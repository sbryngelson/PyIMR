"""Which constitutive model do measured cavitation records actually prefer?

Real laser-induced cavitation data, many trials per condition. The noise is estimated from
the trial-to-trial spread at each time, so it is measured rather than assumed, and the
comparison runs the full candidate set against the whole record.

Run: .venv/bin/python examples/measured_selection.py <dataset> [thermal Nt] [workers]
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any

# Before importing anything that pulls in XLA. Its CPU backend sizes its thread pool from
# the AFFINITY MASK, so 16 spawn workers on a 128-core host took ~400 threads each: 6500
# threads over 128 cores, thrashing this run and every other job on the machine. Measured
# per process: 397 threads unrestricted, 270 with the XLA thread flags, 4 pinned to one
# core. Only affinity actually works, so `_pin` below is the control and these are belt.
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
  os.environ.setdefault(_name, "1")

import numpy as np

import pyimr
from _common import ATOL, RTOL
from pyimr.parallel import pin_worker
from pyimr.noise import (
  STRAIN_RATE_THRESHOLD_PER_S,
  characteristic_time,
  hencky_strain_rate,
  strain_rate_weights,
  weighted_deviation,
)
from pyimr.selection import (
  STANDARD_MODELS,
  bounds_for_invariant,
  compare,
  evaluate_at,
  log_evidence,
  parameter_grid,
  redundancy_over_grid,
  strain_invariant,
)

DATA = Path.home() / "fastscratch/papers/paper_imr_windowing/data"

GRID_COUNT = 10
MODELS = STANDARD_MODELS  # every candidate: #196 made the distributed pair affordable
# Keller-Miksis. Laser cavitation is compressible; PYIMR_DYNAMICS=rayleigh-plesset
# recovers the incompressible results for comparison.
_DYNAMICS = os.environ.get('PYIMR_DYNAMICS', 'keller-miksis')
_LIQUID_EOS = os.environ.get('PYIMR_LIQUID_EOS') or None
# Step budget per solve. Points that collapse to a fraction of a percent of R_max and
# creep instead of rebounding run to any ceiling they are given; the healthy points of
# the same grid finish under 7e3 steps. Against the 1e6 default this runs 6.3x faster
# (205 s to 33 s) and drops 9.4% of the SLS grid rather than 4.4%; every winner and
# every best chi-squared is unchanged, and the losing posteriors move by about 10%
# relative -- on models already 30 or more orders below the winner.
_MAX_STEPS = int(os.environ.get('PYIMR_MAX_STEPS', '50000'))
_CHUNK = 120  # grid points per work unit; keeps the uneven model sizes load-balanced
_MAX_RATIO = 1.05  # R* cannot exceed the maximum it is normalized by, beyond tracking noise

# name -> (file, R_max [m], R_max/R_inf) from the paper's dataset table
DATASETS = {
  "gelatin_15C": ("Ga_t15_exp_data.csv", 277e-6, 7.09),
  "gelatin_23C": ("Ga_t23_exp_data.csv", 298e-6, 7.37),
  "gelatin_33C": ("Ga_t33_exp_data.csv", 312e-6, 6.83),
}

def load(name):
  filename, maximum_radius, stretch = DATASETS[name]
  table = np.loadtxt(DATA / filename, delimiter=",", ndmin=2)
  return table[:, 0], table[:, 1:].T, maximum_radius, maximum_radius / stretch

def screen(trials):
  """Samples to keep: physically possible, and carrying information.

  Two failures, both handled by dropping rather than by inflating an error bar. A trial
  reading above its own maximum radius is a tracking failure, seen only in the last few
  percent of these records. A sample where every trial agrees exactly is the t*=0 point,
  where R/R_max is 1 by construction -- that is a definition, not a measurement, and
  flooring its spread would invent an uncertainty and let it constrain the fit.
  """
  spread = trials.std(axis=0, ddof=1)
  return ~(trials > _MAX_RATIO).any(axis=0) & (spread > 0.0)

def setup(dataset, thermal_nodes, dynamics=_DYNAMICS, liquid_eos=_LIQUID_EOS):
  """Per-dataset state, rebuilt from picklable arguments alone.

  Workers cannot receive a closure, so they reconstruct this from the dataset name. The
  cost is one CSV read against seconds of solving.
  """
  nondimensional_time, trials, maximum_radius, equilibrium = load(dataset)
  usable = screen(trials)
  trials = trials[:, usable]
  characteristic = characteristic_time(maximum_radius)
  times = nondimensional_time[usable] * characteristic
  weights = strain_rate_weights(
    hencky_strain_rate(trials.mean(axis=0), times, characteristic), STRAIN_RATE_THRESHOLD_PER_S * characteristic
  )
  # annotated: an inferred dict[str, int | str] makes every SimulationConfig field
  # int | str through the **options splat, which pyright rejects across the board
  options: dict[str, Any] = {"dynamics": dynamics, "liquid_eos": liquid_eos}
  if thermal_nodes: options |= {"bubtherm": 1, "thermal": "spectral", "Nt": thermal_nodes}

  def solve(material, _config):
    """`(radius, stress)`, or `None` for a point this solver cannot integrate.

    A grid point that will not run drops out of the parameter prior rather than taking
    the study down with it, so the count of drops is a reported number (below).
    """
    try:
      result = pyimr.simulate(
        times,
        pyimr.SimulationConfig(
          maximum_radius, equilibrium, material, rtol=RTOL, atol=ATOL, max_steps=_MAX_STEPS, **options
        ),
      )
    except pyimr.SimulationError:
      return None
    return result.radius_ratio, result.stress_integral_pa

  # the stiffening laws lock up against the deepest COMPRESSION, which only the trace
  # knows, so their bounds come from it rather than from a constant (#199)
  bounds = bounds_for_invariant(strain_invariant(trials.min(axis=0), equilibrium / maximum_radius))
  return trials, times, weights, solve, usable, (maximum_radius, equilibrium), bounds

def solve_chunk(payload):
  """One model over a slice of its grid, plus the children that slice needs.

  Module level and picklable-only because JAX deadlocks under `fork`, so `spawn` is
  required, and `spawn` cannot pickle a closure.
  """
  dataset, thermal_nodes, model, low, high = payload
  _, times, weights, solve, _, _, bounds = setup(dataset, thermal_nodes)
  candidate = MODELS[model]
  points = parameter_grid(candidate.axes, GRID_COUNT, bounds)[0][low:high]
  solved = [evaluate_at(candidate, solve, dict(zip(candidate.axes, row))) for row in points]

  ok = np.array([item is not None for item in solved])
  radii, redundancies = np.full((len(points), len(times)), np.nan), np.zeros(len(points))
  if ok.any():
    kept = [item for item in solved if item is not None]
    radii[ok] = [r for r, _ in kept]
    redundancies[ok] = redundancy_over_grid(
      candidate, STANDARD_MODELS, points[ok], np.array([s for _, s in kept]), solve, weights=weights
    )
  return model, low, radii, redundancies

def main():
  name = sys.argv[1] if len(sys.argv) > 1 else "gelatin_15C"
  # optional second argument: bubble-thermal node count. Accuracy saturates by Nt = 9
  # (agreement to ~1e-9 against Nt = 60), so there is no reason to go finer -- see #181.
  thermal_nodes = int(sys.argv[2]) if len(sys.argv) > 2 else 0
  workers = int(sys.argv[3]) if len(sys.argv) > 3 else 1

  trials, times, weights, _, usable, (maximum_radius, equilibrium), bounds = setup(name, thermal_nodes)
  spread = trials.std(axis=0, ddof=1)
  dropped = int((~usable).sum())

  print(f"{name}: {trials.shape[0]} trials, {trials.shape[1]} samples, "
        f"Rmax={maximum_radius * 1e6:.0f} um, Rmax/Req={maximum_radius / equilibrium:.2f}, "
        f"{'bubble-thermal Nt=%d' % thermal_nodes if thermal_nodes else 'cold'}")
  print(f"  measured spread: median {np.median(spread):.4f}, max {spread.max():.4f} of Rmax")
  print(f"  screened out {dropped} of {dropped + int(usable.sum())} samples "
        f"(unphysical R* or zero spread)\n")

  payloads = [
    (name, thermal_nodes, model.name, low, min(low + _CHUNK, GRID_COUNT**model.dimension))
    for model in MODELS.values()
    for low in range(0, GRID_COUNT**model.dimension, _CHUNK)
  ]
  start = time.perf_counter()
  if workers > 1:
    with mp.get_context("spawn").Pool(workers, initializer=pin_worker, initargs=(workers,)) as pool:
      results = pool.map(solve_chunk, payloads)
  else:
    results = [solve_chunk(item) for item in payloads]

  cached, solves, lost = {}, 0, []
  for candidate in MODELS.values():
    parts = sorted((low, radii, red) for model, low, radii, red in results if model == candidate.name)
    radii = np.concatenate([r for _, r, _ in parts])
    redundancies = np.concatenate([w for _, _, w in parts])
    failed = int(np.isnan(radii).any(axis=1).sum())
    if failed: lost.append(f"{candidate.name} {failed}/{len(radii)}")
    # a model no point of which integrates has no prior to normalize, so it leaves the set
    if failed == len(radii): continue
    cached[candidate.name] = (radii, parameter_grid(candidate.axes, GRID_COUNT, bounds)[1], redundancies, candidate.dimension)
    solves += len(radii) * (1 + len(candidate.contains))
  print(f"{solves} solves in {time.perf_counter() - start:.1f} s on {workers} worker(s)")
  print(f"  grid points that would not integrate: {', '.join(lost) if lost else 'none'}\n")

  deviations = weighted_deviation(spread, weights)
  evidences, fits = {}, {}
  for name, (radii, normalized, redundancies, dimension) in cached.items():
    evidences[name], chi_squared = log_evidence(radii, normalized, redundancies, trials, deviations, dimension=dimension)
    fits[name] = np.nanmin(chi_squared) / trials.size  # dropped points are NaN, not zero
  posterior = compare(evidences)

  print(f"  {'model':10s} {'posterior':>12s} {'best chi2/N':>12s}")
  # best chi2/N is the check that any of this means anything: if no model reaches ~1 the
  # candidates are all wrong and the "winner" is only the least-bad of a bad set
  for name in sorted(posterior, key=lambda n: posterior[n], reverse=True):
    print(f"  {name:10s} {posterior[name]:12.3e} {fits[name]:12.2f}")

if __name__ == "__main__":
  main()
