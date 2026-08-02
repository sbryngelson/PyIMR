"""Does the selected constitutive model depend on the fitting window?

Inferred IMR parameters are known to drift with the window they are fitted over, which is
read as evidence that a constant-parameter model cannot describe a whole cavitation event.
That work fixes the model and watches the parameters move. This asks the sharper question:
whether the *model choice itself* moves. If the collapse and the rebound select different
models, then "which constitutive model describes this material" is not well posed without
naming a window.

Real laser-induced cavitation data, many trials per condition. The noise is estimated from
the trial-to-trial spread at each time, so it is measured rather than assumed.

Every model is simulated once over the full record from the bubble maximum; only the
samples entering the likelihood change between windows. That isolates the window effect
from the separate problem of how to re-initialize a model partway through an event.

Run: .venv/bin/python examples/windowed_selection.py <dataset> [thermal Nt] [workers]
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

import pyimr
from pyimr.noise import (
  STRAIN_RATE_THRESHOLD_PER_S,
  characteristic_time,
  hencky_strain_rate,
  strain_rate_weights,
  weighted_deviation,
)
from pyimr.selection import (
  STANDARD_MODELS,
  compare,
  log_evidence,
  parameter_grid,
  redundancy_over_grid,
)

DATA = Path.home() / "fastscratch/papers/paper_imr_windowing/data"
# Solver tolerance. Adjustable: raise for validation work, lower for speed. At 1e-6 the
# worst observable (internal pressure) carries 2.8e-05 relative error and the radius 3.4e-07,
# against experimental noise of ~2e-02 -- and the model selection is unchanged against 1e-9.
RTOL, ATOL = 1e-6, 1e-8

GRID_COUNT = 10
# Rayleigh-Plesset. Almost certainly the wrong choice for laser cavitation, which is
# compressible, but radial=2 exhausts the step budget at some grid points and needs
# per-point failure handling first. Kept at 1 so this knob changes nothing yet.
_RADIAL = int(os.environ.get('PYIMR_RADIAL', '1'))
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

def collapse_windows(trace, times):
  """Split at the local maxima between collapses, so each window holds one collapse."""
  interior = np.where((trace[1:-1] > trace[:-2]) & (trace[1:-1] >= trace[2:]))[0] + 1
  peaks = [p for p in interior if trace[p] > 0.15]
  edges = [0, *peaks, len(trace)]
  windows = {}
  for index in range(min(len(edges) - 1, 3)):
    mask = np.zeros(len(trace), dtype=bool)
    mask[edges[index]:edges[index + 1]] = True
    if mask.sum() >= 10: windows[f"collapse {index + 1}"] = mask
  windows["full record"] = np.ones(len(trace), dtype=bool)
  return windows

def setup(dataset, thermal_nodes, radial=_RADIAL):
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
  options: dict[str, Any] = {"radial": radial}
  if thermal_nodes: options |= {"bubtherm": 1, "thermal": "spectral", "Nt": thermal_nodes}

  def solve(material):
    result = pyimr.simulate(
      times, pyimr.SimulationConfig(maximum_radius, equilibrium, material, rtol=RTOL, atol=ATOL, **options)
    )
    return result.radius_ratio, result.stress_integral_pa

  return trials, times, weights, solve, usable, (maximum_radius, equilibrium)

def solve_chunk(payload):
  """One model over a slice of its grid, plus the children that slice needs.

  Module level and picklable-only because JAX deadlocks under `fork`, so `spawn` is
  required, and `spawn` cannot pickle a closure.
  """
  os.environ.setdefault("OMP_NUM_THREADS", "1")
  dataset, thermal_nodes, model, low, high = payload
  _, _, weights, solve, _, _ = setup(dataset, thermal_nodes)
  candidate = STANDARD_MODELS[model]
  points = parameter_grid(candidate.axes, GRID_COUNT)[0][low:high]
  solved = [solve(candidate.build(dict(zip(candidate.axes, row)))) for row in points]
  stresses = np.array([s for _, s in solved])
  redundancies = redundancy_over_grid(candidate, STANDARD_MODELS, points, stresses, solve, weights=weights)
  return model, low, np.array([r for r, _ in solved]), redundancies

def main():
  name = sys.argv[1] if len(sys.argv) > 1 else "gelatin_15C"
  # optional second argument: bubble-thermal node count. Accuracy saturates by Nt = 9
  # (agreement to ~1e-9 against Nt = 60), so there is no reason to go finer -- see #181.
  thermal_nodes = int(sys.argv[2]) if len(sys.argv) > 2 else 0
  workers = int(sys.argv[3]) if len(sys.argv) > 3 else 1

  trials, times, weights, _, usable, (maximum_radius, equilibrium) = setup(name, thermal_nodes)
  mean_trace = trials.mean(axis=0)
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
    for model in STANDARD_MODELS.values()
    for low in range(0, GRID_COUNT**model.dimension, _CHUNK)
  ]
  start = time.perf_counter()
  if workers > 1:
    with mp.get_context("spawn").Pool(workers) as pool: results = pool.map(solve_chunk, payloads)
  else:
    results = [solve_chunk(item) for item in payloads]

  cached, solves = {}, 0
  for candidate in STANDARD_MODELS.values():
    parts = sorted((low, radii, red) for model, low, radii, red in results if model == candidate.name)
    radii = np.concatenate([r for _, r, _ in parts])
    cached[candidate.name] = (
      radii, parameter_grid(candidate.axes, GRID_COUNT)[1],
      np.concatenate([w for _, _, w in parts]), candidate.dimension,
    )
    solves += len(radii) * (1 + len(candidate.contains))
  print(f"{solves} solves in {time.perf_counter() - start:.1f} s on {workers} worker(s)\n")

  windows = collapse_windows(mean_trace, times)
  names = list(cached)
  print(f"  {'window':14s} {'samples':>7s}  " + "  ".join(f"{n:>9s}" for n in names))
  for label, mask in windows.items():
    deviations = weighted_deviation(spread[mask], weights[mask])
    evidences, fits = {}, {}
    for name, (radii, normalized, redundancies, dimension) in cached.items():
      evidences[name], chi_squared = log_evidence(
        radii[:, mask], normalized, redundancies, trials[:, mask], deviations, dimension=dimension
      )
      fits[name] = chi_squared.min() / trials[:, mask].size
    posterior = compare(evidences)
    winner = max(posterior, key=lambda n: posterior[n])
    row = "  ".join(f"{posterior[n]:9.2e}" for n in names)
    # best chi2/N is the check that any of this means anything: if no model reaches ~1 the
    # candidates are all wrong and the "winner" is only the least-bad of a bad set
    print(f"  {label:14s} {int(mask.sum()):7d}  {row}   <- {winner}  "
          f"[best chi2/N {min(fits.values()):.1f}, winner {fits[winner]:.1f}]")

if __name__ == "__main__":
  main()
