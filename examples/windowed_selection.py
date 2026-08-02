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

Run: .venv/bin/python examples/windowed_selection.py <dataset>
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

import pyimr
from pyimr.noise import (
  STRAIN_RATE_THRESHOLD_PER_S,
  characteristic_time,
  hencky_strain_rate,
  strain_rate_weights,
  weighted_deviation,
)
from pyimr.selection import STANDARD_MODELS, compare, log_evidence, redundancy_over_grid, solve_grid

DATA = Path.home() / "fastscratch/papers/paper_imr_windowing/data"
GRID_COUNT = 10
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

def main():
  name = sys.argv[1] if len(sys.argv) > 1 else "gelatin_15C"
  nondimensional_time, trials, maximum_radius, equilibrium = load(name)
  characteristic = characteristic_time(maximum_radius)
  times = nondimensional_time * characteristic

  usable = screen(trials)
  dropped = int((~usable).sum())
  nondimensional_time, trials, times = nondimensional_time[usable], trials[:, usable], times[usable]
  mean_trace = trials.mean(axis=0)
  spread = trials.std(axis=0, ddof=1)

  print(f"{name}: {trials.shape[0]} trials, {trials.shape[1]} samples, "
        f"Rmax={maximum_radius * 1e6:.0f} um, Rmax/Req={maximum_radius / equilibrium:.2f}")
  print(f"  measured spread: median {np.median(spread):.4f}, max {spread.max():.4f} of Rmax")
  print(f"  screened out {dropped} of {dropped + int(usable.sum())} samples "
        f"(unphysical R* or zero spread)\n")

  def solve(material):
    result = pyimr.simulate(times, pyimr.SimulationConfig(maximum_radius, equilibrium, material, rtol=1e-9, atol=1e-11))
    return result.radius_ratio, result.stress_integral_pa

  rate = hencky_strain_rate(mean_trace, times, characteristic)
  weights = strain_rate_weights(rate, STRAIN_RATE_THRESHOLD_PER_S * characteristic)

  cached, start, solves = {}, time.perf_counter(), 0
  for candidate in STANDARD_MODELS.values():
    points, normalized, radii, stresses = solve_grid(candidate, solve, count=GRID_COUNT)
    redundancies = redundancy_over_grid(candidate, STANDARD_MODELS, points, stresses, solve, weights=weights)
    solves += len(points) * (1 + len(candidate.contains))
    cached[candidate.name] = (radii, normalized, redundancies, candidate.dimension)
  print(f"{solves} solves in {time.perf_counter() - start:.1f} s\n")

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
