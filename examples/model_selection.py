"""Recover a known constitutive model from a synthetic radius-time curve.

Nested model selection cannot be validated on real data, where only the ranking is
observable. Choosing the truth makes it answerable.

Truth is the standard linear solid at De ~ 1, sitting on a grid node, with one resolution
for every candidate so grid luck cannot decide the winner. Swap `TRUTH_MODEL` for the
other half of the check: `NHKV` tests that spurious complexity is refused, which a method
always answering "simplest" would pass; `SLS` tests that real complexity is still found.

Run: .venv/bin/python examples/model_selection.py
"""

from __future__ import annotations

import time

import numpy as np

import pyimr
from _common import ATOL, RTOL
from pyimr.noise import (
  STRAIN_RATE_THRESHOLD_PER_S,
  characteristic_time,
  elliptical_gate,
  hencky_strain_rate,
  strain_rate_weights,
  weighted_deviation,
)
from pyimr.selection import (
  PARAMETER_BOUNDS,
  STANDARD_MODELS,
  compare,
  evaluate_at,
  log_evidence,
  redundancy_over_grid,
  solve_grid,
)


R0, REQ = 2.25e-4, 5.0e-5
GRID_COUNT = 12
TRUTH_MODEL = "SLS"
TRIALS, NOISE_FRACTION, SEED = 8, 1e-3, 0
WINDOW, SAMPLES = 20e-6, 120


def _node(name, index):
  lower, upper = PARAMETER_BOUNDS[name]
  return float(np.logspace(np.log10(lower), np.log10(upper), GRID_COUNT)[index])

TRUTH = {"mu": _node("mu", 8), "g": _node("g", 5), "lambda1": _node("lambda1", 7)}

def solve(material, _config):
  result = pyimr.simulate(
    np.linspace(0.0, WINDOW, SAMPLES),
    pyimr.SimulationConfig(R0, REQ, material, rtol=RTOL, atol=ATOL),
  )
  return result.radius_ratio, result.stress_integral_pa

def main():
  times = np.linspace(0.0, WINDOW, SAMPLES)
  clean, _ = evaluate_at(STANDARD_MODELS[TRUTH_MODEL], solve, TRUTH)

  characteristic = characteristic_time(R0)
  threshold = STRAIN_RATE_THRESHOLD_PER_S * characteristic
  rate = hencky_strain_rate(clean, times, characteristic)
  strain = np.log(np.maximum(clean, 1e-12) / (REQ / R0))

  keep = elliptical_gate(strain, rate, 0.1 * np.max(np.abs(strain)), threshold)
  weights = strain_rate_weights(rate[keep], threshold)
  sigma = NOISE_FRACTION * float(clean.max())
  deviations = weighted_deviation(np.full(int(keep.sum()), sigma), weights)

  rng = np.random.default_rng(SEED)
  observed = (clean[None, :] + rng.normal(0.0, sigma, size=(TRIALS, SAMPLES)))[:, keep]
  print(f"truth {TRUTH_MODEL} {({k: float(f'{v:.4g}') for k, v in TRUTH.items()})}  "
        f"De={TRUTH['lambda1'] / characteristic:.2f}  sigma/Rmax={NOISE_FRACTION:.0e}")
  print(f"{TRIALS} trials, {int(keep.sum())}/{SAMPLES} samples pass the gate\n")

  evidences, start, solves = {}, time.perf_counter(), 0
  for name, candidate in STANDARD_MODELS.items():
    points, normalized, radii, stresses = solve_grid(candidate, solve, count=GRID_COUNT)
    solves += len(points)
    redundancies = redundancy_over_grid(
      candidate, STANDARD_MODELS, points, stresses[:, keep], solve, weights=weights
    )
    solves += len(points) * len(candidate.contains)
    evidences[name], chi_squared = log_evidence(
      radii[:, keep], normalized, redundancies, observed, deviations, dimension=candidate.dimension
    )
    print(f"  {name:10s} k={candidate.dimension}  nGrid={len(points):5d}  "
          f"logZ={evidences[name]:+11.3f}  best chi2/N={chi_squared.min() / observed.size:.4f}  "
          f"min w_red={redundancies.min():.2e}")

  posterior = compare(evidences)
  print(f"\n{solves} solves in {time.perf_counter() - start:.1f} s\n")
  # scientific notation: losing models sit far below what fixed decimals can show
  print(f"  {'model':10s} {'posterior':>12s}   Bayes factor vs best")
  best = max(posterior, key=lambda name: posterior[name])
  for name in sorted(posterior, key=lambda n: posterior[n], reverse=True):
    mark = "   <- truth" if name == TRUTH_MODEL else ""
    print(f"  {name:10s} {posterior[name]:12.3e}   "
          f"{np.exp(evidences[best] - evidences[name]):10.3g}{mark}")

if __name__ == "__main__":
  main()
