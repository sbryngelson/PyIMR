"""How much measurement precision does identifying a viscoelastic model actually take?

Which model wins is only meaningful where the data separate the candidates. At 1% radius
noise they do not, and preferring the simplest adequate model is then correct rather than
a failure. So this sweeps the noise over one solved grid, and ablates the redundancy prior
to ask whether it decides anything.

Watch `best chi2/N` for the truth: flat at the noise floor means the truth is reachable on
its own grid; climbing as noise falls means the run measures discretization error instead.

Run: .venv/bin/python examples/identifiability.py
"""

from __future__ import annotations

import time

import numpy as np

from model_selection import GRID_COUNT, R0, SAMPLES, TRIALS, TRUTH, TRUTH_MODEL, WINDOW, solve
from pyimr.noise import (
  STRAIN_RATE_THRESHOLD_PER_S,
  characteristic_time,
  hencky_strain_rate,
  strain_rate_weights,
  weighted_deviation,
)
from pyimr.selection import (STANDARD_MODELS, compare, evaluate_at, log_evidence,
                             redundancy_over_grid, solve_grid)

NOISE_FRACTIONS = (1e-2, 3e-3, 1e-3, 3e-4, 1e-4)
SEED = 0

def main():
  times = np.linspace(0.0, WINDOW, SAMPLES)
  clean, _ = evaluate_at(STANDARD_MODELS[TRUTH_MODEL], solve, TRUTH)

  characteristic = characteristic_time(R0)
  weights = strain_rate_weights(hencky_strain_rate(clean, times, characteristic), STRAIN_RATE_THRESHOLD_PER_S * characteristic)
  print(f"truth {TRUTH_MODEL} {({k: float(f'{v:.4g}') for k, v in TRUTH.items()})}  "
        f"De = {TRUTH['lambda1'] / characteristic:.2f}\n")

  # nothing below depends on the noise, so it is computed once
  cached, start, solves = {}, time.perf_counter(), 0
  for name, candidate in STANDARD_MODELS.items():
    points, normalized, radii, stresses = solve_grid(candidate, solve, count=GRID_COUNT)
    redundancies = redundancy_over_grid(candidate, STANDARD_MODELS, points, stresses, solve, weights=weights)
    solves += len(points) * (1 + len(candidate.contains))
    cached[name] = (radii, normalized, redundancies, np.ones_like(redundancies), candidate.dimension)
  print(f"{solves} solves in {time.perf_counter() - start:.1f} s\n")

  print(f"  {'sigma/Rmax':>10s} {'winner':>8s} {'P(winner)':>11s} {'P(truth)':>11s} "
        f"{'BF vs truth':>12s}  {'best chi2/N':>11s}   no-w_red: winner/P(truth)")
  for fraction in NOISE_FRACTIONS:
    rng = np.random.default_rng(SEED)
    sigma = fraction * float(clean.max())
    observed = clean[None, :] + rng.normal(0.0, sigma, size=(TRIALS, SAMPLES))
    deviations = weighted_deviation(np.full(SAMPLES, sigma), weights)

    evidences, ablated, truth_fit = {}, {}, None
    for name, (radii, normalized, redundancies, flat, dimension) in cached.items():
      evidences[name], chi_squared = log_evidence(
        radii, normalized, redundancies, observed, deviations, dimension=dimension
      )
      ablated[name] = log_evidence(
        radii, normalized, flat, observed, deviations, dimension=dimension
      )[0]
      if name == TRUTH_MODEL: truth_fit = chi_squared.min() / observed.size

    posterior, flat_posterior = compare(evidences), compare(ablated)
    winner = max(posterior, key=lambda n: posterior[n])
    flat_winner = max(flat_posterior, key=lambda n: flat_posterior[n])
    factor = float(np.exp(evidences[winner] - evidences[TRUTH_MODEL]))
    mark = "  <- truth recovered" if winner == TRUTH_MODEL else ""
    print(f"  {fraction:10.1e} {winner:>8s} {posterior[winner]:11.3e} {posterior[TRUTH_MODEL]:11.3e} "
          f"{factor:12.3g}  {truth_fit:11.4f}   {flat_winner:>8s}/{flat_posterior[TRUTH_MODEL]:.3e}{mark}")

if __name__ == "__main__":
  main()
