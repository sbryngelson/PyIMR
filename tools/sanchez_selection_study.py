"""Bayesian model selection over the Sanchez et al. model set, on synthetic data.

The paper reports which constitutive model wins for each real material. It cannot check
that answer, because the true model of a real gel is unknown. Here the truth is chosen, so
the pipeline can be asked the question that matters: does it recover it?

The truth is `NHKV`, deliberately. It is nested inside both three-parameter models in the
set -- `qKV` reduces to it at zero strain-stiffening, `SLS` at zero relaxation time -- so
those two can fit the data exactly as well as the truth does. Nothing in the likelihood
prefers the simpler model. Only the Occam model prior and the redundancy parameter prior
do. If either is wrong, an over-parameterized model wins and the failure is visible.

Run: .venv/bin/python tools/sanchez_selection_study.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

import pyimr
from pyimr.noise import elliptical_gate, marginal_log_likelihood, strain_rate_weights, weighted_deviation
from pyimr.prior import (
  harmonic_bottleneck,
  model_posterior,
  model_prior,
  normalize_log_coordinates,
  parameter_prior,
  redundancy_weight,
  stress_scale,
)

R0, REQ = 2.25e-4, 5.0e-5
TRUTH = {"g": 2500.0, "mu": 0.1}
TRIALS, NOISE_FRACTION, SEED = 8, 0.01, 0
WINDOW, SAMPLES = 20e-6, 120

# Table 2 of the paper, in SI units
BOUNDS = {"mu": (1e-4, 1.0), "g": (1e2, 1e5), "lambda1": (1e-7, 1e-3), "alpha": (1e-3, 10.0)}
_TINY_VISCOSITY, _TINY_MODULUS = 1e-9, 1e-6

def _newtonian(t): return pyimr.InstantaneousMaterial(viscous=pyimr.Newtonian(t["mu"]))
def _neo_hookean(t): return pyimr.InstantaneousMaterial(elastic=pyimr.NeoHookean(t["g"]))
def _quadratic_neo_hookean(t): return pyimr.QuadraticKelvinVoigt(t["g"], _TINY_VISCOSITY, t["alpha"])
def _kelvin_voigt(t): return pyimr.NeoHookeanKelvinVoigt(t["g"], t["mu"])
def _quadratic_kelvin_voigt(t): return pyimr.QuadraticKelvinVoigt(t["g"], t["mu"], t["alpha"])
def _linear_maxwell(t): return pyimr.LinearMaxwell(t["mu"], t["lambda1"])
def _standard_linear_solid(t): return pyimr.Zener(t["g"], t["mu"], t["lambda1"], 0.0)

# name -> (builder, axes, contained simpler models, grid points per axis)
MODELS = {
  "newtonian": (_newtonian, ("mu",), (), 24),
  "NH": (_neo_hookean, ("g",), (), 24),
  "NHKV": (_kelvin_voigt, ("mu", "g"), ("newtonian", "NH"), 14),
  "qNH": (_quadratic_neo_hookean, ("g", "alpha"), ("NH",), 14),
  "linmax": (_linear_maxwell, ("mu", "lambda1"), ("newtonian",), 14),
  "qKV": (_quadratic_kelvin_voigt, ("mu", "g", "alpha"), ("NHKV", "qNH"), 9),
  "SLS": (_standard_linear_solid, ("mu", "g", "lambda1"), ("NHKV", "linmax"), 9),
}

def solve(material, times):
  result = pyimr.simulate(times, pyimr.SimulationConfig(R0, REQ, material, rtol=1e-9, atol=1e-11))
  return result.radius_ratio, result.stress_integral_pa

def grid_for(axes, count):
  """Cartesian product of log-spaced axes, plus the same points in normalized coordinates."""
  spans = [np.logspace(np.log10(BOUNDS[a][0]), np.log10(BOUNDS[a][1]), count) for a in axes]
  mesh = np.meshgrid(*spans, indexing="ij")
  points = np.column_stack([m.ravel() for m in mesh])
  normalized = np.column_stack([normalize_log_coordinates(points[:, i], *BOUNDS[a]) for i, a in enumerate(axes)])
  return points, normalized

def main():
  times = np.linspace(0.0, WINDOW, SAMPLES)
  rng = np.random.default_rng(SEED)

  clean, _ = solve(_kelvin_voigt(TRUTH), times)
  sigma = NOISE_FRACTION * float(clean.max())
  observed = clean[None, :] + rng.normal(0.0, sigma, size=(TRIALS, SAMPLES))

  # Hencky strain and rate from the noiseless truth, as the paper does from trial means
  velocity = np.gradient(clean, times)
  characteristic_time = R0 * np.sqrt(998.0 / 101325.0)
  strain = np.log(np.maximum(clean, 1e-12) / (REQ / R0))
  strain_rate = velocity / np.maximum(clean, 1e-12) * characteristic_time

  keep = elliptical_gate(strain, strain_rate, 0.1 * np.max(np.abs(strain)), 1e5 * characteristic_time)
  weights = strain_rate_weights(strain_rate[keep], 1e5 * characteristic_time)
  deviations = weighted_deviation(np.full(int(keep.sum()), sigma), weights)
  effective = int(keep.sum()) * TRIALS
  print(f"truth NHKV(g={TRUTH['g']}, mu={TRUTH['mu']})   sigma={sigma:.3e}   "
        f"{TRIALS} trials, {int(keep.sum())}/{SAMPLES} samples pass the gate\n")

  predictions, log_evidences, names, solves = {}, [], [], 0
  start = time.perf_counter()

  for name, (build, axes, children, count) in MODELS.items():
    points, normalized = grid_for(axes, count)
    radii, stresses = [], []
    for row in points:
      theta = dict(zip(axes, row))
      radius, stress = solve(build(theta), times)
      radii.append(radius[keep]); stresses.append(stress[keep]); solves += 1
    predictions[name] = (points, axes, np.array(radii), np.array(stresses))

    # likelihood: chi-squared against every trial, then marginalize the noise scale
    log_likelihood = np.empty(len(points))
    for index, radius in enumerate(predictions[name][2]):
      residual = (observed[:, keep] - radius[None, :]) / deviations[None, :]
      log_likelihood[index] = marginal_log_likelihood(float(np.sum(residual**2)), effective)

    # parameter prior: harmonic bottleneck times redundancy against contained models
    bottlenecks = np.array([harmonic_bottleneck(row) for row in normalized])
    # The child is SOLVED at the parent's own parameters, not looked up in the child's
    # grid. The two grids are log-spaced with different point counts, so they share only
    # their endpoints -- a lookup hits 2 of 9 points per axis, leaving the redundancy
    # weight at 1 almost everywhere and the prior silently inert.
    redundancies = np.ones(len(points))
    for child in children:
      child_build, child_axes = MODELS[child][0], MODELS[child][1]
      for index, row in enumerate(points):
        theta = dict(zip(axes, row))
        _, child_stress = solve(child_build({a: theta[a] for a in child_axes}), times)
        solves += 1
        gated = child_stress[keep]
        factor = redundancy_weight(
          predictions[name][3][index], [gated], weights=weights, scale=stress_scale(gated)
        )
        redundancies[index] = min(redundancies[index], factor)

    prior = parameter_prior(bottlenecks, redundancies)
    support = prior > 0.0
    evidence = float(np.max(log_likelihood[support]))
    evidence += float(np.log(np.sum(prior[support] * np.exp(log_likelihood[support] - evidence))))

    names.append(name)
    log_evidences.append(evidence + np.log(model_prior(len(axes), float(effective))))
    print(f"  {name:10s} k={len(axes)}  nGrid={len(points):5d}  logZ={evidence:+12.3f}  "
          f"min w_red={redundancies.min():.2e}")

  posterior = model_posterior(np.array(log_evidences))
  print(f"\n{solves} solves in {time.perf_counter() - start:.1f} s\n")
  # scientific notation, not %.6f: the losing models sit at 1e-7 and below, and fixed
  # decimals render every one of them as a flat 0.000000
  best = int(np.argmax(posterior))
  print(f"  {'model':10s} {'posterior':>12s}   Bayes factor vs best")
  for index in np.argsort(-posterior):
    factor = float(np.exp(log_evidences[best] - log_evidences[index]))
    mark = "   <- truth" if names[index] == "NHKV" else ""
    print(f"  {names[index]:10s} {posterior[index]:12.3e}   {factor:10.3g}{mark}")

if __name__ == "__main__":
  main()
