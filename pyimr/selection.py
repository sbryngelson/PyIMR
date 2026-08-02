"""Bayesian comparison of constitutive models against a radius-time curve.

The candidates nest, so a best-fit comparison always prefers the flexible ones. This
assembles `pyimr.noise` (likelihood, marginalized noise scale) and `pyimr.prior`
(redundancy, Occam) into an evidence per model and a posterior over the set.

Method follows Sanchez et al., Soft Matter 2026 (doi:10.1039/D5SM01193K).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ._materials import (
  InstantaneousMaterial,
  LinearMaxwell,
  NeoHookean,
  NeoHookeanKelvinVoigt,
  Newtonian,
  QuadraticKelvinVoigt,
  Zener,
)
from .noise import marginal_log_likelihood
from .prior import (
  harmonic_bottleneck,
  model_posterior,
  model_prior,
  normalize_log_coordinates,
  parameter_prior,
  redundancy_weight,
  stress_scale,
)

__all__ = [
  "PARAMETER_BOUNDS", "STANDARD_MODELS", "CandidateModel", "compare", "log_evidence",
  "parameter_grid", "redundancy_over_grid", "solve_grid",
]

PARAMETER_BOUNDS = {"mu": (1e-4, 1.0), "g": (1e2, 1e5), "lambda1": (1e-7, 1e-3), "alpha": (1e-3, 10.0)}

_NEGLIGIBLE = 1e-9

@dataclass(frozen=True, slots=True)
class CandidateModel:
  """A model, its free parameters, and the models it degenerates into."""

  name: str
  build: Callable[[dict], object]
  axes: tuple[str, ...]
  contains: tuple[str, ...] = ()

  @property
  def dimension(self) -> int: return len(self.axes)

STANDARD_MODELS: dict[str, CandidateModel] = {
  m.name: m for m in (
    CandidateModel("newtonian", lambda t: InstantaneousMaterial(viscous=Newtonian(t["mu"])), ("mu",)),
    CandidateModel("NH", lambda t: InstantaneousMaterial(elastic=NeoHookean(t["g"])), ("g",)),
    CandidateModel("NHKV", lambda t: NeoHookeanKelvinVoigt(t["g"], t["mu"]), ("mu", "g"), ("newtonian", "NH")),
    CandidateModel("qNH", lambda t: QuadraticKelvinVoigt(t["g"], _NEGLIGIBLE, t["alpha"]), ("g", "alpha"), ("NH",)),
    CandidateModel("linmax", lambda t: LinearMaxwell(t["mu"], t["lambda1"]), ("mu", "lambda1"), ("newtonian",)),
    CandidateModel("qKV", lambda t: QuadraticKelvinVoigt(t["g"], t["mu"], t["alpha"]), ("mu", "g", "alpha"), ("NHKV", "qNH")),
    CandidateModel("SLS", lambda t: Zener(t["g"], t["mu"], t["lambda1"], 0.0), ("mu", "g", "lambda1"), ("NHKV", "linmax")),
  )
}

def parameter_grid(axes, count, bounds=None):
  """Cartesian product of log-spaced axes, plus the same points in `[0, 1]`.

  Use one `count` for every model compared: mixed resolutions let grid luck, not the
  models, decide which lands nearest the truth.
  """
  bounds = PARAMETER_BOUNDS if bounds is None else bounds
  if int(count) < 2: raise ValueError("count must be at least 2")
  if missing := [a for a in axes if a not in bounds]: raise ValueError(f"no bounds given for {missing}")

  spans = [np.logspace(np.log10(bounds[a][0]), np.log10(bounds[a][1]), int(count)) for a in axes]
  points = np.column_stack([m.ravel() for m in np.meshgrid(*spans, indexing="ij")])
  normalized = np.column_stack([normalize_log_coordinates(points[:, i], *bounds[a]) for i, a in enumerate(axes)])
  return points, normalized

def solve_grid(candidate, solve, *, count, bounds=None):
  """Evaluate a candidate over its grid. `solve(material)` returns `(radius, stress)`."""
  points, normalized = parameter_grid(candidate.axes, count, bounds)
  solved = [solve(candidate.build(dict(zip(candidate.axes, row)))) for row in points]
  return points, normalized, np.array([r for r, _ in solved]), np.array([s for _, s in solved])

def redundancy_over_grid(candidate, models, points, stresses, solve, *, weights=None):
  """Redundancy weight per grid point against each contained model (eqns 22-24).

  Children are solved at the parent's parameters, not looked up in their own grid: log
  grids of different lengths share only endpoints, so a lookup misses nearly everywhere
  and silently leaves the weight at 1.
  """
  redundancies = np.ones(len(points))
  for child in (models[name] for name in candidate.contains):
    for index, row in enumerate(points):
      theta = dict(zip(candidate.axes, row))
      _, stress = solve(child.build({a: theta[a] for a in child.axes}))
      redundancies[index] = min(
        redundancies[index],
        redundancy_weight(stresses[index], [stress], weights=weights, scale=stress_scale(stress)),
      )
  return redundancies

def log_evidence(radii, normalized, redundancies, observed, deviations, *, dimension, marginalize_noise=True):
  """Grid-quadrature evidence including the Occam prior (eqns 20-21, 27).

  `observed` is `(trial, sample)`, `radii` is `(point, sample)`. Returns
  `(log evidence, chi-squared per point)`.
  """
  observed = np.atleast_2d(np.asarray(observed, dtype=float))
  radii, deviations = np.asarray(radii, dtype=float), np.asarray(deviations, dtype=float)
  if radii.shape[1] != observed.shape[1]: raise ValueError("radii and observations must share a sample axis")

  effective = observed.size
  chi_squared = np.array([float(np.sum(((observed - r[None, :]) / deviations[None, :]) ** 2)) for r in radii])
  log_likelihood = (
    np.array([marginal_log_likelihood(v, effective) for v in chi_squared]) if marginalize_noise else -0.5 * chi_squared
  )

  prior = parameter_prior(np.array([harmonic_bottleneck(row) for row in normalized]), redundancies)
  support = prior > 0.0
  peak = float(np.max(log_likelihood[support]))
  integrated = peak + float(np.log(np.sum(prior[support] * np.exp(log_likelihood[support] - peak))))
  return integrated + float(np.log(model_prior(dimension, float(effective)))), chi_squared

def compare(log_evidences):
  """Normalized posterior over a model set, keyed as given."""
  names = list(log_evidences)
  return dict(zip(names, model_posterior(np.array([log_evidences[n] for n in names]))))
