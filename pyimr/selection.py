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
  ArrudaBoyce,
  Fung,
  Gent,
  Giesekus,
  InstantaneousMaterial,
  LinearPTT,
  LinearMaxwell,
  MooneyRivlin,
  NeoHookean,
  NeoHookeanKelvinVoigt,
  Newtonian,
  OldroydB,
  PowerLaw,
  QuadraticKelvinVoigt,
  QuadraticZener,
  Yeoh,
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
  "PARAMETER_BOUNDS", "STANDARD_MODELS", "CandidateModel", "bounds_for_invariant", "compare", "grid_ready",
  "log_evidence",
  "parameter_grid", "redundancy_over_grid", "solve_grid", "strain_invariant",
]

# Deliberately loose, and log-spaced: a boundary-pinned optimum is charged by the
# harmonic bottleneck, so a range that is too tight makes a model look worse than it is.
#
# `alpha` runs to 100 rather than 10 for that reason, measured rather than assumed: on the
# 15 C record SMC put 97.5% of the qSLS posterior at 9.04, against a ceiling of 10. Widening
# tenfold moved the median only 6.37 to 5.27 and the log evidence 1.5 out of 2173, so the
# parameter is identified and the old ceiling was merely clipping its tail (#216).
# `lam` is the retardation/relaxation RATIO -- the solver's own `LAM` -- because the
# absolute retardation time must stay strictly below the relaxation time, and a free
# absolute axis would put most of the grid outside what the material will construct.
PARAMETER_BOUNDS = {
  "mu": (1e-4, 1.0), "g": (1e2, 1e5), "lambda1": (1e-7, 1e-3), "alpha": (1e-3, 100.0),
  "lam": (1e-3, 9e-1), "mobility": (1e-3, 9e-1), "ptt_eps": (1e-3, 1.0),
  "gent_jm": (1e-1, 1e3), "fung_b": (1e-2, 1e2), "ab_n": (1.1, 1e3),
  "c01": (1e2, 1e5), "yeoh_c2": (1e0, 1e5), "yeoh_c3": (1e0, 1e5),
  "pl_k": (1e-4, 1e1), "pl_n": (1e-2, 2.0),
}

_NEGLIGIBLE = 1e-9
# How many e-foldings of Fung stiffening are allowed across the strain range the record
# covers, and how far above its divergence limit Gent must sit to be integrable at all.
#
# Gent's floor is not a safety margin, it is where the model stops being Gent. Measured on
# a record of stretch 7.09 (span 47.6): every Jm below 1000x span fails to integrate, and
# a 40x larger step budget does not change that, so it is the trajectory and not the
# budget. At 1000x span Gent differs from Neo-Hookean by 1.7e-04 of Rmax, against
# measurement noise near 2e-02 -- 120x below what the data can see. So on records like
# these Gent is either unintegrable or indistinguishable from the simpler model it
# contains, and the redundancy prior is left to say so by driving its weight to zero
# rather than the bound pretending otherwise (#199).
_GENT_MARGIN = (1e3, 1e5)
_FUNG_EFOLDS = (1e-3, 5.0)

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
    # strain-stiffening elastics: four different answers to what happens at extreme
    # strain, where the quadratic term in `qNH`/`qKV` is the only shape on offer
    CandidateModel("gent", lambda t: InstantaneousMaterial(elastic=Gent(t["g"], t["gent_jm"])), ("g", "gent_jm"), ("NH",)),
    CandidateModel("fung", lambda t: InstantaneousMaterial(elastic=Fung(t["g"], t["fung_b"])), ("g", "fung_b"), ("NH",)),
    CandidateModel("arruda", lambda t: InstantaneousMaterial(elastic=ArrudaBoyce(t["g"], t["ab_n"])), ("g", "ab_n"), ("NH",)),
    CandidateModel("mooney", lambda t: InstantaneousMaterial(elastic=MooneyRivlin(t["g"], t["c01"])), ("g", "c01"), ("NH",)),
    CandidateModel("yeoh", lambda t: InstantaneousMaterial(elastic=Yeoh(t["g"], t["yeoh_c2"], t["yeoh_c3"])), ("g", "yeoh_c2", "yeoh_c3"), ("NH",)),
    CandidateModel(
      "powerlaw", lambda t: InstantaneousMaterial(elastic=NeoHookean(t["g"]), viscous=PowerLaw(t["pl_k"], t["pl_n"])),
      ("g", "pl_k", "pl_n"), ("NH",),
    ),
    # memory models the set was missing: stiffening WITH relaxation, and the fluids
    CandidateModel(
      "qSLS", lambda t: QuadraticZener(t["g"], t["mu"], t["lambda1"], 0.0, t["alpha"]),
      ("mu", "g", "lambda1", "alpha"), ("qKV", "SLS"),
    ),
    CandidateModel(
      "oldroydb", lambda t: OldroydB(t["mu"], t["lambda1"], t["lam"] * t["lambda1"]), ("mu", "lambda1", "lam"), ("linmax",)
    ),
    CandidateModel(
      "giesekus", lambda t: Giesekus(t["mu"], t["lambda1"], t["lam"] * t["lambda1"], t["mobility"]),
      ("mu", "lambda1", "lam", "mobility"), ("oldroydb",),
    ),
    CandidateModel(
      "ptt", lambda t: LinearPTT(t["mu"], t["lambda1"], t["lam"] * t["lambda1"], t["ptt_eps"]),
      ("mu", "lambda1", "lam", "ptt_eps"), ("oldroydb",),
    ),
  )
}

def grid_ready(models=None):
  """Names of models whose whole grid shares one compiled program.

  A material whose numbers all travel through the nondimensional groups is keyed by type,
  so a sweep compiles once (#163, #196). What is left is `Ogden`, whose variable-length
  tuples cannot travel that way: it is keyed by content and compiles once per grid point.
  """
  from ._integrate import shares_one_program

  models = STANDARD_MODELS if models is None else models
  centre = lambda a: float(np.sqrt(PARAMETER_BOUNDS[a][0] * PARAMETER_BOUNDS[a][1]))  # noqa: E731
  return frozenset(
    name for name, c in models.items() if shares_one_program(c.build({a: centre(a) for a in c.axes}))
  )

def strain_invariant(radius_ratio, equilibrium_ratio):
  """`I1 - 3` at the deepest point of a trace, which is what the stiffening laws see.

  `_elastic_integrand` uses `lam**-4 + 2*lam**2 - 3`, so COMPRESSION governs: at the
  collapse `lam**-4` runs away, while the expansion at `Rmax` contributes a few tens. On
  the gelatin records the collapse gives 24 to 119 where the expansion form gives 44 to 52,
  and reading the wrong one is what made the first attempt at these bounds miss (#199).
  """
  lam = float(np.min(radius_ratio)) / float(equilibrium_ratio)
  if lam <= 0.0: raise ValueError("radius ratio must stay positive")
  return lam**-4 + 2.0 * lam**2 - 3.0

def bounds_for_invariant(span, bounds=None):
  """`PARAMETER_BOUNDS` with the divergence-limited axes placed against `span = I1 - 3`.

  Gent locks up as `I1 - 3 -> Jm` and Fung grows as `exp(b*(I1 - 3))`, so what either can
  take is set by how far the material is actually driven, not by a constant. Pass
  `strain_invariant` of the measured trace.

  Both axes become a multiple of the span, so the normalized coordinate the prior sees
  means one thing across datasets, which an absolute `Jm` never did.
  """
  span = float(span)
  if span <= 0.0: raise ValueError("the strain invariant must be positive")
  bounds = dict(PARAMETER_BOUNDS if bounds is None else bounds)
  bounds["gent_jm"] = (_GENT_MARGIN[0] * span, _GENT_MARGIN[1] * span)
  bounds["fung_b"] = (_FUNG_EFOLDS[0] / span, _FUNG_EFOLDS[1] / span)
  return bounds

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
  # `logspace` does not reproduce its own endpoints exactly -- through log10 and back, a
  # bound of 24002.829853450417 comes out 1.1e-11 low, which normalizes to -3.9e-16 and
  # trips the prior's non-negativity guard. Every default bound is a power of ten and
  # round-trips exactly, so this only appeared once bounds were derived from data. The
  # points ARE the bounds by construction, so the excursion is round-off, not signal.
  normalized = np.clip(
    np.column_stack([normalize_log_coordinates(points[:, i], *bounds[a]) for i, a in enumerate(axes)]), 0.0, 1.0
  )
  return points, normalized

def solve_grid(candidate, solve, *, count, bounds=None):
  """Evaluate a candidate over its grid. `solve(material)` returns `(radius, stress)`.

  Every point must solve. A sweep whose points can fail wants its own loop, marking the
  failures so they can be dropped from the prior -- `examples/windowed_selection.py`.
  """
  points, normalized = parameter_grid(candidate.axes, count, bounds)
  solved = [solve(candidate.build(dict(zip(candidate.axes, row)))) for row in points]
  return points, normalized, np.array([r for r, _ in solved]), np.array([s for _, s in solved])

def redundancy_over_grid(candidate, models, points, stresses, solve, *, weights=None):
  """Redundancy weight per grid point against each contained model (eqns 22-24).

  Children are solved at the parent's parameters, not looked up in their own grid: log
  grids of different lengths share only endpoints, so a lookup misses nearly everywhere
  and silently leaves the weight at 1.

  `solve` may return `None` for a material it cannot integrate; a child that will not run
  cannot demonstrate redundancy, so that point keeps the weight it already has.
  """
  redundancies = np.ones(len(points))
  for child in (models[name] for name in candidate.contains):
    for index, row in enumerate(points):
      theta = dict(zip(candidate.axes, row))
      solved = solve(child.build({a: theta[a] for a in child.axes}))
      if solved is None: continue
      _, stress = solved
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
