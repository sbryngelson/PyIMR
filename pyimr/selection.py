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

from ._config import OPERATORS
from ._materials import (
  ArrudaBoyce,
  Bingham,
  CarreauYasuda,
  CarreauZener,
  Cross,
  Fung,
  HerschelBulkley,
  ModifiedPowellEyring,
  PowellEyring,
  Gent,
  Giesekus,
  InstantaneousMaterial,
  LinearPTT,
  MaterialModel,
  LinearMaxwell,
  MooneyRivlin,
  NeoHookean,
  NeoHookeanKelvinVoigt,
  Newtonian,
  Ogden,
  OldroydB,
  PowerLaw,
  QuadraticKelvinVoigt,
  QuadraticZener,
  TwoModeQuadraticZener,
  Yeoh,
  Zener,
)
from .discriminate import laplace_log_evidence
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
  "DYNAMICS_MODELS", "EXTENDED_MODELS", "PARAMETER_BOUNDS", "STANDARD_MODELS", "CandidateFit",
  "CandidateModel",
  "bounds_for_invariant", "candidate_log_evidence", "compare", "fit_candidate", "grid_ready",
  "evaluate_at", "log_evidence", "parameter_grid", "physical_from_unit",
  "redundancy_over_grid", "solve_grid",
  "strain_invariant",
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
  # The shear-thinning laws reuse `mu` for the zero-shear viscosity and `lambda1` for the
  # crossover time, because they are the same kinds of quantity measured on the same
  # material and a separate axis would only let the two disagree. What is genuinely new:
  # `cross_m` is Cross's transition sharpness, which is order one rather than a decade
  # count; `yield_pa` spans the shear modulus's own decades extended two below it, since a
  # yield stress far under the modulus cannot be told from none -- and saying so is the
  # redundancy prior's job, not the bound's (#199).
  "cross_m": (1e-1, 2.0), "yield_pa": (1e0, 1e5),
  # `thin_time` is the Carreau crossover inside a Maxwell arm, over the same decades as
  # `lambda1` because it is a time on the same material. It has its own axis rather than
  # sharing one: the crossover and the relaxation time are independent, and a shared axis
  # would assert they move together. Its low end weakly identifies `pl_n` -- with the
  # crossover out of reach the thinning factor is 1 whatever the index -- which is the
  # redundancy prior's to report, not the bound's, exactly as for `tau_ratio` above.
  "thin_time": (1e-7, 1e-3),
  # Ogden's second term. The first reuses `g` and `ab_n`; the second needs its own pair.
  # Exponents are bounded away from zero because the strain-energy form divides by them,
  # and the constructor refuses zero outright.
  "ogden_g2": (1e0, 1e5), "ogden_a2": (1e-1, 2e1),
  # `tau_ratio` is the SECOND relaxation time as a multiple of the first. It is bounded
  # BELOW because the arms are exchangeable: swapping `(lambda1, 1-w)` with `(tau2, w)` is
  # the same trajectory, so a symmetric axis would split every posterior into two identical
  # modes and charge the Occam factor for bookkeeping. Ordering the arms removes that.
  #
  # The floor sits ABOVE 1 rather than at it. At exactly 1 the arms share a timescale and
  # `share` stops doing anything at all -- measured on the 15 C fit, moving it from 0.2 to
  # 0.6 changes the trace by 8e-12, which is solver noise. Its Jacobian column is then zero,
  # `J^T J` is singular, and `laplace_log_evidence` refuses the point outright. So a floor of
  # 1 would put a face of the grid where the evidence cannot be computed. Same reason
  # `ab_n` starts at 1.1 rather than 1.
  #
  # The floor is NOT raised to where `share` clears the measurement noise, which is around
  # a ratio of 10 (1.1e-3 of Rmax at 1.1, 3.2e-2 at 10, against noise near 1.8e-2). Hiding
  # the weakly identified region in the bounds is what #199 rejected for Gent: the
  # redundancy prior exists to drive the weight down there and say so.
  "tau_ratio": (1.1, 1e3), "share": (1e-3, 9e-1),
  # `Req` as a multiple of its inferred value -- a CONFIGURATION axis, not a material one.
  # This bound is a prior, not a measurement, and it is the most consequential one here:
  # `docs/writeup/initial_state.py` finds that a 1.68% error in Req leaves the residual that
  # changing the bubble-dynamics operator does. Too wide and the operator comparison is
  # absorbed and reads as "the operators are indistinguishable"; too narrow and Req is pinned
  # at a precision the experiment does not have. +/-10% log-symmetric is the default, and
  # `req_prior.py` reports how the ranking moves as the width changes rather than asserting
  # that one width is right.
  "req_scale": (0.9, 1.0 / 0.9),
}

# The bubble-dynamics equations are a model choice like any other and the one this package
# has never compared. Every candidate above varies the MATERIAL while holding the forward
# operator at `keller-miksis`; on the 15 C record, changing only this moves the trace by four
# to fourteen times the median noise, more than any constitutive difference measured here.
#
# They are not `CandidateModel`s and should not become them: a candidate is a material, and
# this is the operator the material is pushed through. They compare through the same
# `candidate_log_evidence` by holding the candidate fixed and varying the `solve` callback,
# and because the parameter space is then identical the difference in log evidence is a clean
# Bayes factor with the Occam terms cancelling. The table itself lives in `_config` beside
# the option it names, so there is one of it.
DYNAMICS_MODELS = OPERATORS

_NEGLIGIBLE = 1e-9
# Residual returned where the material will not integrate: far above any real whitened
# misfit, so the optimiser leaves, but finite so it still has somewhere to go.
_UNREACHABLE = 1e4
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
  # a material, not an `object`: every caller that reads a field off what `build` returns --
  # a test checking the parameters went where the builder claims, most of all -- is
  # otherwise reaching into an opaque value, and the type checker is right to say so
  build: Callable[[dict], MaterialModel]
  axes: tuple[str, ...]
  contains: tuple[str, ...] = ()
  # Axes that reach the CONFIGURATION rather than the material. `Req` is the case this
  # exists for: it is inferred, not measured, and a 1.68% error in it leaves the same
  # residual as changing the bubble-dynamics operator, so pinning it asserts a precision
  # the experiment does not have. Named here, such an axis is fitted and paid for by the
  # Occam factor like any other.
  config_axes: tuple[str, ...] = ()

  def __post_init__(self) -> None:
    if stray := [a for a in self.config_axes if a not in self.axes]:
      raise ValueError(f"{self.name}: config_axes {stray} are not among its axes")

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
    # Shear-thinning viscosities. These laws were already implemented and exported but only
    # `powerlaw` had ever been made a candidate, so the rest could be simulated and never
    # compared. Each pairs the thinning law with a neo-Hookean elastic, exactly as
    # `powerlaw` does, so the set differs in ONE thing: the shape of eta(gammadot).
    #
    # Each nests into `NHKV` rather than into `powerlaw`: every one of them becomes a
    # constant viscosity in some limit -- `n = 1` for Carreau and Cross, `lambda -> 0` for
    # the Eyrings, zero yield for Bingham -- and a constant viscosity beside a neo-Hookean
    # elastic IS `NHKV`. Herschel-Bulkley is the exception and nests into `powerlaw`, which
    # it becomes at zero yield stress.
    #
    # `infinite_shear_viscosity_pa_s` is pinned at zero and Carreau's transition exponent at
    # 2, which is the classical Carreau rather than Carreau-Yasuda. Freeing either costs an
    # axis, and at 12 points an axis is a factor of 12 -- the full five-parameter law is in
    # `EXTENDED_MODELS` instead, where it is scored by expansion rather than quadrature.
    CandidateModel(
      "carreau", lambda t: InstantaneousMaterial(
        elastic=NeoHookean(t["g"]), viscous=CarreauYasuda(t["mu"], 0.0, t["lambda1"], 2.0, t["pl_n"]),
      ), ("mu", "g", "lambda1", "pl_n"), ("NHKV", "NH"),
    ),
    CandidateModel(
      "cross", lambda t: InstantaneousMaterial(
        elastic=NeoHookean(t["g"]), viscous=Cross(t["mu"], 0.0, t["lambda1"], t["cross_m"]),
      ), ("mu", "g", "lambda1", "cross_m"), ("NHKV", "NH"),
    ),
    CandidateModel(
      "eyring", lambda t: InstantaneousMaterial(
        elastic=NeoHookean(t["g"]), viscous=PowellEyring(t["mu"], 0.0, t["lambda1"]),
      ), ("mu", "g", "lambda1"), ("NHKV", "NH"),
    ),
    CandidateModel(
      "modeyring", lambda t: InstantaneousMaterial(
        elastic=NeoHookean(t["g"]), viscous=ModifiedPowellEyring(t["mu"], 0.0, t["lambda1"]),
      ), ("mu", "g", "lambda1"), ("NHKV", "NH"),
    ),
    CandidateModel(
      "herschel", lambda t: InstantaneousMaterial(
        elastic=NeoHookean(t["g"]), viscous=HerschelBulkley(t["yield_pa"], t["pl_k"], t["pl_n"]),
      ), ("g", "yield_pa", "pl_k", "pl_n"), ("powerlaw", "NH"),
    ),
    CandidateModel(
      "bingham", lambda t: InstantaneousMaterial(
        elastic=NeoHookean(t["g"]), viscous=Bingham(t["yield_pa"], t["mu"]),
      ), ("mu", "g", "yield_pa"), ("NHKV", "NH"),
    ),
  )
}

# Candidates that exist but do not belong in the grid comparison above.
#
# `solve_grid` is a Cartesian product at one count on every axis, so its cost is
# `count**dimension`: at the `GRID_COUNT = 12` the examples use, six axes is 5,971,968
# solves against 62,208 for the four-axis `qSLS`, and 35x the whole standard sweep put
# together. That is not a tuning problem, it is the shape of the method -- and the reason
# `pyimr.discriminate.laplace_log_evidence` exists. Fit these and take the evidence there.
#
# They are ordinary candidates otherwise: to score one against the models it degenerates
# into, hand `redundancy_over_grid` a merged mapping, `STANDARD_MODELS | EXTENDED_MODELS`.
EXTENDED_MODELS: dict[str, CandidateModel] = {
  m.name: m for m in (
    CandidateModel(
      "qSLS2",
      lambda t: TwoModeQuadraticZener(
        t["g"], t["mu"], t["lambda1"], 0.0, t["alpha"], t["tau_ratio"] * t["lambda1"], t["share"],
      ),
      ("mu", "g", "lambda1", "alpha", "tau_ratio", "share"), ("qSLS",),
    ),
    # The other answer to the one-mode residual: not a second timescale but a moving one.
    # `pl_n = 1` removes the thinning exactly, which is why it nests into `qSLS`.
    CandidateModel(
      "qSLSthin", lambda t: CarreauZener(
        t["g"], t["mu"], t["lambda1"], 0.0, t["alpha"], t["thin_time"], t["pl_n"],
      ), ("mu", "g", "lambda1", "alpha", "thin_time", "pl_n"), ("qSLS",),
    ),
    # Ogden is here for a different reason from the others: at four axes it would fit the
    # grid comfortably, but its variable-length tuples cannot travel through the fixed-width
    # scales vector, so a sweep compiles once PER POINT rather than once (#196). A grid of
    # 20,736 points would mean 20,736 XLA compiles. Scored by expansion there is no grid and
    # the objection disappears, which is what makes it registrable at all.
    CandidateModel(
      "ogden", lambda t: InstantaneousMaterial(
        elastic=Ogden((t["g"], t["ogden_g2"]), (t["ab_n"], t["ogden_a2"])),
      ), ("g", "ab_n", "ogden_g2", "ogden_a2"), ("NH",),
    ),
    # The full five-parameter Carreau-Yasuda: `carreau` above with the infinite-shear
    # viscosity and the transition exponent set free rather than pinned.
    CandidateModel(
      "carreauyasuda", lambda t: InstantaneousMaterial(
        elastic=NeoHookean(t["g"]),
        viscous=CarreauYasuda(t["mu"], t["mu"] * t["lam"], t["lambda1"], t["cross_m"], t["pl_n"]),
      ), ("mu", "g", "lambda1", "cross_m", "pl_n", "lam"), ("carreau", "NHKV", "NH"),
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

def _deviation_for(deviation, samples):
  """A noise scale as an array broadcastable over the samples, scalar or one per sample.

  Real records carry the second: the trial spread varies across the trace, and collapsing
  it to a single number silently reweights which parts of the curve the fit is asked to
  match. `fit_quality.py` has always used the per-sample form.
  """
  scale = np.asarray(deviation, dtype=float)
  if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0): raise ValueError("deviation must be finite and positive")
  if scale.ndim > 1 or (scale.ndim == 1 and scale.size not in (1, samples)):
    raise ValueError(f"deviation must be scalar or one per sample; got {scale.size} for {samples} observations")
  return scale

def evaluate_at(candidate, solve, fields):
  """Build the candidate at `fields` and solve it, splitting off the configuration axes.

  Every caller that turns a parameter point into a trajectory goes through here, so the
  split happens once. `solve` takes `(material, config)` whether or not the candidate has
  configuration axes -- a signature that changed with the candidate would be a `TypeError`
  waiting for whoever first writes a candidate that has them.
  """
  return solve(candidate.build(fields), {a: fields[a] for a in candidate.config_axes})


def physical_from_unit(axes, unit_point, bounds=None):
  """Parameters from a point in the unit cube: the inverse of `normalize_log_coordinates`.

  `parameter_grid` only ever goes the forward way, because the grid IS the parameters. A
  fitted point arrives the other way round -- in the coordinates the prior is uniform on --
  and has to be turned back into something a material can be built from.
  """
  bounds = PARAMETER_BOUNDS if bounds is None else bounds
  if missing := [a for a in axes if a not in bounds]: raise ValueError(f"no bounds given for {missing}")
  unit = np.asarray(unit_point, dtype=float).ravel()
  if unit.size != len(axes): raise ValueError(f"{len(axes)} axes but {unit.size} coordinates")
  lower = np.array([bounds[a][0] for a in axes], dtype=float)
  upper = np.array([bounds[a][1] for a in axes], dtype=float)
  return lower * (upper / lower) ** unit

def solve_grid(candidate, solve, *, count, bounds=None):
  """Evaluate a candidate over its grid. `solve(material, config)` returns `(radius, stress)`.

  Every point must solve. A sweep whose points can fail wants its own loop, marking the
  failures so they can be dropped from the prior -- `examples/windowed_selection.py`.
  """
  points, normalized = parameter_grid(candidate.axes, count, bounds)
  solved = [evaluate_at(candidate, solve, dict(zip(candidate.axes, row))) for row in points]
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
      solved = evaluate_at(child, solve, {a: theta[a] for a in child.axes})
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

@dataclass(frozen=True, slots=True)
class CandidateFit:
  """Where a candidate fits best, and what else the search found.

  `modes` holds the distinct endpoints, best first, because a multistart that keeps only
  its winner throws away the one diagnostic that says whether the winner means anything.
  Two endpoints of nearly equal cost in different corners of the cube is a different
  situation from one basin found eight times, and the evidence should be summed over the
  modes rather than taken at the best -- see `laplace_log_evidence`.
  """

  unit: np.ndarray
  chi_squared: float
  modes: tuple[np.ndarray, ...]
  costs: tuple[float, ...]
  converged: int
  starts: int
  evaluations: int
  failures: int

  @property
  def failure_fraction(self) -> float:
    """Share of forward solves that landed somewhere the material would not integrate.

    Near one this is not a fit, whatever `chi_squared` says: the optimiser spent its budget
    in the penalty region and stopped where it entered.
    """
    return self.failures / self.evaluations if self.evaluations else 0.0

  @property
  def multimodal(self) -> bool:
    """Whether a second basin came within one nat of the best."""
    return len(self.costs) > 1 and (self.costs[1] - self.costs[0]) < 1.0

def fit_candidate(candidate, solve, observed, deviation, *, bounds=None, starts=8, seed=0,
                  max_evaluations=200, separation=1e-2):
  """Fit a candidate by multistart least squares in the prior's unit coordinates.

  `candidate_log_evidence` expands about a fitted point and nothing produced one: a
  candidate's axes need not be material fields -- `tau_ratio` is a ratio of two of them,
  `thin_time` is absolute, `ogden_a2` is neither -- so `PreparedInference.fit_multistart`,
  which works from attribute paths, cannot reach them. This closes that gap, and with it
  the distance between a model existing and a model being rankable.

  Optimising in unit coordinates rather than in the parameters is not a convenience. The
  axes span decades, the box is exactly `[0, 1]^p` there, and it is the space the evidence
  is measured in, so the fit and the Occam factor agree about what a step means.

  Starts are a Latin hypercube plus the centre, because a uniform sample of eight points in
  six dimensions leaves whole faces empty -- the mistake that hid an E-optimality ridge
  earlier in this package. No analytic Jacobian: `least_squares` differences the residual
  itself, which costs `p` extra solves an iteration and is what makes this work for axes
  that are not material fields.

  BUDGET. That differencing makes `max_evaluations` mean far less than it appears to: an
  iteration costs `p + 1` solves, so 80 evaluations is about sixteen steps in four
  dimensions. Measured on a synthetic qSLS record, `starts=4, max_evaluations=80` reaches
  `chi_squared` of 27.3 against the generating point's 0.85 -- worse than the truth, which
  is what an under-converged fit looks like -- while `starts=6, max_evaluations=150` reaches
  0.64. Check `chi_squared` against the truth where you have one, and treat a fit that
  cannot beat it as evidence about the budget rather than about the model.
  """
  from scipy.optimize import least_squares

  bounds = PARAMETER_BOUNDS if bounds is None else bounds
  if int(starts) < 1: raise ValueError("starts must be a positive integer")
  if not 0.0 < float(separation) < 1.0: raise ValueError("separation must lie in (0, 1)")
  measured = np.asarray(observed, dtype=float).ravel()
  scale = _deviation_for(deviation, measured.size)

  # An optimiser walking a six-dimensional box WILL step somewhere the material does not
  # integrate -- Gent locks up, a collapse outruns the step budget, a constructor refuses
  # its own arguments. Raising from inside the residual kills the whole fit for one bad
  # step, so a failed evaluation returns a large finite residual instead and the optimiser
  # backs out of the region by itself. The count is kept because a fit whose evaluations
  # mostly failed is not a fit, and nothing else would say so.
  attempted = failed = 0

  def residual(point):
    nonlocal attempted, failed
    attempted += 1
    values = physical_from_unit(candidate.axes, np.clip(point, 0.0, 1.0), bounds)
    try:
      solved = evaluate_at(candidate, solve, dict(zip(candidate.axes, values, strict=True)))
    except Exception:                                # noqa: BLE001 -- an excursion, not a bug
      failed += 1
      return np.full(measured.size, _UNREACHABLE)
    if solved is None:
      failed += 1
      return np.full(measured.size, _UNREACHABLE)
    radius = np.asarray(solved[0], dtype=float).ravel()
    # a shape mismatch is the caller's mistake rather than a corner of the box, so it is
    # raised rather than penalised -- penalising it would hide it behind a bad fit
    if radius.size != measured.size:
      raise ValueError(f"solve returned {radius.size} samples against {measured.size} observations")
    if not np.all(np.isfinite(radius)):
      failed += 1
      return np.full(measured.size, _UNREACHABLE)
    return (radius - measured) / scale

  rng = np.random.default_rng(int(seed))
  count, dimension = int(starts), candidate.dimension
  strata = np.argsort(rng.random((count, dimension)), axis=0)
  points = (strata + rng.random((count, dimension))) / count
  points[0] = 0.5

  endpoints = []
  for start in points:
    outcome = least_squares(residual, start, bounds=(0.0, 1.0), max_nfev=int(max_evaluations), x_scale="jac")
    # a start that never escaped the penalty region reached no fit at all, and averaging it
    # in with the real ones would put a spurious mode in `modes`
    if np.isfinite(outcome.cost) and 2.0 * outcome.cost / measured.size < _UNREACHABLE**2:
      endpoints.append((float(outcome.cost), np.clip(np.asarray(outcome.x, dtype=float), 0.0, 1.0)))
  if not endpoints: raise ValueError(f"{candidate.name} did not fit from any of {count} starts")

  endpoints.sort(key=lambda item: item[0])
  modes, costs = [], []
  for cost, point in endpoints:
    if all(np.linalg.norm(point - kept) > float(separation) for kept in modes):
      modes.append(point)
      costs.append(cost)
  # `least_squares` reports half the sum of squares, and the residual is already whitened
  return CandidateFit(unit=modes[0], chi_squared=2.0 * costs[0] / measured.size,
                      modes=tuple(modes), costs=tuple(costs), converged=len(endpoints), starts=count,
                      evaluations=attempted, failures=failed)

def candidate_log_evidence(candidate, solve, observed, deviation, unit_point, *, bounds=None, step=1e-4):
  """`log p(Y | M)` for a candidate at a fitted point, by Laplace expansion.

  This is what makes a model comparable without a grid. `log_evidence` above quadratures
  the likelihood over the candidate's whole grid, which costs `count**dimension` and so
  runs out at four or five axes; `laplace_log_evidence` needs only the residual and the
  Jacobian at the fit, but speaks in arrays and knows nothing about candidates. This is the
  join: it turns `unit_point` back into parameters, builds the material, and differences the
  forward model to get both.

  The Jacobian is taken by central differences in the UNIT coordinates rather than in the
  parameters, because that is the space the prior is uniform on and therefore the space
  `laplace_log_evidence` wants its Occam factor measured in. Differences rather than the
  traced sensitivities because a candidate's axes need not be material fields at all --
  `qSLS2` has `tau_ratio`, a ratio of two of them, and `oldroydb` likewise. It costs
  `1 + 2*dimension` solves, which against `count**dimension` is not the expensive part.

  `solve(material, config)` returns `(radius, stress)`, the same callback `solve_grid` takes.
  """
  bounds = PARAMETER_BOUNDS if bounds is None else bounds
  unit = np.asarray(unit_point, dtype=float).ravel()
  if unit.size != candidate.dimension:
    raise ValueError(f"{candidate.name} has {candidate.dimension} axes but got {unit.size} coordinates")
  if np.any(unit < 0.0) or np.any(unit > 1.0):
    raise ValueError("the fitted point must lie in the unit cube the prior is defined on")
  measured = np.asarray(observed, dtype=float).ravel()
  scale = _deviation_for(deviation, measured.size)
  width = float(step)
  if not 0.0 < width < 0.5: raise ValueError("step must lie in (0, 0.5)")

  def trace(point):
    values = physical_from_unit(candidate.axes, point, bounds)
    # a solver that RAISES and one that returns `None` mean the same thing here -- the
    # expansion cannot be taken at this point -- so they leave by the same door. Callers
    # sum over modes and drop the ones that will not score; that is only possible if the
    # failure is one catchable kind rather than whatever the integrator happened to throw.
    try:
      solved = evaluate_at(candidate, solve, dict(zip(candidate.axes, values, strict=True)))
    except Exception as error:                       # noqa: BLE001
      raise ValueError(f"{candidate.name} does not solve at {dict(zip(candidate.axes, values))}: {error}") from error
    if solved is None: raise ValueError(f"{candidate.name} does not solve at {dict(zip(candidate.axes, values))}")
    radius = np.asarray(solved[0], dtype=float).ravel()
    if radius.size != measured.size:
      raise ValueError(f"solve returned {radius.size} samples against {measured.size} observations")
    return radius

  residual = (trace(unit) - measured) / scale
  jacobian = np.empty((residual.size, unit.size))
  for index in range(unit.size):
    # one-sided at a bound rather than stepping outside it: a fit sitting exactly on a
    # boundary is common, and the material may not even exist on the other side
    low, high = unit.copy(), unit.copy()
    low[index], high[index] = max(unit[index] - width, 0.0), min(unit[index] + width, 1.0)
    jacobian[:, index] = (trace(high) - trace(low)) / (scale * (high[index] - low[index]))
  # capped, because candidates are compared ACROSS dimensions and the uncapped Occam factor
  # pays a model for parameters its data cannot see -- see `laplace_log_evidence`
  return laplace_log_evidence(residual, jacobian, scale, cap_at_prior=True)

def compare(log_evidences):
  """Normalized posterior over a model set, keyed as given."""
  names = list(log_evidences)
  return dict(zip(names, model_posterior(np.array([log_evidences[n] for n in names]))))
