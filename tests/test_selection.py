"""Model comparison driver: model-set invariants and the grid it quadratures over.

The nesting declarations fail silently. `redundancy_over_grid` builds each contained model
from the parent's parameters, so a `contains` naming the wrong model yields a plausible
number rather than an error.
"""

import numpy as np
import pytest

from pyimr.selection import (
  EXTENDED_MODELS,
  PARAMETER_BOUNDS,
  bounds_for_invariant,
  candidate_log_evidence,
  physical_from_unit,
  strain_invariant,
  STANDARD_MODELS,
  CandidateModel,
  compare,
  log_evidence,
  parameter_grid,
  redundancy_over_grid,
  solve_grid,
)

SECTION = "16. Model comparison driver"

_GRID = np.linspace(1.0, 2.0, 6)


def _sensitive(material, _config=None):
  """Cheap stand-in forward model that actually varies with the parameters."""
  value = material.viscosity_pa_s + material.shear_modulus_pa
  return _GRID * (1.0 + value), _GRID * (1.0 + value)


_REGISTRY = STANDARD_MODELS | EXTENDED_MODELS


def test_every_contained_model_is_a_genuine_restriction():
  """`redundancy_over_grid` builds children from the parent's parameters, so a child axis
  the parent lacks raises only once that combination is reached.

  Over the merged registry, because an extended candidate nests into a standard one and
  neither mapping can check that on its own.
  """
  for name, candidate in _REGISTRY.items():
    assert candidate.name == name
    for child_name in candidate.contains:
      assert child_name in _REGISTRY, f"{name} contains unknown {child_name!r}"
      assert set(_REGISTRY[child_name].axes) < set(candidate.axes), f"{child_name} is not a restriction of {name}"


def test_every_model_builds_from_exactly_its_own_axes():
  for candidate in _REGISTRY.values():
    theta = {a: float(np.sqrt(PARAMETER_BOUNDS[a][0] * PARAMETER_BOUNDS[a][1])) for a in candidate.axes}
    assert candidate.build(theta) is not None


def test_every_axis_of_every_model_has_bounds():
  # a candidate naming an axis `PARAMETER_BOUNDS` lacks fails only inside a grid sweep,
  # which is a long way from the typo
  for candidate in _REGISTRY.values():
    for axis in candidate.axes:
      assert axis in PARAMETER_BOUNDS, f"{candidate.name} has no bounds for {axis!r}"


def test_the_two_mode_candidate_wires_its_parameters_where_it_claims():
  """A builder that transposed two arguments would still build, still solve, and be wrong.

  Six positional arguments, two of them new, and `shear_modulus`/`viscosity` adjacent in
  the signature but four orders of magnitude apart in the bounds -- so this reads the
  fields back rather than trusting the call.
  """
  import pyimr

  theta = {"mu": 0.05, "g": 200.0, "lambda1": 2e-7, "alpha": 5.0, "tau_ratio": 10.0, "share": 0.3}
  material = EXTENDED_MODELS["qSLS2"].build(theta)
  assert isinstance(material, pyimr.TwoModeQuadraticZener)
  assert material.shear_modulus_pa == 200.0
  assert material.viscosity_pa_s == 0.05
  assert material.relaxation_time_s == 2e-7
  assert material.stiffening == 5.0
  assert material.second_share == 0.3
  # the ratio is relative to the first time, not absolute
  assert material.second_relaxation_time_s == pytest.approx(2e-6)


def test_the_second_arm_is_never_faster_than_the_first():
  """The arms are exchangeable, so the axis is ordered to kill the label-switching mode.

  Were `tau_ratio` allowed below 1, `(lambda1, 1-w)` and `(tau2, w)` would name the same
  trajectory twice and the Occam factor would be charged for bookkeeping. Checked at both
  ends of the axis, since only the lower one carries the constraint.
  """
  import pyimr

  lower, upper = PARAMETER_BOUNDS["tau_ratio"]
  assert lower > 1.0, "at a ratio of exactly 1 the arms share a timescale and `share` dies"
  for ratio in (lower, upper):
    theta = {"mu": 0.05, "g": 200.0, "lambda1": 2e-7, "alpha": 5.0, "tau_ratio": ratio, "share": 0.3}
    material = EXTENDED_MODELS["qSLS2"].build(theta)
    assert isinstance(material, pyimr.TwoModeQuadraticZener)
    assert material.second_relaxation_time_s >= material.relaxation_time_s


def test_the_share_axis_stays_inside_what_the_material_will_accept():
  # `second_share` is a fraction in [0, 1) and the constructor enforces it; a bound at or
  # above 1 would fail only at the last grid point of a long sweep
  lower, upper = PARAMETER_BOUNDS["share"]
  assert 0.0 <= lower < upper < 1.0
  for share in (lower, upper):
    theta = {"mu": 0.05, "g": 200.0, "lambda1": 2e-7, "alpha": 5.0, "tau_ratio": 10.0, "share": share}
    assert EXTENDED_MODELS["qSLS2"].build(theta) is not None


def test_the_extended_models_are_all_genuinely_beyond_the_grid():
  """Membership is not a taste call, and there are exactly two grounds for it.

  Either the candidate has more axes than the grid can afford --- `solve_grid` is
  `count**dimension`, and at the `GRID_COUNT = 12` the examples use, six axes is 5,971,968
  solves against 168,072 for the entire standard sweep --- or its numbers cannot travel
  through the fixed-width scales vector, so a sweep compiles once per POINT instead of
  once, which is Ogden's situation and is independent of how many axes it has.

  A candidate failing both grounds belongs in `STANDARD_MODELS`, and this test should be
  what stops it being parked here instead.
  """
  from pyimr._integrate import shares_one_program

  assert not (set(EXTENDED_MODELS) & set(STANDARD_MODELS))
  widest = max(c.dimension for c in STANDARD_MODELS.values())
  for candidate in EXTENDED_MODELS.values():
    theta = {a: float(np.sqrt(PARAMETER_BOUNDS[a][0] * PARAMETER_BOUNDS[a][1])) for a in candidate.axes}
    too_wide = candidate.dimension > widest
    recompiles = not shares_one_program(candidate.build(theta))
    assert too_wide or recompiles, f"{candidate.name} fits the grid on both counts; move it"


def test_the_grid_normalizes_inside_the_unit_box_for_bounds_that_are_not_round(measured):
  """`logspace` does not reproduce its own endpoints: through log10 and back, a bound of
  24002.829853450417 comes out 1.1e-11 low and normalizes to -3.9e-16, which the prior's
  non-negativity guard rejects. Latent until bounds came from data, because every default
  bound is a power of ten and round-trips exactly.
  """
  awkward = {"gent_jm": (24002.829853450417, 2400282.985345042), "g": (1e2, 1e5)}
  points, normalized = parameter_grid(("gent_jm", "g"), 10, awkward)
  measured("awkward bounds", f"min {normalized.min():.2e}, max {normalized.max():.6f}")
  assert normalized.min() >= 0.0 and normalized.max() <= 1.0
  # and still spans the box rather than being clipped into uselessness
  assert normalized.min() == pytest.approx(0.0, abs=1e-12)
  assert normalized.max() == pytest.approx(1.0, abs=1e-12)
  assert points[:, 0].min() == pytest.approx(awkward["gent_jm"][0])


def test_the_grid_is_log_spaced_and_normalized(measured):
  points, normalized = parameter_grid(("mu", "g"), 5)
  assert points.shape == (25, 2) and normalized.shape == (25, 2)
  assert np.all(normalized >= 0.0) and np.all(normalized <= 1.0)
  assert points[:, 0].min() == pytest.approx(PARAMETER_BOUNDS["mu"][0])
  assert points[:, 1].max() == pytest.approx(PARAMETER_BOUNDS["g"][1])

  distinct = np.unique(points[:, 1])
  ratios = distinct[1:] / distinct[:-1]
  measured("grid spacing", f"ratio spread {ratios.max() - ratios.min():.2e}")
  assert np.allclose(ratios, ratios[0])


def test_the_strain_invariant_reads_the_collapse_not_the_expansion():
  """`_elastic_integrand` uses `lam**-4 + 2*lam**2 - 3`, so the deepest COMPRESSION governs.
  Reading the expansion instead is what made the first attempt at these bounds miss: on
  gelatin it gives 44-52 where the collapse gives 24-119 (#199).
  """
  equilibrium = 1.0 / 7.09
  trace = np.array([1.0, 0.5, 0.056, 0.4])
  span = strain_invariant(trace, equilibrium)
  lam = 0.056 / equilibrium
  assert span == pytest.approx(lam**-4 + 2 * lam**2 - 3)
  assert span > (1.0 / equilibrium) ** 2 + 2 * equilibrium - 3 - 20, "compression must dominate here"


@pytest.mark.parametrize("span", [24.0, 37.4, 118.7])
def test_the_divergence_limited_bounds_scale_with_the_invariant(span, measured):
  """Both axes are multiples of the span, so the normalized coordinate the prior sees means
  one thing across datasets -- which an absolute `Jm` never did.
  """
  bounds = bounds_for_invariant(span)
  assert bounds["gent_jm"][0] > span, "the smallest Jm must exceed the lock-up limit"
  assert bounds["gent_jm"][0] / span == pytest.approx(bounds_for_invariant(24.0)["gent_jm"][0] / 24.0)
  assert bounds["fung_b"][1] * span == pytest.approx(bounds_for_invariant(24.0)["fung_b"][1] * 24.0)
  measured(f"span {span}", f"Jm floor {bounds['gent_jm'][0]:.4g}")

  for axis in ("mu", "g", "lambda1", "alpha"):
    assert bounds[axis] == PARAMETER_BOUNDS[axis]


def test_bounds_for_invariant_refuses_a_non_positive_span():
  with pytest.raises(ValueError, match="strain invariant must be positive"):
    bounds_for_invariant(0.0)


@pytest.mark.parametrize(
  ("axes", "count", "message"),
  [(("mu",), 1, "count must be at least 2"), (("nonsense",), 4, "no bounds given")],
)
def test_the_grid_refuses_malformed_requests(axes, count, message):
  with pytest.raises(ValueError, match=message):
    parameter_grid(axes, count)


def test_solve_grid_covers_every_point():
  points, normalized, radii, stresses = solve_grid(STANDARD_MODELS["NHKV"], _sensitive, count=4)
  assert points.shape == (16, 2) and normalized.shape == (16, 2)
  assert radii.shape == (16, 6) and stresses.shape == (16, 6)


def test_a_model_identical_to_its_child_is_fully_redundant(measured):
  """The property the prior exists for: here the forward model ignores the extra
  parameter, so the parent reproduces its child everywhere, not just at a few points.
  """
  candidate = CandidateModel("parent", STANDARD_MODELS["NHKV"].build, ("mu", "g"), ("newtonian",))
  models = {"parent": candidate, "newtonian": STANDARD_MODELS["newtonian"]}
  identical = lambda _material, _config=None: (_GRID, _GRID)  # noqa: E731

  points, _, _, stresses = solve_grid(candidate, identical, count=4)
  redundancies = redundancy_over_grid(candidate, models, points, stresses, identical)
  measured("identical child", f"max w_red={redundancies.max():.2e} over {len(points)} points")
  assert np.all(redundancies < 1e-9)


def test_an_unsolvable_child_leaves_the_weight_alone():
  """A child that will not integrate is no evidence of redundancy, so it must not be
  scored as either redundant or distinguishable.
  """
  points, _, _, stresses = solve_grid(STANDARD_MODELS["NHKV"], _sensitive, count=3)
  identical = lambda _material, _config=None: (_GRID, _GRID)  # noqa: E731
  assert np.all(redundancy_over_grid(STANDARD_MODELS["NHKV"], STANDARD_MODELS, points, stresses, identical) < 1.0)
  assert np.all(redundancy_over_grid(STANDARD_MODELS["NHKV"], STANDARD_MODELS, points, stresses, lambda _m, _c=None: None) == 1.0)


def test_a_model_that_differs_from_its_child_keeps_its_prior():
  points, _, _, stresses = solve_grid(STANDARD_MODELS["NHKV"], _sensitive, count=3)
  # a flat child: a different SHAPE, since eqn 22 aligns amplitudes and would score a pure
  # rescaling as redundant. Flat also keeps the child's own stress scale tiny, so the
  # difference is well above what it can resolve and the penalty is not earned.
  divergent = lambda _material, _config=None: (_GRID, np.ones_like(_GRID))  # noqa: E731
  assert np.all(redundancy_over_grid(STANDARD_MODELS["NHKV"], STANDARD_MODELS, points, stresses, divergent) > 0.9)


def test_the_evidence_prefers_the_grid_point_that_fits(measured):
  points, normalized, radii, _ = solve_grid(STANDARD_MODELS["NHKV"], _sensitive, count=4)
  redundancies, deviations = np.ones(len(points)), np.full(radii.shape[1], 0.01)

  observed = radii[5][None, :].copy()
  close, chi_squared = log_evidence(radii, normalized, redundancies, observed, deviations, dimension=2)
  assert int(np.argmin(chi_squared)) == 5, "the generating point must fit best"

  far, _ = log_evidence(radii, normalized, redundancies, observed + 5.0, deviations, dimension=2)
  measured("evidence vs fit", f"matched {close:.1f} vs displaced {far:.1f}")
  assert close > far


def test_the_evidence_rejects_mismatched_samples():
  points, normalized, radii, _ = solve_grid(STANDARD_MODELS["NHKV"], _sensitive, count=3)
  with pytest.raises(ValueError, match="share a sample axis"):
    log_evidence(radii, normalized, np.ones(len(points)), np.zeros((1, 3)), np.ones(3), dimension=2)


def test_compare_returns_a_normalized_distribution():
  posterior = compare({"a": -10.0, "b": -12.0, "c": -30.0})
  assert sum(posterior.values()) == pytest.approx(1.0)
  assert posterior["a"] > posterior["b"] > posterior["c"]


def test_the_unit_map_inverts_the_normalization():
  """`physical_from_unit` is the other half of `normalize_log_coordinates`; a mismatch here
  would move every fitted point silently, since both directions look plausible alone.
  """
  from pyimr.prior import normalize_log_coordinates

  axes = ("mu", "g", "lambda1", "alpha", "tau_ratio", "share")
  unit = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 0.375])
  values = physical_from_unit(axes, unit)
  back = np.array([normalize_log_coordinates(v, *PARAMETER_BOUNDS[a]) for v, a in zip(values, axes, strict=True)])
  assert np.allclose(back, unit, atol=1e-12)
  # and the endpoints are the bounds themselves, not merely close to them
  assert physical_from_unit(("g",), [0.0])[0] == pytest.approx(PARAMETER_BOUNDS["g"][0])
  assert physical_from_unit(("g",), [1.0])[0] == pytest.approx(PARAMETER_BOUNDS["g"][1])


def test_the_unit_map_refuses_a_mismatched_point():
  with pytest.raises(ValueError, match="coordinates"):
    physical_from_unit(("g", "mu"), [0.5])
  with pytest.raises(ValueError, match="no bounds"):
    physical_from_unit(("nonesuch",), [0.5])


def _qsls_solve(times, rtol=1e-10):
  import pyimr

  def solve(material, _config):
    config = pyimr.SimulationConfig(277e-6, 277e-6 / 7.09, material, dynamics="keller-miksis",
                                    rtol=rtol, atol=rtol * 1e-2, max_steps=800_000)
    radius = np.asarray(pyimr.simulate(times, config).radius_ratio, dtype=float)
    return radius, radius

  return solve


def test_the_candidate_evidence_agrees_with_the_traced_sensitivities(measured):
  """The Jacobian is differenced, so it needs checking against a different mechanism.

  Forward-mode sensitivities give `dR/dtheta` directly; the chain rule through the log
  normalization gives the same Jacobian in unit coordinates, `theta * dR/dtheta *
  log(hi/lo)`. Feeding that to `laplace_log_evidence` is an independent route to the same
  number, so agreement is evidence about the differencing rather than a restatement of it.

  The second assertion is the load-bearing one: a difference that SHRINKS with the step is
  discretization error, and one that does not is a bug wearing its clothes.
  """
  import pyimr
  from pyimr.discriminate import laplace_log_evidence
  from pyimr.prior import normalize_log_coordinates

  times = np.linspace(0.0, 4e-5, 60)
  fit = {"mu": 0.04651, "g": 204.3, "lambda1": 1.964e-7, "alpha": 5.301}
  paths = {"mu": "material.viscosity_pa_s", "g": "material.shear_modulus_pa",
           "lambda1": "material.relaxation_time_s", "alpha": "material.stiffening"}
  sigma = 0.02
  candidate, solve = STANDARD_MODELS["qSLS"], _qsls_solve(times)
  unit = np.array([normalize_log_coordinates(fit[a], *PARAMETER_BOUNDS[a]) for a in candidate.axes])

  material = candidate.build(fit)
  clean = solve(material, {})[0]
  observed = clean + 0.004 * np.sin(np.arange(times.size))   # the fit must not be exact

  config = pyimr.SimulationConfig(277e-6, 277e-6 / 7.09, material, dynamics="keller-miksis",
                                  rtol=1e-10, atol=1e-12, max_steps=800_000)
  traced = np.asarray(pyimr.prepare(config).solve_with_sensitivities(
    times, tuple(paths[a] for a in candidate.axes)).radius_ratio, dtype=float)
  spans = np.array([np.log(PARAMETER_BOUNDS[a][1] / PARAMETER_BOUNDS[a][0]) for a in candidate.axes])
  reference = laplace_log_evidence((clean - observed) / sigma,
                                   traced * np.array([fit[a] for a in candidate.axes]) * spans / sigma, sigma,
                                   cap_at_prior=True)

  coarse = abs(candidate_log_evidence(candidate, solve, observed, sigma, unit, step=1e-3) - reference)
  default = abs(candidate_log_evidence(candidate, solve, observed, sigma, unit) - reference)
  measured("evidence vs traced", f"log Z = {reference:.4f}, |diff| {default:.2e} at the default step")
  assert default < 1e-2, "the differenced Jacobian disagrees with the traced one"
  assert default < 0.2 * coarse, "a difference that does not shrink with the step is not discretization"


def test_the_candidate_evidence_prefers_the_point_that_fits():
  times = np.linspace(0.0, 4e-5, 40)
  candidate, solve = STANDARD_MODELS["qSLS"], _qsls_solve(times, rtol=1e-9)
  fit = {"mu": 0.04651, "g": 204.3, "lambda1": 1.964e-7, "alpha": 5.301}
  from pyimr.prior import normalize_log_coordinates
  unit = np.array([normalize_log_coordinates(fit[a], *PARAMETER_BOUNDS[a]) for a in candidate.axes])
  observed = solve(candidate.build(fit), {})[0]

  matched = candidate_log_evidence(candidate, solve, observed, 0.02, unit)
  displaced = candidate_log_evidence(candidate, solve, observed, 0.02, np.clip(unit + 0.05, 0.0, 1.0))
  assert matched > displaced, "the evidence must fall away from the fit"


def test_the_candidate_evidence_refuses_a_point_it_cannot_use():
  candidate = STANDARD_MODELS["qSLS"]
  observed = np.ones(6)
  with pytest.raises(ValueError, match="coordinates"):
    candidate_log_evidence(candidate, _sensitive, observed, 1.0, [0.5, 0.5])
  with pytest.raises(ValueError, match="unit cube"):
    candidate_log_evidence(candidate, _sensitive, observed, 1.0, [0.5, 0.5, 0.5, 1.5])
  with pytest.raises(ValueError, match="deviation"):
    candidate_log_evidence(candidate, _sensitive, observed, 0.0, [0.5, 0.5, 0.5, 0.5])
  with pytest.raises(ValueError, match="step"):
    candidate_log_evidence(candidate, _sensitive, observed, 1.0, [0.5] * 4, step=0.9)


def test_the_evidence_reports_an_unsolvable_point_as_one_kind_of_failure():
  """A raising solver and a `None`-returning one mean the same thing and must leave alike.

  They did not: `None` gave a `ValueError` while the integrator's own `SimulationError`
  propagated untouched, so a caller summing evidence over modes and dropping the ones that
  will not score had to know which exception the solver happened to raise. That difference
  killed a real run.
  """
  candidate = STANDARD_MODELS["qSLS"]
  observed, unit = np.ones(6), np.full(4, 0.5)

  def raises(_material): raise RuntimeError("the maximum number of solver steps was reached")

  for solver in (raises, lambda _m: None):
    with pytest.raises(ValueError, match="does not solve"):
      candidate_log_evidence(candidate, solver, observed, 1.0, unit)
