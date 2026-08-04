"""Model comparison driver: model-set invariants and the grid it quadratures over.

The nesting declarations fail silently. `redundancy_over_grid` builds each contained model
from the parent's parameters, so a `contains` naming the wrong model yields a plausible
number rather than an error.
"""

import numpy as np
import pytest

from pyimr.selection import (
  PARAMETER_BOUNDS,
  bounds_for_invariant,
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


def _sensitive(material):
  """Cheap stand-in forward model that actually varies with the parameters."""
  value = material.viscosity_pa_s + material.shear_modulus_pa
  return _GRID * (1.0 + value), _GRID * (1.0 + value)


def test_every_contained_model_is_a_genuine_restriction():
  """`redundancy_over_grid` builds children from the parent's parameters, so a child axis
  the parent lacks raises only once that combination is reached.
  """
  for name, candidate in STANDARD_MODELS.items():
    assert candidate.name == name
    for child_name in candidate.contains:
      assert child_name in STANDARD_MODELS, f"{name} contains unknown {child_name!r}"
      assert set(STANDARD_MODELS[child_name].axes) < set(candidate.axes), f"{child_name} is not a restriction of {name}"


def test_every_model_builds_from_exactly_its_own_axes():
  for candidate in STANDARD_MODELS.values():
    theta = {a: float(np.sqrt(PARAMETER_BOUNDS[a][0] * PARAMETER_BOUNDS[a][1])) for a in candidate.axes}
    assert candidate.build(theta) is not None


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
  identical = lambda _material: (_GRID, _GRID)  # noqa: E731

  points, _, _, stresses = solve_grid(candidate, identical, count=4)
  redundancies = redundancy_over_grid(candidate, models, points, stresses, identical)
  measured("identical child", f"max w_red={redundancies.max():.2e} over {len(points)} points")
  assert np.all(redundancies < 1e-9)


def test_an_unsolvable_child_leaves_the_weight_alone():
  """A child that will not integrate is no evidence of redundancy, so it must not be
  scored as either redundant or distinguishable.
  """
  points, _, _, stresses = solve_grid(STANDARD_MODELS["NHKV"], _sensitive, count=3)
  identical = lambda _material: (_GRID, _GRID)  # noqa: E731
  assert np.all(redundancy_over_grid(STANDARD_MODELS["NHKV"], STANDARD_MODELS, points, stresses, identical) < 1.0)
  assert np.all(redundancy_over_grid(STANDARD_MODELS["NHKV"], STANDARD_MODELS, points, stresses, lambda _m: None) == 1.0)


def test_a_model_that_differs_from_its_child_keeps_its_prior():
  points, _, _, stresses = solve_grid(STANDARD_MODELS["NHKV"], _sensitive, count=3)
  # a flat child: a different SHAPE, since eqn 22 aligns amplitudes and would score a pure
  # rescaling as redundant. Flat also keeps the child's own stress scale tiny, so the
  # difference is well above what it can resolve and the penalty is not earned.
  divergent = lambda _material: (_GRID, np.ones_like(_GRID))  # noqa: E731
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
