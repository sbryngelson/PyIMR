"""Multi-objective design: domination, scalarisation, and the ParEGO search."""

import numpy as np
import pytest

from pyimr.pareto import ParetoResult, chebyshev_scalarize, explore_tradeoff, pareto_indices


def test_pareto_indices_drops_dominated_rows():
  values = np.array([[1.0, 1.0], [2.0, 2.0], [0.5, 3.0], [0.0, 0.0]])
  assert sorted(pareto_indices(values).tolist()) == [1, 2]


def test_pareto_indices_keeps_duplicates():
  # Equal rows do not dominate each other: domination needs a strict improvement somewhere.
  assert sorted(pareto_indices(np.array([[1.0, 2.0], [1.0, 2.0]])).tolist()) == [0, 1]


def test_pareto_indices_keeps_a_row_that_ties_on_one_objective():
  values = np.array([[1.0, 5.0], [1.0, 4.0], [3.0, 1.0]])
  assert sorted(pareto_indices(values).tolist()) == [0, 2]


# The front of f1 = x^2, f2 = 1 - x is f2 = 1 - sqrt(f1): it bulges toward the origin, so
# for maximisation it is NON-convex. A weighted sum w*x^2 + (1-w)*(1-x) is convex in x,
# hence maximised at an endpoint for every weight -- the interior is unreachable. This is
# the case the Chebyshev form exists for, so both halves are asserted.
_GRID = np.linspace(0.0, 1.0, 401)
_NONCONVEX = np.column_stack([_GRID**2, 1.0 - _GRID])


@pytest.mark.parametrize("weight", [0.2, 0.35, 0.5, 0.65, 0.8])
def test_weighted_sum_only_ever_returns_an_endpoint(weight):
  best = int(np.argmax(_NONCONVEX @ np.array([weight, 1.0 - weight])))
  assert best in (0, len(_GRID) - 1)


def test_chebyshev_reaches_the_interior_of_a_nonconvex_front():
  best = int(np.argmax(chebyshev_scalarize(_NONCONVEX, [0.5, 0.5])))
  # max of min(x^2, 1-x) sits where x^2 = 1-x, the golden ratio conjugate.
  assert _GRID[best] == pytest.approx((np.sqrt(5.0) - 1.0) / 2.0, abs=0.01)
  assert 0 < best < len(_GRID) - 1


@pytest.mark.parametrize("objective", [0, 1])
def test_chebyshev_unit_weight_recovers_the_single_objective_optimum(objective):
  # The guided search spends its first iterations on these corners to pin the ends of the
  # front, so a unit weight that does not reduce to plain single-objective scoring makes
  # those iterations useless.
  weights = np.eye(2)[objective]
  assert np.argmax(chebyshev_scalarize(_NONCONVEX, weights)) == np.argmax(_NONCONVEX[:, objective])


def test_chebyshev_weights_slide_the_choice_along_the_front():
  toward_first = _GRID[int(np.argmax(chebyshev_scalarize(_NONCONVEX, [0.9, 0.1])))]
  toward_second = _GRID[int(np.argmax(chebyshev_scalarize(_NONCONVEX, [0.1, 0.9])))]
  assert toward_first > toward_second


def test_chebyshev_rescales_columns_to_a_common_range():
  # Same front, second objective inflated by 1000x. Without rescaling the min would be
  # decided by units alone and the answer would move.
  stretched = _NONCONVEX * np.array([1.0, 1000.0])
  assert np.argmax(chebyshev_scalarize(stretched, [0.5, 0.5])) == np.argmax(
    chebyshev_scalarize(_NONCONVEX, [0.5, 0.5]))


def test_chebyshev_rejects_a_weight_mismatch():
  with pytest.raises(ValueError, match="2 objectives and 3 weights"):
    chebyshev_scalarize(_NONCONVEX, [0.3, 0.3, 0.4])


@pytest.mark.parametrize("weights", [[0.0, 0.0], [-0.5, 1.5]])
def test_chebyshev_rejects_degenerate_weights(weights):
  with pytest.raises(ValueError, match="non-negative"):
    chebyshev_scalarize(_NONCONVEX, weights)


def _conflicting(point):
  """Two objectives pulling opposite ways in the first coordinate."""
  x = float(point[0])
  return [-(x**2), -((x - 1.0) ** 2)]


def test_explore_tradeoff_recovers_both_ends_of_a_conflicting_front():
  result = explore_tradeoff(_conflicting, [[0.0, 1.0]], evaluations=18, initial=6, seed=0)
  assert isinstance(result, ParetoResult)
  assert result.front.size > 1, "a genuine conflict must leave more than one design undominated"
  assert float(result.best_for(0)[0]) < 0.25
  assert float(result.best_for(1)[0]) > 0.75


def test_explore_tradeoff_front_is_actually_undominated():
  result = explore_tradeoff(_conflicting, [[0.0, 1.0]], evaluations=16, initial=6, seed=3)
  values = result.front_values
  for row in values:
    beaten = np.all(values >= row, axis=1) & np.any(values > row, axis=1)
    assert not beaten.any()
  assert result.front_points.shape[0] == values.shape[0]


def test_explore_tradeoff_keeps_infeasible_designs_out_of_the_front():
  def sometimes_fails(point):
    x = float(point[0])
    if x > 0.6: return [np.nan, np.nan]
    return [-(x**2), -((x - 1.0) ** 2)]

  result = explore_tradeoff(sometimes_fails, [[0.0, 1.0]], evaluations=16, initial=8, seed=1)
  assert not result.feasible.all(), "the test needs some design to fail"
  assert np.all(result.points[result.front][:, 0] <= 0.6)
  assert result.feasible[result.front].all()
  assert result.values.shape[0] == result.points.shape[0], "infeasible designs stay in the archive"


@pytest.mark.parametrize("dimension", [1, 2])
def test_explore_tradeoff_initial_design_stratifies_every_axis(dimension):
  # Ten uniform draws over the qSLS design plane put seven above 650 um and never sampled
  # the narrow E-optimality ridge, reporting 0.13 where the ridge holds 1.84. A Latin
  # hypercube puts exactly one point in each of the n strata of every axis, so no
  # coordinate can be left with a gap wide enough to hide a feature.
  initial = 10
  result = explore_tradeoff(lambda point: [-float(point[0]), float(point[0])],
                            [[0.0, 1.0]] * dimension, evaluations=initial, initial=initial, seed=0)
  start = result.points[:initial]
  assert start.shape == (initial, dimension)
  for axis in range(dimension):
    occupied = np.floor(np.clip(start[:, axis], 0.0, 1.0 - 1e-12) * initial).astype(int)
    assert sorted(occupied.tolist()) == list(range(initial)), f"axis {axis} left a stratum empty"


def test_explore_tradeoff_reports_the_weights_it_drew():
  result = explore_tradeoff(_conflicting, [[0.0, 1.0]], evaluations=14, initial=6, seed=0)
  assert result.weights.shape == (8, 2)
  # The first guided steps are the pure single-objective corners, which pin the ends.
  assert np.allclose(result.weights[:2], np.eye(2))
  assert np.all(result.weights >= 0.0)


def test_explore_tradeoff_is_reproducible_under_a_seed():
  first = explore_tradeoff(_conflicting, [[0.0, 1.0]], evaluations=14, initial=6, seed=7)
  second = explore_tradeoff(_conflicting, [[0.0, 1.0]], evaluations=14, initial=6, seed=7)
  assert np.allclose(first.points, second.points)
  assert np.allclose(first.values, second.values)


def test_explore_tradeoff_rejects_a_scalar_objective():
  with pytest.raises(ValueError, match="at least two values"):
    explore_tradeoff(lambda point: [float(point[0])], [[0.0, 1.0]], evaluations=8, initial=4)


def test_explore_tradeoff_refuses_when_too_little_is_feasible():
  with pytest.raises(ValueError, match="feasible"):
    explore_tradeoff(lambda point: [np.nan, np.nan], [[0.0, 1.0]], evaluations=8, initial=4)


@pytest.mark.parametrize(("bounds", "message"), [
  ([[1.0, 0.0]], "upper bound"),
  ([[0.0, 1.0, 2.0]], "dimension, 2"),
])
def test_explore_tradeoff_validates_bounds(bounds, message):
  with pytest.raises(ValueError, match=message):
    explore_tradeoff(_conflicting, bounds, evaluations=8, initial=4)


@pytest.mark.parametrize(("initial", "evaluations", "message"), [(1, 8, "at least two"), (6, 4, "must cover")])
def test_explore_tradeoff_validates_the_budget(initial, evaluations, message):
  with pytest.raises(ValueError, match=message):
    explore_tradeoff(_conflicting, [[0.0, 1.0]], evaluations=evaluations, initial=initial)
