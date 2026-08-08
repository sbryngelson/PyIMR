"""Turning a design measure into a number of runs.

`optimal_measure` certifies optimality over MEASURES. An experiment runs integers, and an
exact design is not a measure, so the certificate does not reach it. `apportion` closes that
gap with the efficient rounding of Pukelsheim and Rieder, and -- because that rule carries no
guarantee at small `runs`, which is the regime IMR reaches -- reports the D-efficiency it
actually achieved rather than asserting one.

The gates here are the two that can fail quietly: an allocation that does not sum to the
budget, and an efficiency number that is not the efficiency. Both are checked against closed
forms rather than against a previous run.
"""

import numpy as np
import pytest

from pyimr.measure import apportion, optimal_measure

# Cubic regression on [-1, 1]. The D-optimal measure is four support points of weight 1/4, so
# every closed form below is exact and the arithmetic can be checked by hand.
GRID = np.linspace(-1.0, 1.0, 41)
CUBIC = np.array([np.outer(v, v) for v in
                  np.vstack([np.ones_like(GRID), GRID, GRID**2, GRID**3]).T])


@pytest.fixture(scope="module")
def measure():
  got = optimal_measure(CUBIC, tolerance=1e-10, iterations=200_000)
  assert got.certified, "the fixture's premise is that this measure is certified"
  return got


def test_the_measure_is_the_known_four_point_design(measure):
  assert measure.support.size == 4
  assert np.allclose(measure.weights[measure.support], 0.25, atol=1e-3)


@pytest.mark.parametrize("runs", [4, 5, 6, 7, 8, 11, 12, 19, 40, 101])
def test_the_allocation_spends_exactly_the_budget(measure, runs):
  got = apportion(measure.weights, runs)
  assert int(got.counts.sum()) == runs
  assert np.all(got.counts >= 0)
  assert np.all(got.counts[got.support] >= 1), "a support point must not be rounded away"
  assert int(np.sum(got.counts[got.support])) == runs, "runs must go only to the support"


@pytest.mark.parametrize("runs", [4, 8, 12, 20, 40, 100])
def test_a_budget_that_divides_evenly_loses_nothing(measure, runs):
  """With four equal weights, any multiple of four reproduces the measure exactly."""
  got = apportion(measure.weights, runs, CUBIC)
  assert np.all(got.counts[got.support] == runs // 4)
  assert got.efficiency == pytest.approx(1.0, rel=1e-9)


def test_the_efficiency_is_the_efficiency():
  """Checked against the closed form, not against a stored number.

  For a design on `p` distinct points with information `V diag(w) V^T`, the determinant is
  `det(V)^2 prod(w)`, so the D-efficiency of an exact design against the measure is
  `(prod(n_i/runs) / prod(w_i))^(1/p)` and `det(V)` cancels. At six runs on four equal
  weights the allocation is 2/2/1/1, giving `((2/6)^2 (1/6)^2 / (1/4)^4)^(1/4) = 0.9428`.
  """
  measure = optimal_measure(CUBIC, tolerance=1e-10, iterations=200_000)
  got = apportion(measure.weights, 6, CUBIC)
  assert sorted(got.counts[got.support].tolist()) == [1, 1, 2, 2]
  closed = float((( (2/6)**2 * (1/6)**2 ) / (0.25**4)) ** 0.25)
  assert closed == pytest.approx(0.9428, abs=1e-4)
  assert got.efficiency == pytest.approx(closed, rel=1e-6)


def test_efficiency_is_reported_as_nan_without_the_matrices():
  # the allocation is available without them; the honesty about what it costs is not
  got = apportion([0.5, 0.3, 0.2], 10)
  assert np.isnan(got.efficiency)
  assert int(got.counts.sum()) == 10


def test_a_budget_below_the_support_is_refused():
  """Silently dropping a support point would return a design for a different problem."""
  with pytest.raises(ValueError, match="cannot cover"):
    apportion([0.25, 0.25, 0.25, 0.25], 3)


@pytest.mark.parametrize("bad", [[0.0, 0.0], [-0.1, 1.1], [np.nan, 0.5]])
def test_impossible_weights_are_refused(bad):
  with pytest.raises(ValueError, match="weights"):
    apportion(bad, 4)


@pytest.mark.parametrize("runs", [0, -3, 2.5])
def test_a_budget_must_be_a_positive_integer(runs):
  with pytest.raises(ValueError, match="runs"):
    apportion([0.5, 0.5], runs)


def test_the_table_reads_out_largest_first():
  got = apportion([0.2, 0.5, 0.3], 10, )
  pairs = got.table
  assert [count for _, count in pairs] == sorted((count for _, count in pairs), reverse=True)
  assert sum(count for _, count in pairs) == 10
  assert {index for index, _ in pairs} == set(got.support.tolist())


def test_it_beats_naive_rounding_where_they_differ():
  """The rule exists because proportional rounding can lose a support point entirely.

  With weights 0.7/0.2/0.1 and four runs, `round(4 w)` gives 3/1/0 -- the third candidate is
  dropped, the information matrix loses a direction, and for these matrices that is singular
  rather than merely worse. Efficient rounding keeps every support point.
  """
  weights = np.array([0.7, 0.2, 0.1])
  naive = np.round(weights * 4).astype(int)
  assert naive[2] == 0, "the premise: naive rounding drops the third candidate"
  got = apportion(weights, 4)
  assert np.all(got.counts >= 1) and int(got.counts.sum()) == 4


def test_efficiency_rises_toward_one_with_budget():
  """The number a collaborator is actually asking for: what does a small budget cost?"""
  measure = optimal_measure(CUBIC, tolerance=1e-10, iterations=200_000)
  awkward = [apportion(measure.weights, runs, CUBIC).efficiency for runs in (5, 6, 7)]
  assert all(0.7 < value < 1.0 for value in awkward), awkward
  # and a large budget is indistinguishable from the measure
  assert apportion(measure.weights, 401, CUBIC).efficiency > 0.999
