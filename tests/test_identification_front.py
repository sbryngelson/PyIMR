"""Batches that trade parameter precision against telling models apart.

The two things a batch of experiments is for pull in different directions, and the classical
answer is a compound criterion: `log det M(xi)` for the parameters, a per-run utility for the
model question, blended. Both parts are concave in the measure, so the blend is too and the
equivalence-theorem certificate survives all along the front -- which is why this is worth
doing over a measure rather than searching single designs.

The gates below are the structural ones, because the numbers depend on the problem: the front
must actually trade, it must spend the budget exactly, it must stay certified, and the
degenerate end must be visible rather than silent.
"""

import numpy as np
import pytest

from pyimr.measure import apportion, identification_front, optimal_measure

# quadratic regression on [-1, 1], with a utility peaked away from the D-optimal support so
# the two purposes genuinely disagree and the front has something to trade
GRID = np.linspace(-1.0, 1.0, 41)
INFORMATION = np.array([np.outer(v, v) for v in
                        np.vstack([np.ones_like(GRID), GRID, GRID**2]).T])
UTILITY = 2.0 * np.exp(-((GRID - 0.3) / 0.25) ** 2)
RUNS = 12


@pytest.fixture(scope="module")
def front():
  return identification_front(INFORMATION, UTILITY, RUNS)


def test_every_batch_spends_the_budget(front):
  for batch in front:
    assert int(batch.counts.sum()) == RUNS
    assert np.all(batch.counts >= 0)
    assert sum(count for _, count in batch.table) == RUNS


def test_every_batch_is_certified(front):
  # the whole reason for working over measures: each point of the front carries a proof
  for batch in front:
    assert batch.certified, f"blend {batch.blend} certified only to gap {batch.gap:.2e}"


def test_the_front_actually_trades(front):
  """A front that does not trade is a scalarization that is not binding."""
  first, last = front[0], front[-1]
  assert last.discrimination > 3.0 * first.discrimination
  assert last.efficiency < 0.5 * first.efficiency


def test_discrimination_rises_and_precision_falls(front):
  """Monotone in the blend, up to the slack integer rounding introduces.

  For measures this is exact -- a standard property of weighted scalarization. Rounding to
  whole runs can perturb it slightly, so the tolerance is on the rounding, not on the claim.
  """
  for coarse, fine in zip(front[:-1], front[1:], strict=True):
    assert fine.discrimination >= coarse.discrimination - 1e-9
    assert fine.efficiency <= coarse.efficiency + 0.05


def test_pure_estimation_is_the_d_optimal_batch(front):
  """`blend = 0` must reproduce `apportion` of the D-optimal measure, efficiency 1 by
  construction -- which is what makes the other rows readable as a cost.
  """
  assert front[0].blend == 0.0
  assert front[0].efficiency == pytest.approx(1.0, rel=1e-9)
  expected = apportion(optimal_measure(INFORMATION).weights, RUNS, INFORMATION).counts
  assert np.array_equal(front[0].counts, expected)


def test_pure_discrimination_collapses_to_one_setting(front):
  """The degenerate end, reported rather than hidden.

  At `blend = 1` the criterion is linear in the measure, so its maximum sits on a single
  candidate and the batch is that setting repeated. The information matrix is then singular,
  no parameter is estimable, and -- the part no criterion here can see -- a batch on one
  setting cannot detect that every model in the set is wrong.
  """
  extreme = front[-1]
  assert extreme.blend == 1.0
  assert extreme.settings == 1
  assert extreme.log_det == float("-inf")
  assert extreme.efficiency == 0.0
  # and it is the best single candidate, which is the only thing a linear criterion can want
  assert extreme.counts[int(np.argmax(UTILITY))] == RUNS


def test_the_discrimination_is_the_batch_total_not_a_rate():
  """Log evidence is additive over independent experiments, so twice the runs is twice the
  nats at the same design. Reporting a rate would make the collaborator do that arithmetic.
  """
  small = identification_front(INFORMATION, UTILITY, 6, blends=(1.0,))[0]
  large = identification_front(INFORMATION, UTILITY, 12, blends=(1.0,))[0]
  assert large.discrimination == pytest.approx(2.0 * small.discrimination, rel=1e-9)


def test_a_utility_that_does_not_discriminate_leaves_the_design_alone():
  """A flat utility adds a constant to the criterion, so the optimum must not move."""
  flat = identification_front(INFORMATION, np.full(GRID.size, 1.5), RUNS, blends=(0.0, 0.5))
  assert np.array_equal(flat[0].counts, flat[1].counts)
  assert flat[1].efficiency == pytest.approx(1.0, rel=1e-9)


@pytest.mark.parametrize(("bad", "message"), [
  ({"utility": np.ones(5)}, "one value per candidate"),
  ({"utility": np.full(GRID.size, np.nan)}, "finite"),
  ({"runs": 0}, "positive integer"),
  ({"blends": (0.5, 1.5)}, r"\[0, 1\]"),
])
def test_impossible_arguments_are_refused(bad, message):
  call = {"matrices": INFORMATION, "utility": UTILITY, "runs": RUNS}
  call.update(bad)
  with pytest.raises(ValueError, match=message):
    identification_front(call.pop("matrices"), call.pop("utility"), call.pop("runs"), **call)


def test_settings_counts_distinct_designs(front):
  for batch in front:
    assert batch.settings == int(np.count_nonzero(batch.counts))
    assert batch.settings == len(batch.table)
