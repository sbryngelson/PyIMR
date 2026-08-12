"""Design measures, and the equivalence-theorem certificate.

The value of this formulation is that the answer can be proved optimal rather than merely
reported, so the tests check it against cases with a known closed form: for polynomial
regression on [-1, 1] the D-optimal measure is equal mass at the roots of
(1 - x^2) P'_d(x), which is +-1 for the line and {-1, 0, 1} for the parabola.
"""

import numpy as np
import pytest

from pyimr.measure import (MeasureResult, apportion, budgeted_measure, constrained_measure,
                           optimal_measure, sensitivity)


def polynomial(degree, points=51):
  grid = np.linspace(-1.0, 1.0, points)
  basis = np.stack([grid**power for power in range(degree + 1)], axis=1)
  return grid, np.einsum("ij,ik->ijk", basis, basis)


def test_line_puts_half_its_mass_at_each_end():
  grid, matrices = polynomial(1)
  result = optimal_measure(matrices)
  assert isinstance(result, MeasureResult)
  assert sorted(np.round(grid[result.support], 6)) == [-1.0, 1.0]
  assert np.allclose(result.weights[result.support], 0.5, atol=1e-3)
  assert result.certified


def test_parabola_puts_a_third_at_each_of_minus_one_zero_one():
  grid, matrices = polynomial(2)
  result = optimal_measure(matrices)
  assert sorted(np.round(grid[result.support], 6)) == [-1.0, 0.0, 1.0]
  assert np.allclose(result.weights[result.support], 1.0 / 3.0, atol=1e-3)
  assert result.certified


def test_the_certificate_is_the_equivalence_theorem_itself():
  # Kiefer-Wolfowitz: at the D-optimum, max_x tr[M^-1 M(x)] equals p exactly, and the
  # maximum is attained on the support. Both halves are checked.
  grid, matrices = polynomial(2)
  result = optimal_measure(matrices)
  information = np.tensordot(result.weights, matrices, axes=(0, 0))
  variance = np.einsum("jk,ikj->i", np.linalg.inv(information), matrices)
  assert variance.max() == pytest.approx(3.0, abs=1e-5), "the bound must be p, the parameter count"
  assert np.allclose(variance[result.support], 3.0, atol=1e-5), "attained on the support"


def test_a_suboptimal_measure_has_a_strictly_positive_gap():
  # The certificate has to discriminate, or it certifies nothing.
  _, matrices = polynomial(2)
  count = matrices.shape[0]
  lopsided = np.zeros(count)
  lopsided[[10, 20, 30]] = 1.0 / 3.0
  assert float(np.max(sensitivity(matrices, lopsided))) > 0.1


def test_a_stopped_search_reports_itself_uncertified():
  # The gap the result CARRIES must be the real one. Testing `sensitivity` directly leaves
  # `MeasureResult.gap` unchecked, and hardwiring it to zero passed the rest of this file.
  _, matrices = polynomial(2)
  stopped = optimal_measure(matrices, iterations=3)
  assert stopped.iterations == 3
  assert stopped.gap > 1e-3, "three steps from a uniform start cannot be optimal"
  assert not stopped.certified
  finished = optimal_measure(matrices)
  assert finished.certified and finished.value > stopped.value


def test_convexity_means_a_finer_grid_cannot_find_a_better_answer():
  # A concave problem on a simplex has one maximum, so refining the candidate set may move
  # the support but cannot raise the criterion beyond the continuous optimum.
  coarse = optimal_measure(polynomial(2, points=51)[1])
  fine = optimal_measure(polynomial(2, points=101)[1])
  assert fine.value == pytest.approx(coarse.value, abs=1e-4)
  assert coarse.certified and fine.certified


def test_a_pure_utility_puts_everything_on_its_best_candidate():
  _, matrices = polynomial(2)
  utility = np.zeros(matrices.shape[0])
  utility[7] = 5.0
  result = optimal_measure(matrices, utility=utility, blend=1.0)
  assert result.support.tolist() == [7]
  assert result.value == pytest.approx(5.0)


def test_blending_moves_the_answer_between_the_two_objectives():
  grid, matrices = polynomial(2)
  utility = np.exp(-((grid - 0.5) ** 2) / 0.02)          # rewards designs near x = 0.5
  pure = optimal_measure(matrices)
  mixed = optimal_measure(matrices, utility=utility, blend=0.5)
  assert mixed.certified
  assert float(mixed.weights @ utility) > float(pure.weights @ utility), "the blend must buy utility"
  determinant = lambda w: np.linalg.slogdet(np.tensordot(w, matrices, axes=(0, 0)))[1]
  assert determinant(mixed.weights) < determinant(pure.weights), "and must pay for it in information"


def test_zero_blend_reproduces_the_pure_criterion():
  _, matrices = polynomial(2)
  utility = np.linspace(0.0, 1.0, matrices.shape[0])
  assert optimal_measure(matrices, utility=utility, blend=0.0).value == pytest.approx(
    optimal_measure(matrices).value)


def test_candidates_that_cannot_determine_the_parameters_are_refused():
  # every candidate informs only the first parameter, so no measure over them is regular
  matrices = np.zeros((5, 2, 2))
  matrices[:, 0, 0] = 1.0
  with pytest.raises(ValueError, match="singular"):
    optimal_measure(matrices)


def test_shapes_and_symmetry_are_validated():
  with pytest.raises(ValueError, match=r"candidates, p, p"):
    optimal_measure(np.zeros((4, 3)))
  crooked = np.tile(np.eye(2), (3, 1, 1))
  crooked[0, 0, 1] = 1.0
  with pytest.raises(ValueError, match="symmetric"):
    optimal_measure(crooked)


def test_blend_requires_a_utility_and_a_valid_range():
  _, matrices = polynomial(1)
  with pytest.raises(ValueError, match="no utility"):
    optimal_measure(matrices, blend=0.5)
  with pytest.raises(ValueError, match=r"\[0, 1\]"):
    optimal_measure(matrices, utility=np.zeros(matrices.shape[0]), blend=1.5)


def test_sensitivity_validates_its_weights():
  _, matrices = polynomial(1)
  with pytest.raises(ValueError, match="sum to one"):
    sensitivity(matrices, np.full(matrices.shape[0], 1.0))
  with pytest.raises(ValueError, match="must have shape"):
    sensitivity(matrices, np.array([0.5, 0.5]))


def test_a_free_optimum_can_be_too_concentrated_to_test():
  # The premise of `constrained_measure`: a criterion is happiest concentrating, and no number
  # of replicates fixes it, because apportionment only allocates within the support it is given.
  _, matrices = polynomial(1)
  free = optimal_measure(matrices)
  assert free.support.size < 3, "a line's D-optimum takes two settings; three parameters need more"
  assert apportion(free.weights, 30, replicates=10).support.size == free.support.size


def test_the_floor_is_met_and_the_extra_settings_actually_get_runs():
  _, matrices = polynomial(1)
  held = constrained_measure(matrices, 5)
  assert held.settings >= 5
  assert held.measure.weights[held.measure.support].min() >= 0.5 / 5 - 1e-9
  # the point of the floor: at a realistic budget every guaranteed setting is visited
  assert np.count_nonzero(apportion(held.measure.weights, 20).counts) >= 5


def test_naming_the_settings_without_a_floor_would_not_have_bound():
  # The failure this was written against: an optimiser merely RESTRICTED to a larger candidate
  # set reproduces the free optimum by leaving the new settings at zero, and reports no cost.
  _, matrices = polynomial(1)
  free = optimal_measure(matrices)
  held = constrained_measure(matrices, 4)
  assert held.nats_lost > 1e-3, "a binding constraint costs something"
  assert held.measure.value < free.value
  assert held.measure.value == pytest.approx(
    float(np.linalg.slogdet(np.tensordot(held.measure.weights, matrices, axes=(0, 0)))[1])), \
    "the reported value must be the criterion at the returned weights, not at eta"


def test_the_price_of_a_testable_batch_rises_with_what_is_demanded():
  # At a FIXED floor the guaranteed sets are nested, so each demand is strictly harder than the
  # last. Under the DEFAULT floor they are not: it is half an equal share, so asking for more
  # settings also asks less of each, and on the operator design seven settings cost less than
  # five. Only the fixed-floor statement is a property of the method.
  _, matrices = polynomial(2)
  costs = [constrained_measure(matrices, k, floor=0.05).nats_lost for k in (3, 5, 9, 15)]
  assert costs == sorted(costs)
  assert all(c >= -1e-9 for c in costs), "the free optimum is optimal; nothing can beat it"


def test_it_stays_certified_under_the_floor():
  # The substitution maps onto a whole simplex, so the equivalence theorem still applies --
  # to the shifted problem. Losing that is the difference between an answer and a proof.
  _, matrices = polynomial(2)
  held = constrained_measure(matrices, 6)
  assert held.certified_under_floor
  assert held.measure.gap <= 1e-6 * max(1.0, abs(held.measure.value))


def test_asking_for_no_more_than_the_free_optimum_costs_nothing():
  _, matrices = polynomial(2)
  held = constrained_measure(matrices, 3)
  assert held.nats_lost == pytest.approx(0.0, abs=1e-6)
  assert sorted(held.measure.support) == sorted(optimal_measure(matrices).support)


def test_the_floor_is_reported_and_can_be_set():
  _, matrices = polynomial(1)
  held = constrained_measure(matrices, 4, floor=0.2)
  assert held.measure.weights[held.measure.support].min() >= 0.2 - 1e-9
  assert "0.2" in str(held)


def test_a_criterion_is_carried_through_the_substitution():
  # The shift is inside the criterion, so a custom one must see the FULL information matrix.
  _, matrices = polynomial(1)
  seen = []

  def a_optimality(information):
    inverse = np.linalg.inv(information)
    seen.append(float(np.trace(information)))
    return -float(np.trace(inverse)), inverse @ inverse

  held = constrained_measure(matrices, 4, criterion=a_optimality)
  assert held.settings >= 4
  information = np.tensordot(held.measure.weights, matrices, axes=(0, 0))
  assert seen[-1] == pytest.approx(float(np.trace(information)), rel=1e-6), \
    "the criterion saw eta's matrix rather than the design's"


def test_it_refuses_what_it_cannot_deliver():
  _, matrices = polynomial(1)
  with pytest.raises(ValueError, match="asked of"):
    constrained_measure(matrices, matrices.shape[0] + 1)
  with pytest.raises(ValueError, match="exceeds all the mass"):
    constrained_measure(matrices, 4, floor=0.25)
  # it passed `unit_interval` and returned two settings of the six asked for, certified
  with pytest.raises(ValueError, match="floor must be positive"):
    constrained_measure(matrices, 6, floor=0.0)
  with pytest.raises(ValueError, match="positive"):
    constrained_measure(matrices, 0)


def test_an_early_transient_does_not_exile_a_candidate_the_optimum_needs():
  # Found via `constrained_measure`: the periodic prune fired at step 200 on a still-moving
  # iterate, dropped a candidate to zero, and a multiplicative update can never resurrect one --
  # so the search ended 1e-5 short of the optimum with a gap of 8e-3 it had forbidden itself to
  # close. More iterations did not help; it was stuck, not slow. The guard is that a positive
  # sensitivity means this step wants MORE mass there, which is never a candidate to drop.
  grid = np.linspace(-1.0, 1.0, 51)
  basis = np.stack([grid**power for power in range(3)], axis=1)
  matrices = np.einsum("ij,ik->ijk", basis, basis)

  def logdet(information):
    return float(np.linalg.slogdet(information)[1]), np.linalg.inv(information)

  # the shape that exposed it: a criterion whose early steps are aggressive enough to push the
  # centre point 20 decades down before the measure has settled at all
  held = constrained_measure(matrices, 6, criterion=logdet)
  assert held.certified_under_floor, "the search stopped short at a gap it could have closed"
  assert held.measure.value > -1.91334, "the exiled candidate was worth 1e-5 of the criterion"


def test_the_summary_does_not_claim_a_certificate_it_does_not_have():
  # A result that reports its own gap honestly and then prints "certified" regardless is worse
  # than one that never mentions it.
  _, matrices = polynomial(2)
  held = constrained_measure(matrices, 5, iterations=3)
  assert not held.certified_under_floor
  assert "NOT certified" in str(held)


def _two_ends():
  """The line on {-1, +1}: `det M(xi) = 4 xi_1 xi_2`, so every quantity below is closed form."""
  basis = np.array([[1.0, -1.0], [1.0, 1.0]])
  return np.einsum("ij,ik->ijk", basis, basis)


def test_a_budget_buys_equal_shares_of_money_not_equal_numbers_of_runs():
  """The closed form, which is the whole content of the cost-normalised theorem.

  Maximising `log(4 xi_1 xi_2)` subject to `sum_i c_i xi_i = 1` gives `xi_i = 1/(2 c_i)`, so
  each candidate takes half the BUDGET whatever it costs, and the run fractions come out
  inversely proportional to price. The criterion at one unit of budget is `-log(c_1 c_2)`.
  """
  costs = np.array([1.0, 3.0])
  held = budgeted_measure(_two_ends(), costs)
  assert held.certified, f"gap {held.gap:.2e}"
  assert np.allclose(held.budget_share, [0.5, 0.5], atol=1e-6)
  assert np.allclose(held.weights, [0.75, 0.25], atol=1e-6)
  assert np.isclose(held.value, -np.log(3.0), atol=1e-6)
  # runs per unit budget: xi_1 + xi_2 = 1/2 + 1/6
  assert np.isclose(held.runs_per_unit_cost, 1.0 / 2.0 + 1.0 / 6.0, rtol=1e-6)


def test_pricing_changes_the_answer_rather_than_relabelling_it():
  # the equal-cost optimum splits evenly; if it also came back for unequal costs, the argument
  # would be run through and have done nothing
  free = optimal_measure(_two_ends())
  assert np.allclose(free.weights, [0.5, 0.5], atol=1e-6)
  priced = budgeted_measure(_two_ends(), [1.0, 3.0])
  assert abs(priced.weights[0] - 0.5) > 0.2


def test_equal_costs_reproduce_the_unpriced_measure():
  _, matrices = polynomial(2)
  free = optimal_measure(matrices)
  priced = budgeted_measure(matrices, np.full(matrices.shape[0], 2.0))
  assert np.allclose(priced.weights, free.weights, atol=1e-6)
  # a uniform price is a change of budget units, so it shifts log det by -p log c and no more
  assert np.isclose(priced.value, free.value - 3.0 * np.log(2.0), atol=1e-6)


def _four_points():
  """The line on four points, priced so the cheap inner pair beats the informative outer one.

  Every closed form above is SATURATED -- support size equal to the parameter count -- and a
  saturated D-optimal design puts equal weight on its points whatever they are, so pricing can
  only rescale the answer and never move it. Those cases cannot tell the cost-normalised
  optimum from dividing the free one by price, and both tests below passed a mutation that did
  exactly that. Here the free optimum takes the ends and the priced one takes the middle pair,
  so the two answers have different SUPPORT and no rescaling connects them.
  """
  points = np.array([-1.0, -0.9, 0.9, 1.0])
  basis = np.stack([np.ones_like(points), points], axis=1)
  return np.einsum("ij,ik->ijk", basis, basis), np.array([4.0, 1.0, 1.0, 4.0])


def test_price_moves_the_support_and_not_merely_the_weights():
  matrices, costs = _four_points()
  free = optimal_measure(matrices)
  held = budgeted_measure(matrices, costs)
  assert held.certified, f"gap {held.gap:.2e}"
  assert list(free.support) == [0, 3], "the free optimum should want the ends"
  assert list(held.support) == [1, 2], "paying four times for the ends should move it inward"
  # dividing the free weights by price -- the thing this is not -- keeps the ends
  naive = free.weights / costs
  assert np.argmax(naive) in (0, 3), "the naive rescaling must differ, or nothing is being tested"
  assert np.isclose(held.value, np.log(0.81), atol=1e-6)


def test_the_certificate_is_the_cost_normalised_one():
  """`tr[M^-1 M(x)] / c_x <= p` everywhere, with equality on the support.

  Checked against the matrices themselves rather than against the scaled ones the search ran
  on, since the scaling is the step the caller has to trust.
  """
  matrices, costs = _four_points()
  held = budgeted_measure(matrices, costs)
  assert held.certified, f"gap {held.gap:.2e}"
  averaged = np.tensordot(held.weights / (held.weights @ costs), matrices, axes=(0, 0))
  ratio = np.einsum("jk,ikj->i", np.linalg.inv(averaged), matrices) / costs
  assert np.max(ratio) <= 2.0 + 1e-6, f"a candidate beats the bound: {np.max(ratio):.6f}"
  assert np.allclose(ratio[held.support], 2.0, atol=1e-5), "the support must attain it"
  assert np.max(ratio[[0, 3]]) < 1.0, "the priced-out candidates must be strictly under it"


def test_costs_are_validated():
  matrices = _two_ends()
  for bad in ([0.0, 1.0], [-1.0, 1.0], [np.inf, 1.0]):
    with pytest.raises(ValueError, match="costs"):
      budgeted_measure(matrices, bad)
  with pytest.raises(ValueError, match="costs"):
    budgeted_measure(matrices, [1.0, 1.0, 1.0])
