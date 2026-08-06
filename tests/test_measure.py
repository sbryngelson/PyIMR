"""Design measures, and the equivalence-theorem certificate.

The value of this formulation is that the answer can be proved optimal rather than merely
reported, so the tests check it against cases with a known closed form: for polynomial
regression on [-1, 1] the D-optimal measure is equal mass at the roots of
(1 - x^2) P'_d(x), which is +-1 for the line and {-1, 0, 1} for the parabola.
"""

import numpy as np
import pytest

from pyimr.measure import MeasureResult, optimal_measure, sensitivity


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
  with pytest.raises(ValueError, match="entries"):
    sensitivity(matrices, np.array([0.5, 0.5]))
