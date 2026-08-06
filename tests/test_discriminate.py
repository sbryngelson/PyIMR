"""Model discrimination by marginal likelihood: the integral form of T-optimality."""

import numpy as np
import pytest

from scipy.special import logsumexp

from pyimr.discriminate import DiscriminationEvaluation, expected_log_bayes_factor, laplace_log_evidence, log_evidence

GRID = np.linspace(0.0, 1.0, 24)


def bank(seed, *, draws=160, offset=0.0, curvature=1.0):
  """Trajectories from a two-parameter family; `offset` moves the family, not the noise."""
  rng = np.random.default_rng(seed)
  slope = rng.standard_normal((draws, 1))
  bend = rng.standard_normal((draws, 1))
  return slope * GRID + curvature * bend * GRID**2 + offset


def test_log_evidence_of_a_constant_bank_is_the_gaussian_density():
  # Two identical draws: the average likelihood is that one likelihood, so the evidence is
  # the exact Gaussian log density and the normalisation must be right.
  curve = np.sin(GRID)
  seen = curve + 0.1
  value, _ = log_evidence(seen[None, :], np.vstack([curve, curve]), 0.5)
  exact = -0.5 * np.sum((seen - curve) ** 2) / 0.25 - 0.5 * GRID.size * np.log(2 * np.pi * 0.25)
  assert float(value[0]) == pytest.approx(exact, rel=1e-12)


def test_log_evidence_effective_draws_collapse_when_one_draw_dominates():
  near = np.zeros_like(GRID)
  far = np.vstack([near + 40.0 * (index + 1) for index in range(9)])
  _, effective = log_evidence(near[None, :], np.vstack([near, far]), 0.3)
  assert float(effective[0]) == pytest.approx(1.0, abs=0.05)


def test_log_evidence_effective_draws_are_full_when_every_draw_fits_equally():
  curve = np.zeros_like(GRID)
  _, effective = log_evidence(curve[None, :], np.vstack([curve] * 8), 0.5)
  assert float(effective[0]) == pytest.approx(8.0, rel=1e-9)


def test_log_evidence_exclusion_removes_exactly_one_draw():
  table = bank(0, draws=6)
  seen = table[2:3]
  with_all, _ = log_evidence(seen, table, 0.4)
  without, _ = log_evidence(seen, table, 0.4, exclude=np.array([2]))
  # dropping the generating row can only lower the average likelihood
  assert float(without[0]) < float(with_all[0])
  manual, _ = log_evidence(seen, np.delete(table, 2, axis=0), 0.4)
  assert float(without[0]) == pytest.approx(float(manual[0]), rel=1e-12)


@pytest.mark.parametrize(("first", "second", "seed"), [(1, 2, 0), (3, 4, 1), (5, 6, 2), (7, 8, 3)])
def test_a_model_cannot_discriminate_against_itself(first, second, seed):
  # Two banks drawn independently from the SAME family, so the true score is zero and any
  # systematic excess is bias. The noise scale matters for whether this test can fail:
  # the bias comes from the generating trajectory sitting in its own bank, and at a loose
  # scale enough other draws compete that one extra member barely moves the average
  # (measured bias 0.09 at sigma=0.6, and this test passed with the correction removed).
  # At sigma=0.2 the correct value stays under 0.35 across seeds while the uncorrected
  # estimator reads 1.19, so the threshold below separates them.
  result = expected_log_bayes_factor(bank(first), bank(second), 0.2, seed=seed)
  assert isinstance(result, DiscriminationEvaluation)
  assert result.expected_log_bayes_factor < 0.7


def test_separated_families_are_discriminated():
  apart = expected_log_bayes_factor(bank(1), bank(2, offset=3.0), 0.35, seed=0)
  together = expected_log_bayes_factor(bank(1), bank(2), 0.35, seed=0)
  assert apart.expected_log_bayes_factor > 10.0
  assert apart.expected_log_bayes_factor > together.expected_log_bayes_factor


@pytest.mark.parametrize("offset", [0.5, 1.0, 2.0])
def test_discrimination_grows_as_the_families_separate(offset):
  closer = expected_log_bayes_factor(bank(1), bank(2, offset=offset), 0.35, seed=0)
  further = expected_log_bayes_factor(bank(1), bank(2, offset=offset + 1.0), 0.35, seed=0)
  assert further.expected_log_bayes_factor > closer.expected_log_bayes_factor


def test_louder_noise_discriminates_less():
  quiet = expected_log_bayes_factor(bank(1), bank(2, offset=2.0), 0.25, seed=0)
  loud = expected_log_bayes_factor(bank(1), bank(2, offset=2.0), 1.5, seed=0)
  assert loud.expected_log_bayes_factor < quiet.expected_log_bayes_factor


def test_reliability_flag_follows_the_effective_sample_size():
  healthy = expected_log_bayes_factor(bank(1), bank(2, offset=1.0), 0.6, seed=0)
  assert healthy.reliable
  # a sharp likelihood against a diffuse prior collapses the integral onto one draw
  sharp = expected_log_bayes_factor(bank(1, draws=40), bank(2, draws=40), 0.002, seed=0)
  assert not sharp.reliable
  assert min(sharp.effective_draws_true, sharp.effective_draws_rival) < 2.0


def test_outer_limits_how_many_observations_are_used():
  result = expected_log_bayes_factor(bank(1), bank(2), 0.35, seed=0, outer=25)
  assert result.draws == 25


def test_results_are_reproducible_under_a_seed():
  first = expected_log_bayes_factor(bank(1), bank(2, offset=1.0), 0.4, seed=5)
  second = expected_log_bayes_factor(bank(1), bank(2, offset=1.0), 0.4, seed=5)
  assert first.expected_log_bayes_factor == pytest.approx(second.expected_log_bayes_factor)


@pytest.mark.parametrize(("deviation", "message"), [(0.0, "positive"), (-1.0, "positive"), (np.inf, "finite")])
def test_deviation_is_validated(deviation, message):
  with pytest.raises(ValueError, match=message):
    expected_log_bayes_factor(bank(1), bank(2), deviation)


def test_banks_must_agree_on_sample_count():
  with pytest.raises(ValueError, match="disagree on samples"):
    expected_log_bayes_factor(bank(1), bank(2)[:, :10], 0.4)


def test_a_single_draw_is_not_an_integral():
  with pytest.raises(ValueError, match="at least two draws"):
    expected_log_bayes_factor(bank(1)[:1], bank(2), 0.4)


def test_non_finite_trajectories_are_refused():
  broken = bank(1)
  broken[3, 4] = np.nan
  with pytest.raises(ValueError, match="non-finite"):
    expected_log_bayes_factor(broken, bank(2), 0.4)


def test_outer_must_be_a_positive_integer():
  with pytest.raises(ValueError, match="positive integer"):
    expected_log_bayes_factor(bank(1), bank(2), 0.4, outer=0)


def test_laplace_evidence_is_exact_for_a_linear_model():
  # A linear model with Gaussian noise makes the Laplace expansion exact, so the closed
  # form can be checked against a direct numerical integral rather than against itself.
  rng = np.random.default_rng(0)
  design = rng.standard_normal((40, 2))
  data = design @ np.array([0.4, -0.9]) + 0.1 * rng.standard_normal(40)
  fit, *_ = np.linalg.lstsq(design, data, rcond=None)
  residual = design @ fit - data
  value = laplace_log_evidence(residual, design, 1.0)

  # brute force: log int exp(-||X b - y||^2/2) db over a grid wide enough to contain it
  axis = np.linspace(-6.0, 6.0, 900)
  step = axis[1] - axis[0]
  first, second = np.meshgrid(axis, axis, indexing="ij")
  offsets = np.stack([first.ravel() - fit[0], second.ravel() - fit[1]], axis=1)
  quadratic = np.sum((offsets @ (design.T @ design)) * offsets, axis=1)
  brute = (logsumexp(-0.5 * quadratic) + 2 * np.log(step)
           - 0.5 * residual @ residual - 0.5 * 40 * np.log(2 * np.pi))
  assert value == pytest.approx(brute, abs=1e-6)


def test_laplace_evidence_penalises_a_flatter_likelihood():
  # Same fit quality, weaker curvature: the Occam factor must charge the broader posterior.
  rng = np.random.default_rng(1)
  design = rng.standard_normal((30, 2))
  residual = 0.1 * rng.standard_normal(30)
  sharp = laplace_log_evidence(residual, design, 1.0)
  flat = laplace_log_evidence(residual, 0.1 * design, 1.0)
  assert flat > sharp, "a flatter likelihood occupies more prior volume, so Z is larger"


def test_laplace_evidence_falls_as_the_fit_worsens():
  rng = np.random.default_rng(2)
  design = rng.standard_normal((30, 3))
  good = laplace_log_evidence(0.1 * rng.standard_normal(30), design, 1.0)
  bad = laplace_log_evidence(3.0 * rng.standard_normal(30), design, 1.0)
  assert bad < good


def test_laplace_evidence_refuses_an_undetermined_parameter():
  design = np.zeros((10, 2))
  design[:, 0] = 1.0                                    # second column carries no information
  with pytest.raises(ValueError, match="singular"):
    laplace_log_evidence(np.zeros(10), design, 1.0)


def test_laplace_evidence_checks_its_shapes():
  with pytest.raises(ValueError, match="rows"):
    laplace_log_evidence(np.zeros(10), np.ones((7, 2)), 1.0)
