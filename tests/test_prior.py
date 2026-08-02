"""Redundancy-based parameter prior, Occam model prior, normalized model posteriors.

The pieces are algebraic and tested as such, but the claim the construction actually makes
is physical: that a flexible model reduced to a simpler one earns less prior mass there.
The last test checks that on real stress histories from a genuine nesting rather than on
constructed arrays, because that is the property the prior exists to have.
"""

import numpy as np
import pytest

from pyimr.prior import (
  harmonic_bottleneck,
  model_posterior,
  model_prior,
  normalize_log_coordinates,
  parameter_prior,
  redundancy_factor,
  redundancy_weight,
  relative_mismatch,
  stress_scale,
)

SECTION = "15. Redundancy prior and model selection"


def test_the_stress_scale_is_not_moved_by_collapse_spikes(measured):
  """Why median and MAD rather than mean and standard deviation: the stress history is
  dominated by a few collapse spikes, and a non-robust spread would let them set the
  resolution floor, making every model look indistinguishable from every other.
  """
  rng = np.random.default_rng(0)
  quiet = 1.0 + 0.05 * rng.standard_normal(400)
  spiked = quiet.copy()
  spiked[:8] = 60.0

  robust = stress_scale(quiet), stress_scale(spiked)
  naive = quiet.std() / abs(np.mean(quiet)), spiked.std() / abs(np.mean(spiked))
  measured("scale under spikes", f"robust {robust[0]:.4f} -> {robust[1]:.4f}, naive {naive[0]:.4f} -> {naive[1]:.4f}")
  assert robust[1] == pytest.approx(robust[0], rel=0.25), "spikes must not move the robust scale much"
  assert naive[1] > 15.0 * naive[0], "the non-robust scale is the thing being avoided"


def test_an_identical_history_has_no_mismatch():
  history = np.array([1.0, -2.0, 3.0, 4.0])
  assert relative_mismatch(history, history) == pytest.approx(0.0)
  assert relative_mismatch(history, np.zeros_like(history)) == pytest.approx(1.0)


def test_weights_move_the_mismatch_toward_where_they_are_large():
  parent = np.array([1.0, 1.0, 1.0, 1.0])
  child = np.array([1.0, 1.0, 1.0, 0.0])  # differs only in the last sample
  down_weighted = np.array([1.0, 1.0, 1.0, 0.01])
  assert relative_mismatch(parent, child, down_weighted) < relative_mismatch(parent, child)


def test_the_redundancy_factor_turns_over_at_the_resolvable_scale(measured):
  scale = 0.2
  values = [redundancy_factor(f, scale) for f in (0.0, 0.02, 0.2, 2.0)]
  assert values[0] == pytest.approx(0.0)
  assert values[2] == pytest.approx(0.5), "at the resolvable scale the factor is one half"
  assert values[1] < 0.02 and values[3] > 0.98
  assert values == sorted(values), "more mismatch must never mean more redundancy"
  measured("redundancy factor", "  ".join(f"{v:.3f}" for v in values))


def test_a_higher_exponent_demands_a_clearer_distinction():
  """The reference implementation raises the exponent to the difference in parameter
  count, so a model buying more parameters must earn them more convincingly.
  """
  assert redundancy_factor(0.1, 0.2, exponent=4) < redundancy_factor(0.1, 0.2, exponent=2)
  assert redundancy_factor(0.4, 0.2, exponent=4) > redundancy_factor(0.4, 0.2, exponent=2)


def test_redundancy_is_set_by_the_most_similar_child(measured):
  """The minimum, not an average: a parent that can emulate any one simpler model is
  redundant there, however different it looks from the others.
  """
  parent = np.array([1.0, 2.0, 3.0, 4.0])
  near, far = parent * 1.001, np.array([-4.0, 9.0, 0.5, -2.0])
  scale = 0.05

  both = redundancy_weight(parent, [near, far], scale=scale)
  assert both == pytest.approx(redundancy_weight(parent, [near], scale=scale))
  assert both < redundancy_weight(parent, [far], scale=scale)
  assert redundancy_weight(parent, [], scale=scale) == 1.0, "nothing to be redundant with"
  measured("min over children", f"near {both:.4f} vs far {redundancy_weight(parent, [far], scale=scale):.4f}")


def test_the_bottleneck_is_dragged_down_by_a_single_pinned_coordinate():
  assert harmonic_bottleneck(np.array([0.5, 0.5, 0.5])) == pytest.approx(0.5)
  assert harmonic_bottleneck(np.array([1.0, 1.0, 1e-6])) < 3e-6, "one pinned coordinate must dominate"
  assert harmonic_bottleneck(np.array([0.9, 0.9])) > harmonic_bottleneck(np.array([0.9, 0.1]))


def test_log_coordinates_are_range_invariant():
  """Normalizing in log space is what makes the prior independent of where the bounds
  were drawn -- the geometric midpoint maps to 0.5 whatever the decade span.
  """
  assert normalize_log_coordinates(np.array([10.0]), 1.0, 100.0)[0] == pytest.approx(0.5)
  assert normalize_log_coordinates(np.array([1e-3]), 1e-6, 1.0)[0] == pytest.approx(0.5)
  np.testing.assert_allclose(normalize_log_coordinates(np.array([1.0, 100.0]), 1.0, 100.0), [0.0, 1.0])


def test_the_parameter_prior_is_a_normalized_grid_prior():
  prior = parameter_prior(np.array([1.0, 2.0, 3.0]), np.array([1.0, 0.5, 0.0]))
  assert prior.sum() == pytest.approx(1.0)
  assert prior[2] == 0.0, "a fully redundant grid point earns no prior mass"


def test_the_model_prior_penalizes_every_extra_parameter(measured):
  values = [model_prior(k, 100.0) for k in (1, 2, 3, 4)]
  assert values == sorted(values, reverse=True)
  assert values[1] / values[0] == pytest.approx(0.1), "each parameter costs sqrt(N_eff)"
  measured("Occam penalty", "  ".join(f"k={k}: {v:.2e}" for k, v in zip((1, 2, 3, 4), values)))


def test_posteriors_are_normalized_and_survive_a_huge_dynamic_range(measured):
  posterior = model_posterior(np.array([-1000.0, -1002.0, -1010.0]))
  assert posterior.sum() == pytest.approx(1.0)
  # differences alone decide it, so the same gaps give the same answer at any offset
  np.testing.assert_allclose(posterior, model_posterior(np.array([0.0, -2.0, -10.0])))
  assert posterior[0] > posterior[1] > posterior[2]

  # underflow check: exp(-1000) is zero in double precision, so a naive implementation
  # would return nan here rather than a valid distribution
  assert np.all(np.isfinite(posterior))
  measured("model posterior", "  ".join(f"{p:.4f}" for p in posterior))


def test_the_model_prior_can_overturn_a_better_fit(measured):
  """The point of the Occam term. A model that fits 2 nats better but spends two more
  parameters must lose, and the arithmetic is checked against direct exponentiation.
  """
  evidences = np.array([-50.0, -48.0])
  priors = np.log([model_prior(2, 100.0), model_prior(4, 100.0)])
  posterior = model_posterior(evidences, priors)

  direct = np.exp(evidences + priors - np.max(evidences + priors))
  np.testing.assert_allclose(posterior, direct / direct.sum())
  assert posterior[0] > posterior[1], "two extra parameters must cost more than 2 nats here"
  measured("Occam overturns fit", f"simpler {posterior[0]:.4f} vs better-fitting {posterior[1]:.4f}")


@pytest.mark.slow
def test_a_flexible_model_reduced_to_a_simpler_one_earns_less_prior(measured):
  """The physical claim, on real stress histories rather than constructed arrays.

  `QuadraticKelvinVoigt` contains `NeoHookeanKelvinVoigt` at zero strain-stiffening, so
  its redundancy weight must vanish as the extra parameter goes to zero and grow as the
  parameter starts doing real work. A prior that failed this would still look correct in
  every algebraic test above.
  """
  import pyimr
  from _validation_support import R0, REQ

  times = np.linspace(0.0, 20e-6, 200)

  def stress(material):
    return pyimr.simulate(times, pyimr.SimulationConfig(R0, REQ, material, rtol=1e-11, atol=1e-13)).stress_integral_pa

  child = stress(pyimr.NeoHookeanKelvinVoigt(2500.0, 0.1))
  scale = stress_scale(child)

  alphas = (1e-4, 1e-3, 1e-2)
  mismatches, weights = [], []
  for alpha in alphas:
    parent = stress(pyimr.QuadraticKelvinVoigt(2500.0, 0.1, alpha))
    mismatches.append(relative_mismatch(parent, child))
    weights.append(redundancy_weight(parent, [child], scale=scale))

  measured("redundancy vs stiffening", "  ".join(f"a={a:g}: F={f:.2e} w={w:.2e}" for a, f, w in zip(alphas, mismatches, weights)))
  assert weights == sorted(weights), "redundancy must rise as the extra parameter does more work"

  # the sharp signature: the mismatch is linear in the stiffening, so with F well below
  # tau the weight is F^2/tau^2 and must grow as its square. A prior that merely ordered
  # correctly -- which any monotone function would -- passes the ordering check above but
  # not this one.
  for coarse, fine in zip(mismatches, mismatches[1:]):
    assert fine / coarse == pytest.approx(10.0, rel=0.02), "mismatch must be linear in the stiffening"
  for coarse, fine in zip(weights, weights[1:]):
    assert fine / coarse == pytest.approx(100.0, rel=0.05), "the weight must grow as the mismatch squared"
  assert weights[0] < 1e-6, f"at vanishing stiffening the model is redundant, got {weights[0]:.2e}"
