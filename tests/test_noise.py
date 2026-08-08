"""Strain-rate weighting and the marginalized noise scale.

The marginalization is the part worth guarding. It is a quadrature standing in for an
integral with no closed form, and a quadrature that is quietly wrong still returns a
plausible number -- so it is checked against independent adaptive integration rather than
against itself, and the grid is shown to converge.
"""

import numpy as np
import pytest
from scipy.integrate import quad

from pyimr.noise import (
  beta_quadrature,
  characteristic_time,
  hencky_strain_rate,
  elliptical_gate,
  marginal_log_likelihood,
  marginalize_evaluation,
  strain_rate_weights,
  weighted_deviation,
)

SECTION = "14. Observation weighting and noise scale"


def _reference_marginal(chi_squared, count, low=0.05, high=10.0):
  """Adaptive integration of the same integral, independent of the grid under test."""
  peak = -0.5 * (count + count * np.log(chi_squared / count))  # log L at beta = sqrt(chi2/N)

  def integrand(beta):
    return np.exp(-0.5 * (chi_squared / beta**2 + count * np.log(beta**2)) - peak) * (2.0 / np.pi) / (1.0 + beta**2)

  value = quad(integrand, low, high, limit=200)[0]
  mass = quad(lambda b: (2.0 / np.pi) / (1.0 + b**2), low, high, limit=200)[0]
  return peak + np.log(value) - np.log(mass)


def test_weights_fall_with_strain_rate_and_stop_at_the_floor(measured):
  rates = np.array([0.0, 0.5, 1.0, 2.0, 10.0, 1e6, 1e300])
  weights = strain_rate_weights(rates, 1.0)

  # strictly decreasing until it saturates, non-increasing after -- the tail sits exactly
  # on the floor, so demanding strict decrease everywhere would fail on correct behaviour
  assert np.all(np.diff(weights) <= 0.0), "the weight must never rise with strain rate"
  assert np.all(np.diff(weights[:4]) < 0.0), "it must be strictly decreasing before saturation"
  assert weights.min() >= 0.1 and np.isfinite(weights).all()
  # at the threshold the logistic argument is zero, so the weight is the midpoint
  assert weights[2] == pytest.approx(0.1 + 0.9 * 0.5)
  # 1e300 saturates: the naive 1/(1+exp(-x)) overflows here instead of reaching the floor
  assert weights[-1] == pytest.approx(0.1)
  measured("logistic weight", f"rate 0 -> {weights[0]:.3f}, threshold -> {weights[2]:.3f}, saturated -> {weights[-1]:.3f}")


def test_a_down_weighted_sample_gets_a_wider_error_bar(measured):
  """Eqn 12: `sigma^2 = sigma0^2 / w`, so the floor of 0.1 inflates sigma by sqrt(10)."""
  deviations = weighted_deviation(np.full(3, 2.0), np.array([1.0, 0.5, 0.1]))
  np.testing.assert_allclose(deviations, 2.0 / np.sqrt([1.0, 0.5, 0.1]))
  measured("variance inflation", f"w=0.1 widens sigma by {deviations[2] / 2.0:.3f}x")


def test_the_gate_drops_only_near_equilibrium_samples():
  strain = np.array([0.0, 0.5, 2.0, 0.0])
  rate = np.array([0.0, 0.5, 0.0, 3.0])
  kept = elliptical_gate(strain, rate, 1.0, 1.0)
  np.testing.assert_array_equal(kept, [False, False, True, True])


def test_the_prior_grid_is_normalized_and_holds_the_mass_it_actually_holds(measured):
  """The paper calls [0.05, 10] "more than 99.9%" of the half-Cauchy mass. For the
  scale-1 half-Cauchy of eqn 16 it is 90.5%. Pinned so the discrepancy stays visible:
  renormalization makes this a prior conditioned on the range, which is proper either
  way, but it means `beta > 10` is excluded rather than negligible.
  """
  nodes, weights = beta_quadrature()
  assert weights.sum() == pytest.approx(1.0)
  assert np.all(weights > 0.0) and nodes[0] == pytest.approx(0.05) and nodes[-1] == pytest.approx(10.0)

  mass = (2.0 / np.pi) * (np.arctan(10.0) - np.arctan(0.05))
  measured("truncated prior mass", f"{mass:.4f} on [0.05, 10] (paper states >0.999)")
  assert mass == pytest.approx(0.9047, abs=1e-4)


@pytest.mark.parametrize(("chi_squared", "count"), [(100.0, 100), (400.0, 100), (25.0, 100), (5000.0, 500)])
def test_the_grid_matches_adaptive_integration(chi_squared, count, measured):
  nodes, weights = beta_quadrature(count=800)
  value = marginal_log_likelihood(chi_squared, count, nodes=nodes, weights=weights)
  reference = _reference_marginal(chi_squared, count)
  measured(f"marginal chi2={chi_squared:g} N={count}", f"grid {value:.6f} vs quad {reference:.6f}")
  assert value == pytest.approx(reference, abs=1e-4)


def test_the_grid_converges(measured):
  values = [marginal_log_likelihood(25.0, 100, **dict(zip(("nodes", "weights"), beta_quadrature(count=n)))) for n in (200, 800, 3200)]
  measured("grid convergence", "  ".join(f"{v:.6f}" for v in values))
  assert abs(values[1] - values[2]) < abs(values[0] - values[2]), "refining must not move it further away"
  assert abs(values[1] - values[2]) < 1e-5


def test_a_model_that_needs_a_large_noise_scale_is_penalized(measured):
  """The reason for marginalizing rather than fitting `beta`. A model that only matches
  the data by inflating its error bars must score worse, and must lose more than the
  improvement in raw fit it bought -- otherwise inflation would be free.
  """
  count = 100
  values = [marginal_log_likelihood(chi, count) for chi in (100.0, 400.0, 2500.0)]
  assert values[0] > values[1] > values[2], "a larger required beta must score worse"

  # profiling at the best beta would report -0.5*N*(1 + log(chi2/N)); marginalizing is
  # strictly worse, and that gap is the Occam factor the paper relies on
  for chi, value in zip((100.0, 400.0, 2500.0), values):
    profiled = -0.5 * (count + count * np.log(chi / count))
    assert value < profiled, "marginalizing must cost something relative to the best-fit beta"
  measured("noise-scale penalty", f"chi2/N = 1, 4, 25 -> {values[0]:.1f}, {values[1]:.1f}, {values[2]:.1f}")


def test_the_normalization_is_a_pure_offset():
  base = marginal_log_likelihood(100.0, 100)
  assert marginal_log_likelihood(100.0, 100, normalization=7.0) == pytest.approx(base - 3.5)


@pytest.mark.parametrize(
  ("call", "message"),
  [
    (lambda: strain_rate_weights(np.zeros(3), 0.0), "threshold must be finite and positive"),
    (lambda: strain_rate_weights(np.zeros(3), 1.0, floor=0.0), "floor must lie in"),
    (lambda: strain_rate_weights(np.zeros(3), 1.0, steepness=0.0), "steepness must be finite and positive"),
    (lambda: strain_rate_weights(np.array([np.nan]), 1.0), "strain_rate must be finite"),
    (lambda: weighted_deviation(np.ones(2), np.array([1.0, 0.0])), "weights must be positive"),
    (lambda: elliptical_gate(np.zeros(2), np.zeros(2), 0.0, 1.0), "strain_threshold must be finite"),
    (lambda: beta_quadrature(minimum=0.0), "require 0 < minimum < maximum"),
    (lambda: beta_quadrature(count=1), "count must be at least 2"),
    (lambda: marginal_log_likelihood(-1.0, 10), "chi_squared must be non-negative"),
    (lambda: marginal_log_likelihood(1.0, 0), "count must be positive"),
    (lambda: marginal_log_likelihood(1.0, 10, nodes=np.ones(3)), "pass both nodes and weights"),
  ],
)
def test_it_refuses_malformed_input(call, message):
  with pytest.raises(ValueError, match=message):
    call()


def test_it_marginalizes_a_real_likelihood_evaluation(measured):
  """The bridge to `pyimr.inference`, on a real prepared inference rather than a stub.

  What is asserted is that the helper reads the right quantity off the evaluation and
  does not silently collapse to the `beta = 1` likelihood. The quadrature itself is
  checked against adaptive integration above; this test does not re-check it.
  """
  import pyimr
  from pyimr.inference import InferenceParameter, RadiusObservation, prepare_inference
  from _validation_support import NHKV, R0, REQ

  times = np.linspace(2e-6, 20e-6, 24)
  truth = pyimr.simulate(times, pyimr.SimulationConfig(R0, REQ, NHKV))
  radii = truth.radius_ratio * R0 + np.random.default_rng(0).normal(0.0, 2e-7, times.size)

  inference = prepare_inference(
    pyimr.SimulationConfig(R0, REQ, NHKV),
    RadiusObservation(times, radii, 2e-7),
    (InferenceParameter("material.shear_modulus_pa", 1500.0, 4000.0),),
  )
  evaluation = inference.evaluate(np.array([0.5]))
  residual = np.asarray(evaluation.residual)
  chi_squared = float(residual @ residual)

  value = marginalize_evaluation(evaluation)
  assert value == pytest.approx(marginal_log_likelihood(chi_squared, residual.size))
  # marginalizing must not simply reproduce the beta = 1 likelihood
  assert value != pytest.approx(-0.5 * chi_squared, rel=1e-6)
  measured("marginalized evaluation", f"chi2={chi_squared:.1f} over {residual.size} samples -> {value:.3f}")


def test_the_characteristic_time_and_strain_rate_are_consistent(measured):
  """Both feed `strain_rate_weights`, which is only meaningful in nondimensional rate."""
  scale = characteristic_time(277e-6)
  assert scale == pytest.approx(277e-6 * np.sqrt(998.0 / 101325.0))
  assert characteristic_time(554e-6) == pytest.approx(2.0 * scale), "linear in the radius"

  times = np.linspace(0.0, 1.0, 400)
  rate = hencky_strain_rate(np.exp(-0.5 * times), times, 1.0)
  measured("Hencky rate", f"t_c(277um)={scale * 1e6:.2f} us, d(lnR)/dt={rate[200]:.4f}")
  assert rate[200] == pytest.approx(-0.5, abs=1e-4), "d(ln R)/dt of exp(-t/2) is -1/2"
  np.testing.assert_allclose(hencky_strain_rate(np.exp(-0.5 * times), times, 3.0), 3.0 * rate)


@pytest.mark.parametrize(
  ("call", "message"),
  [
    (lambda: characteristic_time(0.0), "maximum_radius must be finite and positive"),
    (lambda: characteristic_time(1e-4, density=0.0), "must be positive"),
    (lambda: hencky_strain_rate(np.ones(3), np.ones(4), 1.0), "same shape"),
    (lambda: hencky_strain_rate(np.array([1.0, 0.0]), np.array([0.0, 1.0]), 1.0), "must be positive"),
  ],
)
def test_the_nondimensionalization_helpers_refuse_bad_input(call, message):
  with pytest.raises(ValueError, match=message):
    call()


def test_predicted_spread_separates_a_conditioned_model_from_a_chaotic_one(measured):
  """The scatter is data, not just a noise scale. A quadratic Zener at strong stiffening
  amplifies preparation scatter into an absurd spread, and the measured records -- which
  repeat to about 2% -- exclude it whatever its chi-squared (#203).
  """
  import pyimr
  from pyimr.noise import predicted_spread

  times = np.linspace(0.0, 1.4e-4, 201)

  def spread(alpha, guard: float | None = 50.0):
    material = pyimr.QuadraticZener(4640.0, 1e-4, 2.78e-7, 0.0, alpha)
    config = pyimr.SimulationConfig(
      277e-6, 277e-6 / 7.09, material, dynamics="keller-miksis", rtol=1e-4, atol=1e-6, max_steps=200_000, max_radius_ratio=guard
    )
    return predicted_spread(config, times)

  # The strong-stiffening case is now refused outright: it expands to R/R0 = 2132 after
  # collapse, which is the model failing rather than merely being sensitive. That is a
  # stronger exclusion than the spread argument, and it comes first.
  assert spread(3.594) is None, "a runaway must not reach the spread estimator at all"

  # With the guard off, the estimator still has to separate the two, which is what this
  # test is for. Disabling it here endorses nothing about the model.
  conditioned, chaotic = spread(0.0215), spread(3.594, guard=None)
  assert conditioned is not None and chaotic is not None, "both samples must integrate for the comparison"
  measured("predicted spread", f"alpha=0.02 -> {conditioned:.4f}, alpha=3.6 -> {chaotic:.2f}")
  # gelatin repeats to a spread near 0.02 of Rmax; the conditioned model predicts less
  assert conditioned < 0.02, "a well-conditioned model must not predict more scatter than was observed"
  assert chaotic > 1.0, "the sensitive region must predict an absurd spread, which is what excludes it"


@pytest.mark.parametrize(("options", "message"), [({"samples": 1}, "at least 2"), ({"relative": 0.0}, "must lie in")])
def test_predicted_spread_refuses_a_meaningless_request(options, message):
  import pyimr
  from pyimr.noise import predicted_spread

  config = pyimr.SimulationConfig(277e-6, 277e-6 / 7.09, pyimr.NeoHookeanKelvinVoigt(2500.0, 0.1))
  with pytest.raises(ValueError, match=message):
    predicted_spread(config, np.linspace(0.0, 2e-5, 10), **options)


def test_white_residuals_are_reported_as_independent():
  from pyimr.noise import check_residuals

  found = check_residuals(np.random.default_rng(0).normal(size=400))
  assert found.independent, found.summary
  assert abs(found.lag_one) < 0.15
  assert found.effective_samples > 300, "white noise must not lose most of its samples"
  assert found.inflation < 1.2


def test_correlated_residuals_are_caught_and_the_cost_quantified(measured):
  """The failure this exists for: a fit whose chi-squared says it is excellent while its
  residuals are smooth. Measured on a real record, `chi2/N = 0.974` came with lag-one
  autocorrelation `0.90` and roughly ten effective samples out of 201 (#216).
  """
  from pyimr.noise import check_residuals

  rng = np.random.default_rng(1)
  walk = np.cumsum(rng.normal(size=400))
  walk = (walk - walk.mean()) / walk.std()          # chi2/N is ~1 by construction
  found = check_residuals(walk)

  measured("correlated residuals", f"chi2/N {found.chi_squared_per_sample:.3f}, "
                                   f"lag-1 {found.lag_one:.2f}, {found.effective_samples:.0f} effective")
  assert not found.independent, found.summary
  assert found.chi_squared_per_sample == pytest.approx(1.0, abs=0.05), (
    "the point is that chi-squared looks fine while the residuals do not"
  )
  assert found.effective_samples < 40, "a random walk must lose most of its samples"
  assert found.inflation > 3.0
  assert "too small" in found.summary


def test_the_effective_count_never_exceeds_the_real_one():
  """Negative autocorrelation would otherwise inflate it past the number of samples."""
  from pyimr.noise import check_residuals

  alternating = np.resize([1.0, -1.0], 200) + 1e-9
  found = check_residuals(alternating)
  assert found.effective_samples <= found.samples * 1.0000001, found.summary


@pytest.mark.parametrize(("values", "message"), [
  (np.ones(4), "at least 8"),
  (np.full(20, 3.0), "constant"),
  (np.where(np.arange(20) == 3, np.nan, 1.0), "finite"),
])
def test_check_residuals_refuses_what_it_cannot_answer(values, message):
  from pyimr.noise import check_residuals

  with pytest.raises(ValueError, match=message):
    check_residuals(values)
