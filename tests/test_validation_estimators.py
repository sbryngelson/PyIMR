"""Trace-side estimators (`imr_data`) and the prepared inference layer."""

import numpy as np
import pytest

from pyimr import data
import pyimr
from _validation_support import NHKV, R0, REQ
from pyimr.inference import FieldObservation, InferenceParameter, RadiusObservation, prepare_inference

SECTION = "3b. Trace estimators and prepared inference"


def test_equilibrium_radius_round_trip(measured):
  """The estimator must invert the solver's own pressure/radius relation."""
  gas_pressure = (pyimr.P8 + 2 * pyimr.SURF / REQ) * (REQ / R0) ** (3 * pyimr.KAPPA)
  error = abs(data.equilibrium_radius(R0, gas_pressure) - REQ) / REQ
  measured("equilibrium radius round-trip", f"rel={error:.2e}")
  assert error < 1e-12


def test_natural_frequency_reduces_to_minnaert(measured):
  """The gas-only limit must reproduce Minnaert exactly."""
  computed, _ = data.natural_frequency(R0, REQ, 1e-12, 1e-12, surface_tension_n_m=0.0)
  minnaert = np.sqrt(3 * pyimr.KAPPA * pyimr.P8 / pyimr.RHO) / REQ
  error = abs(computed - minnaert) / minnaert
  measured("natural frequency -> Minnaert", f"rel={error:.2e}")
  assert error < 1e-12


@pytest.fixture(scope="module")
def rebound_trace():
  times = np.linspace(0, 300e-6, 8000)
  radius = pyimr.simulate(times, pyimr.SimulationConfig(R0=R0, Req=REQ, material=NHKV)).radius_m
  return data.collapse_features(times, radius)


def test_natural_frequency_matches_measured_rebound(rebound_trace, measured):
  collapse_times, _, _ = rebound_trace
  observed = float(np.median(2 * np.pi / np.diff(collapse_times)[1:5]))
  predicted, _ = data.natural_frequency(R0, REQ, 2500.0, 0.1)
  error = abs(predicted - observed) / observed
  measured("vs measured rebound", f"predicted={predicted:.3e} measured={observed:.3e} rel={error:.2e}")
  assert error < 0.10


def test_collapse_features_decay(rebound_trace, measured):
  """Feature extraction must find a monotonically decaying rebound sequence."""
  collapse_times, peaks, _ = rebound_trace
  measured("collapse features", f"{len(collapse_times)} collapses, {len(peaks)} peaks")
  assert len(collapse_times) >= 3 and len(peaks) >= 3
  assert np.all(np.diff(peaks[:3]) < 0.0)


def test_thermal_grid_convergence(measured):
  """Thermal grid refinement must converge monotonically."""
  convergence = data.resolution_convergence(
    pyimr.SimulationConfig(R0=R0, Req=REQ, material=NHKV, bubtherm=1, Nt=10, Mt=10), np.linspace(0, 60e-6, 200), [(10, 10), (20, 20), (40, 40)]
  )
  errors = [error for _, error in convergence]
  measured("thermal grid convergence", " ".join(f"{error:.1e}" for error in errors))
  assert errors[0] > errors[1] > errors[2] == 0.0


@pytest.fixture(scope="module")
def prepared_inference():
  config = pyimr.SimulationConfig(R0, REQ, NHKV)
  times = np.linspace(0.0, 20e-6, 50)
  truth = pyimr.simulate(times, config)
  return times, prepare_inference(
    config,
    RadiusObservation(times, truth.radius_m, 1e-8),
    (InferenceParameter("material.shear_modulus_pa", 2000.0, 3000.0), InferenceParameter("material.viscosity_pa_s", 0.05, 0.15)),
  )


def test_likelihood_and_jacobian(prepared_inference, measured):
  times, inference = prepared_inference
  centre = np.array([0.5, 0.5])
  evaluation = inference.evaluate(centre)
  jacobian = inference.jacobian(centre)
  measured("likelihood and Jacobian", f"max|residual|={np.max(np.abs(evaluation.residual)):.2e}")
  assert np.max(np.abs(evaluation.residual)) == 0.0
  assert jacobian.shape == (times.size, 2)
  assert np.all(np.isfinite(jacobian))


def test_multistart_is_deterministic(prepared_inference, measured):
  _, inference = prepared_inference
  multistart = inference.fit_multistart(2, seed=7, max_evaluations=20)
  measured("multistart", f"{len(multistart.endpoints)} endpoints, best cost={multistart.best.cost:.2e}")
  assert len(multistart.endpoints) == 2
  assert multistart.best is not None and multistart.best.cost < 1e-12


_VELOCITY_SIGMA = 2.0


@pytest.fixture(scope="module")
def multi_observable():
  """Radius on one grid, wall velocity on a coarser one -- deliberately not the"""
  times = np.linspace(0.0, 4e-5, 50)
  config = pyimr.SimulationConfig(R0, REQ, NHKV, rtol=1e-10, atol=1e-12)
  truth = pyimr.simulate(times, config)
  rng = np.random.default_rng(3)
  radius = RadiusObservation(times, np.asarray(truth.radius_m) + rng.normal(0.0, 5e-7, times.size), 5e-7)
  coarse = times[::3]
  velocity = FieldObservation(
    "wall_velocity_m_s", coarse, np.asarray(truth.wall_velocity_m_s)[::3] + rng.normal(0.0, _VELOCITY_SIGMA, coarse.size), _VELOCITY_SIGMA
  )
  parameters = (InferenceParameter("material.shear_modulus_pa", 1500.0, 4000.0), InferenceParameter("material.viscosity_pa_s", 0.05, 0.2))
  return prepare_inference(config, radius, parameters), prepare_inference(config, (radius, velocity), parameters)


def test_a_second_observable_stacks_onto_the_residual(multi_observable, measured):
  """One sensitivity solve already returns tangents for every observable; the"""
  radius_only, both = multi_observable
  assert radius_only.observation_size == 50
  assert both.observation_size == 67, "50 radius plus 17 velocity samples"
  unit = np.array([0.42, 0.37])
  assert np.asarray(radius_only.jacobian(unit)).shape == (50, 2)
  assert np.asarray(both.jacobian(unit)).shape == (67, 2)
  measured("stacked observables", f"{radius_only.observation_size} -> {both.observation_size} values")


def test_the_stacked_gradient_is_still_the_derivative(multi_observable, measured):
  """Stacking is only correct if the log-likelihood and its gradient still agree"""
  _, both = multi_observable
  unit = np.array([0.42, 0.37])
  evaluation, jacobian = both.evaluate_with_jacobian(unit)
  analytic = -np.asarray(evaluation.residual) @ jacobian

  step, difference = 1e-5, np.zeros(2)
  for index in range(2):
    offset = np.zeros(2)
    offset[index] = step
    ahead = both.evaluate_with_jacobian(unit + offset)[0].log_likelihood
    behind = both.evaluate_with_jacobian(unit - offset)[0].log_likelihood
    difference[index] = (ahead - behind) / (2.0 * step)

  error = float(np.max(np.abs(analytic - difference))) / max(float(np.max(np.abs(difference))), 1e-30)
  measured("stacked gradient", f"rel={error:.2e}")
  assert error < 1e-4


def test_an_unknown_field_is_refused():
  with pytest.raises(ValueError, match="field must be one of"):
    FieldObservation("bubble_temperature_k", np.array([0.0, 1e-5]), np.array([300.0, 310.0]), 1.0)


def test_gauss_newton_is_exact_where_eig_uses_it(measured):
  """`J^T J` is not an approximation to the Fisher information for a correctly"""
  times = np.linspace(0.0, 4e-5, 60)
  config = pyimr.SimulationConfig(R0, REQ, NHKV)
  truth = pyimr.simulate(times, config)
  observed = np.asarray(truth.radius_m) + np.random.default_rng(11).normal(0.0, 5e-7, times.size)
  parameters = (InferenceParameter("material.shear_modulus_pa", 1500.0, 4000.0), InferenceParameter("material.viscosity_pa_s", 0.05, 0.2))
  inference = prepare_inference(config, RadiusObservation(times, observed, 5e-7), parameters)
  ratio = inference.curvature_ratio(np.array([(2500.0 - 1500.0) / 2500.0, (0.1 - 0.05) / 0.15]))
  measured("Gauss-Newton dropped term at truth", f"||r.H||/||J^T J|| = {ratio:.2e}")
  assert ratio < 1e-2


def test_the_dropped_term_detects_misspecification(measured):
  """The diagnostic's actual use. Data from Zener, fitted with NHKV: the model"""
  times = np.linspace(0.0, 4e-5, 60)
  config = pyimr.SimulationConfig(R0, REQ, NHKV)
  other = pyimr.SimulationConfig(R0, REQ, pyimr.Zener(0.1, 2500.0, 2e-6, 2e-7))
  observed = np.asarray(pyimr.simulate(times, other).radius_m)
  observed = observed + np.random.default_rng(11).normal(0.0, 5e-7, times.size)
  parameters = (InferenceParameter("material.shear_modulus_pa", 1500.0, 4000.0), InferenceParameter("material.viscosity_pa_s", 0.05, 0.2))
  inference = prepare_inference(config, RadiusObservation(times, observed, 5e-7), parameters)
  ratio = inference.curvature_ratio(np.array([(2500.0 - 1500.0) / 2500.0, (0.1 - 0.05) / 0.15]))
  measured("Gauss-Newton dropped term, misspecified", f"||r.H||/||J^T J|| = {ratio:.2e}")
  assert ratio > 0.05


_TAU = 3e-6


@pytest.fixture(scope="module")
def correlated():
  times = np.linspace(2e-6, 4e-5, 40)
  config = pyimr.SimulationConfig(R0, REQ, NHKV, rtol=1e-10, atol=1e-12)
  truth = np.asarray(pyimr.simulate(times, config).radius_m)
  observed = truth + np.random.default_rng(5).normal(0.0, 5e-7, times.size)
  parameters = (InferenceParameter("material.shear_modulus_pa", 1500.0, 4000.0), InferenceParameter("material.viscosity_pa_s", 0.05, 0.2))
  independent = prepare_inference(config, FieldObservation("radius_m", times, observed, 5e-7), parameters)
  linked = prepare_inference(config, FieldObservation("radius_m", times, observed, 5e-7, correlation_time_s=_TAU), parameters)
  return times, observed, independent, linked


def test_correlated_likelihood_matches_a_direct_multivariate_gaussian(correlated, measured):
  """Whitening by the Cholesky factor must reproduce the full"""
  times, observed, _, linked = correlated
  unit = np.array([0.42, 0.37])
  lag = np.abs(times[:, None] - times[None, :])
  covariance = (5e-7) ** 2 * np.exp(-lag / _TAU)
  deviation = np.asarray(pyimr.simulate(times, linked.config_from_unit(unit)).radius_m) - observed
  direct = -0.5 * (deviation @ np.linalg.solve(covariance, deviation) + np.linalg.slogdet(2 * np.pi * covariance)[1])

  computed = linked.evaluate(unit).log_likelihood
  measured("correlated logL", f"{computed:.6f} vs direct {direct:.6f}")
  assert abs(computed - direct) < 1e-9 * abs(direct)


def test_the_correlated_gradient_is_still_the_derivative(correlated, measured):
  _, _, _, linked = correlated
  unit = np.array([0.42, 0.37])
  evaluation, jacobian = linked.evaluate_with_jacobian(unit)
  analytic = -np.asarray(evaluation.residual) @ jacobian
  step, difference = 1e-5, np.zeros(2)
  for index in range(2):
    offset = np.zeros(2)
    offset[index] = step
    ahead = linked.evaluate_with_jacobian(unit + offset)[0].log_likelihood
    behind = linked.evaluate_with_jacobian(unit - offset)[0].log_likelihood
    difference[index] = (ahead - behind) / (2.0 * step)
  error = float(np.max(np.abs(analytic - difference))) / max(float(np.max(np.abs(difference))), 1e-30)
  measured("correlated gradient", f"rel={error:.2e}")
  assert error < 1e-5


def test_vanishing_correlation_time_reduces_to_independent_noise(correlated):
  """The limit that must hold exactly, not approximately: at tau -> 0 the"""
  times, observed, independent, _ = correlated
  tiny = prepare_inference(independent.config, FieldObservation("radius_m", times, observed, 5e-7, correlation_time_s=1e-15), independent.parameters)
  unit = np.array([0.42, 0.37])
  assert tiny.evaluate(unit).log_likelihood == independent.evaluate(unit).log_likelihood


def test_correlated_noise_carries_less_information(correlated, measured):
  """The reason this matters for design. Neighbouring frames that share noise"""
  from pyimr.design import expected_information_gain

  _, _, independent, linked = correlated
  loose = expected_information_gain(independent, draws=6).expected_information_gain
  tight = expected_information_gain(linked, draws=6).expected_information_gain
  measured("EIG independent vs correlated", f"{loose:.3f} -> {tight:.3f} nats")
  assert tight < loose


def _shear_modulus_error(deborah, window_t0):
  """Cramer-Rao sd(G) at the truth, from the exact Jacobian."""
  from pyimr.design import _fisher

  from _validation_support import T0

  config = pyimr.SimulationConfig(R0=R0, Req=REQ, material=pyimr.Zener(2500.0, 0.1, deborah * T0, 0.2 * deborah * T0))
  times = np.linspace(1e-6, window_t0 * T0, 60)
  observed = np.asarray(pyimr.simulate(times, config).radius_m)
  parameters = (InferenceParameter("material.shear_modulus_pa", 500.0, 8000.0), InferenceParameter("material.viscosity_pa_s", 0.01, 0.5))
  inference = prepare_inference(config, FieldObservation("radius_m", times, observed, 5e-7), parameters)
  fisher = _fisher(inference, np.array([(2500.0 - 500.0) / 7500.0, (0.1 - 0.01) / 0.49]))
  return float(np.sqrt(np.linalg.inv(fisher)[0, 0]) * 7500.0)


def test_a_slowly_relaxing_material_hides_its_shear_modulus(measured):
  """G stops being identifiable as the relaxation time outgrows the observation window.

  The design module's whole job is to say what an experiment can teach, and this is the
  case where the answer is "not much". A grid scan of the likelihood gets this wrong --
  at long windows the surface is too sharp to resolve and the apparent maximum lands
  8000 log units below the truth -- so the Fisher information is the instrument, and
  asserting it here is what keeps that distinction from being rediscovered.
  """
  errors = [_shear_modulus_error(de, 1.7) for de in (2, 10, 50)]
  measured("sd(G)/G vs De at 1.7 T0", "  ".join(f"De={de}:{e / 2500:.2f}x" for de, e in zip((2, 10, 50), errors, strict=True)))
  assert errors[0] < errors[1] < errors[2], "G must get harder to identify as the material relaxes more slowly"
  assert errors[0] / 2500.0 < 0.1, "a fast-relaxing material should pin G"
  assert errors[-1] / 2500.0 > 0.4, "a slowly-relaxing one should not"


def test_watching_longer_recovers_the_shear_modulus(measured):
  """And the converse, which is the actionable half: it is the WINDOW, not the material.

  At De=50 the relaxation time is about 30x the short window, so the experiment ends
  before the material relaxes. Extending it restores G, which is a design
  recommendation rather than a limitation to document.
  """
  short, long = _shear_modulus_error(50, 1.7), _shear_modulus_error(50, 10)
  measured("sd(G)/G at De=50", f"1.7 T0: {short / 2500:.2f}x -> 10 T0: {long / 2500:.2f}x")
  assert long < short / 5.0, "a longer window must recover the shear modulus"


def test_batched_evaluation_matches_the_per_point_route(measured):
  """`evaluate_batch` runs ONE traced program over the stack; it must agree with the
  per-point forward solve it replaces.

  Worth asserting because the two routes are genuinely different solves -- the batch
  goes through the traced program with a `ForwardMode` adjoint, the per-point route
  through `simulate` -- so agreement is a claim, not a tautology. The batch is ~800x
  cheaper per point, which is what makes a posterior scan affordable (#129).
  """
  times = np.linspace(1e-6, 4e-5, 40)
  config = pyimr.SimulationConfig(R0=R0, Req=REQ, material=NHKV)
  observed = np.asarray(pyimr.simulate(times, config).radius_m) + np.random.default_rng(7).normal(0.0, 5e-7, times.size)
  parameters = (InferenceParameter("material.shear_modulus_pa", 1500.0, 4000.0), InferenceParameter("material.viscosity_pa_s", 0.05, 0.2))
  inference = prepare_inference(config, FieldObservation("radius_m", times, observed, 5e-7), parameters)

  points = np.random.default_rng(4).uniform(0.1, 0.9, size=(5, 2))
  batched = inference.evaluate_batch(points)
  worst = max(abs(item.log_likelihood - inference.evaluate(point).log_likelihood) for item, point in zip(batched, points, strict=True))
  measured("batched vs per-point logL", f"max |d|={worst:.1e}")
  assert len(batched) == len(points)
  assert worst < 1e-6, worst


@pytest.mark.slow
def test_the_parallel_batch_matches_the_serial_one(prepared_inference):
  """`workers > 1` was doubly broken: unpicklable, then deadlocked on fork with an

  initialised jax runtime -- the same failure that makes `pymc_op.sample_smc` force
  `cores=1`. It runs under `spawn`. This is slow because each worker re-imports jax,
  which is also why parallelism only pays for large batches.
  """
  _, inference = prepared_inference
  points = np.array([[0.3, 0.4], [0.6, 0.55]])

  serial = inference.evaluate_batch(points, workers=1)
  parallel = inference.evaluate_batch(points, workers=2)

  # Relative, because these are two computational routes, not two answers: `workers=1`
  # runs one vmapped program and `workers > 1` evaluates point by point. The residual
  # difference is NOT from the parallelism -- in-process, batched-vs-single reproduces it
  # exactly (2.086e-09 on this case, 2.4e-13 relative).
  worst = max(
    abs(a.log_likelihood - b.log_likelihood) / abs(a.log_likelihood) for a, b in zip(serial, parallel, strict=True)
  )
  assert worst < 1e-11, f"parallel batch disagrees with the serial one by {worst:.3e} relative"
