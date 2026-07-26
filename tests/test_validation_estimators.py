"""Trace-side estimators (`imr_data`) and the prepared inference layer.

Numerical content unchanged from `run_validation.py`; see issue #32.
"""

import numpy as np
import pytest

import imr_data
import imr_fast
from _validation_support import NHKV, R0, REQ
from imr_inference import InferenceParameter, RadiusObservation, prepare_inference

SECTION = "3b. Trace estimators and prepared inference"


def test_equilibrium_radius_round_trip(measured):
  """The estimator must invert the solver's own pressure/radius relation."""
  gas_pressure = (imr_fast.P8 + 2 * imr_fast.SURF / REQ) * (REQ / R0) ** (3 * imr_fast.KAPPA)
  error = abs(imr_data.equilibrium_radius(R0, gas_pressure) - REQ) / REQ
  measured("equilibrium radius round-trip", f"rel={error:.2e}")
  assert error < 1e-12


def test_natural_frequency_reduces_to_minnaert(measured):
  """The gas-only limit must reproduce Minnaert exactly."""
  computed, _ = imr_data.natural_frequency(R0, REQ, 1e-12, 1e-12, surface_tension_n_m=0.0)
  minnaert = np.sqrt(3 * imr_fast.KAPPA * imr_fast.P8 / imr_fast.RHO) / REQ
  error = abs(computed - minnaert) / minnaert
  measured("natural frequency -> Minnaert", f"rel={error:.2e}")
  assert error < 1e-12


@pytest.fixture(scope="module")
def rebound_trace():
  times = np.linspace(0, 300e-6, 8000)
  radius = imr_fast.simulate(times, imr_fast.SimulationConfig(R0=R0, Req=REQ, material=NHKV)).radius_m
  return imr_data.collapse_features(times, radius)


def test_natural_frequency_matches_measured_rebound(rebound_trace, measured):
  collapse_times, _, _ = rebound_trace
  # The first rebound is strongly nonlinear and the late tail is numerical
  # wiggle, so take the median of the intermediate periods.
  observed = float(np.median(2 * np.pi / np.diff(collapse_times)[1:5]))
  predicted, _ = imr_data.natural_frequency(R0, REQ, 2500.0, 0.1)
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
  convergence = imr_data.resolution_convergence(
    imr_fast.SimulationConfig(R0=R0, Req=REQ, material=NHKV, bubtherm=1, Nt=10, Mt=10),
    np.linspace(0, 60e-6, 200),
    [(10, 10), (20, 20), (40, 40)],
  )
  errors = [error for _, error in convergence]
  measured("thermal grid convergence", " ".join(f"{error:.1e}" for error in errors))
  assert errors[0] > errors[1] > errors[2] == 0.0


@pytest.fixture(scope="module")
def prepared_inference():
  config = imr_fast.SimulationConfig(R0, REQ, NHKV)
  times = np.linspace(0.0, 20e-6, 50)
  truth = imr_fast.simulate(times, config)
  return times, prepare_inference(
    config,
    RadiusObservation(times, truth.radius_m, 1e-8),
    (
      InferenceParameter("material.shear_modulus_pa", 2000.0, 3000.0),
      InferenceParameter("material.viscosity_pa_s", 0.05, 0.15),
    ),
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
