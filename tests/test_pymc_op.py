"""The PyMC bridge (#25, piece 1)."""

import sys
import types

import numpy as np
import pytest

import pyimr
from _validation_support import R0, REQ
from pyimr.inference import InferenceParameter, RadiusObservation, prepare_inference

pytest.importorskip("pymc")
from pyimr import pymc_op  # noqa: E402

SECTION = "5. PyMC bridge"
_TIMES = np.linspace(0.0, 20e-6, 60)
_NOISE = 5e-7


@pytest.fixture(scope="module")
def inference():
  config = pyimr.SimulationConfig(R0, REQ, pyimr.NeoHookeanKelvinVoigt(2500.0, 0.1))
  truth = pyimr.simulate(_TIMES, config)
  observed = truth.radius_m + np.random.default_rng(7).normal(0.0, _NOISE, _TIMES.size)
  return prepare_inference(
    config,
    RadiusObservation(_TIMES, observed, _NOISE),
    (InferenceParameter("material.shear_modulus_pa", 1500.0, 4000.0), InferenceParameter("material.viscosity_pa_s", 0.05, 0.2)),
  )


@pytest.mark.parametrize("point", ([0.4, 0.35], [0.5, 0.5], [0.62, 0.44]))
def test_gradient_matches_central_difference(inference, point, measured):
  """The bridge is one contraction, `-r @ J`. A central difference is an"""
  unit = np.array(point)
  analytic = pymc_op._log_likelihood_and_gradient(inference, unit)[1]

  step, difference = 1e-5, np.zeros(inference.size)
  for index in range(inference.size):
    offset = np.zeros(inference.size)
    offset[index] = step
    ahead = pymc_op._log_likelihood_and_gradient(inference, unit + offset)[0]
    behind = pymc_op._log_likelihood_and_gradient(inference, unit - offset)[0]
    difference[index] = (ahead - behind) / (2.0 * step)

  error = float(np.max(np.abs(analytic - difference))) / max(float(np.max(np.abs(difference))), 1e-30)
  measured(f"pymc gradient at {point}", f"rel={error:.2e}")
  assert error < 1e-6


def test_one_solve_serves_both_halves(inference, monkeypatch):
  """A leapfrog step used to pay two forward solves plus two sensitivity solves,"""
  calls = []
  real = type(inference).evaluate_with_jacobian
  monkeypatch.setattr(type(inference), "evaluate_with_jacobian", lambda self, unit: (calls.append(np.asarray(unit).copy()), real(self, unit))[1])
  import pytensor.tensor as tensor

  operation = pymc_op.IMRLogLikelihood(inference)
  unit = tensor.as_tensor_variable(np.array([0.5, 0.5]))
  operation.log_likelihood(unit).eval()
  operation.gradient(unit).eval()
  assert len(calls) == 1, f"expected one solve for logp and dlogp at the same point, got {len(calls)}"


def test_failed_solves_are_rejected_not_smoothed(inference):
  """A stiff solve can fail at values a sampler proposes. Returning -inf rejects"""
  log_likelihood, gradient = pymc_op._log_likelihood_and_gradient(inference, np.array([np.nan, 0.5]))
  assert log_likelihood == -np.inf
  assert gradient.shape == (inference.size,) and np.all(gradient == 0.0)


def test_model_exposes_physical_parameters(inference):
  """A trace should carry what was fitted, not unit-cube coordinates."""
  model = pymc_op.build_model(inference)
  names = {variable.name for variable in model.deterministics}
  assert names == {"material.shear_modulus_pa", "material.viscosity_pa_s"}


def test_operation_rejects_foreign_input():
  with pytest.raises(TypeError, match="PreparedInference"):
    pymc_op.IMRLogLikelihood(object())


def test_missing_pymc_gives_an_actionable_error(monkeypatch):
  """The dependency is optional, so a core install must not see an ImportError"""
  import builtins

  real_import = builtins.__import__

  def blocked(name, *args, **kwargs):
    if name.split(".")[0] in ("pymc", "pytensor"):
      raise ImportError(f"No module named {name!r}")
    return real_import(name, *args, **kwargs)

  monkeypatch.setattr(builtins, "__import__", blocked)
  for module in [name for name in sys.modules if name.split(".")[0] in ("pymc", "pytensor")]:
    monkeypatch.delitem(sys.modules, module)

  with pytest.raises(ImportError, match=r"PyIMR\[inference\]"):
    pymc_op._pymc()


def test_sampling_plumbing_runs(inference):
  """Smoke test only: two draws, asserting shape rather than statistics."""
  trace = pymc_op.sample_posterior(inference, draws=2, tune=2, chains=1, progressbar=False, random_seed=3, compute_convergence_checks=False)
  for name in ("material.shear_modulus_pa", "material.viscosity_pa_s"):
    assert np.asarray(trace.posterior[name]).size == 2


_FIXTURE_LOG_EVIDENCE = 786.43724


def test_smc_evidence_matches_exact_quadrature(inference, measured):
  """The quantity that justifies SMC's existence here, against a known answer."""
  trace = pymc_op.sample_smc(inference, draws=64, chains=2, progressbar=False, random_seed=3)
  assert np.asarray(trace.posterior["material.shear_modulus_pa"]).size == 128
  evidence = pymc_op.log_marginal_likelihood(trace)
  measured("SMC evidence vs quadrature", f"{evidence:.3f} vs {_FIXTURE_LOG_EVIDENCE:.3f}")
  assert abs(evidence - _FIXTURE_LOG_EVIDENCE) < 1.0, (
    f"SMC evidence {evidence:.3f} is not the {_FIXTURE_LOG_EVIDENCE:.3f} that exact quadrature gives"
  )


def test_log_marginal_likelihood_survives_ragged_tempering():
  """Adaptive tempering does not give every chain the same number of stages."""
  stub = types.SimpleNamespace(
    sample_stats={"log_marginal_likelihood": types.SimpleNamespace(values=np.array([[1.0, 2.0, 3.0], [4.0, 5.0]], dtype=object))}
  )
  assert pymc_op.log_marginal_likelihood(stub) == pytest.approx(5.0 + np.log((np.exp(-2.0) + 1.0) / 2.0))

  even = types.SimpleNamespace(sample_stats={"log_marginal_likelihood": types.SimpleNamespace(values=np.array([[1.0, 3.0], [2.0, 5.0]]))})
  combined = pymc_op.log_marginal_likelihood(even)
  assert combined == pytest.approx(5.0 + np.log((np.exp(-2.0) + 1.0) / 2.0))
  assert combined != pytest.approx(5.0), "keeping only the last chain would give 5.0 and discard the other"
