"""The PyMC bridge (#25, piece 1).

PyMC is optional, so every test here skips without it rather than failing. The
core install stays numpy/scipy/numba.
"""

import sys

import numpy as np
import pytest

import imr_fast
from _validation_support import R0, REQ
from imr_fast.inference import InferenceParameter, RadiusObservation, prepare_inference

pytest.importorskip("pymc")
from imr_fast import pymc_op  # noqa: E402

SECTION = "5. PyMC bridge"
_TIMES = np.linspace(0.0, 20e-6, 60)
_NOISE = 5e-7


@pytest.fixture(scope="module")
def inference():
  config = imr_fast.SimulationConfig(R0, REQ, imr_fast.NeoHookeanKelvinVoigt(2500.0, 0.1))
  truth = imr_fast.simulate(_TIMES, config)
  observed = truth.radius_m + np.random.default_rng(7).normal(0.0, _NOISE, _TIMES.size)
  return prepare_inference(
    config,
    RadiusObservation(_TIMES, observed, _NOISE),
    (
      InferenceParameter("material.shear_modulus_pa", 1500.0, 4000.0),
      InferenceParameter("material.viscosity_pa_s", 0.05, 0.2),
    ),
  )


@pytest.mark.parametrize("point", ([0.4, 0.35], [0.5, 0.5], [0.62, 0.44]))
def test_gradient_matches_central_difference(inference, point, measured):
  """The bridge is one contraction, `-r @ J`. A central difference is an
  adequate reference *here* -- unlike in #44, what is being checked is the chain
  rule, not the Jacobian, and that Jacobian is verified against exact tangents
  elsewhere.

  The difference is taken of the log-likelihood the Op itself returns, not of
  `inference.evaluate`. Those are now different integrations: the Op takes both
  halves from one sensitivity solve, while `evaluate` runs a plain forward solve
  converged separately to the same tolerance. Differencing the wrong one
  measures the gap between two integrations rather than an error in the
  gradient, and reports ~1e-04 for a gradient that is right to 5e-08.
  """
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
  """A leapfrog step used to pay two forward solves plus two sensitivity solves,
  because each Op discarded half of what it computed. It should now pay one."""
  calls = []
  real = type(inference).evaluate_with_jacobian
  monkeypatch.setattr(
    type(inference),
    "evaluate_with_jacobian",
    lambda self, unit: (calls.append(np.asarray(unit).copy()), real(self, unit))[1],
  )
  import pytensor.tensor as tensor

  operation = pymc_op.IMRLogLikelihood(inference)
  unit = tensor.as_tensor_variable(np.array([0.5, 0.5]))
  operation.log_likelihood(unit).eval()
  operation.gradient(unit).eval()
  assert len(calls) == 1, f"expected one solve for logp and dlogp at the same point, got {len(calls)}"


def test_failed_solves_are_rejected_not_smoothed(inference):
  """A stiff solve can fail at values a sampler proposes. Returning -inf rejects
  the proposal; a large finite number would bias the posterior toward it."""
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
  """The dependency is optional, so a core install must not see an ImportError
  raised three frames inside pytensor. Simulating absence is the only way to
  test this where PyMC *is* installed -- and it is the configuration every user
  without the extra will actually have."""
  import builtins

  real_import = builtins.__import__

  def blocked(name, *args, **kwargs):
    if name.split(".")[0] in ("pymc", "pytensor"):
      raise ImportError(f"No module named {name!r}")
    return real_import(name, *args, **kwargs)

  monkeypatch.setattr(builtins, "__import__", blocked)
  for module in [name for name in sys.modules if name.split(".")[0] in ("pymc", "pytensor")]:
    monkeypatch.delitem(sys.modules, module)

  with pytest.raises(ImportError, match=r"imr-fast\[inference\]"):
    pymc_op._pymc()


def test_sampling_plumbing_runs(inference):
  """Smoke test only: two draws, asserting shape rather than statistics.

  A posterior-recovery test is deliberately NOT here. Sampling cost is unbounded
  in a way that does not belong in a suite -- NUTS proposes parameter values
  during tuning where the stiff solve is very slow, so the wall time depends on
  the seed rather than on the draw count. Measured: 60 draws x 2 chains took 62 s,
  while 25 x 1 with a different configuration had not finished in five minutes.

  Recovery was checked by hand instead, and the numbers are in the module
  docstring. What this suite owns is the gradient, which is tested above; PyMC's
  sampler is not ours to validate.
  """
  trace = pymc_op.sample_posterior(
    inference, draws=2, tune=2, chains=1, progressbar=False, random_seed=3, compute_convergence_checks=False
  )
  for name in ("material.shear_modulus_pa", "material.viscosity_pa_s"):
    assert np.asarray(trace.posterior[name]).size == 2
