"""PyMC bridge: drive NUTS from the exact forward sensitivities (#25, piece 1)."""

from __future__ import annotations

import numpy as np

from .inference import PreparedInference

__all__ = ["IMRLogLikelihood", "build_model", "log_marginal_likelihood", "sample_posterior", "sample_smc"]
_MISSING = "imr_fast.pymc_op requires PyMC: pip install 'imr-fast[inference]'"

def _pymc():
  try:
    import pymc
    import pytensor.tensor as tensor
    from pytensor.graph.op import Op
  except ImportError as error:  # pragma: no cover - exercised only without pymc
    raise ImportError(_MISSING) from error
  return pymc, tensor, Op

def _log_likelihood_and_gradient(inference, unit):
  try:
    evaluation, jacobian = inference.evaluate_with_jacobian(unit)
  except Exception:  # noqa: BLE001 - any solver failure is a rejected proposal
    return -np.inf, np.zeros(inference.size)
  gradient = -np.asarray(evaluation.residual) @ np.asarray(jacobian)
  return float(evaluation.log_likelihood), np.asarray(gradient, dtype=float)

def _make_ops(inference):
  _, tensor, Op = _pymc()
  memo = {}

  def evaluate(unit):
    key = np.asarray(unit, dtype=float).tobytes()
    if memo.get("key") != key:
      memo["key"] = key
      memo["value"] = _log_likelihood_and_gradient(inference, unit)
    return memo["value"]

  class Gradient(Op):
    itypes, otypes = [tensor.dvector], [tensor.dvector]

    def perform(self, node, inputs, output_storage): output_storage[0][0] = evaluate(inputs[0])[1]

  class LogLikelihood(Op):
    itypes, otypes = [tensor.dvector], [tensor.dscalar]

    def perform(self, node, inputs, output_storage): output_storage[0][0] = np.array(evaluate(inputs[0])[0])
    def grad(self, inputs, output_grads): return [output_grads[0] * gradient_op(inputs[0])]

  gradient_op = Gradient()
  return LogLikelihood(), gradient_op

class IMRLogLikelihood:
  """Callable PyTensor `Op` pair for one `PreparedInference`."""

  def __init__(self, inference):
    if not isinstance(inference, PreparedInference): raise TypeError("inference must be a PreparedInference")
    self.inference = inference
    self.log_likelihood, self.gradient = _make_ops(inference)

  def __call__(self, unit_parameters): return self.log_likelihood(unit_parameters)

def build_model(inference, name="unit"):
  """A PyMC model sampling the unit cube, with the IMR likelihood as a Potential."""
  pymc, tensor, _ = _pymc()
  operation = IMRLogLikelihood(inference)
  with pymc.Model() as model:
    unit = pymc.Uniform(name, 0.0, 1.0, shape=inference.size)
    pymc.Potential("imr_log_likelihood", operation(unit))
    for index, parameter in enumerate(inference.parameters):
      lower, upper = parameter.lower, parameter.upper
      value = lower + unit[index] * (upper - lower)
      if parameter.transform == "log": value = lower * (upper / lower) ** unit[index]
      pymc.Deterministic(parameter.path, value)
  return model

def sample_posterior(inference, draws=1000, tune=1000, chains=4, **kwargs):
  """NUTS over the unit cube. `kwargs` pass straight through to `pymc.sample`."""
  pymc, _, _ = _pymc()
  with build_model(inference):
    return pymc.sample(draws=draws, tune=tune, chains=chains, **kwargs)

def sample_smc(inference, draws=1000, chains=4, **kwargs):
  """Sequential Monte Carlo over the unit cube (#25, piece 2)."""
  pymc, _, _ = _pymc()
  kwargs.setdefault("cores", 1)
  with build_model(inference):
    return pymc.sample_smc(draws=draws, chains=chains, **kwargs)

def log_marginal_likelihood(trace):
  """The evidence from a `sample_smc` trace, as one number."""
  raw = np.atleast_1d(trace.sample_stats["log_marginal_likelihood"].values)
  per_chain = np.array([float(np.asarray(chain, dtype=float).ravel()[-1]) for chain in raw])
  peak = per_chain.max()
  return float(peak + np.log(np.mean(np.exp(per_chain - peak))))
