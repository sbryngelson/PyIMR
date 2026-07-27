"""PyMC bridge: drive NUTS from the exact forward sensitivities (#25, piece 1).

`PreparedInference` already returns the standardised residual and its Jacobian
with respect to the unit parameters, so the gradient of the Gaussian
log-likelihood is one contraction:

    log L = -0.5 * sum(r**2 + log(2*pi*sigma**2))
    d log L / du_k = -sum(r * dr/du_k)

That is the whole bridge. Nothing here re-derives physics, and no finite
differences are involved -- the Jacobian is the same exact tangent the solver
produces, which is the reason this package exists.

PyMC is an OPTIONAL dependency (`pip install imr-fast[inference]`); importing
this module without it raises with an actionable message rather than an
ImportError from three frames down.

Sampling happens on the unit cube, where every parameter is Uniform(0, 1) and
`InferenceParameter` owns the transform back to physical units (including the
log option). That keeps the prior specification in one place and means the
sampler never sees a bound it can walk through.

On robustness: a stiff IMR solve can fail outright at parameter values a sampler
will propose. Those return -inf with a zero gradient, which PyMC treats as a
rejected proposal. Silently returning a large finite number instead would bias
the posterior toward the failure region.

Verified by hand on synthetic data (NHKV, 60 observations at sigma = 5e-7 m,
truth G = 2500 Pa and mu = 0.1 Pa s): 60 draws x 2 chains in 62 s recovered
G = 2779 +- 216 and mu = 0.1097 +- 0.0069, both inside 1.5 sd of the truth.

Sampling cost is dominated by the solver and is not proportional to the draw
count: tuning proposes parameter values where the stiff integration is very
slow, so wall time varies strongly with the seed. Budget by trial, not by
arithmetic on a per-draw figure.
"""

from __future__ import annotations

import numpy as np

import imr_inference

__all__ = ["IMRLogLikelihood", "build_model", "sample_posterior"]

_MISSING = "imr_pymc requires PyMC: pip install 'imr-fast[inference]'"


def _pymc():
  try:
    import pymc
    import pytensor.tensor as tensor
    from pytensor.graph.op import Op
  except ImportError as error:  # pragma: no cover - exercised only without pymc
    raise ImportError(_MISSING) from error
  return pymc, tensor, Op


def _log_likelihood_and_gradient(inference, unit):
  """Both at once. Returns (-inf, zeros) when the solve fails."""
  try:
    evaluation = inference.evaluate(unit)
    jacobian = inference.jacobian(unit)
  except Exception:  # noqa: BLE001 - any solver failure is a rejected proposal
    return -np.inf, np.zeros(inference.size)
  gradient = -np.asarray(evaluation.residual) @ np.asarray(jacobian)
  return float(evaluation.log_likelihood), np.asarray(gradient, dtype=float)


def _make_ops(inference):
  _, tensor, Op = _pymc()

  class Gradient(Op):
    itypes, otypes = [tensor.dvector], [tensor.dvector]

    def perform(self, node, inputs, output_storage):
      output_storage[0][0] = _log_likelihood_and_gradient(inference, inputs[0])[1]

  class LogLikelihood(Op):
    itypes, otypes = [tensor.dvector], [tensor.dscalar]

    def perform(self, node, inputs, output_storage):
      output_storage[0][0] = np.array(_log_likelihood_and_gradient(inference, inputs[0])[0])

    def grad(self, inputs, output_grads):
      return [output_grads[0] * gradient_op(inputs[0])]

  gradient_op = Gradient()
  return LogLikelihood(), gradient_op


class IMRLogLikelihood:
  """Callable PyTensor `Op` pair for one `PreparedInference`.

  Held as a class rather than a bare `Op` so the two operations share a single
  prepared problem: the `Op` instances close over it, and rebuilding them per
  call would re-run `imr_fast.prepare` on every gradient evaluation.
  """

  def __init__(self, inference):
    if not isinstance(inference, imr_inference.PreparedInference):
      raise TypeError("inference must be a PreparedInference")
    self.inference = inference
    self.log_likelihood, self.gradient = _make_ops(inference)

  def __call__(self, unit_parameters):
    return self.log_likelihood(unit_parameters)


def build_model(inference, name="unit"):
  """A PyMC model sampling the unit cube, with the IMR likelihood as a Potential.

  The physical parameters are recorded as deterministics, so a trace carries the
  quantities that were fitted rather than requiring the caller to re-apply each
  transform afterwards.
  """
  pymc, tensor, _ = _pymc()
  operation = IMRLogLikelihood(inference)
  with pymc.Model() as model:
    unit = pymc.Uniform(name, 0.0, 1.0, shape=inference.size)
    pymc.Potential("imr_log_likelihood", operation(unit))
    for index, parameter in enumerate(inference.parameters):
      lower, upper = parameter.lower, parameter.upper
      value = lower + unit[index] * (upper - lower)
      if parameter.transform == "log":
        value = lower * (upper / lower) ** unit[index]
      pymc.Deterministic(parameter.path, value)
  return model


def sample_posterior(inference, draws=1000, tune=1000, chains=4, **kwargs):
  """NUTS over the unit cube. `kwargs` pass straight through to `pymc.sample`."""
  pymc, _, _ = _pymc()
  with build_model(inference):
    return pymc.sample(draws=draws, tune=tune, chains=chains, **kwargs)
