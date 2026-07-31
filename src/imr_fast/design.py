"""Laplace/Fisher expected information gain for experiment design (#25, piece 3)."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import warnings
from numbers import Integral

import numpy as np

from .inference import PreparedInference, RadiusObservation

__all__ = ["DesignEvaluation", "DesignInference", "design_inference", "design_information", "expected_information_gain"]
UNIFORM_VARIANCE = 1.0 / 12.0

@dataclass(frozen=True, slots=True)
class DesignEvaluation:
  """One scored design. `expected_information_gain` is in nats."""

  expected_information_gain: float
  standard_error: float
  draws: int
  successful: int
  failures: int

class DesignInference(PreparedInference):
  """A `PreparedInference` whose observations are placeholders."""

  __slots__ = ()

  def _refuse(self, name):
    raise TypeError(
      f"{name}() needs measured radii, and this inference holds placeholders. "
      "Build one with imr_fast.inference.prepare_inference once the experiment has been run."
    )

  def residual(self, unit_parameters): self._refuse("residual")
  def evaluate(self, unit_parameters): self._refuse("evaluate")
  def evaluate_batch(self, unit_parameters, workers=1): self._refuse("evaluate_batch")
  def fit_multistart(self, starts, **kwargs): self._refuse("fit_multistart")

def design_inference(config, time_s, standard_deviation_m, parameters):
  """A `DesignInference` for a design that has not been run."""
  placeholder = np.full(np.shape(np.asarray(time_s, dtype=float)), float(config.R0))
  return DesignInference(config, RadiusObservation(time_s, placeholder, standard_deviation_m), tuple(parameters))

def _fisher(inference, unit):
  jacobian = np.asarray(inference.jacobian(unit), dtype=float)
  return jacobian.T @ jacobian

def _gain_from_fisher(fisher, variance):
  scale = np.sqrt(variance)
  matrix = np.eye(len(variance)) + scale[:, None] * fisher * scale[None, :]
  return float(np.sum(np.log(np.diag(np.linalg.cholesky(matrix)))))

def _gain(inference, unit, variance): return _gain_from_fisher(_fisher(inference, unit), variance)

def _fisher_worker(argument):
  inference, unit = argument
  try:
    return _fisher(inference, unit)
  except Exception as error:  # noqa: BLE001 - any solver or factorisation failure
    return error

def design_information(inference, *, draws=128, seed=0, workers=1, max_failure_fraction=0.0, batched=False):
  """The `J^T J` of every prior draw, stacked."""
  _validate(inference, draws, workers, max_failure_fraction)
  points = np.random.default_rng(seed).random((int(draws), inference.size))
  if batched:
    if max_failure_fraction: raise ValueError("batched=True cannot honour max_failure_fraction: one traced program fails as a whole")
    jacobians = inference.jacobians(points)
    return np.einsum("dop,doq->dpq", jacobians, jacobians), len(points), 0
  arguments = [(inference, point) for point in points]
  if workers == 1:
    outcomes = [_fisher_worker(argument) for argument in arguments]
  else:
    with ProcessPoolExecutor(max_workers=workers) as executor:
      outcomes = list(executor.map(_fisher_worker, arguments))
  matrices = [value for value in outcomes if not isinstance(value, Exception)]
  errors = [value for value in outcomes if isinstance(value, Exception)]
  requested = len(outcomes)
  if errors:
    fraction = len(errors) / requested
    if not matrices or fraction > max_failure_fraction:
      raise RuntimeError(
        f"{len(errors)} of {requested} design draws failed "
        f"({fraction:.1%} > max_failure_fraction={max_failure_fraction:.1%}); first failure shown as the cause"
      ) from errors[0]
    warnings.warn(
      f"{len(errors)} of {requested} design draws failed ({fraction:.1%}); "
      f"the reported gain is conditional on the {len(matrices)} that succeeded. "
      f"First failure: {type(errors[0]).__name__}: {errors[0]}",
      RuntimeWarning,
      stacklevel=3,
    )
  return np.array(matrices), requested, len(errors)

def _validate(inference, draws, workers, max_failure_fraction):
  if not isinstance(inference, PreparedInference): raise TypeError("inference must be a PreparedInference")
  if not isinstance(draws, Integral) or draws < 1: raise ValueError("draws must be a positive integer")
  if not isinstance(workers, Integral) or workers < 1: raise ValueError("workers must be a positive integer")
  if not 0.0 <= max_failure_fraction < 1.0: raise ValueError("max_failure_fraction must be in [0, 1)")

def expected_information_gain(inference, *, draws=128, seed=0, prior_variance=None, workers=1, max_failure_fraction=0.0, information=None, batched=False):
  """Prior-averaged Laplace EIG for one design, with its Monte Carlo error bar."""
  _validate(inference, draws, workers, max_failure_fraction)
  if prior_variance is None:
    variance = np.full(inference.size, UNIFORM_VARIANCE)
  else:
    variance = np.broadcast_to(np.asarray(prior_variance, dtype=float), (inference.size,)).astype(float)
    if np.any(variance <= 0.0) or not np.all(np.isfinite(variance)): raise ValueError("prior_variance must be finite and positive")
  if information is None:
    information = design_information(inference, draws=draws, seed=seed, workers=workers, max_failure_fraction=max_failure_fraction, batched=batched)
  matrices, requested, failures = information
  gains = np.array([_gain_from_fisher(matrix, variance) for matrix in matrices], dtype=float)
  error = float(np.std(gains, ddof=1) / np.sqrt(gains.size)) if gains.size > 1 else float("inf")
  return DesignEvaluation(float(np.mean(gains)), error, requested, int(gains.size), failures)

def _time_gradient(inference, unit, variance):
  jacobian, derivative = inference.jacobian_with_time_derivative(unit)
  scale = np.sqrt(variance)
  scaled = np.asarray(jacobian, dtype=float) * scale
  scaled_rate = np.asarray(derivative, dtype=float) * scale
  matrix = np.eye(len(variance)) + scaled.T @ scaled
  return np.einsum("ij,ij->i", scaled_rate, np.linalg.solve(matrix, scaled.T).T)

def information_time_gradient(inference, *, draws=128, seed=0, prior_variance=None):
  """Prior-averaged `d(EIG)/d(observation time)`, one entry per observed value."""
  if not isinstance(inference, PreparedInference): raise TypeError("inference must be a PreparedInference")
  if not isinstance(draws, Integral) or draws < 1: raise ValueError("draws must be a positive integer")
  if prior_variance is None:
    variance = np.full(inference.size, UNIFORM_VARIANCE)
  else:
    variance = np.broadcast_to(np.asarray(prior_variance, dtype=float), (inference.size,)).astype(float)
  points = np.random.default_rng(seed).random((int(draws), inference.size))
  return np.mean([_time_gradient(inference, point, variance) for point in points], axis=0)
