"""Laplace/Fisher expected information gain for experiment design (#25, piece 3).

For Gaussian observation noise and a Gaussian prior of covariance `Sigma`, the
expected information gain of a design is

    EIG ~ 0.5 * E_theta[ log det(I + Sigma * J^T J) ]

where `J` is the Jacobian of the standardised residual. `PreparedInference`
already returns exactly that `J` -- divided by sigma and chain-ruled to the unit
parameters -- so `J^T J` *is* the Fisher information in unit coordinates and no
new derivative work is needed. This module is the expectation and the
determinant; the physics is upstream.

A design is scored without data. `PreparedInference.jacobian` never reads
`observation.radius_m`: the Fisher information depends on the observation times,
the noise level, and the configuration being probed, not on what would come back
from the experiment. `design_inference` fills that slot with a placeholder,
which is why a design can be ranked before it is run.

The prior is the unit cube, matching `imr_fast.pymc_bridge`: each parameter is Uniform(0, 1)
there, and a uniform variance is 1/12. Moment-matching a uniform to a Gaussian
is an approximation, and it is the second one here -- the first being the
linearisation itself. Both are stated rather than hidden, and `prior_variance`
overrides the default when a previous experiment has tightened the prior.

Draws are iid, not a Sobol or Latin sequence. A low-discrepancy sequence would
converge faster, but then `standard_error` -- the whole point of reporting one --
would no longer be valid, and without an error bar there is no way to say whether
two designs actually differ. Variance reduction is available by raising `draws`.

On the linearisation: the criterion uses the tangent at each draw, so it inherits
whatever the tangent misses about curvature. It is averaged over the prior rather
than evaluated at a nominal theta precisely because the per-draw gains vary
strongly -- tests/test_design.py measures that spread, which is the honest bound
on how much a nominal-tangent design score can be trusted.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from .inference import PreparedInference, RadiusObservation, prepare_inference

__all__ = ["DesignEvaluation", "design_inference", "expected_information_gain"]

UNIFORM_VARIANCE = 1.0 / 12.0


@dataclass(frozen=True, slots=True)
class DesignEvaluation:
  """One scored design. `expected_information_gain` is in nats."""

  expected_information_gain: float
  standard_error: float
  draws: int
  failures: int


def design_inference(config, time_s, standard_deviation_m, parameters):
  """A `PreparedInference` for a design that has not been run.

  The placeholder radii are never read by `jacobian`; see the module docstring.
  """
  placeholder = np.full(np.shape(np.asarray(time_s, dtype=float)), float(config.R0))
  return prepare_inference(config, RadiusObservation(time_s, placeholder, standard_deviation_m), parameters)


def _gain(inference, unit, variance):
  """0.5 * log det(I + Sigma * J^T J), via Cholesky on the symmetrised form.

  `sqrt(Sigma) J^T J sqrt(Sigma)` has the same determinant as `Sigma J^T J` but
  is symmetric positive definite, so the factorisation both stays stable and
  fails loudly if the information matrix is not what it should be.
  """
  jacobian = np.asarray(inference.jacobian(unit), dtype=float)
  scale = np.sqrt(variance)
  matrix = np.eye(inference.size) + scale[:, None] * (jacobian.T @ jacobian) * scale[None, :]
  return float(np.sum(np.log(np.diag(np.linalg.cholesky(matrix)))))


def _gain_worker(argument):
  """A failed solve is a NaN, dropped and counted -- not a zero, which would
  read as an informative design that happens to teach nothing."""
  inference, unit, variance = argument
  try:
    return _gain(inference, unit, variance)
  except Exception:  # noqa: BLE001 - any solver or factorisation failure
    return float("nan")


def expected_information_gain(inference, *, draws=128, seed=0, prior_variance=None, workers=1):
  """Prior-averaged Laplace EIG for one design, with its Monte Carlo error bar."""
  if not isinstance(inference, PreparedInference):
    raise TypeError("inference must be a PreparedInference")
  if not isinstance(draws, Integral) or draws < 1:
    raise ValueError("draws must be a positive integer")
  if not isinstance(workers, Integral) or workers < 1:
    raise ValueError("workers must be a positive integer")
  if prior_variance is None:
    variance = np.full(inference.size, UNIFORM_VARIANCE)
  else:
    variance = np.broadcast_to(np.asarray(prior_variance, dtype=float), (inference.size,)).astype(float)
    if np.any(variance <= 0.0) or not np.all(np.isfinite(variance)):
      raise ValueError("prior_variance must be finite and positive")

  points = np.random.default_rng(seed).random((int(draws), inference.size))
  arguments = [(inference, point, variance) for point in points]
  if workers == 1:
    gains = np.array([_gain_worker(argument) for argument in arguments])
  else:
    with ProcessPoolExecutor(max_workers=workers) as executor:
      gains = np.array(list(executor.map(_gain_worker, arguments)))

  finite = gains[np.isfinite(gains)]
  if finite.size == 0:
    raise RuntimeError("every design draw failed to produce a Jacobian")
  error = float(np.std(finite, ddof=1) / np.sqrt(finite.size)) if finite.size > 1 else float("inf")
  return DesignEvaluation(float(np.mean(finite)), error, int(finite.size), int(gains.size - finite.size))
