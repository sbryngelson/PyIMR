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

The prior is the unit cube, matching `imr_fast.pymc_op`: each parameter is Uniform(0, 1)
there, and a uniform variance is 1/12. Moment-matching a uniform to a Gaussian
is an approximation, and it is the second one here -- the first being the
linearisation itself. Both are stated rather than hidden, and `prior_variance`
overrides the default when a previous experiment has tightened the prior.

Draws are iid, not a Sobol or Latin sequence. A low-discrepancy sequence would
converge faster, but then `standard_error` -- the whole point of reporting one --
would no longer be valid, and without an error bar there is no way to say whether
two designs actually differ. Variance reduction is available by raising `draws`.

That reasoning is also why a failed draw raises rather than being dropped. An
average over the survivors estimates `E[gain | the solve succeeded]`, and the
error bar computed from them describes that conditional quantity while looking
exactly like a prior average. `max_failure_fraction` opts in to censoring, and
the result then reports `draws`, `successful` and `failures` separately.

On the linearisation: the criterion uses the tangent at each draw, so it inherits
whatever the tangent misses about curvature. It is averaged over the prior rather
than evaluated at a nominal theta precisely because the per-draw gains vary
strongly -- tests/test_design.py measures that spread, which is the honest bound
on how much a nominal-tangent design score can be trusted.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import warnings
from numbers import Integral

import numpy as np

from .inference import PreparedInference, RadiusObservation

__all__ = ["DesignEvaluation", "DesignInference", "design_inference", "expected_information_gain"]

UNIFORM_VARIANCE = 1.0 / 12.0


@dataclass(frozen=True, slots=True)
class DesignEvaluation:
  """One scored design. `expected_information_gain` is in nats.

  `draws` is what was requested and `successful` is what reached the average, so
  censoring is visible in the result rather than inferable by comparing `draws`
  against the argument that produced it.
  """

  expected_information_gain: float
  standard_error: float
  draws: int
  successful: int
  failures: int


class DesignInference(PreparedInference):
  """A `PreparedInference` whose observations are placeholders.

  A design is scored before it is run, so there are no measured radii. The
  Fisher information does not need any -- `jacobian` never reads
  `observation.radius_m` -- but the resulting object is otherwise a fully
  functional likelihood, and calling `evaluate` or `fit_multistart` on it would
  fit against a constant fabricated trace and return a confident, meaningless
  answer. `RadiusObservation` cannot catch that: it validates positivity and
  finiteness, which a placeholder satisfies.

  So the data-consuming methods refuse instead. Scoring works; fitting does not.
  """

  __slots__ = ()

  def _refuse(self, name):
    raise TypeError(
      f"{name}() needs measured radii, and this inference holds placeholders. "
      "Build one with imr_fast.inference.prepare_inference once the experiment has been run."
    )

  def residual(self, unit_parameters):
    self._refuse("residual")

  def evaluate(self, unit_parameters):
    self._refuse("evaluate")

  def evaluate_batch(self, unit_parameters, workers=1):
    self._refuse("evaluate_batch")

  def fit_multistart(self, starts, **kwargs):
    self._refuse("fit_multistart")


def design_inference(config, time_s, standard_deviation_m, parameters):
  """A `DesignInference` for a design that has not been run."""
  placeholder = np.full(np.shape(np.asarray(time_s, dtype=float)), float(config.R0))
  return DesignInference(config, RadiusObservation(time_s, placeholder, standard_deviation_m), tuple(parameters))


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
  """Returns the gain, or the exception that prevented it.

  The exception is carried rather than collapsed to NaN so the caller can chain
  it. A bare handler here would turn a stale signature or a bad parameter path
  -- programming errors, not stiff solves -- into a silent failure count.
  """
  inference, unit, variance = argument
  try:
    return _gain(inference, unit, variance)
  except Exception as error:  # noqa: BLE001 - any solver or factorisation failure
    return error


def expected_information_gain(
  inference, *, draws=128, seed=0, prior_variance=None, workers=1, max_failure_fraction=0.0
):
  """Prior-averaged Laplace EIG for one design, with its Monte Carlo error bar.

  Failed draws are **not** silently dropped. A censored average estimates
  `E[gain | the solve succeeded]`, not `E_theta[gain]`, and the two differ most
  exactly where it matters: a design whose solves fail in the violent, highly
  informative region of the prior scores as though that region did not exist,
  under an error bar that looks just as tight as a clean design's. Ranking two
  designs then compares two different quantities.

  So any failure raises by default. `max_failure_fraction` opts in to censoring
  explicitly, and the result records `draws`, `successful` and `failures` so the
  censoring stays visible downstream.
  """
  if not isinstance(inference, PreparedInference):
    raise TypeError("inference must be a PreparedInference")
  if not isinstance(draws, Integral) or draws < 1:
    raise ValueError("draws must be a positive integer")
  if not isinstance(workers, Integral) or workers < 1:
    raise ValueError("workers must be a positive integer")
  if not 0.0 <= max_failure_fraction < 1.0:
    raise ValueError("max_failure_fraction must be in [0, 1)")
  if prior_variance is None:
    variance = np.full(inference.size, UNIFORM_VARIANCE)
  else:
    variance = np.broadcast_to(np.asarray(prior_variance, dtype=float), (inference.size,)).astype(float)
    if np.any(variance <= 0.0) or not np.all(np.isfinite(variance)):
      raise ValueError("prior_variance must be finite and positive")

  points = np.random.default_rng(seed).random((int(draws), inference.size))
  arguments = [(inference, point, variance) for point in points]
  if workers == 1:
    outcomes = [_gain_worker(argument) for argument in arguments]
  else:
    with ProcessPoolExecutor(max_workers=workers) as executor:
      outcomes = list(executor.map(_gain_worker, arguments))

  gains = np.array([value for value in outcomes if not isinstance(value, Exception)], dtype=float)
  errors = [value for value in outcomes if isinstance(value, Exception)]
  requested = len(outcomes)

  if errors:
    fraction = len(errors) / requested
    if not gains.size or fraction > max_failure_fraction:
      raise RuntimeError(
        f"{len(errors)} of {requested} design draws failed "
        f"({fraction:.1%} > max_failure_fraction={max_failure_fraction:.1%}); first failure shown as the cause"
      ) from errors[0]
    warnings.warn(
      f"{len(errors)} of {requested} design draws failed ({fraction:.1%}); "
      f"the reported gain is conditional on the {gains.size} that succeeded. "
      f"First failure: {type(errors[0]).__name__}: {errors[0]}",
      RuntimeWarning,
      stacklevel=2,
    )

  error = float(np.std(gains, ddof=1) / np.sqrt(gains.size)) if gains.size > 1 else float("inf")
  return DesignEvaluation(float(np.mean(gains)), error, requested, int(gains.size), len(errors))
