"""Ensemble and variational state estimation on top of the prepared flow."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
  "EnsembleAnalysis",
  "VariationalAnalysis",
  "enkf_analysis",
  "four_dvar",
  "kalman_analysis",
  "variational_cost",
]

@dataclass(frozen=True, slots=True)
class EnsembleAnalysis:
  """The updated ensemble and the moments it implies."""

  members: np.ndarray
  mean: np.ndarray
  covariance: np.ndarray

def _observation_matrix(operator, width):
  matrix = np.asarray(operator, dtype=float)
  if matrix.ndim != 2 or matrix.shape[1] != width:
    raise ValueError(f"operator must be 2-D with {width} columns; got shape {matrix.shape}")
  return matrix

def _noise_matrix(noise, count):
  values = np.asarray(noise, dtype=float)
  if values.ndim == 0: return np.eye(count) * float(values) ** 2
  if values.ndim == 1:
    if values.size != count: raise ValueError(f"noise must supply {count} deviations; got {values.size}")
    return np.diag(values**2)
  if values.shape != (count, count): raise ValueError(f"noise covariance must be ({count}, {count}); got {values.shape}")
  return values

def kalman_analysis(mean, covariance, observation, operator, noise):
  """The exact linear-Gaussian update, for checking the ensemble one against."""
  mean = np.asarray(mean, dtype=float)
  covariance = np.asarray(covariance, dtype=float)
  H = _observation_matrix(operator, mean.size)
  observation = np.asarray(observation, dtype=float)
  R = _noise_matrix(noise, observation.size)

  innovation_covariance = H @ covariance @ H.T + R
  gain = np.linalg.solve(innovation_covariance.T, (covariance @ H.T).T).T
  updated_mean = mean + gain @ (observation - H @ mean)
  updated_covariance = covariance - gain @ H @ covariance
  return updated_mean, updated_covariance

def enkf_analysis(members, observation, operator, noise, *, rng=None, perturb=True):
  """One stochastic EnKF analysis step.

  `members` is `(member, state)`. Observations are perturbed by default, which is what
  makes the updated ensemble carry the analysis covariance rather than an ensemble that
  is too tight -- set `perturb=False` only to inspect the deterministic shift.
  """
  ensemble = np.asarray(members, dtype=float)
  if ensemble.ndim != 2: raise ValueError(f"members must be a 2-D array of states; got shape {ensemble.shape}")
  count = ensemble.shape[0]
  if count < 2: raise ValueError("an ensemble analysis needs at least two members")
  H = _observation_matrix(operator, ensemble.shape[1])
  observation = np.asarray(observation, dtype=float)
  R = _noise_matrix(noise, observation.size)

  mean = ensemble.mean(axis=0)
  deviations = ensemble - mean
  predicted = deviations @ H.T
  # 1/(n-1) on both, so the gain is a ratio of unbiased estimates
  cross = deviations.T @ predicted / (count - 1)
  innovation_covariance = predicted.T @ predicted / (count - 1) + R
  gain = np.linalg.solve(innovation_covariance.T, cross.T).T

  targets = np.broadcast_to(observation, (count, observation.size))
  if perturb:
    generator = np.random.default_rng() if rng is None else rng
    targets = targets + generator.multivariate_normal(np.zeros(observation.size), R, size=count)
  updated = ensemble + (targets - ensemble @ H.T) @ gain.T
  centred = updated - updated.mean(axis=0)
  return EnsembleAnalysis(updated, updated.mean(axis=0), centred.T @ centred / (count - 1))

@dataclass(frozen=True, slots=True)
class VariationalAnalysis:
  """The minimising initial state and what the optimiser reported getting there."""

  state: np.ndarray
  cost: np.ndarray
  gradient_norm: float
  iterations: int
  success: bool
  message: str

def _observation_stack(observations, times, width):
  values = np.asarray(observations, dtype=float)
  if values.ndim != 2 or values.shape[0] != times:
    raise ValueError(f"observations must be ({times}, m); got shape {values.shape}")
  if values.shape[1] != width:
    raise ValueError(f"observations must have {width} columns to match the operator; got {values.shape[1]}")
  return values

def variational_cost(problem, times, state, observations, operator, noise, background, background_precision):
  """`(cost, gradient)` of the strong-constraint 4D-Var objective at `state`.

  The gradient is exact: `sum_k M_k^T H^T R^-1 r_k`, with `M_k` the tangent linear
  operator this package computes rather than an ensemble approximation to it. That is
  the whole reason true 4D-Var is reachable here and En4D-Var is not required.
  """
  grid = np.asarray(times, dtype=float)
  states, jacobian = problem.state_tangents(grid, state)
  H = _observation_matrix(operator, states.shape[1])
  targets = _observation_stack(observations, grid.size, H.shape[0])
  R = _noise_matrix(noise, H.shape[0])

  residual = states @ H.T - targets
  seen = np.asarray(~np.isnan(targets).any(axis=1))  # rows of NaN mean 'not observed here'
  residual = np.where(seen[:, None], residual, 0.0)
  weighted = np.linalg.solve(R, residual.T).T

  departure = np.asarray(state, dtype=float) - np.asarray(background, dtype=float)
  precision = np.asarray(background_precision, dtype=float)
  cost = 0.5 * float(np.sum(residual * weighted)) + 0.5 * float(departure @ precision @ departure)
  # jacobian[k][i][j] is d y_k[i] / d y0[j], so `M^T v` contracts the OUTPUT index i and
  # leaves j. Contracting j instead transposes the operator and is right only when the
  # Jacobian is symmetric, which it is not.
  gradient = precision @ departure + np.einsum("kij,ki->j", jacobian, weighted @ H)
  return cost, gradient

def four_dvar(problem, times, observations, operator, noise, background, background_precision, *, maximum_iterations=100):
  """Minimise the 4D-Var cost over the initial state, with exact gradients."""
  from scipy.optimize import minimize

  start = np.asarray(background, dtype=float)

  def objective(state):
    cost, gradient = variational_cost(problem, times, state, observations, operator, noise, background, background_precision)
    return cost, gradient

  outcome = minimize(objective, start, jac=True, method="L-BFGS-B", options={"maxiter": int(maximum_iterations)})
  return VariationalAnalysis(
    state=np.asarray(outcome.x, dtype=float),
    cost=np.asarray(outcome.fun, dtype=float),
    gradient_norm=float(np.linalg.norm(outcome.jac)),
    iterations=int(outcome.nit),
    success=bool(outcome.success),
    message=str(outcome.message),
  )
