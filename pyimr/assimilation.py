"""Ensemble and variational state estimation on top of the prepared flow."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
  "EnsembleAnalysis",
  "VariationalAnalysis",
  "enkf_analysis",
  "ensemble_smoother",
  "ensemble_update",
  "four_dvar",
  "ienks",
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

def ensemble_update(members, predictions, observation, noise, *, rng=None, perturb=True):
  """One stochastic ensemble analysis from *predicted observations*, not a linear operator.

  This is the form nonlinear assimilation needs: `predictions[i]` is whatever the
  observation operator produces for member `i`, however it was computed. The gain comes
  from the sample cross-covariance between states and predictions, so the operator never
  has to be written down or linearised.
  """
  ensemble = np.asarray(members, dtype=float)
  predicted = np.asarray(predictions, dtype=float)
  if ensemble.ndim != 2: raise ValueError(f"members must be a 2-D array of states; got shape {ensemble.shape}")
  if predicted.ndim != 2 or predicted.shape[0] != ensemble.shape[0]:
    raise ValueError(f"predictions must be (member, observation); got shape {predicted.shape}")
  count = ensemble.shape[0]
  if count < 2: raise ValueError("an ensemble analysis needs at least two members")
  observation = np.asarray(observation, dtype=float)
  if predicted.shape[1] != observation.size:
    raise ValueError(f"predictions have {predicted.shape[1]} columns but the observation has {observation.size}")
  R = _noise_matrix(noise, observation.size)

  deviations = ensemble - ensemble.mean(axis=0)
  spread = predicted - predicted.mean(axis=0)
  cross = deviations.T @ spread / (count - 1)
  innovation_covariance = spread.T @ spread / (count - 1) + R
  gain = np.linalg.solve(innovation_covariance.T, cross.T).T

  targets = np.broadcast_to(observation, (count, observation.size))
  if perturb:
    generator = np.random.default_rng() if rng is None else rng
    targets = targets + generator.multivariate_normal(np.zeros(observation.size), R, size=count)
  updated = ensemble + (targets - predicted) @ gain.T
  centred = updated - updated.mean(axis=0)
  return EnsembleAnalysis(updated, updated.mean(axis=0), centred.T @ centred / (count - 1))

def ensemble_smoother(problem, times, members, observations, operator, noise, *, rng=None):
  """Analyse the INITIAL state against a whole window at once, the ensemble way.

  The counterpart to `four_dvar`: same observations, same window, same quantity
  estimated -- but the relationship between the initial state and the observations is
  taken from ensemble statistics rather than from the tangent operator.
  """
  ensemble = np.asarray(members, dtype=float)
  # a diverged tail member is a fact about the draw, not an error: drop it and analyse
  # with the survivors, which is what an ensemble method does anyway
  trajectories, ok = problem.solve_ensemble(ensemble, times, drop_failures=True)
  ensemble = ensemble[ok]
  if ensemble.shape[0] < 2: raise ValueError(f"only {ensemble.shape[0]} of {ok.size} members integrated; cannot form an analysis")
  H = _observation_matrix(operator, trajectories.shape[2])
  targets = _observation_stack(observations, np.asarray(times).size, H.shape[0])
  seen = np.asarray(~np.isnan(targets).any(axis=1))
  predicted = np.einsum("mtj,ij->mti", trajectories, H)[:, seen].reshape(ensemble.shape[0], -1)
  flat = targets[seen].reshape(-1)
  spread = _noise_matrix(noise, H.shape[0])
  block = np.kron(np.eye(int(seen.sum())), spread)
  return ensemble_update(ensemble, predicted, flat, block, rng=rng)

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
    # A line search will propose states the solver cannot integrate. That is a fact about
    # the search, not an error: report the point as infeasible and let the optimiser back
    # off, rather than letting one bad trial state end the minimisation.
    try:
      return variational_cost(problem, times, state, observations, operator, noise, background, background_precision)
    except Exception:  # noqa: BLE001 - any failure to integrate means "do not go here"
      return np.inf, np.zeros_like(np.asarray(state, dtype=float))

  outcome = minimize(objective, start, jac=True, method="L-BFGS-B", options={"maxiter": int(maximum_iterations)})
  return VariationalAnalysis(
    state=np.asarray(outcome.x, dtype=float),
    cost=np.asarray(outcome.fun, dtype=float),
    gradient_norm=float(np.linalg.norm(outcome.jac)),
    iterations=int(outcome.nit),
    success=bool(outcome.success),
    message=str(outcome.message),
  )

def _window_predictions(problem, times, members, operator, seen):
  """Predicted observations for each member, flattened over the observed times."""
  trajectories, ok = problem.solve_ensemble(members, times, drop_failures=True)
  H = _observation_matrix(operator, trajectories.shape[2])
  predicted = np.einsum("mtj,ij->mti", trajectories, H)[:, seen].reshape(trajectories.shape[0], -1)
  return predicted, ok

def ienks(problem, times, members, observations, operator, noise, *, iterations=12, epsilon=1e-3, tolerance=1e-10):
  """Iterative ensemble Kalman smoother over the initial state, bundle variant.

  The one-shot `ensemble_smoother` linearises once about the prior mean, which is what
  costs it accuracy across a nonlinear window. This re-linearises about the current
  estimate each iteration -- Gauss-Newton in the ensemble subspace -- so it closes on the
  same minimum `four_dvar` finds, without ever forming the tangent operator.

  `epsilon` scales the bundle used to estimate the ensemble-space sensitivity. Too large
  and the finite difference is nonlinear; too small and it is roundoff.
  """
  ensemble = np.asarray(members, dtype=float)
  if ensemble.ndim != 2: raise ValueError(f"members must be a 2-D array of states; got shape {ensemble.shape}")
  count = ensemble.shape[0]
  if count < 2: raise ValueError("an ensemble smoother needs at least two members")

  grid = np.asarray(times, dtype=float)
  H = _observation_matrix(operator, ensemble.shape[1])
  targets = _observation_stack(observations, grid.size, H.shape[0])
  seen = np.asarray(~np.isnan(targets).any(axis=1))
  flat = targets[seen].reshape(-1)
  block = np.kron(np.eye(int(seen.sum())), _noise_matrix(noise, H.shape[0]))

  anchor = ensemble.mean(axis=0)
  anomalies = (ensemble - anchor) / np.sqrt(count - 1.0)
  coefficients = np.zeros(count)

  for _ in range(int(iterations)):
    centre = anchor + anomalies.T @ coefficients
    bundle = centre + epsilon * anomalies
    predicted, ok = _window_predictions(problem, grid, bundle, operator, seen)
    if ok.sum() < 2: raise ValueError(f"only {int(ok.sum())} of {count} bundle members integrated")
    sensitivity = (predicted - predicted.mean(axis=0)) / epsilon
    departure = predicted.mean(axis=0) - flat

    weighted = np.linalg.solve(block, sensitivity.T).T
    hessian = np.eye(ok.sum()) + sensitivity @ weighted.T
    gradient = coefficients[ok] + weighted @ departure
    step = np.zeros(count)
    step[ok] = -np.linalg.solve(hessian, gradient)
    coefficients = coefficients + step
    if np.linalg.norm(step) < tolerance: break

  centre = anchor + anomalies.T @ coefficients
  bundle = centre + epsilon * anomalies
  predicted, ok = _window_predictions(problem, grid, bundle, operator, seen)
  sensitivity = (predicted - predicted.mean(axis=0)) / epsilon
  weighted = np.linalg.solve(block, sensitivity.T).T
  transform = np.linalg.inv(np.eye(ok.sum()) + sensitivity @ weighted.T)
  eigenvalues, vectors = np.linalg.eigh(transform)
  root = vectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0.0))) @ vectors.T
  updated = centre + np.sqrt(count - 1.0) * (root @ anomalies[ok])
  centred = updated - updated.mean(axis=0)
  return EnsembleAnalysis(updated, centre, centred.T @ centred / max(updated.shape[0] - 1, 1))
