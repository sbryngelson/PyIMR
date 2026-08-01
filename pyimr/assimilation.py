"""Ensemble and variational state estimation on top of the prepared flow."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
  "EnsembleAnalysis",
  "enkf_analysis",
  "kalman_analysis",
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
