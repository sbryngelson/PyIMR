"""Chebyshev collocation operators for the IMR thermal PDEs."""

from __future__ import annotations

import numpy as np

__all__ = ["chebyshev_diff_mat", "nodes"]

def _chebyshev(count):
  if count < 2: raise ValueError("count must be at least 2")
  last = count - 1
  x = np.cos(np.pi * np.arange(count) / last)
  scale = np.ones(count)
  scale[0] = 2.0
  scale[last] = 2.0
  scale = scale * (-1.0) ** np.arange(count)
  offset = x[:, None] - x[None, :]
  matrix = np.outer(scale, 1.0 / scale) / (offset + np.eye(count))
  matrix -= np.diag(matrix.sum(axis=1))
  return x, matrix

def nodes(count, tm_check=0):
  """Collocation nodes matching :func:`chebyshev_diff_mat`."""
  if tm_check == 0:
    full, _ = _chebyshev(2 * count - 1)
    return np.ascontiguousarray(full[count - 1 :: -1])
  return _chebyshev(count)[0]

def chebyshev_diff_mat(count, order, tm_check=0):
  """Spectral analogue of ``thermal_fd.finite_diff_mat``."""
  if order not in (1, 2): raise ValueError("order must be 1 or 2")
  if tm_check != 0:
    _, first = _chebyshev(count)
    return first if order == 1 else first @ first
  full_count = 2 * count - 1
  full_x, full_first = _chebyshev(full_count)
  keep = np.arange(count - 1, -1, -1)
  mirror = np.arange(count, full_count)

  def fold(matrix):
    folded = matrix[np.ix_(keep, keep)].copy()
    folded[:, 1:] += matrix[np.ix_(keep, mirror)]
    return folded

  if order == 1:
    operator = fold(full_first)
    operator[0, :] = 0.0  # regularity: f'(0) = 0 exactly
    return operator
  full_second = full_first @ full_first
  laplacian = full_second.copy()
  centre = int(np.argmin(np.abs(full_x)))
  away = np.array([j for j in range(full_count) if j != centre])
  laplacian[away, :] += (2.0 / full_x[away])[:, None] * full_first[away, :]
  laplacian[centre, :] = 3.0 * full_second[centre, :]
  return fold(laplacian)
