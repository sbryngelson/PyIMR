"""Chebyshev collocation operators for the IMR thermal PDEs.

Drop-in spectral alternative to :mod:`thermal_fd`. Same two grids, same
meaning for ``order`` and ``tm_check``, but the returned matrices are dense
and spectrally accurate rather than tridiagonal and second order.

Unlike the distributed stress -- whose Lagrangian form has no spatial
derivatives at all -- the thermal fields genuinely need a spatial
discretisation, so collocation buys something real here.

Two grids, two treatments:

  tm_check=0  interior (bubble gas) grid, y in [0,1]. The ``order=2`` matrix
              is the spherically symmetric Laplacian
              ``(1/y^2) d/dy (y^2 df/dy) = f'' + (2/y) f'``, not ``d^2/dy^2``.
              Regularity at ``y = 0`` is imposed by building the operator on
              the even extension to ``[-1,1]``: a function smooth at the
              origin of a sphere is even in ``y``, so the even-symmetric
              Chebyshev grid resolves it without ever evaluating ``2/y`` at
              the pole. This is the standard treatment (Trefethen, *Spectral
              Methods in MATLAB*, ch. 13) and avoids the L'Hopital special
              case ``thermal_fd`` needs at row 0.

  tm_check=1  exterior (medium) grid, xi in [1,-1]. Plain derivatives; the
              physical stretching and spherical factors are applied
              separately in the RHS, exactly as for the finite-difference
              operators. Chebyshev nodes already live on ``[-1,1]``, so this
              grid needs no mapping at all.

Node positions are **not** uniform, so a caller cannot mix these matrices with
uniform-grid assumptions. Use :func:`nodes` to get the matching grid, and take
boundary gradients from the relevant row of the ``order=1`` matrix rather than
from a hardcoded one-sided stencil.
"""

from __future__ import annotations

import numpy as np

__all__ = ["chebyshev_diff_mat", "nodes"]

def _chebyshev(count):
  # Chebyshev-Gauss-Lobatto nodes on [1,-1] and the differentiation matrix.
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
  """Collocation nodes matching :func:`chebyshev_diff_mat`.

  ``tm_check=0`` returns ``y`` increasing on ``[0,1]`` (bubble centre first,
  wall last), matching ``np.linspace(0, 1, count)``. ``tm_check=1`` returns
  ``xi`` decreasing on ``[1,-1]``, matching the finite-difference convention.
  """
  if tm_check == 0:
    # even extension: use the upper half of a (2*count - 1) point grid
    full, _ = _chebyshev(2 * count - 1)
    return np.ascontiguousarray(full[count - 1 :: -1])
  return _chebyshev(count)[0]

def chebyshev_diff_mat(count, order, tm_check=0):
  """Spectral analogue of ``thermal_fd.finite_diff_mat``.

  ``order=1`` is ``d/dy``. ``order=2`` on the interior grid is the spherical
  Laplacian; on the exterior grid it is ``d^2/dxi^2``.
  """
  if order not in (1, 2): raise ValueError("order must be 1 or 2")
  if tm_check != 0:
    _, first = _chebyshev(count)
    return first if order == 1 else first @ first
  # Interior grid. Work on the even extension y in [-1,1], then restrict.
  full_count = 2 * count - 1
  full_x, full_first = _chebyshev(full_count)
  # x_j = cos(pi j / (2N)) so x_{2N-j} = -x_j: column 2N-j is the mirror of
  # column j. keep = [N, N-1, ..., 0] gives y = 0 .. 1 increasing, and the
  # mirrors of keep[1:] are exactly columns count .. full_count-1, in order.
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
  # spherical Laplacian f'' + (2/y) f'; the 2/y term is removable at y = 0,
  # where the limit is 3 f''(0)
  laplacian = full_second.copy()
  centre = int(np.argmin(np.abs(full_x)))
  away = np.array([j for j in range(full_count) if j != centre])
  laplacian[away, :] += (2.0 / full_x[away])[:, None] * full_first[away, :]
  laplacian[centre, :] = 3.0 * full_second[centre, :]
  return fold(laplacian)
