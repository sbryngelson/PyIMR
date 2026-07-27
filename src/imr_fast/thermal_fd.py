"""
Finite-difference operators for the IMRv2 thermal-PDE fields, ported from
IMRv2/src/f_imr_fd.m :: f_finite_diff_mat.

Two grids, two different operators:

  tm_check=0  interior (bubble gas) grid, y in [0,1].
              The order=2 ("second derivative") matrix is NOT d^2/dy^2 -- it is
              the spherically-symmetric Laplacian
                  (1/y^2) d/dy( y^2 df/dy ) = f'' + (2/y) f'
              folded directly into one stencil via off-diagonal coefficients
              (1 +- dy/y_i). Row 0 (the bubble center, y=0) uses the standard
              L'Hopital/mirror-symmetry treatment for the removable 1/y
              singularity: Lap f(0) = 3 f''(0), discretized as
                  6*(f_1 - f_0) / dy^2
              which is what the -6, 6 coefficients encode.
              Row 0 of the order=1 matrix is left as all-zero: regularity at
              the bubble center means f'(0)=0 by construction, so no
              approximation is needed there.
              The last row of the order=1 matrix uses a one-sided 2nd-order
              backward stencil (the wall gradient, needed for the pressure
              flux term); the last row of the order=2 matrix is left as zero
              because the physics RHS always overwrites that field component
              directly rather than reading its raw Laplacian.

  tm_check=1  exterior (medium/liquid) grid, xi in [1,-1]. PLAIN (non-spherical)
              central-difference stencils. The physical stretching (the Lt
              parameter) and spherical geometry factors are applied
              separately, multiplicatively, in the RHS -- NOT baked into this
              matrix. Both end rows are left as zero for the same reason as
              above (the physics RHS clamps Tm at both ends directly).

Validated standalone (see validate_thermal_fd.py) before being wired into any
ODE state -- this file has no simulate()-side dependencies.
"""

import numpy as np


def finite_diff_mat(nodes, order, tm_check=0):
  N = nodes - 1
  if tm_check == 0:
    deltaY = 1.0 / N
    y = np.arange(nodes) * deltaY  # y[0]=0 ... y[N]=1
  else:
    deltaY = -2.0 / N
    y = 1.0 + np.arange(nodes) * deltaY  # xi[0]=1 ... xi[N]=-1

  dmatrix = np.zeros((nodes, nodes))

  if order == 1:
    for i in range(1, N):
      dmatrix[i, i + 1] = 0.5
      dmatrix[i, i - 1] = -0.5
    if tm_check == 0:
      dmatrix[N, N] = 1.5
      dmatrix[N, N - 1] = -2.0
      dmatrix[N, N - 2] = 0.5
    dmatrix /= deltaY

  elif order == 2:
    for i in range(1, N):
      if tm_check == 0:
        dmatrix[i, i + 1] = 1.0 + deltaY / y[i]
        dmatrix[i, i] = -2.0
        dmatrix[i, i - 1] = 1.0 - deltaY / y[i]
      else:
        dmatrix[i, i + 1] = 1.0
        dmatrix[i, i] = -2.0
        dmatrix[i, i - 1] = 1.0
    if tm_check == 0:
      dmatrix[0, 0] = -6.0
      dmatrix[0, 1] = 6.0
    dmatrix /= deltaY**2

  else:
    raise ValueError("order must be 1 or 2")

  return dmatrix
