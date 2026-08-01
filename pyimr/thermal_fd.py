"""Finite-difference operators for the IMRv2 thermal-PDE fields, ported from"""

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
