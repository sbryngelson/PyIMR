"""Thermal PDE discretization: finite difference vs Chebyshev collocation.

Unlike the distributed stress, the thermal fields genuinely carry spatial
derivatives, so a spectral discretization buys something here. thermal_fd is
cleanly second order (verified standalone in validate_thermal_fd.py); the
Chebyshev operators are spectral.

Marked slow: the convergence study solves at Nt up to 300 on an 8001-point
output grid at tightened integrator tolerance.

Numerical content unchanged from `run_validation.py`; see issue #32.
"""

import functools

import numpy as np
import pytest

import imr_fast
import thermal_fd
import thermal_spectral
from _validation_support import NHKV, R0, REQ

SECTION = "2d. Thermal PDE discretization"

# Measure collapse depth and timing on a grid fine enough to resolve the
# minimum. A coarse output grid samples the sharp collapse at slightly
# different phases as the solution shifts, which shows up as an apparent error
# floor around 1e-4 that has nothing to do with the discretization. That
# artifact is what an earlier version of this check mistook for a spectral
# convergence plateau.
_FINE_TIMES = np.linspace(0.0, 60e-6, 8001)

# Resolving the output grid is necessary but not sufficient. At the default
# rtol=1e-8/atol=1e-10 the collapse minimum is only reproducible to ~5e-07
# regardless of Nt, so a spectral run at Nt=100 differs from one at Nt=200 by
# the integrator's noise rather than by any discretization error. Measured:
#
#   Nt        100        150        200        300        400
#   R/R0  ...582427  ...579289  ...577297  ...580435  ...580495   (default tol)
#
# -- non-monotone, spread 5.1e-07, which is exactly the size of the "error"
# being attributed to Nt=100. The verdict then depends on which reference is
# picked (3.5x against Nt=200, 7.7x against Nt=400) and on the SciPy version.
#
# Tightening the time tolerance pushes the temporal error below the spatial
# one, which is the only regime in which a spatial convergence rate means
# anything. The Nt >= 200 values then agree to 1.4e-09.
_CONVERGENCE_RTOL, _CONVERGENCE_ATOL = 1e-11, 1e-13


def _cosine(y):
  return np.cos(2.0 * y)


def _laplacian(y):
  return np.where(y == 0, -12.0, -4.0 * np.cos(2.0 * y) - 4.0 * np.sin(2.0 * y) / np.where(y == 0, 1, y))


def test_spectral_laplacian_operator(measured):
  """Operators first, isolated from the ODE."""
  nodes = thermal_spectral.nodes(17, 0)
  spectral = float(np.max(np.abs(thermal_spectral.chebyshev_diff_mat(17, 2, 0) @ _cosine(nodes) - _laplacian(nodes))))
  uniform = np.linspace(0.0, 1.0, 17)
  finite = float(
    np.max(np.abs((thermal_fd.finite_diff_mat(17, 2, 0) @ _cosine(uniform))[:-1] - _laplacian(uniform)[:-1]))
  )
  measured("spherical Laplacian N=17", f"spectral={spectral:.2e}  finite difference={finite:.2e}")
  assert spectral < 1e-9


@functools.lru_cache(maxsize=None)
def _collapse_metrics(nt, backend):
  radius = imr_fast.simulate(
    _FINE_TIMES,
    imr_fast.SimulationConfig(
      R0=R0, Req=REQ, material=NHKV, bubtherm=1, Nt=nt, thermal=backend, rtol=_CONVERGENCE_RTOL, atol=_CONVERGENCE_ATOL
    ),
  ).radius_ratio
  index = int(radius.argmin())
  before, at, after = radius[index - 1], radius[index], radius[index + 1]
  shift = 0.5 * (before - after) / (before - 2.0 * at + after)
  minimum = at - 0.25 * (before - after) * shift
  time = _FINE_TIMES[index] + 0.5 * shift * (_FINE_TIMES[index + 1] - _FINE_TIMES[index - 1])
  return minimum, time


@pytest.mark.slow
def test_spectral_beats_fine_finite_difference_on_depth(measured):
  """Spectral at Nt=25 must beat finite difference at Nt=200."""
  reference, _ = _collapse_metrics(200, "spectral")
  spectral, _ = _collapse_metrics(25, "spectral")
  finite, _ = _collapse_metrics(200, "fd")
  measured("min radius", f"spectral(25)={abs(spectral - reference):.2e}  fd(200)={abs(finite - reference):.2e}")
  assert abs(spectral - reference) < abs(finite - reference)


@pytest.mark.slow
def test_spectral_beats_fine_finite_difference_on_timing(measured):
  _, reference = _collapse_metrics(200, "spectral")
  _, spectral = _collapse_metrics(25, "spectral")
  _, finite = _collapse_metrics(200, "fd")
  measured(
    "collapse time",
    f"spectral(25)={abs(spectral - reference) * 1e9:.4f}ns  fd(200)={abs(finite - reference) * 1e9:.4f}ns",
  )
  assert abs(spectral - reference) < abs(finite - reference)


@pytest.mark.slow
def test_spectral_keeps_converging(measured):
  """Spectral must keep converging -- there is no floor. Assert the overall
  improvement rather than a strict chain: near the reference's own accuracy the
  individual steps are not monotone, and asserting that they are is a flaky test
  rather than a stronger one."""
  reference, _ = _collapse_metrics(200, "spectral")
  coarse, _ = _collapse_metrics(25, "spectral")
  fine, _ = _collapse_metrics(100, "spectral")
  measured("spectral 25 -> 100", f"{abs(coarse - reference) / abs(fine - reference):.1f}x (no floor)")
  assert abs(fine - reference) < abs(coarse - reference) / 4.0


@pytest.mark.slow
def test_convergence_is_measured_above_the_reference_floor(measured):
  """Report the reference's own uncertainty alongside the ratio. It is what
  decides whether that ratio means anything, and leaving it out is what let this
  check read as a solver result for as long as it did."""
  reference, _ = _collapse_metrics(200, "spectral")
  fine, _ = _collapse_metrics(100, "spectral")
  floor = abs(_collapse_metrics(300, "spectral")[0] - reference)
  measured("reference floor |Nt=300 - Nt=200|", f"{floor:.2e}  vs measured error at Nt=100 {abs(fine - reference):.2e}")
  assert abs(fine - reference) > floor
