"""Thermal PDE discretization: finite difference vs Chebyshev collocation."""

import functools

import numpy as np
import pytest

import imr_fast
from imr_fast import thermal_fd
from imr_fast import thermal_spectral
from _validation_support import NHKV, R0, REQ

SECTION = "2d. Thermal PDE discretization"

_FINE_TIMES = np.linspace(0.0, 60e-6, 8001)

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
  finite = float(np.max(np.abs((thermal_fd.finite_diff_mat(17, 2, 0) @ _cosine(uniform))[:-1] - _laplacian(uniform)[:-1])))
  measured("spherical Laplacian N=17", f"spectral={spectral:.2e}  finite difference={finite:.2e}")
  assert spectral < 1e-9


@functools.lru_cache(maxsize=None)
def _collapse_metrics(nt, backend):
  radius = imr_fast.simulate(
    _FINE_TIMES,
    imr_fast.SimulationConfig(R0=R0, Req=REQ, material=NHKV, bubtherm=1, Nt=nt, thermal=backend, rtol=_CONVERGENCE_RTOL, atol=_CONVERGENCE_ATOL),
  ).radius_ratio
  index = int(radius.argmin())
  before, at, after = radius[index - 1], radius[index], radius[index + 1]
  shift = 0.5 * (before - after) / (before - 2.0 * at + after)
  minimum = at - 0.25 * (before - after) * shift
  time = _FINE_TIMES[index] + 0.5 * shift * (_FINE_TIMES[index + 1] - _FINE_TIMES[index - 1])
  return minimum, time


_REFERENCE_NT = 200
_SPECTRAL_NT = 25
_FINITE_NT = 100
_FLOOR_NT = 150


@pytest.mark.slow
def test_spectral_beats_fine_finite_difference_on_depth(measured):
  """Spectral at Nt=25 must beat finite difference at four times the resolution."""
  reference, _ = _collapse_metrics(_REFERENCE_NT, "spectral")
  spectral, _ = _collapse_metrics(_SPECTRAL_NT, "spectral")
  finite, _ = _collapse_metrics(_FINITE_NT, "fd")
  measured("min radius", f"spectral({_SPECTRAL_NT})={abs(spectral - reference):.2e}  fd({_FINITE_NT})={abs(finite - reference):.2e}")
  assert abs(spectral - reference) < abs(finite - reference)


@pytest.mark.slow
def test_spectral_beats_fine_finite_difference_on_timing(measured):
  _, reference = _collapse_metrics(_REFERENCE_NT, "spectral")
  _, spectral = _collapse_metrics(_SPECTRAL_NT, "spectral")
  _, finite = _collapse_metrics(_FINITE_NT, "fd")
  measured("collapse time", f"spectral({_SPECTRAL_NT})={abs(spectral - reference) * 1e9:.4f}ns  fd({_FINITE_NT})={abs(finite - reference) * 1e9:.4f}ns")
  assert abs(spectral - reference) < abs(finite - reference)


@pytest.mark.slow
def test_spectral_keeps_converging(measured):
  """Spectral must keep converging -- there is no floor. The overall improvement is"""
  reference, _ = _collapse_metrics(_REFERENCE_NT, "spectral")
  coarse, _ = _collapse_metrics(_SPECTRAL_NT, "spectral")
  fine, _ = _collapse_metrics(_FINITE_NT, "spectral")
  measured(f"spectral {_SPECTRAL_NT} -> {_FINITE_NT}", f"{abs(coarse - reference) / abs(fine - reference):.1f}x (no floor)")
  assert abs(fine - reference) < abs(coarse - reference) / 4.0


@pytest.mark.slow
def test_convergence_is_measured_above_the_reference_floor(measured):
  """The ratio above means nothing unless it is measured above the reference's own"""
  reference, _ = _collapse_metrics(_REFERENCE_NT, "spectral")
  fine, _ = _collapse_metrics(_FINITE_NT, "spectral")
  floor = abs(_collapse_metrics(_FLOOR_NT, "spectral")[0] - reference)
  measured(f"reference floor |Nt={_FLOOR_NT} - Nt={_REFERENCE_NT}|", f"{floor:.2e}  vs error at Nt={_FINITE_NT} {abs(fine - reference):.2e}")
  assert abs(fine - reference) > floor
