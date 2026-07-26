"""Distributed stress quadrature convergence.

The distributed constitutive equations are pointwise ODEs -- no spatial
derivatives -- so the only spatial approximation is the quadrature for
I = int_R^inf 2 (t_rr - t_hh) / r dr. Mapping to the Lagrangian coordinate
turns it into a fixed-weight sum, which makes Gauss-Legendre available and makes
the integral's time derivative exact instead of a discrete difference.

Marked slow: the reference solves at 1920 and 3840 points.

Numerical content unchanged from `run_validation.py`; see issue #32.
"""

import functools

import numpy as np
import pytest

import imr_fast
from _validation_support import R0, REQ, T0, deviation

SECTION = "2c. Distributed stress quadrature"

pytestmark = pytest.mark.slow

_TIMES = np.linspace(0.0, 120e-6, 300)

# Past ~1e-7 the residual is the ODE solver's own tolerance floor and does not
# decrease monotonically. Assertions below stay above it -- requiring monotone
# improvement at the floor is a flaky test, not a stronger one.
_FLOOR = 1e-6


@functools.lru_cache(maxsize=None)
def _giesekus(points, quadrature, mobility=0.2):
  return imr_fast.simulate(
    _TIMES,
    imr_fast.SimulationConfig(
      R0=R0, Req=REQ, material=imr_fast.Giesekus(0.1, 2 * T0, 0.4 * T0, mobility, points=points, quadrature=quadrature)
    ),
  ).radius_ratio


def _error(points, quadrature="gauss"):
  return deviation(_giesekus(points, quadrature), _giesekus(1920, "gauss"))


@pytest.mark.parametrize("points", (60, 120, 240))
def test_gauss_error_recorded(points, measured):
  """Reported for the convergence table; the assertions are below."""
  measured(f"gauss({points}) vs gauss(1920)", f"max|dR|={_error(points):.2e}")


def test_gauss_doubling_buys_orders_of_magnitude(measured):
  """Spectral convergence: one doubling must buy orders, well above the floor."""
  coarse, fine = _error(60), _error(120)
  measured("gauss 60 -> 120", f"{coarse / fine:.0f}x (floor {_FLOOR:.0e})")
  assert coarse > _FLOOR, "60 points already at the solver floor; the ratio below would be meaningless"
  assert fine < coarse / 100.0


def test_default_and_half_default_are_converged(measured):
  """At and beyond the default, only require having reached the floor."""
  measured("gauss 120 and 240", f"{_error(120):.2e}, {_error(240):.2e}")
  assert _error(120) < 1e-5 and _error(240) < 1e-5


def test_gauss_agrees_with_trapezoid(measured):
  """Independent-rule cross-check: the two quadratures must agree once both are
  resolved. This is the only independent check the distributed models have --
  IMRv2 cannot run Giesekus or PTT at all."""
  worst = deviation(_giesekus(240, "gauss"), _giesekus(3840, "trapezoid"))
  measured("gauss(240) vs trapezoid(3840)", f"max|dR|={worst:.2e}")
  assert worst < 5e-3


def test_former_trapezoid_default_was_not_converged(measured):
  """The trapezoid rule at the former default carries percent-level error,
  which the Oldroyd-B reduction limit alone did not reveal. This is why the
  default moved in 0.3.0."""
  worst = _error(480, "trapezoid")
  measured("former default trapezoid(480)", f"max|dR|={worst:.2e}")
  assert worst > 1e-3, "if this ever drops, the 0.3.0 default change no longer has a justification"
