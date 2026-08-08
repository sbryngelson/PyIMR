"""Distributed stress quadrature convergence."""

import functools

import numpy as np
import pytest

import pyimr
from _validation_support import R0, REQ, T0, deviation

SECTION = "2c. Distributed stress quadrature"

pytestmark = pytest.mark.slow

_TIMES = np.linspace(0.0, 120e-6, 300)

_FLOOR = 1e-6


@functools.lru_cache(maxsize=None)
def _giesekus(points, quadrature, mobility=0.2):
  return pyimr.simulate(
    _TIMES,
    pyimr.SimulationConfig(R0=R0, Req=REQ, material=pyimr.Giesekus(0.1, 2 * T0, 0.4 * T0, mobility, points=points, quadrature=quadrature)),
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
  """Independent-rule cross-check: the two quadratures must agree once both are"""
  worst = deviation(_giesekus(240, "gauss"), _giesekus(3840, "trapezoid"))
  measured("gauss(240) vs trapezoid(3840)", f"max|dR|={worst:.2e}")
  assert worst < 5e-3


def test_former_trapezoid_default_was_not_converged(measured):
  """The trapezoid rule at the former default carries percent-level error,"""
  worst = _error(480, "trapezoid")
  measured("former default trapezoid(480)", f"max|dR|={worst:.2e}")
  assert worst > 1e-3, "if this ever drops, the default quadrature no longer has a justification"


@pytest.mark.parametrize("quadrature", ["gauss", "trapezoid"])
def test_the_distributed_stress_rate_is_the_time_derivative(quadrature, measured):
  """`radial != 1` asks for an explicit dS/dt. The trapezoid branch of that had no test.

  Checked as an identity rather than a line: along the trajectory the returned rate must
  equal `dS/dR * Rd + dS/dZ . dZ`, which a centered difference in the direction the state
  is actually moving reproduces without needing either partial separately.
  """
  from pyimr._solver import prepare
  from pyimr._stress import _distributed_stress

  material = pyimr.Giesekus(0.1, 2 * T0, 0.4 * T0, 0.2, points=64, quadrature=quadrature)
  config = pyimr.SimulationConfig(R0=R0, Req=REQ, material=material, dynamics="keller-miksis")
  problem = prepare(config)
  p = problem.parameters
  state = np.asarray(problem.initial_state, dtype=float)
  R, Rd = 0.75, -0.4  # mid-collapse: both the radius and the memory are moving
  Z = state[problem.layout.stress]

  stress, rate, dZ, _ = _distributed_stress(material, problem.distributed_stress, p, R, Rd, Z, True)

  def integral_at(step):
    moved, _, _, _ = _distributed_stress(material, problem.distributed_stress, p, R + step * Rd, Rd, Z + step * np.asarray(dZ), True)
    return moved

  step = 1e-7
  difference = (integral_at(step) - integral_at(-step)) / (2.0 * step)
  error = abs(rate - difference) / abs(difference)
  measured(f"{quadrature} stress rate", f"rel={error:.2e}")
  assert error < 1e-6, f"{quadrature}: rate {rate:.8e} vs centered difference {difference:.8e}"
