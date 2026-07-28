"""Constitutive suite: closed-form equivalence, reduction limits, analytic
stress rates, and the nonlinear-memory limits.

Numerical content unchanged from `run_validation.py`; see issue #32.
"""

from typing import Any

import numpy as np
import pytest

import imr_fast
from imr_fast._stress import _stress
from _validation_support import NHKV, R0, REQ, T0, deviation, oldroyd_b, reference_times, solve_radius

SECTION = "2. Constitutive suite"

# Tight tolerances: the closed-form and composable paths must agree to solver
# noise, not to physical accuracy.
_EQUIVALENCE = dict(rtol=1e-10, atol=1e-12)
_TRAJECTORY_TOLERANCE = 1e-7

_GENERIC_NH = imr_fast.InstantaneousMaterial(imr_fast.NeoHookean(2500.0), imr_fast.Newtonian(0.1))


def _instantaneous_values(material, radius=0.5, velocity=-0.3, need_rate=True):
  config = imr_fast.SimulationConfig(R0=R0, Req=REQ, material=material)
  problem = imr_fast.prepare(config)
  return _stress(material, problem.parameters, radius, velocity, None, problem.instantaneous_material, need_rate)


@pytest.mark.parametrize("radial", (1, 2))
def test_composable_matches_closed_form(radial, measured):
  closed = solve_radius(reference_times(), NHKV, radial=radial, **_EQUIVALENCE)
  generic = solve_radius(reference_times(), _GENERIC_NH, radial=radial, **_EQUIVALENCE)
  worst = float(np.max(np.abs(generic - closed)))
  measured(f"composable NH/Newtonian radial={radial}", f"max|dR|={worst:.2e}")
  assert worst < _TRAJECTORY_TOLERANCE


def test_composable_matches_closed_form_with_thermal(measured):
  options = dict(bubtherm=1, medtherm=1, Nt=9, Mt=9, **_EQUIVALENCE)
  closed = solve_radius(reference_times(), NHKV, **options)
  generic = solve_radius(reference_times(), _GENERIC_NH, **options)
  worst = float(np.max(np.abs(generic - closed)))
  measured("composable NH/Newtonian thermal", f"max|dR|={worst:.2e}")
  assert worst < _TRAJECTORY_TOLERANCE


_ELASTIC_REDUCTIONS = [
  ("Mooney-Rivlin", imr_fast.MooneyRivlin(1250.0, 0.0), 1e-12),
  ("Yeoh", imr_fast.Yeoh(1250.0), 1e-12),
  ("Fung", imr_fast.Fung(2500.0, 0.0), 1e-12),
  ("Gent", imr_fast.Gent(2500.0, 1e9), 1e-8),
  ("Arruda-Boyce", imr_fast.ArrudaBoyce(2500.0, 1e9), 1e-8),
  # Ogden is exact rather than asymptotic in its limit: one term at alpha = 2
  # IS neo-Hookean, so it is held to the same 1e-12 as Mooney-Rivlin and Yeoh.
  ("Ogden one term", imr_fast.Ogden((2500.0,), (2.0,)), 1e-12),
  # Two terms summing to the same small-strain modulus is not the same law, so
  # it must NOT reduce -- covered by test_ogden_multi_term_is_distinct below.
]


@pytest.mark.parametrize("label,elastic,tolerance", _ELASTIC_REDUCTIONS, ids=[c[0] for c in _ELASTIC_REDUCTIONS])
def test_elastic_reduces_to_neo_hookean(label, elastic, tolerance, measured):
  expected = _instantaneous_values(imr_fast.InstantaneousMaterial(elastic=imr_fast.NeoHookean(2500.0)))[0]
  value = _instantaneous_values(imr_fast.InstantaneousMaterial(elastic=elastic))[0]
  error = abs(value - expected) / abs(expected)
  measured(f"{label} -> neo-Hookean", f"rel={error:.2e}")
  assert error < tolerance


_VISCOUS_REDUCTIONS = [
  ("power law", imr_fast.PowerLaw(0.1, 1.0)),
  ("Carreau-Yasuda", imr_fast.CarreauYasuda(0.1, 0.1, 1.0, 2.0, 0.5)),
  ("Cross", imr_fast.Cross(0.1, 0.1, 1.0, 2.0)),
  ("Powell-Eyring", imr_fast.PowellEyring(0.1, 0.1, 1.0)),
  ("mod Powell-Eyring", imr_fast.ModifiedPowellEyring(0.1, 0.1, 1.0)),
  ("Powell-Eyring lam=0", imr_fast.PowellEyring(0.1, 0.05, 0.0)),
  ("mod Powell-Eyring lam=0", imr_fast.ModifiedPowellEyring(0.1, 0.05, 0.0)),
  ("Herschel-Bulkley", imr_fast.HerschelBulkley(0.0, 0.1, 1.0)),
  ("Bingham", imr_fast.Bingham(0.0, 0.1)),
]


@pytest.mark.parametrize("label,viscous", _VISCOUS_REDUCTIONS, ids=[c[0] for c in _VISCOUS_REDUCTIONS])
def test_viscous_reduces_to_newtonian(label, viscous, measured):
  expected = _instantaneous_values(imr_fast.InstantaneousMaterial(viscous=imr_fast.Newtonian(0.1)))[0]
  value = _instantaneous_values(imr_fast.InstantaneousMaterial(viscous=viscous))[0]
  error = abs(value - expected) / abs(expected)
  measured(f"{label} -> Newtonian", f"rel={error:.2e}")
  assert error < 1e-12


_RATE_MATERIALS = [
  ("Mooney-Rivlin", imr_fast.InstantaneousMaterial(elastic=imr_fast.MooneyRivlin(1000.0, 400.0))),
  ("Yeoh", imr_fast.InstantaneousMaterial(elastic=imr_fast.Yeoh(1000.0, 100.0, 10.0))),
  ("Fung", imr_fast.InstantaneousMaterial(elastic=imr_fast.Fung(2500.0, 0.2))),
  ("Gent", imr_fast.InstantaneousMaterial(elastic=imr_fast.Gent(2500.0, 500.0))),
  ("Arruda-Boyce", imr_fast.InstantaneousMaterial(elastic=imr_fast.ArrudaBoyce(2500.0, 50.0))),
  ("power law", imr_fast.InstantaneousMaterial(viscous=imr_fast.PowerLaw(0.1, 0.7))),
  ("Carreau-Yasuda", imr_fast.InstantaneousMaterial(viscous=imr_fast.CarreauYasuda(0.1, 0.01, 1e-5, 2.0, 0.5))),
  ("Cross", imr_fast.InstantaneousMaterial(viscous=imr_fast.Cross(0.1, 0.01, 1e-5, 2.0))),
  ("Herschel-Bulkley", imr_fast.InstantaneousMaterial(viscous=imr_fast.HerschelBulkley(100.0, 0.1, 0.8))),
  ("Bingham", imr_fast.InstantaneousMaterial(viscous=imr_fast.Bingham(100.0, 0.1))),
  ("Powell-Eyring", imr_fast.InstantaneousMaterial(viscous=imr_fast.PowellEyring(0.5, 0.1, 2e-5))),
  ("mod Powell-Eyring", imr_fast.InstantaneousMaterial(viscous=imr_fast.ModifiedPowellEyring(0.5, 0.1, 2e-5))),
]


@pytest.mark.parametrize("label,material", _RATE_MATERIALS, ids=[c[0] for c in _RATE_MATERIALS])
def test_analytic_stress_rate_matches_centered_difference(label, material, measured):
  """The solver evaluates stress rates analytically, including the viscosity
  tangent -- no finite difference inside the radial dynamics. This is what
  checks that derivation."""
  radius, velocity, acceleration, step = 0.5, -0.3, 0.2, 1e-6
  _, rate, _, coefficient = _instantaneous_values(material, radius, velocity)
  ahead = _instantaneous_values(material, radius + step * velocity, velocity + step * acceleration, False)[0]
  behind = _instantaneous_values(material, radius - step * velocity, velocity - step * acceleration, False)[0]
  difference = (ahead - behind) / (2 * step)
  predicted = rate - coefficient / radius * acceleration
  error = abs(difference - predicted) / max(1.0, abs(difference))
  measured(f"stress rate {label}", f"rel={error:.2e}")
  assert error < 2e-7


@pytest.mark.parametrize("stretch", (0.4, 0.9, 1.0, 1.0005, 1.3, 2.5))
def test_ogden_matches_neo_hookean_through_the_series_switch(stretch, measured):
  """Ogden's (1 - u**a)/(1 - u) factor is 0/0 at u = 1 and is covered by a
  binomial series nearby. Check the seam, not just the smooth regions -- a wrong
  series or a mis-set switch threshold shows up only within it."""
  reference = imr_fast.InstantaneousMaterial(elastic=imr_fast.NeoHookean(2500.0))
  ogden = imr_fast.InstantaneousMaterial(elastic=imr_fast.Ogden((2500.0,), (2.0,)))
  expected = _instantaneous_values(reference, radius=stretch)[0]
  actual = _instantaneous_values(ogden, radius=stretch)[0]
  error = abs(actual - expected) / abs(expected)
  measured(f"Ogden vs neo-Hookean, stretch={stretch}", f"rel={error:.2e}")
  assert error < 1e-12


def test_ogden_multi_term_is_distinct(measured):
  """A reduction limit alone would pass for an implementation that ignored all
  but the first term. Terms with a fractional and a negative exponent must move
  the stress away from neo-Hookean."""
  reference = imr_fast.InstantaneousMaterial(elastic=imr_fast.NeoHookean(2500.0))
  multi = imr_fast.InstantaneousMaterial(elastic=imr_fast.Ogden((1800.0, 600.0, -300.0), (1.3, 4.0, -2.0)))
  expected = _instantaneous_values(reference, radius=0.6)[0]
  actual = _instantaneous_values(multi, radius=0.6)[0]
  separation = abs(actual - expected) / abs(expected)
  measured("Ogden 3-term vs neo-Hookean", f"rel={separation:.2e}")
  assert separation > 0.05


def test_gent_lockup_becomes_a_solver_failure():
  with pytest.raises(imr_fast.SimulationError, match="Gent lock-up"):
    solve_radius(reference_times()[:3], imr_fast.InstantaneousMaterial(elastic=imr_fast.Gent(2500.0, 5.0)))


# Section 2b. Nonlinear memory (Giesekus / PTT) reduction limits.
_DE, _LAM = 2.0, 0.2
_MEMORY_TIMES = np.linspace(0, 1.2e-4, 300)
_RELAXATION = _DE * T0
_RETARDATION = _LAM * _RELAXATION


@pytest.fixture(scope="module")
def ucm_trajectory():
  return solve_radius(_MEMORY_TIMES, oldroyd_b())


@pytest.mark.parametrize(
  "label,model",
  [("giesekus", imr_fast.Giesekus(0.1, _RELAXATION, _RETARDATION)), ("linear PTT", imr_fast.LinearPTT(0.1, _RELAXATION, _RETARDATION))],
  ids=["giesekus", "linear-ptt"],
)
def test_zero_nonlinearity_reproduces_ucm(label, model, ucm_trajectory, measured):
  worst = deviation(solve_radius(_MEMORY_TIMES, model), ucm_trajectory)
  measured(f"{label} -> UCM", f"max|dR|={worst:.2e}")
  assert worst < 5e-3  # discretisation-limited, converging in points


def test_zero_nonlinearity_reproduces_ucm_keller_miksis(measured):
  ucm = solve_radius(_MEMORY_TIMES, oldroyd_b(), radial=2)
  distributed = solve_radius(_MEMORY_TIMES, imr_fast.Giesekus(0.1, _RELAXATION, _RETARDATION), radial=2)
  worst = deviation(distributed, ucm)
  measured("KM Giesekus -> UCM", f"max|dR|={worst:.2e}")
  assert worst < 2e-3


def test_zero_nonlinearity_reproduces_ucm_coupled(measured):
  """With `medtherm` the two do NOT reduce to each other at fixed De, and the
  original form of this test asserted that they did (#47).

  They take different medium-heating paths: `_dissipation` carries the polymer
  at its quasi-steady value, `_distributed_dissipation` carries its actual
  stress. So the gap is physics, not discretisation -- it GROWS with Mt
  (2.0e-03, 2.7e-03, 7.2e-03 at Mt = 9, 17, 33), which is why the old fixed
  tolerance passed at Mt=9 and failed by Mt=33. Resolution was never the
  variable. What is true is that the gap vanishes as De -> 0, so that is what
  gets asserted.
  """
  # `thermal="fd"` pinned deliberately: the claim is about the two constitutive
  # paths, which share whatever thermal scheme is in use, so the discretisation
  # is not a variable here. Pinning keeps the test independent of the default
  # (#26) -- and this case is by far the most expensive in the suite, so
  # inheriting a stiffer operator costs minutes for no added coverage.
  options: dict[str, Any] = dict(bubtherm=1, medtherm=1, vapor=1, masstrans=1, Nt=9, Mt=9, thermal="fd")

  def gap(relaxation):
    retardation = _LAM * relaxation
    ucm = solve_radius(_MEMORY_TIMES, imr_fast.OldroydB(0.1, relaxation, retardation), **options)
    distributed = solve_radius(_MEMORY_TIMES, imr_fast.Giesekus(0.1, relaxation, retardation), **options)
    return deviation(distributed, ucm)

  slow, fast = gap(_RELAXATION), gap(0.25 * _RELAXATION)
  measured("coupled Giesekus -> UCM", f"max|dR|={slow:.2e} -> {fast:.2e} at De/4")
  assert fast < 0.8 * slow, "the two paths must converge as the polymer approaches quasi-steady"


@pytest.mark.parametrize(
  "label,model",
  [
    ("giesekus", imr_fast.Giesekus(0.1, _RELAXATION, _RETARDATION, mobility=0.2)),
    ("linear PTT", imr_fast.LinearPTT(0.1, _RELAXATION, _RETARDATION, extensibility=0.2)),
  ],
  ids=["giesekus", "linear-ptt"],
)
def test_nonlinear_parameter_produces_distinct_physics(label, model, ucm_trajectory, measured):
  """A reduction limit alone would pass for a model that ignores its own
  nonlinear parameter. This is the other half."""
  worst = deviation(solve_radius(_MEMORY_TIMES, model), ucm_trajectory)
  measured(f"{label} parameter=0.2 vs UCM", f"max|dR|={worst:.2e}")
  assert worst > 0.05


@pytest.mark.parametrize(
  ("label", "viscous"), (("Bingham", imr_fast.Bingham(100.0, 0.1)), ("Herschel-Bulkley", imr_fast.HerschelBulkley(100.0, 0.1, 0.8)))
)
def test_yield_stress_needs_no_suppression(label, viscous, measured):
  """The yield-stress regularisation used `np.where` around a divide by the
  shear rate, wrapped in `np.errstate` (#35). `np.where` evaluates BOTH arms, so
  the 0/0 at every zero-rate node was computed and then discarded -- the
  suppression hid a value nothing used.

  The rate passes through zero at every rebound, so a real solve reaches it.
  Underflow is excluded deliberately: it fires inside SciPy and NumPy ignores it
  by default.
  """
  material = imr_fast.InstantaneousMaterial(elastic=imr_fast.NeoHookean(2500.0), viscous=viscous)
  config = imr_fast.SimulationConfig(R0=R0, Req=REQ, material=material)
  with np.errstate(divide="raise", invalid="raise", over="raise"):
    result = imr_fast.simulate(np.linspace(0.0, 6e-5, 200), config)
  radius = np.asarray(result.radius_ratio)
  measured(f"{label} no suppression", f"min R/R0={radius.min():.4f}")
  assert np.all(np.isfinite(radius))
