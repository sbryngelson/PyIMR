"""Constitutive suite: closed-form equivalence, reduction limits, analytic"""

from typing import Any

import numpy as np
import pytest

import pyimr
from pyimr._stress import _stress
from _validation_support import NHKV, R0, REQ, T0, deviation, oldroyd_b, reference_times, solve_radius

SECTION = "2. Constitutive suite"

_EQUIVALENCE = dict(rtol=1e-10, atol=1e-12)
_TRAJECTORY_TOLERANCE = 1e-7

_GENERIC_NH = pyimr.InstantaneousMaterial(pyimr.NeoHookean(2500.0), pyimr.Newtonian(0.1))


def _instantaneous_values(material, radius=0.5, velocity=-0.3, need_rate=True):
  config = pyimr.SimulationConfig(R0=R0, Req=REQ, material=material)
  problem = pyimr.prepare(config)
  return _stress(material, problem.parameters, radius, velocity, None, problem.instantaneous_material, need_rate)


@pytest.mark.parametrize("dynamics", ("rayleigh-plesset", "keller-miksis"))
def test_composable_matches_closed_form(dynamics, measured):
  closed = solve_radius(reference_times(), NHKV, dynamics=dynamics, **_EQUIVALENCE)
  generic = solve_radius(reference_times(), _GENERIC_NH, dynamics=dynamics, **_EQUIVALENCE)
  worst = float(np.max(np.abs(generic - closed)))
  measured(f"composable NH/Newtonian {dynamics}", f"max|dR|={worst:.2e}")
  assert worst < _TRAJECTORY_TOLERANCE


def test_composable_matches_closed_form_with_thermal(measured):
  options = dict(bubtherm=1, medtherm=1, Nt=9, Mt=9, **_EQUIVALENCE)
  closed = solve_radius(reference_times(), NHKV, **options)
  generic = solve_radius(reference_times(), _GENERIC_NH, **options)
  worst = float(np.max(np.abs(generic - closed)))
  measured("composable NH/Newtonian thermal", f"max|dR|={worst:.2e}")
  assert worst < _TRAJECTORY_TOLERANCE


_ELASTIC_REDUCTIONS = [
  ("Mooney-Rivlin", pyimr.MooneyRivlin(1250.0, 0.0), 1e-12),
  ("Yeoh", pyimr.Yeoh(1250.0), 1e-12),
  ("Fung", pyimr.Fung(2500.0, 0.0), 1e-12),
  ("Gent", pyimr.Gent(2500.0, 1e9), 1e-8),
  ("Arruda-Boyce", pyimr.ArrudaBoyce(2500.0, 1e9), 1e-8),
  ("Ogden one term", pyimr.Ogden((2500.0,), (2.0,)), 1e-12),
]


@pytest.mark.parametrize("label,elastic,tolerance", _ELASTIC_REDUCTIONS, ids=[c[0] for c in _ELASTIC_REDUCTIONS])
def test_elastic_reduces_to_neo_hookean(label, elastic, tolerance, measured):
  expected = _instantaneous_values(pyimr.InstantaneousMaterial(elastic=pyimr.NeoHookean(2500.0)))[0]
  value = _instantaneous_values(pyimr.InstantaneousMaterial(elastic=elastic))[0]
  error = abs(value - expected) / abs(expected)
  measured(f"{label} -> neo-Hookean", f"rel={error:.2e}")
  assert error < tolerance


_VISCOUS_REDUCTIONS = [
  ("power law", pyimr.PowerLaw(0.1, 1.0)),
  ("Carreau-Yasuda", pyimr.CarreauYasuda(0.1, 0.1, 1.0, 2.0, 0.5)),
  ("Cross", pyimr.Cross(0.1, 0.1, 1.0, 2.0)),
  ("Powell-Eyring", pyimr.PowellEyring(0.1, 0.1, 1.0)),
  ("mod Powell-Eyring", pyimr.ModifiedPowellEyring(0.1, 0.1, 1.0)),
  ("Powell-Eyring lam=0", pyimr.PowellEyring(0.1, 0.05, 0.0)),
  ("mod Powell-Eyring lam=0", pyimr.ModifiedPowellEyring(0.1, 0.05, 0.0)),
  ("Herschel-Bulkley", pyimr.HerschelBulkley(0.0, 0.1, 1.0)),
  ("Bingham", pyimr.Bingham(0.0, 0.1)),
]


@pytest.mark.parametrize("label,viscous", _VISCOUS_REDUCTIONS, ids=[c[0] for c in _VISCOUS_REDUCTIONS])
def test_viscous_reduces_to_newtonian(label, viscous, measured):
  expected = _instantaneous_values(pyimr.InstantaneousMaterial(viscous=pyimr.Newtonian(0.1)))[0]
  value = _instantaneous_values(pyimr.InstantaneousMaterial(viscous=viscous))[0]
  error = abs(value - expected) / abs(expected)
  measured(f"{label} -> Newtonian", f"rel={error:.2e}")
  assert error < 1e-12


_RATE_MATERIALS = [
  ("Mooney-Rivlin", pyimr.InstantaneousMaterial(elastic=pyimr.MooneyRivlin(1000.0, 400.0))),
  ("Yeoh", pyimr.InstantaneousMaterial(elastic=pyimr.Yeoh(1000.0, 100.0, 10.0))),
  ("Fung", pyimr.InstantaneousMaterial(elastic=pyimr.Fung(2500.0, 0.2))),
  ("Gent", pyimr.InstantaneousMaterial(elastic=pyimr.Gent(2500.0, 500.0))),
  ("Arruda-Boyce", pyimr.InstantaneousMaterial(elastic=pyimr.ArrudaBoyce(2500.0, 50.0))),
  ("power law", pyimr.InstantaneousMaterial(viscous=pyimr.PowerLaw(0.1, 0.7))),
  ("Carreau-Yasuda", pyimr.InstantaneousMaterial(viscous=pyimr.CarreauYasuda(0.1, 0.01, 1e-5, 2.0, 0.5))),
  ("Cross", pyimr.InstantaneousMaterial(viscous=pyimr.Cross(0.1, 0.01, 1e-5, 2.0))),
  ("Herschel-Bulkley", pyimr.InstantaneousMaterial(viscous=pyimr.HerschelBulkley(100.0, 0.1, 0.8))),
  ("Bingham", pyimr.InstantaneousMaterial(viscous=pyimr.Bingham(100.0, 0.1))),
  ("Powell-Eyring", pyimr.InstantaneousMaterial(viscous=pyimr.PowellEyring(0.5, 0.1, 2e-5))),
  ("mod Powell-Eyring", pyimr.InstantaneousMaterial(viscous=pyimr.ModifiedPowellEyring(0.5, 0.1, 2e-5))),
]


@pytest.mark.parametrize("label,material", _RATE_MATERIALS, ids=[c[0] for c in _RATE_MATERIALS])
def test_analytic_stress_rate_matches_centered_difference(label, material, measured):
  """The solver evaluates stress rates analytically, including the viscosity"""
  radius, velocity, acceleration, step = 0.5, -0.3, 0.2, 1e-6
  _, rate, _, coefficient = _instantaneous_values(material, radius, velocity)
  ahead = _instantaneous_values(material, radius + step * velocity, velocity + step * acceleration, False)[0]
  behind = _instantaneous_values(material, radius - step * velocity, velocity - step * acceleration, False)[0]
  difference = (ahead - behind) / (2 * step)
  predicted = rate - coefficient / radius * acceleration
  error = abs(difference - predicted) / max(1.0, abs(difference))
  measured(f"stress rate {label}", f"rel={error:.2e}")
  assert error < 2e-7


@pytest.mark.parametrize("label,material", _RATE_MATERIALS, ids=[c[0] for c in _RATE_MATERIALS])
def test_rate_materials_evaluate_the_same_under_both_namespaces(label, material, measured):
  """Every material must evaluate under `jnp` as well as `np`, and agree."""
  from pyimr import _jax  # noqa: PLC0415
  from pyimr._stress import _instantaneous_stress  # noqa: PLC0415

  _, jnp, _ = _jax._jax()

  config = pyimr.SimulationConfig(R0=R0, Req=REQ, material=material)
  problem = pyimr.prepare(config)
  radius, velocity = 0.5, -0.3
  reference = _instantaneous_stress(material, problem.instantaneous_material, problem.parameters, radius, velocity, True, xp=np)
  traced = _instantaneous_stress(
    material, problem.instantaneous_material, problem.parameters, jnp.asarray(radius), jnp.asarray(velocity), True, xp=jnp
  )
  assert [value is None for value in reference] == [value is None for value in traced], f"{label}: namespaces disagree on which terms exist"
  pairs = [(float(a), float(b)) for a, b in zip(reference, traced, strict=True) if a is not None and b is not None]
  worst = max(abs(a - b) / max(1.0, abs(a)) for a, b in pairs)
  measured(f"namespace agreement {label}", f"rel={worst:.2e}")
  assert all(np.isfinite(b) for _, b in pairs), f"{label}: traced evaluation is not finite"
  assert worst < 1e-12, f"{label}: {worst:.3e}"


@pytest.mark.parametrize("stretch", (0.4, 0.9, 1.0, 1.0005, 1.3, 2.5))
def test_ogden_matches_neo_hookean_through_the_series_switch(stretch, measured):
  """Ogden's (1 - u**a)/(1 - u) factor is 0/0 at u = 1 and is covered by a"""
  reference = pyimr.InstantaneousMaterial(elastic=pyimr.NeoHookean(2500.0))
  ogden = pyimr.InstantaneousMaterial(elastic=pyimr.Ogden((2500.0,), (2.0,)))
  expected = _instantaneous_values(reference, radius=stretch)[0]
  actual = _instantaneous_values(ogden, radius=stretch)[0]
  error = abs(actual - expected) / abs(expected)
  measured(f"Ogden vs neo-Hookean, stretch={stretch}", f"rel={error:.2e}")
  assert error < 1e-12


def test_ogden_multi_term_is_distinct(measured):
  """A reduction limit alone would pass for an implementation that ignored all"""
  reference = pyimr.InstantaneousMaterial(elastic=pyimr.NeoHookean(2500.0))
  multi = pyimr.InstantaneousMaterial(elastic=pyimr.Ogden((1800.0, 600.0, -300.0), (1.3, 4.0, -2.0)))
  expected = _instantaneous_values(reference, radius=0.6)[0]
  actual = _instantaneous_values(multi, radius=0.6)[0]
  separation = abs(actual - expected) / abs(expected)
  measured("Ogden 3-term vs neo-Hookean", f"rel={separation:.2e}")
  assert separation > 0.05


def test_gent_lockup_becomes_a_solver_failure():
  with pytest.raises(pyimr.SimulationError, match="Gent lock-up"):
    solve_radius(reference_times()[:3], pyimr.InstantaneousMaterial(elastic=pyimr.Gent(2500.0, 5.0)))


_DE, _LAM = 2.0, 0.2
_MEMORY_TIMES = np.linspace(0, 1.2e-4, 300)
_RELAXATION = _DE * T0
_RETARDATION = _LAM * _RELAXATION


@pytest.fixture(scope="module")
def ucm_trajectory():
  return solve_radius(_MEMORY_TIMES, oldroyd_b())


@pytest.mark.parametrize(
  "label,model",
  [("giesekus", pyimr.Giesekus(0.1, _RELAXATION, _RETARDATION)), ("linear PTT", pyimr.LinearPTT(0.1, _RELAXATION, _RETARDATION))],
  ids=["giesekus", "linear-ptt"],
)
def test_zero_nonlinearity_reproduces_ucm(label, model, ucm_trajectory, measured):
  worst = deviation(solve_radius(_MEMORY_TIMES, model), ucm_trajectory)
  measured(f"{label} -> UCM", f"max|dR|={worst:.2e}")
  assert worst < 5e-3  # discretisation-limited, converging in points


def test_zero_nonlinearity_reproduces_ucm_keller_miksis(measured):
  ucm = solve_radius(_MEMORY_TIMES, oldroyd_b(), dynamics="keller-miksis")
  distributed = solve_radius(_MEMORY_TIMES, pyimr.Giesekus(0.1, _RELAXATION, _RETARDATION), dynamics="keller-miksis")
  worst = deviation(distributed, ucm)
  measured("KM Giesekus -> UCM", f"max|dR|={worst:.2e}")
  assert worst < 2e-3


def test_zero_nonlinearity_reproduces_ucm_coupled(measured):
  """With `medtherm` the two do NOT reduce to each other at fixed De, and the"""
  options: dict[str, Any] = dict(bubtherm=1, medtherm=1, vapor=1, masstrans=1, Nt=9, Mt=9, thermal="fd")

  def gap(relaxation):
    retardation = _LAM * relaxation
    ucm = solve_radius(_MEMORY_TIMES, pyimr.OldroydB(0.1, relaxation, retardation), **options)
    distributed = solve_radius(_MEMORY_TIMES, pyimr.Giesekus(0.1, relaxation, retardation), **options)
    return deviation(distributed, ucm)

  slow, fast = gap(_RELAXATION), gap(0.25 * _RELAXATION)
  measured("coupled Giesekus -> UCM", f"max|dR|={slow:.2e} -> {fast:.2e} at De/4")
  assert fast < 0.8 * slow, "the two paths must converge as the polymer approaches quasi-steady"


@pytest.mark.parametrize(
  "label,model",
  [
    ("giesekus", pyimr.Giesekus(0.1, _RELAXATION, _RETARDATION, mobility=0.2)),
    ("linear PTT", pyimr.LinearPTT(0.1, _RELAXATION, _RETARDATION, extensibility=0.2)),
  ],
  ids=["giesekus", "linear-ptt"],
)
def test_nonlinear_parameter_produces_distinct_physics(label, model, ucm_trajectory, measured):
  """A reduction limit alone would pass for a model that ignores its own"""
  worst = deviation(solve_radius(_MEMORY_TIMES, model), ucm_trajectory)
  measured(f"{label} parameter=0.2 vs UCM", f"max|dR|={worst:.2e}")
  assert worst > 0.05


@pytest.mark.parametrize(
  ("label", "viscous"), (("Bingham", pyimr.Bingham(100.0, 0.1)), ("Herschel-Bulkley", pyimr.HerschelBulkley(100.0, 0.1, 0.8)))
)
def test_yield_stress_needs_no_suppression(label, viscous, measured):
  """The yield-stress regularisation used `np.where` around a divide by the"""
  material = pyimr.InstantaneousMaterial(elastic=pyimr.NeoHookean(2500.0), viscous=viscous)
  config = pyimr.SimulationConfig(R0=R0, Req=REQ, material=material)
  with np.errstate(divide="raise", invalid="raise", over="raise"):
    result = pyimr.simulate(np.linspace(0.0, 6e-5, 200), config)
  radius = np.asarray(result.radius_ratio)
  measured(f"{label} no suppression", f"min R/R0={radius.min():.4f}")
  assert np.all(np.isfinite(radius))


_MAXWELL_VISCOSITY = 0.1


@pytest.mark.parametrize("dynamics", ("rayleigh-plesset", "keller-miksis"))
def test_linear_maxwell_reduces_to_newtonian_as_relaxation_vanishes(dynamics, measured):
  """A Maxwell fluid with no memory is a Newtonian one. Convergence must be first order
  in the relaxation time with no floor -- a floor would mean a term that survives the
  limit, which is how the `Zener` coefficient bug in #174 shows up.

  This also validates `LinearMaxwell`'s `acceleration_coefficient` of 0, which is only
  live in the compressible forms: `S = Z1/R^3` carries no instantaneous `Rd`, so no `R-ddot` term
  moves to the left of Keller-Miksis. The Newtonian reference declares `4/Re8`, and the
  two still agree because in the stiff limit that coupling re-emerges through `Z1`'s ODE.
  """
  # First collapse and rebound only, not `reference_times()`. The two models differ at
  # O(relaxation) by construction, and over the full 120 us reference window that seed
  # difference is amplified across repeated collapses -- a max-norm there measures chaotic
  # divergence, not the limit. Tests that compare algebraically identical formulations can
  # use the full window; this one cannot.
  times = np.linspace(0.0, 20e-6, 200)
  newtonian = solve_radius(times, pyimr.NeoHookeanKelvinVoigt(1e-6, _MAXWELL_VISCOSITY), dynamics=dynamics, **_EQUIVALENCE)

  errors = []
  for relaxation in (1e-7, 1e-8, 1e-9):
    radius = solve_radius(times, pyimr.LinearMaxwell(_MAXWELL_VISCOSITY, relaxation), dynamics=dynamics, **_EQUIVALENCE)
    errors.append(float(np.max(np.abs(radius - newtonian))))

  measured(f"Maxwell -> Newtonian {dynamics}", "  ".join(f"{e:.2e}" for e in errors))
  for coarse, fine in zip(errors, errors[1:]):
    assert coarse / fine > 8.0, f"{coarse:.3e} -> {fine:.3e} is not first-order decay"
  assert errors[-1] < 1e-5, f"error floor at {errors[-1]:.3e} means a term survives the limit"


@pytest.mark.parametrize("dynamics", ("rayleigh-plesset", "keller-miksis"))
def test_linear_maxwell_is_zener_without_the_parallel_spring(dynamics, measured):
  """`Zener` is `LinearMaxwell` plus an elastic branch, so removing the modulus recovers it.

  `dynamics="keller-miksis"` is the case that matters: the acceleration coefficient is unused at
  `dynamics="rayleigh-plesset"`, so only the compressible branch exercises it. This limit stalled at 1.57e-03
  there while `Zener` carried IMRv2's `4/Re8`, and converges since it was corrected to
  `4*LAM/Re8` (#174). It is the test that catches that coefficient.
  """
  times = reference_times()
  maxwell = solve_radius(times, pyimr.LinearMaxwell(_MAXWELL_VISCOSITY, 2e-6), dynamics=dynamics, **_EQUIVALENCE)

  errors = []
  for modulus in (1e0, 1e-2, 1e-4):
    zener = solve_radius(times, pyimr.Zener(modulus, _MAXWELL_VISCOSITY, 2e-6, 0.0), dynamics=dynamics, **_EQUIVALENCE)
    errors.append(float(np.max(np.abs(zener - maxwell))))

  measured(f"Zener(G->0) -> Maxwell {dynamics}", "  ".join(f"{e:.2e}" for e in errors))
  for coarse, fine in zip(errors, errors[1:]):
    assert coarse / fine > 80.0, f"{coarse:.3e} -> {fine:.3e} is not first-order in the modulus"
  assert errors[-1] < 1e-7


def test_linear_maxwell_refuses_a_missing_relaxation_time():
  with pytest.raises(ValueError):
    pyimr.LinearMaxwell(_MAXWELL_VISCOSITY, 0.0)
