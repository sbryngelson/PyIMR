"""Closed forms the radial dynamics must satisfy, owing nothing to IMRv2."""

from dataclasses import replace
import functools
from typing import Any

import numpy as np
import pytest

import pyimr
from pyimr import _thermal

SECTION = "1d. Closed forms independent of IMRv2"

RAYLEIGH = 0.9146808342
_R0 = 225e-6
_RHO, _P8 = 1064.0, 101325.0
_ANALYTIC = RAYLEIGH * _R0 * np.sqrt(_RHO / _P8)

_SIGMA = 1e-12
_INVISCID = pyimr.PhysicalParameters(surface_tension_n_m=_SIGMA, far_field_pressure_pa=_P8, medium_density_kg_m3=_RHO)

_SAMPLES = 40001
_WINDOW = 1.3 * _ANALYTIC
_RESOLUTION = _WINDOW / (_SAMPLES - 1) / _ANALYTIC


def _collapse_time(ratio):
  config = pyimr.SimulationConfig(R0=_R0, Req=_R0 * ratio, material=pyimr.NoStress(), dynamics="rayleigh-plesset", physics=_INVISCID, rtol=1e-11, atol=1e-13)
  times = np.linspace(0.0, _WINDOW, _SAMPLES)
  radius = np.asarray(pyimr.simulate(times, config).radius_ratio)
  return float(times[int(np.argmin(radius))])


def test_collapse_time_converges_to_rayleigh(measured):
  """Residual gas cushions the collapse, so the time is long and falls toward"""
  ratios = (1.0 / 6.0, 0.1, 0.05, 0.02)
  errors = [abs(_collapse_time(ratio) - _ANALYTIC) / _ANALYTIC for ratio in ratios]

  measured("Rayleigh t_c", f"analytic={_ANALYTIC * 1e6:.4f}us  rel={' -> '.join(f'{e:.1e}' for e in errors)}")
  assert errors[0] > errors[1] > errors[2], "collapse time must converge as the gas content vanishes"
  assert errors[-1] < max(4.0 * _RESOLUTION, 1e-5), "did not reach the analytic value within grid resolution"


def test_the_step_budget_failure_is_retried_at_higher_order(measured):
  """The retry that replaces LSODA's switching, asserted on the case that needed it."""
  config = pyimr.SimulationConfig(
    R0=_R0, Req=_R0 * 0.02, material=pyimr.NoStress(), dynamics="rayleigh-plesset", physics=_INVISCID, rtol=1e-11, atol=1e-13
  )
  result = pyimr.simulate(np.linspace(0.0, _WINDOW, 2001), config)
  measured("step-budget retry", f"backend={result.stats.backend}")
  assert "dopri8" in result.stats.backend, f"expected the higher-order retry, got {result.stats.backend}"
  assert result.stats.success

  easy = pyimr.simulate(np.linspace(0.0, _WINDOW, 2001), replace(config, Req=_R0 / 6.0))
  assert "tsit5" in easy.stats.backend, f"the retry fired where it was not needed: {easy.stats.backend}"


def test_gas_content_lengthens_the_collapse(measured):
  """Direction, not just magnitude. Gas resists compression, so a fuller bubble"""
  loose, tight = _collapse_time(1.0 / 6.0), _collapse_time(0.05)
  measured("gas cushioning", f"Req/R0=1/6: {loose * 1e6:.4f}us  Req/R0=0.05: {tight * 1e6:.4f}us")
  assert loose > tight > _ANALYTIC * (1.0 - _RESOLUTION)


def _trace(ratio, dynamics, liquid_eos=None, sound_speed=None, window=60e-6, samples=4000):
  physics = (
    _INVISCID
    if sound_speed is None
    else pyimr.PhysicalParameters(surface_tension_n_m=_SIGMA, far_field_pressure_pa=_P8, medium_density_kg_m3=_RHO, sound_speed_m_s=sound_speed)
  )
  config = pyimr.SimulationConfig(R0=_R0, Req=_R0 * ratio, material=pyimr.NoStress(), dynamics=dynamics, liquid_eos=liquid_eos, physics=physics, rtol=1e-12, atol=1e-14)
  return pyimr.simulate(np.linspace(0.0, window, samples), config)


def _first_integral_residual(ratio, dynamics="rayleigh-plesset", liquid_eos=None):
  result = _trace(ratio, dynamics, liquid_eos)
  radius = np.asarray(result.radius_ratio) * _R0
  velocity = np.asarray(result.wall_velocity_m_s)

  equilibrium = _R0 * ratio
  gas_at_r0 = (_P8 + 2.0 * _SIGMA / equilibrium) * (equilibrium / _R0) ** (3.0 * pyimr.KAPPA)
  exponent = 3.0 - 3.0 * pyimr.KAPPA
  gas = 2.0 * gas_at_r0 * _R0 ** (3.0 * pyimr.KAPPA) / (_RHO * exponent) * (radius**exponent - _R0**exponent)
  ambient = 2.0 * _P8 / (3.0 * _RHO) * (radius**3 - _R0**3)

  actual = radius**3 * velocity**2
  return float(np.max(np.abs(actual - (gas - ambient)))) / float(np.max(np.abs(actual)))


@pytest.mark.parametrize("ratio", (1.0 / 6.0, 0.1, 0.3))
def test_rayleigh_plesset_conserves_its_first_integral(ratio, measured):
  """Pointwise, so it constrains the entire right-hand side rather than one"""
  residual = _first_integral_residual(ratio)
  measured(f"RP first integral Req/R0={ratio:.3f}", f"rel={residual:.2e}")
  assert residual < 1e-8


def test_keller_miksis_is_compressible_at_all(measured):
  """The converse of the test above. At a physical sound speed Keller-Miksis"""
  incompressible = _first_integral_residual(1.0 / 6.0, dynamics="rayleigh-plesset")
  compressible = _first_integral_residual(1.0 / 6.0, dynamics="keller-miksis")
  measured("KM violates the RP invariant", f"RP rel={incompressible:.1e}  KM rel={compressible:.2f}")
  assert compressible > 0.1, "KM satisfies the incompressible invariant -- is it actually solving RP?"
  assert incompressible < 1e-8


def test_keller_miksis_approaches_rayleigh_plesset_as_first_order_in_one_over_c(measured):
  """`c -> inf` is the easy half. The order matters more: Keller-Miksis carries"""
  gaps = []
  for sound_speed in (1e7, 1e9):
    reference = np.asarray(_trace(1.0 / 6.0, dynamics="rayleigh-plesset", sound_speed=sound_speed).radius_ratio)
    compressible = np.asarray(_trace(1.0 / 6.0, dynamics="keller-miksis", sound_speed=sound_speed).radius_ratio)
    gaps.append(float(np.max(np.abs(compressible - reference))))

  ratio = gaps[0] / gaps[1]
  measured("KM -> RP as 1/c", f"c=1e7: {gaps[0]:.2e}, c=1e9: {gaps[1]:.2e}, ratio={ratio:.1f} (first order = 100)")
  assert gaps[1] < gaps[0]
  assert 50.0 < ratio < 200.0, "gap must fall like 1/c, not faster or slower"


_BASE_PHYSICS = pyimr.PhysicalParameters()
_T8 = 298.15  # SimulationConfig default far-field temperature, and the wall value when medtherm=0


def _scaled_conductivity(scale):
  return pyimr.PhysicalParameters(
    gas_conductivity_slope=_BASE_PHYSICS.gas_conductivity_slope * scale,
    gas_conductivity_offset=_BASE_PHYSICS.gas_conductivity_offset * scale,
    vapor_conductivity_slope=_BASE_PHYSICS.vapor_conductivity_slope * scale,
    vapor_conductivity_offset=_BASE_PHYSICS.vapor_conductivity_offset * scale,
  )


_MEDIUM = {"medtherm": 1, "Mt": 25}


@functools.lru_cache(maxsize=None)
def _polytropic_state(conductivity_scale, exponent, samples=600, thermal=None, medium=False):
  options: dict[str, Any] = dict(_MEDIUM) if medium else {}
  if thermal is not None:
    options["thermal"] = thermal
  config = pyimr.SimulationConfig(
    R0=_R0,
    Req=_R0 / 6,
    material=pyimr.NoStress(),
    dynamics="rayleigh-plesset",
    bubtherm=1,
    Nt=25,
    physics=_scaled_conductivity(conductivity_scale),
    rtol=1e-10,
    atol=1e-12,
    **options,
  )
  result = pyimr.simulate(np.linspace(0.0, 40e-6, samples), config)
  invariant = np.asarray(result.internal_pressure_pa) * np.asarray(result.radius_ratio) ** exponent
  temperature = np.asarray(result.bubble_temperature_k)
  drift = float(np.max(np.abs(invariant - invariant[0])) / abs(invariant[0]))
  return drift, float(np.min(temperature)), float(np.max(temperature))


_ADIABATIC = 3.0 * pyimr.KAPPA
_ISOTHERMAL = 3.0


@pytest.mark.parametrize("thermal", ("spectral", "fd"))
@pytest.mark.parametrize("medium", (False, True), ids=("gas only", "with medtherm"))
def test_both_schemes_anchor_at_the_adiabatic_limit(thermal, medium, measured):
  """The closed forms must hold for the scheme that actually ships."""
  drift, _, hottest = _polytropic_state(1e-8, _ADIABATIC, thermal=thermal, medium=medium)
  measured(f"adiabatic {thermal}{' +medtherm' if medium else ''}", f"drift={drift:.2e}  Tmax={hottest:.0f}K")
  assert drift < 1e-3
  assert hottest > 3000.0, "no compression heating -- the reduction is degenerate, not adiabatic"


def test_thermal_pde_reduces_to_the_adiabatic_polytropic_law(measured):
  """Switch conduction off and the thermal PDE must reproduce `P R^(3*gamma) ="""
  drift, coldest, hottest = _polytropic_state(1e-8, _ADIABATIC)
  measured("thermal -> adiabatic", f"P*R^(3g) drift={drift:.2e}  T in [{coldest:.1f}, {hottest:.1f}] K")
  assert drift < 1e-3
  assert hottest > 3000.0, "no compression heating -- the reduction is degenerate, not adiabatic"


def test_thermal_pde_reduces_to_the_isothermal_polytropic_law(measured):
  """The other limit. Fast conduction pins the gas at the wall temperature, so"""
  drift, coldest, hottest = _polytropic_state(1e5, _ISOTHERMAL)
  measured("thermal -> isothermal", f"P*R^3 drift={drift:.2e}  T in [{coldest:.1f}, {hottest:.1f}] K")
  assert drift < 5e-2
  assert abs(coldest - _T8) < 10.0 and abs(hottest - _T8) < 10.0


def test_conduction_carries_the_gas_between_both_limits(measured):
  """One knob, two endpoints. Raising `chi` must break the adiabatic invariant"""
  scales = (1e-8, 1e-4, 1e-2, 1.0, 1e2, 1e5)
  adiabatic = [_polytropic_state(scale, _ADIABATIC)[0] for scale in scales]
  states = [_polytropic_state(scale, _ISOTHERMAL) for scale in scales]
  isothermal = [row[0] for row in states]
  spread = [row[2] - row[1] for row in states]

  measured(
    "chi sweeps adiabatic -> isothermal",
    f"P*R^(3g) {adiabatic[0]:.1e}->{adiabatic[-1]:.1e}; "
    f"P*R^3 {isothermal[0]:.1e}->{isothermal[-1]:.1e}; T spread {spread[0]:.0f}K->{spread[-1]:.0f}K",
  )
  assert all(a < b for a, b in zip(adiabatic[:-1], adiabatic[1:], strict=True)), "adiabatic must degrade with chi"
  assert all(a > b for a, b in zip(isothermal[:-1], isothermal[1:], strict=True)), "isothermal must improve with chi"
  assert spread[-1] < 0.01 * spread[0], "the gas must end up pinned at the wall temperature"


@pytest.mark.parametrize("options", ({"bubtherm": 1, "Nt": 25}, {"bubtherm": 1, "medtherm": 1, "Nt": 25, "Mt": 25}))
def test_a_dry_run_ignores_the_vapour_conductivity(options, measured):
  """With no vapour present, the vapour conductivity cannot affect anything."""
  times = np.linspace(0.0, 40e-6, 400)
  baseline, worst = None, 0.0
  for scale in (1.0, 1.5, 3.0, 10.0):
    physics = pyimr.PhysicalParameters(
      vapor_conductivity_slope=_BASE_PHYSICS.vapor_conductivity_slope * scale,
      vapor_conductivity_offset=_BASE_PHYSICS.vapor_conductivity_offset * scale,
    )
    config = pyimr.SimulationConfig(
      R0=_R0, Req=_R0 / 6, material=pyimr.NoStress(), dynamics="rayleigh-plesset", physics=physics, rtol=1e-10, atol=1e-12, **options
    )
    trace = np.asarray(pyimr.simulate(times, config).radius_ratio)
    baseline = trace if baseline is None else baseline
    worst = max(worst, float(np.max(np.abs(trace - baseline))))
  measured(f"vapour-K invariance {'+medtherm' if 'medtherm' in options else 'gas'}", f"max|dR|={worst:.2e}")
  assert worst < 1e-7


@pytest.mark.parametrize("dynamics", ("keller-enthalpy", "gilmore"))
def test_the_mie_gruneisen_domain_is_one_boundary_not_two(dynamics, measured):
  """The Hugoniot's density root and its sound speed fail at the same place."""
  s, nog = _thermal._HUGONIOT_S, _thermal._NOG
  limit = -1.0 / (4.0 * (s + nog))
  assert _thermal._mu_of_A(limit, s, nog) == pytest.approx(-1.0 / (s + 2.0 * nog), rel=1e-12)

  grid = np.concatenate([np.linspace(limit, 0.0, 4000), np.logspace(-6, 4, 4000)])
  mu = _thermal._mu_of_A(grid, s, nog)
  assert np.all(np.isfinite(mu)), "the closed-form discriminant must stay real on its whole domain"
  radicand = (1.0 + (s + 2.0 * nog) * mu) / (1.0 - s * mu) ** 3
  assert np.all(radicand >= 0.0), "a real density root must imply a real sound speed"

  config = pyimr.SimulationConfig(R0=_R0, Req=_R0 / 6, material=pyimr.NeoHookeanKelvinVoigt(2500.0, 0.1), dynamics=dynamics, liquid_eos="mie-gruneisen")
  parameters = pyimr.prepare(config).parameters
  result = pyimr.simulate(np.linspace(0.0, 60e-6, 2000), config)
  bubble = np.asarray(result.internal_pressure_pa) + np.asarray(result.stress_integral_pa)
  reached = (bubble / parameters["P8"] - parameters["iWe"] / np.asarray(result.radius_ratio)) / parameters["Cstar"] ** 2
  measured(f"Mie-Gruneisen A range {dynamics}", f"[{reached.min():.2e}, {reached.max():.2e}] vs limit {limit:.4f}")
  assert reached.min() > limit, "the trajectory must stay inside the Hugoniot's domain"
