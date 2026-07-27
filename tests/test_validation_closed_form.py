"""Closed forms the radial dynamics must satisfy, owing nothing to IMRv2.

Every other trajectory check in this suite compares against IMRv2 -- pinned
output from another *implementation*. Agreement there means the two codes agree,
not that either is right. Three checks here do not:

1. **Rayleigh collapse time** (1917), `t_c = 0.914681 * R0 * sqrt(rho / dp)` for
   an empty cavity in an incompressible, inviscid liquid. The constant is
   `sqrt(3*pi/2) * Gamma(5/6) / Gamma(1/3)`. Approached rather than reached: an
   empty cavity collapses through the solver's radius floor, so the gas content
   is driven toward zero through `Req/R0` and the collapse time watched as it
   converges.

2. **The exact first integral of Rayleigh-Plesset.** Multiplying by `2 R^2 R'`
   makes the left side `d/dt (R^3 R'^2)`, and for a polytropic gas the right
   side integrates in closed form. Unlike the other two this is a *pointwise*
   statement -- it constrains the whole trajectory, so it tests the entire
   radial right-hand side rather than one scalar summary of it.

3. **Keller-Miksis reduces to Rayleigh-Plesset as `c -> inf`,** at first order
   in `1/c`. That the *order* is right, not merely the limit, is what makes this
   a test of the compressibility correction rather than of a coincidence.

4. **The thermal PDE reduces to both polytropic limits.** No conduction gives
   `P R^(3*gamma) = const`; conduction fast enough to pin the gas at the wall
   gives `P R^3 = const`. One knob, `chi`, carries it monotonically between
   them.

Each limit is paired with its converse -- Keller-Miksis must *violate* the
incompressible invariant, conduction must *break* the adiabatic one. Without
those, a Keller-Miksis branch that silently fell back to Rayleigh-Plesset, or a
conduction term that was inert, would score perfectly.
"""

import numpy as np
import pytest

import imr_fast

SECTION = "1d. Closed forms independent of IMRv2"

RAYLEIGH = 0.9146808342
_R0 = 225e-6
_RHO, _P8 = 1064.0, 101325.0
_ANALYTIC = RAYLEIGH * _R0 * np.sqrt(_RHO / _P8)

# Surface tension must be positive, so it is driven to zero rather than set
# there. At 1e-12 N/m the Laplace pressure is ~1e-8 Pa against 1e5 Pa ambient.
_SIGMA = 1e-12
_INVISCID = imr_fast.PhysicalParameters(
  surface_tension_n_m=_SIGMA, far_field_pressure_pa=_P8, medium_density_kg_m3=_RHO
)

_SAMPLES = 40001
_WINDOW = 1.3 * _ANALYTIC
# argmin over a uniform grid cannot resolve the collapse time better than one
# spacing, which sets the floor these assertions can meaningfully test against.
_RESOLUTION = _WINDOW / (_SAMPLES - 1) / _ANALYTIC


def _collapse_time(ratio):
  config = imr_fast.SimulationConfig(
    R0=_R0, Req=_R0 * ratio, material=imr_fast.NoStress(), radial=1, physics=_INVISCID, rtol=1e-11, atol=1e-13
  )
  times = np.linspace(0.0, _WINDOW, _SAMPLES)
  radius = np.asarray(imr_fast.simulate(times, config).radius_ratio)
  return float(times[int(np.argmin(radius))])


def test_collapse_time_converges_to_rayleigh(measured):
  """Residual gas cushions the collapse, so the time is long and falls toward
  the analytic value as the cavity empties. Monotonicity is asserted as well as
  the endpoint: a solver that happened to sit near 0.9147 without converging
  toward it would pass a single-tolerance check."""
  ratios = (1.0 / 6.0, 0.1, 0.05, 0.02)
  errors = [abs(_collapse_time(ratio) - _ANALYTIC) / _ANALYTIC for ratio in ratios]

  measured("Rayleigh t_c", f"analytic={_ANALYTIC * 1e6:.4f}us  rel={' -> '.join(f'{e:.1e}' for e in errors)}")
  assert errors[0] > errors[1] > errors[2], "collapse time must converge as the gas content vanishes"
  assert errors[-1] < max(4.0 * _RESOLUTION, 1e-5), "did not reach the analytic value within grid resolution"


def test_gas_content_lengthens_the_collapse(measured):
  """Direction, not just magnitude. Gas resists compression, so a fuller bubble
  must collapse later than an emptier one -- and later than Rayleigh, never
  sooner. A sign error in the gas pressure term passes the convergence test
  above but fails this."""
  loose, tight = _collapse_time(1.0 / 6.0), _collapse_time(0.05)
  measured("gas cushioning", f"Req/R0=1/6: {loose * 1e6:.4f}us  Req/R0=0.05: {tight * 1e6:.4f}us")
  assert loose > tight > _ANALYTIC * (1.0 - _RESOLUTION)


def _trace(ratio, radial, sound_speed=None, window=60e-6, samples=4000):
  physics = (
    _INVISCID
    if sound_speed is None
    else imr_fast.PhysicalParameters(
      surface_tension_n_m=_SIGMA, far_field_pressure_pa=_P8, medium_density_kg_m3=_RHO, sound_speed_m_s=sound_speed
    )
  )
  config = imr_fast.SimulationConfig(
    R0=_R0, Req=_R0 * ratio, material=imr_fast.NoStress(), radial=radial, physics=physics, rtol=1e-12, atol=1e-14
  )
  return imr_fast.simulate(np.linspace(0.0, window, samples), config)


def _first_integral_residual(ratio, radial=1):
  """max |R^3 R'^2 - closed form| along the trajectory, relative to its scale."""
  result = _trace(ratio, radial=radial)
  radius = np.asarray(result.radius_ratio) * _R0
  velocity = np.asarray(result.wall_velocity_m_s)

  equilibrium = _R0 * ratio
  gas_at_r0 = (_P8 + 2.0 * _SIGMA / equilibrium) * (equilibrium / _R0) ** (3.0 * imr_fast.KAPPA)
  exponent = 3.0 - 3.0 * imr_fast.KAPPA
  gas = 2.0 * gas_at_r0 * _R0 ** (3.0 * imr_fast.KAPPA) / (_RHO * exponent) * (radius**exponent - _R0**exponent)
  ambient = 2.0 * _P8 / (3.0 * _RHO) * (radius**3 - _R0**3)

  actual = radius**3 * velocity**2
  return float(np.max(np.abs(actual - (gas - ambient)))) / float(np.max(np.abs(actual)))


@pytest.mark.parametrize("ratio", (1.0 / 6.0, 0.1, 0.3))
def test_rayleigh_plesset_conserves_its_first_integral(ratio, measured):
  """Pointwise, so it constrains the entire right-hand side rather than one
  scalar. A collapse-time check passes for any model that happens to arrive on
  schedule; this one does not.

  `wall_velocity_m_s` is the solver's own state, so no finite difference enters
  and the residual is bounded by integration tolerance rather than by a stencil.
  """
  residual = _first_integral_residual(ratio)
  measured(f"RP first integral Req/R0={ratio:.3f}", f"rel={residual:.2e}")
  assert residual < 1e-8


def test_keller_miksis_is_compressible_at_all(measured):
  """The converse of the test above. At a physical sound speed Keller-Miksis
  must *violate* the incompressible invariant, and by an O(1) amount. Without
  this, a KM branch that silently fell back to Rayleigh-Plesset would pass every
  other check in this module."""
  incompressible = _first_integral_residual(1.0 / 6.0, radial=1)
  compressible = _first_integral_residual(1.0 / 6.0, radial=2)
  measured("KM violates the RP invariant", f"RP rel={incompressible:.1e}  KM rel={compressible:.2f}")
  assert compressible > 0.1, "KM satisfies the incompressible invariant -- is it actually solving RP?"
  assert incompressible < 1e-8


def test_keller_miksis_approaches_rayleigh_plesset_as_first_order_in_one_over_c(measured):
  """`c -> inf` is the easy half. The order matters more: Keller-Miksis carries
  `O(1/c)` corrections, so a hundredfold rise in `c` must cut the gap a
  hundredfold. A correction attached at the wrong order still converges and
  would pass a limit-only check."""
  gaps = []
  for sound_speed in (1e7, 1e9):
    reference = np.asarray(_trace(1.0 / 6.0, radial=1, sound_speed=sound_speed).radius_ratio)
    compressible = np.asarray(_trace(1.0 / 6.0, radial=2, sound_speed=sound_speed).radius_ratio)
    gaps.append(float(np.max(np.abs(compressible - reference))))

  ratio = gaps[0] / gaps[1]
  measured("KM -> RP as 1/c", f"c=1e7: {gaps[0]:.2e}, c=1e9: {gaps[1]:.2e}, ratio={ratio:.1f} (first order = 100)")
  assert gaps[1] < gaps[0]
  assert 50.0 < ratio < 200.0, "gap must fall like 1/c, not faster or slower"


_BASE_PHYSICS = imr_fast.PhysicalParameters()
_T8 = 298.15  # SimulationConfig default far-field temperature, and the wall value when medtherm=0


def _polytropic_drift(conductivity_scale, exponent, samples=600):
  """Relative drift of `P * R^exponent`, which is constant for a polytropic gas."""
  physics = imr_fast.PhysicalParameters(
    gas_conductivity_slope=_BASE_PHYSICS.gas_conductivity_slope * conductivity_scale,
    gas_conductivity_offset=_BASE_PHYSICS.gas_conductivity_offset * conductivity_scale,
  )
  config = imr_fast.SimulationConfig(
    R0=_R0,
    Req=_R0 / 6,
    material=imr_fast.NoStress(),
    radial=1,
    bubtherm=1,
    Nt=25,
    physics=physics,
    rtol=1e-10,
    atol=1e-12,
  )
  result = imr_fast.simulate(np.linspace(0.0, 40e-6, samples), config)
  invariant = np.asarray(result.internal_pressure_pa) * np.asarray(result.radius_ratio) ** exponent
  return float(np.max(np.abs(invariant - invariant[0])) / abs(invariant[0]))


def test_thermal_pde_reduces_to_the_adiabatic_polytropic_law(measured):
  """Switch conduction off and the thermal PDE must reproduce `P R^(3*gamma) =
  const` -- a closed form, not another code's output.

  This is the only external anchor the thermal model has. `validate_bubtherm_
  adiabatic.py` checks the same reduction by forcing `chi = 0` through the
  internal RHS; this reaches the limit through the public API instead, so it
  tests the model a user can actually configure.
  """
  drift = _polytropic_drift(1e-8, 3.0 * imr_fast.KAPPA)
  measured("thermal -> adiabatic", f"P*R^(3g) drift={drift:.2e}")
  assert drift < 1e-3


def test_conduction_actually_breaks_the_adiabatic_invariant(measured):
  """The converse. At physical conductivity the gas exchanges heat with the
  wall, so `P R^(3*gamma)` must NOT hold. Without this, a thermal branch whose
  conduction term was inert would pass the test above perfectly."""
  scales = (1e-8, 1e-4, 1e-2, 1.0)
  drifts = [_polytropic_drift(scale, 3.0 * imr_fast.KAPPA) for scale in scales]
  measured("conduction breaks it", " -> ".join(f"{d:.1e}" for d in drifts))
  assert all(a < b for a, b in zip(drifts[:-1], drifts[1:], strict=True)), "drift must grow with conductivity"
  assert drifts[-1] > 0.1, "conduction at physical strength barely moved the invariant -- is it inert?"


def _scaled_conductivity(scale):
  """Scale gas AND vapour conductivity together.

  `K8` averages the two, and `alpha_g = gas_slope * T8 / K8`. Scaling only the
  gas terms therefore changes `alpha_g` as a side effect -- and since `theta` is
  a Kirchhoff transform of a conductivity linear in T, the temperature it can
  represent is bounded below by `1 - 1/alpha_g`. Raising `alpha_g` lifts that
  floor into the physical range and the solve fails on a negative square root,
  which looks exactly like a thermal-model breakdown and is not one.

  Scaling both keeps `alpha_g` fixed and moves only the magnitude `chi`.
  """
  return imr_fast.PhysicalParameters(
    gas_conductivity_slope=_BASE_PHYSICS.gas_conductivity_slope * scale,
    gas_conductivity_offset=_BASE_PHYSICS.gas_conductivity_offset * scale,
    vapor_conductivity_slope=_BASE_PHYSICS.vapor_conductivity_slope * scale,
    vapor_conductivity_offset=_BASE_PHYSICS.vapor_conductivity_offset * scale,
  )


def _isothermal_state(scale, samples=600):
  config = imr_fast.SimulationConfig(
    R0=_R0,
    Req=_R0 / 6,
    material=imr_fast.NoStress(),
    radial=1,
    bubtherm=1,
    Nt=25,
    physics=_scaled_conductivity(scale),
    rtol=1e-10,
    atol=1e-12,
  )
  result = imr_fast.simulate(np.linspace(0.0, 40e-6, samples), config)
  invariant = np.asarray(result.internal_pressure_pa) * np.asarray(result.radius_ratio) ** 3
  temperature = np.asarray(result.bubble_temperature_k)
  drift = float(np.max(np.abs(invariant - invariant[0])) / abs(invariant[0]))
  return drift, float(np.min(temperature)), float(np.max(temperature))


def test_thermal_pde_reduces_to_the_isothermal_polytropic_law(measured):
  """The other limit. Fast conduction pins the gas at the wall temperature, so
  `P R^3 = const` -- and the check is not only the invariant but that the gas
  temperature actually collapses onto `T8`. An invariant can be satisfied for
  the wrong reason; a bubble sitting at 298 K cannot."""
  drift, coldest, hottest = _isothermal_state(1e5)
  measured("thermal -> isothermal", f"P*R^3 drift={drift:.2e}  T in [{coldest:.1f}, {hottest:.1f}] K")
  assert drift < 5e-2
  assert abs(coldest - _T8) < 10.0 and abs(hottest - _T8) < 10.0


def test_conduction_carries_the_gas_between_both_limits(measured):
  """The two limits are endpoints of one knob. Raising `chi` must move the gas
  monotonically off adiabatic and onto isothermal -- neither invariant holding
  at both ends, and the temperature range narrowing onto the wall."""
  scales = (1e0, 1e2, 1e5)
  drifts = [_isothermal_state(scale) for scale in scales]
  isothermal = [row[0] for row in drifts]
  spread = [row[2] - row[1] for row in drifts]
  measured(
    "chi sweeps adiabatic -> isothermal",
    f"P*R^3 drift {' -> '.join(f'{d:.1e}' for d in isothermal)}; T spread {' -> '.join(f'{s:.0f}K' for s in spread)}",
  )
  assert all(a > b for a, b in zip(isothermal[:-1], isothermal[1:], strict=True))
  assert all(a > b for a, b in zip(spread[:-1], spread[1:], strict=True))
