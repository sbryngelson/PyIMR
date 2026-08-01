"""Problem preparation: nondimensionalisation, grids, and the collapse precursor."""

from __future__ import annotations

from typing import Any, Callable

import copy
from types import MappingProxyType

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

from ._config import (
  CollapseStats,
  MediumOperators,
  PhysicalParameters,
  PreparedDistributedStress,
  PreparedForcing,
  PreparedInstantaneousMaterial,
  PreparedProblem,
  SimulationConfig,
  SimulationError,
  StateLayout,
  _freeze_array,
)
from ._materials import (
  Giesekus,
  InstantaneousMaterial,
  LinearPTT,
  NeoHookeanKelvinVoigt,
  NoStress,
  OldroydB,
  QuadraticKelvinVoigt,
  QuadraticZener,
  Zener,
  _is_distributed_stress,
  _stress_state_count,
)
from ._rhs import _rhs
import imr_fast as _solver
from ._arrays import at_set
from ._thermal import _far_field_singular_index, _mie_F, _mu_of_A, kirchhoff_theta, mixture_kirchhoff, pvsat
from .thermal_fd import finite_diff_mat
from .thermal_spectral import chebyshev_diff_mat
from .thermal_spectral import nodes as chebyshev_nodes

__all__ = [
  "_collapse_memory_state",
  "_collapse_zener_rhs",
  "_material_scales",
  "_prepare_distributed_stress",
  "_prepare_forcing",
  "_prepare_instantaneous_material",
  "_thermal_state",
  "params",
  "prepare",
]

def _material_scales(material):
  if isinstance(material, NoStress): return 0.0, 0.0, 0.0, 0.0, 0.0
  if isinstance(material, (NeoHookeanKelvinVoigt, QuadraticKelvinVoigt, Zener, QuadraticZener)):
    modulus = material.shear_modulus_pa
  else:
    modulus = 0.0
  if isinstance(material, (NeoHookeanKelvinVoigt, QuadraticKelvinVoigt, Zener, QuadraticZener, OldroydB, Giesekus, LinearPTT)):
    viscosity = material.viscosity_pa_s
  else:
    viscosity = 0.0
  if isinstance(material, (Zener, QuadraticZener, OldroydB, Giesekus, LinearPTT)):
    relaxation = material.relaxation_time_s
    retardation = material.retardation_time_s
  else:
    relaxation = 0.0
    retardation = 0.0
  stiffening = material.stiffening if isinstance(material, (QuadraticKelvinVoigt, QuadraticZener)) else 0.0
  return modulus, viscosity, relaxation, retardation, stiffening

def params(R0, Req, material, vapor=0, T8=298.15, pA=0.0, omega=0.0, TW=0.0, DT=0.0, mn=0.0, wave_type=0, bubtherm=0, masstrans=0, physics=None, *, xp=np, scales=None):
  physics = PhysicalParameters() if physics is None else physics
  P8_value = physics.far_field_pressure_pa
  density = physics.medium_density_kg_m3
  surface_tension = physics.surface_tension_n_m
  kappa = physics.polytropic_exponent
  Uc = xp.sqrt(P8_value / density)
  t0 = R0 / Uc
  concrete = _material_scales(material)
  G, mu, lam1, lam2, alphax = concrete if scales is None else scales
  relaxing = concrete[2] > 0.0
  Ca = P8_value / G if concrete[0] > 0 else xp.inf
  Re8 = P8_value * R0 / (mu * Uc) if concrete[1] > 0 else xp.inf
  We = P8_value * R0 / (2 * surface_tension)
  Pv = vapor * pvsat(T8, xp=xp)
  P0_exp = 3 if bubtherm else 3 * kappa
  P0 = (P8_value + 2 * surface_tension / Req - Pv) * (Req / R0) ** P0_exp
  Pv_star = Pv / P8_value
  Pb = P0 / P8_value + Pv_star
  K8 = 0.5 * (
    physics.gas_conductivity_slope * T8 + physics.gas_conductivity_offset + physics.vapor_conductivity_slope * T8 + physics.vapor_conductivity_offset
  )
  chi = T8 * K8 / (P8_value * R0 * Uc)
  alpha_g = physics.gas_conductivity_slope * T8 / K8
  beta_g = physics.gas_conductivity_offset / K8
  alpha_v = physics.vapor_conductivity_slope * T8 / K8
  beta_v = physics.vapor_conductivity_offset / K8
  Dm = physics.medium_conductivity_w_m_k / (density * physics.medium_specific_heat_j_kg_k)
  Foh = Dm / (Uc * R0)
  iota = physics.medium_conductivity_w_m_k / (K8 * physics.medium_grid_length)
  Br = Uc**2 / (physics.medium_specific_heat_j_kg_k * T8)
  Rv = physics.universal_gas_constant_j_mol_k / physics.vapor_molar_mass_kg_mol
  Rg = physics.universal_gas_constant_j_mol_k / physics.gas_molar_mass_kg_mol
  Rnondim = P8_value / (density * T8)
  Rv_star = Rv / Rnondim
  Rg_star = Rg / Rnondim
  Fom = physics.mass_diffusivity_m2_s / (Uc * R0)
  L_heat_star = physics.latent_heat_j_kg / Uc**2
  if masstrans and vapor == 0: raise ValueError("masstrans=1 requires vapor=1 (kv0's formula is only physically meaningful with Pv_star>0)")
  kv0 = 1.0 / (1.0 + (Rv_star / Rg_star) * (Pb / Pv_star - 1.0)) if vapor else 0.0
  tait_gamma = physics.tait_pressure_pa / P8_value
  tait_sam = 1.0 + tait_gamma
  tait_no = (physics.tait_exponent - 1.0) / physics.tait_exponent
  Cstar = physics.sound_speed_m_s / Uc
  nog = (physics.tait_exponent - 1.0) / 2.0
  mie_reference = _mie_F(_mu_of_A(1.0 / Cstar**2, physics.hugoniot_slope, nog, xp=xp), physics.hugoniot_slope, nog, xp=xp)
  return dict(
    t0=t0,
    Uc=Uc,
    P8=P8_value,
    viscosity_scale=P8_value * R0 / Uc,
    Ca=Ca,
    Re8=Re8,
    iWe=1.0 / We,
    req=Req / R0,
    Pb=Pb,
    Pv=Pv_star,
    T8=T8,
    kappa=kappa,
    kapover=(kappa - 1.0) / kappa,
    De=(lam1 * Uc / R0 if relaxing else 0.0),
    LAM=(lam2 / lam1 if relaxing else 0.0),
    Cstar=Cstar,
    alphax=alphax,
    tait_gamma=tait_gamma,
    tait_sam=tait_sam,
    tait_no=tait_no,
    tait_exponent=physics.tait_exponent,
    hugoniot_slope=physics.hugoniot_slope,
    nog=nog,
    mie_reference=mie_reference,
    ee=pA / P8_value,
    om=omega * t0,
    tw=TW / t0,
    dt=DT / t0,
    mn=mn,
    wave_type=wave_type,
    chi=chi,
    alpha_g=alpha_g,
    beta_g=beta_g,
    alpha_v=alpha_v,
    beta_v=beta_v,
    Foh=Foh,
    iota=iota,
    Br=Br,
    Lt=physics.medium_grid_length,
    Rv_star=Rv_star,
    Rg_star=Rg_star,
    Fom=Fom,
    L_heat_star=L_heat_star,
    kv0=kv0,
  )

def _prepare_forcing(config, parameters):
  if config.sampled_forcing is None: return None
  forcing = config.sampled_forcing
  knots = np.asarray(forcing.time_s) / parameters["t0"]
  values = np.asarray(forcing.pressure_pa) / parameters["P8"]
  interpolant = PchipInterpolator(knots, values, extrapolate=False)
  return PreparedForcing(knots=_freeze_array(knots), coefficients=_freeze_array(interpolant.c))

def _prepare_instantaneous_material(material):
  if not isinstance(material, InstantaneousMaterial): return None
  nodes, weights = np.polynomial.legendre.leggauss(material.quadrature_points)
  return PreparedInstantaneousMaterial(interval_nodes=_freeze_array(nodes), interval_weights=_freeze_array(weights))

def _prepare_distributed_stress(material):
  """Lagrangian grid for the distributed stress, plus its quadrature weights."""
  if not _is_distributed_stress(material): return None
  span = material.extent - 1.0
  if material.quadrature == "gauss":
    nodes, weights = np.polynomial.legendre.leggauss(material.points)
    unit_grid = 0.5 * (nodes + 1.0)
    geometric = 0.5 * weights * 4.0 * span * unit_grid**3
  else:
    unit_grid = np.linspace(0.0, 1.0, material.points)
    geometric = None
  reference_radius = 1.0 + span * unit_grid**4
  return PreparedDistributedStress(
    reference_radius=_freeze_array(reference_radius),
    reference_radius_cubed=_freeze_array(reference_radius**3),
    weights=None if geometric is None else _freeze_array(geometric * reference_radius**2),
  )

def _thermal_state(temperature_ratio, alpha, beta): return kirchhoff_theta(temperature_ratio, alpha, beta)

def medium_with_parameters(medium, p, *, xp=np):
  """The medium's parameter-dependent wall weights, rebuilt for a new `p`."""
  if medium is None: return None
  updated = copy.copy(medium)
  bubble, med = xp.asarray(medium.bubble_wall_stencil), xp.asarray(medium.medium_wall_stencil)
  for name, value in (
    ("grad_Tm", 2.0 * p["chi"] * p["iota"] * med),
    ("grad_Trans", p["chi"] * bubble),
    ("grad_C", p["Fom"] * p["L_heat_star"] * bubble),
  ):
    object.__setattr__(updated, name, value)
  return updated

def forcing_with_parameters(forcing, p, reference, *, xp=np):
  """A prepared sampled forcing, rescaled for a new `p`."""
  if forcing is None: return None
  t0_reference, pressure_reference = reference
  knots = xp.asarray(forcing.knots) * (t0_reference / p["t0"])
  prepared = xp.asarray(forcing.coefficients)
  rows = [
    prepared[row] * ((pressure_reference / t0_reference**degree) * (p["t0"] ** degree / p["P8"]))
    for row, degree in enumerate((3, 2, 1, 0))
  ]
  return _solver.PreparedForcing(knots=knots, coefficients=xp.stack(rows) if hasattr(xp, "stack") else np.array(rows))

def initial_state_vector(config, layout, p, collapse_state, *, xp=np, initial=None):
  """The state the solve starts from, in whichever arithmetic `p` is built in."""
  initial = config.initial if initial is None else initial
  state = xp.zeros(layout.size)
  state = at_set(state, 0, 1.0)
  state = at_set(state, 1, initial.wall_velocity_m_s / p["Uc"])
  if collapse_state is not None:
    state = at_set(state, layout.stress, xp.asarray(collapse_state))
  elif initial.stress_state is not None:
    state = at_set(state, layout.stress, xp.asarray(initial.stress_state))
  if not config.bubtherm: return state
  state = at_set(state, layout.pressure, p["Pb"])
  vapor_fraction = p["kv0"] if initial.vapor_mass_fraction is None else initial.vapor_mass_fraction
  if config.masstrans: state = at_set(state, layout.vapor_fraction, vapor_fraction)
  temperature_ratio = 1.0 if initial.bubble_temperature_k is None else initial.bubble_temperature_k / config.T8
  alpha, beta = mixture_kirchhoff(vapor_fraction, p, config.masstrans)
  state = at_set(state, layout.bubble_thermal, _thermal_state(temperature_ratio, alpha, beta))
  if config.medtherm:
    ratio = 1.0 if initial.medium_temperature_k is None else initial.medium_temperature_k / config.T8
    state = at_set(state, layout.medium_thermal, ratio)
  return state

def _collapse_zener_rhs(state, p):
  radius, velocity, stress = state
  equilibrium = p["req"]
  pressure_prefactor = 1.0 - p["Pv"] + p["iWe"] / equilibrium
  pressure = p["Pv"] + pressure_prefactor * (equilibrium / radius) ** 3
  pressure_rate = -3.0 * pressure_prefactor * (equilibrium / radius) ** 3 * velocity / radius
  stress_rate = (
    -4.0 * velocity / (p["Re8"] * radius)
    - 2.0 * p["De"] / p["Ca"] * velocity / radius * ((equilibrium / radius) ** 4 + equilibrium / radius)
    - (stress + 0.5 / p["Ca"] * (5.0 - (equilibrium / radius) ** 4 - 4.0 * equilibrium / radius))
  ) / p["De"]
  sound = p["Cstar"]
  numerator = (
    (1.0 + velocity / sound) * (pressure - 1.0 + stress - p["iWe"] / radius)
    + radius / sound * (pressure_rate + p["iWe"] * velocity / radius**2 + stress_rate)
    - 1.5 * (1.0 - velocity / (3.0 * sound)) * velocity**2
  )
  denominator = (1.0 - velocity / sound) * radius + 4.0 / (p["Re8"] * sound)
  return velocity, numerator / denominator, stress_rate

def _collapse_memory_state(config, instantaneous_material, distributed_stress):
  settings = config.collapse
  if settings is None: return None, None
  precursor = params(config.R0, config.Req, config.material, config.vapor, config.T8, physics=config.physics, bubtherm=1, masstrans=config.masstrans)
  precursor["kappa"] = 1.0
  state_width = _stress_state_count(config.material)
  equilibrium_radius = precursor["req"]
  integration_evaluations = 0
  shooting_evaluations = 0
  upstream_zener = isinstance(config.material, Zener)

  def maximum_event(_time, state): return state[1]

  maximum_event.terminal = True
  maximum_event.direction = -1

  def integrate(initial_velocity):
    nonlocal integration_evaluations
    initial = np.zeros(2 + state_width)
    initial[0] = equilibrium_radius
    initial[1] = initial_velocity

    def production_rhs(time, state):
      return _rhs(
        time, state, precursor, config.material, config.radial, instantaneous_material=instantaneous_material, distributed_stress=distributed_stress
      )

    def zener_precursor_rhs(_time, state): return _collapse_zener_rhs(state, precursor)

    # same static type: `_rhs` returns a list on the mechanical path and an array
    collapse_rhs: Callable[..., Any] = zener_precursor_rhs if upstream_zener else production_rhs
    solution = solve_ivp(
      collapse_rhs,
      (0.0, settings.maximum_time_nondimensional),
      initial,
      events=maximum_event,
      method="LSODA",
      rtol=min(config.rtol, 1e-9),
      atol=min(config.atol, 1e-11),
    )
    integration_evaluations += int(solution.nfev)
    if not solution.success: raise SimulationError(f"collapse precursor integration failed: {solution.message}")
    if solution.t_events[0].size == 0:
      raise SimulationError(f"collapse precursor did not reach a maximum radius within t={settings.maximum_time_nondimensional:g}")
    return solution.y_events[0][-1], float(solution.t_events[0][-1])

  def residual(initial_velocity):
    nonlocal shooting_evaluations
    shooting_evaluations += 1
    return integrate(initial_velocity)[0][0] - 1.0

  lower_velocity = max(settings.initial_velocity_guess * 1e-8, np.finfo(float).eps)
  lower_residual = residual(lower_velocity)
  if lower_residual >= 0.0: raise SimulationError("collapse precursor equilibrium radius is not below the observed maximum radius")
  upper_velocity = settings.initial_velocity_guess
  upper_residual = residual(upper_velocity)
  expansions = 0
  while upper_residual < 0.0 and expansions < settings.maximum_bracket_expansions:
    upper_velocity *= 2.0
    upper_residual = residual(upper_velocity)
    expansions += 1
  if upper_residual < 0.0:
    raise SimulationError(f"collapse shooting could not bracket an initial velocity after {settings.maximum_bracket_expansions} expansions")
  initial_velocity = brentq(
    residual, lower_velocity, upper_velocity, xtol=settings.radius_tolerance, rtol=max(settings.radius_tolerance, 4.0 * np.finfo(float).eps)
  )
  maximum_state, maximum_time = integrate(initial_velocity)
  memory_state = _freeze_array(maximum_state[2:])
  stats = CollapseStats(
    initial_velocity_nondimensional=float(initial_velocity),
    maximum_time_nondimensional=maximum_time,
    maximum_radius_ratio=float(maximum_state[0]),
    shooting_evaluations=shooting_evaluations,
    integration_evaluations=integration_evaluations,
    stress_state=memory_state,
  )
  return memory_state, stats

def prepare(config: SimulationConfig) -> PreparedProblem:
  """Prepare reusable grids, operators, parameters, and state layout."""
  if not isinstance(config, SimulationConfig): raise TypeError("config must be a SimulationConfig")
  p = params(
    config.R0,
    config.Req,
    config.material,
    config.vapor,
    config.T8,
    config.pA,
    config.omega,
    config.TW,
    config.DT,
    config.mn,
    config.wave_type,
    config.bubtherm,
    config.masstrans,
    config.physics,
  )
  instantaneous_material = _prepare_instantaneous_material(config.material)
  distributed_stress = _prepare_distributed_stress(config.material)
  collapse_state, collapse_stats = _collapse_memory_state(config, instantaneous_material, distributed_stress)
  initial = config.initial
  if initial.internal_pressure_pa is not None: p["Pb"] = initial.internal_pressure_pa / p["P8"]
  layout = StateLayout.from_config(config)
  if initial.bubble_temperature_k is not None and not config.bubtherm: raise ValueError("initial bubble temperature requires bubtherm=1")
  if initial.medium_temperature_k is not None and not config.medtherm: raise ValueError("initial medium temperature requires medtherm=1")
  if initial.vapor_mass_fraction is not None and not config.masstrans: raise ValueError("initial vapor mass fraction requires masstrans=1")
  # saturation ties these two. Setting one alone leaves the vapour out of equilibrium with the
  # temperature it is supposed to be saturated at -- kv0 is saturation at T8, so a bubble
  # temperature of 300 K against T8=298.15 K starts 0.144 off. Demand both so the choice is
  # deliberate rather than inherited from a default (#133).
  if config.masstrans and (initial.bubble_temperature_k is None) != (initial.vapor_mass_fraction is None):
    raise ValueError(
      "masstrans=1 couples initial.bubble_temperature_k and initial.vapor_mass_fraction through saturation; "
      "set both or neither"
    )
  stress_width = layout.stress.stop - layout.stress.start
  if initial.stress_state is not None and len(initial.stress_state) != stress_width:
    raise ValueError(f"initial stress_state requires exactly {stress_width} values")
  initial_state = initial_state_vector(config, layout, p, collapse_state)
  bubble_grid = None
  bubble_D1 = None
  bubble_D2 = None
  medium = None
  if config.bubtherm:
    spectral = config.thermal == "spectral"
    _diff = chebyshev_diff_mat if spectral else finite_diff_mat
    bubble_y = chebyshev_nodes(config.Nt, 0) if spectral else np.linspace(0.0, 1.0, config.Nt)
    bubble_first = _diff(config.Nt, 1, 0)
    bubble_grid = _freeze_array(bubble_y)
    bubble_D1 = _freeze_array(bubble_first)
    bubble_D2 = _freeze_array(_diff(config.Nt, 2, 0))
    if config.medtherm:
      Nm = config.Mt - 1
      deltaYm = -2.0 / Nm
      xi = chebyshev_nodes(config.Mt, 1) if spectral else np.linspace(1.0, -1.0, config.Mt)
      _far_field_singular_index(xi)
      stretched = np.empty_like(xi)
      stretched[:-1] = 2.0 / (xi[:-1] + 1.0)
      stretched[-1] = np.inf
      yT = (stretched - 1.0) * p["Lt"] + 1.0
      yT2, yT3 = yT**2, yT**3
      iyT3, iyT4, iyT6 = yT**-3, yT**-4, yT**-6
      coeff = np.array([-1.5, 2.0, -0.5])
      deltaY = 1.0 / (config.Nt - 1)
      medium_first = _diff(config.Mt, 1, 1)

      def _pad(values, length):
        padded = np.zeros(length)
        padded[: values.size] = values
        return padded

      bubble_wall_stencil = bubble_first[-1, ::-1] if spectral else _pad(-coeff / deltaY, config.Nt)
      medium_wall_stencil = medium_first[0] if spectral else _pad(coeff / deltaYm, config.Mt)
      medium = MediumOperators(
        xi=_freeze_array(xi),
        yT=_freeze_array(yT),
        yT2=_freeze_array(yT2),
        yT3=_freeze_array(yT3),
        iyT3=_freeze_array(iyT3),
        iyT4=_freeze_array(iyT4),
        iyT6=_freeze_array(iyT6),
        D1=_freeze_array(medium_first),
        D2=_freeze_array(_diff(config.Mt, 2, 1)),
        grad_Tm=_freeze_array(
          2 * p["chi"] * p["iota"] * medium_first[0] if spectral else _pad(2 * p["chi"] * p["iota"] / deltaYm * coeff, config.Mt)
        ),
        grad_Trans=_freeze_array(p["chi"] * bubble_first[-1, ::-1] if spectral else _pad(-coeff * p["chi"] / deltaY, config.Nt)),
        grad_C=_freeze_array(
          p["Fom"] * p["L_heat_star"] * bubble_first[-1, ::-1] if spectral else _pad(-coeff * p["Fom"] * p["L_heat_star"] / deltaY, config.Nt)
        ),
        bubble_wall_stencil=_freeze_array(bubble_wall_stencil),
        medium_wall_stencil=_freeze_array(medium_wall_stencil),
      )
  return PreparedProblem(
    config=config,
    parameters=MappingProxyType(p),
    layout=layout,
    initial_state=_freeze_array(initial_state),
    bubble_grid=bubble_grid,
    bubble_D1=bubble_D1,
    bubble_D2=bubble_D2,
    medium=medium,
    forcing=_prepare_forcing(config, p),
    instantaneous_material=instantaneous_material,
    distributed_stress=distributed_stress,
    collapse_stats=collapse_stats,
  )
