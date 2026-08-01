"""Fast, validated solvers for inertial microcavitation rheometry."""

from __future__ import annotations


from dataclasses import replace

import numpy as np


__all__ = [
  "ArrudaBoyce",
  "Bingham",
  "C8",
  "CarreauYasuda",
  "CollapseInitialization",
  "CollapseStats",
  "Cross",
  "ElasticModel",
  "Fung",
  "Gent",
  "Giesekus",
  "HerschelBulkley",
  "InitialState",
  "InstantaneousMaterial",
  "KAPPA",
  "LinearPTT",
  "MaterialModel",
  "ModifiedPowellEyring",
  "MooneyRivlin",
  "NeoHookean",
  "NeoHookeanKelvinVoigt",
  "Newtonian",
  "NoStress",
  "Ogden",
  "OldroydB",
  "P8",
  "PhysicalParameters",
  "PowellEyring",
  "PowerLaw",
  "PreparedProblem",
  "QuadraticKelvinVoigt",
  "QuadraticZener",
  "RHO",
  "SURF",
  "SampledForcing",
  "SimulationConfig",
  "SimulationError",
  "SimulationResult",
  "SolverStats",
  "ViscousModel",
  "Yeoh",
  "Zener",
  "prepare",
  "simulate",
  "simulate_with_sensitivities",
]

from ._integrate import integrate as _integrate
from ._config import (  # noqa: F401
  C8,
  CollapseInitialization,
  CollapseStats,
  InitialState,
  KAPPA,
  MediumOperators,
  P8,
  PhysicalParameters,
  PreparedDistributedStress,
  PreparedForcing,
  PreparedInstantaneousMaterial,
  PreparedProblem,
  RHO,
  SURF,
  SampledForcing,
  SimulationConfig,
  SimulationError,
  SimulationResult,
  SolverStats,
  StateLayout,
  _CP,
  _D0,
  _KM,
  _LHEAT,
  _LT,
  _MWG,
  _MWV,
  _RU,
  _freeze_array,
  _readonly_float_array,
  _readonly_optional,
  _validate_inputs,
)

from ._thermal import (  # noqa: F401
  _GAM_TAIT,
  _HUGONIOT_S,
  _NOG,
  _NSTATE_TAIT,
  _apply_thermal_boundaries,
  _dissipation,
  _distributed_dissipation,
  _instantaneous_dissipation,
  _kv_of_T,
  _mie_F,
  _mie_gruneisen,
  _mu_of_A,
  _T_of_kv,
  _bracketed_root,
  _wall_theta_bw,
  _wall_theta_bw_full,
  pvsat,
)


from ._materials import (  # noqa: F401
  _is_distributed_stress,
  _stress_state_count,
  ArrudaBoyce,
  Bingham,
  CarreauYasuda,
  Cross,
  ElasticModel,
  Fung,
  Gent,
  Giesekus,
  HerschelBulkley,
  InstantaneousMaterial,
  LinearPTT,
  MaterialModel,
  ModifiedPowellEyring,
  MooneyRivlin,
  NeoHookean,
  NeoHookeanKelvinVoigt,
  Ogden,
  Newtonian,
  NoStress,
  OldroydB,
  PowellEyring,
  PowerLaw,
  QuadraticKelvinVoigt,
  QuadraticZener,
  ViscousModel,
  Yeoh,
  Zener,
  _finite_nonnegative,
  _finite_positive,
)

from ._stress import (  # noqa: F401
  _MaterialDomainError,
  _PE_SERIES_LIMIT,
  _distributed_stress,
  _distributed_stress_integral,
  _elastic_integrand,
  _instantaneous_stress,
  _powell_eyring_terms,
  _stress,
  _viscosity_and_tangent,
)

from ._prepare import (  # noqa: F401
  _collapse_memory_state,
  _collapse_zener_rhs,
  _material_scales,
  _prepare_distributed_stress,
  _prepare_forcing,
  _prepare_instantaneous_material,
  _thermal_state,
  params,
  prepare,
)

from ._rhs import (  # noqa: F401
  _nZ,
  _pinf,
  _rhs,
  _rhs_args,
  _sampled_pressure,
)


def _thermal_outputs(problem: PreparedProblem, states: np.ndarray):
  config = problem.config
  if not config.bubtherm: return None, None, None
  layout = problem.layout
  p = problem.parameters
  count = states.shape[0]
  bubble_temperature = np.empty((count, config.Nt))
  medium_temperature = np.empty((count, config.Mt)) if config.medtherm else None
  vapor_fraction = np.empty((count, config.Nt)) if config.masstrans else None
  for index, state in enumerate(states):
    theta = state[layout.bubble_thermal].copy()
    Tm = state[layout.medium_thermal].copy() if layout.medium_thermal is not None else None
    kv = state[layout.vapor_fraction].copy() if layout.vapor_fraction is not None else None
    *_, temperature, _ = _apply_thermal_boundaries(theta, Tm, kv, state[layout.pressure], p, problem.medium, config.masstrans)
    bubble_temperature[index] = config.T8 * temperature
    if medium_temperature is not None and Tm is not None: medium_temperature[index] = config.T8 * Tm
    if vapor_fraction is not None: vapor_fraction[index] = kv
  return bubble_temperature, medium_temperature, vapor_fraction

def _build_result(problem: PreparedProblem, time_s: np.ndarray, states, stats: SolverStats) -> SimulationResult:
  config = problem.config
  p = problem.parameters
  layout = problem.layout
  states = np.asarray(states).T
  radius_ratio = states[:, 0]
  velocity = states[:, 1]
  Uc = p["Uc"]
  pressure_scale = p["P8"]
  kappa = p["kappa"]
  if layout.pressure is None:
    pressure = (p["Pb"] - p["Pv"]) * radius_ratio ** (-3 * kappa) + p["Pv"]
  else:
    pressure = states[:, layout.pressure]
  stress_integral = np.empty(states.shape[0])
  for index, state in enumerate(states):
    stress_state = state[layout.stress] if layout.stress.stop > layout.stress.start else None
    if problem.distributed_stress is None:
      stress_integral[index] = _stress(config.material, p, state[0], state[1], stress_state, problem.instantaneous_material, False)[0]
    else:
      stress_integral[index] = _distributed_stress_integral(problem.distributed_stress, p, state[0], state[1], stress_state)
  bubble_temperature, medium_temperature, vapor_fraction = _thermal_outputs(problem, states)
  internal_stress_state = states[:, layout.stress] if layout.stress.stop > layout.stress.start else None
  return SimulationResult(
    time_s=_readonly_float_array(time_s),
    radius_ratio=_readonly_float_array(radius_ratio),
    wall_velocity_m_s=_readonly_float_array(Uc * velocity),
    internal_pressure_pa=_readonly_float_array(pressure_scale * pressure),
    stress_integral_pa=_readonly_float_array(pressure_scale * stress_integral),
    bubble_temperature_k=_readonly_optional(bubble_temperature),
    medium_temperature_k=_readonly_optional(medium_temperature),
    vapor_mass_fraction=_readonly_optional(vapor_fraction),
    stress_state=_readonly_optional(internal_stress_state),
    stress_reference_radius_ratio=(problem.distributed_stress.reference_radius if problem.distributed_stress is not None else None),
    stats=stats,
    config=config,
  )

def _integrate_prepared(problem: PreparedProblem, tv):
  config = problem.config
  time_s = _validate_inputs(tv, config)
  p = problem.parameters
  tn = time_s / p["t0"]
  args = _rhs_args(problem, p, medium=problem.medium)
  states, stats = _integrate(
    _rhs, tn, problem.initial_state, args=args,
    rtol=config.rtol, atol=config.atol, failure="IMR integration failed", config=config,
    max_step=None if config.max_step_s is None else config.max_step_s / p["t0"],
  )
  return time_s, states, stats

def _solve_prepared(problem: PreparedProblem, tv) -> SimulationResult:
  time_s, states, stats = _integrate_prepared(problem, tv)
  try:
    return _build_result(problem, time_s, states, stats)
  except _MaterialDomainError as error:
    raise SimulationError(f"IMR integration failed: {error}", replace(stats, success=False, message=str(error))) from error

def simulate(tv, config: SimulationConfig) -> SimulationResult:
  """Run one simulation and return immutable physical histories."""
  return prepare(config).solve(tv)

def simulate_with_sensitivities(tv, config: SimulationConfig, parameters):
  """Run one simulation with forward sensitivities."""
  return prepare(config).solve_with_sensitivities(tv, parameters)
