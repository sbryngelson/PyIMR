"""Fast Python IMR solver -- a validated slice of IMRv2.

Covers:
  radial   1 (Rayleigh-Plesset), 2 (Keller-Miksis, pressure form),
           3 (Keller-Miksis, enthalpy form, Tait EoS liquid),
           4 (Gilmore, Tait EoS liquid),
           5 (Keller-Miksis, enthalpy form, Mie-Gruneisen EoS liquid)
           radial=6 (Gilmore, Mie-Gruneisen EoS) NOT supported -- confirmed
           dead/broken upstream (goes complex in real IMRv2 for essentially
           every polytropic-gas P, mild or extreme; f_radial_eq.m evaluates
           the EoS at raw P with no pressure-reference correction, unlike
           radial=5's Pb=P-iWe/R)
  bubtherm 0 (polytropic, kappa) or 1 (gas thermal PDE)
           medtherm 0 or 1 (liquid boundary layer, requires bubtherm=1)
           masstrans 0 or 1 (vapor mass transfer, requires bubtherm=1 and
                             vapor=1; may be combined with medtherm=1)
  material closed-form NHKV, quadratic KV, Zener, quadratic Zener, Oldroyd-B;
           composable neo-Hookean, Mooney-Rivlin, Yeoh, Fung, Gent, or
           Arruda-Boyce elasticity with Newtonian, power-law, Carreau-Yasuda,
           Cross, Herschel-Bulkley, or Bingham viscosity; and distributed
           Giesekus or linear PTT memory
  forcing  wave_type 0 (constant offset), 1 (Gaussian), 2 (histotripsy),
           3 (Heaviside step), or a dimensional sampled pressure history

Equations transcribed from IMRv2/src/{f_radial_eq,f_stress,f_call_params,
f_imr_fd}.m. Validated against IMRv2 reference trajectories -- see
tests/run_validation.py.

The default physical constants match the pinned reference trajectories.
They are configurable through PhysicalParameters; IMRv2's own shipped
polytropic exponent is 1.47, whereas this solver defaults to 1.4 because that
is the value used to generate its reference data.

bubtherm=1 implements IMRv2's "elseif bubtherm" branch (f_imr_fd.m): gas-phase
thermal PDE, dry gas (kv0=0, vapor=0). With medtherm=0, the wall is an
isothermal-equivalent clamp (thetadot[-1]=0). Its Pdot uses bare P (kappa*P),
NOT (P-Pv) -- this is IMRv2's actual equation for this branch, not a
simplification; the bubtherm=0 polytropic branch's Pdot uses (P-Pv) and the
two are NOT reconciled/harmonized, since they are genuinely different
equations in the source.

medtherm=1 (requires bubtherm=1) adds the liquid boundary layer: a stretched
exterior grid (Mt points, Lt controls the stretching), advection+diffusion+
viscous-dissipation RHS for Tm, and the wall temperature theta[-1] is NOT a
free state -- it is solved every RHS call via a 1-D root-find (scipy.optimize
style secant iteration; f_bubble_wall_thermal_bc) enforcing heat-flux
continuity across the interface. A warm-start value is scoped to one
integration, matching IMRv2's continuation behavior without leaking state
between repeated prepared solves.
thetadot[-1]=0 and Tmdot[0]=0 always because both slots are algebraic boundary
values rather than evolved states. Forward sensitivities differentiate the
converged scalar boundary solve.

masstrans=1 (requires bubtherm=1, vapor=1) implements IMRv2's "if bubtherm &&
masstrans" branch: a wall vapor mass fraction field kv(y,t), a mixture
thermal conductivity/diffusivity (kv-weighted gas/vapor), extra mass-transfer
terms in Pdot/Uvel/thetadot, and a new kvdot equation. kv[-1] (the wall value)
is set algebraically every RHS call via vapor-liquid equilibrium (_kv_of_T,
f_kv_of_T.m) using T[-1] computed with the STALE (pre-update) kv[-1] --
IMRv2's own one-step lag, replicated exactly, not reconciled. With medtherm=0,
theta[-1] never evolves (frozen at its initial value), so T[-1]===1
identically and no implicit solve is needed for the wall BC itself. With
medtherm=1 also set, theta[-1] is instead solved via a 3-term coupled
root-find (_wall_theta_bw_full, f_bubble_wall_full_bc) that additionally
enforces vapor-mass-flux continuity (via a preliminary Clausius-Clapeyron
kv estimate at the candidate wall temperature) alongside heat-flux
continuity; alpha_m in that solve also uses the stale kv[-1], same lag.
Forward sensitivities cover this coupled algebraic condition.

The main API is not valid for radial=6 or arbitrary constitutive callbacks.
Re-validate before extending any branch.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np
from scipy.integrate import solve_ivp

from ._autodiff import primal, primal_array  # noqa: F401

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
  _WallState,
  _freeze_array,
  _readonly_float_array,
  _readonly_optional,
  _validate_inputs,
)

# Tait equation-of-state constants for the liquid, IMRv2 defaults
# (default_case.m: GAM, nstate), used by radial=3,4.
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
  _secant_root,
  _wall_theta_bw,
  _wall_theta_bw_full,
  pvsat,
)

# Mie-Gruneisen EoS constants for radial=5,6, IMRv2 defaults (default_case.m:
# hugoniot_s; f_imr_fd.m: nog=(nstate-1)/2, same nstate as the Tait branch).

# gas / vapor thermal-conductivity linear-in-T fit coefficients, IMRv2 defaults
# (default_case.m). K8 is IMRv2's reference conductivity: it mixes gas AND
# vapor coefficients even when vapor=0, because it is used purely as a
# normalization constant, not a physical mixture average at a given state.

# liquid (medium) thermal properties, IMRv2 defaults (default_case.m); water-like

# mass-transfer / vapor-species properties, IMRv2 defaults (default_case.m)


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


from ._constitutive import (  # noqa: F401
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
  _prepare_distributed_jacobian,
  _prepare_distributed_stress,
  _prepare_forcing,
  _prepare_instantaneous_material,
  _thermal_state,
  params,
  prepare,
)


from ._equations import (  # noqa: F401
  _nZ,
  _pinf,
  _radius_floor_event,
  _rhs,
  _sampled_pressure,
)


_radius_floor_event.terminal = True
_radius_floor_event.direction = -1


def _thermal_outputs(problem: PreparedProblem, states: np.ndarray):
  config = problem.config
  if not config.bubtherm:
    return None, None, None
  layout = problem.layout
  p = problem.parameters
  count = states.shape[0]
  bubble_temperature = np.empty((count, config.Nt))
  medium_temperature = np.empty((count, config.Mt)) if config.medtherm else None
  vapor_fraction = np.empty((count, config.Nt)) if config.masstrans else None
  wall_state = _WallState()
  for index, state in enumerate(states):
    theta = state[layout.bubble_thermal].copy()
    Tm = state[layout.medium_thermal].copy() if layout.medium_thermal is not None else None
    kv = state[layout.vapor_fraction].copy() if layout.vapor_fraction is not None else None
    temperature, _ = _apply_thermal_boundaries(
      theta, Tm, kv, state[layout.pressure], p, problem.medium, config.masstrans, wall_state
    )
    bubble_temperature[index] = config.T8 * temperature
    if medium_temperature is not None:
      medium_temperature[index] = config.T8 * Tm
    if vapor_fraction is not None:
      vapor_fraction[index] = kv
  return bubble_temperature, medium_temperature, vapor_fraction


def _build_result(problem: PreparedProblem, time_s: np.ndarray, solution, stats: SolverStats) -> SimulationResult:
  config = problem.config
  p = problem.parameters
  layout = problem.layout
  states = solution.y.T
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
      stress_integral[index] = _stress(
        config.material, p, state[0], state[1], stress_state, problem.instantaneous_material, False
      )[0]
    else:
      stress_integral[index] = _distributed_stress_integral(
        problem.distributed_stress, p, state[0], state[1], stress_state
      )

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
    stress_reference_radius_ratio=(
      problem.distributed_stress.reference_radius if problem.distributed_stress is not None else None
    ),
    stats=stats,
    config=config,
  )


def _integrate_prepared(problem: PreparedProblem, tv):
  config = problem.config
  time_s = _validate_inputs(
    tv,
    config.R0,
    config.Req,
    config.material,
    config.radial,
    config.vapor,
    config.T8,
    config.pA,
    config.omega,
    config.TW,
    config.DT,
    config.mn,
    config.wave_type,
    config.bubtherm,
    config.Nt,
    config.medtherm,
    config.Mt,
    config.masstrans,
    config.rtol,
    config.atol,
  )
  p = problem.parameters
  tn = time_s / p["t0"]
  args = (
    p,
    config.material,
    config.radial,
    config.bubtherm,
    problem.bubble_D1,
    problem.bubble_D2,
    problem.bubble_grid,
    config.medtherm,
    problem.medium,
    config.masstrans,
    _WallState(),
    problem.forcing,
    problem.instantaneous_material,
    problem.distributed_stress,
  )
  started = perf_counter()
  method = "BDF" if problem.jacobian_sparsity is not None else "LSODA"
  solver_options = {"jac_sparsity": problem.jacobian_sparsity} if problem.jacobian_sparsity is not None else {}
  if config.max_step_s is not None:
    solver_options["max_step"] = config.max_step_s / problem.parameters["t0"]
  try:
    solution = solve_ivp(
      _rhs,
      (tn[0], tn[-1]),
      problem.initial_state,
      t_eval=tn,
      args=args,
      events=_radius_floor_event,
      method=method,
      rtol=config.rtol,
      atol=config.atol,
      **solver_options,
    )
  except _MaterialDomainError as error:
    elapsed = perf_counter() - started
    message = f"material domain failure: {error}"
    stats = SolverStats(
      backend=f"scipy-{method.lower()}", success=False, message=message, nfev=0, njev=0, nlu=0, elapsed_s=elapsed
    )
    raise SimulationError(f"IMR integration failed: {message}", stats) from error
  elapsed = perf_counter() - started
  complete = solution.y.shape[1] == time_s.size
  finite = bool(np.all(np.isfinite(solution.y)))
  success = bool(solution.success and complete and finite)
  message = str(solution.message)
  if solution.success and not complete:
    message = f"{message}; terminated before the final requested time"
  elif solution.success and not finite:
    message = f"{message}; solution contains non-finite states"
  stats = SolverStats(
    backend=f"scipy-{method.lower()}",
    success=success,
    message=message,
    nfev=int(solution.nfev),
    njev=int(solution.njev),
    nlu=int(solution.nlu),
    elapsed_s=elapsed,
  )
  if not success:
    raise SimulationError(f"IMR integration failed: {message}", stats)
  return time_s, solution, stats


def _solve_prepared(problem: PreparedProblem, tv) -> SimulationResult:
  time_s, solution, stats = _integrate_prepared(problem, tv)
  return _build_result(problem, time_s, solution, stats)


def simulate(tv, config: SimulationConfig) -> SimulationResult:
  """Run one simulation and return immutable physical histories."""
  return prepare(config).solve(tv)


def simulate_with_sensitivities(tv, config: SimulationConfig, parameters):
  """Run one simulation with forward sensitivities.

  Parameter paths identify continuous configuration fields such as ``R0`` or
  ``material.shear_modulus_pa``.
  """
  return prepare(config).solve_with_sensitivities(tv, parameters)
