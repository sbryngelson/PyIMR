"""Running a prepared problem: the integration call, and the results it builds.

Separate from `__init__` because everything here is implementation. With it in the
package root, any module that needed to solve had to import the root, and the root
imported them back -- `_prepare` and `__init__` were a genuine top-level cycle, and
`PreparedProblem` reached the solver through five function-level imports.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

from numbers import Integral
from types import MappingProxyType
from typing import Mapping

import numpy as np

from ._config import (
  CollapseStats,
  MediumOperators,
  PreparedDistributedStress,
  PreparedForcing,
  PreparedInstantaneousMaterial,
  SimulationConfig,
  SimulationError,
  SimulationResult,
  SolverStats,
  StateLayout,
  SweepResult,
  _freeze_array,
  _readonly_optional,
  _validate_inputs,
)
from ._integrate import integrate as _integrate
from ._prepare import (
  _collapse_memory_state,
  _prepare_distributed_stress,
  _prepare_forcing,
  _prepare_instantaneous_material,
  initial_state_vector,
  params,
)
from ._rhs import _rhs, _rhs_args
from ._stress import _MaterialDomainError, _distributed_stress_integral, _stress
from ._thermal import _apply_thermal_boundaries, _far_field_singular_index
from .thermal_fd import finite_diff_mat
from .thermal_spectral import chebyshev_diff_mat, nodes as chebyshev_nodes

__all__ = ["PreparedProblem", "prepare", "simulate", "simulate_with_sensitivities"]


@dataclass(frozen=True, slots=True)
class PreparedProblem:
  """Reusable static data for repeated simulations of one configuration."""

  config: SimulationConfig
  parameters: Mapping[str, float]
  layout: StateLayout
  initial_state: np.ndarray
  bubble_grid: np.ndarray | None = None
  bubble_D1: np.ndarray | None = None
  bubble_D2: np.ndarray | None = None
  medium: MediumOperators | None = None
  forcing: PreparedForcing | None = None
  instantaneous_material: PreparedInstantaneousMaterial | None = None
  distributed_stress: PreparedDistributedStress | None = None
  collapse_stats: CollapseStats | None = None

  # `parameters` is a mappingproxy, which pickle refuses. Without these, every
  # `workers > 1` path -- `evaluate_batch`, `fit_multistart`, `design_information` --
  # dies with "cannot pickle 'mappingproxy' object" before running any work.
  def __getstate__(self):
    state = {item.name: getattr(self, item.name) for item in fields(self)}
    state["parameters"] = dict(state["parameters"])
    return state

  def __setstate__(self, state):
    for name, value in state.items():
      object.__setattr__(self, name, MappingProxyType(value) if name == "parameters" else value)

  def solve(self, tv) -> SimulationResult:

    return _solve_prepared(self, tv)

  def solve_states(self, tv, state=None) -> np.ndarray:
    """The raw internal trajectory, shaped `(time, state)`.

    `solve` returns physical histories; ensemble and assimilation code needs the
    vector the integrator actually advances, and `solve_from` needs one back.
    """

    _, states, _ = _integrate_prepared(self, tv, state)
    return _freeze_array(np.asarray(states).T)

  def solve_ensemble(self, members, tv, *, drop_failures: bool = False, chunk: int | None = None):
    """`(states, ok)` for a batch of initial states, advanced together.

    One `vmap` over the whole ensemble rather than a loop of solves, so the members share
    a compiled program. `ok[i]` says whether member `i` integrated successfully: an
    ensemble is drawn from a distribution and its tails do diverge, and because the whole
    batch shares one program a single failure would otherwise take all of it down.

    Raises if any member failed, unless `drop_failures`, in which case the survivors are
    returned and `ok` records who they were.

    `chunk` splits the batch and solves the pieces in turn. Per-member cost is not
    monotone in batch width -- measured 20 ms at 32 members and 74-94 ms at 36-48, with
    the same members duplicated into a wider batch reproducing it, so it is the width and
    not the draw (#162). Chunking at 32 was 1.8-2.6x faster for 64-256 members on one
    machine. It is not the default because that number is a fact about that machine: XLA
    picks kernels per shape and 34 was fast where 33 and 36 were not. Measure before
    setting it. Chunking is not exactly a no-op: full-width pieces come back
    bit-identical, but a remainder piece has a different width and so a different kernel
    and summation order -- measured 1.0e-15 absolute, three orders below the solver rtol.
    """

    from ._jax import ensemble_states_jax

    times = _validate_inputs(tv, self.config)
    batch = np.asarray(members, dtype=float)
    if batch.ndim != 2: raise ValueError(f"members must be a 2-D array of states; got shape {batch.shape}")
    stacked = np.stack([_validate_state(self, row) for row in batch])
    if chunk is None:
      states, ok = ensemble_states_jax(self, times, stacked)
    else:
      if not isinstance(chunk, Integral) or chunk < 1: raise ValueError("chunk must be a positive integer")
      pieces = [ensemble_states_jax(self, times, stacked[start : start + int(chunk)]) for start in range(0, len(stacked), int(chunk))]
      states = np.concatenate([piece[0] for piece in pieces])
      ok = np.concatenate([piece[1] for piece in pieces])
    if not ok.all() and not drop_failures:
      failed = np.flatnonzero(~ok)
      raise SimulationError(f"{failed.size} of {ok.size} ensemble members failed to integrate: {failed[:8].tolist()}")
    keep = np.ones(ok.size, dtype=bool) if not drop_failures else ok
    return _freeze_array(states[keep]), _freeze_array(ok).astype(bool)

  def solve_sweep(self, tv, parameters, values) -> SweepResult:
    """Solve one traced program at many parameter sets at once.

    A grid or sample campaign is the case this exists for: it traces once and `vmap`s,
    rather than paying a fresh solve per point.

    Whether that is FASTER depends on the solver, and the direction reverses. With an
    explicit solver (`thermal="fd"`) batching won: 144.6 ms a point at width 1 against
    45.7 ms at width 256. With the implicit solver `thermal="spectral"` selects, batching
    LOST: 1572 ms at width 1 against 3927 at width 8 and 2797 at width 32. Under `vmap`
    the Newton solves batch together and one member needing small steps imposes them on
    all. Measure at your configuration; for coupled spectral, prefer separate processes.

    `parameters` are paths as `solve_with_sensitivities` takes them; `values` is
    `(point, parameter)` in DIMENSIONAL units.
    """
    from ._jax import _DERIVED_ORDER, sensitivities_jax

    times = _validate_inputs(tv, self.config)
    paths = tuple(parameters)
    if not paths: raise ValueError("solve_sweep needs at least one parameter path")
    grid = np.atleast_2d(np.asarray(values, dtype=float))
    if grid.ndim != 2 or grid.shape[1] != len(paths):
      raise ValueError(f"values must be (point, {len(paths)}) to match the paths given; got {grid.shape}")
    if not np.all(np.isfinite(grid)): raise ValueError("values must be finite")

    outputs, _ = sensitivities_jax(self, times, list(paths), at=grid, values_only=True)
    derived = np.asarray(outputs.derived)
    fields = {name: _freeze_array(derived[:, :, index]) for index, name in enumerate(_DERIVED_ORDER)}
    return SweepResult(
      parameters=paths, values=_freeze_array(grid), time_s=_freeze_array(times),
      state=_freeze_array(np.asarray(outputs.states)),
      steps=_freeze_array(np.atleast_1d(np.asarray(outputs.steps))),
      **fields,
    )

  def state_tangents(self, tv, state=None):
    """`(states, jacobian)` where `jacobian[k]` is `d state(t_k) / d state(t_0)`.

    The tangent linear operator of the flow. `jacobian[0]` is the identity by
    construction, and `jacobian[k] @ v` propagates a perturbation `v` to `t_k`.
    """

    from ._jax import state_tangents_jax

    times = _validate_inputs(tv, self.config)
    start = self.initial_state if state is None else _validate_state(self, state)
    return state_tangents_jax(self, times, start)

  def solve_from(self, state, tv) -> SimulationResult:
    """Solve from an arbitrary state rather than the configured initial one.

    `state` is the raw internal vector, laid out as `self.layout` describes and
    nondimensionalised the same way `initial_state` is -- not physical units.
    """

    return _solve_prepared(self, tv, state)

  def solve_with_sensitivities(self, tv, parameters):
    from .sensitivity import solve_with_sensitivities

    return solve_with_sensitivities(self, tv, parameters)

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

def _reject_runaway(config, radius_ratio, stats):
  """The bubble starts at its maximum radius, so growth far past it is the model failing.

  qSLS at strong stiffening with relaxation near the bubble period reaches R/R0 = 2132,
  identically at rtol 1e-3, 1e-4 and 1e-5 -- a converged property of the equations, not a
  tolerance artifact -- and returns without raising. Silent is the danger: inference,
  design and model selection all treat a returned result as usable.
  """
  if config.max_radius_ratio is None: return
  largest = float(np.max(radius_ratio))
  if largest > config.max_radius_ratio:
    raise SimulationError(
      f"radius ran away to R/R0 = {largest:.4g}, above max_radius_ratio "
      f"{config.max_radius_ratio:g}; the model is not physical here",
      replace(stats, success=False, message="radius runaway"),
    )

def _build_result(problem: PreparedProblem, time_s: np.ndarray, states, stats: SolverStats) -> SimulationResult:
  config = problem.config
  p = problem.parameters
  layout = problem.layout
  states = np.asarray(states).T
  radius_ratio = states[:, 0]
  velocity = states[:, 1]
  _reject_runaway(config, radius_ratio, stats)
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
    time_s=_freeze_array(time_s),
    radius_ratio=_freeze_array(radius_ratio),
    wall_velocity_m_s=_freeze_array(Uc * velocity),
    internal_pressure_pa=_freeze_array(pressure_scale * pressure),
    stress_integral_pa=_freeze_array(pressure_scale * stress_integral),
    bubble_temperature_k=_readonly_optional(bubble_temperature),
    medium_temperature_k=_readonly_optional(medium_temperature),
    vapor_mass_fraction=_readonly_optional(vapor_fraction),
    stress_state=_readonly_optional(internal_stress_state),
    stress_reference_radius_ratio=(problem.distributed_stress.reference_radius if problem.distributed_stress is not None else None),
    stats=stats,
    config=config,
  )

def _validate_state(problem: PreparedProblem, state):
  """A restart state has to match the layout the problem was prepared for."""
  values = np.asarray(state, dtype=float)
  expected = problem.initial_state.size
  if values.shape != (expected,):
    raise ValueError(f"state must have shape ({expected},) for this configuration; got {values.shape}")
  if not np.all(np.isfinite(values)): raise ValueError("state must contain only finite values")
  return values

def _integrate_prepared(problem: PreparedProblem, tv, state=None):
  config = problem.config
  time_s = _validate_inputs(tv, config)
  p = problem.parameters
  tn = time_s / p["t0"]
  args = _rhs_args(problem, p, medium=problem.medium)
  start = problem.initial_state if state is None else _validate_state(problem, state)
  states, stats = _integrate(
    _rhs, tn, start, args=args,
    rtol=config.rtol, atol=config.atol, failure="IMR integration failed", config=config,
    max_step=None if config.max_step_s is None else config.max_step_s / p["t0"],
  )
  return time_s, states, stats

def _solve_prepared(problem: PreparedProblem, tv, state=None) -> SimulationResult:
  time_s, states, stats = _integrate_prepared(problem, tv, state)
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
