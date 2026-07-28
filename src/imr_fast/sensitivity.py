"""Forward sensitivities for the production IMR solver."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import lil_matrix

from . import _complex
import imr_fast as _solver
from ._stress import _stress
from ._autodiff import Dual
from ._dual import (
  SensitivityParameter,
  _dual_config,
  _dual_forcing,
  _dual_medium,
  _dual_parameters,
  _initial_matrix,
  _material_parameters,
  _mechanical_parameters,
  _normalize_parameters,
  _rhs_physical,
)
from ._mechanical import mechanical_stress_tangent, mechanical_tangent_rhs

__all__ = ["SensitivityParameter", "SensitivityResult", "simulate_with_sensitivities", "solve_with_sensitivities"]

@dataclass(frozen=True, slots=True)
class SensitivityResult:
  """Simulation and dimensional derivatives at every requested output time."""

  simulation: _solver.SimulationResult
  parameters: tuple[SensitivityParameter, ...]
  state: np.ndarray
  radius_ratio: np.ndarray
  radius_m: np.ndarray
  wall_velocity_m_s: np.ndarray
  internal_pressure_pa: np.ndarray
  stress_integral_pa: np.ndarray
  bubble_temperature_k: np.ndarray | None = None
  medium_temperature_k: np.ndarray | None = None
  vapor_mass_fraction: np.ndarray | None = None

def _rhs_mechanical_compiled(time_s, packed, *, problem, parameter_values, parameter_tangents, material_data, width):
  state_width = problem.layout.size
  matrix = packed.reshape(state_width, width + 1)
  (material_code, elastic_code, elastic_values, elastic_tangents, viscous_code, viscous_values, viscous_tangents) = material_data
  prepared = problem.instantaneous_material
  nodes = prepared.interval_nodes if prepared is not None else np.empty(0)
  weights = prepared.interval_weights if prepared is not None else np.empty(0)
  distributed = problem.distributed_stress
  reference_radius = distributed.reference_radius if distributed is not None else np.empty(0)
  reference_radius_cubed = distributed.reference_radius_cubed if distributed is not None else np.empty(0)
  stress_weights = distributed.weights if distributed is not None and distributed.weights is not None else np.empty(0)
  return mechanical_tangent_rhs(
    time_s,
    matrix,
    parameter_values,
    parameter_tangents,
    material_code,
    elastic_code,
    elastic_values,
    elastic_tangents,
    viscous_code,
    viscous_values,
    viscous_tangents,
    nodes,
    weights,
    reference_radius,
    reference_radius_cubed,
    stress_weights,
    problem.config.radial,
  ).ravel()

def _augmented_sparsity(base_sparsity, width):
  if base_sparsity is None: return None
  components = width + 1
  size = base_sparsity.shape[0]
  pattern = lil_matrix((size * components, size * components), dtype=bool)
  rows, columns = base_sparsity.nonzero()
  for row, column in zip(rows, columns, strict=True):
    base_row = row * components
    base_column = column * components
    pattern[base_row, base_column] = True
    for direction in range(1, components):
      pattern[base_row + direction, base_column] = True
      pattern[base_row + direction, base_column + direction] = True
  return pattern.tocsr()

def _readonly(values):
  result = np.asarray(values, dtype=float)
  result.setflags(write=False)
  return result

def _compiled_mechanical_outputs(problem, config, parameters, states, width, compiled):
  count = states.shape[0]
  outputs = tuple(np.empty((count, width)) for _ in range(5))
  (parameter_values, parameter_tangents, material_data) = compiled
  (material_code, elastic_code, elastic_values, elastic_tangents, viscous_code, viscous_values, viscous_tangents) = material_data
  prepared = problem.instantaneous_material
  nodes = prepared.interval_nodes if prepared is not None else np.empty(0)
  weights = prepared.interval_weights if prepared is not None else np.empty(0)
  distributed = problem.distributed_stress
  reference_radius = distributed.reference_radius if distributed is not None else np.empty(0)
  reference_radius_cubed = distributed.reference_radius_cubed if distributed is not None else np.empty(0)
  stress_weights = distributed.weights if distributed is not None and distributed.weights is not None else np.empty(0)
  for time_index, row in enumerate(states):
    radius = Dual(row[0, 0], row[0, 1:])
    wall_velocity = Dual(row[1, 0], row[1, 1:])
    pressure = (parameters["Pb"] - parameters["Pv"]) * radius ** (-3.0 * parameters["kappa"]) + parameters["Pv"]
    stress_data = mechanical_stress_tangent(
      row,
      parameter_values,
      parameter_tangents,
      material_code,
      elastic_code,
      elastic_values,
      elastic_tangents,
      viscous_code,
      viscous_values,
      viscous_tangents,
      nodes,
      weights,
      reference_radius,
      reference_radius_cubed,
      stress_weights,
    )
    stress = Dual(stress_data[0], stress_data[1:])
    values = (radius, config.R0 * radius, parameters["Uc"] * wall_velocity, parameters["P8"] * pressure, parameters["P8"] * stress)
    for target, value in zip(outputs, values, strict=True):
      target[time_index] = value.tangent
  return (*outputs, None, None, None)

def _output_duals(problem, config, parameters, states, width, compiled=None):
  if compiled is not None: return _compiled_mechanical_outputs(problem, config, parameters, states, width, compiled)
  count = states.shape[0]
  radius_ratio = np.empty((count, width))
  radius_m = np.empty_like(radius_ratio)
  velocity = np.empty_like(radius_ratio)
  pressure = np.empty_like(radius_ratio)
  stress = np.empty_like(radius_ratio)
  bubble_temperature = np.empty((count, config.Nt, width)) if config.bubtherm else None
  medium_temperature = np.empty((count, config.Mt, width)) if config.medtherm else None
  vapor_fraction = np.empty((count, config.Nt, width)) if config.masstrans else None
  wall_state = _solver._WallState()
  dual_medium = _dual_medium(problem, parameters)
  for time_index, row in enumerate(states):
    dual_state = np.array([Dual(row[index, 0], row[index, 1:]) for index in range(row.shape[0])], dtype=object)
    radius = dual_state[0]
    wall_velocity = dual_state[1]
    pressure_value = (
      dual_state[problem.layout.pressure]
      if config.bubtherm
      else ((parameters["Pb"] - parameters["Pv"]) * radius ** (-3.0 * parameters["kappa"]) + parameters["Pv"])
    )
    stress_state = dual_state[problem.layout.stress] if problem.layout.stress.stop > problem.layout.stress.start else None
    if problem.distributed_stress is None:
      stress_value = _stress(config.material, parameters, radius, wall_velocity, stress_state, problem.instantaneous_material, False)[0]
    else:
      stress_value = _solver._distributed_stress_integral(problem.distributed_stress, parameters, radius, wall_velocity, stress_state)
    output = (radius, config.R0 * radius, parameters["Uc"] * wall_velocity, parameters["P8"] * pressure_value, parameters["P8"] * stress_value)
    targets = (radius_ratio, radius_m, velocity, pressure, stress)
    for target, value in zip(targets, output, strict=True):
      target[time_index] = value.tangent if isinstance(value, Dual) else np.zeros(width)
    if config.bubtherm:
      theta = dual_state[problem.layout.bubble_thermal].copy()
      medium_state = dual_state[problem.layout.medium_thermal].copy() if config.medtherm else None
      vapor_state = dual_state[problem.layout.vapor_fraction].copy() if config.masstrans else None
      temperature, _ = _solver._apply_thermal_boundaries(
        theta, medium_state, vapor_state, pressure_value, parameters, dual_medium, config.masstrans, wall_state
      )
      bubble_temperature[time_index] = _tangent_values(config.T8 * temperature, width)
      if medium_temperature is not None: medium_temperature[time_index] = _tangent_values(config.T8 * medium_state, width)
      if vapor_fraction is not None: vapor_fraction[time_index] = _tangent_values(vapor_state, width)
  return (radius_ratio, radius_m, velocity, pressure, stress, bubble_temperature, medium_temperature, vapor_fraction)

def _tangent_values(values, width):
  array = np.asarray(values, dtype=object)
  result = np.zeros(array.shape + (width,))
  for index, value in np.ndenumerate(array):
    if isinstance(value, Dual): result[index] = value.tangent
  return result

def solve_with_sensitivities(problem, tv, parameters):
  """Solve one prepared problem and all requested forward sensitivities."""
  if not isinstance(problem, _solver.PreparedProblem): raise TypeError("problem must be a PreparedProblem")
  config = problem.config
  time_s = _solver._validate_inputs(tv, config)
  normalized, values, scales = _normalize_parameters(config, parameters)
  dual_config = _dual_config(config, normalized, values, scales)
  dual_parameters = _dual_parameters(dual_config)
  dual_medium = _dual_medium(problem, dual_parameters)
  dual_forcing = _dual_forcing(dual_config, dual_parameters)
  width = len(normalized)
  initial = _initial_matrix(problem, dual_config, dual_parameters, width)
  started = perf_counter()
  use_compiled_mechanical = not config.bubtherm and problem.forcing is None
  if use_compiled_mechanical:
    parameter_values, parameter_tangents = _mechanical_parameters(dual_parameters, width)
    material_data = _material_parameters(dual_config.material, width)
    rhs = partial(
      _rhs_mechanical_compiled,
      problem=problem,
      parameter_values=parameter_values,
      parameter_tangents=parameter_tangents,
      material_data=material_data,
      width=width,
    )
  elif _complex.complex_step_supported(problem):
    # Same packed layout, ~7-46x faster on the thermal path (#44).
    rhs = partial(
      _complex.rhs_complex,
      problem=problem,
      prepared=_complex.directions(dual_config, dual_parameters, dual_medium, dual_forcing, width),
      wall_states=[_solver._WallState() for _ in range(width)],
      width=width,
    )
  else:
    rhs = partial(
      _rhs_physical,
      problem=problem,
      config=dual_config,
      parameters=dual_parameters,
      medium=dual_medium,
      wall_state=_solver._WallState(),
      forcing=dual_forcing,
      width=width,
    )

  def radius_floor(_time, packed): return packed[0] - 1e-8

  radius_floor.terminal = True
  radius_floor.direction = -1
  jacobian_sparsity = _augmented_sparsity(problem.jacobian_sparsity, width)
  method = "BDF" if jacobian_sparsity is not None else "LSODA"
  solver_options = {"jac_sparsity": jacobian_sparsity} if jacobian_sparsity is not None else {}
  try:
    solution = solve_ivp(
      rhs,
      (time_s[0], time_s[-1]),
      initial.ravel(),
      t_eval=time_s,
      args=(),
      method=method,
      rtol=config.rtol,
      atol=config.atol,
      events=radius_floor,
      **solver_options,
    )
  except _solver._MaterialDomainError as error:
    elapsed = perf_counter() - started
    message = f"material domain failure: {error}"
    stats = _solver.SolverStats(backend=f"scipy-{method.lower()}-forward", success=False, message=message, nfev=0, njev=0, nlu=0, elapsed_s=elapsed)
    raise _solver.SimulationError(f"IMR sensitivity integration failed: {message}", stats) from error
  elapsed = perf_counter() - started
  complete = solution.y.shape[1] == time_s.size
  finite = bool(np.all(np.isfinite(solution.y)))
  success = bool(solution.success and complete and finite)
  message = str(solution.message)
  if solution.success and not complete:
    message = f"{message}; terminated before the final requested time"
  elif solution.success and not finite: message = f"{message}; solution contains non-finite states"
  stats = _solver.SolverStats(
    backend=f"scipy-{method.lower()}-forward",
    success=success,
    message=message,
    nfev=int(solution.nfev),
    njev=int(solution.njev),
    nlu=int(solution.nlu),
    elapsed_s=elapsed,
  )
  if not success: raise _solver.SimulationError(f"IMR sensitivity integration failed: {message}", stats)
  packed = solution.y.T.reshape(time_s.size, problem.layout.size, width + 1)
  base_states = packed[:, :, 0]
  base_solution = SimpleNamespace(y=base_states.T)
  simulation = _solver._build_result(problem, time_s, base_solution, stats)
  normalized_state = packed[:, :, 1:] / scales
  compiled_output = (parameter_values, parameter_tangents, material_data) if use_compiled_mechanical else None
  outputs = _output_duals(problem, dual_config, dual_parameters, packed, width, compiled_output)
  outputs = tuple(None if output is None else output / scales for output in outputs)
  return SensitivityResult(
    simulation=simulation,
    parameters=normalized,
    state=_readonly(normalized_state),
    radius_ratio=_readonly(outputs[0]),
    radius_m=_readonly(outputs[1]),
    wall_velocity_m_s=_readonly(outputs[2]),
    internal_pressure_pa=_readonly(outputs[3]),
    stress_integral_pa=_readonly(outputs[4]),
    bubble_temperature_k=(None if outputs[5] is None else _readonly(outputs[5])),
    medium_temperature_k=(None if outputs[6] is None else _readonly(outputs[6])),
    vapor_mass_fraction=(None if outputs[7] is None else _readonly(outputs[7])),
  )

def simulate_with_sensitivities(tv, config, parameters):
  """Prepare and solve a configuration with forward sensitivities."""
  return solve_with_sensitivities(_solver.prepare(config), tv, parameters)
