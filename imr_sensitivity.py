"""Forward sensitivities for the production IMR solver."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from functools import partial
from numbers import Real
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator
from scipy.sparse import lil_matrix

import imr_fast as _solver
from _imr_autodiff import Dual, seed, unpack
from _imr_mechanical import mechanical_stress_tangent, mechanical_tangent_rhs

__all__ = ["SensitivityParameter", "SensitivityResult", "simulate_with_sensitivities", "solve_with_sensitivities"]

_MECHANICAL_PARAMETER_KEYS = (
  "Pv",
  "kappa",
  "Pb",
  "req",
  "Ca",
  "Re8",
  "De",
  "LAM",
  "alphax",
  "P8",
  "t0",
  "viscosity_scale",
  "Cstar",
  "iWe",
  "tait_gamma",
  "tait_sam",
  "tait_no",
  "tait_exponent",
  "hugoniot_slope",
  "nog",
  "mie_reference",
  "ee",
  "om",
  "tw",
  "dt",
  "mn",
  "wave_type",
)


@dataclass(frozen=True, slots=True)
class SensitivityParameter:
  """One differentiable configuration field.

  ``path`` uses dataclass field notation, for example ``"R0"``,
  ``"material.shear_modulus_pa"``, or
  ``"physics.polytropic_exponent"``. ``scale`` is the dimensional parameter
  perturbation represented by a unit tangent; it affects integrator scaling,
  not the returned dimensional derivative.
  """

  path: str
  scale: float | None = None

  def __post_init__(self):
    if not isinstance(self.path, str) or not self.path:
      raise ValueError("sensitivity parameter path must be a non-empty string")
    if self.scale is not None and (not np.isfinite(self.scale) or self.scale <= 0.0):
      raise ValueError("sensitivity parameter scale must be finite and positive")


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


_NONDIFFERENTIABLE_FIELDS = {
  "radial",
  "vapor",
  "wave_type",
  "bubtherm",
  "Nt",
  "medtherm",
  "Mt",
  "masstrans",
  "quadrature_points",
  "points",
  "extent",
  "rtol",
  "atol",
  "collapse",
}


def _path_parts(path):
  parts = path.split(".")
  if any(not part.isidentifier() for part in parts):
    raise ValueError(f"invalid sensitivity parameter path: {path!r}")
  return parts


def _path_value(root, parts, full_path):
  value = root
  for part in parts:
    if not hasattr(value, part):
      raise ValueError(f"unknown sensitivity parameter path: {full_path!r}")
    value = getattr(value, part)
  return value


def _seed_path(root, parts, replacement):
  clone = copy.copy(root)
  if len(parts) == 1:
    object.__setattr__(clone, parts[0], replacement)
    return clone
  child = getattr(root, parts[0])
  object.__setattr__(clone, parts[0], _seed_path(child, parts[1:], replacement))
  return clone


def _normalize_parameters(config, parameters):
  normalized = tuple(
    parameter if isinstance(parameter, SensitivityParameter) else SensitivityParameter(parameter)
    for parameter in parameters
  )
  if not normalized:
    raise ValueError("at least one sensitivity parameter is required")
  paths = [parameter.path for parameter in normalized]
  if len(set(paths)) != len(paths):
    raise ValueError("sensitivity parameter paths must be unique")

  values = []
  scales = []
  for parameter in normalized:
    parts = _path_parts(parameter.path)
    if any(part in _NONDIFFERENTIABLE_FIELDS for part in parts):
      raise ValueError(f"{parameter.path!r} is discrete or controls solver preparation")
    value = _path_value(config, parts, parameter.path)
    if not isinstance(value, Real) or not np.isfinite(value):
      raise ValueError(f"{parameter.path!r} must identify one finite scalar field")
    scale = parameter.scale
    if scale is None:
      scale = abs(float(value)) if value != 0.0 else 1.0
    values.append(float(value))
    scales.append(float(scale))
  return normalized, values, np.asarray(scales)


def _dual_config(config, normalized, values, scales):
  width = len(normalized)
  result = config
  for index, (parameter, value, scale) in enumerate(zip(normalized, values, scales, strict=True)):
    replacement = seed(value, width, index, scale)
    result = _seed_path(result, _path_parts(parameter.path), replacement)
  return result


def _dual_parameters(config):
  parameters = _solver.params(
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
  if config.initial.internal_pressure_pa is not None:
    parameters["Pb"] = config.initial.internal_pressure_pa / parameters["P8"]
  return parameters


def _initial_matrix(problem, config, parameters, width):
  state = np.asarray(problem.initial_state)
  matrix = np.zeros((state.size, width + 1))
  matrix[:, 0] = state
  p = parameters
  initial = config.initial

  initial_dual = np.empty(state.size, dtype=object)
  for index, value in enumerate(state):
    initial_dual[index] = seed(value, width)
  initial_dual[0] = seed(1.0, width)
  initial_dual[1] = initial.wall_velocity_m_s / p["Uc"]
  if config.bubtherm:
    initial_dual[problem.layout.pressure] = p["Pb"]
    vapor_fraction = p["kv0"] if initial.vapor_mass_fraction is None else initial.vapor_mass_fraction
    if config.masstrans:
      for index in range(problem.layout.vapor_fraction.start, problem.layout.vapor_fraction.stop):
        initial_dual[index] = vapor_fraction
    temperature_ratio = 1.0 if initial.bubble_temperature_k is None else initial.bubble_temperature_k / config.T8
    alpha = vapor_fraction * p["alpha_v"] + (1.0 - vapor_fraction) * p["alpha_g"] if config.masstrans else p["alpha_g"]
    thermal_state = _solver._thermal_state(temperature_ratio, alpha)
    for index in range(problem.layout.bubble_thermal.start, problem.layout.bubble_thermal.stop):
      initial_dual[index] = thermal_state
    if config.medtherm:
      medium_temperature_ratio = (
        1.0 if initial.medium_temperature_k is None else initial.medium_temperature_k / config.T8
      )
      for index in range(problem.layout.medium_thermal.start, problem.layout.medium_thermal.stop):
        initial_dual[index] = medium_temperature_ratio
  if problem.collapse_stats is not None:
    collapse_tangents = _collapse_initial_tangents(problem, config, width)
    for offset, index in enumerate(range(problem.layout.stress.start, problem.layout.stress.stop)):
      initial_dual[index] = Dual(problem.collapse_stats.stress_state[offset], collapse_tangents[offset])
  elif initial.stress_state is not None:
    for index, value in zip(
      range(problem.layout.stress.start, problem.layout.stress.stop), initial.stress_state, strict=True
    ):
      initial_dual[index] = value
  values, tangents = unpack(initial_dual, width)
  matrix[:, 0] = values
  matrix[:, 1:] = tangents
  return matrix


def _pad_dual(value, width):
  if isinstance(value, Dual):
    return Dual(value.value, np.concatenate((value.tangent, [0.0])))
  return seed(value, width + 1)


def _ensure_dual(value, width):
  if isinstance(value, Dual):
    if value.tangent.size != width:
      raise ValueError("inconsistent tangent width in collapse precursor")
    return value
  return seed(value, width)


def _pad_object(value, width):
  clone = copy.copy(value)
  for name in value.__dataclass_fields__:
    field_value = getattr(value, name)
    if hasattr(field_value, "__dataclass_fields__"):
      replacement = _pad_object(field_value, width)
    elif isinstance(field_value, Dual):
      replacement = _pad_dual(field_value, width)
    else:
      continue
    object.__setattr__(clone, name, replacement)
  return clone


def _collapse_initial_tangents(problem, config, width):
  settings = config.collapse
  material = _pad_object(config.material, width)
  physics = _pad_object(config.physics, width)
  parameters = _solver.params(
    _pad_dual(config.R0, width),
    _pad_dual(config.Req, width),
    material,
    config.vapor,
    _pad_dual(config.T8, width),
    physics=physics,
    bubtherm=1,
    masstrans=config.masstrans,
  )
  parameters["kappa"] = seed(1.0, width + 1)
  parameters["req"] = _ensure_dual(parameters["req"], width + 1)
  state_width = problem.layout.stress.stop - problem.layout.stress.start
  direction_width = width + 1
  initial = np.zeros((2 + state_width, direction_width + 1))
  initial[0, 0] = parameters["req"].value
  initial[0, 1:] = parameters["req"].tangent
  initial[1, 0] = problem.collapse_stats.initial_velocity_nondimensional
  initial[1, -1] = 1.0
  upstream_zener = isinstance(material, _solver.Zener)

  wall_state = _solver._WallState()

  def tangent_rhs(time, packed):
    matrix = packed.reshape(2 + state_width, direction_width + 1)
    state = np.array([Dual(row[0], row[1:]) for row in matrix], dtype=object)
    if upstream_zener:
      output = _solver._collapse_zener_rhs(state, parameters)
    else:
      output = _solver._rhs(
        time,
        state,
        parameters,
        material,
        problem.config.radial,
        wall_state=wall_state,
        instantaneous_material=problem.instantaneous_material,
        distributed_stress=problem.distributed_stress,
      )
    values, tangents = unpack(output, direction_width)
    result = np.empty_like(matrix)
    result[:, 0] = values
    result[:, 1:] = tangents
    return result.ravel()

  def maximum_event(_time, packed):
    matrix = packed.reshape(2 + state_width, direction_width + 1)
    return matrix[1, 0]

  maximum_event.terminal = True
  maximum_event.direction = -1
  solution = solve_ivp(
    tangent_rhs,
    (0.0, settings.maximum_time_nondimensional),
    initial.ravel(),
    events=maximum_event,
    method="LSODA",
    rtol=min(problem.config.rtol, 1e-9),
    atol=min(problem.config.atol, 1e-11),
  )
  if not solution.success or solution.t_events[0].size == 0:
    raise _solver.SimulationError("collapse sensitivity precursor failed to reach maximum radius")
  event = solution.y_events[0][-1].reshape(2 + state_width, direction_width + 1)
  event_rhs = tangent_rhs(solution.t_events[0][-1], solution.y_events[0][-1]).reshape(
    2 + state_width, direction_width + 1
  )
  acceleration = event_rhs[1, 0]
  event_time_tangents = -event[1, 1:] / acceleration
  memory_tangents = event[2:, 1:] + event_rhs[2:, [0]] * event_time_tangents
  radius_tangents = event[0, 1:]
  velocity_direction = direction_width - 1
  radius_velocity = radius_tangents[velocity_direction]
  if abs(radius_velocity) < 1e-14:
    raise _solver.SimulationError("collapse shooting root has a singular velocity derivative")
  velocity_tangents = -radius_tangents[:width] / radius_velocity
  return memory_tangents[:, :width] + memory_tangents[:, [velocity_direction]] * velocity_tangents


def _rhs_physical(time_s, packed, *, problem, config, parameters, medium, wall_state, forcing, width):
  state_width = problem.layout.size
  matrix = packed.reshape(state_width, width + 1)
  state = np.empty(state_width, dtype=object)
  for index in range(state_width):
    state[index] = Dual(matrix[index, 0], matrix[index, 1:])
  nondimensional_time = time_s / parameters["t0"]
  output = _solver._rhs(
    nondimensional_time,
    state,
    parameters,
    config.material,
    config.radial,
    config.bubtherm,
    problem.bubble_D1,
    problem.bubble_D2,
    problem.bubble_grid,
    config.medtherm,
    medium,
    config.masstrans,
    wall_state,
    forcing,
    problem.instantaneous_material,
    problem.distributed_stress,
  )
  physical_output = np.asarray([value / parameters["t0"] for value in output], dtype=object)
  values, tangents = unpack(physical_output, width)
  result = np.empty_like(matrix)
  result[:, 0] = values
  result[:, 1:] = tangents
  return result.ravel()


def _dual_medium(problem, parameters):
  if problem.medium is None:
    return None
  medium = copy.copy(problem.medium)
  xi = problem.medium.xi
  length = parameters["Lt"]
  with np.errstate(divide="ignore", invalid="ignore"):
    y_t = (2.0 / (xi + 1.0) - 1.0) * length + 1.0
    y_t2 = y_t**2
    y_t3 = y_t**3
    inverse_y_t3 = y_t**-3
    inverse_y_t4 = y_t**-4
    inverse_y_t6 = y_t**-6
  coefficients = np.array([-1.5, 2.0, -0.5])
  delta_medium = -2.0 / (problem.config.Mt - 1)
  delta_bubble = 1.0 / (problem.config.Nt - 1)

  def _pad(values, length):
    # Boundary weights are stored full length (see _prepare_* in _imr_prepare);
    # the finite-difference tail is zero.
    padded = np.zeros(length, dtype=object)
    padded[:] = 0.0
    padded[: len(values)] = values
    return padded

  replacements = {
    "yT": y_t,
    "yT2": y_t2,
    "yT3": y_t3,
    "iyT3": inverse_y_t3,
    "iyT4": inverse_y_t4,
    "iyT6": inverse_y_t6,
    "grad_Tm": _pad(2.0 * parameters["chi"] * parameters["iota"] / delta_medium * coefficients, problem.config.Mt),
    "grad_Trans": _pad(-coefficients * parameters["chi"] / delta_bubble, problem.config.Nt),
    "grad_C": _pad(-coefficients * parameters["Fom"] * parameters["L_heat_star"] / delta_bubble, problem.config.Nt),
  }
  for name, value in replacements.items():
    object.__setattr__(medium, name, value)
  return medium


def _dual_forcing(config, parameters):
  forcing = config.sampled_forcing
  if forcing is None:
    return None
  physical = PchipInterpolator(np.asarray(forcing.time_s), np.asarray(forcing.pressure_pa), extrapolate=False)
  coefficients = np.empty(physical.c.shape, dtype=object)
  for row, degree in enumerate((3, 2, 1, 0)):
    coefficients[row] = physical.c[row] * parameters["t0"] ** degree / parameters["P8"]
  return _solver.PreparedForcing(
    knots=np.asarray(forcing.time_s, dtype=object) / parameters["t0"], coefficients=coefficients
  )


def _packed_values(values, width, size):
  result_values = np.zeros(size)
  result_tangents = np.zeros((size, width))
  for index, value in enumerate(values):
    if isinstance(value, Dual):
      result_values[index] = value.value
      result_tangents[index] = value.tangent
    else:
      result_values[index] = value
  return result_values, result_tangents


def _mechanical_parameters(parameters, width):
  return _packed_values([parameters[key] for key in _MECHANICAL_PARAMETER_KEYS], width, len(_MECHANICAL_PARAMETER_KEYS))


def _material_parameters(material, width):
  elastic_code = 0
  viscous_code = 0
  elastic_fields = ()
  viscous_fields = ()
  if isinstance(material, _solver.NoStress):
    material_code = 0
  elif isinstance(material, _solver.NeoHookeanKelvinVoigt):
    material_code = 1
  elif isinstance(material, _solver.QuadraticKelvinVoigt):
    material_code = 2
  elif isinstance(material, _solver.Zener):
    material_code = 3
  elif isinstance(material, _solver.QuadraticZener):
    material_code = 4
  elif isinstance(material, _solver.OldroydB):
    material_code = 5
  elif isinstance(material, _solver.Giesekus):
    material_code = 7
    elastic_fields = (material.mobility,)
  elif isinstance(material, _solver.LinearPTT):
    material_code = 8
    elastic_fields = (material.extensibility,)
  elif isinstance(material, _solver.InstantaneousMaterial):
    material_code = 6
    elastic = material.elastic
    if isinstance(elastic, _solver.NeoHookean):
      elastic_code = 1
      elastic_fields = (elastic.shear_modulus_pa,)
    elif isinstance(elastic, _solver.MooneyRivlin):
      elastic_code = 2
      elastic_fields = (elastic.c10_pa, elastic.c01_pa)
    elif isinstance(elastic, _solver.Yeoh):
      elastic_code = 3
      elastic_fields = (elastic.c1_pa, elastic.c2_pa, elastic.c3_pa)
    elif isinstance(elastic, _solver.Fung):
      elastic_code = 4
      elastic_fields = (elastic.shear_modulus_pa, elastic.stiffening)
    elif isinstance(elastic, _solver.Gent):
      elastic_code = 5
      elastic_fields = (elastic.shear_modulus_pa, elastic.extensibility)
    elif isinstance(elastic, _solver.ArrudaBoyce):
      elastic_code = 6
      elastic_fields = (elastic.shear_modulus_pa, elastic.chain_segments)
    viscous = material.viscous
    if isinstance(viscous, _solver.Newtonian):
      viscous_code = 1
      viscous_fields = (viscous.viscosity_pa_s,)
    elif isinstance(viscous, _solver.PowerLaw):
      viscous_code = 2
      viscous_fields = (viscous.consistency_pa_s_n, viscous.exponent, viscous.regularization_rate_per_s)
    elif isinstance(viscous, _solver.CarreauYasuda):
      viscous_code = 3
      viscous_fields = (
        viscous.zero_shear_viscosity_pa_s,
        viscous.infinite_shear_viscosity_pa_s,
        viscous.time_constant_s,
        viscous.transition_exponent,
        viscous.power_index,
      )
    elif isinstance(viscous, _solver.Cross):
      viscous_code = 4
      viscous_fields = (
        viscous.zero_shear_viscosity_pa_s,
        viscous.infinite_shear_viscosity_pa_s,
        viscous.time_constant_s,
        viscous.transition_exponent,
      )
    elif isinstance(viscous, (_solver.PowellEyring, _solver.ModifiedPowellEyring)):
      viscous_code = 8 if isinstance(viscous, _solver.ModifiedPowellEyring) else 7
      viscous_fields = (
        viscous.zero_shear_viscosity_pa_s,
        viscous.infinite_shear_viscosity_pa_s,
        viscous.time_constant_s,
      )
    elif isinstance(viscous, _solver.HerschelBulkley):
      viscous_code = 5
      viscous_fields = (
        viscous.yield_stress_pa,
        viscous.consistency_pa_s_n,
        viscous.exponent,
        viscous.regularization_rate_per_s,
      )
    elif isinstance(viscous, _solver.Bingham):
      viscous_code = 6
      viscous_fields = (viscous.yield_stress_pa, viscous.plastic_viscosity_pa_s, viscous.regularization_rate_per_s)
  else:
    raise TypeError("unsupported mechanical material")
  elastic_values, elastic_tangents = _packed_values(elastic_fields, width, 5)
  viscous_values, viscous_tangents = _packed_values(viscous_fields, width, 5)
  return (material_code, elastic_code, elastic_values, elastic_tangents, viscous_code, viscous_values, viscous_tangents)


def _rhs_mechanical_compiled(time_s, packed, *, problem, parameter_values, parameter_tangents, material_data, width):
  state_width = problem.layout.size
  matrix = packed.reshape(state_width, width + 1)
  (material_code, elastic_code, elastic_values, elastic_tangents, viscous_code, viscous_values, viscous_tangents) = (
    material_data
  )
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
  if base_sparsity is None:
    return None
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
  (material_code, elastic_code, elastic_values, elastic_tangents, viscous_code, viscous_values, viscous_tangents) = (
    material_data
  )
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
    values = (
      radius,
      config.R0 * radius,
      parameters["Uc"] * wall_velocity,
      parameters["P8"] * pressure,
      parameters["P8"] * stress,
    )
    for target, value in zip(outputs, values, strict=True):
      target[time_index] = value.tangent
  return (*outputs, None, None, None)


def _output_duals(problem, config, parameters, states, width, compiled=None):
  if compiled is not None:
    return _compiled_mechanical_outputs(problem, config, parameters, states, width, compiled)
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
    stress_state = (
      dual_state[problem.layout.stress] if problem.layout.stress.stop > problem.layout.stress.start else None
    )
    if problem.distributed_stress is None:
      stress_value = _solver._stress(
        config.material, parameters, radius, wall_velocity, stress_state, problem.instantaneous_material, False
      )[0]
    else:
      stress_value = _solver._distributed_stress_integral(
        problem.distributed_stress, parameters, radius, wall_velocity, stress_state
      )
    output = (
      radius,
      config.R0 * radius,
      parameters["Uc"] * wall_velocity,
      parameters["P8"] * pressure_value,
      parameters["P8"] * stress_value,
    )
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
      if medium_temperature is not None:
        medium_temperature[time_index] = _tangent_values(config.T8 * medium_state, width)
      if vapor_fraction is not None:
        vapor_fraction[time_index] = _tangent_values(vapor_state, width)
  return (radius_ratio, radius_m, velocity, pressure, stress, bubble_temperature, medium_temperature, vapor_fraction)


def _tangent_values(values, width):
  array = np.asarray(values, dtype=object)
  result = np.zeros(array.shape + (width,))
  for index, value in np.ndenumerate(array):
    if isinstance(value, Dual):
      result[index] = value.tangent
  return result


def solve_with_sensitivities(problem, tv, parameters):
  """Solve one prepared problem and all requested forward sensitivities."""
  if not isinstance(problem, _solver.PreparedProblem):
    raise TypeError("problem must be a PreparedProblem")
  config = problem.config
  time_s = _solver._validate_inputs(
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

  def radius_floor(_time, packed):
    return packed[0] - 1e-8

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
    stats = _solver.SolverStats(
      backend=f"scipy-{method.lower()}-forward",
      success=False,
      message=message,
      nfev=0,
      njev=0,
      nlu=0,
      elapsed_s=elapsed,
    )
    raise _solver.SimulationError(f"IMR sensitivity integration failed: {message}", stats) from error
  elapsed = perf_counter() - started
  complete = solution.y.shape[1] == time_s.size
  finite = bool(np.all(np.isfinite(solution.y)))
  success = bool(solution.success and complete and finite)
  message = str(solution.message)
  if solution.success and not complete:
    message = f"{message}; terminated before the final requested time"
  elif solution.success and not finite:
    message = f"{message}; solution contains non-finite states"
  stats = _solver.SolverStats(
    backend=f"scipy-{method.lower()}-forward",
    success=success,
    message=message,
    nfev=int(solution.nfev),
    njev=int(solution.njev),
    nlu=int(solution.nlu),
    elapsed_s=elapsed,
  )
  if not success:
    raise _solver.SimulationError(f"IMR sensitivity integration failed: {message}", stats)

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
