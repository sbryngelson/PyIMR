"""Dual-mode problem construction for the sensitivity solver.

Everything needed to turn a `SimulationConfig` into a forward-mode problem:
parameter paths and seeding, `Dual` copies of the config, prepared medium
operators and forcing, collapse-precursor initial tangents, and the flat
parameter packing the compiled kernels take. The integration itself and the
assembly of results stay in `imr_fast.sensitivity`.

Split out under #27. `_dual_medium` here is where the spectral wall-stencil
defect in #43 lived, which is part of why this file is worth having on its own.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from numbers import Real

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator

import imr_fast as _solver
from ._autodiff import Dual, seed, unpack
from ._thermal import _far_field_singular_index


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


def _dual_medium(problem, parameters):
  if problem.medium is None:
    return None
  medium = copy.copy(problem.medium)
  xi = problem.medium.xi
  length = parameters["Lt"]
  # Same far-field singularity as _prepare, and the same limits. Worth
  # stating what that means for the tangent: yT -> inf and its inverse powers
  # -> 0 whatever Lt is, so the far-field entries carry NO dependence on the
  # parameters and their derivatives are exactly zero. The suppressed form
  # instead produced inf * Dual, giving an inf or nan tangent there, which was
  # harmless only because Tmdot[-1] is overwritten downstream.
  _far_field_singular_index(xi)
  interior = np.asarray(xi)[:-1]
  y_t = np.empty(np.asarray(xi).size, dtype=object)
  y_t[:-1] = (2.0 / (interior + 1.0) - 1.0) * length + 1.0
  y_t[-1] = float("inf")
  y_t2 = np.empty_like(y_t)
  y_t3 = np.empty_like(y_t)
  inverse_y_t3 = np.empty_like(y_t)
  inverse_y_t4 = np.empty_like(y_t)
  inverse_y_t6 = np.empty_like(y_t)
  y_t2[:-1], y_t3[:-1] = y_t[:-1] ** 2, y_t[:-1] ** 3
  inverse_y_t3[:-1] = y_t[:-1] ** -3
  inverse_y_t4[:-1] = y_t[:-1] ** -4
  inverse_y_t6[:-1] = y_t[:-1] ** -6
  y_t2[-1] = y_t3[-1] = float("inf")
  inverse_y_t3[-1] = inverse_y_t4[-1] = inverse_y_t6[-1] = 0.0

  # The wall-flux weights carry Dual parameter dependence and so must be
  # rebuilt, but their stencils are pure grid geometry. Take those from the
  # prepared problem rather than assuming a shape: they are dense for Chebyshev
  # collocation and three-point for finite difference, and hardcoding the
  # latter made every thermal="spectral" tangent differentiate a different
  # operator than the forward solve integrated -- a step-independent 2.5e-01
  # relative error. See issue #31, which lists exactly this pair.
  def _dual_weights(scalar, stencil):
    # Structural zeros stay plain floats, as they were before this rebuilt the
    # weights from a stencil: the finite-difference tail is exactly zero and
    # carries no parameter dependence, so making it Dual only adds arithmetic.
    weights = np.zeros(stencil.size, dtype=object)
    weights[:] = 0.0
    for index, value in enumerate(stencil):
      if value != 0.0:
        weights[index] = scalar * float(value)
    return weights

  bubble_stencil = np.asarray(problem.medium.bubble_wall_stencil)
  medium_stencil = np.asarray(problem.medium.medium_wall_stencil)

  replacements = {
    "yT": y_t,
    "yT2": y_t2,
    "yT3": y_t3,
    "iyT3": inverse_y_t3,
    "iyT4": inverse_y_t4,
    "iyT6": inverse_y_t6,
    "grad_Tm": _dual_weights(2.0 * parameters["chi"] * parameters["iota"], medium_stencil),
    "grad_Trans": _dual_weights(parameters["chi"], bubble_stencil),
    "grad_C": _dual_weights(parameters["Fom"] * parameters["L_heat_star"], bubble_stencil),
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
    elif isinstance(elastic, _solver.Ogden):
      elastic_code = 7
      # [n_terms, mu_1..mu_n, alpha_1..alpha_n]; the compiled kernel takes a
      # flat array, so the term count travels with the data.
      elastic_fields = (float(len(elastic.exponents)), *elastic.shear_moduli_pa, *elastic.exponents)
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
  # Sized to the data, not to a constant. Every law before Ogden had at most
  # three parameters, so a hardcoded 5 was invisible slack; Ogden needs
  # 1 + 2 * terms and overflowed it. max() keeps the padding the compiled
  # kernels have always been able to index past the end of.
  elastic_values, elastic_tangents = _packed_values(elastic_fields, width, max(5, len(elastic_fields)))
  viscous_values, viscous_tangents = _packed_values(viscous_fields, width, max(5, len(viscous_fields)))
  return (material_code, elastic_code, elastic_values, elastic_tangents, viscous_code, viscous_values, viscous_tangents)
