"""Forward sensitivities for the production IMR solver."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np

from ._config import SimulationResult, SolverStats, _validate_inputs
from ._solver import PreparedProblem, _build_result, prepare


__all__ = ["SensitivityParameter", "SensitivityResult", "simulate_with_sensitivities", "solve_with_sensitivities"]


@dataclass(frozen=True, slots=True)
class SensitivityParameter:
  """One differentiable configuration field."""

  path: str
  scale: float | None = None

  def __post_init__(self):
    if not isinstance(self.path, str) or not self.path: raise ValueError("sensitivity parameter path must be a non-empty string")
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
  "dynamics",
  "liquid_eos",
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
  "max_steps",
  "max_radius_ratio",
  "collapse",
}

def _path_parts(path, kind="sensitivity"):
  parts = path.split(".")
  if any(not part.isidentifier() for part in parts): raise ValueError(f"invalid {kind} parameter path: {path!r}")
  return parts

def _path_value(root, parts, full_path, kind="sensitivity"):
  """Walk a dotted path. `kind` only names the caller in the error -- `inference` and
  `sensitivity` had a copy each, identical but for that noun."""
  value = root
  for part in parts:
    if not hasattr(value, part): raise ValueError(f"unknown {kind} parameter path: {full_path!r}")
    value = getattr(value, part)
  return value

def _normalize_parameters(config, parameters):
  normalized = tuple(parameter if isinstance(parameter, SensitivityParameter) else SensitivityParameter(parameter) for parameter in parameters)
  if not normalized: raise ValueError("at least one sensitivity parameter is required")
  paths = [parameter.path for parameter in normalized]
  if len(set(paths)) != len(paths): raise ValueError("sensitivity parameter paths must be unique")
  values = []
  scales = []
  for parameter in normalized:
    parts = _path_parts(parameter.path)
    if any(part in _NONDIFFERENTIABLE_FIELDS for part in parts): raise ValueError(f"{parameter.path!r} is discrete or controls solver preparation")
    value = _path_value(config, parts, parameter.path)
    if not isinstance(value, Real) or not np.isfinite(float(value)): raise ValueError(f"{parameter.path!r} must identify one finite scalar field")
    scale = parameter.scale
    if scale is None: scale = abs(float(value)) if value != 0.0 else 1.0
    values.append(float(value))
    scales.append(float(scale))
  return normalized, values, np.asarray(scales)

@dataclass(frozen=True, slots=True)
class SensitivityResult:
  """Simulation and dimensional derivatives at every requested output time."""

  simulation: SimulationResult
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

from ._config import _freeze_array as _readonly, _readonly_optional

def _jax_sensitivities(problem, time_s, normalized):
  from ._jax import CONFIG_PATHS, INITIAL_PATHS, PHYSICS_PATHS, SCALE_PATHS, sensitivities_jax

  paths = [parameter.path for parameter in normalized]
  covered = set(SCALE_PATHS) | set(CONFIG_PATHS) | set(PHYSICS_PATHS) | set(INITIAL_PATHS) | set(PHYSICS_PATHS) | set(INITIAL_PATHS)
  unknown = [path for path in paths if path not in covered]
  if unknown:
    raise NotImplementedError(f"jax sensitivities cover {sorted(covered)}; got {unknown}")

  values, tangents = sensitivities_jax(problem, time_s, paths)
  assert tangents is not None  # noqa: S101 - only `values_only=True` returns None, and this asks for tangents
  derived_tangent = tangents.derived
  stats = SolverStats(
    backend="jax-tsit5-forward", success=True, message="jacfwd through the diffrax solve",
    nfev=0, njev=0, nlu=0, elapsed_s=0.0,
  )
  simulation = _build_result(problem, time_s, values.states.T, stats)
  return SensitivityResult(
    simulation=simulation,
    parameters=normalized,
    state=_readonly(tangents.states),
    radius_ratio=_readonly(derived_tangent[:, 0, :]),
    radius_m=_readonly(derived_tangent[:, 1, :]),
    wall_velocity_m_s=_readonly(derived_tangent[:, 2, :]),
    internal_pressure_pa=_readonly(derived_tangent[:, 3, :]),
    stress_integral_pa=_readonly(derived_tangent[:, 4, :]),
    bubble_temperature_k=_readonly_optional(tangents.bubble_temperature),
    medium_temperature_k=_readonly_optional(tangents.medium_temperature),
    vapor_mass_fraction=_readonly_optional(tangents.vapor_fraction),
  )

def solve_with_sensitivities(problem, tv, parameters):
  """Solve one prepared problem and all requested forward sensitivities."""
  if not isinstance(problem, PreparedProblem): raise TypeError("problem must be a PreparedProblem")
  time_s = _validate_inputs(tv, problem.config)
  normalized, _values, _scales = _normalize_parameters(problem.config, parameters)
  return _jax_sensitivities(problem, time_s, normalized)

def simulate_with_sensitivities(tv, config, parameters):
  """Prepare and solve a configuration with forward sensitivities."""
  return solve_with_sensitivities(prepare(config), tv, parameters)
