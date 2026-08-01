"""Prepared likelihood, batch evaluation, and deterministic multistart tools."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from itertools import repeat
from numbers import Integral

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import least_squares
from scipy.stats import qmc

import pyimr
from .sensitivity import SensitivityParameter, _normalize_parameters

__all__ = [
  "InferenceParameter",
  "LikelihoodEvaluation",
  "MultistartEndpoint",
  "MultistartResult",
  "PreparedInference",
  "RadiusObservation",
  "prepare_inference",
]

def _readonly(values):
  array = np.array(values, dtype=float, copy=True)
  array.setflags(write=False)
  return array

def _path_parts(path):
  parts = path.split(".")
  if not parts or any(not part.isidentifier() for part in parts): raise ValueError(f"invalid inference parameter path: {path!r}")
  return parts

def _path_value(root, parts, full_path):
  value = root
  for part in parts:
    if not hasattr(value, part): raise ValueError(f"unknown inference parameter path: {full_path!r}")
    value = getattr(value, part)
  return value

def _replace_path(root, parts, value):
  if len(parts) == 1: return replace(root, **{parts[0]: value})
  child = _replace_path(getattr(root, parts[0]), parts[1:], value)
  return replace(root, **{parts[0]: child})

@dataclass(frozen=True, slots=True)
class InferenceParameter:
  """One bounded continuous configuration field."""

  path: str
  lower: float
  upper: float
  transform: str = "linear"

  def __post_init__(self):
    _path_parts(self.path)
    if not np.isfinite(self.lower) or not np.isfinite(self.upper) or self.lower >= self.upper:
      raise ValueError("inference bounds must be finite and increasing")
    if self.transform not in ("linear", "log"): raise ValueError("inference transform must be 'linear' or 'log'")
    if self.transform == "log" and self.lower <= 0.0: raise ValueError("log-transformed inference bounds must be positive")

  def physical_value(self, unit_value):
    if self.transform == "linear": return self.lower + unit_value * (self.upper - self.lower)
    return self.lower * (self.upper / self.lower) ** unit_value

  def derivative(self, unit_value):
    if self.transform == "linear": return self.upper - self.lower
    value = self.physical_value(unit_value)
    return value * np.log(self.upper / self.lower)

def _validate_observation(time_s, values, deviation, *, minimum, value_label, deviation_label):
  time = np.asarray(time_s, dtype=float)
  observed = np.asarray(values, dtype=float)
  spread = np.asarray(deviation, dtype=float)
  if time.ndim != 1 or time.size < minimum:
    raise ValueError(f"observation time_s must be one-dimensional with at least {minimum} point(s)")
  if observed.shape != time.shape: raise ValueError(f"observation {value_label} must match time_s")
  if spread.ndim > 1 or (spread.ndim == 1 and spread.shape != time.shape):
    raise ValueError(f"{deviation_label} must be scalar or match time_s")
  if not (np.all(np.isfinite(time)) and np.all(np.isfinite(observed)) and np.all(np.isfinite(spread))):
    raise ValueError("observations and deviations must be finite")
  if np.any(spread <= 0.0): raise ValueError(f"{deviation_label} must be positive")
  if time[0] < 0.0 or np.any(np.diff(time) <= 0.0): raise ValueError("observation time_s must be non-negative and increasing")
  if spread.ndim == 0: spread = np.full(time.shape, float(spread))
  return time, observed, spread

@dataclass(frozen=True, slots=True)
class RadiusObservation:
  """Dimensional radius observations with independent Gaussian noise."""

  time_s: np.ndarray
  radius_m: np.ndarray
  standard_deviation_m: float | np.ndarray

  def __post_init__(self):
    time, radius, deviation = _validate_observation(
      self.time_s, self.radius_m, self.standard_deviation_m,
      minimum=2, value_label="radius_m", deviation_label="standard_deviation_m",
    )
    if np.any(radius <= 0.0): raise ValueError("observed radii must be positive")
    object.__setattr__(self, "time_s", _readonly(time))
    object.__setattr__(self, "radius_m", _readonly(radius))
    object.__setattr__(self, "standard_deviation_m", _readonly(deviation))

_TIME_DERIVATIVE_OF = {"radius_m": "wall_velocity_m_s", "radius_ratio": "wall_velocity_m_s"}
OBSERVABLE_FIELDS = ("radius_m", "radius_ratio", "wall_velocity_m_s", "internal_pressure_pa", "stress_integral_pa")

@dataclass(frozen=True, slots=True)
class FieldObservation:
  """Observations of any scalar trace the solver produces."""

  field: str
  time_s: np.ndarray
  values: np.ndarray
  standard_deviation: float | np.ndarray
  correlation_time_s: float | None = None

  def __post_init__(self):
    if self.field not in OBSERVABLE_FIELDS: raise ValueError(f"field must be one of {OBSERVABLE_FIELDS}, got {self.field!r}")
    time, values, deviation = _validate_observation(
      self.time_s, self.values, self.standard_deviation,
      minimum=1, value_label="values", deviation_label="standard_deviation",
    )
    if self.correlation_time_s is not None:
      correlation = float(self.correlation_time_s)
      if not np.isfinite(correlation) or correlation <= 0.0:
        raise ValueError("correlation_time_s must be finite and positive, or None for independent noise")
      object.__setattr__(self, "correlation_time_s", correlation)
    object.__setattr__(self, "time_s", _readonly(time))
    object.__setattr__(self, "values", _readonly(values))
    object.__setattr__(self, "standard_deviation", _readonly(deviation))

def _whitening_factor(item):
  if item.correlation_time_s is None: return None
  deviation = np.asarray(item.standard_deviation)
  lag = np.abs(item.time_s[:, None] - item.time_s[None, :])
  covariance = np.outer(deviation, deviation) * np.exp(-lag / item.correlation_time_s)
  return np.linalg.cholesky(covariance)

def _whiten(values, item, factor):
  if factor is None:
    deviation = np.asarray(item.standard_deviation)
    return values / (deviation[:, None] if values.ndim == 2 else deviation)
  return solve_triangular(factor, values, lower=True)

def _log_determinant(item, factor):
  count = item.time_s.size
  if factor is None: return float(np.sum(np.log(2.0 * np.pi * np.asarray(item.standard_deviation) ** 2)))
  return float(count * np.log(2.0 * np.pi) + 2.0 * np.sum(np.log(np.diag(factor))))

def _as_field_observations(observation):
  items = observation if isinstance(observation, (tuple, list)) else (observation,)
  if not items: raise ValueError("at least one observation is required")
  normalized = []
  for item in items:
    if isinstance(item, RadiusObservation):
      normalized.append(FieldObservation("radius_m", item.time_s, item.radius_m, item.standard_deviation_m))
    elif isinstance(item, FieldObservation):
      normalized.append(item)
    else:
      raise TypeError("observations must be RadiusObservation or FieldObservation")
  return tuple(normalized)

@dataclass(frozen=True, slots=True)
class LikelihoodEvaluation:
  """One retained likelihood evaluation."""

  unit_parameters: np.ndarray
  physical_parameters: np.ndarray
  residual: np.ndarray
  log_likelihood: float
  stats: pyimr.SolverStats

@dataclass(frozen=True, slots=True)
class MultistartEndpoint:
  """One optimizer endpoint, including unsuccessful starts."""

  start_unit_parameters: np.ndarray
  unit_parameters: np.ndarray
  physical_parameters: np.ndarray
  cost: float
  optimality: float
  evaluations: int
  success: bool
  message: str

@dataclass(frozen=True, slots=True)
class MultistartResult:
  """All deterministic multistart endpoints."""

  endpoints: tuple[MultistartEndpoint, ...]

  @property
  def best(self):
    successful = [endpoint for endpoint in self.endpoints if endpoint.success]
    if not successful: return None
    return min(successful, key=lambda endpoint: endpoint.cost)

@dataclass(frozen=True, slots=True)
class PreparedInference:
  """Reusable parameterization and Gaussian radius likelihood."""

  config: pyimr.SimulationConfig
  observation: RadiusObservation | FieldObservation | tuple
  parameters: tuple[InferenceParameter, ...]
  _observations: tuple = ()
  _grid: np.ndarray = field(default_factory=lambda: np.empty(0))
  _index: tuple = ()
  _whiteners: tuple = ()
  _problem: object = None

  def __post_init__(self):
    if not isinstance(self.config, pyimr.SimulationConfig): raise TypeError("config must be SimulationConfig")
    observations = _as_field_observations(self.observation)
    grid = np.unique(np.concatenate([item.time_s for item in observations]))
    object.__setattr__(self, "_observations", observations)
    object.__setattr__(self, "_grid", _readonly(grid))
    object.__setattr__(self, "_index", tuple(np.searchsorted(grid, item.time_s) for item in observations))
    object.__setattr__(self, "_whiteners", tuple(_whitening_factor(item) for item in observations))
    parameters = tuple(self.parameters)
    if not parameters or not all(isinstance(parameter, InferenceParameter) for parameter in parameters):
      raise TypeError("parameters must contain at least one InferenceParameter")
    paths = [parameter.path for parameter in parameters]
    if len(set(paths)) != len(paths): raise ValueError("inference parameter paths must be unique")
    for parameter in parameters:
      value = _path_value(self.config, _path_parts(parameter.path), parameter.path)
      if not np.isscalar(value) or not np.isfinite(value): raise ValueError(f"{parameter.path!r} must identify a finite scalar field")
      _normalize_parameters(self.config, (SensitivityParameter(parameter.path),))
    object.__setattr__(self, "_problem", pyimr.prepare(self.config))
    object.__setattr__(self, "parameters", parameters)

  @property
  def size(self):
    return len(self.parameters)

  @property
  def _normalization(self):
    return float(sum(_log_determinant(item, factor) for item, factor in zip(self._observations, self._whiteners, strict=True)))

  def physical_parameters(self, unit_parameters):
    unit = self._validate_unit_parameters(unit_parameters)
    return np.array([parameter.physical_value(value) for parameter, value in zip(self.parameters, unit, strict=True)])

  def config_from_unit(self, unit_parameters):
    physical = self.physical_parameters(unit_parameters)
    config = self.config
    for parameter, value in zip(self.parameters, physical, strict=True):
      config = _replace_path(config, _path_parts(parameter.path), value)
    return config

  def _stack_residual(self, result):
    parts = []
    for item, index, factor in zip(self._observations, self._index, self._whiteners, strict=True):
      predicted = np.asarray(getattr(result, item.field))[index]
      parts.append(_whiten(predicted - item.values, item, factor))
    return np.concatenate(parts)

  @property
  def observation_size(self):
    """Total number of observed values, across every field."""
    return sum(item.time_s.size for item in self._observations)

  def _evaluation(self, unit, residual, stats):
    return LikelihoodEvaluation(
      unit_parameters=_readonly(unit),
      physical_parameters=_readonly(self.physical_parameters(unit)),
      residual=_readonly(residual),
      log_likelihood=float(-0.5 * (np.sum(residual**2) + self._normalization)),
      stats=stats,
    )

  def residual(self, unit_parameters):
    config = self.config_from_unit(unit_parameters)
    return self._stack_residual(pyimr.simulate(self._grid, config))

  def jacobian(self, unit_parameters):
    unit = self._validate_unit_parameters(unit_parameters)
    return self.jacobians(np.atleast_2d(unit))[0]

  def _traced_at(self, points, values_only):
    from ._jax import sensitivities_jax

    points = np.atleast_2d(np.asarray(points, dtype=float))
    if points.shape[1] != self.size: raise ValueError(f"each point needs {self.size} unit parameters; got {points.shape}")
    physical = np.array([[parameter.physical_value(value) for parameter, value in zip(self.parameters, point, strict=True)] for point in points])
    paths = [parameter.path for parameter in self.parameters]
    return sensitivities_jax(self._problem, self._grid, paths, at=physical, values_only=values_only)

  @staticmethod
  def _by_field(group):
    from ._jax import _DERIVED_ORDER

    derived = np.asarray(group.derived)
    fields = {name: derived[:, :, index] for index, name in enumerate(_DERIVED_ORDER)}
    for name, array in (("bubble_temperature_k", group.bubble_temperature),
                        ("medium_temperature_k", group.medium_temperature),
                        ("vapor_mass_fraction", group.vapor_fraction)):
      if array is not None: fields[name] = np.asarray(array)
    return fields

  def _traced_values(self, points):
    values, _ = self._traced_at(points, True)
    return self._by_field(values)

  def _traced_fields(self, points):
    values, tangents = self._traced_at(points, False)
    assert tangents is not None  # noqa: S101 - values_only=False always returns both
    return self._by_field(values), self._by_field(tangents)

  def _stacked(self, kind, fields, chain=None, subtract=False):
    parts = []
    for item, index, factor in zip(self._observations, self._index, self._whiteners, strict=True):
      if item.field not in fields: raise NotImplementedError(f"no traced {kind} for observation field {item.field!r}")
      selected = fields[item.field][index]
      if subtract: selected = selected - item.values
      whitened = _whiten(selected, item, factor)
      parts.append(whitened if chain is None else whitened * chain)
    return np.concatenate(parts, axis=0)

  def jacobians(self, unit_points):
    """`J` for many draws, from one traced program where the backend allows it."""
    points = np.atleast_2d(np.asarray(unit_points, dtype=float))
    for point in points: self._validate_unit_parameters(point)
    _values, tangents = self._traced_fields(points)
    stacked = []
    for draw, point in enumerate(points):
      chain = np.array([parameter.derivative(value) for parameter, value in zip(self.parameters, point, strict=True)])
      stacked.append(self._stacked("tangent", {name: array[draw] for name, array in tangents.items()}, chain))
    return np.array(stacked)

  def evaluate(self, unit_parameters):
    unit = self._validate_unit_parameters(unit_parameters)
    config = self.config_from_unit(unit)
    result = pyimr.simulate(self._grid, config)
    return self._evaluation(unit, self._stack_residual(result), result.stats)

  def evaluate_with_jacobian(self, unit_parameters):
    """Likelihood and Jacobian from a single sensitivity solve."""
    unit = self._validate_unit_parameters(unit_parameters)
    values, tangents = self._traced_fields(np.atleast_2d(unit))
    chain = np.array([parameter.derivative(value) for parameter, value in zip(self.parameters, unit, strict=True)])
    residual = self._stacked("value", {name: array[0] for name, array in values.items()}, subtract=True)
    jacobian = self._stacked("tangent", {name: array[0] for name, array in tangents.items()}, chain)
    stats = pyimr.SolverStats(
      backend="jax-tsit5-forward", success=True, message="jacfwd through the diffrax solve", nfev=0, njev=0, nlu=0, elapsed_s=0.0
    )
    return self._evaluation(unit, residual, stats), jacobian

  def jacobian_with_time_derivative(self, unit_parameters):
    """`(J, dJ/dt)` from one solve. `dJ/dt` is what a design needs -- what a design needs to move its own"""
    unit = self._validate_unit_parameters(unit_parameters)
    missing = sorted({item.field for item in self._observations} - set(_TIME_DERIVATIVE_OF))
    if missing: raise NotImplementedError(f"no time derivative available for {missing}; only {sorted(_TIME_DERIVATIVE_OF)} are supported")
    _values, tangents = self._traced_fields(np.atleast_2d(unit))
    chain = np.array([parameter.derivative(value) for parameter, value in zip(self.parameters, unit, strict=True)])
    fields = {name: array[0] for name, array in tangents.items()}
    parts = []
    for item, index, factor in zip(self._observations, self._index, self._whiteners, strict=True):
      tangent = fields[_TIME_DERIVATIVE_OF[item.field]][index]
      if item.field == "radius_ratio": tangent = tangent / self.config.R0
      parts.append(_whiten(tangent, item, factor) * chain)
    return self._stacked("tangent", fields, chain), np.concatenate(parts, axis=0)

  def jacobian_time_derivative(self, unit_parameters):
    """Just the `d/dt` half; see :meth:`jacobian_with_time_derivative`."""
    return self.jacobian_with_time_derivative(unit_parameters)[1]

  def curvature_ratio(self, unit_parameters, step=1e-5):
    """`||sum_k r_k H^k|| / ||J^T J||` -- the size of what Gauss-Newton drops."""
    unit = self._validate_unit_parameters(unit_parameters)
    residual = np.asarray(self.evaluate(unit).residual)
    jacobian = np.asarray(self.jacobian(unit))
    dropped = np.zeros((self.size, self.size))
    for index in range(self.size):
      offset = np.zeros(self.size)
      offset[index] = step
      ahead = np.asarray(self.jacobian(np.clip(unit + offset, 0.0, 1.0)))
      behind = np.asarray(self.jacobian(np.clip(unit - offset, 0.0, 1.0)))
      dropped[index] = residual @ ((ahead - behind) / (2.0 * step))
    dropped = 0.5 * (dropped + dropped.T)
    return float(np.linalg.norm(dropped) / np.linalg.norm(jacobian.T @ jacobian))

  def evaluate_batch(self, unit_parameters, workers=1):
    points = self._validate_batch(unit_parameters)
    if not isinstance(workers, Integral) or workers < 1: raise ValueError("workers must be a positive integer")
    # One traced program over the stack, not one per point. Every point is a distinct
    # configuration, so the per-point route missed the compile cache every time and
    # retraced: 1053 ms against 6.9 ms batched (#129).
    if workers == 1: return self._evaluate_batched(points)
    with ProcessPoolExecutor(max_workers=workers) as executor:
      return tuple(executor.map(_evaluate_worker, repeat(self), points))

  def _evaluate_batched(self, points):
    values = self._traced_values(points)
    stats = pyimr.SolverStats(
      backend="jax-batched", success=True, message="one traced program over the batch", nfev=0, njev=0, nlu=0, elapsed_s=0.0
    )
    return tuple(
      self._evaluation(point, self._stacked("value", {n: a[k] for n, a in values.items()}, subtract=True), stats)
      for k, point in enumerate(points)
    )

  def fit_multistart(self, starts, *, seed=0, max_evaluations=200, workers=1):
    if not isinstance(starts, Integral) or starts < 1: raise ValueError("starts must be a positive integer")
    if not isinstance(max_evaluations, Integral) or max_evaluations < 1: raise ValueError("max_evaluations must be a positive integer")
    if not isinstance(workers, Integral) or workers < 1: raise ValueError("workers must be a positive integer")
    # `rng=`, not the legacy `seed=`: scipy keeps `seed` only as an alias and will drop it.
    # They are NOT equivalent -- `rng=n` seeds `default_rng(n)`, so the start points moved.
    sampler = qmc.LatinHypercube(d=self.size, rng=seed)
    start_points = sampler.random(int(starts))
    start_points[0] = 0.5
    arguments = [(self, point, max_evaluations) for point in start_points]
    if workers == 1:
      endpoints = tuple(_fit_worker(argument) for argument in arguments)
    else:
      with ProcessPoolExecutor(max_workers=workers) as executor:
        endpoints = tuple(executor.map(_fit_worker, arguments))
    return MultistartResult(endpoints=endpoints)

  def _validate_unit_parameters(self, unit_parameters):
    unit = np.asarray(unit_parameters, dtype=float)
    if unit.shape != (self.size,): raise ValueError(f"unit parameters must have shape ({self.size},)")
    if not np.all(np.isfinite(unit)) or np.any((unit < 0.0) | (unit > 1.0)): raise ValueError("unit parameters must be finite and within [0, 1]")
    return unit

  def _validate_batch(self, unit_parameters):
    points = np.asarray(unit_parameters, dtype=float)
    if points.ndim != 2 or points.shape[1] != self.size: raise ValueError(f"batch parameters must have shape (n, {self.size})")
    if not np.all(np.isfinite(points)) or np.any((points < 0.0) | (points > 1.0)):
      raise ValueError("batch parameters must be finite and within [0, 1]")
    return points

def _evaluate_worker(inference, point): return inference.evaluate(point)

def _fit_worker(argument):
  inference, start, max_evaluations = argument
  try:
    result = least_squares(inference.residual, start, jac=inference.jacobian, bounds=(0.0, 1.0), max_nfev=max_evaluations, x_scale="jac")
    physical = inference.physical_parameters(result.x)
    return MultistartEndpoint(
      start_unit_parameters=_readonly(start),
      unit_parameters=_readonly(result.x),
      physical_parameters=_readonly(physical),
      cost=float(result.cost),
      optimality=float(result.optimality),
      evaluations=int(result.nfev),
      success=bool(result.success),
      message=str(result.message),
    )
  except Exception as error:
    return MultistartEndpoint(
      start_unit_parameters=_readonly(start),
      unit_parameters=_readonly(start),
      physical_parameters=_readonly(inference.physical_parameters(start)),
      cost=float("inf"),
      optimality=float("inf"),
      evaluations=0,
      success=False,
      message=f"{type(error).__name__}: {error}",
    )

def prepare_inference(config, observation, parameters):
  """Prepare a reusable IMR likelihood and parameterization."""
  return PreparedInference(config, observation, tuple(parameters))
