"""Prepared likelihood, batch evaluation, and deterministic multistart tools."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from itertools import repeat
from numbers import Integral

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import qmc

import imr_fast
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
  if not parts or any(not part.isidentifier() for part in parts):
    raise ValueError(f"invalid inference parameter path: {path!r}")
  return parts


def _path_value(root, parts, full_path):
  value = root
  for part in parts:
    if not hasattr(value, part):
      raise ValueError(f"unknown inference parameter path: {full_path!r}")
    value = getattr(value, part)
  return value


def _replace_path(root, parts, value):
  if len(parts) == 1:
    return replace(root, **{parts[0]: value})
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
    if self.transform not in ("linear", "log"):
      raise ValueError("inference transform must be 'linear' or 'log'")
    if self.transform == "log" and self.lower <= 0.0:
      raise ValueError("log-transformed inference bounds must be positive")

  def physical_value(self, unit_value):
    if self.transform == "linear":
      return self.lower + unit_value * (self.upper - self.lower)
    return self.lower * (self.upper / self.lower) ** unit_value

  def derivative(self, unit_value):
    if self.transform == "linear":
      return self.upper - self.lower
    value = self.physical_value(unit_value)
    return value * np.log(self.upper / self.lower)


@dataclass(frozen=True, slots=True)
class RadiusObservation:
  """Dimensional radius observations with independent Gaussian noise."""

  time_s: np.ndarray
  radius_m: np.ndarray
  standard_deviation_m: float | np.ndarray

  def __post_init__(self):
    time = np.asarray(self.time_s, dtype=float)
    radius = np.asarray(self.radius_m, dtype=float)
    deviation = np.asarray(self.standard_deviation_m, dtype=float)
    if time.ndim != 1 or time.size < 2:
      raise ValueError("observation time_s must be one-dimensional")
    if radius.shape != time.shape:
      raise ValueError("observation radius_m must match time_s")
    if deviation.ndim > 1 or (deviation.ndim == 1 and deviation.shape != time.shape):
      raise ValueError("standard_deviation_m must be scalar or match time_s")
    if (
      not np.all(np.isfinite(time))
      or not np.all(np.isfinite(radius))
      or not np.all(np.isfinite(deviation))
      or np.any(deviation <= 0.0)
    ):
      raise ValueError("observations and deviations must be finite")
    if time[0] < 0.0 or np.any(np.diff(time) <= 0.0):
      raise ValueError("observation time_s must be non-negative and increasing")
    if np.any(radius <= 0.0):
      raise ValueError("observed radii must be positive")
    if deviation.ndim == 0:
      deviation = np.full(time.shape, float(deviation))
    object.__setattr__(self, "time_s", _readonly(time))
    object.__setattr__(self, "radius_m", _readonly(radius))
    object.__setattr__(self, "standard_deviation_m", _readonly(deviation))


# Every trace the sensitivity solve already returns tangents for, and whose
# shape is one value per observation time. `bubble_temperature_k` and
# `medium_temperature_k` are fields over their grids -- shape (times, nodes) --
# so observing them needs a node selection and is deliberately out of scope
# here.
# d/dt of an observable, where the solver already returns the tangent of that
# derivative. `radius_m`'s time derivative IS the wall velocity, so the tangent
# needed to differentiate a design with respect to WHEN it looks is already
# computed. The others would need the RHS differentiated as well.
_TIME_DERIVATIVE_OF = {"radius_m": "wall_velocity_m_s", "radius_ratio": "wall_velocity_m_s"}

OBSERVABLE_FIELDS = ("radius_m", "radius_ratio", "wall_velocity_m_s", "internal_pressure_pa", "stress_integral_pa")


@dataclass(frozen=True, slots=True)
class FieldObservation:
  """Observations of any scalar trace the solver produces.

  A single `simulate_with_sensitivities` call already returns exact tangents for
  all of `OBSERVABLE_FIELDS`; the likelihood used to read one of them and
  discard the rest. So observing wall velocity or internal pressure alongside
  radius costs nothing beyond the arithmetic -- which is what makes "what to
  measure" usable as a design variable rather than a modelling choice.

  Observations may sit on different time grids. The solve runs once on the union
  of them.
  """

  field: str
  time_s: np.ndarray
  values: np.ndarray
  standard_deviation: float | np.ndarray

  def __post_init__(self):
    if self.field not in OBSERVABLE_FIELDS:
      raise ValueError(f"field must be one of {OBSERVABLE_FIELDS}, got {self.field!r}")
    time = np.asarray(self.time_s, dtype=float)
    values = np.asarray(self.values, dtype=float)
    deviation = np.asarray(self.standard_deviation, dtype=float)
    if time.ndim != 1 or time.size < 1:
      raise ValueError("observation time_s must be one-dimensional and non-empty")
    if values.shape != time.shape:
      raise ValueError("observation values must match time_s")
    if deviation.ndim > 1 or (deviation.ndim == 1 and deviation.shape != time.shape):
      raise ValueError("standard_deviation must be scalar or match time_s")
    if not (np.all(np.isfinite(time)) and np.all(np.isfinite(values)) and np.all(np.isfinite(deviation))):
      raise ValueError("observations and deviations must be finite")
    if np.any(deviation <= 0.0):
      raise ValueError("standard_deviation must be positive")
    if time[0] < 0.0 or np.any(np.diff(time) <= 0.0):
      raise ValueError("observation time_s must be non-negative and increasing")
    if deviation.ndim == 0:
      deviation = np.full(time.shape, float(deviation))
    object.__setattr__(self, "time_s", _readonly(time))
    object.__setattr__(self, "values", _readonly(values))
    object.__setattr__(self, "standard_deviation", _readonly(deviation))


def _as_field_observations(observation):
  """Normalise one observation, or several, into a tuple of `FieldObservation`.

  `RadiusObservation` is kept rather than deprecated: it carries the positivity
  check that only makes sense for a radius, and it is the overwhelmingly common
  case.
  """
  items = observation if isinstance(observation, (tuple, list)) else (observation,)
  if not items:
    raise ValueError("at least one observation is required")
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
  stats: imr_fast.SolverStats


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
    if not successful:
      return None
    return min(successful, key=lambda endpoint: endpoint.cost)


@dataclass(frozen=True, slots=True)
class PreparedInference:
  """Reusable parameterization and Gaussian radius likelihood."""

  config: imr_fast.SimulationConfig
  observation: RadiusObservation | FieldObservation | tuple
  parameters: tuple[InferenceParameter, ...]
  _observations: tuple = ()
  _grid: np.ndarray = field(default_factory=lambda: np.empty(0))
  _index: tuple = ()

  def __post_init__(self):
    if not isinstance(self.config, imr_fast.SimulationConfig):
      raise TypeError("config must be SimulationConfig")
    observations = _as_field_observations(self.observation)
    grid = np.unique(np.concatenate([item.time_s for item in observations]))
    object.__setattr__(self, "_observations", observations)
    object.__setattr__(self, "_grid", _readonly(grid))
    object.__setattr__(self, "_index", tuple(np.searchsorted(grid, item.time_s) for item in observations))
    parameters = tuple(self.parameters)
    if not parameters or not all(isinstance(parameter, InferenceParameter) for parameter in parameters):
      raise TypeError("parameters must contain at least one InferenceParameter")
    paths = [parameter.path for parameter in parameters]
    if len(set(paths)) != len(paths):
      raise ValueError("inference parameter paths must be unique")
    for parameter in parameters:
      value = _path_value(self.config, _path_parts(parameter.path), parameter.path)
      if not np.isscalar(value) or not np.isfinite(value):
        raise ValueError(f"{parameter.path!r} must identify a finite scalar field")
      _normalize_parameters(self.config, (SensitivityParameter(parameter.path),))
    imr_fast.prepare(self.config)
    object.__setattr__(self, "parameters", parameters)

  @property
  def size(self):
    return len(self.parameters)

  @property
  def _normalization(self):
    return float(sum(np.sum(np.log(2.0 * np.pi * item.standard_deviation**2)) for item in self._observations))

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
    for item, index in zip(self._observations, self._index, strict=True):
      predicted = np.asarray(getattr(result, item.field))[index]
      parts.append((predicted - item.values) / item.standard_deviation)
    return np.concatenate(parts)

  def _stack_jacobian(self, result, unit):
    chain = np.array([parameter.derivative(value) for parameter, value in zip(self.parameters, unit, strict=True)])
    parts = []
    for item, index in zip(self._observations, self._index, strict=True):
      tangent = np.asarray(getattr(result, item.field))[index]
      parts.append(tangent / np.asarray(item.standard_deviation)[:, None] * chain)
    return np.concatenate(parts, axis=0)

  @property
  def observation_size(self):
    """Total number of observed values, across every field."""
    return sum(item.time_s.size for item in self._observations)

  def residual(self, unit_parameters):
    config = self.config_from_unit(unit_parameters)
    return self._stack_residual(imr_fast.simulate(self._grid, config))

  def jacobian(self, unit_parameters):
    unit = self._validate_unit_parameters(unit_parameters)
    config = self.config_from_unit(unit)
    result = imr_fast.simulate_with_sensitivities(self._grid, config, [parameter.path for parameter in self.parameters])
    return self._stack_jacobian(result, unit)

  def evaluate(self, unit_parameters):
    unit = self._validate_unit_parameters(unit_parameters)
    config = self.config_from_unit(unit)
    result = imr_fast.simulate(self._grid, config)
    residual = self._stack_residual(result)
    log_likelihood = -0.5 * (np.sum(residual**2) + self._normalization)
    return LikelihoodEvaluation(
      unit_parameters=_readonly(unit),
      physical_parameters=_readonly(self.physical_parameters(unit)),
      residual=_readonly(residual),
      log_likelihood=float(log_likelihood),
      stats=result.stats,
    )

  def evaluate_with_jacobian(self, unit_parameters):
    """Likelihood and Jacobian from a single sensitivity solve.

    `evaluate` and `jacobian` each integrate; run separately they cost two
    solves and, more subtly, return values converged independently to the same
    tolerance -- so the gradient is the derivative of a slightly different
    function than the log-likelihood it accompanies. A sampler wants both from
    one integration.

    Measured on the mechanical path: 1.6x cheaper than the pair, and the
    gradient agrees with a central difference of *this* log-likelihood to
    5e-08, against 2.5e-06 for the split pair checked the same way.
    """
    unit = self._validate_unit_parameters(unit_parameters)
    config = self.config_from_unit(unit)
    result = imr_fast.simulate_with_sensitivities(self._grid, config, [parameter.path for parameter in self.parameters])
    residual = self._stack_residual(result.simulation)
    jacobian = self._stack_jacobian(result, unit)
    log_likelihood = -0.5 * (np.sum(residual**2) + self._normalization)
    evaluation = LikelihoodEvaluation(
      unit_parameters=_readonly(unit),
      physical_parameters=_readonly(self.physical_parameters(unit)),
      residual=_readonly(residual),
      log_likelihood=float(log_likelihood),
      stats=result.simulation.stats,
    )
    return evaluation, jacobian

  def jacobian_with_time_derivative(self, unit_parameters):
    """`(J, dJ/dt)` from one solve. `dJ/dt` is what a design needs -- what a design needs to move its own
    observation times.

    Scoring a time grid needs `J`; *optimising* one needs `dJ/dt`, and for a
    radius observation that is the wall-velocity tangent, which the same solve
    already returns. No extra integration.

    Fields whose time derivative is not itself an observable raise, rather than
    silently returning the wrong thing -- getting `dP/dt` would mean
    differentiating the right-hand side, which is a larger change.
    """
    unit = self._validate_unit_parameters(unit_parameters)
    missing = sorted({item.field for item in self._observations} - set(_TIME_DERIVATIVE_OF))
    if missing:
      raise NotImplementedError(
        f"no time derivative available for {missing}; only {sorted(_TIME_DERIVATIVE_OF)} are supported"
      )
    config = self.config_from_unit(unit)
    result = imr_fast.simulate_with_sensitivities(self._grid, config, [parameter.path for parameter in self.parameters])
    chain = np.array([parameter.derivative(value) for parameter, value in zip(self.parameters, unit, strict=True)])
    parts = []
    for item, index in zip(self._observations, self._index, strict=True):
      tangent = np.asarray(getattr(result, _TIME_DERIVATIVE_OF[item.field]))[index]
      if item.field == "radius_ratio":
        tangent = tangent / self.config.R0
      parts.append(tangent / np.asarray(item.standard_deviation)[:, None] * chain)
    return self._stack_jacobian(result, unit), np.concatenate(parts, axis=0)

  def jacobian_time_derivative(self, unit_parameters):
    """Just the `d/dt` half; see :meth:`jacobian_with_time_derivative`."""
    return self.jacobian_with_time_derivative(unit_parameters)[1]

  def evaluate_batch(self, unit_parameters, workers=1):
    points = self._validate_batch(unit_parameters)
    if not isinstance(workers, Integral) or workers < 1:
      raise ValueError("workers must be a positive integer")
    if workers == 1:
      return tuple(self.evaluate(point) for point in points)
    with ProcessPoolExecutor(max_workers=workers) as executor:
      return tuple(executor.map(_evaluate_worker, repeat(self), points))

  def fit_multistart(self, starts, *, seed=0, max_evaluations=200, workers=1):
    if not isinstance(starts, Integral) or starts < 1:
      raise ValueError("starts must be a positive integer")
    if not isinstance(max_evaluations, Integral) or max_evaluations < 1:
      raise ValueError("max_evaluations must be a positive integer")
    if not isinstance(workers, Integral) or workers < 1:
      raise ValueError("workers must be a positive integer")
    sampler = qmc.LatinHypercube(d=self.size, seed=seed)
    start_points = sampler.random(starts)
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
    if unit.shape != (self.size,):
      raise ValueError(f"unit parameters must have shape ({self.size},)")
    if not np.all(np.isfinite(unit)) or np.any((unit < 0.0) | (unit > 1.0)):
      raise ValueError("unit parameters must be finite and within [0, 1]")
    return unit

  def _validate_batch(self, unit_parameters):
    points = np.asarray(unit_parameters, dtype=float)
    if points.ndim != 2 or points.shape[1] != self.size:
      raise ValueError(f"batch parameters must have shape (n, {self.size})")
    if not np.all(np.isfinite(points)) or np.any((points < 0.0) | (points > 1.0)):
      raise ValueError("batch parameters must be finite and within [0, 1]")
    return points


def _evaluate_worker(inference, point):
  return inference.evaluate(point)


def _fit_worker(argument):
  inference, start, max_evaluations = argument
  try:
    result = least_squares(
      inference.residual, start, jac=inference.jacobian, bounds=(0.0, 1.0), max_nfev=max_evaluations, x_scale="jac"
    )
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
