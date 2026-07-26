"""Prepared likelihood, batch evaluation, and deterministic multistart tools."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from itertools import repeat
from numbers import Integral

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import qmc

import imr_fast
from imr_sensitivity import SensitivityParameter, _normalize_parameters

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
  observation: RadiusObservation
  parameters: tuple[InferenceParameter, ...]

  def __post_init__(self):
    if not isinstance(self.config, imr_fast.SimulationConfig):
      raise TypeError("config must be SimulationConfig")
    if not isinstance(self.observation, RadiusObservation):
      raise TypeError("observation must be RadiusObservation")
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

  def physical_parameters(self, unit_parameters):
    unit = self._validate_unit_parameters(unit_parameters)
    return np.array([parameter.physical_value(value) for parameter, value in zip(self.parameters, unit, strict=True)])

  def config_from_unit(self, unit_parameters):
    physical = self.physical_parameters(unit_parameters)
    config = self.config
    for parameter, value in zip(self.parameters, physical, strict=True):
      config = _replace_path(config, _path_parts(parameter.path), value)
    return config

  def residual(self, unit_parameters):
    config = self.config_from_unit(unit_parameters)
    result = imr_fast.simulate(self.observation.time_s, config)
    return (result.radius_m - self.observation.radius_m) / self.observation.standard_deviation_m

  def jacobian(self, unit_parameters):
    unit = self._validate_unit_parameters(unit_parameters)
    config = self.config_from_unit(unit)
    result = imr_fast.simulate_with_sensitivities(
      self.observation.time_s, config, [parameter.path for parameter in self.parameters]
    )
    chain = np.array([parameter.derivative(value) for parameter, value in zip(self.parameters, unit, strict=True)])
    return result.radius_m / self.observation.standard_deviation_m[:, None] * chain

  def evaluate(self, unit_parameters):
    unit = self._validate_unit_parameters(unit_parameters)
    config = self.config_from_unit(unit)
    result = imr_fast.simulate(self.observation.time_s, config)
    residual = (result.radius_m - self.observation.radius_m) / self.observation.standard_deviation_m
    normalization = np.log(2.0 * np.pi * self.observation.standard_deviation_m**2)
    log_likelihood = -0.5 * np.sum(residual**2 + normalization)
    return LikelihoodEvaluation(
      unit_parameters=_readonly(unit),
      physical_parameters=_readonly(self.physical_parameters(unit)),
      residual=_readonly(residual),
      log_likelihood=float(log_likelihood),
      stats=result.stats,
    )

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
