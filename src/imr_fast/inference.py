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
  # Shared shape/finiteness/ordering checks for an observed series, returning the three as float arrays with a scalar
  # deviation broadcast to match. `RadiusObservation` and `FieldObservation` had six of these checks each, written
  # twice -- and had already drifted: the radius form folded `deviation > 0` into its finiteness check, so a sigma of
  # exactly 0.0 was rejected with "observations and deviations must be finite", which is not true of 0.0.
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
  # Lower Cholesky factor of the noise covariance, or None when it is diagonal.
  #
  # With `Sigma = L L^T`, the whitened residual `L^-1 (y - m)` and Jacobian `L^-1 dm/dtheta` mean exactly what the
  # independent-noise versions meant, so everything downstream -- the `-r @ J` gradient, `J^T J` as the Fisher
  # information, EIG -- is unchanged. Correlated noise is a change of one division into one triangular solve.
  #
  # The kernel is exponential, `exp(-|t_i - t_j| / tau)`. Radii recovered by edge detection are correlated over
  # roughly a frame or two, and an exponential is the one-parameter model of that; it is also positive definite for
  # any tau, so the factorisation cannot fail on a user's choice.
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
  # `log det(2 pi Sigma)` for one observation.
  count = item.time_s.size
  if factor is None: return float(np.sum(np.log(2.0 * np.pi * np.asarray(item.standard_deviation) ** 2)))
  return float(count * np.log(2.0 * np.pi) + 2.0 * np.sum(np.log(np.diag(factor))))

def _as_field_observations(observation):
  # Normalise one observation, or several, into a tuple of `FieldObservation`.
  #
  # `RadiusObservation` is kept rather than deprecated: it carries the positivity check that only makes sense for a
  # radius, and it is the overwhelmingly common case.
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
    if not successful: return None
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
  _whiteners: tuple = ()
  # Held rather than rebuilt. `sensitivities_jax` caches its compiled program on the
  # problem's identity, so calling `prepare` per evaluation guarantees a miss and a
  # full retrace -- ~2.5 s against 5.6 ms. `__post_init__` already prepared one to
  # validate the configuration; keeping it is what makes `jacobians` worth having.
  _problem: object = None

  def __post_init__(self):
    if not isinstance(self.config, imr_fast.SimulationConfig): raise TypeError("config must be SimulationConfig")
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
    object.__setattr__(self, "_problem", imr_fast.prepare(self.config))
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
    # The Gaussian log-likelihood and the value object around it, in one place.
    # `_normalization` carries `sum(log(2*pi*sigma**2))`, or `log det(2*pi*Sigma)`
    # when the noise is correlated -- a constant the posterior never sees but the
    # marginal likelihood does, so it is worth having exactly one copy of.
    return LikelihoodEvaluation(
      unit_parameters=_readonly(unit),
      physical_parameters=_readonly(self.physical_parameters(unit)),
      residual=_readonly(residual),
      log_likelihood=float(-0.5 * (np.sum(residual**2) + self._normalization)),
      stats=stats,
    )

  def residual(self, unit_parameters):
    config = self.config_from_unit(unit_parameters)
    return self._stack_residual(imr_fast.simulate(self._grid, config))

  def jacobian(self, unit_parameters):
    unit = self._validate_unit_parameters(unit_parameters)
    return self.jacobians(np.atleast_2d(unit))[0]

  def _traced_fields(self, points):
    """`(values, tangents)` by output field for many parameter points, from ONE
    traced program.

    The draws differ only in the values of the inference parameters, which the traced
    path takes as an ARGUMENT -- so one `prepare` at the reference configuration
    serves all of them and `sensitivities_jax` maps over the points. Building a
    configuration per point and preparing it, which is what `config_from_unit` plus
    `simulate_with_sensitivities` does, forces a distinct traced program per point
    instead: 2.5 s against 5.6 ms.
    """
    from ._jax import _DERIVED_ORDER, sensitivities_jax

    points = np.atleast_2d(np.asarray(points, dtype=float))
    if points.shape[1] != self.size: raise ValueError(f"each point needs {self.size} unit parameters; got {points.shape}")
    physical = np.array([[parameter.physical_value(value) for parameter, value in zip(self.parameters, point, strict=True)] for point in points])
    paths = [parameter.path for parameter in self.parameters]
    values, tangents = sensitivities_jax(self._problem, self._grid, paths, at=physical)

    def by_field(group, derived):
      fields = {name: derived[:, :, index] for index, name in enumerate(_DERIVED_ORDER)}
      for name, array in (("bubble_temperature_k", group.bubble_temperature),
                          ("medium_temperature_k", group.medium_temperature),
                          ("vapor_mass_fraction", group.vapor_fraction)):
        if array is not None: fields[name] = np.asarray(array)
      return fields

    return by_field(values, np.asarray(values.derived)), by_field(tangents, np.asarray(tangents.derived))

  def _stacked(self, kind, fields, chain=None, subtract=False):
    """One observation-stacked block, whitened, for a single point.

    `subtract` reproduces `_stack_residual`: the whitening is applied to the
    DIFFERENCE from the observed values, not to the prediction and the data
    separately. It is linear either way, so the two agree -- but matching the shape of
    the original leaves nothing to argue about.
    """
    parts = []
    for item, index, factor in zip(self._observations, self._index, self._whiteners, strict=True):
      if item.field not in fields: raise NotImplementedError(f"no traced {kind} for observation field {item.field!r}")
      selected = fields[item.field][index]
      if subtract: selected = selected - item.values
      whitened = _whiten(selected, item, factor)
      parts.append(whitened if chain is None else whitened * chain)
    return np.concatenate(parts, axis=0)

  def jacobians(self, unit_points):
    """`J` for many draws, from one traced program where the backend allows it.

    Returns `(draws, observations, parameters)`. Equal to stacking `jacobian` over the
    same points to 6e-13 relative on the same backend, which is the check that
    licenses skipping the per-draw `config_from_unit`.

    """
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
    result = imr_fast.simulate(self._grid, config)
    return self._evaluation(unit, self._stack_residual(result), result.stats)

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
    # Both halves out of ONE traced program, and out of one integration, which is what
    # this method existed for before it was traced: a sampler wants the gradient to be
    # the derivative of the log-likelihood it accompanies, not of a separately
    # converged one.
    values, tangents = self._traced_fields(np.atleast_2d(unit))
    chain = np.array([parameter.derivative(value) for parameter, value in zip(self.parameters, unit, strict=True)])
    residual = self._stacked("value", {name: array[0] for name, array in values.items()}, subtract=True)
    jacobian = self._stacked("tangent", {name: array[0] for name, array in tangents.items()}, chain)
    # The traced solve reports no step counts, so the stats say what they can say --
    # the same placeholder `sensitivity._jax_sensitivities` records.
    stats = imr_fast.SolverStats(
      backend="jax-tsit5-forward", success=True, message="jacfwd through the diffrax solve", nfev=0, njev=0, nlu=0, elapsed_s=0.0
    )
    return self._evaluation(unit, residual, stats), jacobian

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
    if missing: raise NotImplementedError(f"no time derivative available for {missing}; only {sorted(_TIME_DERIVATIVE_OF)} are supported")
    _values, tangents = self._traced_fields(np.atleast_2d(unit))
    chain = np.array([parameter.derivative(value) for parameter, value in zip(self.parameters, unit, strict=True)])
    fields = {name: array[0] for name, array in tangents.items()}
    parts = []
    for item, index, factor in zip(self._observations, self._index, self._whiteners, strict=True):
      tangent = fields[_TIME_DERIVATIVE_OF[item.field]][index]
      # `wall_velocity_m_s` is dimensional and `radius_ratio` is not, so the derivative
      # of the ratio needs the length scale removed.
      if item.field == "radius_ratio": tangent = tangent / self.config.R0
      parts.append(_whiten(tangent, item, factor) * chain)
    return self._stacked("tangent", fields, chain), np.concatenate(parts, axis=0)

  def jacobian_time_derivative(self, unit_parameters):
    """Just the `d/dt` half; see :meth:`jacobian_with_time_derivative`."""
    return self.jacobian_with_time_derivative(unit_parameters)[1]

  def curvature_ratio(self, unit_parameters, step=1e-5):
    """`||sum_k r_k H^k|| / ||J^T J||` -- the size of what Gauss-Newton drops.

    The exact Hessian of the Gaussian log-likelihood is

        -d2L = J^T J + sum_k r_k H^k,     H^k = d2 r_k / dtheta2

    and `J^T J` alone is the Gauss-Newton approximation used throughout this
    package. **For expected information gain that is not an approximation at
    all**: EIG needs the Fisher information `E[-d2L]`, and a correctly specified
    model has `E[r_k] = 0`, so the dropped term has zero expectation. `design.py`
    never sees data, so the term cannot enter there.

    It does matter for the *observed* information -- a Laplace posterior at real
    data -- and it is a misspecification diagnostic, which is the more useful
    reading. Measured on synthetic data:

        correct model, at the truth            3.4e-04
        correct model, 30% off in G            6.1e-03
        correct model, 30% off in both         7.5e-01
        Zener data fitted with NHKV            2.8e-01

    A large value away from the optimum is expected. A large value *at* a fit is
    evidence the model cannot represent the data.

    `H` is taken by central difference of the exact Jacobian, so this costs `2p`
    sensitivity solves and no second-order machinery. That is the whole reason
    it exists in this form: knowing when Gauss-Newton is inadequate is cheap,
    while fixing it would mean second-order sensitivities.
    """
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
    if workers == 1: return tuple(self.evaluate(point) for point in points)
    with ProcessPoolExecutor(max_workers=workers) as executor:
      return tuple(executor.map(_evaluate_worker, repeat(self), points))

  def fit_multistart(self, starts, *, seed=0, max_evaluations=200, workers=1):
    if not isinstance(starts, Integral) or starts < 1: raise ValueError("starts must be a positive integer")
    if not isinstance(max_evaluations, Integral) or max_evaluations < 1: raise ValueError("max_evaluations must be a positive integer")
    if not isinstance(workers, Integral) or workers < 1: raise ValueError("workers must be a positive integer")
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
