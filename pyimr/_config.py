"""Configuration, prepared-problem and result value objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral

import numpy as np

from typing import get_args

from ._materials import (
  Giesekus,
  LinearPTT,
  MaterialModel,
  OldroydB,
  QuadraticZener,
  Zener,
  _finite_positive,
  _stress_state_count,
)
from ._thermal import _GAM_TAIT, _HUGONIOT_S, _NSTATE_TAIT

__all__ = [
  "C8",
  "CollapseInitialization",
  "CollapseStats",
  "InitialState",
  "KAPPA",
  "MediumOperators",
  "P8",
  "PhysicalParameters",
  "PreparedDistributedStress",
  "PreparedForcing",
  "PreparedInstantaneousMaterial",
  "RHO",
  "SURF",
  "SampledForcing",
  "SimulationConfig",
  "SimulationError",
  "SimulationResult",
  "SolverStats",
  "StateLayout",
  "SweepResult",
  "_CP",
  "_D0",
  "_KM",
  "_LHEAT",
  "_LT",
  "_MWG",
  "_MWV",
  "_RU",
  "_freeze_array",
  "_readonly_optional",
  "_validate_inputs",
]
_ATG, _BTG = 5.28e-5, 1.165e-2
_ATV, _BTV = 3.30e-5, 1.742e-2
P8 = 101325.0  # far-field pressure (Pa)
RHO = 1064.0  # far-field density (kg/m^3)
SURF = 0.07  # surface tension (N/m)
KAPPA = 1.4  # polytropic exponent (see module docstring)
C8 = 1484.0  # far-field sound speed (m/s)
_KM = 0.55  # liquid thermal conductivity (W/m/K)
_CP = 4.181e3  # liquid specific heat (J/kg/K)
_LT = 2.0  # exterior-grid stretching length (default_case.m)
_RU = 8.3144598  # universal gas constant (J/mol/K)
_MWV = 18.01528e-3  # molar mass, water vapor (kg/mol)
_MWG = 28.966e-3  # molar mass, non-condensible gas / air (kg/mol)
_D0 = 24.2e-6  # binary (vapor-in-gas) diffusion coefficient (m^2/s)
_LHEAT = 2264.76e3  # latent heat of vaporization (J/kg)

class SimulationError(RuntimeError):
  """Raised when the numerical integrator cannot complete a simulation."""

  def __init__(self, message: str, stats: SolverStats | None = None):
    super().__init__(message)
    self.stats = stats

# The union in `_materials` is the declaration; this is it at runtime. Written out by hand
# it was three lists -- the union, this, and an inline `isinstance` -- that nothing kept in
# step, so adding a material could leave the type error and the validator disagreeing.
_MATERIALS = get_args(MaterialModel)

@dataclass(frozen=True, slots=True)
class PhysicalParameters:
  """Dimensional environment and transport properties."""

  far_field_pressure_pa: float = P8
  medium_density_kg_m3: float = RHO
  surface_tension_n_m: float = SURF
  sound_speed_m_s: float = C8
  polytropic_exponent: float = KAPPA
  tait_pressure_pa: float = _GAM_TAIT
  tait_exponent: float = _NSTATE_TAIT
  hugoniot_slope: float = _HUGONIOT_S
  gas_conductivity_slope: float = _ATG
  gas_conductivity_offset: float = _BTG
  vapor_conductivity_slope: float = _ATV
  vapor_conductivity_offset: float = _BTV
  medium_conductivity_w_m_k: float = _KM
  medium_specific_heat_j_kg_k: float = _CP
  medium_grid_length: float = _LT
  mass_diffusivity_m2_s: float = _D0
  latent_heat_j_kg: float = _LHEAT
  universal_gas_constant_j_mol_k: float = _RU
  vapor_molar_mass_kg_mol: float = _MWV
  gas_molar_mass_kg_mol: float = _MWG

  def __post_init__(self) -> None:
    for name in self.__dataclass_fields__:
      value = getattr(self, name)
      if not np.isfinite(value) or value <= 0.0: raise ValueError(f"physics.{name} must be finite and positive")
    if self.polytropic_exponent <= 1.0: raise ValueError("physics.polytropic_exponent must be greater than 1")
    if self.tait_exponent <= 1.0: raise ValueError("physics.tait_exponent must be greater than 1")

@dataclass(frozen=True, slots=True)
class SampledForcing:
  """Far-field pressure perturbation sampled in dimensional units."""

  time_s: tuple[float, ...]
  pressure_pa: tuple[float, ...]

  def __post_init__(self) -> None:
    times = tuple(float(value) for value in self.time_s)
    pressure = tuple(float(value) for value in self.pressure_pa)
    if len(times) < 2 or len(times) != len(pressure): raise ValueError("sampled forcing requires equal arrays of at least 2 values")
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(pressure)): raise ValueError("sampled forcing values must be finite")
    if times[0] < 0.0 or np.any(np.diff(times) <= 0.0): raise ValueError("sampled forcing times must be non-negative and increasing")
    object.__setattr__(self, "time_s", times)
    object.__setattr__(self, "pressure_pa", pressure)

@dataclass(frozen=True, slots=True)
class InitialState:
  """Optional dimensional initial conditions and internal solver state."""

  wall_velocity_m_s: float = 0.0
  internal_pressure_pa: float | None = None
  bubble_temperature_k: float | None = None
  medium_temperature_k: float | None = None
  vapor_mass_fraction: float | None = None
  stress_state: tuple[float, ...] | None = None

  def __post_init__(self) -> None:
    if not np.isfinite(self.wall_velocity_m_s): raise ValueError("initial.wall_velocity_m_s must be finite")
    for name in ("internal_pressure_pa", "bubble_temperature_k", "medium_temperature_k"):
      value = getattr(self, name)
      if value is not None and (not np.isfinite(value) or value <= 0.0): raise ValueError(f"initial.{name} must be finite and positive")
    fraction = self.vapor_mass_fraction
    if fraction is not None and (not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0):
      raise ValueError("initial.vapor_mass_fraction must be between 0 and 1")
    if self.stress_state is not None:
      state = tuple(float(value) for value in self.stress_state)
      if not np.all(np.isfinite(state)): raise ValueError("initial.stress_state must contain finite values")
      object.__setattr__(self, "stress_state", state)

@dataclass(frozen=True, slots=True)
class CollapseInitialization:
  """History-consistent memory state at the observed maximum radius."""

  maximum_time_nondimensional: float = 4.0
  radius_tolerance: float = 1e-8
  initial_velocity_guess: float = 1.0
  maximum_bracket_expansions: int = 24

  def __post_init__(self) -> None:
    _finite_positive("collapse.maximum_time_nondimensional", self.maximum_time_nondimensional)
    _finite_positive("collapse.radius_tolerance", self.radius_tolerance)
    _finite_positive("collapse.initial_velocity_guess", self.initial_velocity_guess)
    if not isinstance(self.maximum_bracket_expansions, Integral) or self.maximum_bracket_expansions < 1:
      raise ValueError("collapse.maximum_bracket_expansions must be a positive integer")

@dataclass(frozen=True, slots=True)
class SimulationConfig:
  """Validated dimensional inputs for one IMR simulation."""

  R0: float
  Req: float
  material: MaterialModel
  radial: int = 1
  vapor: int = 0
  T8: float = 298.15
  pA: float = 0.0
  omega: float = 0.0
  TW: float = 0.0
  DT: float = 0.0
  mn: float = 0.0
  wave_type: int = 0
  bubtherm: int = 0
  Nt: int = 25
  medtherm: int = 0
  Mt: int = 100
  masstrans: int = 0
  rtol: float = 1e-8
  atol: float = 1e-10
  max_step_s: float | None = None
  max_steps: int = 1_000_000
  thermal: str = "spectral"
  physics: PhysicalParameters = field(default_factory=PhysicalParameters)
  sampled_forcing: SampledForcing | None = None
  initial: InitialState = field(default_factory=InitialState)
  collapse: CollapseInitialization | None = None

  def __post_init__(self) -> None:
    if not isinstance(self.material, _MATERIALS): raise TypeError("material must be a supported material model")
    if not isinstance(self.physics, PhysicalParameters): raise TypeError("physics must be PhysicalParameters")
    if not isinstance(self.initial, InitialState): raise TypeError("initial must be InitialState")
    if self.sampled_forcing is not None and not isinstance(self.sampled_forcing, SampledForcing):
      raise TypeError("sampled_forcing must be SampledForcing")
    if self.collapse is not None and not isinstance(self.collapse, CollapseInitialization): raise TypeError("collapse must be CollapseInitialization")
    if self.collapse is not None:
      if not isinstance(self.material, (Zener, QuadraticZener, OldroydB, Giesekus, LinearPTT)):
        raise ValueError("collapse initialization requires a material with memory")
      if self.initial.stress_state is not None: raise ValueError("collapse initialization cannot be combined with initial.stress_state")
      if self.initial.wall_velocity_m_s != 0.0: raise ValueError("collapse initialization requires zero observed wall velocity")
    _validate_config(self)
    if self.max_step_s is not None: _finite_positive("max_step_s", self.max_step_s)
    if self.thermal not in ("fd", "spectral"): raise ValueError("thermal must be 'fd' or 'spectral'")
    if self.sampled_forcing is not None and (
      self.pA != 0.0 or self.omega != 0.0 or self.TW != 0.0 or self.DT != 0.0 or self.mn != 0.0 or self.wave_type != 0
    ):
      raise ValueError("sampled_forcing cannot be combined with analytic forcing")

@dataclass(frozen=True, slots=True)
class SolverStats:
  """Integrator diagnostics for one completed or failed solve."""

  backend: str
  success: bool
  message: str
  nfev: int
  njev: int
  nlu: int
  elapsed_s: float

@dataclass(frozen=True, slots=True)
class CollapseStats:
  """Diagnostics for a completed precursor shooting solve."""

  initial_velocity_nondimensional: float
  maximum_time_nondimensional: float
  maximum_radius_ratio: float
  shooting_evaluations: int
  integration_evaluations: int
  stress_state: np.ndarray

@dataclass(frozen=True, slots=True)
class StateLayout:
  pressure: int | None
  bubble_thermal: slice | None
  medium_thermal: slice | None
  vapor_fraction: slice | None
  stress: slice
  size: int

  @classmethod
  def of(cls, bubtherm, Nt, medtherm, Mt, masstrans, stress_states) -> StateLayout:
    """Where each block sits in the state vector, from the flags that decide it.

    `_rhs` slices the same vector and takes its flags as arguments rather than from a
    config -- the collapse precursor runs a reduced state under a config that asks for
    more -- so the layout has to be reachable from flags, not just from a config.
    """
    cursor = 2
    pressure = None
    bubble_thermal = None
    medium_thermal = None
    vapor_fraction = None
    if bubtherm:
      pressure = cursor
      cursor += 1
      bubble_thermal = slice(cursor, cursor + Nt)
      cursor += Nt
      if medtherm:
        medium_thermal = slice(cursor, cursor + Mt)
        cursor += Mt
      if masstrans:
        vapor_fraction = slice(cursor, cursor + Nt)
        cursor += Nt
    stress = slice(cursor, cursor + stress_states)
    return cls(
      pressure=pressure, bubble_thermal=bubble_thermal, medium_thermal=medium_thermal, vapor_fraction=vapor_fraction,
      stress=stress, size=stress.stop
    )

  @classmethod
  def from_config(cls, config: SimulationConfig) -> StateLayout:
    return cls.of(
      config.bubtherm, config.Nt, config.medtherm, config.Mt, config.masstrans, _stress_state_count(config.material)
    )

@dataclass(frozen=True, slots=True)
class MediumOperators:
  xi: np.ndarray
  yT: np.ndarray
  yT2: np.ndarray
  yT3: np.ndarray
  iyT3: np.ndarray
  iyT4: np.ndarray
  iyT6: np.ndarray
  D1: np.ndarray
  D2: np.ndarray
  grad_Tm: np.ndarray
  grad_Trans: np.ndarray
  grad_C: np.ndarray
  bubble_wall_stencil: np.ndarray
  medium_wall_stencil: np.ndarray

@dataclass(frozen=True, slots=True)
class PreparedForcing:
  knots: np.ndarray
  coefficients: np.ndarray

@dataclass(frozen=True, slots=True)
class PreparedDistributedStress:
  reference_radius: np.ndarray
  reference_radius_cubed: np.ndarray
  weights: np.ndarray | None = None

@dataclass(frozen=True, slots=True)
class PreparedInstantaneousMaterial:
  interval_nodes: np.ndarray
  interval_weights: np.ndarray

@dataclass(frozen=True, slots=True)
class SweepResult:
  """One trajectory per parameter set, every array shaped `(point, time)`."""

  parameters: tuple[str, ...]
  values: np.ndarray
  time_s: np.ndarray
  radius_ratio: np.ndarray
  radius_m: np.ndarray
  wall_velocity_m_s: np.ndarray
  internal_pressure_pa: np.ndarray
  stress_integral_pa: np.ndarray
  state: np.ndarray
  steps: np.ndarray

@dataclass(frozen=True, slots=True)
class SimulationResult:
  """Immutable physical histories returned by the strict public API."""

  time_s: np.ndarray
  radius_ratio: np.ndarray
  wall_velocity_m_s: np.ndarray
  internal_pressure_pa: np.ndarray
  stress_integral_pa: np.ndarray
  bubble_temperature_k: np.ndarray | None
  medium_temperature_k: np.ndarray | None
  vapor_mass_fraction: np.ndarray | None
  stress_state: np.ndarray | None
  stress_reference_radius_ratio: np.ndarray | None
  stats: SolverStats
  config: SimulationConfig

  @property
  def radius_m(self) -> np.ndarray:
    radius = self.config.R0 * self.radius_ratio
    radius.setflags(write=False)
    return radius

def _readonly_optional(values) -> np.ndarray | None: return None if values is None else _freeze_array(values)

def _validate_config(config) -> None:
  c = config
  for name, value in (("R0", c.R0), ("Req", c.Req), ("T8", c.T8), ("rtol", c.rtol), ("atol", c.atol)):
    if not np.isfinite(value) or value <= 0: raise ValueError(f"{name} must be finite and positive")
  if not isinstance(c.material, _MATERIALS): raise TypeError("material must be a supported material model")
  for name, value in (("pA", c.pA), ("omega", c.omega), ("TW", c.TW), ("DT", c.DT), ("mn", c.mn)):
    if not np.isfinite(value): raise ValueError(f"{name} must be finite")
  for name, value, allowed in (("radial", c.radial, range(1, 7)), ("wave_type", c.wave_type, range(0, 4))):
    if not isinstance(value, Integral) or value not in allowed:
      raise ValueError(f"{name} must be one of: {', '.join(str(choice) for choice in allowed)}")
  for name, value in (("vapor", c.vapor), ("bubtherm", c.bubtherm), ("medtherm", c.medtherm), ("masstrans", c.masstrans)):
    if not isinstance(value, Integral) or value not in (0, 1): raise ValueError(f"{name} must be 0 or 1")
  for name, value in (("Nt", c.Nt), ("Mt", c.Mt)):
    if not isinstance(value, Integral) or value < 3: raise ValueError(f"{name} must be an integer >= 3")
  if not isinstance(c.max_steps, Integral) or c.max_steps < 1: raise ValueError("max_steps must be an integer >= 1")
  if c.medtherm and not c.bubtherm: raise ValueError("medtherm=1 requires bubtherm=1")
  if c.masstrans and not c.bubtherm: raise ValueError("masstrans=1 requires bubtherm=1")
  if c.masstrans and not c.vapor: raise ValueError("masstrans=1 requires vapor=1")
  if c.bubtherm and c.vapor and not c.masstrans: raise ValueError("bubtherm=1 with vapor=1 currently requires masstrans=1")

def _validate_inputs(tv, config) -> np.ndarray:
  """Validate a time grid against a config, returning the grid as an array."""
  times = np.asarray(tv, dtype=float)
  if times.ndim != 1 or times.size < 2: raise ValueError("tv must be a one-dimensional array with at least two times")
  if not np.all(np.isfinite(times)): raise ValueError("tv must contain only finite values")
  if times[0] < 0 or np.any(np.diff(times) <= 0): raise ValueError("tv must be non-negative and strictly increasing")
  _validate_config(config)
  return times

def _freeze_array(values) -> np.ndarray:
  """A read-only float copy. The copy is the point: `asarray` on a float64 array returns
  that same array, so freezing without it froze the CALLER's array -- `solve_sweep` was
  handing back a `times` the caller could no longer write to."""
  array = np.array(values, dtype=float, copy=True)
  array.setflags(write=False)
  return array
