"""Typed, dimensional material models.

Value objects only: every class here is a frozen dataclass that validates its
own parameters. No solver state, no numerics.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np

__all__ = [
  "ArrudaBoyce",
  "Bingham",
  "CarreauYasuda",
  "Cross",
  "ElasticModel",
  "Fung",
  "Gent",
  "Giesekus",
  "HerschelBulkley",
  "InstantaneousMaterial",
  "LinearPTT",
  "MaterialModel",
  "ModifiedPowellEyring",
  "MooneyRivlin",
  "NeoHookean",
  "NeoHookeanKelvinVoigt",
  "Newtonian",
  "NoStress",
  "OldroydB",
  "PowellEyring",
  "PowerLaw",
  "QuadraticKelvinVoigt",
  "QuadraticZener",
  "ViscousModel",
  "Yeoh",
  "Zener",
]


def _finite_positive(name, value) -> None:
  if not np.isfinite(value) or value <= 0.0:
    raise ValueError(f"{name} must be finite and positive")


def _finite_nonnegative(name, value) -> None:
  if not np.isfinite(value) or value < 0.0:
    raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class NoStress:
  """No constitutive stress."""


@dataclass(frozen=True, slots=True)
class NeoHookeanKelvinVoigt:
  """Closed-form neo-Hookean Kelvin-Voigt solid."""

  shear_modulus_pa: float
  viscosity_pa_s: float

  def __post_init__(self) -> None:
    _finite_positive("shear_modulus_pa", self.shear_modulus_pa)
    _finite_positive("viscosity_pa_s", self.viscosity_pa_s)


@dataclass(frozen=True, slots=True)
class QuadraticKelvinVoigt:
  """Closed-form quadratic Kelvin-Voigt solid."""

  shear_modulus_pa: float
  viscosity_pa_s: float
  stiffening: float = 0.25

  def __post_init__(self) -> None:
    _finite_positive("shear_modulus_pa", self.shear_modulus_pa)
    _finite_positive("viscosity_pa_s", self.viscosity_pa_s)
    _finite_nonnegative("stiffening", self.stiffening)


def _validate_memory_parameters(viscosity_pa_s, relaxation_time_s, retardation_time_s, *, polymer_required) -> None:
  _finite_positive("viscosity_pa_s", viscosity_pa_s)
  _finite_positive("relaxation_time_s", relaxation_time_s)
  _finite_nonnegative("retardation_time_s", retardation_time_s)
  limit_ok = retardation_time_s < relaxation_time_s if polymer_required else retardation_time_s <= relaxation_time_s
  if not limit_ok:
    relation = "less than" if polymer_required else "no greater than"
    raise ValueError(f"retardation_time_s must be {relation} relaxation_time_s")


@dataclass(frozen=True, slots=True)
class Zener:
  """Closed-form neo-Hookean Zener/Jeffreys/Maxwell family."""

  shear_modulus_pa: float
  viscosity_pa_s: float
  relaxation_time_s: float
  retardation_time_s: float = 0.0

  def __post_init__(self) -> None:
    _finite_positive("shear_modulus_pa", self.shear_modulus_pa)
    _validate_memory_parameters(
      self.viscosity_pa_s,
      self.relaxation_time_s,
      self.retardation_time_s,
      polymer_required=False,
    )


@dataclass(frozen=True, slots=True)
class QuadraticZener:
  """Closed-form quadratic-elastic Zener family."""

  shear_modulus_pa: float
  viscosity_pa_s: float
  relaxation_time_s: float
  retardation_time_s: float = 0.0
  stiffening: float = 0.25

  def __post_init__(self) -> None:
    _finite_positive("shear_modulus_pa", self.shear_modulus_pa)
    _validate_memory_parameters(
      self.viscosity_pa_s,
      self.relaxation_time_s,
      self.retardation_time_s,
      polymer_required=False,
    )
    _finite_nonnegative("stiffening", self.stiffening)


@dataclass(frozen=True, slots=True)
class OldroydB:
  """Closed-form UCM/Oldroyd-B fluid."""

  viscosity_pa_s: float
  relaxation_time_s: float
  retardation_time_s: float = 0.0

  def __post_init__(self) -> None:
    _validate_memory_parameters(
      self.viscosity_pa_s,
      self.relaxation_time_s,
      self.retardation_time_s,
      polymer_required=False,
    )


@dataclass(frozen=True, slots=True)
class NeoHookean:
  shear_modulus_pa: float

  def __post_init__(self) -> None:
    _finite_positive("shear_modulus_pa", self.shear_modulus_pa)


@dataclass(frozen=True, slots=True)
class MooneyRivlin:
  c10_pa: float
  c01_pa: float

  def __post_init__(self) -> None:
    _finite_nonnegative("c10_pa", self.c10_pa)
    _finite_nonnegative("c01_pa", self.c01_pa)
    if self.c10_pa == 0.0 and self.c01_pa == 0.0:
      raise ValueError("at least one Mooney-Rivlin coefficient must be positive")


@dataclass(frozen=True, slots=True)
class Yeoh:
  c1_pa: float
  c2_pa: float = 0.0
  c3_pa: float = 0.0

  def __post_init__(self) -> None:
    _finite_positive("c1_pa", self.c1_pa)
    for name in ("c2_pa", "c3_pa"):
      if not np.isfinite(getattr(self, name)):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True, slots=True)
class Fung:
  shear_modulus_pa: float
  stiffening: float

  def __post_init__(self) -> None:
    _finite_positive("shear_modulus_pa", self.shear_modulus_pa)
    _finite_nonnegative("stiffening", self.stiffening)


@dataclass(frozen=True, slots=True)
class Gent:
  shear_modulus_pa: float
  extensibility: float

  def __post_init__(self) -> None:
    _finite_positive("shear_modulus_pa", self.shear_modulus_pa)
    _finite_positive("extensibility", self.extensibility)


@dataclass(frozen=True, slots=True)
class ArrudaBoyce:
  shear_modulus_pa: float
  chain_segments: float

  def __post_init__(self) -> None:
    _finite_positive("shear_modulus_pa", self.shear_modulus_pa)
    if not np.isfinite(self.chain_segments) or self.chain_segments <= 1.0:
      raise ValueError("chain_segments must be finite and greater than 1")


ElasticModel = NeoHookean | MooneyRivlin | Yeoh | Fung | Gent | ArrudaBoyce


@dataclass(frozen=True, slots=True)
class Newtonian:
  viscosity_pa_s: float

  def __post_init__(self) -> None:
    _finite_positive("viscosity_pa_s", self.viscosity_pa_s)


@dataclass(frozen=True, slots=True)
class PowerLaw:
  consistency_pa_s_n: float
  exponent: float
  regularization_rate_per_s: float = 1e-3

  def __post_init__(self) -> None:
    _finite_positive("consistency_pa_s_n", self.consistency_pa_s_n)
    _finite_positive("exponent", self.exponent)
    _finite_positive("regularization_rate_per_s", self.regularization_rate_per_s)


@dataclass(frozen=True, slots=True)
class CarreauYasuda:
  zero_shear_viscosity_pa_s: float
  infinite_shear_viscosity_pa_s: float
  time_constant_s: float
  transition_exponent: float
  power_index: float

  def __post_init__(self) -> None:
    _finite_positive("zero_shear_viscosity_pa_s", self.zero_shear_viscosity_pa_s)
    _finite_nonnegative(
      "infinite_shear_viscosity_pa_s",
      self.infinite_shear_viscosity_pa_s,
    )
    _finite_nonnegative("time_constant_s", self.time_constant_s)
    _finite_positive("transition_exponent", self.transition_exponent)
    _finite_positive("power_index", self.power_index)


@dataclass(frozen=True, slots=True)
class PowellEyring:
  """eta_inf + (eta_0 - eta_inf) * asinh(lambda*gdot)/(lambda*gdot).

  IMRv2's f_viscosity.m uses sinh rather than asinh, which is
  shear-thickening and diverges exponentially. This is the standard law.
  """

  zero_shear_viscosity_pa_s: float
  infinite_shear_viscosity_pa_s: float
  time_constant_s: float

  def __post_init__(self) -> None:
    _finite_positive("zero_shear_viscosity_pa_s", self.zero_shear_viscosity_pa_s)
    _finite_nonnegative(
      "infinite_shear_viscosity_pa_s",
      self.infinite_shear_viscosity_pa_s,
    )
    _finite_nonnegative("time_constant_s", self.time_constant_s)


@dataclass(frozen=True, slots=True)
class ModifiedPowellEyring:
  """eta_inf + (eta_0 - eta_inf) * log1p(lambda*gdot)/(lambda*gdot).

  IMRv2 generalizes the denominator to (lambda*gdot)^nc, which has no
  finite zero-shear limit unless nc == 1. This uses the well-posed form.
  """

  zero_shear_viscosity_pa_s: float
  infinite_shear_viscosity_pa_s: float
  time_constant_s: float

  def __post_init__(self) -> None:
    _finite_positive("zero_shear_viscosity_pa_s", self.zero_shear_viscosity_pa_s)
    _finite_nonnegative(
      "infinite_shear_viscosity_pa_s",
      self.infinite_shear_viscosity_pa_s,
    )
    _finite_nonnegative("time_constant_s", self.time_constant_s)


@dataclass(frozen=True, slots=True)
class Cross:
  zero_shear_viscosity_pa_s: float
  infinite_shear_viscosity_pa_s: float
  time_constant_s: float
  transition_exponent: float

  def __post_init__(self) -> None:
    _finite_positive("zero_shear_viscosity_pa_s", self.zero_shear_viscosity_pa_s)
    _finite_nonnegative(
      "infinite_shear_viscosity_pa_s",
      self.infinite_shear_viscosity_pa_s,
    )
    _finite_nonnegative("time_constant_s", self.time_constant_s)
    _finite_positive("transition_exponent", self.transition_exponent)


@dataclass(frozen=True, slots=True)
class HerschelBulkley:
  yield_stress_pa: float
  consistency_pa_s_n: float
  exponent: float
  regularization_rate_per_s: float = 1e-3

  def __post_init__(self) -> None:
    _finite_nonnegative("yield_stress_pa", self.yield_stress_pa)
    _finite_positive("consistency_pa_s_n", self.consistency_pa_s_n)
    _finite_positive("exponent", self.exponent)
    _finite_positive("regularization_rate_per_s", self.regularization_rate_per_s)


@dataclass(frozen=True, slots=True)
class Bingham:
  yield_stress_pa: float
  plastic_viscosity_pa_s: float
  regularization_rate_per_s: float = 1e-3

  def __post_init__(self) -> None:
    _finite_nonnegative("yield_stress_pa", self.yield_stress_pa)
    _finite_positive("plastic_viscosity_pa_s", self.plastic_viscosity_pa_s)
    _finite_positive("regularization_rate_per_s", self.regularization_rate_per_s)


ViscousModel = (
  Newtonian | PowerLaw | CarreauYasuda | Cross | PowellEyring | ModifiedPowellEyring | HerschelBulkley | Bingham
)


@dataclass(frozen=True, slots=True)
class InstantaneousMaterial:
  """Composable hyperelastic and generalized-Newtonian material."""

  elastic: ElasticModel | None = None
  viscous: ViscousModel | None = None
  quadrature_points: int = 32

  def __post_init__(self) -> None:
    if self.elastic is None and self.viscous is None:
      raise ValueError("an instantaneous material requires an elastic or viscous law")
    if self.elastic is not None and not isinstance(
      self.elastic,
      (NeoHookean, MooneyRivlin, Yeoh, Fung, Gent, ArrudaBoyce),
    ):
      raise TypeError("elastic must be a supported elastic model")
    if self.viscous is not None and not isinstance(
      self.viscous,
      (Newtonian, PowerLaw, CarreauYasuda, Cross, PowellEyring, ModifiedPowellEyring, HerschelBulkley, Bingham),
    ):
      raise TypeError("viscous must be a supported viscous model")
    if not isinstance(self.quadrature_points, Integral) or self.quadrature_points < 8:
      raise ValueError("quadrature_points must be an integer >= 8")


def _validate_distributed_model(name, parameter, points, extent) -> None:
  if not np.isfinite(parameter) or parameter < 0.0:
    raise ValueError(f"{name} must be finite and non-negative")
  if not isinstance(points, Integral) or points < 8:
    raise ValueError("points must be an integer >= 8")
  if not np.isfinite(extent) or extent <= 1.0:
    raise ValueError("extent must be finite and greater than 1")


@dataclass(frozen=True, slots=True)
class Giesekus:
  """Distributed Giesekus memory on a prepared Lagrangian radial grid."""

  viscosity_pa_s: float
  relaxation_time_s: float
  retardation_time_s: float = 0.0
  mobility: float = 0.0
  points: int = 480
  extent: float = 60.0

  def __post_init__(self) -> None:
    _validate_memory_parameters(
      self.viscosity_pa_s,
      self.relaxation_time_s,
      self.retardation_time_s,
      polymer_required=True,
    )
    _validate_distributed_model("mobility", self.mobility, self.points, self.extent)
    if self.mobility > 1.0:
      raise ValueError("mobility must not exceed 1")


@dataclass(frozen=True, slots=True)
class LinearPTT:
  """Distributed linear Phan-Thien-Tanner memory."""

  viscosity_pa_s: float
  relaxation_time_s: float
  retardation_time_s: float = 0.0
  extensibility: float = 0.0
  points: int = 480
  extent: float = 60.0

  def __post_init__(self) -> None:
    _validate_memory_parameters(
      self.viscosity_pa_s,
      self.relaxation_time_s,
      self.retardation_time_s,
      polymer_required=True,
    )
    _validate_distributed_model("extensibility", self.extensibility, self.points, self.extent)


MaterialModel = (
  NoStress
  | NeoHookeanKelvinVoigt
  | QuadraticKelvinVoigt
  | Zener
  | QuadraticZener
  | OldroydB
  | InstantaneousMaterial
  | Giesekus
  | LinearPTT
)
