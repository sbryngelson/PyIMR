"""Typed, dimensional material models."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from functools import partial
from numbers import Integral

import numpy as np

from ._validate import positive_scalar

__all__ = [
  "_is_distributed_stress",
  "_stress_state_count",
  "TwoModeQuadraticZener",
  "ArrudaBoyce",
  "Bingham",
  "CarreauYasuda",
  "CarreauZener",
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
  "Ogden",
  "OldroydB",
  "PowellEyring",
  "PowerLaw",
  "QuadraticKelvinVoigt",
  "CubicZener",
  "RelaxingMaterial",
  "QuadraticZener",
  "ViscousModel",
  "Yeoh",
  "Zener",
]

def _validated_fields(**rules):
  """Give a dataclass a `__post_init__` that runs one check per named field."""
  def decorate(cls):
    def __post_init__(self):
      for name, check in rules.items(): check(name, getattr(self, name))
    cls.__post_init__ = __post_init__
    return cls
  return decorate

_non_negative = partial(positive_scalar, allow_zero=True)

@dataclass(frozen=True, slots=True)
class NoStress:
  """No constitutive stress."""

@dataclass(frozen=True, slots=True)
@_validated_fields(shear_modulus_pa=positive_scalar, viscosity_pa_s=positive_scalar)
class NeoHookeanKelvinVoigt:
  """Closed-form neo-Hookean Kelvin-Voigt solid."""

  shear_modulus_pa: float
  viscosity_pa_s: float

@dataclass(frozen=True, slots=True)
@_validated_fields(shear_modulus_pa=positive_scalar, viscosity_pa_s=positive_scalar, stiffening=_non_negative)
class QuadraticKelvinVoigt:
  """Closed-form quadratic Kelvin-Voigt solid."""

  shear_modulus_pa: float
  viscosity_pa_s: float
  stiffening: float = 0.25

def _validate_memory_parameters(viscosity_pa_s, relaxation_time_s, retardation_time_s, *, polymer_required) -> None:
  positive_scalar("viscosity_pa_s", viscosity_pa_s)
  positive_scalar("relaxation_time_s", relaxation_time_s)
  _non_negative("retardation_time_s", retardation_time_s)
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
    positive_scalar("shear_modulus_pa", self.shear_modulus_pa)
    _validate_memory_parameters(self.viscosity_pa_s, self.relaxation_time_s, self.retardation_time_s, polymer_required=False)

@dataclass(frozen=True, slots=True)
class LinearMaxwell:
  """Maxwell fluid: a relaxing viscous stress with no elastic equilibrium.

  Zener without the parallel spring. `Ca` is infinite because there is no modulus, so the
  elastic target the memory relaxes toward is zero, and there is no retardation.
  """

  viscosity_pa_s: float
  relaxation_time_s: float

  def __post_init__(self) -> None:
    _validate_memory_parameters(self.viscosity_pa_s, self.relaxation_time_s, 0.0, polymer_required=False)

@dataclass(frozen=True, slots=True)
class QuadraticZener:
  """Closed-form quadratic-elastic Zener family."""

  shear_modulus_pa: float
  viscosity_pa_s: float
  relaxation_time_s: float
  retardation_time_s: float = 0.0
  stiffening: float = 0.25

  def __post_init__(self) -> None:
    positive_scalar("shear_modulus_pa", self.shear_modulus_pa)
    _validate_memory_parameters(self.viscosity_pa_s, self.relaxation_time_s, self.retardation_time_s, polymer_required=False)
    _non_negative("stiffening", self.stiffening)

@dataclass(frozen=True, slots=True)
class CubicZener:
  """Quadratic Zener with the next term of the strain-energy expansion.

  `QuadraticZener` stiffens as `alpha (I1 - 3)^2`, which is Yeoh truncated after `c2`. On
  these records the collapse reaches `I1 - 3` of 24 to 119, where the next term is not a
  small correction: screened against the identifiable discrepancy, a cubic elastic term
  reaches 30% to 52% of it, and its direction is nearly orthogonal to what an extra
  relaxation arm reaches, so the two are not competing descriptions of one effect.

  `cubic = 0` reduces to `QuadraticZener` exactly -- the added term carries the coefficient
  linearly -- and that identity is the test the branch is gated on. The closed form of its
  elastic integral carries a `log`, which no other `Ze` in this package does, so it is also
  checked against the quadrature of `_elastic_integrand` at finite `cubic`.
  """

  shear_modulus_pa: float
  viscosity_pa_s: float
  relaxation_time_s: float
  retardation_time_s: float = 0.0
  stiffening: float = 0.25
  cubic: float = 0.0

  def __post_init__(self):
    positive_scalar("shear_modulus_pa", self.shear_modulus_pa)
    positive_scalar("viscosity_pa_s", self.viscosity_pa_s)
    positive_scalar("relaxation_time_s", self.relaxation_time_s)
    _non_negative("retardation_time_s", self.retardation_time_s)
    _non_negative("stiffening", self.stiffening)
    _non_negative("cubic", self.cubic)


@dataclass(frozen=True, slots=True)
class RelaxingMaterial:
  """Any elastic law, with a Maxwell arm: the Zener construction freed from one potential.

  Every Zener here carries its own hand-derived `Ze`, so relaxation has been available to the
  neo-Hookean family and to nothing else. Gent, Fung, Arruda-Boyce, Mooney-Rivlin, Yeoh and
  Ogden exist only as `InstantaneousMaterial` -- elastic, with no memory at all -- which is
  the wrong comparison to make against a relaxing model and the reason a purely elastic Yeoh
  scored so much higher in screening than the same term added to qSLS.

  The equilibrium target is taken by the quadrature `InstantaneousMaterial` already uses,
  rather than by a closed form derived per law, so one class covers every elastic model in
  the package and any added later. That costs the quadrature at each step and buys the
  comparison being a controlled one.

  `elastic` reaches the compiled program through `p["el*"]` exactly as it does for
  `InstantaneousMaterial`, so its parameters are traced, differentiable and shared across a
  sweep rather than keying a new compile per value.
  """

  elastic: ElasticModel
  viscosity_pa_s: float
  relaxation_time_s: float
  retardation_time_s: float = 0.0
  quadrature_points: int = 32

  def __post_init__(self):
    if self.elastic is None: raise ValueError("RelaxingMaterial needs an elastic law")
    positive_scalar("viscosity_pa_s", self.viscosity_pa_s)
    positive_scalar("relaxation_time_s", self.relaxation_time_s)
    _non_negative("retardation_time_s", self.retardation_time_s)
    if not isinstance(self.quadrature_points, Integral) or self.quadrature_points < 8:
      raise ValueError("quadrature_points must be an integer >= 8")


@dataclass(frozen=True, slots=True)
class TwoModeQuadraticZener:
  """Quadratic Zener with a second Maxwell arm: two relaxation times, not one.

  Every viscoelastic model in this package carries a single relaxation time, which is a
  strong assumption for a crosslinked biopolymer whose relaxation is distributed. The
  residual of the one-mode fit is correlated at lag one (0.918) and concentrated in the
  first collapse, so a second timescale is the first thing worth trying.

  The elastic target and the viscous forcing are BOTH split by `second_share`, which makes
  `second_share = 0` reduce exactly to `QuadraticZener` -- the second memory is then driven
  by nothing and decays from zero, so it stays zero. That exact reduction is what the
  implementation is tested against; a construction that only reduces approximately would
  hide an error of its own size.
  """

  shear_modulus_pa: float
  viscosity_pa_s: float
  relaxation_time_s: float
  retardation_time_s: float = 0.0
  stiffening: float = 0.25
  second_relaxation_time_s: float = 1e-6
  second_share: float = 0.0

  def __post_init__(self) -> None:
    positive_scalar("shear_modulus_pa", self.shear_modulus_pa)
    _validate_memory_parameters(self.viscosity_pa_s, self.relaxation_time_s, self.retardation_time_s, polymer_required=False)
    _non_negative("stiffening", self.stiffening)
    positive_scalar("second_relaxation_time_s", self.second_relaxation_time_s)
    if not (0.0 <= self.second_share < 1.0):
      raise ValueError("second_share must lie in [0, 1): it is the fraction of the memory carried by the second arm")

@dataclass(frozen=True, slots=True)
class CarreauZener:
  """Quadratic Zener whose dashpot shear-thins: one relaxation time, but a rate-dependent one.

  The other enrichment of the one-mode fit is `TwoModeQuadraticZener`, which answers the
  correlated residual with a second timescale. This answers it with a timescale that moves.
  A Maxwell arm is a spring and a dashpot in series, so its relaxation time is
  `lambda = eta / G`; if the dashpot obeys Carreau rather than Newton, `lambda` falls as the
  medium is sheared harder. A collapse spans decades of shear rate, so the two are different
  claims about the same residual and the comparison between them is the point.

  Only the dashpot thins. The elastic target and the solvent term are untouched, which is
  what keeps the change contained: `S` is unchanged, and because the memory derivative
  carries no `R-ddot`, neither does the acceleration coefficient.

  `power_index = 1` removes the thinning exactly -- the factor is then `1` for every shear
  rate, not merely near it -- so the law reduces to `QuadraticZener` identically. That is
  the reduction the implementation is gated on, and it is the default.
  """

  shear_modulus_pa: float
  viscosity_pa_s: float
  relaxation_time_s: float
  retardation_time_s: float = 0.0
  stiffening: float = 0.25
  thinning_time_s: float = 1e-6
  power_index: float = 1.0

  def __post_init__(self) -> None:
    positive_scalar("shear_modulus_pa", self.shear_modulus_pa)
    _validate_memory_parameters(self.viscosity_pa_s, self.relaxation_time_s, self.retardation_time_s, polymer_required=False)
    _non_negative("stiffening", self.stiffening)
    positive_scalar("thinning_time_s", self.thinning_time_s)
    positive_scalar("power_index", self.power_index)

@dataclass(frozen=True, slots=True)
class OldroydB:
  """Closed-form UCM/Oldroyd-B fluid."""

  viscosity_pa_s: float
  relaxation_time_s: float
  retardation_time_s: float = 0.0

  def __post_init__(self) -> None:
    _validate_memory_parameters(self.viscosity_pa_s, self.relaxation_time_s, self.retardation_time_s, polymer_required=False)

@dataclass(frozen=True, slots=True)
@_validated_fields(shear_modulus_pa=positive_scalar)
class NeoHookean:
  shear_modulus_pa: float

@dataclass(frozen=True, slots=True)
class MooneyRivlin:
  c10_pa: float
  c01_pa: float

  def __post_init__(self) -> None:
    _non_negative("c10_pa", self.c10_pa)
    _non_negative("c01_pa", self.c01_pa)
    if self.c10_pa == 0.0 and self.c01_pa == 0.0: raise ValueError("at least one Mooney-Rivlin coefficient must be positive")

@dataclass(frozen=True, slots=True)
class Yeoh:
  c1_pa: float
  c2_pa: float = 0.0
  c3_pa: float = 0.0

  def __post_init__(self) -> None:
    positive_scalar("c1_pa", self.c1_pa)
    for name in ("c2_pa", "c3_pa"):
      if not np.isfinite(getattr(self, name)): raise ValueError(f"{name} must be finite")

@dataclass(frozen=True, slots=True)
@_validated_fields(shear_modulus_pa=positive_scalar, stiffening=_non_negative)
class Fung:
  shear_modulus_pa: float
  stiffening: float

@dataclass(frozen=True, slots=True)
@_validated_fields(shear_modulus_pa=positive_scalar, extensibility=positive_scalar)
class Gent:
  shear_modulus_pa: float
  extensibility: float

@dataclass(frozen=True, slots=True)
class ArrudaBoyce:
  shear_modulus_pa: float
  chain_segments: float

  def __post_init__(self) -> None:
    positive_scalar("shear_modulus_pa", self.shear_modulus_pa)
    if not np.isfinite(self.chain_segments) or self.chain_segments <= 1.0: raise ValueError("chain_segments must be finite and greater than 1")

@dataclass(frozen=True, slots=True)
class Ogden:
  """Multi-term Ogden, W = sum_p (mu_p / alpha_p) (l1^a_p + l2^a_p + l3^a_p - 3)."""

  shear_moduli_pa: tuple[float, ...]
  exponents: tuple[float, ...]

  def __post_init__(self) -> None:
    moduli = tuple(float(value) for value in self.shear_moduli_pa)
    exponents = tuple(float(value) for value in self.exponents)
    if not moduli or len(moduli) != len(exponents): raise ValueError("Ogden requires equal, non-empty shear_moduli_pa and exponents")
    if not np.all(np.isfinite(moduli)): raise ValueError("Ogden shear_moduli_pa must be finite")
    if not np.all(np.isfinite(exponents)) or any(value == 0.0 for value in exponents): raise ValueError("Ogden exponents must be finite and non-zero")
    if sum(m * a for m, a in zip(moduli, exponents, strict=True)) <= 0.0:
      raise ValueError("Ogden requires sum(shear_moduli_pa * exponents) > 0 for a positive shear modulus")
    object.__setattr__(self, "shear_moduli_pa", moduli)
    object.__setattr__(self, "exponents", exponents)

ElasticModel = NeoHookean | MooneyRivlin | Yeoh | Fung | Gent | ArrudaBoyce | Ogden

@dataclass(frozen=True, slots=True)
@_validated_fields(viscosity_pa_s=positive_scalar)
class Newtonian:
  viscosity_pa_s: float

@dataclass(frozen=True, slots=True)
@_validated_fields(consistency_pa_s_n=positive_scalar, exponent=positive_scalar, regularization_rate_per_s=positive_scalar)
class PowerLaw:
  consistency_pa_s_n: float
  exponent: float
  regularization_rate_per_s: float = 1e-3

@dataclass(frozen=True, slots=True)
@_validated_fields(zero_shear_viscosity_pa_s=positive_scalar, infinite_shear_viscosity_pa_s=_non_negative, time_constant_s=_non_negative, transition_exponent=positive_scalar, power_index=positive_scalar)
class CarreauYasuda:
  zero_shear_viscosity_pa_s: float
  infinite_shear_viscosity_pa_s: float
  time_constant_s: float
  transition_exponent: float
  power_index: float

@dataclass(frozen=True, slots=True)
@_validated_fields(zero_shear_viscosity_pa_s=positive_scalar, infinite_shear_viscosity_pa_s=_non_negative, time_constant_s=_non_negative)
class PowellEyring:
  """eta_inf + (eta_0 - eta_inf) * asinh(lambda*gdot)/(lambda*gdot)."""

  zero_shear_viscosity_pa_s: float
  infinite_shear_viscosity_pa_s: float
  time_constant_s: float

@dataclass(frozen=True, slots=True)
@_validated_fields(zero_shear_viscosity_pa_s=positive_scalar, infinite_shear_viscosity_pa_s=_non_negative, time_constant_s=_non_negative)
class ModifiedPowellEyring:
  """eta_inf + (eta_0 - eta_inf) * log1p(lambda*gdot)/(lambda*gdot)."""

  zero_shear_viscosity_pa_s: float
  infinite_shear_viscosity_pa_s: float
  time_constant_s: float

@dataclass(frozen=True, slots=True)
@_validated_fields(zero_shear_viscosity_pa_s=positive_scalar, infinite_shear_viscosity_pa_s=_non_negative, time_constant_s=_non_negative, transition_exponent=positive_scalar)
class Cross:
  zero_shear_viscosity_pa_s: float
  infinite_shear_viscosity_pa_s: float
  time_constant_s: float
  transition_exponent: float

@dataclass(frozen=True, slots=True)
@_validated_fields(yield_stress_pa=_non_negative, consistency_pa_s_n=positive_scalar, exponent=positive_scalar, regularization_rate_per_s=positive_scalar)
class HerschelBulkley:
  yield_stress_pa: float
  consistency_pa_s_n: float
  exponent: float
  regularization_rate_per_s: float = 1e-3

@dataclass(frozen=True, slots=True)
@_validated_fields(yield_stress_pa=_non_negative, plastic_viscosity_pa_s=positive_scalar, regularization_rate_per_s=positive_scalar)
class Bingham:
  yield_stress_pa: float
  plastic_viscosity_pa_s: float
  regularization_rate_per_s: float = 1e-3

ViscousModel = Newtonian | PowerLaw | CarreauYasuda | Cross | PowellEyring | ModifiedPowellEyring | HerschelBulkley | Bingham

@dataclass(frozen=True, slots=True)
class InstantaneousMaterial:
  """Composable hyperelastic and generalized-Newtonian material."""

  elastic: ElasticModel | None = None
  viscous: ViscousModel | None = None
  quadrature_points: int = 32

  def __post_init__(self) -> None:
    if self.elastic is None and self.viscous is None: raise ValueError("an instantaneous material requires an elastic or viscous law")
    if self.elastic is not None and not isinstance(self.elastic, (NeoHookean, MooneyRivlin, Yeoh, Fung, Gent, ArrudaBoyce, Ogden)):
      raise TypeError("elastic must be a supported elastic model")
    if self.viscous is not None and not isinstance(
      self.viscous, (Newtonian, PowerLaw, CarreauYasuda, Cross, PowellEyring, ModifiedPowellEyring, HerschelBulkley, Bingham)
    ):
      raise TypeError("viscous must be a supported viscous model")
    if not isinstance(self.quadrature_points, Integral) or self.quadrature_points < 8: raise ValueError("quadrature_points must be an integer >= 8")

def _validate_distributed_model(name, parameter, points, extent) -> None:
  if not np.isfinite(parameter) or parameter < 0.0: raise ValueError(f"{name} must be finite and non-negative")
  if not isinstance(points, Integral) or points < 8: raise ValueError("points must be an integer >= 8")
  if not np.isfinite(extent) or extent <= 1.0: raise ValueError("extent must be finite and greater than 1")

@dataclass(frozen=True, slots=True)
class Giesekus:
  """Distributed Giesekus memory on a prepared Lagrangian radial grid."""

  viscosity_pa_s: float
  relaxation_time_s: float
  retardation_time_s: float = 0.0
  mobility: float = 0.0
  points: int = 240
  extent: float = 60.0
  quadrature: str = "gauss"

  def __post_init__(self) -> None:
    _validate_memory_parameters(self.viscosity_pa_s, self.relaxation_time_s, self.retardation_time_s, polymer_required=True)
    _validate_distributed_model("mobility", self.mobility, self.points, self.extent)
    if self.quadrature not in ("trapezoid", "gauss"): raise ValueError("quadrature must be 'trapezoid' or 'gauss'")
    if self.mobility > 1.0: raise ValueError("mobility must not exceed 1")

@dataclass(frozen=True, slots=True)
class LinearPTT:
  """Distributed linear Phan-Thien-Tanner memory."""

  viscosity_pa_s: float
  relaxation_time_s: float
  retardation_time_s: float = 0.0
  extensibility: float = 0.0
  points: int = 240
  extent: float = 60.0
  quadrature: str = "gauss"

  def __post_init__(self) -> None:
    _validate_memory_parameters(self.viscosity_pa_s, self.relaxation_time_s, self.retardation_time_s, polymer_required=True)
    _validate_distributed_model("extensibility", self.extensibility, self.points, self.extent)
    if self.quadrature not in ("trapezoid", "gauss"): raise ValueError("quadrature must be 'trapezoid' or 'gauss'")

MaterialModel = RelaxingMaterial | CubicZener | TwoModeQuadraticZener | CarreauZener | (
  NoStress | NeoHookeanKelvinVoigt | QuadraticKelvinVoigt | Zener | QuadraticZener | OldroydB | InstantaneousMaterial | Giesekus | LinearPTT | LinearMaxwell
)

LAW_WIDTH = 5

def law_values(model):
  """A law's float fields in declaration order, padded to `LAW_WIDTH`.

  `None` when the law has a field a fixed-width float vector cannot carry -- `Ogden`,
  whose moduli and exponents are variable-length tuples. Those stay baked into the
  compiled program and keyed by content (#196).
  """
  if model is None: return (0.0,) * LAW_WIDTH
  values = []
  for field in dataclasses.fields(model):
    value = getattr(model, field.name)
    if not isinstance(value, float): return None
    values.append(value)
  if len(values) > LAW_WIDTH: return None
  return (*values, *(0.0,) * (LAW_WIDTH - len(values)))

def law_with_values(model, values):
  """`model` with its float fields replaced, built WITHOUT running validation.

  The validation already ran on the concrete object; here the values are tracers, which
  `np.isfinite` would reject. `object.__new__` keeps the type, so every `isinstance`
  branch in `_stress` dispatches exactly as before.
  """
  clone = object.__new__(type(model))
  for field, value in zip(dataclasses.fields(model), values, strict=False):
    object.__setattr__(clone, field.name, value)
  return clone

def _is_distributed_stress(material) -> bool: return isinstance(material, (Giesekus, LinearPTT))

def _stress_state_count(material) -> int:
  if _is_distributed_stress(material): return 2 * material.points
  if isinstance(material, (Zener, QuadraticZener, CubicZener, LinearMaxwell, CarreauZener,
                           RelaxingMaterial)): return 1
  if isinstance(material, (OldroydB, TwoModeQuadraticZener)): return 2
  return 0
