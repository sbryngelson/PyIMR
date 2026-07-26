"""Constitutive stress evaluation.

Pure evaluation: material plus kinematic state in, stress and its rate out.
No solver state, no preparation, no dependency on the RHS.
"""

from __future__ import annotations

import numpy as np

from _imr_autodiff import primal, primal_array
from _imr_materials import (
  Bingham,
  CarreauYasuda,
  Cross,
  Fung,
  Gent,
  Giesekus,
  InstantaneousMaterial,
  ModifiedPowellEyring,
  MooneyRivlin,
  NeoHookean,
  NeoHookeanKelvinVoigt,
  Newtonian,
  NoStress,
  OldroydB,
  PowellEyring,
  PowerLaw,
  QuadraticKelvinVoigt,
  QuadraticZener,
  Yeoh,
  Zener,
)

__all__ = [
  "_MaterialDomainError",
  "_PE_SERIES_LIMIT",
  "_distributed_stress",
  "_distributed_stress_integral",
  "_elastic_integrand",
  "_instantaneous_stress",
  "_powell_eyring_terms",
  "_stress",
  "_viscosity_and_tangent",
]


class _MaterialDomainError(RuntimeError):
  """Internal signal for a constitutive model leaving its physical domain."""


def _elastic_integrand(model, stretch, pressure_scale):
  stretch = np.asarray(stretch)
  stretch_values = primal_array(stretch)
  if np.any(stretch_values <= 0.0):
    raise _MaterialDomainError("elastic stretch became non-positive")
  invariant_offset = stretch**-4 + 2.0 * stretch**2 - 3.0
  geometric_factor = (stretch**3 + 1.0) / stretch**5
  if isinstance(model, NeoHookean):
    coefficient = model.shear_modulus_pa / pressure_scale
    result = -2.0 * coefficient * geometric_factor
  elif isinstance(model, MooneyRivlin):
    result = -4.0 * model.c10_pa / pressure_scale * geometric_factor - 4.0 * model.c01_pa / pressure_scale * (
      1.0 + stretch**-3
    )
  else:
    if isinstance(model, Yeoh):
      coefficient = (
        2.0
        * (model.c1_pa + 2.0 * model.c2_pa * invariant_offset + 3.0 * model.c3_pa * invariant_offset**2)
        / pressure_scale
      )
    elif isinstance(model, Fung):
      coefficient = model.shear_modulus_pa / pressure_scale * np.exp(model.stiffening * invariant_offset)
    elif isinstance(model, Gent):
      remaining_extension = 1.0 - invariant_offset / model.extensibility
      if np.any(primal_array(remaining_extension) <= 0.0):
        maximum = float(np.max(primal_array(invariant_offset)))
        raise _MaterialDomainError(
          f"Gent lock-up: I1 - 3 reached {maximum:.6g}, limit is {primal(model.extensibility):.6g}"
        )
      coefficient = model.shear_modulus_pa / pressure_scale / remaining_extension
    else:
      invariant = invariant_offset + 3.0
      coefficients = (0.5, 1 / 20, 11 / 1050, 19 / 7000, 519 / 673750)
      series = sum(
        (order + 1) * coefficient * invariant**order / model.chain_segments**order
        for order, coefficient in enumerate(coefficients)
      )
      coefficient = 2.0 * model.shear_modulus_pa / pressure_scale * series
    result = -2.0 * coefficient * geometric_factor
  if not np.all(np.isfinite(primal_array(result))):
    raise _MaterialDomainError("elastic stress became non-finite")
  return result


_PE_SERIES_LIMIT = 1e-4


def _powell_eyring_terms(u, modified):
  if modified:
    series = (1.0 - u / 2.0 + u**2 / 3.0, -u / 2.0 + 2.0 * u**2 / 3.0)
  else:
    series = (
      1.0 - u**2 / 6.0 + 3.0 * u**4 / 40.0,
      -(u**2) / 3.0 + 3.0 * u**4 / 10.0,
    )
  if isinstance(u, np.ndarray) and u.dtype != object:
    safe = np.maximum(u, _PE_SERIES_LIMIT)
    if modified:
      exact_f = np.log1p(safe) / safe
      exact_s = (safe / (1.0 + safe) - np.log1p(safe)) / safe
    else:
      exact_f = np.arcsinh(safe) / safe
      exact_s = (safe / np.sqrt(1.0 + safe**2) - np.arcsinh(safe)) / safe
    small = u < _PE_SERIES_LIMIT
    return np.where(small, series[0], exact_f), np.where(small, series[1], exact_s)
  if u < _PE_SERIES_LIMIT:
    return series
  if modified:
    logged = np.log1p(u)
    return logged / u, (u / (1.0 + u) - logged) / u
  arc = np.arcsinh(u)
  return arc / u, (u / np.sqrt(1.0 + u**2) - arc) / u


def _viscosity_and_tangent(model, shear_rate):
  shear_rate = np.asarray(shear_rate)
  if isinstance(model, Newtonian):
    viscosity = np.full_like(shear_rate, model.viscosity_pa_s)
    tangent = viscosity
  elif isinstance(model, PowerLaw):
    effective_rate = np.sqrt(shear_rate**2 + model.regularization_rate_per_s**2)
    viscosity = model.consistency_pa_s_n * effective_rate ** (model.exponent - 1.0)
    tangent = viscosity + (
      model.consistency_pa_s_n * (model.exponent - 1.0) * shear_rate**2 * effective_rate ** (model.exponent - 3.0)
    )
  elif isinstance(model, CarreauYasuda):
    scaled = (model.time_constant_s * shear_rate) ** model.transition_exponent
    power = (model.power_index - 1.0) / model.transition_exponent
    difference = model.zero_shear_viscosity_pa_s - model.infinite_shear_viscosity_pa_s
    viscosity = model.infinite_shear_viscosity_pa_s + difference * (1.0 + scaled) ** power
    tangent = viscosity + (difference * (model.power_index - 1.0) * scaled * (1.0 + scaled) ** (power - 1.0))
  elif isinstance(model, (PowellEyring, ModifiedPowellEyring)):
    modified = isinstance(model, ModifiedPowellEyring)
    difference = model.zero_shear_viscosity_pa_s - model.infinite_shear_viscosity_pa_s
    scaled = np.absolute(model.time_constant_s * shear_rate)
    if shear_rate.dtype == object:
      factor = np.empty_like(shear_rate)
      slope = np.empty_like(shear_rate)
      for index, value in np.ndenumerate(scaled):
        factor[index], slope[index] = _powell_eyring_terms(value, modified)
    else:
      factor, slope = _powell_eyring_terms(scaled, modified)
    viscosity = model.infinite_shear_viscosity_pa_s + difference * factor
    tangent = viscosity + difference * slope
  elif isinstance(model, Cross):
    scaled = (model.time_constant_s * shear_rate) ** model.transition_exponent
    difference = model.zero_shear_viscosity_pa_s - model.infinite_shear_viscosity_pa_s
    viscosity = model.infinite_shear_viscosity_pa_s + difference / (1.0 + scaled)
    tangent = viscosity - (difference * model.transition_exponent * scaled / (1.0 + scaled) ** 2)
  else:
    if isinstance(model, Bingham):
      yield_stress = model.yield_stress_pa
      consistency = model.plastic_viscosity_pa_s
      exponent = 1.0
      regularization = model.regularization_rate_per_s
    else:
      yield_stress = model.yield_stress_pa
      consistency = model.consistency_pa_s_n
      exponent = model.exponent
      regularization = model.regularization_rate_per_s
    scaled = shear_rate / regularization
    if shear_rate.dtype == object:
      yield_viscosity = np.empty_like(shear_rate)
      for index, rate in np.ndenumerate(shear_rate):
        yield_viscosity[index] = (
          -yield_stress * np.expm1(-scaled[index]) / rate if primal(rate) > 0.0 else yield_stress / regularization
        )
    else:
      with np.errstate(divide="ignore", invalid="ignore"):
        yield_viscosity = np.where(
          shear_rate > 0.0,
          -yield_stress * np.expm1(-scaled) / shear_rate,
          yield_stress / regularization,
        )
    effective_rate = np.sqrt(shear_rate**2 + regularization**2)
    power_viscosity = consistency * effective_rate ** (exponent - 1.0)
    viscosity = yield_viscosity + power_viscosity
    tangent = (
      yield_stress / regularization * np.exp(-scaled)
      + power_viscosity
      + consistency * (exponent - 1.0) * shear_rate**2 * effective_rate ** (exponent - 3.0)
    )
  viscosity_values = primal_array(viscosity)
  tangent_values = primal_array(tangent)
  if (
    not np.all(np.isfinite(viscosity_values))
    or not np.all(np.isfinite(tangent_values))
    or np.any(viscosity_values < 0.0)
  ):
    raise _MaterialDomainError("generalized viscosity became invalid")
  return viscosity, tangent


def _instantaneous_stress(material, prepared, p, R, Rd, need_rate):
  stress_integral = 0.0
  explicit_rate = 0.0
  acceleration_coefficient = 0.0
  if material.elastic is not None:
    wall_stretch = R / p["req"]
    half_interval = 0.5 * (wall_stretch - 1.0)
    stretch = 1.0 + half_interval * (prepared.interval_nodes + 1.0)
    integrand = _elastic_integrand(material.elastic, stretch, p["P8"])
    stress_integral += half_interval * np.dot(prepared.interval_weights, integrand)
    if need_rate:
      wall_integrand = _elastic_integrand(material.elastic, wall_stretch, p["P8"])
      if isinstance(wall_integrand, np.ndarray):
        wall_integrand = wall_integrand.item()
      explicit_rate += wall_integrand * Rd / p["req"]
  if material.viscous is not None:
    quadrature_radius = 0.5 * (prepared.interval_nodes + 1.0)
    quadrature_weights = 0.5 * prepared.interval_weights
    strain_rate = Rd / R
    shear_rate = 2.0 * np.sqrt(3.0) * abs(strain_rate) / p["t0"] * quadrature_radius**3
    viscosity, tangent = _viscosity_and_tangent(material.viscous, shear_rate)
    weighted_radius = quadrature_radius**2 * quadrature_weights
    viscosity_integral = np.dot(weighted_radius, viscosity)
    stress_integral += -12.0 * strain_rate * viscosity_integral / p["viscosity_scale"]
    if need_rate:
      stress_tangent = -12.0 / p["viscosity_scale"] * np.dot(weighted_radius, tangent)
      explicit_rate -= stress_tangent * strain_rate**2
      acceleration_coefficient -= stress_tangent
  return stress_integral, explicit_rate, None, acceleration_coefficient


def _stress(material, p, R, Rd, Z, instantaneous=None, need_rate=True):
  Rst = p["req"] / R
  Ca, Re8, De, LAM, ax = p["Ca"], p["Re8"], p["De"], p["LAM"], p["alphax"]
  if isinstance(material, NoStress):
    return 0.0, 0.0, None, 0.0
  if isinstance(material, NeoHookeanKelvinVoigt):
    S = -(5 - 4 * Rst - Rst**4) / (2 * Ca) - 4.0 / Re8 * Rd / R
    Sdot = -2 * Rd / R * (Rst + Rst**4) / Ca + 4.0 / Re8 * (Rd / R) ** 2
    return S, Sdot, None, 4.0 / Re8
  if isinstance(material, QuadraticKelvinVoigt):
    S = (
      (3 * ax - 1) * (5 - Rst**4 - 4 * Rst) / (2 * Ca)
      - 4.0 / Re8 * Rd / R
      + (2 * ax / Ca) * (27 / 40 + Rst**8 / 8 + Rst**5 / 5 + Rst**2 - 2 / Rst)
    )
    Sdot = (
      (Rd / R) * ((3 * ax - 1) / (2 * Ca)) * (4 * Rst**4 + 4 * Rst)
      + 4 * (Rd / R) ** 2 / Re8
      - 2 * ax / Ca * Rd / R * (Rst**8 + Rst**5 + 2 * Rst**2 + 2 / Rst)
    )
    return S, Sdot, None, 4.0 / Re8
  if isinstance(material, Zener):
    Z1 = Z[0]
    S = Z1 / R**3 - 4 * LAM / Re8 * Rd / R
    Ze = -0.5 * (R**3 / Ca) * (5 - Rst**4 - 4 * Rst)
    Z1d = -(Z1 - Ze) / De + 4 * (LAM - 1) / (Re8 * De) * R**2 * Rd
    Sdot = Z1d / R**3 - 3 * Rd / R**4 * Z1 + 4 * LAM / Re8 * (Rd / R) ** 2
    # IMRv2's compressible radial equations use the full 4/Re8 implicit
    # coefficient for Zener, not the solvent-only coefficient visible in S.
    return S, Sdot, np.array([Z1d]), 4.0 / Re8
  if isinstance(material, QuadraticZener):
    Z1 = Z[0]
    S = Z1 / R**3 - 4 * LAM / Re8 * Rd / R
    strainhard = (3 * ax - 1) / (2 * Ca)
    Ze = R**3 * (
      strainhard * (5 - Rst**4 - 4 * Rst) + (2 * ax / Ca) * (0.675 + 0.125 * Rst**8 + 0.2 * Rst**5 + Rst**2 - 2 / Rst)
    )
    Z1d = -(Z1 - Ze) / De + 4 * (LAM - 1) / (Re8 * De) * R**2 * Rd
    Sdot = Z1d / R**3 - 3 * Rd / R**4 * Z1 + 4 * LAM / Re8 * Rd**2 / R**2
    # Same upstream implicit coefficient convention as the linear Zener.
    return S, Sdot, np.array([Z1d]), 4.0 / Re8
  if isinstance(material, OldroydB):
    Z1, Z2 = Z[0], Z[1]
    Z1d = -(1 / De - 2 * Rd / R) * Z1 + 2 * (LAM - 1) / (Re8 * De) * R**2 * Rd
    Z2d = -(1 / De + Rd / R) * Z2 + 2 * (LAM - 1) / (Re8 * De) * R**2 * Rd
    S = (Z1 + Z2) / R**3 - 4 * LAM / Re8 * Rd / R
    Sdot = (Z1d + Z2d) / R**3 - 3 * Rd / R**4 * (Z1 + Z2) + 4 * LAM / Re8 * Rd**2 / R**2
    return S, Sdot, np.array([Z1d, Z2d]), 4.0 * LAM / Re8
  if isinstance(material, InstantaneousMaterial):
    return _instantaneous_stress(material, instantaneous, p, R, Rd, need_rate)
  raise TypeError(f"material={material!r} is not an analytic material")


def _distributed_stress(material, prepared, p, R, Rd, state, need_rate):
  points = prepared.reference_radius.size
  radial_stress = state[:points]
  hoop_stress = state[points:]
  radius_cubed = np.maximum(
    prepared.reference_radius_cubed + R**3 - 1.0,
    1e-30,
  )
  inverse_radius_cubed = 1.0 / radius_cubed
  strain_rate = Rd * R**2 * inverse_radius_cubed
  polymer_viscosity = (1.0 - p["LAM"]) / p["Re8"]

  if isinstance(material, Giesekus):
    nonlinear_scale = material.mobility / polymer_viscosity
    radial_rate = (
      -radial_stress / p["De"]
      - 4.0 * strain_rate * radial_stress
      - nonlinear_scale * radial_stress**2
      - 4.0 * polymer_viscosity * strain_rate / p["De"]
    )
    hoop_rate = (
      -hoop_stress / p["De"]
      + 2.0 * strain_rate * hoop_stress
      - nonlinear_scale * hoop_stress**2
      + 2.0 * polymer_viscosity * strain_rate / p["De"]
    )
  else:
    nonlinear_scale = material.extensibility / polymer_viscosity
    trace_factor = 1.0 + nonlinear_scale * (radial_stress + 2.0 * hoop_stress)
    radial_rate = (
      -trace_factor * radial_stress / p["De"]
      - 4.0 * strain_rate * radial_stress
      - 4.0 * polymer_viscosity * strain_rate / p["De"]
    )
    hoop_rate = (
      -trace_factor * hoop_stress / p["De"]
      + 2.0 * strain_rate * hoop_stress
      + 2.0 * polymer_viscosity * strain_rate / p["De"]
    )

  radius = np.cbrt(radius_cubed)
  stress_difference = radial_stress - hoop_stress
  integrand = 2.0 * stress_difference / radius
  polymer_integral = np.trapezoid(integrand, radius)
  polymer_integral_rate = 0.0
  if need_rate:
    material_velocity = R**2 * Rd / radius**2
    integrand_rate = 2.0 * ((radial_rate - hoop_rate) / radius - stress_difference * material_velocity / radius**2)
    intervals = np.diff(radius)
    interval_rates = np.diff(material_velocity)
    polymer_integral_rate = np.sum(
      0.5 * ((integrand_rate[:-1] + integrand_rate[1:]) * intervals + (integrand[:-1] + integrand[1:]) * interval_rates)
    )
  solvent_scale = 4.0 * p["LAM"] / p["Re8"]
  stress_integral = polymer_integral - solvent_scale * Rd / R
  explicit_rate = polymer_integral_rate + solvent_scale * (Rd / R) ** 2
  return (
    stress_integral,
    explicit_rate,
    np.concatenate((radial_rate, hoop_rate)),
    solvent_scale,
  )


def _distributed_stress_integral(prepared, p, R, Rd, state):
  points = prepared.reference_radius.size
  radius = np.cbrt(np.maximum(prepared.reference_radius_cubed + R**3 - 1.0, 1e-30))
  integrand = 2.0 * (state[:points] - state[points:]) / radius
  polymer_integral = np.trapezoid(integrand, radius)
  return polymer_integral - 4.0 * p["LAM"] / p["Re8"] * Rd / R
