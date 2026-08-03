"""Constitutive stress evaluation."""

from __future__ import annotations

import numpy as np

from ._materials import (
  LinearMaxwell,
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
  Ogden,
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

_OGDEN_SERIES_LIMIT = 1e-3

def _ogden_ratio(u, exponent, *, xp=np):
  offset = u - 1.0
  near = xp.abs(offset) < _OGDEN_SERIES_LIMIT
  series = exponent * (
    1.0
    + (exponent - 1.0) / 2.0 * offset
    + (exponent - 1.0) * (exponent - 2.0) / 6.0 * offset**2
    + (exponent - 1.0) * (exponent - 2.0) * (exponent - 3.0) / 24.0 * offset**3
  )
  if xp is np and np.ndim(near) == 0: return series if near else (1.0 - u**exponent) / (1.0 - u)
  safe = xp.where(near, 1.0 + 2.0 * _OGDEN_SERIES_LIMIT, u)
  return xp.where(near, series, (1.0 - safe**exponent) / (1.0 - safe))

def _elastic_integrand(model, stretch, pressure_scale, *, xp=np):
  stretch = xp.asarray(stretch)
  if xp is np:
    # value guards are numpy-only; a tracer has no value to test
    if np.any(stretch <= 0.0): raise _MaterialDomainError("elastic stretch became non-positive")
  invariant_offset = stretch**-4 + 2.0 * stretch**2 - 3.0
  geometric_factor = (stretch**3 + 1.0) / stretch**5
  if isinstance(model, Ogden):
    u = stretch**3
    total = 0.0
    for modulus, exponent in zip(model.shear_moduli_pa, model.exponents, strict=True):
      total = total + modulus * stretch ** (-2.0 * exponent) * _ogden_ratio(u, exponent, xp=xp)
    result = -2.0 / (stretch * pressure_scale) * total
  elif isinstance(model, NeoHookean):
    coefficient = model.shear_modulus_pa / pressure_scale
    result = -2.0 * coefficient * geometric_factor
  elif isinstance(model, MooneyRivlin):
    result = -4.0 * model.c10_pa / pressure_scale * geometric_factor - 4.0 * model.c01_pa / pressure_scale * (1.0 + stretch**-3)
  else:
    if isinstance(model, Yeoh):
      coefficient = 2.0 * (model.c1_pa + 2.0 * model.c2_pa * invariant_offset + 3.0 * model.c3_pa * invariant_offset**2) / pressure_scale
    elif isinstance(model, Fung):
      coefficient = model.shear_modulus_pa / pressure_scale * xp.exp(model.stiffening * invariant_offset)
    elif isinstance(model, Gent):
      remaining_extension = 1.0 - invariant_offset / model.extensibility
      if xp is np and np.any(remaining_extension <= 0.0):
        maximum = float(np.max(invariant_offset))
        raise _MaterialDomainError(f"Gent lock-up: I1 - 3 reached {maximum:.6g}, limit is {float(model.extensibility):.6g}")
      coefficient = model.shear_modulus_pa / pressure_scale / remaining_extension
    else:
      invariant = invariant_offset + 3.0
      coefficients = (0.5, 1 / 20, 11 / 1050, 19 / 7000, 519 / 673750)
      series = sum((order + 1) * coefficient * invariant**order / model.chain_segments**order for order, coefficient in enumerate(coefficients))
      coefficient = 2.0 * model.shear_modulus_pa / pressure_scale * series
    result = -2.0 * coefficient * geometric_factor
  if xp is np and not np.all(np.isfinite(result)): raise _MaterialDomainError("elastic stress became non-finite")
  return result

_PE_SERIES_LIMIT = 1e-4

def _powell_eyring_terms(u, modified, *, xp=np):
  """`(f, s)`: the shape factor `F(u)` and the slope term `u F'(u)`."""
  if modified:
    series = (1.0 - u / 2.0 + u**2 / 3.0, -u / 2.0 + 2.0 * u**2 / 3.0)
  else:
    series = (1.0 - u**2 / 6.0 + 3.0 * u**4 / 40.0, -(u**2) / 3.0 + 3.0 * u**4 / 10.0)
  safe = xp.maximum(u, _PE_SERIES_LIMIT)
  if modified:
    exact_f = xp.log1p(safe) / safe
    exact_s = (safe / (1.0 + safe) - xp.log1p(safe)) / safe
  else:
    exact_f = xp.arcsinh(safe) / safe
    exact_s = (safe / xp.sqrt(1.0 + safe**2) - xp.arcsinh(safe)) / safe
  small = u < _PE_SERIES_LIMIT
  return xp.where(small, series[0], exact_f), xp.where(small, series[1], exact_s)

def _viscosity_and_tangent(model, shear_rate, *, xp=np):
  shear_rate = xp.asarray(shear_rate)
  if isinstance(model, Newtonian):
    viscosity = xp.full_like(shear_rate, model.viscosity_pa_s)
    tangent = viscosity
  elif isinstance(model, PowerLaw):
    effective_rate = xp.sqrt(shear_rate**2 + model.regularization_rate_per_s**2)
    viscosity = model.consistency_pa_s_n * effective_rate ** (model.exponent - 1.0)
    tangent = viscosity + (model.consistency_pa_s_n * (model.exponent - 1.0) * shear_rate**2 * effective_rate ** (model.exponent - 3.0))
  elif isinstance(model, CarreauYasuda):
    scaled = (model.time_constant_s * shear_rate) ** model.transition_exponent
    power = (model.power_index - 1.0) / model.transition_exponent
    difference = model.zero_shear_viscosity_pa_s - model.infinite_shear_viscosity_pa_s
    viscosity = model.infinite_shear_viscosity_pa_s + difference * (1.0 + scaled) ** power
    tangent = viscosity + (difference * (model.power_index - 1.0) * scaled * (1.0 + scaled) ** (power - 1.0))
  elif isinstance(model, (PowellEyring, ModifiedPowellEyring)):
    modified = isinstance(model, ModifiedPowellEyring)
    difference = model.zero_shear_viscosity_pa_s - model.infinite_shear_viscosity_pa_s
    scaled = xp.absolute(model.time_constant_s * shear_rate)
    factor, slope = _powell_eyring_terms(scaled, modified, xp=xp)
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
    positive = shear_rate > 0.0
    denominator = xp.where(positive, shear_rate, 1.0)
    yield_viscosity = xp.where(positive, -yield_stress * xp.expm1(-scaled) / denominator, yield_stress / regularization)
    effective_rate = xp.sqrt(shear_rate**2 + regularization**2)
    power_viscosity = consistency * effective_rate ** (exponent - 1.0)
    viscosity = yield_viscosity + power_viscosity
    tangent = (
      yield_stress / regularization * xp.exp(-scaled)
      + power_viscosity
      + consistency * (exponent - 1.0) * shear_rate**2 * effective_rate ** (exponent - 3.0)
    )
  if xp is np:
    viscosity_values, tangent_values = viscosity, tangent
    if not np.all(np.isfinite(viscosity_values)) or not np.all(np.isfinite(tangent_values)) or np.any(viscosity_values < 0.0):
      raise _MaterialDomainError("generalized viscosity became invalid")
  return viscosity, tangent

def _instantaneous_stress(material, prepared, p, R, Rd, need_rate, *, xp=np):
  stress_integral = 0.0
  explicit_rate = 0.0
  acceleration_coefficient = 0.0
  if material.elastic is not None:
    wall_stretch = R / p["req"]
    half_interval = 0.5 * (wall_stretch - 1.0)
    stretch = 1.0 + half_interval * (prepared.interval_nodes + 1.0)
    integrand = _elastic_integrand(material.elastic, stretch, p["P8"], xp=xp)
    stress_integral += half_interval * xp.dot(prepared.interval_weights, integrand)
    if need_rate:
      wall_integrand = _elastic_integrand(material.elastic, wall_stretch, p["P8"], xp=xp)
      if isinstance(wall_integrand, np.ndarray): wall_integrand = wall_integrand.item()
      explicit_rate += wall_integrand * Rd / p["req"]
  if material.viscous is not None:
    quadrature_radius = 0.5 * (prepared.interval_nodes + 1.0)
    quadrature_weights = 0.5 * prepared.interval_weights
    strain_rate = Rd / R
    shear_rate = 2.0 * xp.sqrt(3.0) * abs(strain_rate) / p["t0"] * quadrature_radius**3
    viscosity, tangent = _viscosity_and_tangent(material.viscous, shear_rate, xp=xp)
    weighted_radius = quadrature_radius**2 * quadrature_weights
    viscosity_integral = xp.dot(weighted_radius, viscosity)
    stress_integral += -12.0 * strain_rate * viscosity_integral / p["viscosity_scale"]
    if need_rate:
      stress_tangent = -12.0 / p["viscosity_scale"] * xp.dot(weighted_radius, tangent)
      explicit_rate -= stress_tangent * strain_rate**2
      acceleration_coefficient -= stress_tangent
  return stress_integral, explicit_rate, None, acceleration_coefficient

def _stress(material, p, R, Rd, Z, instantaneous=None, need_rate=True, *, xp=np):
  Rst = p["req"] / R
  Ca, Re8, De, LAM, ax = p["Ca"], p["Re8"], p["De"], p["LAM"], p["alphax"]
  if isinstance(material, NoStress): return 0.0, 0.0, None, 0.0
  if isinstance(material, NeoHookeanKelvinVoigt):
    S = -(5 - 4 * Rst - Rst**4) / (2 * Ca) - 4.0 / Re8 * Rd / R
    Sdot = -2 * Rd / R * (Rst + Rst**4) / Ca + 4.0 / Re8 * (Rd / R) ** 2
    return S, Sdot, None, 4.0 / Re8
  if isinstance(material, QuadraticKelvinVoigt):
    S = (3 * ax - 1) * (5 - Rst**4 - 4 * Rst) / (2 * Ca) - 4.0 / Re8 * Rd / R + (2 * ax / Ca) * (27 / 40 + Rst**8 / 8 + Rst**5 / 5 + Rst**2 - 2 / Rst)
    Sdot = (
      (Rd / R) * ((3 * ax - 1) / (2 * Ca)) * (4 * Rst**4 + 4 * Rst)
      + 4 * (Rd / R) ** 2 / Re8
      - 2 * ax / Ca * Rd / R * (Rst**8 + Rst**5 + 2 * Rst**2 + 2 / Rst)
    )
    return S, Sdot, None, 4.0 / Re8
  if isinstance(material, LinearMaxwell):
    # Zener with Ca infinite, so the elastic target is zero and the memory decays to it.
    # `acceleration_coefficient` is 0, not 4/Re8: S = Z1/R^3 carries no instantaneous Rd,
    # so no R-ddot term moves to the left-hand side of Keller-Miksis.
    Z1 = Z[0]
    S = Z1 / R**3
    Z1d = -Z1 / De - 4.0 / (Re8 * De) * R**2 * Rd
    Sdot = Z1d / R**3 - 3 * Rd / R**4 * Z1
    return S, Sdot, xp.array([Z1d]), 0.0
  if isinstance(material, Zener):
    Z1 = Z[0]
    S = Z1 / R**3 - 4 * LAM / Re8 * Rd / R
    Ze = -0.5 * (R**3 / Ca) * (5 - Rst**4 - 4 * Rst)
    Z1d = -(Z1 - Ze) / De + 4 * (LAM - 1) / (Re8 * De) * R**2 * Rd
    Sdot = Z1d / R**3 - 3 * Rd / R**4 * Z1 + 4 * LAM / Re8 * (Rd / R) ** 2
    return S, Sdot, xp.array([Z1d]), 4.0 * LAM / Re8
  if isinstance(material, QuadraticZener):
    Z1 = Z[0]
    S = Z1 / R**3 - 4 * LAM / Re8 * Rd / R
    strainhard = (3 * ax - 1) / (2 * Ca)
    Ze = R**3 * (strainhard * (5 - Rst**4 - 4 * Rst) + (2 * ax / Ca) * (0.675 + 0.125 * Rst**8 + 0.2 * Rst**5 + Rst**2 - 2 / Rst))
    Z1d = -(Z1 - Ze) / De + 4 * (LAM - 1) / (Re8 * De) * R**2 * Rd
    Sdot = Z1d / R**3 - 3 * Rd / R**4 * Z1 + 4 * LAM / Re8 * Rd**2 / R**2
    return S, Sdot, xp.array([Z1d]), 4.0 * LAM / Re8
  if isinstance(material, OldroydB):
    Z1, Z2 = Z[0], Z[1]
    Z1d = -(1 / De - 2 * Rd / R) * Z1 + 2 * (LAM - 1) / (Re8 * De) * R**2 * Rd
    Z2d = -(1 / De + Rd / R) * Z2 + 2 * (LAM - 1) / (Re8 * De) * R**2 * Rd
    S = (Z1 + Z2) / R**3 - 4 * LAM / Re8 * Rd / R
    Sdot = (Z1d + Z2d) / R**3 - 3 * Rd / R**4 * (Z1 + Z2) + 4 * LAM / Re8 * Rd**2 / R**2
    return S, Sdot, xp.array([Z1d, Z2d]), 4.0 * LAM / Re8
  if isinstance(material, InstantaneousMaterial): return _instantaneous_stress(material, instantaneous, p, R, Rd, need_rate, xp=xp)
  raise TypeError(f"material={material!r} is not an analytic material")

def _distributed_stress(material, prepared, p, R, Rd, state, need_rate, *, xp=np):
  points = prepared.reference_radius.size
  radial_stress = state[:points]
  hoop_stress = state[points:]
  radius_cubed = xp.maximum(prepared.reference_radius_cubed + R**3 - 1.0, 1e-30)
  inverse_radius_cubed = 1.0 / radius_cubed
  strain_rate = Rd * R**2 * inverse_radius_cubed
  polymer_viscosity = (1.0 - p["LAM"]) / p["Re8"]
  if isinstance(material, Giesekus):
    nonlinear_scale = p["nlx"] / polymer_viscosity
    radial_rate = (
      -radial_stress / p["De"]
      - 4.0 * strain_rate * radial_stress
      - nonlinear_scale * radial_stress**2
      - 4.0 * polymer_viscosity * strain_rate / p["De"]
    )
    hoop_rate = (
      -hoop_stress / p["De"] + 2.0 * strain_rate * hoop_stress - nonlinear_scale * hoop_stress**2 + 2.0 * polymer_viscosity * strain_rate / p["De"]
    )
  else:
    nonlinear_scale = p["nlx"] / polymer_viscosity
    trace_factor = 1.0 + nonlinear_scale * (radial_stress + 2.0 * hoop_stress)
    radial_rate = -trace_factor * radial_stress / p["De"] - 4.0 * strain_rate * radial_stress - 4.0 * polymer_viscosity * strain_rate / p["De"]
    hoop_rate = -trace_factor * hoop_stress / p["De"] + 2.0 * strain_rate * hoop_stress + 2.0 * polymer_viscosity * strain_rate / p["De"]
  radius = xp.cbrt(radius_cubed)
  stress_difference = radial_stress - hoop_stress
  polymer_integral_rate = 0.0
  if prepared.weights is not None:
    polymer_integral = 2.0 * xp.sum(prepared.weights * stress_difference * inverse_radius_cubed)
    if need_rate:
      polymer_integral_rate = 2.0 * xp.sum(
        prepared.weights * ((radial_rate - hoop_rate) * inverse_radius_cubed - 3.0 * stress_difference * R**2 * Rd * inverse_radius_cubed**2)
      )
  else:
    integrand = 2.0 * stress_difference / radius
    polymer_integral = xp.trapezoid(integrand, radius)
    if need_rate:
      material_velocity = R**2 * Rd / radius**2
      integrand_rate = 2.0 * ((radial_rate - hoop_rate) / radius - stress_difference * material_velocity / radius**2)
      intervals = xp.diff(radius)
      interval_rates = xp.diff(material_velocity)
      polymer_integral_rate = xp.sum(
        0.5 * ((integrand_rate[:-1] + integrand_rate[1:]) * intervals + (integrand[:-1] + integrand[1:]) * interval_rates)
      )
  solvent_scale = 4.0 * p["LAM"] / p["Re8"]
  stress_integral = polymer_integral - solvent_scale * Rd / R
  explicit_rate = polymer_integral_rate + solvent_scale * (Rd / R) ** 2
  return (stress_integral, explicit_rate, xp.concatenate((radial_rate, hoop_rate)), solvent_scale)

def _distributed_stress_integral(prepared, p, R, Rd, state, *, xp=np):
  points = prepared.reference_radius.size
  radius = xp.cbrt(xp.maximum(prepared.reference_radius_cubed + R**3 - 1.0, 1e-30))
  integrand = 2.0 * (state[:points] - state[points:]) / radius
  polymer_integral = xp.trapezoid(integrand, radius)
  return polymer_integral - 4.0 * p["LAM"] / p["Re8"] * Rd / R
