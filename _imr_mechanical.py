"""Compiled mechanical RHS used by high-throughput sensitivity solves."""

from __future__ import annotations

import numpy as np
from numba import njit

P_PV = 0
P_KAPPA = 1
P_PB = 2
P_REQ = 3
P_CA = 4
P_RE8 = 5
P_DE = 6
P_LAM = 7
P_ALPHA = 8
P_P8 = 9
P_T0 = 10
P_VISCOSITY_SCALE = 11
P_CSTAR = 12
P_IWE = 13
P_TAIT_GAMMA = 14
P_TAIT_SAM = 15
P_TAIT_NO = 16
P_TAIT_EXPONENT = 17
P_HUGONIOT = 18
P_NOG = 19
P_MIE_REFERENCE = 20
P_EE = 21
P_OM = 22
P_TW = 23
P_DT = 24
P_MN = 25
P_WAVE_TYPE = 26


@njit(cache=True)
def _forcing(time, p):
  wave_type = int(p[P_WAVE_TYPE].real)
  amplitude = p[P_EE]
  if amplitude == 0.0:
    return 0.0j, 0.0j
  omega = p[P_OM]
  width = p[P_TW]
  delay = p[P_DT]
  exponent = p[P_MN]
  if wave_type == 0:
    return amplitude, 0.0j
  if wave_type == 1:
    pulse = np.exp(-((time - delay) ** 2) / width**2)
    return (-amplitude * pulse, amplitude * 2.0 * (time - delay) / width**2 * pulse)
  if wave_type == 2:
    lower = (delay - np.pi / omega).real
    upper = (delay + np.pi / omega).real
    if time.real < lower or time.real > upper:
      return 0.0j, 0.0j
    cosine = 0.5 + 0.5 * np.cos(omega * (time - delay))
    return (
      amplitude * cosine**exponent,
      -amplitude * exponent * cosine ** (exponent - 1.0) * 0.5 * omega * np.sin(omega * (time - delay)),
    )
  return (-amplitude * (1.0 - (1.0 if time.real > width.real else 0.0)), 0.0j)


@njit(cache=True)
def _elastic_integrand(code, parameters, stretch, pressure_scale):
  invariant_offset = stretch**-4 + 2.0 * stretch**2 - 3.0
  geometric = (stretch**3 + 1.0) / stretch**5
  if code == 1:
    coefficient = parameters[0] / pressure_scale
    return -2.0 * coefficient * geometric
  if code == 2:
    return -4.0 * parameters[0] / pressure_scale * geometric - 4.0 * parameters[1] / pressure_scale * (
      1.0 + stretch**-3
    )
  if code == 3:
    coefficient = (
      2.0
      * (parameters[0] + 2.0 * parameters[1] * invariant_offset + 3.0 * parameters[2] * invariant_offset**2)
      / pressure_scale
    )
  elif code == 4:
    coefficient = parameters[0] / pressure_scale * np.exp(parameters[1] * invariant_offset)
  elif code == 5:
    coefficient = parameters[0] / pressure_scale / (1.0 - invariant_offset / parameters[1])
  else:
    invariant = invariant_offset + 3.0
    segments = parameters[1]
    series = (
      0.5
      + 2.0 / 20.0 * invariant / segments
      + 3.0 * 11.0 / 1050.0 * invariant**2 / segments**2
      + 4.0 * 19.0 / 7000.0 * invariant**3 / segments**3
      + 5.0 * 519.0 / 673750.0 * invariant**4 / segments**4
    )
    coefficient = 2.0 * parameters[0] / pressure_scale * series
  return -2.0 * coefficient * geometric


@njit(cache=True)
def _viscosity(code, parameters, shear_rate):
  if code == 1:
    viscosity = parameters[0]
    return viscosity, viscosity
  if code == 2:
    consistency, exponent, regularization = parameters[:3]
    effective = np.sqrt(shear_rate**2 + regularization**2)
    viscosity = consistency * effective ** (exponent - 1.0)
    tangent = viscosity + consistency * (exponent - 1.0) * shear_rate**2 * effective ** (exponent - 3.0)
    return viscosity, tangent
  if code == 3:
    zero, infinite, time_constant, transition, index = parameters[:5]
    if abs(shear_rate) < 1e-25:
      return zero, zero
    scaled = (time_constant * shear_rate) ** transition
    power = (index - 1.0) / transition
    difference = zero - infinite
    viscosity = infinite + difference * (1.0 + scaled) ** power
    tangent = viscosity + difference * (index - 1.0) * scaled * (1.0 + scaled) ** (power - 1.0)
    return viscosity, tangent
  if code == 4:
    zero, infinite, time_constant, transition = parameters[:4]
    if abs(shear_rate) < 1e-25:
      return zero, zero
    scaled = (time_constant * shear_rate) ** transition
    difference = zero - infinite
    viscosity = infinite + difference / (1.0 + scaled)
    tangent = viscosity - difference * transition * scaled / (1.0 + scaled) ** 2
    return viscosity, tangent
  if code == 7 or code == 8:
    # NOTE: mechanical_tangent_rhs differentiates this by complex step, so
    # every operation on the value path must stay analytic. sqrt(x*x) is
    # the analytic |x|; abs() would discard the imaginary perturbation.
    # log1p/arcsinh are spelled via log so numba handles complex input.
    zero, infinite, time_constant = parameters[:3]
    difference = zero - infinite
    scaled = time_constant * shear_rate
    u = np.sqrt(scaled * scaled)
    if abs(u) < 1e-4:
      if code == 8:
        factor = 1.0 - u / 2.0 + u * u / 3.0
        slope = -u / 2.0 + 2.0 * u * u / 3.0
      else:
        factor = 1.0 - u * u / 6.0 + 3.0 * u**4 / 40.0
        slope = -u * u / 3.0 + 3.0 * u**4 / 10.0
    elif code == 8:
      logged = np.log(1.0 + u)
      factor = logged / u
      slope = (u / (1.0 + u) - logged) / u
    else:
      arc = np.log(u + np.sqrt(u * u + 1.0))
      factor = arc / u
      slope = (u / np.sqrt(1.0 + u * u) - arc) / u
    viscosity = infinite + difference * factor
    return viscosity, viscosity + difference * slope
  if code == 5:
    yield_stress, consistency, exponent, regularization = parameters[:4]
  else:
    yield_stress, consistency, regularization = parameters[:3]
    exponent = 1.0
  scaled = shear_rate / regularization
  if shear_rate == 0.0:
    yield_viscosity = yield_stress / regularization
  else:
    yield_viscosity = -yield_stress * np.expm1(-scaled) / shear_rate
  effective = np.sqrt(shear_rate**2 + regularization**2)
  power_viscosity = consistency * effective ** (exponent - 1.0)
  viscosity = yield_viscosity + power_viscosity
  tangent = (
    yield_stress / regularization * np.exp(-scaled)
    + power_viscosity
    + consistency * (exponent - 1.0) * shear_rate**2 * effective ** (exponent - 3.0)
  )
  return viscosity, tangent


@njit(cache=True)
def _instantaneous_stress(
  elastic_code, elastic_parameters, viscous_code, viscous_parameters, nodes, weights, p, radius, velocity, need_rate
):
  stress = 0.0j
  explicit_rate = 0.0j
  acceleration = 0.0j
  if elastic_code:
    wall_stretch = radius / p[P_REQ]
    half = 0.5 * (wall_stretch - 1.0)
    integral = 0.0j
    for index in range(nodes.size):
      stretch = 1.0 + half * (nodes[index] + 1.0)
      integral += weights[index] * _elastic_integrand(elastic_code, elastic_parameters, stretch, p[P_P8])
    stress += half * integral
    if need_rate:
      explicit_rate += _elastic_integrand(elastic_code, elastic_parameters, wall_stretch, p[P_P8]) * velocity / p[P_REQ]
  if viscous_code:
    strain_rate = velocity / radius
    sign = -1.0 if strain_rate.real < 0.0 else 1.0
    viscosity_integral = 0.0j
    tangent_integral = 0.0j
    for index in range(nodes.size):
      quadrature_radius = 0.5 * (nodes[index] + 1.0)
      quadrature_weight = 0.5 * weights[index]
      shear_rate = 2.0 * np.sqrt(3.0) * sign * strain_rate / p[P_T0] * quadrature_radius**3
      viscosity, tangent = _viscosity(viscous_code, viscous_parameters, shear_rate)
      weighted_radius = quadrature_radius**2 * quadrature_weight
      viscosity_integral += weighted_radius * viscosity
      tangent_integral += weighted_radius * tangent
    stress += -12.0 * strain_rate * viscosity_integral / p[P_VISCOSITY_SCALE]
    if need_rate:
      stress_tangent = -12.0 / p[P_VISCOSITY_SCALE] * tangent_integral
      explicit_rate -= stress_tangent * strain_rate**2
      acceleration -= stress_tangent
  return stress, explicit_rate, acceleration


@njit(cache=True)
def _stress(
  material_code,
  elastic_code,
  elastic_parameters,
  viscous_code,
  viscous_parameters,
  nodes,
  weights,
  p,
  radius,
  velocity,
  state,
  need_rate,
  reference_radius,
  reference_radius_cubed,
):
  ratio = p[P_REQ] / radius
  cauchy = p[P_CA]
  reynolds = p[P_RE8]
  relaxation = p[P_DE]
  retardation = p[P_LAM]
  stiffening = p[P_ALPHA]
  rates = np.zeros(state.size, dtype=np.complex128)
  if material_code == 0:
    return 0.0j, 0.0j, 0.0j, rates
  if material_code == 1:
    stress = -(5.0 - 4.0 * ratio - ratio**4) / (2.0 * cauchy) - 4.0 / reynolds * velocity / radius
    stress_rate = -2.0 * velocity / radius * (ratio + ratio**4) / cauchy + 4.0 / reynolds * (velocity / radius) ** 2
    return stress, stress_rate, 4.0 / reynolds, rates
  if material_code == 2:
    stress = (
      (3.0 * stiffening - 1.0) * (5.0 - ratio**4 - 4.0 * ratio) / (2.0 * cauchy)
      - 4.0 / reynolds * velocity / radius
      + 2.0 * stiffening / cauchy * (27.0 / 40.0 + ratio**8 / 8.0 + ratio**5 / 5.0 + ratio**2 - 2.0 / ratio)
    )
    stress_rate = (
      velocity / radius * (3.0 * stiffening - 1.0) / (2.0 * cauchy) * (4.0 * ratio**4 + 4.0 * ratio)
      + 4.0 * (velocity / radius) ** 2 / reynolds
      - 2.0 * stiffening / cauchy * velocity / radius * (ratio**8 + ratio**5 + 2.0 * ratio**2 + 2.0 / ratio)
    )
    return stress, stress_rate, 4.0 / reynolds, rates
  if material_code == 3 or material_code == 4:
    z1 = state[0]
    stress = z1 / radius**3 - 4.0 * retardation / reynolds * velocity / radius
    if material_code == 3:
      equilibrium = -0.5 * radius**3 / cauchy * (5.0 - ratio**4 - 4.0 * ratio)
    else:
      equilibrium = radius**3 * (
        (3.0 * stiffening - 1.0) / (2.0 * cauchy) * (5.0 - ratio**4 - 4.0 * ratio)
        + 2.0 * stiffening / cauchy * (0.675 + 0.125 * ratio**8 + 0.2 * ratio**5 + ratio**2 - 2.0 / ratio)
      )
    rates[0] = (
      -(z1 - equilibrium) / relaxation + 4.0 * (retardation - 1.0) / (reynolds * relaxation) * radius**2 * velocity
    )
    stress_rate = (
      rates[0] / radius**3 - 3.0 * velocity / radius**4 * z1 + 4.0 * retardation / reynolds * (velocity / radius) ** 2
    )
    return stress, stress_rate, 4.0 / reynolds, rates
  if material_code == 5:
    z1, z2 = state[0], state[1]
    rates[0] = (
      -(1.0 / relaxation - 2.0 * velocity / radius) * z1
      + 2.0 * (retardation - 1.0) / (reynolds * relaxation) * radius**2 * velocity
    )
    rates[1] = (
      -(1.0 / relaxation + velocity / radius) * z2
      + 2.0 * (retardation - 1.0) / (reynolds * relaxation) * radius**2 * velocity
    )
    stress = (z1 + z2) / radius**3 - 4.0 * retardation / reynolds * velocity / radius
    stress_rate = (
      (rates[0] + rates[1]) / radius**3
      - 3.0 * velocity / radius**4 * (z1 + z2)
      + 4.0 * retardation / reynolds * (velocity / radius) ** 2
    )
    return (stress, stress_rate, 4.0 * retardation / reynolds, rates)
  if material_code == 7 or material_code == 8:
    points = reference_radius.size
    radius_values = np.empty(points, dtype=np.complex128)
    material_velocity = np.empty(points, dtype=np.complex128)
    integrand = np.empty(points, dtype=np.complex128)
    integrand_rate = np.empty(points, dtype=np.complex128)
    polymer_viscosity = (1.0 - retardation) / reynolds
    nonlinear_parameter = elastic_parameters[0]
    for index in range(points):
      radius_cubed = reference_radius_cubed[index] + radius**3 - 1.0
      material_radius = radius_cubed ** (1.0 / 3.0)
      radius_values[index] = material_radius
      local_rate = velocity * radius**2 / radius_cubed
      radial_stress = state[index]
      hoop_stress = state[index + points]
      if material_code == 7:
        nonlinear_scale = nonlinear_parameter / polymer_viscosity
        radial_rate = (
          -radial_stress / relaxation
          - 4.0 * local_rate * radial_stress
          - nonlinear_scale * radial_stress**2
          - 4.0 * polymer_viscosity * local_rate / relaxation
        )
        hoop_rate = (
          -hoop_stress / relaxation
          + 2.0 * local_rate * hoop_stress
          - nonlinear_scale * hoop_stress**2
          + 2.0 * polymer_viscosity * local_rate / relaxation
        )
      else:
        nonlinear_scale = nonlinear_parameter / polymer_viscosity
        trace_factor = 1.0 + nonlinear_scale * (radial_stress + 2.0 * hoop_stress)
        radial_rate = (
          -trace_factor * radial_stress / relaxation
          - 4.0 * local_rate * radial_stress
          - 4.0 * polymer_viscosity * local_rate / relaxation
        )
        hoop_rate = (
          -trace_factor * hoop_stress / relaxation
          + 2.0 * local_rate * hoop_stress
          + 2.0 * polymer_viscosity * local_rate / relaxation
        )
      rates[index] = radial_rate
      rates[index + points] = hoop_rate
      difference = radial_stress - hoop_stress
      integrand[index] = 2.0 * difference / material_radius
      material_velocity[index] = radius**2 * velocity / material_radius**2
      integrand_rate[index] = 2.0 * (
        (radial_rate - hoop_rate) / material_radius - difference * material_velocity[index] / material_radius**2
      )
    polymer_integral = 0.0j
    polymer_rate = 0.0j
    for index in range(points - 1):
      interval = radius_values[index + 1] - radius_values[index]
      interval_rate = material_velocity[index + 1] - material_velocity[index]
      polymer_integral += 0.5 * (integrand[index] + integrand[index + 1]) * interval
      if need_rate:
        polymer_rate += 0.5 * (
          (integrand_rate[index] + integrand_rate[index + 1]) * interval
          + (integrand[index] + integrand[index + 1]) * interval_rate
        )
    solvent = 4.0 * retardation / reynolds
    stress = polymer_integral - solvent * velocity / radius
    stress_rate = polymer_rate + solvent * (velocity / radius) ** 2
    return stress, stress_rate, solvent, rates
  stress, stress_rate, acceleration = _instantaneous_stress(
    elastic_code, elastic_parameters, viscous_code, viscous_parameters, nodes, weights, p, radius, velocity, need_rate
  )
  return stress, stress_rate, acceleration, rates


@njit(cache=True)
def _mie_mu(compression, slope, nog):
  a = compression * slope**2 - nog
  b = -2.0 * compression * slope - 1.0
  discriminant = b**2 - 4.0 * a * compression
  return (-b + np.sqrt(discriminant)) / (2.0 * a)


@njit(cache=True)
def _mie_antiderivative(mu, slope, nog):
  shifted = 1.0 - slope * mu
  return (
    (2.0 * nog + slope - 1.0) / (slope + 1.0) ** 3 * np.log(shifted / (mu + 1.0))
    + (nog + slope) / (slope * (slope + 1.0) * shifted**2)
    - (2.0 * nog + slope - 1.0) / ((slope + 1.0) ** 2 * shifted)
  )


@njit(cache=True)
def mechanical_rhs(
  time,
  state,
  p,
  material_code,
  elastic_code,
  elastic_parameters,
  viscous_code,
  viscous_parameters,
  nodes,
  weights,
  reference_radius,
  reference_radius_cubed,
  radial,
):
  radius = state[0]
  velocity = state[1]
  pressure = (p[P_PB] - p[P_PV]) * radius ** (-3.0 * p[P_KAPPA]) + p[P_PV]
  pressure_rate = -3.0 * p[P_KAPPA] * (p[P_PB] - p[P_PV]) * radius ** (-3.0 * p[P_KAPPA] - 1.0) * velocity
  stress, stress_rate, acceleration, material_rates = _stress(
    material_code,
    elastic_code,
    elastic_parameters,
    viscous_code,
    viscous_parameters,
    nodes,
    weights,
    p,
    radius,
    velocity,
    state[2:],
    radial != 1,
    reference_radius,
    reference_radius_cubed,
  )
  forcing, forcing_rate = _forcing(time, p)
  surface = p[P_IWE]
  if radial == 1:
    radial_acceleration = (pressure - 1.0 - forcing - surface / radius + stress - 1.5 * velocity**2) / radius
  elif radial == 2:
    sound = p[P_CSTAR]
    numerator = (
      (1.0 + velocity / sound) * (pressure - 1.0 - forcing - surface / radius + stress)
      + radius / sound * (pressure_rate + surface * velocity / radius**2 + stress_rate - forcing_rate)
      - 1.5 * (1.0 - velocity / (3.0 * sound)) * velocity**2
    )
    denominator = (1.0 - velocity / sound) * radius + acceleration / sound
    radial_acceleration = numerator / denominator
  elif radial == 3 or radial == 4:
    bubble_pressure = pressure - surface / radius + p[P_TAIT_GAMMA] + stress
    density_factor = (p[P_TAIT_SAM] / bubble_pressure) ** (1.0 / p[P_TAIT_EXPONENT])
    enthalpy = p[P_TAIT_SAM] / p[P_TAIT_NO] * ((bubble_pressure / p[P_TAIT_SAM]) ** p[P_TAIT_NO] - 1.0)
    sound = p[P_CSTAR]
    if radial == 4:
      density = (bubble_pressure / p[P_TAIT_SAM]) ** (1.0 / p[P_TAIT_EXPONENT])
      sound = np.sqrt(p[P_TAIT_EXPONENT] * bubble_pressure / density)
    numerator = (
      (1.0 + velocity / sound) * (enthalpy - forcing)
      - radius / sound * forcing_rate
      + radius / sound * density_factor * (pressure_rate + surface * velocity / radius**2 + stress_rate)
      - 1.5 * (1.0 - velocity / (3.0 * sound)) * velocity**2
    )
    denominator = (1.0 - velocity / sound) * radius + acceleration * density_factor / sound
    radial_acceleration = numerator / denominator
  else:
    sound = p[P_CSTAR]
    bubble_pressure = pressure - surface / radius
    compression = bubble_pressure / sound**2
    mu = _mie_mu(compression, p[P_HUGONIOT], p[P_NOG])
    density_factor = 1.0 / (1.0 + mu)
    enthalpy = sound**2 * (_mie_antiderivative(mu, p[P_HUGONIOT], p[P_NOG]) - p[P_MIE_REFERENCE])
    numerator = (
      (1.0 + velocity / sound) * (enthalpy - forcing)
      - radius / sound * forcing_rate
      + radius / sound * density_factor * (pressure_rate + surface * velocity / radius**2 + stress_rate)
      - 1.5 * (1.0 - velocity / (3.0 * sound)) * velocity**2
    )
    denominator = (1.0 - velocity / sound) * radius + acceleration * density_factor / sound
    radial_acceleration = numerator / denominator
  output = np.empty(state.size, dtype=np.complex128)
  output[0] = velocity
  output[1] = radial_acceleration
  output[2:] = material_rates
  return output


@njit(cache=True)
def mechanical_tangent_rhs(
  time_s,
  matrix,
  parameter_values,
  parameter_tangents,
  material_code,
  elastic_code,
  elastic_values,
  elastic_tangents,
  viscous_code,
  viscous_values,
  viscous_tangents,
  nodes,
  weights,
  reference_radius,
  reference_radius_cubed,
  radial,
):
  width = matrix.shape[1] - 1
  result = np.empty_like(matrix)
  base_parameters = parameter_values.astype(np.complex128)
  base_state = matrix[:, 0].astype(np.complex128)
  base_elastic = elastic_values.astype(np.complex128)
  base_viscous = viscous_values.astype(np.complex128)
  nondimensional_time = time_s / base_parameters[P_T0]
  base_output = (
    mechanical_rhs(
      nondimensional_time,
      base_state,
      base_parameters,
      material_code,
      elastic_code,
      base_elastic,
      viscous_code,
      base_viscous,
      nodes,
      weights,
      reference_radius,
      reference_radius_cubed,
      radial,
    )
    / base_parameters[P_T0]
  )
  result[:, 0] = base_output.real
  step = 1e-30
  for direction in range(width):
    complex_parameters = base_parameters + 1j * step * parameter_tangents[:, direction]
    complex_state = base_state + 1j * step * matrix[:, direction + 1]
    complex_elastic = base_elastic + 1j * step * elastic_tangents[:, direction]
    complex_viscous = base_viscous + 1j * step * viscous_tangents[:, direction]
    nondimensional_time = time_s / complex_parameters[P_T0]
    output = (
      mechanical_rhs(
        nondimensional_time,
        complex_state,
        complex_parameters,
        material_code,
        elastic_code,
        complex_elastic,
        viscous_code,
        complex_viscous,
        nodes,
        weights,
        reference_radius,
        reference_radius_cubed,
        radial,
      )
      / complex_parameters[P_T0]
    )
    result[:, direction + 1] = output.imag / step
  return result


@njit(cache=True)
def mechanical_stress_tangent(
  matrix,
  parameter_values,
  parameter_tangents,
  material_code,
  elastic_code,
  elastic_values,
  elastic_tangents,
  viscous_code,
  viscous_values,
  viscous_tangents,
  nodes,
  weights,
  reference_radius,
  reference_radius_cubed,
):
  width = matrix.shape[1] - 1
  result = np.empty(width + 1)
  base_parameters = parameter_values.astype(np.complex128)
  base_state = matrix[:, 0].astype(np.complex128)
  base_elastic = elastic_values.astype(np.complex128)
  base_viscous = viscous_values.astype(np.complex128)
  stress = _stress(
    material_code,
    elastic_code,
    base_elastic,
    viscous_code,
    base_viscous,
    nodes,
    weights,
    base_parameters,
    base_state[0],
    base_state[1],
    base_state[2:],
    False,
    reference_radius,
    reference_radius_cubed,
  )[0]
  result[0] = stress.real
  step = 1e-30
  for direction in range(width):
    complex_parameters = base_parameters + 1j * step * parameter_tangents[:, direction]
    complex_state = base_state + 1j * step * matrix[:, direction + 1]
    complex_elastic = base_elastic + 1j * step * elastic_tangents[:, direction]
    complex_viscous = base_viscous + 1j * step * viscous_tangents[:, direction]
    stress = _stress(
      material_code,
      elastic_code,
      complex_elastic,
      viscous_code,
      complex_viscous,
      nodes,
      weights,
      complex_parameters,
      complex_state[0],
      complex_state[1],
      complex_state[2:],
      False,
      reference_radius,
      reference_radius_cubed,
    )[0]
    result[direction + 1] = stress.imag / step
  return result
