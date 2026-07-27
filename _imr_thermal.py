"""Equation of state, thermal dissipation, and wall boundary closure.

Owns the Tait and Mie-Gruneisen parameters, saturated vapour pressure, the
viscous dissipation terms feeding the medium temperature equation, and the
implicit wall-temperature solve.
"""

from __future__ import annotations

import numpy as np

from _imr_autodiff import primal_array
from _imr_materials import InstantaneousMaterial, NoStress, QuadraticKelvinVoigt
from _imr_stress import _elastic_integrand, _viscosity_and_tangent

__all__ = [
  "_GAM_TAIT",
  "_HUGONIOT_S",
  "_NOG",
  "_NSTATE_TAIT",
  "_apply_thermal_boundaries",
  "_dissipation",
  "_distributed_dissipation",
  "_instantaneous_dissipation",
  "_kv_of_T",
  "_mie_F",
  "_mie_gruneisen",
  "_mu_of_A",
  "_secant_root",
  "_wall_theta_bw",
  "_wall_theta_bw_full",
  "pvsat",
]


_GAM_TAIT = 3049.13e5

_NSTATE_TAIT = 7.15

_HUGONIOT_S = 1.65

_NOG = (_NSTATE_TAIT - 1.0) / 2.0


def pvsat(T):
  return 1.17e11 * np.exp(-5200.0 / T)


def _mu_of_A(A, s=_HUGONIOT_S, nog=_NOG):
  a = A * s**2 - nog
  b = -2.0 * A * s - 1.0
  d = b**2 - 4.0 * a * A
  return (-b - np.sqrt(d)) / (2.0 * a)


def _mie_F(mu, s=_HUGONIOT_S, nog=_NOG):
  w = 1.0 - s * mu
  return (
    (2 * nog + s - 1) / (s + 1) ** 3 * np.log(w / (mu + 1.0))
    + (nog + s) / (s * (s + 1) * w**2)
    - (2 * nog + s - 1) / ((s + 1) ** 2 * w)
  )


def _mie_gruneisen(P, Cstar, s, nog, reference):
  A = P / Cstar**2
  mu = _mu_of_A(A, s, nog)
  w = 1.0 - s * mu
  # transient negative discriminant on rejected LSODA trial steps only --
  # the accepted trajectory stays real (confirmed against real IMRv2, see
  # module docstring); harmless, same pattern as the xi[-1]=-1 case below.
  with np.errstate(invalid="ignore"):
    C = Cstar * np.sqrt(((1 + 2 * nog * mu) * w**2 + 2 * s * mu * (1 + nog * mu) * w) / w**4)
  hH = 1.0 / (1.0 + mu)
  hB = Cstar**2 * (_mie_F(mu, s, nog) - reference)
  return C, hB, hH


def _far_field_singular_index(xi) -> int:
  """Index of the node where the medium grid map 2 / (xi + 1) is singular.

  Raises unless there is exactly one and it is the last. The wall closure and
  every yT power assume that; suppressing the divide instead lets a moved or
  duplicated singularity produce inf at an interior node in silence. See #35.
  """
  values = np.asarray(xi, dtype=float)
  singular = np.flatnonzero(values + 1.0 == 0.0)
  if singular.size != 1 or singular[0] != values.size - 1:
    raise ValueError(
      f"medium grid singularity must be the far-field node alone: xi + 1 == 0 at {singular.tolist()} "
      f"of {values.size} nodes"
    )
  return int(singular[0])


def _instantaneous_dissipation(material, p, R, Rd, yT, yT3, iyT3):
  strain_rate = Rd / R * iyT3
  heating = np.zeros_like(yT)
  if material.elastic is not None:
    # yT and yT3 are +inf at the far-field node, so R*yT/reference_radius is
    # inf/inf there -- a nan that the next line overwrote with 1.0. Compute the
    # interior and set the wall value directly; unstretched is what inf/inf was
    # standing in for. #35.
    reference_radius = np.cbrt(np.maximum(R**3 * (yT3[:-1] - 1.0) + p["req"] ** 3, 1e-30))
    stretch = np.ones_like(yT)
    stretch[:-1] = R * yT[:-1] / reference_radius
    integrand = _elastic_integrand(material.elastic, stretch, p["P8"])
    stress_difference = 0.5 * integrand * stretch * (stretch**3 - 1.0)
    heating -= 2.0 * strain_rate * stress_difference
  if material.viscous is not None:
    shear_rate = 2.0 * np.sqrt(3.0) * abs(strain_rate) / p["t0"]
    viscosity, _ = _viscosity_and_tangent(material.viscous, shear_rate)
    heating += 12.0 * viscosity / p["viscosity_scale"] * strain_rate**2
  return p["Br"] * heating


def _dissipation(material, p, R, Rd, yT, yT2, yT3, iyT3, iyT4, iyT6):
  Ca, Re8, Br, ax = p["Ca"], p["Re8"], p["Br"], p["alphax"]
  Rst = p["req"] / R
  x2 = (yT3 - 1.0 + Rst**3) ** (2.0 / 3.0)
  ix2 = 1.0 / x2
  x4 = x2**2
  base = 12.0 * (Br / Re8) * (Rd / R) ** 2 * iyT6 + 2.0 * Br / Ca * iyT3 * (Rd / R) * (yT2 * ix2 - iyT4 * x4)
  if isinstance(material, InstantaneousMaterial):
    return _instantaneous_dissipation(material, p, R, Rd, yT, yT3, iyT3)
  if isinstance(material, NoStress):
    return np.zeros_like(yT)
  if isinstance(material, QuadraticKelvinVoigt):
    return base * (1.0 + ax * (x4 * iyT4 + 2.0 * yT2 * ix2 - 3.0))
  return base


def _distributed_dissipation(state, prepared, p, R, Rd, yT, iyT3):
  points = prepared.reference_radius.size
  stress_difference = state[:points] - state[points:]
  spatial_radius = R * yT
  reference_radius = np.cbrt(np.maximum(spatial_radius**3 - R**3 + 1.0, 1.0))
  if reference_radius.dtype == object or stress_difference.dtype == object:
    sampled_difference = np.empty_like(reference_radius)
    reference_values = primal_array(reference_radius)
    source_radius = prepared.reference_radius
    for index, (radius, radius_value) in enumerate(zip(reference_radius, reference_values, strict=True)):
      if radius_value <= source_radius[0]:
        sampled_difference[index] = stress_difference[0]
      elif radius_value >= source_radius[-1]:
        sampled_difference[index] = 0.0
      else:
        left = np.searchsorted(source_radius, radius_value) - 1
        fraction = (radius - source_radius[left]) / (source_radius[left + 1] - source_radius[left])
        sampled_difference[index] = stress_difference[left] + fraction * (
          stress_difference[left + 1] - stress_difference[left]
        )
  else:
    sampled_difference = np.interp(reference_radius, prepared.reference_radius, stress_difference, right=0.0)
  strain_rate = Rd / R * iyT3
  polymer_heating = -2.0 * strain_rate * sampled_difference
  solvent_heating = 12.0 * p["LAM"] / p["Re8"] * strain_rate**2
  return p["Br"] * (polymer_heating + solvent_heating)


def _kv_of_T(Tw, P, T8, Rvg_ratio, pressure_scale):
  theta_var = Rvg_ratio * (P / (pvsat(Tw * T8) / pressure_scale) - 1.0)
  return 1.0 / (1.0 + theta_var)


def _secant_root(function, guess, *, tol=1e-13, maxiter=100):
  p0 = float(guess)
  p1 = p0 * 1.0001
  p1 += 1e-4 if p1 >= 0.0 else -1e-4
  q0 = function(p0)
  q1 = function(p1)
  if abs(q1) < abs(q0):
    p0, p1, q0, q1 = p1, p0, q1, q0

  for _ in range(maxiter):
    if q1 == q0:
      raise RuntimeError("wall boundary secant solve encountered zero slope")
    if abs(q1) > abs(q0):
      ratio = q0 / q1
      root = (-ratio * p1 + p0) / (1.0 - ratio)
    else:
      ratio = q1 / q0
      root = (-ratio * p0 + p1) / (1.0 - ratio)
    if abs(root - p1) <= tol:
      return root
    p0, q0 = p1, q1
    p1 = root
    q1 = function(p1)
  raise RuntimeError(f"wall boundary secant solve failed to converge after {maxiter} iterations")


def _wall_theta_bw(guess, theta_tail, Tm_tail, alpha_g, grad_Tm, grad_Trans):

  def resid(theta_bw):
    Tw = (alpha_g - 1.0 + np.sqrt(1.0 + 2.0 * theta_bw * alpha_g)) / alpha_g
    lhs = grad_Tm[0] * Tw + np.sum(grad_Tm[1:] * Tm_tail)
    rhs = grad_Trans[0] * theta_bw + np.sum(grad_Trans[1:] * theta_tail)
    return lhs + rhs

  return _secant_root(resid, guess)


def _wall_theta_bw_full(
  guess,
  theta_tail,
  Tm_tail,
  kv_tail,
  kv_end_stale,
  P,
  alpha_v,
  alpha_g,
  T8,
  Rvg_ratio,
  Rva_diff,
  Rg_star,
  pressure_scale,
  grad_Tm,
  grad_Trans,
  grad_C,
):
  alpha_m = kv_end_stale * alpha_v + (1.0 - kv_end_stale) * alpha_g

  def resid(theta_bw):
    Tw = (alpha_m - 1.0 + np.sqrt(1.0 + 2.0 * theta_bw * alpha_m)) / alpha_m
    kvw = _kv_of_T(Tw, P, T8, Rvg_ratio, pressure_scale)
    lhs = grad_Tm[0] * Tw + np.sum(grad_Tm[1:] * Tm_tail)
    rhs = grad_Trans[0] * theta_bw + np.sum(grad_Trans[1:] * theta_tail)
    scalar = P / ((kvw * Rva_diff + Rg_star) * (Tw * (1.0 - kvw)))
    extra = scalar * (grad_C[0] * kvw + np.sum(grad_C[1:] * kv_tail))
    return lhs + rhs + extra

  failure = None
  with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
    try:
      return _secant_root(resid, guess)
    except RuntimeError as error:
      failure = error
      roots = []
      for fallback in (0.0, 0.5 * guess, 1.5 * guess, theta_tail[0]):
        try:
          roots.append(_secant_root(resid, fallback))
        except RuntimeError:
          pass
  if roots:
    return min(roots, key=lambda root: abs(root - guess))
  raise failure


def _apply_thermal_boundaries(theta, Tm, kv, P, p, medium, masstrans, wall_state):
  if medium is not None and masstrans:
    theta[-1] = _wall_theta_bw_full(
      wall_state.theta,
      theta[-2::-1],
      Tm[1:],
      kv[-2::-1],
      kv[-1],
      P,
      p["alpha_v"],
      p["alpha_g"],
      p["T8"],
      p["Rv_star"] / p["Rg_star"],
      p["Rv_star"] - p["Rg_star"],
      p["Rg_star"],
      p["P8"],
      medium.grad_Tm,
      medium.grad_Trans,
      medium.grad_C,
    )
    wall_state.theta = theta[-1]
  elif medium is not None:
    theta[-1] = _wall_theta_bw(wall_state.theta, theta[-2::-1], Tm[1:], p["alpha_g"], medium.grad_Tm, medium.grad_Trans)
    wall_state.theta = theta[-1]

  alpha_m = None
  if masstrans:
    alpha_m = kv * p["alpha_v"] + (1.0 - kv) * p["alpha_g"]
    temperature = (alpha_m - 1.0 + np.sqrt(1.0 + 2.0 * theta * alpha_m)) / alpha_m
    kv[-1] = _kv_of_T(temperature[-1], P, p["T8"], p["Rv_star"] / p["Rg_star"], p["P8"])
  else:
    alpha_g = p["alpha_g"]
    temperature = (alpha_g - 1.0 + np.sqrt(1.0 + 2.0 * theta * alpha_g)) / alpha_g
  if Tm is not None:
    Tm[0] = temperature[-1]
  return temperature, alpha_m
