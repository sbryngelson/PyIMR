"""Equation of state, thermal dissipation, and wall boundary closure."""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from ._arrays import at_set
from ._materials import InstantaneousMaterial, NoStress, QuadraticKelvinVoigt
from ._stress import _elastic_integrand, _viscosity_and_tangent

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
  "kirchhoff_temperature",
  "kirchhoff_theta",
  "mixture_kirchhoff",
  "_mie_F",
  "_mie_gruneisen",
  "_mu_of_A",
  "_T_of_kv",
  "_bracketed_root",
  "_traced_root",
  "_wall_theta_bw",
  "_wall_theta_bw_full",
  "pvsat",
]
_GAM_TAIT = 3049.13e5
_NSTATE_TAIT = 7.15
_HUGONIOT_S = 1.65
_NOG = (_NSTATE_TAIT - 1.0) / 2.0
_KV_EPS = 1e-13
_HALVINGS = 20
_POLISH = 3
_DOMAIN_FLOOR = 1e-12  # keeps sqrt/log arguments defined at unphysical trial states (#133)

def kirchhoff_theta(temperature, alpha, beta):
  """The Kirchhoff transform of a conductivity linear in temperature."""
  return 0.5 * alpha * (temperature**2 - 1.0) + beta * (temperature - 1.0)

def kirchhoff_temperature(theta, alpha, beta, *, xp=np):
  """Inverse of :func:`kirchhoff_theta`."""
  # implicit solvers evaluate this at trial states with theta < 0, where the root is of a
  # negative number. NaN there is fatal: the primal rejects such a step, but optimistix
  # differentiates the root find before that happens and hands the NaN to lineax (#133).
  return (-beta + xp.sqrt(xp.maximum((alpha + beta) ** 2 + 2.0 * alpha * theta, _DOMAIN_FLOOR))) / alpha

def mixture_kirchhoff(vapor_fraction, p, masstrans):
  """The `(alpha, beta)` a gas/vapour mixture presents to :func:`kirchhoff_theta`."""
  if not masstrans: return p["alpha_g"], p["beta_g"]
  alpha = vapor_fraction * p["alpha_v"] + (1.0 - vapor_fraction) * p["alpha_g"]
  beta = vapor_fraction * p["beta_v"] + (1.0 - vapor_fraction) * p["beta_g"]
  return alpha, beta

def pvsat(T, *, xp=np): return 1.17e11 * xp.exp(-5200.0 / T)

def _mu_of_A(A, s=_HUGONIOT_S, nog=_NOG, *, xp=np):
  return (2.0 * A * s + 1.0 - xp.sqrt(1.0 + 4.0 * A * (s + nog))) / (2.0 * (A * s**2 - nog))

def _mie_F(mu, s=_HUGONIOT_S, nog=_NOG, *, xp=np):
  w = 1.0 - s * mu
  return (2 * nog + s - 1) / (s + 1) ** 3 * xp.log(w / (mu + 1.0)) + (nog + s) / (s * (s + 1) * w**2) - (2 * nog + s - 1) / ((s + 1) ** 2 * w)

def _mie_gruneisen(P, Cstar, s, nog, reference, *, xp=np):
  A = P / Cstar**2
  mu = _mu_of_A(A, s, nog, xp=xp)
  C = Cstar * xp.sqrt((1.0 + (s + 2.0 * nog) * mu) / (1.0 - s * mu) ** 3)
  hH = 1.0 / (1.0 + mu)
  hB = Cstar**2 * (_mie_F(mu, s, nog, xp=xp) - reference)
  return C, hB, hH

def _far_field_singular_index(xi) -> int:
  values = np.asarray(xi, dtype=float)
  singular = np.flatnonzero(values + 1.0 == 0.0)
  if singular.size != 1 or singular[0] != values.size - 1:
    # exactly one, and last: the wall closure and every yT power assume it
    raise ValueError(f"medium grid singularity must be the far-field node alone: xi + 1 == 0 at {singular.tolist()} of {values.size} nodes")
  return int(singular[0])

def _instantaneous_dissipation(material, p, R, Rd, yT, yT3, iyT3, *, xp=np):
  strain_rate = Rd / R * iyT3
  heating = xp.zeros_like(yT)
  if material.elastic is not None:
    reference_radius = xp.cbrt(xp.maximum(R**3 * (yT3[:-1] - 1.0) + p["req"] ** 3, 1e-30))
    stretch = at_set(xp.ones_like(yT), slice(None, -1), R * yT[:-1] / reference_radius)
    integrand = _elastic_integrand(material.elastic, stretch, p["P8"], xp=xp)
    stress_difference = 0.5 * integrand * stretch * (stretch**3 - 1.0)
    heating -= 2.0 * strain_rate * stress_difference
  if material.viscous is not None:
    shear_rate = 2.0 * xp.sqrt(3.0) * abs(strain_rate) / p["t0"]
    viscosity, _ = _viscosity_and_tangent(material.viscous, shear_rate, xp=xp)
    heating += 12.0 * viscosity / p["viscosity_scale"] * strain_rate**2
  return p["Br"] * heating

def _dissipation(material, p, R, Rd, yT, yT2, yT3, iyT3, iyT4, iyT6, *, xp=np):
  """Medium heating for the closed-form materials."""
  Ca, Re8, Br, ax = p["Ca"], p["Re8"], p["Br"], p["alphax"]
  Rst = p["req"] / R
  inner = slice(0, -1)
  x2 = (yT3[inner] - 1.0 + Rst**3) ** (2.0 / 3.0)
  ix2 = 1.0 / x2
  x4 = x2**2
  base = at_set(
    xp.zeros_like(yT), inner,
    12.0 * (Br / Re8) * (Rd / R) ** 2 * iyT6[inner] + 2.0 * Br / Ca * iyT3[inner] * (Rd / R) * (yT2[inner] * ix2 - iyT4[inner] * x4),
  )
  if isinstance(material, InstantaneousMaterial): return _instantaneous_dissipation(material, p, R, Rd, yT, yT3, iyT3, xp=xp)
  if isinstance(material, NoStress): return xp.zeros_like(yT)
  if isinstance(material, QuadraticKelvinVoigt):
    stiffening = at_set(xp.ones_like(yT), inner, 1.0 + ax * (x4 * iyT4[inner] + 2.0 * yT2[inner] * ix2 - 3.0))
    return base * stiffening
  return base

def _distributed_dissipation(state, prepared, p, R, Rd, yT, iyT3, *, xp=np):
  points = prepared.reference_radius.size
  stress_difference = state[:points] - state[points:]
  spatial_radius = R * yT
  reference_radius = xp.cbrt(xp.maximum(spatial_radius**3 - R**3 + 1.0, 1.0))
  sampled_difference = xp.interp(reference_radius, prepared.reference_radius, stress_difference, right=0.0)
  strain_rate = Rd / R * iyT3
  polymer_heating = -2.0 * strain_rate * sampled_difference
  solvent_heating = 12.0 * p["LAM"] / p["Re8"] * strain_rate**2
  return p["Br"] * (polymer_heating + solvent_heating)

def _kv_of_T(Tw, P, T8, Rvg_ratio, pressure_scale, *, xp=np):
  theta_var = Rvg_ratio * (P / (pvsat(Tw * T8, xp=xp) / pressure_scale) - 1.0)
  return 1.0 / (1.0 + theta_var)

def _T_of_kv(kv, P, T8, Rvg_ratio, pressure_scale, *, xp=np):
  """Closed-form inverse of :func:`_kv_of_T`."""
  ps = P * kv * Rvg_ratio / (kv * Rvg_ratio + 1.0 - kv)
  # same trial-state problem as `kirchhoff_temperature`: P < 0 or kv outside [0, 1] drives the
  # partial pressure to zero or negative, and the log argument with it. Two floors, one so the
  # division stays finite and one so the log stays positive (#133).
  saturation = 1.17e11 / (pressure_scale * xp.maximum(ps, _DOMAIN_FLOOR))
  return 5200.0 / (T8 * xp.log(xp.maximum(saturation, 1.0 + _DOMAIN_FLOOR)))

def _value(x):
  return getattr(x, "value", x).real

def _traced_root(residual, bracket, xp, *, halvings=_HALVINGS, polish=_POLISH):
  """The bracketed solve again, for a namespace that cannot branch on a value.

  The tangent comes from the implicit function theorem via `lax.custom_root`, not from
  differentiating the iteration. For `F(r; p) = 0` it is `dr/dp = -(dF/dp) / (dF/dr)`,
  exact for any iteration count, so the primal only has to satisfy its own accuracy.

  That decoupling is the point. Differentiating the iteration made the tangent exact only
  because 20 halvings drove `residual(root)` to ~1e-6 and killed the term the finite
  difference contributed -- exact by margin, and it silently degraded if the iteration
  budget was cut (#161, #181). Newton polish now uses an AD slope, one `value_and_grad`
  rather than three residual evaluations per step.
  """
  import jax

  low0, high0 = bracket

  def solve(function, guess):
    low, high = low0 * xp.ones_like(guess), high0 * xp.ones_like(guess)
    below = function(low) >= 0.0
    for _ in range(halvings):
      middle = 0.5 * (low + high)
      left = (function(middle) >= 0.0) == below
      low = xp.where(left, middle, low)
      high = xp.where(left, high, middle)
    root = 0.5 * (low + high)
    for _ in range(polish):
      value, slope = jax.value_and_grad(function)(root)
      root = root - value / slope
    return root

  def tangent_solve(linearized, y):
    return y / linearized(xp.ones_like(y))

  return jax.lax.custom_root(residual, xp.asarray(0.5 * (low0 + high0)), solve, tangent_solve)

def _bracketed_root(residual, *, bracket=(_KV_EPS, 1.0 - _KV_EPS), xp=np):
  """Solve `residual(kv) = 0` on a bracket, then attach the exact tangent."""
  if xp is not np: return _traced_root(residual, bracket, xp)
  low, high = bracket
  try:
    root = float(brentq(lambda kv: _value(residual(kv)), low, high, xtol=1e-15))  # pyright: ignore[reportArgumentType]
  except ValueError as error:
    ends = (_value(residual(low)), _value(residual(high)))
    raise RuntimeError(f"no admissible wall vapour fraction in {bracket}: residual {ends[0]:+.3e} to {ends[1]:+.3e}") from error
  step = 1e-7 * root
  slope = (_value(residual(root + step)) - _value(residual(root - step))) / (2.0 * step)
  polished = root - residual(root) / slope
  return polished - residual(polished) / slope

def _wall_theta_bw(theta_tail, Tm_tail, alpha_g, beta_g, grad_Tm, grad_Trans, *, xp=np):
  """Solve the flux match at the wall exactly, rather than iterating on it (#57)."""
  b, c = grad_Tm[0], grad_Trans[0]
  span = (alpha_g + beta_g) ** 2
  k = -b * beta_g / alpha_g + xp.sum(grad_Tm[1:] * Tm_tail)
  k += xp.sum(grad_Trans[1:] * theta_tail) - c * span / (2.0 * alpha_g)
  s = (-b + xp.sqrt(b * b - 2.0 * alpha_g * c * k)) / c
  return (s * s - span) / (2.0 * alpha_g)

def _wall_theta_bw_full(
  theta_tail,
  Tm_tail,
  kv_tail,
  kv_end_stale,
  P,
  alpha_v,
  alpha_g,
  beta_v,
  beta_g,
  T8,
  Rvg_ratio,
  Rva_diff,
  Rg_star,
  pressure_scale,
  grad_Tm,
  grad_Trans,
  grad_C,
  *,
  xp=np,
):
  """Wall energy balance with equilibrium phase change, solved on `kv in (0, 1)`."""
  alpha_m = kv_end_stale * alpha_v + (1.0 - kv_end_stale) * alpha_g
  beta_m = kv_end_stale * beta_v + (1.0 - kv_end_stale) * beta_g

  # Hoisted: these three contractions do not depend on `kv`, and the root find evaluates
  # `resid` ~29 times per RHS call. Left inside, the Nt- and Mt-length dot products were
  # recomputed every iteration; everything that actually varies with `kv` is scalar.
  flux_fixed = xp.sum(grad_Tm[1:] * Tm_tail) + xp.sum(grad_Trans[1:] * theta_tail)
  carrier_fixed = xp.sum(grad_C[1:] * kv_tail)

  def resid(kv):
    Tw = _T_of_kv(kv, P, T8, Rvg_ratio, pressure_scale, xp=xp)
    theta_bw = kirchhoff_theta(Tw, alpha_m, beta_m)
    flux = grad_Tm[0] * Tw + grad_Trans[0] * theta_bw + flux_fixed
    denominator = (kv * Rva_diff + Rg_star) * Tw * (1.0 - kv)
    return denominator * flux + P * (grad_C[0] * kv + carrier_fixed)

  kv = _bracketed_root(resid, xp=xp)
  return kirchhoff_theta(_T_of_kv(kv, P, T8, Rvg_ratio, pressure_scale, xp=xp), alpha_m, beta_m)

def _apply_thermal_boundaries(theta, Tm, kv, P, p, medium, masstrans, *, xp=np):
  if medium is not None and masstrans:
    theta = at_set(theta, -1, _wall_theta_bw_full(
      theta[-2::-1],
      Tm[1:],
      kv[-2::-1],
      kv[-1],
      P,
      p["alpha_v"],
      p["alpha_g"],
      p["beta_v"],
      p["beta_g"],
      p["T8"],
      p["Rv_star"] / p["Rg_star"],
      p["Rv_star"] - p["Rg_star"],
      p["Rg_star"],
      p["P8"],
      medium.grad_Tm,
      medium.grad_Trans,
      medium.grad_C,
      xp=xp,
    ))
  elif medium is not None:
    theta = at_set(theta, -1, _wall_theta_bw(theta[-2::-1], Tm[1:], p["alpha_g"], p["beta_g"], medium.grad_Tm, medium.grad_Trans, xp=xp))
  alpha_m = None
  if masstrans:
    alpha_m = kv * p["alpha_v"] + (1.0 - kv) * p["alpha_g"]
    beta_m = kv * p["beta_v"] + (1.0 - kv) * p["beta_g"]
    temperature = kirchhoff_temperature(theta, alpha_m, beta_m, xp=xp)
    kv = at_set(kv, -1, _kv_of_T(temperature[-1], P, p["T8"], p["Rv_star"] / p["Rg_star"], p["P8"], xp=xp))
  else:
    temperature = kirchhoff_temperature(theta, p["alpha_g"], p["beta_g"], xp=xp)
  if Tm is not None: Tm = at_set(Tm, 0, temperature[-1])
  return theta, Tm, kv, temperature, alpha_m
