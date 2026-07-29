"""Equation of state, thermal dissipation, and wall boundary closure.

Owns the Tait and Mie-Gruneisen parameters, saturated vapour pressure, the
viscous dissipation terms feeding the medium temperature equation, and the
implicit wall-temperature solve.
"""

from __future__ import annotations

import numpy as np

from ._autodiff import at_set, primal_array
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
  "_secant_root",
  "_wall_theta_bw",
  "_wall_theta_bw_full",
  "pvsat",
]
_GAM_TAIT = 3049.13e5
_NSTATE_TAIT = 7.15
_HUGONIOT_S = 1.65
_NOG = (_NSTATE_TAIT - 1.0) / 2.0

def kirchhoff_theta(temperature, alpha, beta):
  """The Kirchhoff transform of a conductivity linear in temperature.

  The bubble thermal PDE integrates `theta` rather than `T` so that a
  temperature-dependent conductivity enters linearly. The transform is defined
  by exactly one requirement -- that its derivative BE the conductivity the
  diffusion term uses:

      d(theta)/dT = K*(T) = alpha*T + beta
      theta(T)    = alpha*(T**2 - 1)/2 + beta*(T - 1)

  Every other form in this package follows from that line, and none of them may
  re-derive it. They used to: five sites each inlined their own algebra under
  the tacit assumption `beta = 1 - alpha`, which holds only when the
  normalisation `K8` is the gas conductivity at `T8`. `K8` is in fact the
  gas/vapour average, so the assumption was wrong by 0.24% for air and
  arbitrarily wrong for any other gas -- and since `K8` cancels from
  `chi * K*(T)`, it made trajectories depend on the vapour conductivity even
  with no vapour present. See #75.
  """
  return 0.5 * alpha * (temperature**2 - 1.0) + beta * (temperature - 1.0)

def kirchhoff_temperature(theta, alpha, beta, *, xp=np):
  """Inverse of :func:`kirchhoff_theta`.

  Completing the square on `alpha*T**2/2 + beta*T = alpha/2 + beta + theta`
  gives `(alpha*T + beta)**2 = (alpha + beta)**2 + 2*alpha*theta`, so with
  `s = alpha*T + beta` -- the conductivity itself, and non-negative -- both
  directions are one line:

      T     = (s - beta) / alpha
      theta = (s**2 - (alpha + beta)**2) / (2*alpha)
  """
  return (-beta + xp.sqrt((alpha + beta) ** 2 + 2.0 * alpha * theta)) / alpha

def mixture_kirchhoff(vapor_fraction, p, masstrans):
  """The `(alpha, beta)` a gas/vapour mixture presents to :func:`kirchhoff_theta`.

  Mass fraction weighted, or the dry-gas pair when there is no vapour to mix.
  Shared for the same reason the transform itself is: #75 was five sites each
  inlining their own Kirchhoff algebra, and two of them had grown this pair back
  independently -- the primal initial state and its dual, which must agree
  exactly or the tangents differentiate a different conductivity than the
  forward solve integrates.
  """
  if not masstrans: return p["alpha_g"], p["beta_g"]
  alpha = vapor_fraction * p["alpha_v"] + (1.0 - vapor_fraction) * p["alpha_g"]
  beta = vapor_fraction * p["beta_v"] + (1.0 - vapor_fraction) * p["beta_g"]
  return alpha, beta

def pvsat(T, *, xp=np): return 1.17e11 * xp.exp(-5200.0 / T)

def _mu_of_A(A, s=_HUGONIOT_S, nog=_NOG, *, xp=np):
  # The discriminant of a*mu**2 + b*mu + A collapses: b**2 - 4*a*A is
  # 4*A**2*s**2 + 4*A*s + 1 - 4*A**2*s**2 + 4*nog*A, whose quartic terms cancel
  # exactly. Writing it out gives the Hugoniot's domain in one line -- a real
  # density root needs A > -1/(4*(s + nog)) = -0.0529 -- and avoids the
  # cancellation the subtracted form suffers. See #35.
  return (2.0 * A * s + 1.0 - xp.sqrt(1.0 + 4.0 * A * (s + nog))) / (2.0 * (A * s**2 - nog))

def _mie_F(mu, s=_HUGONIOT_S, nog=_NOG, *, xp=np):
  w = 1.0 - s * mu
  return (2 * nog + s - 1) / (s + 1) ** 3 * xp.log(w / (mu + 1.0)) + (nog + s) / (s * (s + 1) * w**2) - (2 * nog + s - 1) / ((s + 1) ** 2 * w)

def _mie_gruneisen(P, Cstar, s, nog, reference, *, xp=np):
  A = P / Cstar**2
  mu = _mu_of_A(A, s, nog, xp=xp)
  # The sound-speed radicand reduces to (1 + (s + 2*nog)*mu) / (1 - s*mu)**3,
  # which vanishes at mu = -1/(s + 2*nog) -- the same mu `_mu_of_A` returns at
  # its own discriminant's root. The two boundaries coincide exactly, so a
  # negative argument here is unreachable: mu is already nan by then, and
  # sqrt(nan) is quiet. The suppression this used to carry never fired (#35).
  C = Cstar * xp.sqrt((1.0 + (s + 2.0 * nog) * mu) / (1.0 - s * mu) ** 3)
  hH = 1.0 / (1.0 + mu)
  hB = Cstar**2 * (_mie_F(mu, s, nog, xp=xp) - reference)
  return C, hB, hH

def _far_field_singular_index(xi) -> int:
  # Index of the node where the medium grid map 2 / (xi + 1) is singular.
  #
  # Raises unless there is exactly one and it is the last. The wall closure and every yT power assume that;
  # suppressing the divide instead lets a moved or duplicated singularity produce inf at an interior node in silence.
  # See #35.
  values = np.asarray(xi, dtype=float)
  singular = np.flatnonzero(values + 1.0 == 0.0)
  if singular.size != 1 or singular[0] != values.size - 1:
    raise ValueError(f"medium grid singularity must be the far-field node alone: xi + 1 == 0 at {singular.tolist()} of {values.size} nodes")
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

def _dissipation(material, p, R, Rd, yT, yT2, yT3, iyT3, iyT4, iyT6, *, xp=np):
  """Medium heating for the closed-form materials.

  The `12/Re8` below is the TOTAL quasi-steady stress power, not a solvent term
  -- the resemblance to `_distributed_dissipation`'s `12*LAM/Re8` is what makes
  the missing `LAM` look like a bug (#47). With D = diag(-2e, e, e) for
  spherically symmetric incompressible flow, a quasi-steady polymer
  `tau_p = 2*mu_p*D` contributes `12*mu_p*e^2`, so solvent plus polymer is
  `12*(mu_s + mu_p)*e^2 = 12/Re8*e^2` and the split collapses.

  That makes this exact for purely viscous and Kelvin-Voigt materials, which
  have no polymer at all, and a De << 1 approximation for the viscoelastic ones:
  it carries no relaxation memory. `_distributed_dissipation` keeps the two
  terms apart and is the general form. Measured, the gap between them falls like
  O(De) -- see the issue for the table -- and is ~5% by De = 0.02.
  """
  Ca, Re8, Br, ax = p["Ca"], p["Re8"], p["Br"], p["alphax"]
  Rst = p["req"] / R
  # Everything here is interior-only. yT3 is +inf at the far-field node, so x2
  # is inf there and its reciprocal 0; the base expression then multiplies the
  # two and yields nan. On the Dual path the same inf reaches Dual.__pow__ and
  # produces a nan tangent as well.
  inner = slice(0, -1)
  x2 = (yT3[inner] - 1.0 + Rst**3) ** (2.0 / 3.0)
  ix2 = 1.0 / x2
  x4 = x2**2
  # Interior only: yT2 is +inf at the far-field node and iyT3 is exactly 0
  # there, so the second term is 0 * inf, a nan. It was invisible because the
  # caller in _imr_rhs suppressed invalid around the whole block -- a blanket
  # suppression hiding a nan in a function it does not own. Tmdot[-1] is
  # overwritten with 0.0, which is the value set here. #35.
  base = at_set(
    xp.zeros_like(yT), inner,
    12.0 * (Br / Re8) * (Rd / R) ** 2 * iyT6[inner] + 2.0 * Br / Ca * iyT3[inner] * (Rd / R) * (yT2[inner] * ix2 - iyT4[inner] * x4),
  )
  if isinstance(material, InstantaneousMaterial): return _instantaneous_dissipation(material, p, R, Rd, yT, yT3, iyT3)
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
        sampled_difference[index] = stress_difference[left] + fraction * (stress_difference[left + 1] - stress_difference[left])
  else:
    sampled_difference = xp.interp(reference_radius, prepared.reference_radius, stress_difference, right=0.0)
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
  if abs(q1) < abs(q0): p0, p1, q0, q1 = p1, p0, q1, q0
  for _ in range(maxiter):
    if q1 == q0: raise RuntimeError("wall boundary secant solve encountered zero slope")
    if abs(q1) > abs(q0):
      ratio = q0 / q1
      root = (-ratio * p1 + p0) / (1.0 - ratio)
    else:
      ratio = q1 / q0
      root = (-ratio * p0 + p1) / (1.0 - ratio)
    if abs(root - p1) <= tol: return root
    p0, q0 = p1, q1
    p1 = root
    q1 = function(p1)
  raise RuntimeError(f"wall boundary secant solve failed to converge after {maxiter} iterations")

def _wall_theta_bw(guess, theta_tail, Tm_tail, alpha_g, beta_g, grad_Tm, grad_Trans, *, xp=np):
  """Solve the flux match at the wall exactly, rather than iterating on it (#57).

  Substituting Tw = (s - beta_g) / alpha_g and theta_bw = (s*s - (alpha_g +
  beta_g)**2) / (2*alpha_g), where s = alpha_g*Tw + beta_g >= 0 is the
  conductivity itself, turns the residual into a quadratic
    c*s**2 + 2*b*s + 2*alpha_g*k = 0,   b = grad_Tm[0], c = grad_Trans[0],
  whose roots multiply to 2*alpha_g*k/c. That product is negative here (c > 0 and
  k < 0 for a one-sided wall stencil), so the roots straddle zero, the physical
  one is unique and the discriminant b*b - c*(2*alpha_g*k) cannot be negative.

  The secant this replaces would fail outright on ~1 call in 4000 -- enough to
  abort an integration -- for ordinary retardation ratios, and carried only a
  single-step tangent on the Dual path where this is exact.

  `_wall_theta_bw_full` keeps its fallback ladder: with mass transfer the vapour
  fraction makes the residual transcendental in Tw, so no closed form exists.
  """
  b, c = grad_Tm[0], grad_Trans[0]
  span = (alpha_g + beta_g) ** 2
  k = -b * beta_g / alpha_g + xp.sum(grad_Tm[1:] * Tm_tail)
  k += xp.sum(grad_Trans[1:] * theta_tail) - c * span / (2.0 * alpha_g)
  s = (-b + xp.sqrt(b * b - 2.0 * alpha_g * c * k)) / c
  return (s * s - span) / (2.0 * alpha_g)

def _wall_theta_bw_full(
  guess,
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
):
  alpha_m = kv_end_stale * alpha_v + (1.0 - kv_end_stale) * alpha_g
  beta_m = kv_end_stale * beta_v + (1.0 - kv_end_stale) * beta_g

  def resid(theta_bw):
    # A secant iterate can leave the Kirchhoff transform's range, where the
    # inverse has no real root: measured once per 30 us solve, out to
    # theta = -4.3e+04 against a guess of 0.50. The nan is what tells
    # `_secant_root` to back off, so it is suppressed here and only here. The
    # ladder below and the solver's own arithmetic raise nothing -- measured,
    # not assumed -- and are left honest. See #35.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
      Tw = kirchhoff_temperature(theta_bw, alpha_m, beta_m)
      kvw = _kv_of_T(Tw, P, T8, Rvg_ratio, pressure_scale)
      lhs = grad_Tm[0] * Tw + np.sum(grad_Tm[1:] * Tm_tail)
      rhs = grad_Trans[0] * theta_bw + np.sum(grad_Trans[1:] * theta_tail)
      scalar = P / ((kvw * Rva_diff + Rg_star) * (Tw * (1.0 - kvw)))
      extra = scalar * (grad_C[0] * kvw + np.sum(grad_C[1:] * kv_tail))
    return lhs + rhs + extra

  failure = None
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
  if roots: return min(roots, key=lambda root: abs(root - guess))
  raise failure

def _apply_thermal_boundaries(theta, Tm, kv, P, p, medium, masstrans, wall_state, *, xp=np):
  # Returns the fields as well as the temperature: jax arrays are immutable, so
  # the wall values cannot be written into the caller's arrays and have to come
  # back. numpy still mutates in place inside `at_set`, so the returned objects
  # are the same ones and nothing downstream can tell the difference.
  if medium is not None and masstrans:
    theta = at_set(theta, -1, _wall_theta_bw_full(
      wall_state.theta,
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
    ))
    wall_state.theta = theta[-1]
  elif medium is not None:
    theta = at_set(theta, -1, _wall_theta_bw(wall_state.theta, theta[-2::-1], Tm[1:], p["alpha_g"], p["beta_g"], medium.grad_Tm, medium.grad_Trans, xp=xp))
    wall_state.theta = theta[-1]
  alpha_m = None
  if masstrans:
    alpha_m = kv * p["alpha_v"] + (1.0 - kv) * p["alpha_g"]
    beta_m = kv * p["beta_v"] + (1.0 - kv) * p["beta_g"]
    temperature = kirchhoff_temperature(theta, alpha_m, beta_m, xp=xp)
    kv = at_set(kv, -1, _kv_of_T(temperature[-1], P, p["T8"], p["Rv_star"] / p["Rg_star"], p["P8"]))
  else:
    temperature = kirchhoff_temperature(theta, p["alpha_g"], p["beta_g"], xp=xp)
  if Tm is not None: Tm = at_set(Tm, 0, temperature[-1])
  return theta, Tm, kv, temperature, alpha_m
