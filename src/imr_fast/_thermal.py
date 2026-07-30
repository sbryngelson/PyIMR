"""Equation of state, thermal dissipation, and wall boundary closure.

Owns the Tait and Mie-Gruneisen parameters, saturated vapour pressure, the
viscous dissipation terms feeding the medium temperature equation, and the
implicit wall-temperature solve.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

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
# Endpoint inset for the vapour-fraction bracket: `kv` is a mass fraction, so the
# bracket IS its physical range (0, 1), opened just enough that the residual is
# evaluated inside it rather than at its removable ends.
_KV_EPS = 1e-13
# Traced bisection budget. 20 halvings take a width-1 bracket to 1e-06, which is
# well inside Newton's basin for this residual; 3 quadratic polish steps then run
# out of double precision. Both are unrolled into the traced graph, so they are
# also a compile-time cost -- measured in PLAN.md W11.
_HALVINGS = 20
_POLISH = 3

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

def _kv_of_T(Tw, P, T8, Rvg_ratio, pressure_scale, *, xp=np):
  theta_var = Rvg_ratio * (P / (pvsat(Tw * T8, xp=xp) / pressure_scale) - 1.0)
  return 1.0 / (1.0 + theta_var)

def _T_of_kv(kv, P, T8, Rvg_ratio, pressure_scale, *, xp=np):
  """Closed-form inverse of :func:`_kv_of_T`.

  `_kv_of_T` is strictly increasing in `Tw` -- `pvsat` is -- so it inverts, and
  inverting it is what turns the wall solve's physical admissibility condition
  into a constant bracket. Setting `kv = 1/(1 + Rvg*(P/ps - 1))` and solving for
  the saturation pressure gives `ps = P*kv*Rvg / (kv*Rvg + 1 - kv)`, then
  `pvsat(Tw*T8) = pressure_scale*ps` inverts in one log.

  `kv -> 1` sends `ps -> P` and so `Tw -> Tsat(P)`; `kv -> 0` sends both to
  zero. Every `kv` in `(0, 1)` therefore maps to a `Tw` below saturation, which
  is precisely the admissible range: `kv` is a MASS FRACTION.
  """
  ps = P * kv * Rvg_ratio / (kv * Rvg_ratio + 1.0 - kv)
  return 5200.0 / (T8 * xp.log(1.17e11 / (pressure_scale * ps)))

def _value(x):
  # The residual is evaluated in whichever arithmetic the caller brought: plain
  # float on the value path, `Dual` on the tangent path, complex128 under
  # complex step. All three yield their real value this way.
  return getattr(x, "value", x).real

def _traced_root(residual, bracket, xp, *, halvings=_HALVINGS, polish=_POLISH):
  """The bracketed solve again, for a namespace that cannot branch on a value.

  Under tracing there is no `brentq`: it wants concrete floats, and its control
  flow is data-dependent besides. Bisection is the one bracketing method whose
  iteration count does NOT depend on the values -- it is a fixed number of
  halvings, so the traced program has a fixed shape -- and `xp.where` carries the
  interval update without a Python branch. That is the whole reason this reads as
  a plain unrolled loop and needs no `lax` primitive: `where` and `sign` are the
  only namespace operations involved, so `_thermal` stays free of a jax import.

  Bisection alone would need 52 halvings for a width-1 bracket, which is both
  slow and a large graph. It only has to reach Newton's basin: `halvings` gets
  the error to ~1e-6, then the polish steps converge quadratically because the
  slope is recomputed each time rather than shared. That is the opposite trade
  from `_bracketed_root`, and for the opposite reason -- there, Brent had already
  converged and the steps existed only to carry a tangent, so a shared slope with
  `1 - d**2` error was the cheaper way to get one.

  Differentiating this gives the right answer for the same reason: the halvings
  are comparisons and contribute no tangent, and the Newton steps supply the
  implicit-function derivative from a residual that is differentiated exactly.
  """
  low, high = bracket
  # A boolean rather than `xp.sign`, which returns 0 at an exact zero and would
  # then match neither side. No `_value` either: this path runs only for a traced
  # namespace, where the residual is already real.
  below = residual(low) >= 0.0
  for _ in range(halvings):
    middle = 0.5 * (low + high)
    left = (residual(middle) >= 0.0) == below
    low = xp.where(left, middle, low)
    high = xp.where(left, high, middle)
  root = 0.5 * (low + high)
  for _ in range(polish):
    step = 1e-7 * root
    slope = (residual(root + step) - residual(root - step)) / (2.0 * step)
    root = root - residual(root) / slope
  return root

def _bracketed_root(residual, *, bracket=(_KV_EPS, 1.0 - _KV_EPS), xp=np):
  """Solve `residual(kv) = 0` on a bracket, then attach the exact tangent.

  Brent on the bracket -- bisection, secant and inverse quadratic interpolation
  -- is the method Barajas & Johnsen prescribe for this closure, and bracketing
  is the whole point: it cannot leave the interval, so the root it returns is
  admissible by construction. See the module note on `_wall_theta_bw_full`.

  Brent needs floats, which would drop the tangent, so Newton steps in the
  caller's own arithmetic put it back. They do not move the value --
  `residual(root)` is already zero there -- but by the implicit function theorem
  the tangent becomes `-(dG/dp)/(dG/dkv)`, the derivative of the root with
  respect to every parameter.

  TWO steps, sharing one approximate slope. The slope comes from a central
  difference, whose cancellation error is `eps/h ~ 1e-9` -- and a single step
  passes that error straight through to the tangent, which is measurable:
  complex-step against `Dual` on the coupled mass-transfer RHS degrades from
  1e-13 to 8e-9. Writing the slope as `m = G'(1 + d)`, one step scales the
  tangent by `1/(1 + d)` and two by `u*(2 - u)` with `u = 1/(1 + d)`, which
  expands to `1 - d**2`. So the second step costs one residual evaluation and
  squares the error rather than merely shrinking it -- no step-size tuning and
  no wider stencil, both of which only move `d` around.
  """
  if xp is not np: return _traced_root(residual, bracket, xp)
  low, high = bracket
  # `brentq` checks the endpoint signs itself, so checking them here first would
  # pay for two of the ~12 residual evaluations a wall solve costs to say
  # something it already says. Naming the values is worth it only on the way out.
  #
  # Its stub returns `float | tuple[float, RootResults]` regardless of
  # `full_output`, so the scalar overload cannot be selected and pyright rejects
  # arithmetic on the result. The runtime type is a float here.
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

  `_wall_theta_bw_full` cannot be reduced this far -- with mass transfer the
  vapour fraction makes the residual transcendental -- but it is bracketed
  rather than iterated from a guess, for the reasons recorded there.
  """
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
  """Wall energy balance with equilibrium phase change, solved on `kv in (0, 1)`.

  Barajas & Johnsen close the wall with `rho_w C_w = pvsat(Tw)/(Rv Tw)` and
  solve the resulting algebraic condition "using an algorithm based on a
  combination of bisection, secant, and inverse quadratic interpolation" -- that
  is Brent's method, a BRACKETING solve. Two things follow, and this function
  was doing neither (#111).

  First, the unknown is taken to be the vapour mass fraction rather than
  `theta_bw`. `_T_of_kv` inverts the equilibrium condition in closed form, so
  `kv`'s physical range -- a mass fraction lies in `(0, 1)` -- becomes the
  bracket, the same constant for every state. Written in `Tw` the admissible
  interval is instead `(0, Tsat(P))`, which has to be recomputed per state.

  Second, the residual is multiplied through by its own denominator
  `D = (kv*Rva_diff + Rg_star)*Tw*(1 - kv)`. As shipped the `1/(1 - kv)` factor
  put a POLE at `kv = 1`, exactly at the edge of admissibility, and iterating
  from a guess walked through it: measured over a 25 us NHKV collapse, 26 of
  1480 wall solves returned mass fractions outside `(0, 1)` -- as far as +179
  and -603 -- all of them in the last 2% of the solve, at deepest collapse.
  `D > 0` strictly inside the bracket, so multiplying it out removes the pole
  while preserving every root, and the product extends continuously to both
  endpoints. Measured on the same 1480 solves: exactly one sign change each,
  finite throughout, roots in `[0.472, 0.804]`.

  So the root is unique on the bracket and no branch has to be chosen. That is
  what makes this closure a function of state alone -- it previously warm-started
  from the previous call's answer, which made the whole right-hand side depend on
  the integrator's step history and is why the jax backend could not host mass
  transfer. IMRv2 warm-starts a plain secant here and inherits the excursion;
  the pinned reference does too, so the deep-collapse trajectory moves.
  """
  alpha_m = kv_end_stale * alpha_v + (1.0 - kv_end_stale) * alpha_g
  beta_m = kv_end_stale * beta_v + (1.0 - kv_end_stale) * beta_g

  def resid(kv):
    Tw = _T_of_kv(kv, P, T8, Rvg_ratio, pressure_scale, xp=xp)
    theta_bw = kirchhoff_theta(Tw, alpha_m, beta_m)
    flux = grad_Tm[0] * Tw + xp.sum(grad_Tm[1:] * Tm_tail)
    flux = flux + grad_Trans[0] * theta_bw + xp.sum(grad_Trans[1:] * theta_tail)
    denominator = (kv * Rva_diff + Rg_star) * Tw * (1.0 - kv)
    return denominator * flux + P * (grad_C[0] * kv + xp.sum(grad_C[1:] * kv_tail))

  kv = _bracketed_root(resid, xp=xp)
  return kirchhoff_theta(_T_of_kv(kv, P, T8, Rvg_ratio, pressure_scale, xp=xp), alpha_m, beta_m)

def _apply_thermal_boundaries(theta, Tm, kv, P, p, medium, masstrans, *, xp=np):
  # Returns the fields as well as the temperature: jax arrays are immutable, so
  # the wall values cannot be written into the caller's arrays and have to come
  # back. numpy still mutates in place inside `at_set`, so the returned objects
  # are the same ones and nothing downstream can tell the difference.
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
