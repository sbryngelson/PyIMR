"""Right-hand side of the bubble-dynamics system."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from ._arrays import at_set
from ._config import ENTHALPY_FORMS, SECOND_ORDER_THETA, StateLayout
from ._materials import _stress_state_count
from ._stress import _distributed_stress, _stress
from ._thermal import _apply_thermal_boundaries, _dissipation, _distributed_dissipation, _mie_gruneisen

__all__ = ["RhsArgs", "_pinf", "_rhs", "_rhs_args", "_sampled_pressure"]

def _sampled_pressure(tn, forcing, *, xp=np):
  """The PCHIP forcing history, without a Python branch on the integration time."""
  knots = xp.asarray(forcing.knots)
  interval = xp.clip(xp.searchsorted(knots, tn, side="right") - 1, 0, knots.size - 2)
  offset = tn - knots[interval]
  coefficients = xp.asarray(forcing.coefficients)
  c0, c1, c2, c3 = (coefficients[row][interval] for row in range(4))
  pressure = ((c0 * offset + c1) * offset + c2) * offset + c3
  pressure_rate = (3.0 * c0 * offset + 2.0 * c1) * offset + c2
  inside = xp.where((tn >= knots[0]) & (tn <= knots[-1]), 1.0, 0.0)
  return pressure * inside, pressure_rate * inside

def _pinf(tn, p, forcing=None, *, xp=np):
  if forcing is not None: return _sampled_pressure(tn, forcing, xp=xp)
  wt, ee, om, tw, dt, mn = (p["wave_type"], p["ee"], p["om"], p["tw"], p["dt"], p["mn"])
  if wt == 0:  # constant offset impulse
    return ee, 0.0
  if wt == 1:  # Gaussian
    e = xp.exp(-((tn - dt) ** 2) / tw**2)
    return -ee * e, ee * (2 * (tn - dt) / tw**2) * e
  if wt == 2:  # histotripsy pulse
    inside = (tn >= dt - np.pi / om) & (tn <= dt + np.pi / om)
    # clamped: c<0 makes c**(mn-1) nan, and a nan selected against still poisons the gradient
    c = xp.maximum(0.5 + 0.5 * xp.cos(om * (tn - dt)), 1e-300)
    pressure = ee * c**mn
    rate = -ee * mn * c ** (mn - 1) * 0.5 * om * xp.sin(om * (tn - dt))
    return xp.where(inside, pressure, 0.0), xp.where(inside, rate, 0.0)
  if wt == 3:  # Heaviside step
    return -ee * xp.where(tn > tw, 0.0, 1.0), 0.0 * tn
  raise ValueError(f"wave_type={wt} not supported")

class RhsArgs(NamedTuple):
  """What `_rhs` takes after `(tn, y)`, in that order, so `*args` still splats.

  A plain tuple here meant callers reached in by index -- `args[8]` for the medium,
  `(merged, *args[1:8], rebuilt, *args[9:])` to substitute two of them. Reordering
  `_rhs`'s parameters would have left that arithmetic pointing at the wrong fields
  with nothing to catch it.
  """

  p: dict
  material: object
  radial: int
  bubtherm: int
  D1: object
  D2: object
  ygrid: object
  medtherm: int
  mt: object
  masstrans: int
  forcing: object
  instantaneous_material: object
  distributed_stress: object

def _rhs_args(problem, p, *, medium):
  """The positional arguments `_rhs` takes, for a prepared problem."""
  config = problem.config
  return RhsArgs(
    p, config.material, config.radial, config.bubtherm, problem.bubble_D1, problem.bubble_D2, problem.bubble_grid,
    config.medtherm, medium, config.masstrans, problem.forcing, problem.instantaneous_material,
    problem.distributed_stress,
  )

def _rhs(
  tn,
  y,
  p,
  material,
  radial,
  bubtherm=0,
  D1=None,
  D2=None,
  ygrid=None,
  medtherm=0,
  mt=None,
  masstrans=0,
  forcing=None,
  instantaneous_material=None,
  distributed_stress=None,
  *,
  xp=np,
):
  R = xp.maximum(y[0], 1e-8)
  Rd = y[1]
  Pv = p["Pv"]
  kappa = p["kappa"]
  # `bubtherm`, `medtherm` and `masstrans` each pair a flag with companion arguments that are
  # non-None exactly when it is set, and `_validate_config` enforces medtherm/masstrans =>
  # bubtherm. None of that is visible to a type checker, and these values flow between three
  # separate flag blocks, so hoist the bindings and state the invariant where each is used.
  kv = None
  theta = None
  Tm = None
  T = None
  # one layout, shared with `StateLayout`: this used to recompute the same offsets by
  # hand, and a disagreement would have read the wrong block rather than raising
  layout = StateLayout.of(bubtherm, ygrid.size if ygrid is not None else 0, medtherm,
                          mt.xi.size if mt is not None else 0, masstrans, _stress_state_count(material))
  if bubtherm:
    assert D1 is not None and D2 is not None and ygrid is not None
    P = y[layout.pressure]
    theta = y[layout.bubble_thermal].copy()
    if medtherm:
      assert mt is not None
      Tm = y[layout.medium_thermal].copy()
    if masstrans:
      kv = y[layout.vapor_fraction].copy()
    theta, Tm, kv, T, alpha_m = _apply_thermal_boundaries(theta, Tm, kv, P, p, mt, masstrans, xp=xp)
  else:
    P = (p["Pb"] - Pv) * R ** (-3 * kappa) + Pv  # f_imr_fd.m:412
  nz = layout.stress.stop - layout.stress.start
  Z = y[layout.stress] if nz else None
  if distributed_stress is None:
    S, Sdot, dZ, acceleration_coefficient = _stress(material, p, R, Rd, Z, instantaneous_material, radial != 1, xp=xp)
  else:
    S, Sdot, dZ, acceleration_coefficient = _distributed_stress(material, distributed_stress, p, R, Rd, Z, radial != 1, xp=xp)
  Pf8, Pf8dot = _pinf(tn, p, forcing, xp=xp)
  iWe = p["iWe"]
  thetadot = None
  kvdot = None
  if bubtherm:
    assert D1 is not None and D2 is not None and ygrid is not None and theta is not None and T is not None
    alpha_g, beta_g, chi = p["alpha_g"], p["beta_g"], p["chi"]
    if masstrans:
      assert kv is not None  # masstrans => the vapour field was sliced out above
      # T below uses the stale kv[-1]. IMRv2's own one-step lag, replicated deliberately.
      alpha_v, beta_v = p["alpha_v"], p["beta_v"]
      Rv_star, Rg_star = p["Rv_star"], p["Rg_star"]
      Rva_diff = Rv_star - Rg_star
      Fom = p["Fom"]
      dtheta = D1 @ theta
      ddtheta = D2 @ theta
      dkv = D1 @ kv
      ddkv = D2 @ kv
      Rmix = kv * Rv_star + (1.0 - kv) * Rg_star
      RDkv = (Rva_diff / Rmix) * dkv
      Pdot = (
        3.0
        / R
        * (chi * (kappa - 1.0) * dtheta[-1] / R - kappa * P * Rd + kappa * P * Fom * Rv_star * dkv[-1] / (T[-1] * R * Rmix[-1] * (1.0 - kv[-1])))
      )
      Uvel = (chi / R * (kappa - 1.0) * dtheta - ygrid * R * Pdot / 3.0) / (kappa * P) + Fom / R * RDkv
      Kstar_g = alpha_g * T + beta_g
      Kstar_v = alpha_v * T + beta_v
      Kstar = kv * Kstar_v + (1.0 - kv) * Kstar_g
      nonlinear_term = (chi * ddtheta / R**2 + Pdot) * (p["kapover"] * Kstar * T / P)
      advection_term = -dtheta * (Uvel - ygrid * Rd) / R
      mass_diffusion = (Fom / R**2) * (Rva_diff / Rmix) * dkv * dtheta
      thetadot = at_set(advection_term + nonlinear_term + mass_diffusion, -1, 0.0)
      nonlinear_diffusion = dkv * (dtheta / (Kstar * T) + RDkv)
      advection_term2 = (Uvel - Rd * ygrid) / R * dkv
      kvdot = at_set(Fom / R**2 * (ddkv - nonlinear_diffusion) - advection_term2, -1, 0.0)
    else:
      dtheta = D1 @ theta
      ddtheta = D2 @ theta
      Pdot = 3.0 / R * (chi * (kappa - 1.0) * dtheta[-1] / R - kappa * P * Rd)
      Uvel = (chi / R * (kappa - 1.0) * dtheta - ygrid * R * Pdot / 3.0) / (kappa * P)
      Kstar = alpha_g * T + beta_g
      diffusion = (chi * ddtheta / R**2 + Pdot) * (p["kapover"] * Kstar * T / P)
      advection = -dtheta * (Uvel - ygrid * Rd) / R
      thetadot = at_set(advection + diffusion, -1, 0.0)
  else:
    Pdot = -3 * kappa * (p["Pb"] - Pv) * R ** (-3 * kappa - 1) * Rd
  Tmdot = None
  if medtherm:
    assert mt is not None and Tm is not None  # medtherm => bubtherm, so both were bound above
    dTm = mt.D1 @ Tm
    ddTm = mt.D2 @ Tm
    xi, yT, yT2, yT3, iyT3, iyT4, iyT6 = (mt.xi, mt.yT, mt.yT2, mt.yT3, mt.iyT3, mt.iyT4, mt.iyT6)
    Lt, Foh = p["Lt"], p["Foh"]
    inner = slice(0, -1)
    med_advection = at_set(
      xp.zeros_like(yT), inner,
      (1 + xi[inner]) ** 2 / (Lt * R) * (Rd / yT2[inner] * (1 - yT3[inner]) / 2 + Foh / R * ((xi[inner] + 1) / (2 * Lt) - 1 / yT[inner])) * dTm[inner],
    )
    med_diffusion = Foh / R**2 * (xi + 1) ** 4 / Lt**2 * ddTm / 4
    if distributed_stress is None:
      taugradu = _dissipation(material, p, R, Rd, yT, yT2, yT3, iyT3, iyT4, iyT6, xp=xp)
    else:
      taugradu = _distributed_dissipation(Z, distributed_stress, p, R, Rd, yT, iyT3, xp=xp)
    Tmdot = at_set(at_set(med_advection + med_diffusion + taugradu, 0, 0.0), -1, 0.0)
  if radial == 1:  # Rayleigh-Plesset
    Rdd = (P - 1 - Pf8 - iWe / R + S - 1.5 * Rd**2) / R
  elif radial == 2:  # Keller-Miksis (pressure form)
    Cs = p["Cstar"]
    num = (1 + Rd / Cs) * (P - 1 - Pf8 - iWe / R + S) + R / Cs * (Pdot + iWe * Rd / R**2 + Sdot - Pf8dot) - 1.5 * (1 - Rd / (3 * Cs)) * Rd**2
    den = (1 - Rd / Cs) * R + acceleration_coefficient / Cs
    Rdd = num / den
  # One equation, reached nine ways. Prosperetti & Lezzi (1986) eq. 17 is a one-parameter
  # family in `lam`, first order in the wall Mach number,
  #
  #   [1 - (lam+1) Rd/c] R Rdd + (3/2)[1 - (3 lam + 1)/3 Rd/c] Rd^2
  #       = [1 + (1 - lam) Rd/c] hB + (R/c) hBdot,
  #
  # with lam = 0 the Keller form and lam = 1 the Herring form. Gilmore is not a member -- it
  # is Kirkwood-Bethe, lam = 0 with the LOCAL wall sound speed rather than `Cstar` -- so the
  # two axes below are `lam` and `local`, and the equation of state is the third.
  #
  # As published, eq. 17 prints `=` where this has `+` between the two left-hand terms, which
  # would leave the equation with two equals signs. The `+` is what reduces to the Keller
  # equation at lam = 0, and it is what this branch computed before the family was added.
  elif radial in ENTHALPY_FORMS:
    lam, local, eos, order = ENTHALPY_FORMS[radial]
    if eos == "tait":
      Pb = P - iWe / R + p["tait_gamma"] + S
      hB = p["tait_sam"] / p["tait_no"] * ((Pb / p["tait_sam"]) ** p["tait_no"] - 1.0)
      hH = (p["tait_sam"] / Pb) ** (1.0 / p["tait_exponent"])
      C = xp.sqrt(p["tait_exponent"] * Pb * hH)
    elif eos == "nasg":
      # Noble-Abel stiffened gas. The isentrope is (v - b)(p + p_inf)^(1/g) = const, so with
      # `bstar = b rho_inf` the specific volume ratio, the enthalpy `int v dp` and the sound
      # speed `-v^2 (dp/dv)_s` are all closed form. At `bstar = 0` each line below becomes the
      # Tait line above it, term for term.
      bstar = p["nasg_b"]
      Pb = P - iWe / R + p["nasg_gamma"] + S
      hH = bstar + (1.0 - bstar) * (p["nasg_sam"] / Pb) ** (1.0 / p["nasg_exponent"])
      hB = bstar * (Pb - p["nasg_sam"]) + (1.0 - bstar) * p["nasg_sam"] / p["nasg_no"] * (
        (Pb / p["nasg_sam"]) ** p["nasg_no"] - 1.0)
      C = xp.sqrt(p["nasg_exponent"] * Pb * hH**2 / (hH - bstar))
    else:  # Mie-Gruneisen. Upstream omits +S from Pb here; restoring it is what
      Pb = P - iWe / R + S
      C, hB, hH = _mie_gruneisen(Pb, p["Cstar"], p["hugoniot_slope"], p["nog"], p["mie_reference"], xp=xp)
    Cs = C if local else p["Cstar"]
    M = Rd / Cs
    enthalpy_rate = hH * (Pdot + iWe * Rd / R**2 + Sdot) - Pf8dot
    if order == 1:
      num = ((1 + (1.0 - lam) * M) * (hB - Pf8) + R / Cs * enthalpy_rate
             - 1.5 * (1 - (3.0 * lam + 1.0) / 3.0 * M) * Rd**2)
      den = (1 - (lam + 1.0) * M) * R + acceleration_coefficient * hH / Cs
      Rdd = num / den
    else:
      # Lezzi & Prosperetti (1987) eq. 8.7, at their own recommended (lambda, theta):
      #
      #   [1 - (lam+1)M + (14/5 + 2 lam + th) M^2] R Rdd
      #     + (3/2)[1 - (lam + 1/3)M + (16/15 + 4 lam/3 + th) M^2] Rd^2 + R^2 Rdd^2/c^2
      #   = [1 + (1-lam)M + th M^2] hB + [1 - (1+lam)M] (R/c) hBdot
      #
      # The `R^2 Rdd^2` term makes this QUADRATIC in Rdd -- the first equation here that is
      # not solved by a division -- and is why the second-order forms are known to be
      # delicate. Dropping every M^2 term returns eq. 8.7 to the first-order branch above,
      # term for term, which is the reduction `test_lezzi_prosperetti.py` pins.
      th = SECOND_ORDER_THETA
      quadratic = R**2 / Cs**2
      linear = (R * (1 - (lam + 1.0) * M + (2.8 + 2.0 * lam + th) * M**2)
                + (1 - (1.0 + lam) * M) * acceleration_coefficient * hH / Cs)
      constant = (1.5 * (1 - (lam + 1.0 / 3.0) * M + (16.0 / 15.0 + 4.0 * lam / 3.0 + th) * M**2) * Rd**2
                  - (1 + (1.0 - lam) * M + th * M**2) * (hB - Pf8)
                  - (1 - (1.0 + lam) * M) * R / Cs * enthalpy_rate)
      # The root that survives `c -> infinity`, where `quadratic -> 0` and the equation must
      # return `-constant/linear`. Taking `(-b + sqrt(b^2-4ac))/(2a)` directly loses that limit
      # to cancellation; the `constant/q` branch is the numerically stable statement of the
      # same root and is exact in the limit.
      discriminant = linear**2 - 4.0 * quadratic * constant
      # A negative discriminant means no real acceleration satisfies the equation -- the known
      # failure of second-order forms at high Mach. Clamping keeps the integrator on a finite
      # trajectory rather than poisoning every downstream gradient with a nan; the trajectory
      # is then wrong, and `max_radius_ratio` and the residual are what catch it.
      root = xp.sqrt(xp.maximum(discriminant, 0.0))
      q = -0.5 * (linear + xp.where(linear >= 0.0, 1.0, -1.0) * root)
      Rdd = constant / xp.where(xp.abs(q) < 1e-300, 1e-300, q)
  else:
    raise ValueError(f"radial={radial} not supported")
  if distributed_stress is None:
    out = [Rd, Rdd]
    if thetadot is not None:
      out.append(Pdot)
      out.extend(list(thetadot))
    if Tmdot is not None: out.extend(list(Tmdot))
    if kvdot is not None: out.extend(list(kvdot))
    if dZ is not None: out.extend(list(dZ))
    return out
  # written through the same layout the state was read with, rather than walking the
  # cursor a second time -- the read and the write have to agree, and now they cannot not
  out = at_set(at_set(xp.zeros_like(y), 0, Rd), 1, Rdd)
  if thetadot is not None:
    out = at_set(at_set(out, layout.pressure, Pdot), layout.bubble_thermal, thetadot)
  if Tmdot is not None: out = at_set(out, layout.medium_thermal, Tmdot)
  if kvdot is not None: out = at_set(out, layout.vapor_fraction, kvdot)
  if dZ is not None: out = at_set(out, layout.stress, dZ)
  return out
