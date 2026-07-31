"""Right-hand side of the bubble-dynamics system."""

from __future__ import annotations

import numpy as np

from ._arrays import at_set
from ._materials import _stress_state_count
from ._stress import _distributed_stress, _stress
from ._thermal import _apply_thermal_boundaries, _dissipation, _distributed_dissipation, _mie_gruneisen

__all__ = ["_nZ", "_pinf", "_rhs", "_rhs_args", "_sampled_pressure"]

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

def _nZ(material): return _stress_state_count(material)

def _rhs_args(problem, p, *, medium):
  """The positional arguments `_rhs` takes, for a prepared problem."""
  config = problem.config
  return (
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
  kv = None
  if bubtherm:
    P = y[2]
    Nt = ygrid.size
    theta = y[3 : 3 + Nt].copy()
    idx = 3 + Nt
    Tm = None
    if medtherm:
      Tm = y[idx : idx + mt.xi.size].copy()
      idx += mt.xi.size
    if masstrans:
      kv = y[idx : idx + Nt].copy()
      idx += Nt
    theta, Tm, kv, T, alpha_m = _apply_thermal_boundaries(theta, Tm, kv, P, p, mt, masstrans, xp=xp)
    Zstart = idx
  else:
    P = (p["Pb"] - Pv) * R ** (-3 * kappa) + Pv  # f_imr_fd.m:412
    Zstart = 2
  nz = _nZ(material)
  Z = y[Zstart : Zstart + nz] if nz else None
  if distributed_stress is None:
    S, Sdot, dZ, acceleration_coefficient = _stress(material, p, R, Rd, Z, instantaneous_material, radial != 1, xp=xp)
  else:
    S, Sdot, dZ, acceleration_coefficient = _distributed_stress(material, distributed_stress, p, R, Rd, Z, radial != 1, xp=xp)
  Pf8, Pf8dot = _pinf(tn, p, forcing, xp=xp)
  iWe = p["iWe"]
  thetadot = None
  kvdot = None
  if bubtherm:
    alpha_g, beta_g, chi = p["alpha_g"], p["beta_g"], p["chi"]
    if masstrans:
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
  elif radial in (3, 4, 5, 6):  # enthalpy forms: 3/5 Keller-Miksis, 4/6 Gilmore
    if radial in (3, 4):  # Tait
      Pb = P - iWe / R + p["tait_gamma"] + S
      hB = p["tait_sam"] / p["tait_no"] * ((Pb / p["tait_sam"]) ** p["tait_no"] - 1.0)
      hH = (p["tait_sam"] / Pb) ** (1.0 / p["tait_exponent"])
      Cs = p["Cstar"] if radial == 3 else xp.sqrt(p["tait_exponent"] * Pb * hH)
    else:  # Mie-Gruneisen. Upstream omits +S from Pb here; restoring it is what
      Pb = P - iWe / R + S
      C, hB, hH = _mie_gruneisen(Pb, p["Cstar"], p["hugoniot_slope"], p["nog"], p["mie_reference"], xp=xp)
      Cs = p["Cstar"] if radial == 5 else C
    num = (1 + Rd / Cs) * (hB - Pf8) - R / Cs * Pf8dot + R / Cs * hH * (Pdot + iWe * Rd / R**2 + Sdot) - 1.5 * (1 - Rd / (3 * Cs)) * Rd**2
    den = (1 - Rd / Cs) * R + acceleration_coefficient * hH / Cs
    Rdd = num / den
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
  out = at_set(at_set(xp.zeros_like(y), 0, Rd), 1, Rdd)
  cursor = 2
  if thetadot is not None:
    out = at_set(out, cursor, Pdot)
    cursor += 1
    out = at_set(out, slice(cursor, cursor + thetadot.size), thetadot)
    cursor += thetadot.size
  if Tmdot is not None:
    out = at_set(out, slice(cursor, cursor + Tmdot.size), Tmdot)
    cursor += Tmdot.size
  if kvdot is not None:
    out = at_set(out, slice(cursor, cursor + kvdot.size), kvdot)
    cursor += kvdot.size
  if dZ is not None: out = at_set(out, slice(cursor, cursor + dZ.size), dZ)
  return out
