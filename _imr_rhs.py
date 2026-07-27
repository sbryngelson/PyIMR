"""Right-hand side of the bubble-dynamics system.

Evaluates the radial equation, the gas and medium thermal PDEs, vapour
transport and the constitutive state for one point in time. Pure evaluation:
it reads a prepared problem but never builds one.
"""

from __future__ import annotations

import numpy as np

from _imr_autodiff import primal, primal_array
from _imr_materials import _stress_state_count
from _imr_stress import _distributed_stress, _stress
from _imr_thermal import _apply_thermal_boundaries, _dissipation, _distributed_dissipation, _mie_gruneisen

__all__ = ["_nZ", "_pinf", "_radius_floor_event", "_rhs", "_sampled_pressure"]


def _sampled_pressure(tn, forcing):
  time_value = primal(tn)
  knot_values = primal_array(forcing.knots)
  if time_value < knot_values[0] or time_value > knot_values[-1]:
    return 0.0, 0.0
  interval = np.searchsorted(knot_values, time_value, side="right") - 1
  interval = min(interval, knot_values.size - 2)
  offset = tn - forcing.knots[interval]
  c0, c1, c2, c3 = forcing.coefficients[:, interval]
  pressure = ((c0 * offset + c1) * offset + c2) * offset + c3
  pressure_rate = (3.0 * c0 * offset + 2.0 * c1) * offset + c2
  return pressure, pressure_rate


def _pinf(tn, p, forcing=None):
  if forcing is not None:
    return _sampled_pressure(tn, forcing)
  wt, ee, om, tw, dt, mn = (p["wave_type"], p["ee"], p["om"], p["tw"], p["dt"], p["mn"])
  if ee == 0.0:
    return 0.0, 0.0
  if wt == 0:  # constant offset impulse
    return ee, 0.0
  if wt == 1:  # Gaussian
    e = np.exp(-((tn - dt) ** 2) / tw**2)
    return -ee * e, ee * (2 * (tn - dt) / tw**2) * e
  if wt == 2:  # histotripsy pulse
    if tn < dt - np.pi / om or tn > dt + np.pi / om:
      return 0.0, 0.0
    c = 0.5 + 0.5 * np.cos(om * (tn - dt))
    return (ee * c**mn, -ee * mn * c ** (mn - 1) * 0.5 * om * np.sin(om * (tn - dt)))
  if wt == 3:  # Heaviside step
    return (-ee * (1.0 - (1.0 if tn > tw else 0.0)), 0.0)
  raise ValueError(f"wave_type={wt} not supported")


def _nZ(material):
  return _stress_state_count(material)


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
  wall_state=None,
  forcing=None,
  instantaneous_material=None,
  distributed_stress=None,
):
  R = max(y[0], 1e-8)
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
    T, alpha_m = _apply_thermal_boundaries(theta, Tm, kv, P, p, mt, masstrans, wall_state)
    Zstart = idx
  else:
    P = (p["Pb"] - Pv) * R ** (-3 * kappa) + Pv  # f_imr_fd.m:412
    Zstart = 2

  nz = _nZ(material)
  Z = y[Zstart : Zstart + nz] if nz else None
  if distributed_stress is None:
    S, Sdot, dZ, acceleration_coefficient = _stress(material, p, R, Rd, Z, instantaneous_material, radial != 1)
  else:
    S, Sdot, dZ, acceleration_coefficient = _distributed_stress(material, distributed_stress, p, R, Rd, Z, radial != 1)
  Pf8, Pf8dot = _pinf(tn, p, forcing)
  iWe = p["iWe"]

  thetadot = None
  kvdot = None
  if bubtherm:
    alpha_g, beta_g, chi = p["alpha_g"], p["beta_g"], p["chi"]
    if masstrans:
      # f_imr_fd.m, "if bubtherm && masstrans" branch. T is computed with
      # the STALE (pre-update) kv[-1] -- matches source order exactly
      # (T computed once, THEN kv[-1] is freshly overwritten below); this
      # one-step lag is IMRv2's own behavior, not reconciled/"fixed" here.
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
        * (
          chi * (kappa - 1.0) * dtheta[-1] / R
          - kappa * P * Rd
          + kappa * P * Fom * Rv_star * dkv[-1] / (T[-1] * R * Rmix[-1] * (1.0 - kv[-1]))
        )
      )
      Uvel = (chi / R * (kappa - 1.0) * dtheta - ygrid * R * Pdot / 3.0) / (kappa * P) + Fom / R * RDkv
      Kstar_g = alpha_g * T + beta_g
      Kstar_v = alpha_v * T + beta_v
      Kstar = kv * Kstar_v + (1.0 - kv) * Kstar_g
      nonlinear_term = (chi * ddtheta / R**2 + Pdot) * (p["kapover"] * Kstar * T / P)
      advection_term = -dtheta * (Uvel - ygrid * Rd) / R
      mass_diffusion = (Fom / R**2) * (Rva_diff / Rmix) * dkv * dtheta
      thetadot = advection_term + nonlinear_term + mass_diffusion
      thetadot[-1] = 0.0
      nonlinear_diffusion = dkv * (dtheta / (np.sqrt(1.0 + 2.0 * alpha_m * theta) * T) + RDkv)
      advection_term2 = (Uvel - Rd * ygrid) / R * dkv
      kvdot = Fom / R**2 * (ddkv - nonlinear_diffusion) - advection_term2
      kvdot[-1] = 0.0
    else:
      # f_imr_fd.m, "elseif bubtherm" branch, kv0=0 (dry gas) simplification.
      # Identical whether medtherm is on or off -- only theta[-1]'s VALUE
      # differs (wall-BC solve above vs. frozen at its initial value).
      dtheta = D1 @ theta
      ddtheta = D2 @ theta
      Pdot = 3.0 / R * (chi * (kappa - 1.0) * dtheta[-1] / R - kappa * P * Rd)
      Uvel = (chi / R * (kappa - 1.0) * dtheta - ygrid * R * Pdot / 3.0) / (kappa * P)
      Kstar = alpha_g * T + beta_g
      diffusion = (chi * ddtheta / R**2 + Pdot) * (p["kapover"] * Kstar * T / P)
      advection = -dtheta * (Uvel - ygrid * Rd) / R
      thetadot = advection + diffusion
      thetadot[-1] = 0.0
  else:
    Pdot = -3 * kappa * (p["Pb"] - Pv) * R ** (-3 * kappa - 1) * Rd

  Tmdot = None
  if medtherm:
    # f_imr_fd.m "surrounding temperature" block. xi[-1]=-1 exactly (the
    # far-field point) makes yT[-1]=inf -- an algebraic singularity
    # inherent to IMRv2's own xi=1+(j-1)*deltaYm formula (confirmed:
    # xi_last = 1+Nm*(-2/Nm) = -1 identically, not introduced by this
    # port), producing a transient inf*0=nan in the last entry of
    # med_advection/taugradu. Harmless: Tmdot[-1]=0.0 unconditionally
    # overwrites it and that state slot's value never depends on it
    # (same frozen-slot pattern as theta[-1] without medtherm).
    dTm = mt.D1 @ Tm
    ddTm = mt.D2 @ Tm
    xi, yT, yT2, yT3, iyT3, iyT4, iyT6 = (mt.xi, mt.yT, mt.yT2, mt.yT3, mt.iyT3, mt.iyT4, mt.iyT6)
    Lt, Foh = p["Lt"], p["Foh"]
    # Interior only: at the far-field node yT2 and yT3 are +inf, so
    # Rd/yT2 * (1 - yT3) is 0 * -inf, a nan that Tmdot[-1] = 0.0 below
    # overwrote. The wall entry is set to zero here instead, which is the value
    # that overwrite produced. #35.
    inner = slice(0, -1)
    med_advection = np.zeros_like(yT)
    med_advection[inner] = (
      (1 + xi[inner]) ** 2
      / (Lt * R)
      * (Rd / yT2[inner] * (1 - yT3[inner]) / 2 + Foh / R * ((xi[inner] + 1) / (2 * Lt) - 1 / yT[inner]))
      * dTm[inner]
    )
    med_diffusion = Foh / R**2 * (xi + 1) ** 4 / Lt**2 * ddTm / 4
    if distributed_stress is None:
      taugradu = _dissipation(material, p, R, Rd, yT, yT2, yT3, iyT3, iyT4, iyT6)
    else:
      taugradu = _distributed_dissipation(Z, distributed_stress, p, R, Rd, yT, iyT3)
    Tmdot = med_advection + med_diffusion + taugradu
    Tmdot[0] = 0.0
    Tmdot[-1] = 0.0

  if radial == 1:  # Rayleigh-Plesset
    Rdd = (P - 1 - Pf8 - iWe / R + S - 1.5 * Rd**2) / R
  elif radial == 2:  # Keller-Miksis (pressure form)
    Cs = p["Cstar"]
    num = (
      (1 + Rd / Cs) * (P - 1 - Pf8 - iWe / R + S)
      + R / Cs * (Pdot + iWe * Rd / R**2 + Sdot - Pf8dot)
      - 1.5 * (1 - Rd / (3 * Cs)) * Rd**2
    )
    den = (1 - Rd / Cs) * R + acceleration_coefficient / Cs
    Rdd = num / den
  elif radial == 3:  # Keller-Miksis, enthalpy, Tait EoS
    Cs = p["Cstar"]
    Pb = P - iWe / R + p["tait_gamma"] + S
    hB = p["tait_sam"] / p["tait_no"] * ((Pb / p["tait_sam"]) ** p["tait_no"] - 1.0)
    hH = (p["tait_sam"] / Pb) ** (1.0 / p["tait_exponent"])
    num = (
      (1 + Rd / Cs) * (hB - Pf8)
      - R / Cs * Pf8dot
      + R / Cs * hH * (Pdot + iWe * Rd / R**2 + Sdot)
      - 1.5 * (1 - Rd / (3 * Cs)) * Rd**2
    )
    den = (1 - Rd / Cs) * R + acceleration_coefficient * hH / Cs
    Rdd = num / den
  elif radial == 4:  # Gilmore, Tait EoS
    Pb = P - iWe / R + p["tait_gamma"] + S
    rho = (Pb / p["tait_sam"]) ** (1.0 / p["tait_exponent"])
    Cs = np.sqrt(p["tait_exponent"] * Pb / rho)
    hB = p["tait_sam"] / p["tait_no"] * ((Pb / p["tait_sam"]) ** p["tait_no"] - 1.0)
    hH = (p["tait_sam"] / Pb) ** (1.0 / p["tait_exponent"])
    num = (
      (1 + Rd / Cs) * (hB - Pf8)
      - R / Cs * Pf8dot
      + R / Cs * hH * (Pdot + iWe * Rd / R**2 + Sdot)
      - 1.5 * (1 - Rd / (3 * Cs)) * Rd**2
    )
    den = (1 - Rd / Cs) * R + acceleration_coefficient * hH / Cs
    Rdd = num / den
  elif radial == 5:  # Keller-Miksis, enthalpy, Mie-Gruneisen EoS
    # Pb matches radial=3/4. Upstream omits +S here; restoring it is what
    # brings this branch back into agreement with the Tait forms (PLAN W9).
    Cs = p["Cstar"]
    Pb = P - iWe / R + S
    _, hB, hH = _mie_gruneisen(Pb, Cs, p["hugoniot_slope"], p["nog"], p["mie_reference"])
    num = (
      (1 + Rd / Cs) * (hB - Pf8)
      - R / Cs * Pf8dot
      + R / Cs * hH * (Pdot + iWe * Rd / R**2 + Sdot)
      - 1.5 * (1 - Rd / (3 * Cs)) * Rd**2
    )
    den = (1 - Rd / Cs) * R + acceleration_coefficient * hH / Cs
    Rdd = num / den
  elif radial == 6:  # Gilmore, Mie-Gruneisen EoS
    Pb = P - iWe / R + S
    Cs, hB, hH = _mie_gruneisen(Pb, p["Cstar"], p["hugoniot_slope"], p["nog"], p["mie_reference"])
    num = (
      (1 + Rd / Cs) * (hB - Pf8)
      - R / Cs * Pf8dot
      + R / Cs * hH * (Pdot + iWe * Rd / R**2 + Sdot)
      - 1.5 * (1 - Rd / (3 * Cs)) * Rd**2
    )
    den = (1 - Rd / Cs) * R + acceleration_coefficient * hH / Cs
    Rdd = num / den
  else:
    raise ValueError(f"radial={radial} not supported")

  if distributed_stress is None:
    out = [Rd, Rdd]
    if bubtherm:
      out.append(Pdot)
      out.extend(thetadot.tolist())
    if medtherm:
      out.extend(Tmdot.tolist())
    if masstrans:
      out.extend(kvdot.tolist())
    if dZ is not None:
      out.extend(dZ.tolist())
    return out

  out = np.empty_like(y)
  out[0] = Rd
  out[1] = Rdd
  cursor = 2
  if bubtherm:
    out[cursor] = Pdot
    cursor += 1
    out[cursor : cursor + thetadot.size] = thetadot
    cursor += thetadot.size
  if medtherm:
    out[cursor : cursor + Tmdot.size] = Tmdot
    cursor += Tmdot.size
  if masstrans:
    out[cursor : cursor + kvdot.size] = kvdot
    cursor += kvdot.size
  if dZ is not None:
    out[cursor : cursor + dZ.size] = dZ
  return out


def _radius_floor_event(_tn, y, *_args):
  return y[0] - 1e-8
