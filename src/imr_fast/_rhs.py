"""Right-hand side of the bubble-dynamics system.

Evaluates the radial equation, the gas and medium thermal PDEs, vapour
transport and the constitutive state for one point in time. Pure evaluation:
it reads a prepared problem but never builds one.
"""

from __future__ import annotations

import numpy as np

from ._autodiff import at_set, primal, primal_array
from ._materials import _stress_state_count
from ._stress import _distributed_stress, _stress
from ._thermal import _apply_thermal_boundaries, _dissipation, _distributed_dissipation, _mie_gruneisen

__all__ = ["_nZ", "_pinf", "_rhs", "_rhs_args", "_sampled_pressure"]

def _sampled_pressure(tn, forcing, *, xp=np):
  """The PCHIP forcing history, without a Python branch on the integration time.

  Three things here were data-dependent on `tn`, which a tracer supplies: the
  out-of-range test, the knot search, and the index into the coefficient rows.
  None of them needed a new algorithm.

  `searchsorted` exists in both namespaces and takes a traced needle. The index it
  returns is then clamped rather than compared, and the coefficients are read
  through `xp` because a tracer cannot index a numpy array at all.

  The range test becomes a 0/1 MASK carried by an ordinary multiply, not an
  `xp.where` over the results. `where` is not a ufunc, so `np.where` on a `Dual`
  would return an object array rather than a `Dual` and quietly break the tangent
  path -- whereas the mask itself is built from plain floats in every arithmetic,
  and the polymorphic multiply is what applies it. Both arms are evaluated, on the
  same argument as `_pinf`'s windows: a cubic at a clamped interval stays finite,
  so nothing poisons a gradient.
  """
  time_value = primal(tn)
  # `primal_array` strips `Dual` tangents so the SEARCH runs on values, which is all
  # it needs. Under tracing there is nothing to strip and nothing that can be: the
  # tracer is the value, and forcing it to float64 raises. The arithmetic below still
  # uses `forcing.knots` itself, so either way the tangent is kept where it matters.
  knot_values = primal_array(forcing.knots) if xp is np else xp.asarray(forcing.knots)
  interval = xp.clip(xp.searchsorted(xp.asarray(knot_values), time_value, side="right") - 1, 0, knot_values.size - 2)
  # The offset keeps the ORIGINAL knots, not the primal copy above: `knots` is the
  # sampled time divided by `t0`, so it carries a tangent whenever `t0` does. The
  # search needs only values; the arithmetic needs the derivative.
  offset = tn - xp.asarray(forcing.knots)[interval]
  coefficients = xp.asarray(forcing.coefficients)
  c0, c1, c2, c3 = (coefficients[row][interval] for row in range(4))
  pressure = ((c0 * offset + c1) * offset + c2) * offset + c3
  pressure_rate = (3.0 * c0 * offset + 2.0 * c1) * offset + c2
  inside = xp.where((time_value >= knot_values[0]) & (time_value <= knot_values[-1]), 1.0, 0.0)
  return pressure * inside, pressure_rate * inside

def _pinf(tn, p, forcing=None, *, xp=np):
  if forcing is not None: return _sampled_pressure(tn, forcing, xp=xp)
  wt, ee, om, tw, dt, mn = (p["wave_type"], p["ee"], p["om"], p["tw"], p["dt"], p["mn"])
  # `wave_type` is configuration and stays a Python branch. The WINDOWS are not:
  # they test `tn`, the integration time, which a tracer supplies, so they become
  # `where`. Same values on numpy -- verified bit-identical -- but both arms are
  # now evaluated, which is why the histotripsy cosine is clamped rather than
  # guarded. Outside its window `c` would otherwise go negative and `c ** (mn - 1)`
  # produce a nan that `where` would then select against, and a nan selected
  # against still poisons a gradient.
  #
  # `ee` is not configuration either, any more: it scales with `pA`, which the
  # traced sensitivity path differentiates. The `if ee == 0.0` early-out this used
  # to carry was a Python branch on that value. Removing it costs one branch on an
  # unforced solve and needs no compensating arithmetic, because every arm below is
  # already proportional to `ee` -- verified bit-identical on all four arithmetics.
  if wt == 0:  # constant offset impulse
    return ee, 0.0
  if wt == 1:  # Gaussian
    e = xp.exp(-((tn - dt) ** 2) / tw**2)
    return -ee * e, ee * (2 * (tn - dt) / tw**2) * e
  if wt == 2:  # histotripsy pulse
    inside = (tn >= dt - np.pi / om) & (tn <= dt + np.pi / om)
    c = xp.maximum(0.5 + 0.5 * xp.cos(om * (tn - dt)), 1e-300)
    pressure = ee * c**mn
    rate = -ee * mn * c ** (mn - 1) * 0.5 * om * xp.sin(om * (tn - dt))
    return xp.where(inside, pressure, 0.0), xp.where(inside, rate, 0.0)
  if wt == 3:  # Heaviside step
    return -ee * xp.where(tn > tw, 0.0, 1.0), 0.0 * tn
  raise ValueError(f"wave_type={wt} not supported")

def _nZ(material): return _stress_state_count(material)

def _rhs_args(problem, p, *, medium):
  """The positional arguments `_rhs` takes, for a prepared problem.

  `medium` is required rather than read off `problem`, and that is deliberate. Its
  wall weights are built from `chi`, `iota`, `Fom` and `L_heat_star`, so the traced
  sensitivity path has to substitute a rebuilt one -- and every other consumer of
  those weights has to substitute the SAME one. Passing it silently defaulted is
  how `_thermal_fields` came to use the prepared medium while `_rhs` used the
  rebuilt one, which left the temperature output tangents 5.7e-04 wrong against a
  finite difference while every state tangent converged.

  One definition, because there were two and they had drifted. The forward path
  built the full tuple; `_jax.sensitivities_jax` built its own with every thermal
  slot zeroed, which silently made the jax tangents mechanical-only and is what
  the `bubtherm=1` refusal in `sensitivity.py` was standing in front of.

  `p` is passed rather than read off `problem` because the traced sensitivity path
  rebuilds it from tracers -- that is the whole point there -- while the forward
  path uses `problem.parameters` unchanged.
  """
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
  # `xp` is the array namespace. numpy by default, so every existing caller and
  # every pinned trajectory is untouched; `jax.numpy` is what a second backend
  # passes. W11 stage 2a -- the point is that ONE right-hand side serves both,
  # rather than a transcribed copy that can drift from this one the way
  # `_mechanical` already has.
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
      # Kstar is alpha_m*T + beta_m, the mixture conductivity, already formed above.
      nonlinear_diffusion = dkv * (dtheta / (Kstar * T) + RDkv)
      advection_term2 = (Uvel - Rd * ygrid) / R * dkv
      kvdot = at_set(Fom / R**2 * (ddkv - nonlinear_diffusion) - advection_term2, -1, 0.0)
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
      thetadot = at_set(advection + diffusion, -1, 0.0)
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
    # One formula, four equations of state. It used to be four copies of the
    # `num`/`den` block below, byte-identical, differing only in how (Cs, hB, hH)
    # were obtained -- and the compiled mirror in `_mechanical` had already
    # collapsed them to two. Four copies is how one radial branch drifts from
    # its siblings while they stay right, which is exactly what #18 found.
    if radial in (3, 4):  # Tait
      Pb = P - iWe / R + p["tait_gamma"] + S
      hB = p["tait_sam"] / p["tait_no"] * ((Pb / p["tait_sam"]) ** p["tait_no"] - 1.0)
      hH = (p["tait_sam"] / Pb) ** (1.0 / p["tait_exponent"])
      # hH is 1/rho, so Gilmore's sqrt(gamma*Pb/rho) needs no second power.
      Cs = p["Cstar"] if radial == 3 else xp.sqrt(p["tait_exponent"] * Pb * hH)
    else:  # Mie-Gruneisen. Upstream omits +S from Pb here; restoring it is what
      # brings these branches back into agreement with the Tait forms (PLAN W9).
      Pb = P - iWe / R + S
      C, hB, hH = _mie_gruneisen(Pb, p["Cstar"], p["hugoniot_slope"], p["nog"], p["mie_reference"], xp=xp)
      Cs = p["Cstar"] if radial == 5 else C
    num = (1 + Rd / Cs) * (hB - Pf8) - R / Cs * Pf8dot + R / Cs * hH * (Pdot + iWe * Rd / R**2 + Sdot) - 1.5 * (1 - Rd / (3 * Cs)) * Rd**2
    den = (1 - Rd / Cs) * R + acceleration_coefficient * hH / Cs
    Rdd = num / den
  else:
    raise ValueError(f"radial={radial} not supported")
  if distributed_stress is None:
    # `list(...)` rather than `.tolist()`: both give one element per entry, but
    # `.tolist()` demands concrete values and a traced array has none. Shapes
    # here are static, so iterating is fine under `jit`. numpy is unaffected --
    # the elements become `np.float64` instead of `float` and every downstream
    # value is identical (verified bit-for-bit).
    # Keyed on the fields rather than the flags they came from. Each is None
    # exactly when its flag is off, so this is the same condition -- but it is
    # the one a type checker can narrow, and it ties the packing to the data.
    out = [Rd, Rdd]
    if thetadot is not None:
      out.append(Pdot)
      out.extend(list(thetadot))
    if Tmdot is not None: out.extend(list(Tmdot))
    if kvdot is not None: out.extend(list(kvdot))
    if dZ is not None: out.extend(list(dZ))
    return out
  # The distributed branch keeps its preallocated buffer rather than joining the
  # list above: `dZ` here is 2*points long -- 480 entries at the default -- and a
  # Python list of that is not the same thing. `at_set` mutates in place on numpy
  # and rebuilds functionally on jax, so both get what they need.
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
