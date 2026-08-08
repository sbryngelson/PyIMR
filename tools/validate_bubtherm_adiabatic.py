"""
Adiabatic-limit self-check for bubtherm=1, NO IMRv2 round-trip needed.

With chi=0 exactly and vapor=0 (Pv=0), the thermal branch's
  Pdot = 3/R*(chi*(kappa-1)*dtheta[-1]/R - kappa*P*Rd)
collapses EXACTLY to the polytropic-FORM branch's
  Pdot = -3*kappa*(P-Pv)*Rd/R = -3*kappa*P*Rd/R          (since Pv=0)
and theta's own dynamics decouple from (R, Rdot, P) entirely (nothing feeds
theta back into Pdot once chi=0).

IMPORTANT: bubtherm=1 and bubtherm=0 use DIFFERENT P0 initial-condition
formulas (f_call_params.m:158-163 -- exponent 3 vs 3*kappa; this is a real,
intentional physical difference, not a bug -- see pyimr.py's params()
docstring). So comparing against F.simulate(...,bubtherm=0) directly is
WRONG: it starts from a different P(0) and the two trajectories will
legitimately differ. The correct reduction check isolates the RHS-structure
question from that: run a "shadow polytropic" ODE (bubtherm=0 equation
FORM) using bubtherm=1's OWN Pb, so both sides start from the identical
initial condition and only the RHS structure is being tested.
"""

import numpy as np
from scipy.integrate import solve_ivp

import pyimr as F
from pyimr._rhs import _rhs
from pyimr.thermal_fd import finite_diff_mat

R0, Req, G, mu = 225e-6, 225e-6 * 0.15, 2500.0, 0.1
tv = np.linspace(0, 1.2e-4, 400)

print("=" * 72)
print("ADIABATIC LIMIT: bubtherm=1 with chi forced to 0 vs shadow-polytropic")
print("(same Pb/P0 convention on both sides -- isolates RHS structure only)")
print("=" * 72)


def _shadow_polytropic(p, rtol, atol):
  """bubtherm=0 RHS form, but using p['Pb'] as given (bubtherm=1's Pb)."""
  tn = tv / p["t0"]
  y0 = [1.0, 0.0] + [0.0] * F._stress_state_count(material)
  s = solve_ivp(
    _rhs,
    (tn[0], tn[-1]),
    y0,
    t_eval=tn,
    args=(p, material, 1, 0),  # bubtherm=0 RHS form
    method="LSODA",
    rtol=rtol,
    atol=atol,
  )
  return s.y[0]


def _thermal_chi0(p, rtol, atol, Nt=25):
  p = dict(p)
  p["chi"] = 0.0  # not a physically meaningful override in general --
  tn = tv / p["t0"]  # intentionally not exposed via simulate()'s public API
  D1 = finite_diff_mat(Nt, 1, tm_check=0)
  D2 = finite_diff_mat(Nt, 2, tm_check=0)
  ygrid = np.linspace(0.0, 1.0, Nt)
  y0 = [1.0, 0.0, p["Pb"]] + [0.0] * Nt
  s = solve_ivp(_rhs, (tn[0], tn[-1]), y0, t_eval=tn, args=(p, material, 1, 1, D1, D2, ygrid), method="LSODA", rtol=rtol, atol=atol)
  return s.y[0]


# The (R,Rdot,P) subsystem is decoupled from theta once chi=0 (Pdot no longer
# references dtheta at all), so it is MATHEMATICALLY identical to the
# shadow-polytropic system -- but LSODA still integrates the larger, coupled
# state vector (theta keeps evolving via its advection term even at chi=0,
# just with no diffusion and no feedback), so it takes different adaptive
# steps than the smaller system. That produces a residual set by solver
# TOLERANCE, not by the equations. Confirm this directly: a fixed algebraic
# error would not shrink as tolerance tightens; this should.
print("  tolerance scaling (a real equation bug would NOT shrink here):")
material = F.NeoHookeanKelvinVoigt(G, mu)
for rtol, atol in [(1e-8, 1e-10), (1e-10, 1e-12), (1e-12, 1e-14)]:
  p = F.params(R0, Req, material, bubtherm=1)
  R_shadow = _shadow_polytropic(p, rtol, atol)
  err_t = np.max(np.abs(_thermal_chi0(p, rtol, atol) - R_shadow))
  print(f"    rtol={rtol:.0e} atol={atol:.0e}  ->  err={err_t:.3e}")

p = F.params(R0, Req, material, bubtherm=1)
R_shadow = _shadow_polytropic(p, 1e-10, 1e-12)
R_thermal_chi0 = _thermal_chi0(p, 1e-10, 1e-12)
err = np.max(np.abs(R_thermal_chi0 - R_shadow))
tag = "PASS" if err < 1e-6 else "FAIL"
print(f"\n  at rtol=1e-10: max|R_thermal(chi=0) - R_shadow_polytropic| = {err:.3e}   {tag}")

print()
print("=" * 72)
print("SANITY: bubtherm=1 with REAL chi>0 should DIFFER from shadow-polytropic")
print("        (confirms the thermal terms aren't silently inert)")
print("=" * 72)
config = F.SimulationConfig(R0=R0, Req=Req, material=material, dynamics="rayleigh-plesset", bubtherm=1, Nt=25)
R_thermal_real = F.simulate(tv, config).radius_ratio
diff = np.max(np.abs(R_thermal_real - R_shadow))
nan_frac = np.mean(np.isnan(R_thermal_real))
print(f"  max|R_thermal(real chi) - R_shadow_polytropic| = {diff:.3e}")
print(f"  fraction NaN in thermal run: {nan_frac:.2f}")
print(f"  {'PASS (differs, as physically expected)' if diff > 1e-3 and nan_frac == 0 else 'FLAG -- inspect'}")
