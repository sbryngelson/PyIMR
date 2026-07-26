"""Full validation suite. Run: python3 tests/run_validation.py"""

import os
import sys
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import imr_fast
import _imr_thermal
from imr_fast import params
from imr_inference import InferenceParameter, RadiusObservation, prepare_inference


def solve_radius(times, R0, Req, material, **options):
  config = imr_fast.SimulationConfig(R0=R0, Req=Req, material=material, **options)
  return imr_fast.simulate(times, config).radius_ratio


fail = 0
print("=" * 64)
print("1. FORWARD SOLVER vs IMRv2 reference trajectories")
print("=" * 64)
_d = os.path.dirname(os.path.abspath(__file__))
_R0 = 225e-6
_t0 = _R0 / np.sqrt(101325 / 1064)
_t2 = np.loadtxt(f"{_d}/imr2_t.csv")
_A = np.loadtxt(f"{_d}/imr2_s06.csv", delimiter=",")
_Gg = np.loadtxt(f"{_d}/imr2_G.csv")
_Mg = np.loadtxt(f"{_d}/imr2_M.csv")
_checks = [
  ("Zener truth De=2 s=6", solve_radius(_t2, _R0, _R0 / 6, imr_fast.Zener(2500.0, 0.1, 2 * _t0, 0.4 * _t0)), _A[:, 0])
]
for _k in [0, 30, len(_Gg) * len(_Mg) - 1]:
  _gi, _mi = _k // len(_Mg), _k % len(_Mg)
  _checks.append(
    (
      f"NHKV G={_Gg[_gi]:.0f} mu={_Mg[_mi]:.4f}",
      solve_radius(_t2, _R0, _R0 / 6, imr_fast.NeoHookeanKelvinVoigt(_Gg[_gi], _Mg[_mi])),
      _A[:, 1 + _k],
    )
  )
for _lab, _py, _ml in _checks:
  _mx = np.nanmax(np.abs(_ml - _py))
  _ok = _mx < 2e-3
  fail += not _ok
  print(f"    {_lab:30s} max|dR|={_mx:.2e}  {'PASS' if _ok else 'FAIL'}")

print("\n" + "=" * 64)
print("1b. EXTENDED FEATURES vs IMRv2 references")
print("=" * 64)
_d = os.path.dirname(os.path.abspath(__file__))
_R0 = 225e-6
_t0 = _R0 / np.sqrt(101325 / 1064)
_t = np.loadtxt(f"{_d}/ref_t.csv")
for lab, kw, ref in [
  ("qKV alphax=0.10", dict(material=imr_fast.QuadraticKelvinVoigt(2500.0, 0.1, 0.10)), "ref_qkv_a010.csv"),
  ("qKV alphax=0.25", dict(material=imr_fast.QuadraticKelvinVoigt(2500.0, 0.1, 0.25)), "ref_qkv_a025.csv"),
  ("UCM/OldB De=0.5", dict(material=imr_fast.OldroydB(0.1, 0.5 * _t0, 0.1 * _t0)), "ref_ucm_De005.csv"),
  ("UCM/OldB De=2.0", dict(material=imr_fast.OldroydB(0.1, 2 * _t0, 0.4 * _t0)), "ref_ucm_De020.csv"),
  ("Keller-Miksis NHKV", dict(material=imr_fast.NeoHookeanKelvinVoigt(2500.0, 0.1), radial=2), "ref_km_nhkv.csv"),
  ("Keller-Miksis Zener", dict(material=imr_fast.Zener(2500.0, 0.1, 2 * _t0, 0.4 * _t0), radial=2), "ref_km_zener.csv"),
  ("Gaussian forcing pA=5e4", dict(wave_type=1, pA=5e4, TW=5e-6, DT=2e-5), "ref_gauss_pA50.csv"),
  ("Gaussian forcing pA=2e5", dict(wave_type=1, pA=2e5, TW=5e-6, DT=2e-5), "ref_gauss_pA200.csv"),
  ("constant offset pA=3e4", dict(wave_type=0, pA=3e4), "ref_imp_pA30.csv"),
  ("Heaviside step pA=5e4", dict(wave_type=3, pA=5e4, TW=3e-5), "ref_heav_pA50.csv"),
  ("histotripsy pulse", dict(wave_type=2, pA=1e5, omega=2 * np.pi / 2e-5, DT=3e-5, mn=2), "ref_histo.csv"),
  ("vapor=1 (T=298.15K)", dict(vapor=1, T8=298.15), "ref_vapor.csv"),
  ("bubtherm=1 (thermal PDE)", dict(bubtherm=1, Nt=25), "ref_bubtherm.csv"),
  ("medtherm=1 (liquid layer)", dict(bubtherm=1, medtherm=1, Nt=25, Mt=25), "ref_medtherm.csv"),
  ("masstrans=1 (vapor transfer)", dict(bubtherm=1, vapor=1, masstrans=1, Nt=25), "ref_masstrans.csv"),
  (
    "masstrans=1+medtherm=1 (coupled)",
    dict(bubtherm=1, vapor=1, masstrans=1, medtherm=1, Nt=25, Mt=25),
    "ref_masstrans_medtherm.csv",
  ),
  ("no constitutive stress", dict(material=imr_fast.NoStress()), "ref_stress0.csv"),
  ("quadratic Zener", dict(material=imr_fast.QuadraticZener(2500.0, 0.1, 2 * _t0, 0.4 * _t0, 0.25)), "ref_stress4.csv"),
  ("radial=3 (KM enthalpy, Tait)", dict(radial=3), "ref_radial3.csv"),
  ("radial=4 (Gilmore, Tait)", dict(radial=4), "ref_radial4.csv"),
  ("radial=3+Zener", dict(radial=3, material=imr_fast.Zener(2500.0, 0.1, 2 * _t0, 0.4 * _t0)), "ref_radial3_zener.csv"),
  ("radial=4+Zener", dict(radial=4, material=imr_fast.Zener(2500.0, 0.1, 2 * _t0, 0.4 * _t0)), "ref_radial4_zener.csv"),
]:
  ml = np.loadtxt(f"{_d}/{ref}")
  material = kw.pop("material", imr_fast.NeoHookeanKelvinVoigt(2500.0, 0.1))
  py = solve_radius(_t, _R0, _R0 / 6, material, **kw)
  mx = np.nanmax(np.abs(ml - py))
  ok = mx < 2e-3
  fail += not ok
  print(f"    {lab:24s} max|dR|={mx:.2e}  {'PASS' if ok else 'FAIL'}")

print("\n" + "=" * 64)
print("1d. MIE-GRUNEISEN EOS -- DELIBERATE DIVERGENCE FROM IMRv2")
print("=" * 64)
# IMRv2's Mie-Gruneisen branch takes the wrong root of its own density
# quadratic. a*mu^2 + b*mu + A = 0 has roots that tend to 0 and to -1/nog as
# A -> 0; upstream takes (-b + sqrt(d))/(2a), which is the -1/nog branch --
# a 32.5% density deficit at ambient pressure. It also omits the stress term
# from Pb, which radial=3 and 4 both include.
#
# Both are corrected here, so radial=5 no longer reproduces ref_radial5.csv.
# The checks below are what justify that divergence.
_R = imr_fast.RHO
_p = imr_fast.params(
  _R0,
  _R0 / 6,
  imr_fast.NeoHookeanKelvinVoigt(2500.0, 0.1),
  0,
  298.15,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0,
  0,
  0,
  imr_fast.PhysicalParameters(),
)
_Cs, _s, _nog = _p["Cstar"], _p["hugoniot_slope"], _p["nog"]

# 1. density is undisturbed at ambient pressure
_mu = _imr_thermal._mu_of_A(1.0 / _Cs**2, _s, _nog)
_ok = abs(_mu) < 1e-4
fail += not _ok
print(f"    rho/rho0 - 1 at ambient = {_mu:.3e}  {'PASS' if _ok else 'FAIL'}")

# 2. sound speed at ambient recovers c0 -- the EoS calibration point
_C, _, _ = _imr_thermal._mie_gruneisen(1.0, _Cs, _s, _nog, _p["mie_reference"])
_rel = abs(float(_C) - _Cs) / _Cs
_ok = _rel < 1e-3
fail += not _ok
print(f"    c(ambient)/c0 - 1       = {_rel:.3e}  {'PASS' if _ok else 'FAIL'}")

# 3. enthalpy matches its weakly-compressible limit h ~ P - 1
_worst = 0.0
for _P in (2.0, 10.0, 100.0):
  _, _hB, _ = _imr_thermal._mie_gruneisen(_P, _Cs, _s, _nog, _p["mie_reference"])
  _worst = max(_worst, abs(float(_hB) - (_P - 1.0)) / (_P - 1.0))
_ok = _worst < 5e-3
fail += not _ok
print(f"    max |h - (P-1)|/(P-1)   = {_worst:.3e}  {'PASS' if _ok else 'FAIL'}")

# 4. the corrected branches must agree with the independent Tait forms to
#    within the spread the Tait forms already have among themselves
_rad = {
  _n: imr_fast.simulate(
    _t, imr_fast.SimulationConfig(R0=_R0, Req=_R0 / 6, material=imr_fast.NeoHookeanKelvinVoigt(2500.0, 0.1), radial=_n)
  ).radius_ratio
  for _n in (2, 3, 4, 5, 6)
}
_spread = np.nanmax(np.abs(_rad[3] - _rad[4]))
for _lab, _a, _b, _tol in [
  ("radial=5 (KM/Mie-G) vs radial=3 (KM/Tait)", 5, 3, 2e-3),
  ("radial=6 (Gilmore/Mie-G) vs radial=4 (Gilmore/Tait)", 6, 4, 3e-2),
  ("radial=6 vs radial=5 (Gilmore vs KM, same EoS)", 6, 5, 3e-2),
]:
  _mx = np.nanmax(np.abs(_rad[_a] - _rad[_b]))
  _ok = _mx < _tol
  fail += not _ok
  print(f"    {_lab:52s} {_mx:.2e}  {'PASS' if _ok else 'FAIL'}")
print(f"    (reference spread, radial=3 vs radial=4: {_spread:.2e})")

# 5. radial=6 is unavailable upstream at all; assert it runs clean here
_ok = np.all(np.isfinite(_rad[6])) and np.isrealobj(_rad[6])
fail += not _ok
print(f"    radial=6 finite and real (IMRv2 returns complex)  {'PASS' if _ok else 'FAIL'}")

print("\n" + "=" * 64)
print("1c. COLLAPSE INITIALIZATION vs IMRv2")
print("=" * 64)
# References from tools/gen_imrv2_cases.m. IMRv2 requires the full coupled
# model for collapse (f_call_params.m:234-236), so these run bubtherm +
# medtherm + masstrans + vapor together.
gaps = []
_full = dict(bubtherm=1, medtherm=1, masstrans=1, vapor=1, Nt=25, Mt=25)

# IMRv2 applies a collapse precursor only for stress 3 and 4 (Zener family).
# f_call_params.m:447-460 leaves Szero empty for stress < 3 and returns zeros
# under an explicit "TODO" for stress == 5, so collapse=1 is a no-op for NHKV
# and Oldroyd-B. Those two references therefore pin the fully coupled model
# without a precursor; only the Zener case tests collapse itself.
_ml = np.loadtxt(f"{_d}/ref_collapse_zener.csv")
_cfg = imr_fast.SimulationConfig(
  R0=_R0,
  Req=_R0 / 6,
  material=imr_fast.Zener(2500.0, 0.1, 2 * _t0, 0.4 * _t0),
  collapse=imr_fast.CollapseInitialization(),
  **_full,
)
_mx = np.nanmax(np.abs(_ml - imr_fast.simulate(_t, _cfg).radius_ratio))
_ok = _mx < 2e-3
fail += not _ok
print(f"    collapse Zener        max|dR|={_mx:.2e}  {'PASS' if _ok else 'FAIL'}")

# The residual above is entirely one number: the precursor stress at maximum
# radius. imr-fast root-finds the true maximum (velocity == 0) and gets
# S0 = -0.15994511, converged and integrator-independent. IMRv2 takes a
# discrete argmax over ode23tb output points and gets -0.16004691.
#
# That gap is inherent, not a tolerance choice. R is locally quadratic at the
# peak, so it moves only 2.1e-06 over the 1.9e-03 of nondimensional time that
# separates the two; S is locally linear there (dS/dt = +0.053), so it moves
# 1.0e-04. Locating a quadratic maximum by argmax costs O(sqrt(tol)) in
# position and therefore O(sqrt(tol)) in the stress carried out of it.
#
# Injecting IMRv2's own Szero reproduces the reference in the normal pinned
# band, which proves the coupled solve is exact and isolates the whole
# difference to the precursor.
_IMRV2_SZERO = -0.1600469117114953
_cfg_upstream = imr_fast.SimulationConfig(
  R0=_R0,
  Req=_R0 / 6,
  material=imr_fast.Zener(2500.0, 0.1, 2 * _t0, 0.4 * _t0),
  initial=imr_fast.InitialState(stress_state=(_IMRV2_SZERO,)),
  **_full,
)
_mx = np.nanmax(np.abs(_ml - imr_fast.simulate(_t, _cfg_upstream).radius_ratio))
_ok = _mx < 5e-5
fail += not _ok
print(f"    collapse Zener w/ IMRv2 Szero  max|dR|={_mx:.2e}  {'PASS' if _ok else 'FAIL'}")

for lab, mat, ref in [
  ("Oldroyd-B", imr_fast.OldroydB(0.1, 2 * _t0, 0.4 * _t0), "ref_coupled_oldb.csv"),
  ("NHKV", imr_fast.NeoHookeanKelvinVoigt(2500.0, 0.1), "ref_coupled_nhkv.csv"),
]:
  ml = np.loadtxt(f"{_d}/{ref}")
  cfg = imr_fast.SimulationConfig(R0=_R0, Req=_R0 / 6, material=mat, **_full)
  mx = np.nanmax(np.abs(ml - imr_fast.simulate(_t, cfg).radius_ratio))
  ok = mx < 2e-3
  fail += not ok
  print(f"    coupled {lab:14s} max|dR|={mx:.2e}  {'PASS' if ok else 'FAIL'}")

# imr-fast refuses collapse without memory; IMRv2 accepts the flag and ignores
# it. Refusing is the stricter, correct behaviour -- assert it stays that way.
try:
  imr_fast.simulate(
    _t,
    imr_fast.SimulationConfig(
      R0=_R0,
      Req=_R0 / 6,
      material=imr_fast.NeoHookeanKelvinVoigt(2500.0, 0.1),
      collapse=imr_fast.CollapseInitialization(),
      **_full,
    ),
  )
  _ok = False
except ValueError:
  _ok = True
fail += not _ok
print(f"    memoryless collapse refused (IMRv2 silently ignores)  {'PASS' if _ok else 'FAIL'}")

print("\n" + "=" * 64)
print("2. CONSTITUTIVE SUITE")
print("=" * 64)


def _instantaneous_values(material, radius=0.5, velocity=-0.3, need_rate=True):
  config = imr_fast.SimulationConfig(R0=_R0, Req=_R0 / 6, material=material)
  problem = imr_fast.prepare(config)
  return imr_fast._stress(
    material, problem.parameters, radius, velocity, None, problem.instantaneous_material, need_rate
  )


print("  composable NH/Newtonian trajectory vs closed-form fast path")
_generic_nh = imr_fast.InstantaneousMaterial(imr_fast.NeoHookean(2500.0), imr_fast.Newtonian(0.1))
_equivalence_options = dict(rtol=1e-10, atol=1e-12)
_trajectory_tolerance = 1e-7
for _radial in (1, 2):
  _closed = solve_radius(
    _t, _R0, _R0 / 6, imr_fast.NeoHookeanKelvinVoigt(2500.0, 0.1), radial=_radial, **_equivalence_options
  )
  _generic = solve_radius(_t, _R0, _R0 / 6, _generic_nh, radial=_radial, **_equivalence_options)
  _mx = np.max(np.abs(_generic - _closed))
  _ok = _mx < _trajectory_tolerance
  fail += not _ok
  print(f"    radial={_radial} max|dR|={_mx:.2e}  {'PASS' if _ok else 'FAIL'}")
_thermal_options = dict(bubtherm=1, medtherm=1, Nt=9, Mt=9, **_equivalence_options)
_closed = solve_radius(_t, _R0, _R0 / 6, imr_fast.NeoHookeanKelvinVoigt(2500.0, 0.1), **_thermal_options)
_generic = solve_radius(_t, _R0, _R0 / 6, _generic_nh, **_thermal_options)
_mx = np.max(np.abs(_generic - _closed))
_ok = _mx < _trajectory_tolerance
fail += not _ok
print(f"    thermal max|dR|={_mx:.2e}  {'PASS' if _ok else 'FAIL'}")

print("  elastic model reductions to neo-Hookean")
_elastic_reference = _instantaneous_values(imr_fast.InstantaneousMaterial(elastic=imr_fast.NeoHookean(2500.0)))[0]
for _name, _elastic, _tol in [
  ("Mooney-Rivlin", imr_fast.MooneyRivlin(1250.0, 0.0), 1e-12),
  ("Yeoh", imr_fast.Yeoh(1250.0), 1e-12),
  ("Fung", imr_fast.Fung(2500.0, 0.0), 1e-12),
  ("Gent", imr_fast.Gent(2500.0, 1e9), 1e-8),
  ("Arruda-Boyce", imr_fast.ArrudaBoyce(2500.0, 1e9), 1e-8),
]:
  _value = _instantaneous_values(imr_fast.InstantaneousMaterial(elastic=_elastic))[0]
  _rel = abs(_value - _elastic_reference) / abs(_elastic_reference)
  _ok = _rel < _tol
  fail += not _ok
  print(f"    {_name:16s} rel={_rel:.2e}  {'PASS' if _ok else 'FAIL'}")

print("  viscous model reductions to Newtonian")
_viscous_reference = _instantaneous_values(imr_fast.InstantaneousMaterial(viscous=imr_fast.Newtonian(0.1)))[0]
for _name, _viscous in [
  ("power law", imr_fast.PowerLaw(0.1, 1.0)),
  ("Carreau-Yasuda", imr_fast.CarreauYasuda(0.1, 0.1, 1.0, 2.0, 0.5)),
  ("Cross", imr_fast.Cross(0.1, 0.1, 1.0, 2.0)),
  ("Powell-Eyring", imr_fast.PowellEyring(0.1, 0.1, 1.0)),
  ("mod Powell-Eyring", imr_fast.ModifiedPowellEyring(0.1, 0.1, 1.0)),
  ("Powell-Eyring lam=0", imr_fast.PowellEyring(0.1, 0.05, 0.0)),
  ("mod Powell-Eyring lam=0", imr_fast.ModifiedPowellEyring(0.1, 0.05, 0.0)),
  ("Herschel-Bulkley", imr_fast.HerschelBulkley(0.0, 0.1, 1.0)),
  ("Bingham", imr_fast.Bingham(0.0, 0.1)),
]:
  _value = _instantaneous_values(imr_fast.InstantaneousMaterial(viscous=_viscous))[0]
  _rel = abs(_value - _viscous_reference) / abs(_viscous_reference)
  _ok = _rel < 1e-12
  fail += not _ok
  print(f"    {_name:16s} rel={_rel:.2e}  {'PASS' if _ok else 'FAIL'}")

print("  analytic stress rates vs centered differences")
_materials = [
  imr_fast.InstantaneousMaterial(elastic=imr_fast.MooneyRivlin(1000.0, 400.0)),
  imr_fast.InstantaneousMaterial(elastic=imr_fast.Yeoh(1000.0, 100.0, 10.0)),
  imr_fast.InstantaneousMaterial(elastic=imr_fast.Fung(2500.0, 0.2)),
  imr_fast.InstantaneousMaterial(elastic=imr_fast.Gent(2500.0, 500.0)),
  imr_fast.InstantaneousMaterial(elastic=imr_fast.ArrudaBoyce(2500.0, 50.0)),
  imr_fast.InstantaneousMaterial(viscous=imr_fast.PowerLaw(0.1, 0.7)),
  imr_fast.InstantaneousMaterial(viscous=imr_fast.CarreauYasuda(0.1, 0.01, 1e-5, 2.0, 0.5)),
  imr_fast.InstantaneousMaterial(viscous=imr_fast.Cross(0.1, 0.01, 1e-5, 2.0)),
  imr_fast.InstantaneousMaterial(viscous=imr_fast.HerschelBulkley(100.0, 0.1, 0.8)),
  imr_fast.InstantaneousMaterial(viscous=imr_fast.Bingham(100.0, 0.1)),
  imr_fast.InstantaneousMaterial(viscous=imr_fast.PowellEyring(0.5, 0.1, 2e-5)),
  imr_fast.InstantaneousMaterial(viscous=imr_fast.ModifiedPowellEyring(0.5, 0.1, 2e-5)),
]
_rate_errors = []
for _material in _materials:
  _R, _Rd, _Rdd, _h = 0.5, -0.3, 0.2, 1e-6
  _S, _Sdot, _, _A = _instantaneous_values(_material, _R, _Rd)
  _plus = _instantaneous_values(_material, _R + _h * _Rd, _Rd + _h * _Rdd, False)[0]
  _minus = _instantaneous_values(_material, _R - _h * _Rd, _Rd - _h * _Rdd, False)[0]
  _finite_difference = (_plus - _minus) / (2 * _h)
  _predicted = _Sdot - _A / _R * _Rdd
  _rate_errors.append(abs(_finite_difference - _predicted) / max(1.0, abs(_finite_difference)))
_mx = max(_rate_errors)
_ok = _mx < 2e-7
fail += not _ok
print(f"    maximum relative error={_mx:.2e}  {'PASS' if _ok else 'FAIL'}")

print("  Gent lock-up becomes a solver failure")
try:
  solve_radius(_t[:3], _R0, _R0 / 6, imr_fast.InstantaneousMaterial(elastic=imr_fast.Gent(2500.0, 5.0)))
  _ok = False
except imr_fast.SimulationError as _error:
  _ok = "Gent lock-up" in str(_error)
fail += not _ok
print(f"    {'PASS' if _ok else 'FAIL'}")

print("\n" + "=" * 64)
print("2b. NONLINEAR MEMORY (Giesekus / PTT) reduction limits")
print("=" * 64)
_De, _LAM = 2.0, 0.2
_tv = np.linspace(0, 1.2e-4, 300)
_p0 = params(225e-6, 225e-6 / 6, imr_fast.NeoHookeanKelvinVoigt(2500.0, 0.1))
_relaxation = _De * _p0["t0"]
_retardation = _LAM * _relaxation
_ucm_material = imr_fast.OldroydB(0.1, _relaxation, _retardation)
_ucm = solve_radius(_tv, 225e-6, 225e-6 / 6, _ucm_material)
print("  zero nonlinearity must reproduce UCM/Oldroyd-B")
for _name, _model in [
  ("giesekus", imr_fast.Giesekus(0.1, _relaxation, _retardation)),
  ("linear PTT", imr_fast.LinearPTT(0.1, _relaxation, _retardation)),
]:
  _R = solve_radius(_tv, 225e-6, 225e-6 / 6, _model)
  _mx = np.nanmax(np.abs(_R - _ucm))
  _ok = _mx < 5e-3
  fail += not _ok  # discretisation-limited, converging in points
  print(f"    {_name:>10} -> UCM   max|dR|={_mx:.2e}  {'PASS' if _ok else 'FAIL'}")
_ucm_km = solve_radius(_tv, 225e-6, 225e-6 / 6, _ucm_material, radial=2)
_distributed_km = solve_radius(_tv, 225e-6, 225e-6 / 6, imr_fast.Giesekus(0.1, _relaxation, _retardation), radial=2)
_mx = np.nanmax(np.abs(_distributed_km - _ucm_km))
_ok = _mx < 2e-3
fail += not _ok
print(f"    {'KM Giesekus':>10} -> UCM   max|dR|={_mx:.2e}  {'PASS' if _ok else 'FAIL'}")
_coupled_options = dict(bubtherm=1, medtherm=1, vapor=1, masstrans=1, Nt=9, Mt=9)
_ucm_coupled = solve_radius(_tv, 225e-6, 225e-6 / 6, _ucm_material, **_coupled_options)
_distributed_coupled = solve_radius(
  _tv, 225e-6, 225e-6 / 6, imr_fast.Giesekus(0.1, _relaxation, _retardation), **_coupled_options
)
_mx = np.nanmax(np.abs(_distributed_coupled - _ucm_coupled))
_ok = _mx < 3e-3
fail += not _ok
print(f"    {'coupled':>10} -> UCM   max|dR|={_mx:.2e}  {'PASS' if _ok else 'FAIL'}")
print("  nonlinear parameter must produce distinct physics")
for _name, _model in [
  ("giesekus", imr_fast.Giesekus(0.1, _relaxation, _retardation, mobility=0.2)),
  ("linear PTT", imr_fast.LinearPTT(0.1, _relaxation, _retardation, extensibility=0.2)),
]:
  _R = solve_radius(_tv, 225e-6, 225e-6 / 6, _model)
  _mx = np.nanmax(np.abs(_R - _ucm))
  _ok = _mx > 0.05
  fail += not _ok
  print(f"    {_name:>10} parameter=0.2  max|dR| vs UCM={_mx:.2e}  {'PASS' if _ok else 'FAIL'}")

print("\n" + "=" * 64)
print("2c. DISTRIBUTED STRESS QUADRATURE")
print("=" * 64)
# The distributed constitutive equations are pointwise ODEs -- no spatial
# derivatives -- so the only spatial approximation is the quadrature for
# I = int_R^inf 2 (t_rr - t_hh) / r dr. Mapping to the Lagrangian coordinate
# turns it into a fixed-weight sum, which makes Gauss-Legendre available and
# makes the integral's time derivative exact instead of a discrete difference.
_qt = np.linspace(0.0, 120e-6, 300)


def _giesekus(points, quadrature, mobility=0.2):
  return imr_fast.simulate(
    _qt,
    imr_fast.SimulationConfig(
      R0=_R0,
      Req=_R0 / 6,
      material=imr_fast.Giesekus(0.1, 2 * _t0, 0.4 * _t0, mobility, points=points, quadrature=quadrature),
    ),
  ).radius_ratio


_reference = _giesekus(1920, "gauss")
print("  Gauss-Legendre convergence (nonlinear, mobility=0.2)")
_previous = None
for _pts in (60, 120, 240):
  _mx = np.nanmax(np.abs(_giesekus(_pts, "gauss") - _reference))
  _ok = _previous is None or _mx < _previous
  fail += not _ok
  print(f"    points={_pts:4d}  max|dR| vs gauss(1920)={_mx:.2e}  {'PASS' if _ok else 'FAIL'}")
  _previous = _mx
_ok = _previous < 1e-5
fail += not _ok
print(f"    default resolution converged below 1e-5  {'PASS' if _ok else 'FAIL'}")

# Independent-rule cross-check: the two quadratures must agree once both are
# resolved. This is the only independent check the distributed models have --
# IMRv2 cannot run Giesekus or PTT at all.
_mx = np.nanmax(np.abs(_giesekus(240, "gauss") - _giesekus(3840, "trapezoid")))
_ok = _mx < 5e-3
fail += not _ok
print(f"    gauss(240) vs trapezoid(3840)   max|dR|={_mx:.2e}  {'PASS' if _ok else 'FAIL'}")

# The trapezoid rule at the former default carries percent-level error, which
# the Oldroyd-B reduction limit alone did not reveal.
_mx = np.nanmax(np.abs(_giesekus(480, "trapezoid") - _reference))
print(f"    (for reference, former default trapezoid(480): {_mx:.2e})")

print("\n" + "=" * 64)
print("3. UNIFIED FORWARD SENSITIVITIES")
print("=" * 64)
_sensitivity_times = np.linspace(0.0, 20e-6, 80)


def _material_offset(config, field, amount):
  return replace(config, material=replace(config.material, **{field: getattr(config.material, field) + amount}))


def _centered_output(times, config, field, step, output):
  plus = output(imr_fast.simulate(times, _material_offset(config, field, step)))
  minus = output(imr_fast.simulate(times, _material_offset(config, field, -step)))
  return (plus - minus) / (2.0 * step)


for _radial in range(1, 6):
  _config = imr_fast.SimulationConfig(
    _R0, _R0 / 6, imr_fast.NeoHookeanKelvinVoigt(2500.0, 0.1), radial=_radial, rtol=1e-10, atol=1e-12
  )
  _sensitivity = imr_fast.simulate_with_sensitivities(
    _sensitivity_times, _config, ["material.shear_modulus_pa"]
  ).radius_ratio[:, 0]
  # A one-percent material step stays in the centered-difference regime
  # while remaining above radial=5's adaptive-integration noise floor.
  _step = 25.0
  _difference = _centered_output(
    _sensitivity_times, _config, "shear_modulus_pa", _step, lambda result: result.radius_ratio
  )
  _relative = np.linalg.norm(_sensitivity - _difference) / np.linalg.norm(_difference)
  _ok = _relative < 2e-4
  fail += not _ok
  print(f"    radial={_radial} material tangent rel={_relative:.2e}  {'PASS' if _ok else 'FAIL'}")

print("  coupled heat/mass-transfer output tangent")
_coupled_config = imr_fast.SimulationConfig(
  _R0,
  _R0 / 6,
  imr_fast.NeoHookeanKelvinVoigt(2500.0, 0.1),
  bubtherm=1,
  medtherm=1,
  masstrans=1,
  vapor=1,
  Nt=7,
  Mt=7,
  rtol=1e-9,
  atol=1e-11,
)
_coupled_times = np.linspace(0.0, 2e-6, 8)
_coupled_sensitivity = imr_fast.simulate_with_sensitivities(
  _coupled_times, _coupled_config, ["material.shear_modulus_pa"]
)
_step = 0.025
_temperature_difference = _centered_output(
  _coupled_times, _coupled_config, "shear_modulus_pa", _step, lambda result: result.medium_temperature_k
)
_relative = np.linalg.norm(
  _coupled_sensitivity.medium_temperature_k[..., 0] - _temperature_difference
) / np.linalg.norm(_temperature_difference)
_ok = _relative < 2e-3
fail += not _ok
print(f"    medium temperature rel={_relative:.2e}  {'PASS' if _ok else 'FAIL'}")

print("  collapse shooting tangent")
_collapse_config = imr_fast.SimulationConfig(
  _R0, _R0 / 6, imr_fast.Zener(2500.0, 0.1, 40e-6, 8e-6), radial=2, collapse=imr_fast.CollapseInitialization()
)
_collapse_sensitivity = imr_fast.simulate_with_sensitivities(
  np.array([0.0, 1e-8]), _collapse_config, ["material.shear_modulus_pa"]
).state[0, -1, 0]
_collapse_problem = imr_fast.prepare(_collapse_config)
_collapse_difference = (
  imr_fast.prepare(_material_offset(_collapse_config, "shear_modulus_pa", _step)).initial_state[-1]
  - imr_fast.prepare(_material_offset(_collapse_config, "shear_modulus_pa", -_step)).initial_state[-1]
) / (2.0 * _step)
_relative = abs(_collapse_sensitivity - _collapse_difference) / abs(_collapse_difference)
_shooting_error = abs(_collapse_problem.collapse_stats.maximum_radius_ratio - 1.0)
_ok = _relative < 1e-5 and _shooting_error < 2e-8
fail += not _ok
print(
  f"    initial memory tangent rel={_relative:.2e}; "
  f"shooting residual={_shooting_error:.2e}  "
  f"{'PASS' if _ok else 'FAIL'}"
)

print("\n" + "=" * 64)
print("3b. TRACE ESTIMATORS (imr_data)")
print("=" * 64)
import imr_data

# equilibrium radius must invert the solver's own pressure/radius relation
_Pg0 = (imr_fast.P8 + 2 * imr_fast.SURF / (_R0 / 6)) * ((_R0 / 6) / _R0) ** (3 * imr_fast.KAPPA)
_rel = abs(imr_data.equilibrium_radius(_R0, _Pg0) - _R0 / 6) / (_R0 / 6)
_ok = _rel < 1e-12
fail += not _ok
print(f"    equilibrium radius round-trip  rel={_rel:.2e}  {'PASS' if _ok else 'FAIL'}")

# gas-only limit must reproduce Minnaert exactly
_w, _ = imr_data.natural_frequency(_R0, _R0 / 6, 1e-12, 1e-12, surface_tension_n_m=0.0)
_minnaert = np.sqrt(3 * imr_fast.KAPPA * imr_fast.P8 / imr_fast.RHO) / (_R0 / 6)
_rel = abs(_w - _minnaert) / _minnaert
_ok = _rel < 1e-12
fail += not _ok
print(f"    natural frequency -> Minnaert  rel={_rel:.2e}  {'PASS' if _ok else 'FAIL'}")

# and it must land near the measured late-time rebound frequency
_tt = np.linspace(0, 300e-6, 8000)
_rr = imr_fast.simulate(
  _tt, imr_fast.SimulationConfig(R0=_R0, Req=_R0 / 6, material=imr_fast.NeoHookeanKelvinVoigt(2500.0, 0.1))
).radius_m
_tc, _rp, _ = imr_data.collapse_features(_tt, _rr)
# The first rebound is strongly nonlinear and the late tail is numerical
# wiggle, so take the median of the intermediate periods.
_measured = float(np.median(2 * np.pi / np.diff(_tc)[1:5]))
_predicted, _ = imr_data.natural_frequency(_R0, _R0 / 6, 2500.0, 0.1)
_rel = abs(_predicted - _measured) / _measured
_ok = _rel < 0.10
fail += not _ok
print(
  f"    vs measured rebound  predicted={_predicted:.3e} measured={_measured:.3e} "
  f"rel={_rel:.2e}  {'PASS' if _ok else 'FAIL'}"
)

# feature extraction must find a monotonically decaying rebound sequence
_ok = len(_tc) >= 3 and len(_rp) >= 3 and np.all(np.diff(_rp[:3]) < 0.0)
fail += not _ok
print(f"    collapse features: {len(_tc)} collapses, {len(_rp)} peaks, decaying  {'PASS' if _ok else 'FAIL'}")

# thermal grid refinement must converge monotonically
_conv = imr_data.resolution_convergence(
  imr_fast.SimulationConfig(
    R0=_R0, Req=_R0 / 6, material=imr_fast.NeoHookeanKelvinVoigt(2500.0, 0.1), bubtherm=1, Nt=10, Mt=10
  ),
  np.linspace(0, 60e-6, 200),
  [(10, 10), (20, 20), (40, 40)],
)
_errs = [e for _, e in _conv]
_ok = _errs[0] > _errs[1] > _errs[2] == 0.0
fail += not _ok
print(f"    thermal grid convergence {[f'{e:.1e}' for e in _errs]}  {'PASS' if _ok else 'FAIL'}")

print("\n" + "=" * 64)
print("4. PREPARED INFERENCE")
print("=" * 64)
_inference_config = imr_fast.SimulationConfig(_R0, _R0 / 6, imr_fast.NeoHookeanKelvinVoigt(2500.0, 0.1))
_inference_times = np.linspace(0.0, 20e-6, 50)
_truth = imr_fast.simulate(_inference_times, _inference_config)
_inference = prepare_inference(
  _inference_config,
  RadiusObservation(_inference_times, _truth.radius_m, 1e-8),
  (
    InferenceParameter("material.shear_modulus_pa", 2000.0, 3000.0),
    InferenceParameter("material.viscosity_pa_s", 0.05, 0.15),
  ),
)
_evaluation = _inference.evaluate(np.array([0.5, 0.5]))
_jacobian = _inference.jacobian(np.array([0.5, 0.5]))
_ok = (
  np.max(np.abs(_evaluation.residual)) == 0.0
  and _jacobian.shape == (_inference_times.size, 2)
  and np.all(np.isfinite(_jacobian))
)
fail += not _ok
print(f"    likelihood and Jacobian  {'PASS' if _ok else 'FAIL'}")
_multistart = _inference.fit_multistart(2, seed=7, max_evaluations=20)
_ok = len(_multistart.endpoints) == 2 and _multistart.best is not None and _multistart.best.cost < 1e-12
fail += not _ok
print(f"    retained deterministic multistart endpoints  {'PASS' if _ok else 'FAIL'}")

if gaps:
  print("\n" + "=" * 64)
  print("KNOWN GAPS vs IMRv2 (tracked in PLAN.md W1)")
  print("=" * 64)
  for lab, mx, tol in gaps:
    detail = "unsupported here" if mx is None else f"max|dR|={mx:.2e} (target {tol:.0e})"
    print(f"    {lab:52s} {detail}")

print("\n" + ("ALL VALIDATION PASSED" if fail == 0 else f"{fail} CHECK(S) FAILED"))
sys.exit(1 if fail else 0)
