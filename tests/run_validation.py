"""Full validation suite. Run: python3 tests/run_validation.py"""

import os
import sys
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import imr_fast
from imr_fast import params
from imr_inference import (
  InferenceParameter,
  RadiusObservation,
  prepare_inference,
)


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
  (
    "Zener truth De=2 s=6",
    solve_radius(
      _t2,
      _R0,
      _R0 / 6,
      imr_fast.Zener(2500.0, 0.1, 2 * _t0, 0.4 * _t0),
    ),
    _A[:, 0],
  )
]
for _k in [0, 30, len(_Gg) * len(_Mg) - 1]:
  _gi, _mi = _k // len(_Mg), _k % len(_Mg)
  _checks.append(
    (
      f"NHKV G={_Gg[_gi]:.0f} mu={_Mg[_mi]:.4f}",
      solve_radius(
        _t2,
        _R0,
        _R0 / 6,
        imr_fast.NeoHookeanKelvinVoigt(_Gg[_gi], _Mg[_mi]),
      ),
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
  ("radial=5 (KM enthalpy, Mie-Gruneisen)", dict(radial=5), "ref_radial5.csv"),
]:
  ml = np.loadtxt(f"{_d}/{ref}")
  material = kw.pop("material", imr_fast.NeoHookeanKelvinVoigt(2500.0, 0.1))
  py = solve_radius(_t, _R0, _R0 / 6, material, **kw)
  mx = np.nanmax(np.abs(ml - py))
  ok = mx < 2e-3
  fail += not ok
  print(f"    {lab:24s} max|dR|={mx:.2e}  {'PASS' if ok else 'FAIL'}")

print("\n" + "=" * 64)
print("2. CONSTITUTIVE SUITE")
print("=" * 64)


def _instantaneous_values(material, radius=0.5, velocity=-0.3, need_rate=True):
  config = imr_fast.SimulationConfig(R0=_R0, Req=_R0 / 6, material=material)
  problem = imr_fast.prepare(config)
  return imr_fast._stress(
    material,
    problem.parameters,
    radius,
    velocity,
    None,
    problem.instantaneous_material,
    need_rate,
  )


print("  composable NH/Newtonian trajectory vs closed-form fast path")
_generic_nh = imr_fast.InstantaneousMaterial(imr_fast.NeoHookean(2500.0), imr_fast.Newtonian(0.1))
_equivalence_options = dict(rtol=1e-10, atol=1e-12)
_trajectory_tolerance = 1e-7
for _radial in (1, 2):
  _closed = solve_radius(
    _t,
    _R0,
    _R0 / 6,
    imr_fast.NeoHookeanKelvinVoigt(2500.0, 0.1),
    radial=_radial,
    **_equivalence_options,
  )
  _generic = solve_radius(_t, _R0, _R0 / 6, _generic_nh, radial=_radial, **_equivalence_options)
  _mx = np.max(np.abs(_generic - _closed))
  _ok = _mx < _trajectory_tolerance
  fail += not _ok
  print(f"    radial={_radial} max|dR|={_mx:.2e}  {'PASS' if _ok else 'FAIL'}")
_thermal_options = dict(bubtherm=1, medtherm=1, Nt=9, Mt=9, **_equivalence_options)
_closed = solve_radius(
  _t,
  _R0,
  _R0 / 6,
  imr_fast.NeoHookeanKelvinVoigt(2500.0, 0.1),
  **_thermal_options,
)
_generic = solve_radius(
  _t,
  _R0,
  _R0 / 6,
  _generic_nh,
  **_thermal_options,
)
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
  solve_radius(
    _t[:3],
    _R0,
    _R0 / 6,
    imr_fast.InstantaneousMaterial(elastic=imr_fast.Gent(2500.0, 5.0)),
  )
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
_ucm_km = solve_radius(
  _tv,
  225e-6,
  225e-6 / 6,
  _ucm_material,
  radial=2,
)
_distributed_km = solve_radius(
  _tv,
  225e-6,
  225e-6 / 6,
  imr_fast.Giesekus(0.1, _relaxation, _retardation),
  radial=2,
)
_mx = np.nanmax(np.abs(_distributed_km - _ucm_km))
_ok = _mx < 2e-3
fail += not _ok
print(f"    {'KM Giesekus':>10} -> UCM   max|dR|={_mx:.2e}  {'PASS' if _ok else 'FAIL'}")
_coupled_options = dict(
  bubtherm=1,
  medtherm=1,
  vapor=1,
  masstrans=1,
  Nt=9,
  Mt=9,
)
_ucm_coupled = solve_radius(
  _tv,
  225e-6,
  225e-6 / 6,
  _ucm_material,
  **_coupled_options,
)
_distributed_coupled = solve_radius(
  _tv,
  225e-6,
  225e-6 / 6,
  imr_fast.Giesekus(0.1, _relaxation, _retardation),
  **_coupled_options,
)
_mx = np.nanmax(np.abs(_distributed_coupled - _ucm_coupled))
_ok = _mx < 3e-3
fail += not _ok
print(f"    {'coupled':>10} -> UCM   max|dR|={_mx:.2e}  {'PASS' if _ok else 'FAIL'}")
print("  nonlinear parameter must produce distinct physics")
for _name, _model in [
  (
    "giesekus",
    imr_fast.Giesekus(0.1, _relaxation, _retardation, mobility=0.2),
  ),
  (
    "linear PTT",
    imr_fast.LinearPTT(0.1, _relaxation, _retardation, extensibility=0.2),
  ),
]:
  _R = solve_radius(_tv, 225e-6, 225e-6 / 6, _model)
  _mx = np.nanmax(np.abs(_R - _ucm))
  _ok = _mx > 0.05
  fail += not _ok
  print(f"    {_name:>10} parameter=0.2  max|dR| vs UCM={_mx:.2e}  {'PASS' if _ok else 'FAIL'}")

print("\n" + "=" * 64)
print("3. UNIFIED FORWARD SENSITIVITIES")
print("=" * 64)
_sensitivity_times = np.linspace(0.0, 20e-6, 80)


def _material_offset(config, field, amount):
  return replace(
    config,
    material=replace(
      config.material,
      **{field: getattr(config.material, field) + amount},
    ),
  )


def _centered_output(times, config, field, step, output):
  plus = output(imr_fast.simulate(times, _material_offset(config, field, step)))
  minus = output(imr_fast.simulate(times, _material_offset(config, field, -step)))
  return (plus - minus) / (2.0 * step)


for _radial in range(1, 6):
  _config = imr_fast.SimulationConfig(
    _R0,
    _R0 / 6,
    imr_fast.NeoHookeanKelvinVoigt(2500.0, 0.1),
    radial=_radial,
    rtol=1e-10,
    atol=1e-12,
  )
  _sensitivity = imr_fast.simulate_with_sensitivities(
    _sensitivity_times,
    _config,
    ["material.shear_modulus_pa"],
  ).radius_ratio[:, 0]
  # A one-percent material step stays in the centered-difference regime
  # while remaining above radial=5's adaptive-integration noise floor.
  _step = 25.0
  _difference = _centered_output(
    _sensitivity_times,
    _config,
    "shear_modulus_pa",
    _step,
    lambda result: result.radius_ratio,
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
  _coupled_times,
  _coupled_config,
  ["material.shear_modulus_pa"],
)
_step = 0.025
_temperature_difference = _centered_output(
  _coupled_times,
  _coupled_config,
  "shear_modulus_pa",
  _step,
  lambda result: result.medium_temperature_k,
)
_relative = np.linalg.norm(
  _coupled_sensitivity.medium_temperature_k[..., 0] - _temperature_difference
) / np.linalg.norm(_temperature_difference)
_ok = _relative < 2e-3
fail += not _ok
print(f"    medium temperature rel={_relative:.2e}  {'PASS' if _ok else 'FAIL'}")

print("  collapse shooting tangent")
_collapse_config = imr_fast.SimulationConfig(
  _R0,
  _R0 / 6,
  imr_fast.Zener(2500.0, 0.1, 40e-6, 8e-6),
  radial=2,
  collapse=imr_fast.CollapseInitialization(),
)
_collapse_sensitivity = imr_fast.simulate_with_sensitivities(
  np.array([0.0, 1e-8]),
  _collapse_config,
  ["material.shear_modulus_pa"],
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
print("4. PREPARED INFERENCE")
print("=" * 64)
_inference_config = imr_fast.SimulationConfig(
  _R0,
  _R0 / 6,
  imr_fast.NeoHookeanKelvinVoigt(2500.0, 0.1),
)
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

print("\n" + ("ALL VALIDATION PASSED" if fail == 0 else f"{fail} CHECK(S) FAILED"))
sys.exit(1 if fail else 0)
