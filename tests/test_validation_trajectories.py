"""Pinned IMRv2 trajectories, collapse initialization, and the deliberate"""

from typing import Any

import numpy as np
import pytest

from pyimr import _thermal
import pyimr
from _validation_support import NHKV, R0, REQ, T0, deviation, median_deviation, oldroyd_b, reference, reference_times, solve_radius, zener

SECTION = "1. Forward solver vs IMRv2 reference trajectories"

_PINNED_TOLERANCE = 2e-3

_PINNED_MEDIAN_BOUNDS = {
  "Zener truth De=2 s=6": 7e-06,
  "NHKV grid": 7e-06,
  "qKV alphax=0.10": 4e-07,
  "qKV alphax=0.25": 4e-07,
  "UCM/OldB De=0.5": 6e-06,
  "UCM/OldB De=2.0": 6e-06,
  "Keller-Miksis NHKV": 2e-07,
  "Keller-Miksis Zener": 7e-06,
  "Gaussian forcing pA=5e4": 7e-07,
  "Gaussian forcing pA=2e5": 3e-07,
  "constant offset pA=3e4": 4e-07,
  "Heaviside step pA=5e4": 4e-07,
  "histotripsy pulse": 5e-07,
  "vapor=1 (T=298.15K)": 4e-07,
  "bubtherm=1 (thermal PDE)": 8e-06,
  "medtherm=1 (liquid layer)": 8e-06,
  "masstrans=1+medtherm=1 (coupled)": 7e-06,
  "no constitutive stress": 7e-06,
  "quadratic Zener": 6e-06,
  "radial=3 (KM enthalpy, Tait)": 2e-07,
  "radial=4 (Gilmore, Tait)": 2e-07,
  "radial=3+Zener": 7e-06,
  "radial=4+Zener": 8e-06,
  "coupled Oldroyd-B": 5e-06,
  "coupled NHKV": 7e-06,
}


def _median_bound(label):
  if label.startswith("NHKV G="):
    return _PINNED_MEDIAN_BOUNDS["NHKV grid"]
  bound = _PINNED_MEDIAN_BOUNDS.get(label)
  assert bound is not None, f"no median bound recorded for pinned case {label!r}"
  return bound


def _imr2_cases():
  gg, mg = reference("imr2_G.csv"), reference("imr2_M.csv")
  cases = [("Zener truth De=2 s=6", zener(), 0)]
  for k in (0, 30, gg.size * mg.size - 1):
    gi, mi = k // mg.size, k % mg.size
    cases.append((f"NHKV G={gg[gi]:.0f} mu={mg[mi]:.4f}", pyimr.NeoHookeanKelvinVoigt(gg[gi], mg[mi]), 1 + k))
  return cases


@pytest.mark.parametrize("label,material,column", _imr2_cases(), ids=lambda v: None)
def test_imr2_trajectory(label, material, column, measured):
  times = reference("imr2_t.csv")
  computed = solve_radius(times, material)
  upstream = reference("imr2_s06.csv")[:, column]
  worst, typical = deviation(upstream, computed), median_deviation(upstream, computed)
  measured(label, f"max|dR|={worst:.2e}  median={typical:.2e}")
  assert worst < _PINNED_TOLERANCE
  assert typical < _median_bound(label)


_EXTENDED: list[tuple[str, dict[str, Any], str]] = [
  ("qKV alphax=0.10", dict(material=pyimr.QuadraticKelvinVoigt(2500.0, 0.1, 0.10)), "ref_qkv_a010.csv"),
  ("qKV alphax=0.25", dict(material=pyimr.QuadraticKelvinVoigt(2500.0, 0.1, 0.25)), "ref_qkv_a025.csv"),
  ("UCM/OldB De=0.5", dict(material=pyimr.OldroydB(0.1, 0.5 * T0, 0.1 * T0)), "ref_ucm_De005.csv"),
  ("UCM/OldB De=2.0", dict(material=oldroyd_b()), "ref_ucm_De020.csv"),
  ("Keller-Miksis NHKV", dict(material=NHKV, radial=2), "ref_km_nhkv.csv"),
  ("Keller-Miksis Zener", dict(material=zener(), radial=2), "ref_km_zener.csv"),
  ("Gaussian forcing pA=5e4", dict(wave_type=1, pA=5e4, TW=5e-6, DT=2e-5), "ref_gauss_pA50.csv"),
  ("Gaussian forcing pA=2e5", dict(wave_type=1, pA=2e5, TW=5e-6, DT=2e-5), "ref_gauss_pA200.csv"),
  ("constant offset pA=3e4", dict(wave_type=0, pA=3e4), "ref_imp_pA30.csv"),
  ("Heaviside step pA=5e4", dict(wave_type=3, pA=5e4, TW=3e-5), "ref_heav_pA50.csv"),
  ("histotripsy pulse", dict(wave_type=2, pA=1e5, omega=2 * np.pi / 2e-5, DT=3e-5, mn=2), "ref_histo.csv"),
  ("vapor=1 (T=298.15K)", dict(vapor=1, T8=298.15), "ref_vapor.csv"),
  ("bubtherm=1 (thermal PDE)", dict(bubtherm=1, Nt=25, thermal="fd"), "ref_bubtherm.csv"),
  ("medtherm=1 (liquid layer)", dict(bubtherm=1, medtherm=1, Nt=25, Mt=25, thermal="fd"), "ref_medtherm.csv"),
  ("masstrans=1+medtherm=1 (coupled)", dict(bubtherm=1, vapor=1, masstrans=1, medtherm=1, Nt=25, Mt=25, thermal="fd"), "ref_masstrans_medtherm.csv"),
  ("no constitutive stress", dict(material=pyimr.NoStress()), "ref_stress0.csv"),
  ("quadratic Zener", dict(material=pyimr.QuadraticZener(2500.0, 0.1, 2 * T0, 0.4 * T0, 0.25)), "ref_stress4.csv"),
  ("radial=3 (KM enthalpy, Tait)", dict(radial=3), "ref_radial3.csv"),
  ("radial=4 (Gilmore, Tait)", dict(radial=4), "ref_radial4.csv"),
  ("radial=3+Zener", dict(radial=3, material=zener()), "ref_radial3_zener.csv"),
  ("radial=4+Zener", dict(radial=4, material=zener()), "ref_radial4_zener.csv"),
]


@pytest.mark.parametrize("label,options,reference_file", _EXTENDED, ids=[case[0] for case in _EXTENDED])
def test_extended_feature_trajectory(label, options, reference_file, measured):
  options = dict(options)
  material = options.pop("material", NHKV)
  computed = solve_radius(reference_times(), material, **options)
  upstream = reference(reference_file)
  worst, typical = deviation(upstream, computed), median_deviation(upstream, computed)
  measured(label, f"max|dR|={worst:.2e}  median={typical:.2e}")
  assert worst < _PINNED_TOLERANCE
  assert typical < _median_bound(label)


_FULL: dict[str, Any] = dict(bubtherm=1, medtherm=1, masstrans=1, vapor=1, Nt=25, Mt=25, thermal="fd")

_IMRV2_SZERO = -0.1600469117114953


def test_collapse_zener(measured):
  config = pyimr.SimulationConfig(R0=R0, Req=REQ, material=zener(), collapse=pyimr.CollapseInitialization(), **_FULL)
  computed = pyimr.simulate(reference_times(), config).radius_ratio
  upstream = reference("ref_collapse_zener.csv")
  worst, typical = deviation(upstream, computed), median_deviation(upstream, computed)
  measured("collapse Zener", f"max|dR|={worst:.2e}  median={typical:.2e}")
  assert worst < _PINNED_TOLERANCE


def test_collapse_zener_with_upstream_szero(measured):
  """The residual above is entirely one number: the precursor stress at maximum"""
  config = pyimr.SimulationConfig(R0=R0, Req=REQ, material=zener(), initial=pyimr.InitialState(stress_state=(_IMRV2_SZERO,)), **_FULL)
  worst = deviation(reference("ref_collapse_zener.csv"), pyimr.simulate(reference_times(), config).radius_ratio)
  measured("collapse Zener w/ IMRv2 Szero", f"max|dR|={worst:.2e}")
  assert worst < 2e-4


@pytest.mark.parametrize(
  "label,material_factory,reference_file",
  [("Oldroyd-B", oldroyd_b, "ref_coupled_oldb.csv"), ("NHKV", lambda: NHKV, "ref_coupled_nhkv.csv")],
  ids=["oldroyd-b", "nhkv"],
)
def test_coupled_without_precursor(label, material_factory, reference_file, measured):
  config = pyimr.SimulationConfig(R0=R0, Req=REQ, material=material_factory(), **_FULL)
  computed = pyimr.simulate(reference_times(), config).radius_ratio
  upstream = reference(reference_file)
  worst, typical = deviation(upstream, computed), median_deviation(upstream, computed)
  measured(f"coupled {label}", f"max|dR|={worst:.2e}  median={typical:.2e}")
  assert worst < _PINNED_TOLERANCE
  assert typical < _median_bound(f"coupled {label}")


def test_memoryless_collapse_is_refused():
  """PyIMR refuses collapse without memory; IMRv2 accepts the flag and"""
  with pytest.raises(ValueError):
    pyimr.simulate(
      reference_times(), pyimr.SimulationConfig(R0=R0, Req=REQ, material=NHKV, collapse=pyimr.CollapseInitialization(), **_FULL)
    )


def test_masstrans_diverges_from_upstream_by_the_kirchhoff_correction(measured):
  """Bounded, so a future regression cannot hide inside an expected divergence."""
  options: dict[str, Any] = dict(bubtherm=1, vapor=1, masstrans=1, Nt=25, thermal="fd")
  computed = solve_radius(reference_times(), NHKV, **options)
  upstream = reference("ref_masstrans.csv")
  worst, typical = deviation(upstream, computed), median_deviation(upstream, computed)
  measured("masstrans vs upstream (#75)", f"max|dR|={worst:.2e}  median={typical:.2e}")
  assert worst < 1e-3, "divergence from upstream grew beyond the Kirchhoff correction"
  assert typical > 1e-5, "no divergence at all -- is the correction still applied?"


@pytest.fixture(scope="module")
def mie_parameters():
  p = pyimr.params(R0, REQ, NHKV, 0, 298.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, pyimr.PhysicalParameters())
  return p, p["Cstar"], p["hugoniot_slope"], p["nog"]


def test_mie_density_undisturbed_at_ambient(mie_parameters, measured):
  _, sound, slope, nog = mie_parameters
  mu = _thermal._mu_of_A(1.0 / sound**2, slope, nog)
  measured("rho/rho0 - 1 at ambient", f"{mu:.3e}")
  assert abs(mu) < 1e-4


def test_mie_sound_speed_recovers_c0(mie_parameters, measured):
  """The EoS calibration point."""
  p, sound, slope, nog = mie_parameters
  computed, _, _ = _thermal._mie_gruneisen(1.0, sound, slope, nog, p["mie_reference"])
  error = abs(float(computed) - sound) / sound
  measured("c(ambient)/c0 - 1", f"{error:.3e}")
  assert error < 1e-3


def test_mie_enthalpy_weakly_compressible_limit(mie_parameters, measured):
  """Enthalpy matches its weakly-compressible limit h ~ P - 1."""
  p, sound, slope, nog = mie_parameters
  worst = 0.0
  for pressure in (2.0, 10.0, 100.0):
    _, enthalpy, _ = _thermal._mie_gruneisen(pressure, sound, slope, nog, p["mie_reference"])
    worst = max(worst, abs(float(enthalpy) - (pressure - 1.0)) / (pressure - 1.0))
  measured("max |h - (P-1)|/(P-1)", f"{worst:.3e}")
  assert worst < 5e-3


@pytest.fixture(scope="module")
def radial_trajectories():
  return {
    n: pyimr.simulate(reference_times(), pyimr.SimulationConfig(R0=R0, Req=REQ, material=NHKV, radial=n)).radius_ratio for n in (2, 3, 4, 5, 6)
  }


@pytest.mark.parametrize(
  "label,left,right,tolerance",
  [
    ("radial=5 (KM/Mie-G) vs radial=3 (KM/Tait)", 5, 3, 2e-3),
    ("radial=6 (Gilmore/Mie-G) vs radial=4 (Gilmore/Tait)", 6, 4, 3e-2),
    ("radial=6 vs radial=5 (Gilmore vs KM, same EoS)", 6, 5, 3e-2),
  ],
  ids=["5-vs-3", "6-vs-4", "6-vs-5"],
)
def test_corrected_mie_agrees_with_tait(label, left, right, tolerance, radial_trajectories, measured):
  """The corrected branches must agree with the independent Tait forms to"""
  worst = deviation(radial_trajectories[left], radial_trajectories[right])
  spread = deviation(radial_trajectories[3], radial_trajectories[4])
  measured(label, f"{worst:.2e}  (reference spread radial=3 vs 4: {spread:.2e})")
  assert worst < tolerance


def test_radial6_is_finite_and_real(radial_trajectories):
  """radial=6 is unavailable upstream at all; assert it runs clean here."""
  trajectory = radial_trajectories[6]
  assert np.all(np.isfinite(trajectory)) and np.isrealobj(trajectory)
