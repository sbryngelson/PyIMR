"""Pinned IMRv2 trajectories, collapse initialization, and the deliberate
Mie-Gruneisen divergence.

Numerical content unchanged from `run_validation.py`; see issue #32.
"""

import numpy as np
import pytest

import _imr_thermal
import imr_fast
from _validation_support import NHKV, R0, REQ, T0, deviation, oldroyd_b, reference, reference_times, solve_radius, zener

SECTION = "1. Forward solver vs IMRv2 reference trajectories"

_PINNED_TOLERANCE = 2e-3


def _imr2_cases():
  """The Zener truth case plus three NHKV points spanning the (G, mu) grid."""
  gg, mg = reference("imr2_G.csv"), reference("imr2_M.csv")
  cases = [("Zener truth De=2 s=6", zener(), 0)]
  for k in (0, 30, gg.size * mg.size - 1):
    gi, mi = k // mg.size, k % mg.size
    cases.append((f"NHKV G={gg[gi]:.0f} mu={mg[mi]:.4f}", imr_fast.NeoHookeanKelvinVoigt(gg[gi], mg[mi]), 1 + k))
  return cases


@pytest.mark.parametrize("label,material,column", _imr2_cases(), ids=lambda v: None)
def test_imr2_trajectory(label, material, column, measured):
  times = reference("imr2_t.csv")
  computed = solve_radius(times, material)
  worst = deviation(reference("imr2_s06.csv")[:, column], computed)
  measured(label, f"max|dR|={worst:.2e}")
  assert worst < _PINNED_TOLERANCE


# Section 1b. Every extended feature with a pinned reference: constitutive
# variants, all four forcing shapes, the thermal and mass-transfer branches,
# and the Tait radial models.
_EXTENDED = [
  ("qKV alphax=0.10", dict(material=imr_fast.QuadraticKelvinVoigt(2500.0, 0.1, 0.10)), "ref_qkv_a010.csv"),
  ("qKV alphax=0.25", dict(material=imr_fast.QuadraticKelvinVoigt(2500.0, 0.1, 0.25)), "ref_qkv_a025.csv"),
  ("UCM/OldB De=0.5", dict(material=imr_fast.OldroydB(0.1, 0.5 * T0, 0.1 * T0)), "ref_ucm_De005.csv"),
  ("UCM/OldB De=2.0", dict(material=oldroyd_b()), "ref_ucm_De020.csv"),
  ("Keller-Miksis NHKV", dict(material=NHKV, radial=2), "ref_km_nhkv.csv"),
  ("Keller-Miksis Zener", dict(material=zener(), radial=2), "ref_km_zener.csv"),
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
  ("quadratic Zener", dict(material=imr_fast.QuadraticZener(2500.0, 0.1, 2 * T0, 0.4 * T0, 0.25)), "ref_stress4.csv"),
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
  worst = deviation(reference(reference_file), computed)
  measured(label, f"max|dR|={worst:.2e}")
  assert worst < _PINNED_TOLERANCE


# Section 1c. IMRv2 requires the full coupled model for collapse
# (f_call_params.m:234-236), so these run bubtherm + medtherm + masstrans +
# vapor together.
_FULL = dict(bubtherm=1, medtherm=1, masstrans=1, vapor=1, Nt=25, Mt=25)

# IMRv2 applies a collapse precursor only for stress 3 and 4 (Zener family).
# f_call_params.m:447-460 leaves Szero empty for stress < 3 and returns zeros
# under an explicit "TODO" for stress == 5, so collapse=1 is a no-op for NHKV
# and Oldroyd-B. Those two references therefore pin the fully coupled model
# without a precursor; only the Zener case tests collapse itself.
_IMRV2_SZERO = -0.1600469117114953


def test_collapse_zener(measured):
  config = imr_fast.SimulationConfig(
    R0=R0, Req=REQ, material=zener(), collapse=imr_fast.CollapseInitialization(), **_FULL
  )
  worst = deviation(reference("ref_collapse_zener.csv"), imr_fast.simulate(reference_times(), config).radius_ratio)
  measured("collapse Zener", f"max|dR|={worst:.2e}")
  assert worst < _PINNED_TOLERANCE


def test_collapse_zener_with_upstream_szero(measured):
  """The residual above is entirely one number: the precursor stress at maximum
  radius. imr-fast root-finds the true maximum (velocity == 0) and gets
  S0 = -0.15994511, converged and integrator-independent. IMRv2 takes a discrete
  argmax over ode23tb output points and gets -0.16004691.

  That gap is inherent, not a tolerance choice. R is locally quadratic at the
  peak, so it moves only 2.1e-06 over the 1.9e-03 of nondimensional time that
  separates the two; S is locally linear there (dS/dt = +0.053), so it moves
  1.0e-04. Locating a quadratic maximum by argmax costs O(sqrt(tol)) in position
  and therefore O(sqrt(tol)) in the stress carried out of it.

  Injecting IMRv2's own Szero reproduces the reference in the normal pinned
  band, which proves the coupled solve is exact and isolates the whole
  difference to the precursor.
  """
  config = imr_fast.SimulationConfig(
    R0=R0, Req=REQ, material=zener(), initial=imr_fast.InitialState(stress_state=(_IMRV2_SZERO,)), **_FULL
  )
  worst = deviation(reference("ref_collapse_zener.csv"), imr_fast.simulate(reference_times(), config).radius_ratio)
  measured("collapse Zener w/ IMRv2 Szero", f"max|dR|={worst:.2e}")
  assert worst < 5e-5


@pytest.mark.parametrize(
  "label,material_factory,reference_file",
  [("Oldroyd-B", oldroyd_b, "ref_coupled_oldb.csv"), ("NHKV", lambda: NHKV, "ref_coupled_nhkv.csv")],
  ids=["oldroyd-b", "nhkv"],
)
def test_coupled_without_precursor(label, material_factory, reference_file, measured):
  config = imr_fast.SimulationConfig(R0=R0, Req=REQ, material=material_factory(), **_FULL)
  worst = deviation(reference(reference_file), imr_fast.simulate(reference_times(), config).radius_ratio)
  measured(f"coupled {label}", f"max|dR|={worst:.2e}")
  assert worst < _PINNED_TOLERANCE


def test_memoryless_collapse_is_refused():
  """imr-fast refuses collapse without memory; IMRv2 accepts the flag and
  ignores it. Refusing is the stricter, correct behaviour."""
  with pytest.raises(ValueError):
    # The refusal is a configuration error, so it fires on construction --
    # keep both statements inside the assertion.
    imr_fast.simulate(
      reference_times(),
      imr_fast.SimulationConfig(R0=R0, Req=REQ, material=NHKV, collapse=imr_fast.CollapseInitialization(), **_FULL),
    )


# Section 1d. IMRv2's Mie-Gruneisen branch takes the wrong root of its own
# density quadratic. a*mu^2 + b*mu + A = 0 has roots that tend to 0 and to
# -1/nog as A -> 0; upstream takes (-b + sqrt(d))/(2a), which is the -1/nog
# branch -- a 32.5% density deficit at ambient pressure. It also omits the
# stress term from Pb, which radial=3 and 4 both include.
#
# Both are corrected here, so radial=5 no longer reproduces ref_radial5.csv.
# The checks below are what justify that divergence.


@pytest.fixture(scope="module")
def mie_parameters():
  p = imr_fast.params(R0, REQ, NHKV, 0, 298.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, imr_fast.PhysicalParameters())
  return p, p["Cstar"], p["hugoniot_slope"], p["nog"]


def test_mie_density_undisturbed_at_ambient(mie_parameters, measured):
  _, sound, slope, nog = mie_parameters
  mu = _imr_thermal._mu_of_A(1.0 / sound**2, slope, nog)
  measured("rho/rho0 - 1 at ambient", f"{mu:.3e}")
  assert abs(mu) < 1e-4


def test_mie_sound_speed_recovers_c0(mie_parameters, measured):
  """The EoS calibration point."""
  p, sound, slope, nog = mie_parameters
  computed, _, _ = _imr_thermal._mie_gruneisen(1.0, sound, slope, nog, p["mie_reference"])
  error = abs(float(computed) - sound) / sound
  measured("c(ambient)/c0 - 1", f"{error:.3e}")
  assert error < 1e-3


def test_mie_enthalpy_weakly_compressible_limit(mie_parameters, measured):
  """Enthalpy matches its weakly-compressible limit h ~ P - 1."""
  p, sound, slope, nog = mie_parameters
  worst = 0.0
  for pressure in (2.0, 10.0, 100.0):
    _, enthalpy, _ = _imr_thermal._mie_gruneisen(pressure, sound, slope, nog, p["mie_reference"])
    worst = max(worst, abs(float(enthalpy) - (pressure - 1.0)) / (pressure - 1.0))
  measured("max |h - (P-1)|/(P-1)", f"{worst:.3e}")
  assert worst < 5e-3


@pytest.fixture(scope="module")
def radial_trajectories():
  return {
    n: imr_fast.simulate(
      reference_times(), imr_fast.SimulationConfig(R0=R0, Req=REQ, material=NHKV, radial=n)
    ).radius_ratio
    for n in (2, 3, 4, 5, 6)
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
  """The corrected branches must agree with the independent Tait forms to
  within the spread the Tait forms already have among themselves."""
  worst = deviation(radial_trajectories[left], radial_trajectories[right])
  spread = deviation(radial_trajectories[3], radial_trajectories[4])
  measured(label, f"{worst:.2e}  (reference spread radial=3 vs 4: {spread:.2e})")
  assert worst < tolerance


def test_radial6_is_finite_and_real(radial_trajectories):
  """radial=6 is unavailable upstream at all; assert it runs clean here."""
  trajectory = radial_trajectories[6]
  assert np.all(np.isfinite(trajectory)) and np.isrealobj(trajectory)
