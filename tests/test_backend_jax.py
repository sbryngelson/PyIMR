"""The traced backend: the program it builds, and its tangents against a difference."""

from typing import Any

import numpy as np
import pytest

import imr_fast
import imr_fast.sensitivity
from imr_fast import _jax
from _validation_support import NHKV, R0, REQ, oldroyd_b, tangent_deviation, zener


SECTION = "7. jax backend"

_TIMES = np.linspace(0.0, 40e-6, 300)

_MAX_BOUND, _MEDIAN_BOUND = 1e-05, 1e-06

_CASES = [(radial, "NHKV", NHKV) for radial in range(1, 7)] + [
  (2, "NoStress", imr_fast.NoStress()),
  (2, "qKV", imr_fast.QuadraticKelvinVoigt(2500.0, 0.1, 0.25)),
  (2, "Zener", zener()),
  (2, "OldroydB", oldroyd_b()),
  (4, "Zener", zener()),
]

_THERMAL_CASES = [
  ("bubtherm fd", dict(bubtherm=1, Nt=17, thermal="fd")),
  ("bubtherm spectral", dict(bubtherm=1, Nt=17, thermal="spectral")),
  ("coupled fd", dict(bubtherm=1, medtherm=1, Nt=13, Mt=13, thermal="fd")),
  ("coupled spectral", dict(bubtherm=1, medtherm=1, Nt=13, Mt=13, thermal="spectral")),
  ("coupled+mass fd", dict(bubtherm=1, vapor=1, masstrans=1, medtherm=1, Nt=13, Mt=13, thermal="fd")),
]

_COVERAGE_CASES = [
  ("gaussian forcing", NHKV, dict(wave_type=1, pA=5e4, TW=5e-6, DT=2e-5)),
  ("heaviside step", NHKV, dict(wave_type=3, pA=5e4, TW=3e-5)),
  ("histotripsy pulse", NHKV, dict(wave_type=2, pA=1e5, omega=2 * np.pi / 2e-5, DT=3e-5, mn=2)),
  ("giesekus", imr_fast.Giesekus(0.1, 80e-6, 16e-6, 0.2, points=12), {}),
  ("linear PTT", imr_fast.LinearPTT(0.1, 80e-6, 16e-6, 0.2, points=12), {}),
  ("giesekus + medtherm", imr_fast.Giesekus(0.1, 80e-6, 16e-6, 0.2, points=12), dict(bubtherm=1, medtherm=1, Nt=9, Mt=9, thermal="fd")),
]

_TANGENT_FIELDS = ("radius_ratio", "radius_m", "wall_velocity_m_s", "internal_pressure_pa", "stress_integral_pa")

def test_jax_sensitivities_refuse_only_what_is_not_a_scalar_field():
  """Everything this test used to list is covered now."""
  times = np.linspace(0.0, 20e-6, 40)
  problem = imr_fast.prepare(imr_fast.SimulationConfig(R0=R0, Req=REQ, material=NHKV))
  with pytest.raises(ValueError, match="finite scalar"):
    imr_fast.sensitivity.solve_with_sensitivities(problem, times, ("initial.stress_state",))
  with pytest.raises(ValueError, match="unknown sensitivity parameter path"):
    imr_fast.sensitivity.solve_with_sensitivities(problem, times, ("physics.not_a_field",))

def test_the_traced_bisection_stays_inside_the_physical_bracket(measured):
  """`_traced_root`'s counterpart to `test_thermal_grid`'s admissibility check."""
  _, jnp, _ = _jax._jax()

  from imr_fast import _thermal

  root = 0.6
  found = float(_thermal._traced_root(lambda kv: (kv - root) * (2.0 + kv), (_thermal._KV_EPS, 1.0 - _thermal._KV_EPS), jnp))
  measured("traced bisection root", f"{found:.15f} vs {root}")
  assert 0.0 < found < 1.0
  assert abs(found - root) < 1e-14

def test_traced_and_brentq_roots_agree_on_the_shipped_closure(measured):
  """The two solvers on one captured wall state, so a disagreement shows up here"""
  _, jnp, _ = _jax._jax()

  from imr_fast import _thermal

  captured = []
  original = _thermal._bracketed_root

  def record(residual, **options):
    if options.get("xp", np) is np: captured.append(residual)
    return original(residual, **options)

  _thermal._bracketed_root = record
  try:
    config = imr_fast.SimulationConfig(R0=R0, Req=REQ, material=NHKV, bubtherm=1, vapor=1, masstrans=1, medtherm=1, Nt=9, Mt=9, thermal="fd")
    imr_fast.simulate(np.linspace(0.0, 20e-6, 60), config)
  finally:
    _thermal._bracketed_root = original

  bracket = (_thermal._KV_EPS, 1.0 - _thermal._KV_EPS)
  worst = 0.0
  for residual in captured[::29]:
    brent = float(original(residual, bracket=bracket))
    traced = float(_thermal._traced_root(residual, bracket, jnp))
    worst = max(worst, abs(brent - traced))
  measured("traced vs brentq root", f"max |dkv| = {worst:.2e} over {len(captured[::29])} states")
  assert worst < 1e-12


_THERMAL_TANGENT_CASES: list[tuple[str, dict[str, Any], float]] = [
  ("bubtherm fd", dict(bubtherm=1, Nt=13, thermal="fd"), 1e-05),
  ("bubtherm spectral", dict(bubtherm=1, Nt=13, thermal="spectral"), 5e-05),
  ("coupled fd", dict(bubtherm=1, medtherm=1, Nt=11, Mt=11, thermal="fd"), 1e-05),
  ("coupled spectral", dict(bubtherm=1, medtherm=1, Nt=11, Mt=11, thermal="spectral"), 3e-04),
  ("coupled+mass fd", dict(bubtherm=1, vapor=1, masstrans=1, medtherm=1, Nt=11, Mt=11, thermal="fd"), 5e-06),
]

_ALL_TANGENT_FIELDS = (*_TANGENT_FIELDS, "bubble_temperature_k", "medium_temperature_k", "vapor_mass_fraction")

_SAMPLED_CASES: list[tuple[str, np.ndarray, dict[str, Any]]] = [
  ("mechanical", np.linspace(0.0, 25e-6, 200), {}),
  ("coupled fd", np.linspace(0.0, 25e-6, 200), dict(bubtherm=1, medtherm=1, Nt=11, Mt=11, thermal="fd")),
  ("past last knot", np.linspace(0.0, 45e-6, 200), {}),
]

_CONFIG_TANGENT_CASES: list[tuple[str, dict[str, Any], tuple[str, ...], float]] = [
  ("mechanical R0", dict(radial=2), ("R0",), 1e-05),
  ("mechanical Req", dict(radial=2), ("Req",), 1e-05),
  ("mechanical pA", dict(radial=2, wave_type=1, pA=5e4, TW=5e-6, DT=2e-5), ("pA",), 1e-05),
  ("mechanical T8 w/ vapor", dict(radial=2, vapor=1), ("T8",), 1e-05),
  ("coupled R0", dict(bubtherm=1, medtherm=1, Nt=11, Mt=11, thermal="fd"), ("R0",), 1e-04),
  ("coupled Req", dict(bubtherm=1, medtherm=1, Nt=11, Mt=11, thermal="fd"), ("Req",), 5e-04),
  ("coupled T8", dict(bubtherm=1, medtherm=1, Nt=11, Mt=11, thermal="fd"), ("T8",), 1e-04),
  ("coupled G and R0", dict(bubtherm=1, medtherm=1, Nt=11, Mt=11, thermal="fd"), ("material.shear_modulus_pa", "R0"), 1e-04),
  ("mass transfer R0", dict(bubtherm=1, vapor=1, masstrans=1, medtherm=1, Nt=11, Mt=11, thermal="fd"), ("R0",), 1e-04),
]

def test_traced_sensitivities_are_compiled_once(measured):
  """`integrate_jax` caches a `jax.jit`; `sensitivities_jax` did not, so every call"""
  from imr_fast import _jax

  times = np.linspace(0.0, 20e-6, 60)
  paths = ("material.shear_modulus_pa",)
  problem = imr_fast.prepare(imr_fast.SimulationConfig(R0=R0, Req=REQ, material=NHKV))
  _jax._COMPILED.clear()
  first = _jax.sensitivities_jax(problem, times, paths)
  after_one = dict(_jax._COMPILED)
  second = _jax.sensitivities_jax(problem, times, paths)
  measured("jax sensitivity cache entries", f"{len(after_one)} after one call, {len(_jax._COMPILED)} after two")
  assert len(after_one) == 1, f"expected one cache entry, got {sorted(after_one)}"
  assert list(_jax._COMPILED) == list(after_one), "a second identical call retraced instead of reusing"
  assert _jax._COMPILED[next(iter(after_one))] is after_one[next(iter(after_one))]
  first_tangent, second_tangent = first[1], second[1]
  assert first_tangent is not None and second_tangent is not None
  assert np.array_equal(first_tangent.derived, second_tangent.derived), "cached call returned a different tangent"

def test_params_branches_only_on_concrete_configuration():
  """Two guards inside `params` tested values that the traced path differentiates."""
  import jax  # noqa: PLC0415  # pyright: ignore[reportMissingImports]

  from imr_fast._prepare import params

  _, jnp, _ = _jax._jax()

  def build(traced):
    p = params(R0, REQ, NHKV, 1, traced[0], 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, imr_fast.PhysicalParameters(), xp=jnp,
               scales=(2500.0, traced[1], 0.0, 0.0, 0.0))
    return jnp.asarray([p["kv0"], p["De"], p["LAM"], p["Pv"], p["chi"]])

  jax.jit(build)(jnp.asarray([298.15, 0.1]))
  jax.jit(jax.jacfwd(build))(jnp.asarray([298.15, 0.1]))


_STRUCTURE_TANGENT_CASES: list[tuple[str, dict[str, Any], tuple[str, ...], float]] = [
  ("mechanical P8", dict(radial=2), ("physics.far_field_pressure_pa",), 1e-05),
  ("mechanical density", dict(radial=2), ("physics.medium_density_kg_m3",), 1e-05),
  ("mechanical surface tension", dict(radial=2), ("physics.surface_tension_n_m",), 1e-05),
  ("mechanical sound speed", dict(radial=2), ("physics.sound_speed_m_s",), 1e-05),
  ("mechanical initial velocity", dict(radial=2), ("initial.wall_velocity_m_s",), 1e-05),
  ("mass transfer conductivity", dict(bubtherm=1, vapor=1, masstrans=1, medtherm=1, Nt=9, Mt=9, thermal="fd"), ("physics.medium_conductivity_w_m_k",), 1e-05),
  ("mass transfer latent heat", dict(bubtherm=1, vapor=1, masstrans=1, medtherm=1, Nt=9, Mt=9, thermal="fd"), ("physics.latent_heat_j_kg",), 1e-05),
  ("mass transfer diffusivity", dict(bubtherm=1, vapor=1, masstrans=1, medtherm=1, Nt=9, Mt=9, thermal="fd"), ("physics.mass_diffusivity_m2_s",), 1e-05),
]

def test_the_traced_path_covers_every_differentiable_scalar_field():
  """The gate on deleting the numpy sensitivity route, asserted rather than assumed."""
  import dataclasses

  config = imr_fast.SimulationConfig(
    R0=R0, Req=REQ, material=NHKV, pA=1e4, wave_type=2, omega=2 * np.pi / 2e-5, DT=3e-5, mn=2.0, TW=1e-5, vapor=1
  )
  candidates = []
  for field in dataclasses.fields(config):
    value = getattr(config, field.name)
    if isinstance(value, (int, float)) and not isinstance(value, bool): candidates.append(field.name)
  for group in ("physics", "initial", "material"):
    for field in dataclasses.fields(getattr(config, group)):
      value = getattr(getattr(config, group), field.name)
      if isinstance(value, (int, float)) and not isinstance(value, bool): candidates.append(f"{group}.{field.name}")

  covered = set(_jax.SCALE_PATHS) | set(_jax.CONFIG_PATHS) | set(_jax.PHYSICS_PATHS) | set(_jax.INITIAL_PATHS)

  accepted = []
  for path in sorted(set(candidates)):
    try:
      imr_fast.sensitivity._normalize_parameters(config, (path,))
    except ValueError:
      continue
    accepted.append(path)
  assert len(accepted) > 20, f"only {len(accepted)} paths accepted; the enumeration has gone stale"
  assert not [p for p in accepted if p not in covered], f"the traced path does not cover {[p for p in accepted if p not in covered]}"


_TANGENT_CASES: list[tuple[str, str, dict[str, Any], str, float]] = [
  ("material G", "material.shear_modulus_pa", dict(radial=2), "radius_ratio", 5e-06),
  ("material mu", "material.viscosity_pa_s", dict(radial=2), "radius_ratio", 5e-06),
  ("R0", "R0", dict(radial=2), "radius_ratio", 5e-06),
  ("Req", "Req", dict(radial=2), "radius_ratio", 5e-06),
  ("pA", "pA", dict(radial=2, wave_type=1, pA=5e4, TW=5e-6, DT=2e-5), "radius_ratio", 5e-05),
  ("physics P8", "physics.far_field_pressure_pa", dict(radial=2), "radius_ratio", 5e-06),
  ("physics density", "physics.medium_density_kg_m3", dict(radial=2), "radius_ratio", 5e-06),
  ("physics surface tension", "physics.surface_tension_n_m", dict(radial=2), "radius_ratio", 5e-06),
  ("initial velocity", "initial.wall_velocity_m_s", dict(radial=2, initial=imr_fast.InitialState(wall_velocity_m_s=-2.0)), "radius_ratio", 5e-06),
  ("bubtherm G", "material.shear_modulus_pa", dict(bubtherm=1, Nt=13, thermal="fd"), "bubble_temperature_k", 5e-05),
  ("coupled T8", "T8", dict(bubtherm=1, medtherm=1, Nt=11, Mt=11, thermal="fd"), "bubble_temperature_k", 5e-04),
  ("coupled medium", "physics.medium_conductivity_w_m_k", dict(bubtherm=1, medtherm=1, Nt=11, Mt=11, thermal="fd"), "medium_temperature_k", 5e-04),
  ("mass transfer G", "material.shear_modulus_pa", dict(bubtherm=1, vapor=1, masstrans=1, medtherm=1, Nt=11, Mt=11, thermal="fd"), "vapor_mass_fraction", 5e-05),
  ("mass transfer latent heat", "physics.latent_heat_j_kg", dict(bubtherm=1, vapor=1, masstrans=1, medtherm=1, Nt=11, Mt=11, thermal="fd"), "radius_ratio", 5e-05),
  ("spectral G", "material.shear_modulus_pa", dict(bubtherm=1, medtherm=1, Nt=11, Mt=11, thermal="spectral"), "radius_ratio", 5e-05),
]

@pytest.mark.parametrize("label,path,options,field,bound", _TANGENT_CASES, ids=[c[0] for c in _TANGENT_CASES])
def test_traced_tangents_match_a_central_difference(label, path, options, field, bound, measured):
  config = imr_fast.SimulationConfig(R0=R0, Req=REQ, material=NHKV, **options)
  deviation = tangent_deviation(config, path, np.linspace(0.0, 15e-6, 60), field)
  measured(f"traced vs difference, {label}", f"{field} rel={deviation:.1e}")
  assert deviation < bound, f"{label}: {deviation:.3e} exceeds {bound:.0e}"


def test_the_collapse_tangent_matches_a_central_difference(measured):
  """Tightened tolerances, because at the default this reads 2.8e-02 and is"""
  config = imr_fast.SimulationConfig(
    R0=R0, Req=REQ, material=zener(), collapse=imr_fast.CollapseInitialization(), rtol=1e-12, atol=1e-12
  )
  deviation = tangent_deviation(config, "Req", np.linspace(0.0, 25e-6, 60), "radius_ratio", relative_step=1e-4)
  measured("traced vs difference, collapse Req", f"rel={deviation:.1e}")
  assert deviation < 1e-05, deviation


def test_the_sampled_forcing_tangent_matches_a_central_difference(measured):
  """A LARGER step than elsewhere, and that is the interesting part: the deviation runs"""
  rng = np.random.default_rng(3)
  knots = np.linspace(0.0, 3e-5, 24)
  history = imr_fast.SampledForcing(
    time_s=tuple(knots), pressure_pa=tuple(6e4 * np.sin(2 * np.pi * knots / 1.5e-5) + 1e4 * rng.standard_normal(knots.size))
  )
  config = imr_fast.SimulationConfig(R0=R0, Req=REQ, material=NHKV, sampled_forcing=history, rtol=1e-10, atol=1e-10)
  worst = {p: tangent_deviation(config, p, np.linspace(0.0, 20e-6, 60), "radius_ratio", relative_step=1e-3)
           for p in ("R0", "physics.far_field_pressure_pa")}
  measured("traced vs difference, sampled forcing", "  ".join(f"{k.split('.')[-1]}={v:.1e}" for k, v in worst.items()))
  assert max(worst.values()) < 1e-03, worst
