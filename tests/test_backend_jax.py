"""The traced backend: the program it builds, and its tangents against a difference.

This module used to compare two implementations. W11 stage 5 left one, so the tests that
had scipy for a reference are replaced below by comparison against a CENTRAL DIFFERENCE
of the forward solve.

That is a real reduction in strength, stated rather than glossed. A difference cannot
distinguish two already-correct routes at 1e-13, which is what the deleted `Dual`
reference did. It resolves 1e-09 to 1e-04 here -- and every defect this suite has caught
was larger by two orders or more. Missing terms are what the tests are for, and a
difference finds those.

The rest of the module is about the traced program itself -- its bracket, its cache, its
concrete branches -- and never needed a second implementation.
"""

from typing import Any

import numpy as np
import pytest

import imr_fast
import imr_fast.sensitivity
from imr_fast import _jax
from _validation_support import NHKV, R0, REQ, oldroyd_b, tangent_deviation, zener


SECTION = "7. jax backend"

_TIMES = np.linspace(0.0, 40e-6, 300)

# scipy itself sits at max 8.6e-06 / median 3.1e-08 against the pinned IMRv2
# trajectories, so requiring the two backends to agree an order of magnitude
# INSIDE that is requiring them to agree better than either agrees with the
# reference. Measured worst across these cases: 2.9e-06 max, 2.1e-07 median.
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
  """Everything this test used to list is covered now.

  `bubtherm=1` went in the thermal work, `R0` with the config scalars, and
  `physics.*`/`initial.*` with `_Overridden`. What is left refuses for a reason that
  is not about jax: a path must name one finite SCALAR, so `initial.stress_state` --
  a sequence -- is out on both routes, and an unknown path is a typo.
  """
  times = np.linspace(0.0, 20e-6, 40)
  problem = imr_fast.prepare(imr_fast.SimulationConfig(R0=R0, Req=REQ, material=NHKV))
  with pytest.raises(ValueError, match="finite scalar"):
    imr_fast.sensitivity.solve_with_sensitivities(problem, times, ("initial.stress_state",))
  with pytest.raises(ValueError, match="unknown sensitivity parameter path"):
    imr_fast.sensitivity.solve_with_sensitivities(problem, times, ("physics.not_a_field",))

def test_the_traced_bisection_stays_inside_the_physical_bracket(measured):
  """`_traced_root`'s counterpart to `test_thermal_grid`'s admissibility check.

  Bisection cannot leave its bracket, so `kv` in `(0, 1)` is structural here in a
  way it is not for a local iteration -- but the polish steps that follow are
  Newton, and Newton can. Asserted on the traced path specifically, because the
  numpy test cannot reach this solver.
  """
  # `_jax._jax()` rather than a bare `import jax.numpy`: it is what enables x64,
  # and without it this runs in float32 and lands 2.4e-08 off -- which is exactly
  # what this assertion caught the first time it was written.
  _, jnp, _ = _jax._jax()

  from imr_fast import _thermal

  # A residual with the same shape as the wall closure's: one root, steep near
  # the upper end. `_traced_root` is handed jnp so it takes the traced branch.
  root = 0.6
  found = float(_thermal._traced_root(lambda kv: (kv - root) * (2.0 + kv), (_thermal._KV_EPS, 1.0 - _thermal._KV_EPS), jnp))
  measured("traced bisection root", f"{found:.15f} vs {root}")
  assert 0.0 < found < 1.0
  assert abs(found - root) < 1e-14

def test_traced_and_brentq_roots_agree_on_the_shipped_closure(measured):
  """The two solvers on one captured wall state, so a disagreement shows up here
  rather than as trajectory drift it would be easy to blame on the integrator."""
  _, jnp, _ = _jax._jax()

  from imr_fast import _thermal

  captured = []
  original = _thermal._bracketed_root

  def record(residual, **options):
    # Only the NUMPY calls. The forward solve is traced, so its residuals close over
    # tracers and `brentq` on one leaks. `_thermal_outputs` still runs the closure in
    # numpy when it builds a result's temperatures, which is where the concrete wall
    # states come from -- and both routines stay live, `brentq` for the outputs and
    # `_traced_root` inside the solve, so the comparison is still worth having.
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


# Thermal forward sensitivities. `sensitivities_jax` built its own `_rhs` argument
# tuple with every thermal slot zeroed, so these were mechanical-only and the
# `bubtherm=1` refusal in sensitivity.py stood in front of that. Both are gone:
# `_rhs_args` is now one definition shared with the forward path.
#
# Bounds are per case at roughly 5x the measured value, on the precedent argued
# in test_validation_trajectories: a single bound is set by the worst case and
# throws away the sensitivity of every other one. Here the worst is the spectral
# medium temperature, 70x the best case, and it is INTEGRATOR error rather than a
# tangent discrepancy -- test_jax_thermal_tangents_converge below is what
# establishes that, and it is the property-level claim.

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
  # Past the last knot, where the mask has to zero the forcing rather than
  # extrapolate the clamped cubic it still evaluates.
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
  """`integrate_jax` caches a `jax.jit`; `sensitivities_jax` did not, so every call
  retraced the whole `jacfwd`. Measured at 1121 ms against the compiled numba
  route's 47 ms -- 24x slower for no reason other than the missing cache, which is
  the opposite of the ordering the migration assumes.

  Asserted structurally rather than by timing: a wall-clock threshold on a shared
  machine is a flaky test. What matters is that a second identical call adds no
  cache entry and reuses the first.
  """
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
  # And the compiled pair really is the same object, not an equal-keyed rebuild.
  assert _jax._COMPILED[next(iter(after_one))] is after_one[next(iter(after_one))]
  assert np.array_equal(first[1].derived, second[1].derived), "cached call returned a different tangent"

def test_params_branches_only_on_concrete_configuration():
  """Two guards inside `params` tested values that the traced path differentiates.

  `Pv_star > 0` is really asking whether vapour is on -- `pvsat` is an exponential
  -- and `lam1 > 0` whether the material HAS a relaxation time, which its type
  fixes. Both worked under `jacfwd`, which keeps primals concrete, and both raised
  `TracerBoolConversionError` under `jit`. Rewritten to read `vapor` and the
  concrete scales, so this is what stops them coming back.
  """
  import jax  # noqa: PLC0415  # pyright: ignore[reportMissingImports]

  from imr_fast._prepare import params

  _, jnp, _ = _jax._jax()

  def build(traced):
    p = params(R0, REQ, NHKV, 1, traced[0], 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, imr_fast.PhysicalParameters(), xp=jnp,
               scales=(2500.0, traced[1], 0.0, 0.0, 0.0))
    return jnp.asarray([p["kv0"], p["De"], p["LAM"], p["Pv"], p["chi"]])

  # jit, not just jacfwd: the abstract trace is what the concrete-branch rewrite
  # was for, and jacfwd alone would pass even with the old code.
  jax.jit(build)(jnp.asarray([298.15, 0.1]))
  jax.jit(jax.jacfwd(build))(jnp.asarray([298.15, 0.1]))


# Structure-valued parameters: `physics.*` and `initial.*`. These are read off a
# dataclass rather than passed as arguments, and tracing one cannot be done by
# building a new dataclass -- `__post_init__` validates with `np.isfinite`, which
# converts a tracer. `_jax._Overridden` substitutes at attribute access instead, so
# the twenty-five `physics` reads in `params` and the five `initial` reads in
# `initial_state_vector` need no rewriting.

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
  """The gate on deleting the numpy sensitivity route, asserted rather than assumed.

  Enumerates the scalar fields of the config, `physics`, `initial` and the material,
  keeps the ones the scipy route actually accepts, and requires the traced path to
  cover all of them. `mn` was the last holdout and was found by this check, not by
  reading the code.
  """
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

  # `_normalize_parameters` is what decides acceptance, so a solve per candidate buys
  # nothing. The direction that catches the bug is unchanged: a path the front end
  # accepts but the traced substitution does not know would be silently ignored.
  accepted = []
  for path in sorted(set(candidates)):
    try:
      imr_fast.sensitivity._normalize_parameters(config, (path,))
    except ValueError:
      continue
    accepted.append(path)
  assert len(accepted) > 20, f"only {len(accepted)} paths accepted; the enumeration has gone stale"
  assert not [p for p in accepted if p not in covered], f"the traced path does not cover {[p for p in accepted if p not in covered]}"


# The tangent coverage the deleted `Dual` reference provided, against a central
# difference instead. Bounds are per case at roughly five times the measured deviation,
# because a difference's accuracy is case-dependent and one bound would be set by the
# worst. Every bound is at least two orders below the smallest defect this suite has
# caught, which is what lets them fail for the right reason.
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
  """Tightened tolerances, because at the default this reads 2.8e-02 and is
  STEP-INDEPENDENT -- shrinking the difference does not move it, so it is the
  integrators' error rather than the differencing's, and the Zener collapse amplifies
  it. At 1e-12 it is 1.1e-06. The traced path once received the collapse stress state as
  a constant, giving a tangent of zero, which a difference resolves trivially."""
  config = imr_fast.SimulationConfig(
    R0=R0, Req=REQ, material=zener(), collapse=imr_fast.CollapseInitialization(), rtol=1e-12, atol=1e-12
  )
  deviation = tangent_deviation(config, "Req", np.linspace(0.0, 25e-6, 60), "radius_ratio", relative_step=1e-4)
  measured("traced vs difference, collapse Req", f"rel={deviation:.1e}")
  assert deviation < 1e-05, deviation


def test_the_sampled_forcing_tangent_matches_a_central_difference(measured):
  """A LARGER step than elsewhere, and that is the interesting part: the deviation runs
  4.6e-04 at a 1e-03 step, 1.6e-02 at 1e-05 and 2.6e+00 at 1e-07. Inverted, because the
  forcing is a piecewise cubic and shrinking the step does not shrink the error from
  knots crossing evaluation points -- it only shrinks the signal.

  This is the one place in the suite where a difference is close to unable to do the
  job: 1.8e-04 against a defect that was 3.1e-02, a 170x margin rather than the usual
  10^4. `prepare` divides the knots by `t0` and the coefficients by `P8`, and handing
  the prepared forcing to the traced solve as a constant is what lost both terms."""
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
