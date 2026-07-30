"""The jax backend against the scipy one (PLAN.md W11 stage 2b).

jax and diffrax are optional, so every test here skips without them, exactly as
`test_pymc_op.py` does for PyMC. The core install stays numpy/scipy/numba.

What makes this cheap to assert is that there is no second implementation to
compare against. Stage 2a made `_rhs` namespace-agnostic, so both backends
integrate the SAME right-hand side and any disagreement is the integrator's
alone -- diffrax's Tsit5 against scipy's LSODA.
"""

import importlib.util

import numpy as np
import pytest

import imr_fast
import imr_fast.sensitivity
from imr_fast import _jax
from _validation_support import NHKV, R0, REQ, oldroyd_b, zener

# Scoped to the cross-backend test rather than the module: the refusals below
# need no jax at all -- `_jax.unsupported_reason` imports only numpy -- and CI
# has no jax, so a module-level skip would drop the coverage that runs there.
_HAS_JAX = all(importlib.util.find_spec(name) is not None for name in ("jax", "diffrax"))
requires_jax = pytest.mark.skipif(not _HAS_JAX, reason="jax and diffrax are optional; not installed")

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

@requires_jax
@pytest.mark.parametrize("radial,label,material", _CASES, ids=[f"radial{r}-{n}" for r, n, _ in _CASES])
def test_jax_backend_matches_scipy(radial, label, material, measured):
  reference = np.asarray(
    imr_fast.simulate(_TIMES, imr_fast.SimulationConfig(R0=R0, Req=REQ, material=material, radial=radial)).radius_ratio
  )
  result = imr_fast.simulate(_TIMES, imr_fast.SimulationConfig(R0=R0, Req=REQ, material=material, radial=radial, backend="jax"))
  computed = np.asarray(result.radius_ratio)
  worst, typical = float(np.nanmax(np.abs(reference - computed))), float(np.nanmedian(np.abs(reference - computed)))
  measured(f"jax vs scipy r={radial} {label}", f"max={worst:.2e} median={typical:.2e}")
  assert result.stats.backend == "jax-tsit5", "the jax backend was not actually selected"
  assert worst < _MAX_BOUND and typical < _MEDIAN_BOUND

def test_unsupported_configurations_are_refused_by_name():
  """A refusal at construction, not a tracer error several frames into diffrax.

  The list is down to one: a sampled forcing history, because its interpolation
  searches its own knots. `test_mass_transfer_is_accepted` covers what left it.
  """
  sampled = imr_fast.SampledForcing(time_s=(0.0, 1e-5, 2e-5), pressure_pa=(0.0, 5e4, 0.0))
  with pytest.raises(ValueError, match="sampled_forcing"):
    imr_fast.SimulationConfig(R0=R0, Req=REQ, material=NHKV, backend="jax", sampled_forcing=sampled)

def test_backend_field_rejects_anything_else():
  with pytest.raises(ValueError, match="backend must be"):
    imr_fast.SimulationConfig(R0=R0, Req=REQ, material=NHKV, backend="torch")


_THERMAL_CASES = [
  ("bubtherm fd", dict(bubtherm=1, Nt=17, thermal="fd")),
  ("bubtherm spectral", dict(bubtherm=1, Nt=17, thermal="spectral")),
  ("coupled fd", dict(bubtherm=1, medtherm=1, Nt=13, Mt=13, thermal="fd")),
  ("coupled spectral", dict(bubtherm=1, medtherm=1, Nt=13, Mt=13, thermal="spectral")),
  ("coupled+mass fd", dict(bubtherm=1, vapor=1, masstrans=1, medtherm=1, Nt=13, Mt=13, thermal="fd")),
]

@requires_jax
@pytest.mark.parametrize("label,options", _THERMAL_CASES, ids=[c[0] for c in _THERMAL_CASES])
def test_jax_backend_matches_scipy_on_the_thermal_path(label, options, measured):
  """The thermal fields, which slice assignment kept off this backend until W11
  stage 4 routed them through `at_set`.

  `medtherm` needs no iterative wall solve at all -- #57 made that closure closed
  form. `masstrans` does, and it is here because #111 turned that solve into a
  bracket on the vapour fraction's own physical range: `_thermal._traced_root`
  bisects a fixed number of times, so the traced program has a fixed shape and
  needs no `lax` primitive.
  """
  times = np.linspace(0.0, 25e-6, 200)
  reference = np.asarray(imr_fast.simulate(times, imr_fast.SimulationConfig(R0=R0, Req=REQ, material=NHKV, **options)).radius_ratio)
  result = imr_fast.simulate(times, imr_fast.SimulationConfig(R0=R0, Req=REQ, material=NHKV, backend="jax", **options))
  computed = np.asarray(result.radius_ratio)
  worst, typical = float(np.nanmax(np.abs(reference - computed))), float(np.nanmedian(np.abs(reference - computed)))
  measured(f"jax vs scipy {label}", f"max={worst:.2e} median={typical:.2e} steps={result.stats.nfev}")
  assert worst < _MAX_BOUND and typical < _MEDIAN_BOUND

def test_mass_transfer_is_accepted():
  """The inverse of the assertion this file used to carry.

  It was refused because the wall closure warm-started a secant from the previous
  call's answer, which made the right-hand side a function of the integrator's
  step history rather than of `(t, y)` -- a tracer stored there escapes the trace,
  so no namespace swap could have fixed it. #111 removed the warm start.
  """
  config = imr_fast.SimulationConfig(
    R0=R0, Req=REQ, material=NHKV, backend="jax", bubtherm=1, vapor=1, masstrans=1, medtherm=1, Nt=9, Mt=9
  )
  assert _jax.unsupported_reason(config) is None

_COVERAGE_CASES = [
  ("gaussian forcing", NHKV, dict(wave_type=1, pA=5e4, TW=5e-6, DT=2e-5)),
  ("heaviside step", NHKV, dict(wave_type=3, pA=5e4, TW=3e-5)),
  ("histotripsy pulse", NHKV, dict(wave_type=2, pA=1e5, omega=2 * np.pi / 2e-5, DT=3e-5, mn=2)),
  ("giesekus", imr_fast.Giesekus(0.1, 80e-6, 16e-6, 0.2, points=12), {}),
  ("linear PTT", imr_fast.LinearPTT(0.1, 80e-6, 16e-6, 0.2, points=12), {}),
  ("giesekus + medtherm", imr_fast.Giesekus(0.1, 80e-6, 16e-6, 0.2, points=12), dict(bubtherm=1, medtherm=1, Nt=9, Mt=9, thermal="fd")),
]

@requires_jax
@pytest.mark.parametrize("label,material,options", _COVERAGE_CASES, ids=[c[0] for c in _COVERAGE_CASES])
def test_jax_backend_covers_forcing_and_distributed_memory(label, material, options, measured):
  """Analytic forcing and the distributed-memory materials.

  Two different blockers, both now gone. The windowed forcings branch on the
  INTEGRATION TIME, which a tracer supplies, so those tests became `where`. The
  distributed materials pack their output into a preallocated buffer -- 2*points
  entries, too many for the list the other branch builds -- so that buffer is
  filled through `at_set` instead.
  """
  times = np.linspace(0.0, 25e-6, 150)
  reference = np.asarray(imr_fast.simulate(times, imr_fast.SimulationConfig(R0=R0, Req=REQ, material=material, **options)).radius_ratio)
  result = imr_fast.simulate(times, imr_fast.SimulationConfig(R0=R0, Req=REQ, material=material, backend="jax", **options))
  computed = np.asarray(result.radius_ratio)
  worst = float(np.nanmax(np.abs(reference - computed)))
  measured(f"jax vs scipy {label}", f"max={worst:.2e}")
  assert worst < _MAX_BOUND

_TANGENT_FIELDS = ("radius_ratio", "radius_m", "wall_velocity_m_s", "internal_pressure_pa", "stress_integral_pa")

@requires_jax
@pytest.mark.parametrize(
  "label,material,paths",
  [
    ("NHKV G", NHKV, ("material.shear_modulus_pa",)),
    ("NHKV G+mu", NHKV, ("material.shear_modulus_pa", "material.viscosity_pa_s")),
    ("Zener G", zener(), ("material.shear_modulus_pa",)),
  ],
  ids=["nhkv-G", "nhkv-G-mu", "zener-G"],
)
def test_jax_tangents_match_the_scipy_sensitivity_path(label, material, paths, measured):
  """Every output's tangent, from one `jacfwd`.

  What is being checked is not just the radius. `internal_pressure_pa` and
  `stress_integral_pa` are nonlinear in state and parameters, and on the scipy
  route their tangents come from `_output_duals` deriving each one. Here the
  traced function returns the primal outputs and `jacfwd` differentiates all of
  them together, so they cost no extra code -- and no extra chance to be wrong.
  """
  from imr_fast._jax import sensitivities_jax

  times = np.linspace(0.0, 20e-6, 80)
  config = imr_fast.SimulationConfig(R0=R0, Req=REQ, material=material, radial=2)
  problem = imr_fast.prepare(config)
  reference = imr_fast.sensitivity.solve_with_sensitivities(problem, times, paths)
  *_, tangent = sensitivities_jax(problem, times, paths)

  worst = {}
  for index, field in enumerate(_TANGENT_FIELDS):
    expected = np.asarray(getattr(reference, field))
    scale = max(float(np.max(np.abs(expected))), 1e-30)
    worst[field] = float(np.max(np.abs(expected - tangent[:, index, :]))) / scale
  measured(f"jax tangents {label}", "  ".join(f"{f.split('_')[0]}={w:.1e}" for f, w in worst.items()))
  assert max(worst.values()) < 1e-05, worst

@requires_jax
def test_jax_tangents_converge_to_the_scipy_ones(measured):
  """The gate is convergence under refinement, not a fixed threshold.

  JAX cannot join the 5e-13 agreement complex-step and Dual reach, and PLAN.md
  W11 originally asked it to. Those two differentiate the SAME integration; JAX
  differentiates a different integrator and carries that integrator's error. So
  the question is whether the residual is error or a wrong derivative, and only
  refinement distinguishes them -- a fixed bound can be met by a derivative that
  is wrong but close.
  """
  from imr_fast._jax import sensitivities_jax

  times = np.linspace(0.0, 20e-6, 80)
  paths = ("material.shear_modulus_pa",)
  errors = []
  for tolerance in (1e-8, 1e-10, 1e-12):
    config = imr_fast.SimulationConfig(R0=R0, Req=REQ, material=NHKV, radial=2, rtol=tolerance, atol=tolerance * 1e-2)
    problem = imr_fast.prepare(config)
    expected = np.asarray(imr_fast.sensitivity.solve_with_sensitivities(problem, times, paths).radius_m)
    *_, tangent = sensitivities_jax(problem, times, paths)
    scale = max(float(np.max(np.abs(expected))), 1e-30)
    errors.append(float(np.max(np.abs(expected - tangent[:, 1, :]))) / scale)
  measured("jax tangent convergence", " -> ".join(f"{e:.1e}" for e in errors))
  assert errors[-1] < errors[0] / 100.0, f"tangent error did not converge under refinement: {errors}"


@requires_jax
@pytest.mark.parametrize(
  "label,material,paths",
  [("NHKV G", NHKV, ("material.shear_modulus_pa",)), ("Zener G+mu", zener(), ("material.shear_modulus_pa", "material.viscosity_pa_s"))],
  ids=["nhkv", "zener"],
)
def test_solve_with_sensitivities_dispatches_on_the_backend(label, material, paths, measured):
  """`backend="jax"` has to mean the same thing for derivatives as for
  trajectories.

  Until this dispatch existed, `sensitivities_jax` was written, tested and
  unreachable: `solve_with_sensitivities` ignored the field entirely, so a
  config asking for jax got a jax forward solve and Dual-route tangents. Every
  field of the result is compared here, not just the radius -- including `state`
  and the `simulation` embedded in it.
  """
  times = np.linspace(0.0, 20e-6, 80)
  scipy_problem = imr_fast.prepare(imr_fast.SimulationConfig(R0=R0, Req=REQ, material=material, radial=2))
  jax_problem = imr_fast.prepare(imr_fast.SimulationConfig(R0=R0, Req=REQ, material=material, radial=2, backend="jax"))
  reference = imr_fast.sensitivity.solve_with_sensitivities(scipy_problem, times, paths)
  computed = imr_fast.sensitivity.solve_with_sensitivities(jax_problem, times, paths)
  worst = {}
  for field in (*_TANGENT_FIELDS, "state"):
    expected, got = np.asarray(getattr(reference, field)), np.asarray(getattr(computed, field))
    assert expected.shape == got.shape, f"{field}: {expected.shape} vs {got.shape}"
    worst[field] = float(np.max(np.abs(expected - got))) / max(float(np.max(np.abs(expected))), 1e-30)
  trajectory = float(np.max(np.abs(np.asarray(reference.simulation.radius_ratio) - np.asarray(computed.simulation.radius_ratio))))
  measured(f"jax sensitivity dispatch {label}", f"worst={max(worst.values()):.1e} sim={trajectory:.1e}")
  assert max(worst.values()) < 1e-05, worst
  assert trajectory < _MAX_BOUND

def test_jax_sensitivities_refuse_what_they_do_not_cover():
  """Refused by name rather than quietly falling back to the Dual route, which
  would make the backend field mean one thing for trajectories and another for
  their derivatives."""
  times = np.linspace(0.0, 20e-6, 40)
  thermal = imr_fast.prepare(imr_fast.SimulationConfig(R0=R0, Req=REQ, material=NHKV, backend="jax", bubtherm=1, Nt=9))
  with pytest.raises(NotImplementedError, match="bubtherm"):
    imr_fast.sensitivity.solve_with_sensitivities(thermal, times, ("material.shear_modulus_pa",))
  mechanical = imr_fast.prepare(imr_fast.SimulationConfig(R0=R0, Req=REQ, material=NHKV, backend="jax"))
  with pytest.raises(NotImplementedError, match="material scales"):
    imr_fast.sensitivity.solve_with_sensitivities(mechanical, times, ("R0",))


@requires_jax
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

@requires_jax
def test_traced_and_brentq_roots_agree_on_the_shipped_closure(measured):
  """The two solvers on one captured wall state, so a disagreement shows up here
  rather than as trajectory drift it would be easy to blame on the integrator."""
  _, jnp, _ = _jax._jax()

  from imr_fast import _thermal

  captured = []
  original = _thermal._bracketed_root

  def record(residual, **options):
    captured.append(residual)
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
