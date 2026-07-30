"""The jax backend against the scipy one (PLAN.md W11 stage 2b).

Nothing here skips. jax and diffrax are core dependencies as of stage 5, so the
`requires_jax` mark this module used to carry -- and the careful scoping that kept
the refusal assertions running on a jax-less CI job -- are both gone.

What makes this cheap to assert is that there is no second implementation to
compare against. Stage 2a made `_rhs` namespace-agnostic, so both backends
integrate the SAME right-hand side and any disagreement is the integrator's
alone -- diffrax's Tsit5 against scipy's LSODA.
"""

from typing import Any

import numpy as np
import pytest

import imr_fast
import imr_fast.sensitivity
from imr_fast import _jax
from _validation_support import NHKV, R0, REQ, oldroyd_b, zener


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

_COVERAGE_CASES = [
  ("gaussian forcing", NHKV, dict(wave_type=1, pA=5e4, TW=5e-6, DT=2e-5)),
  ("heaviside step", NHKV, dict(wave_type=3, pA=5e4, TW=3e-5)),
  ("histotripsy pulse", NHKV, dict(wave_type=2, pA=1e5, omega=2 * np.pi / 2e-5, DT=3e-5, mn=2)),
  ("giesekus", imr_fast.Giesekus(0.1, 80e-6, 16e-6, 0.2, points=12), {}),
  ("linear PTT", imr_fast.LinearPTT(0.1, 80e-6, 16e-6, 0.2, points=12), {}),
  ("giesekus + medtherm", imr_fast.Giesekus(0.1, 80e-6, 16e-6, 0.2, points=12), dict(bubtherm=1, medtherm=1, Nt=9, Mt=9, thermal="fd")),
]

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
  tangent = sensitivities_jax(problem, times, paths)[1].derived

  worst = {}
  for index, field in enumerate(_TANGENT_FIELDS):
    expected = np.asarray(getattr(reference, field))
    scale = max(float(np.max(np.abs(expected))), 1e-30)
    worst[field] = float(np.max(np.abs(expected - tangent[:, index, :]))) / scale
  measured(f"jax tangents {label}", "  ".join(f"{f.split('_')[0]}={w:.1e}" for f, w in worst.items()))
  assert max(worst.values()) < 1e-05, worst

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
    tangent = sensitivities_jax(problem, times, paths)[1].derived
    scale = max(float(np.max(np.abs(expected))), 1e-30)
    errors.append(float(np.max(np.abs(expected - tangent[:, 1, :]))) / scale)
  measured("jax tangent convergence", " -> ".join(f"{e:.1e}" for e in errors))
  assert errors[-1] < errors[0] / 100.0, f"tangent error did not converge under refinement: {errors}"


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

def test_jax_sensitivities_refuse_only_what_is_not_a_scalar_field():
  """Everything this test used to list is covered now.

  `bubtherm=1` went in the thermal work, `R0` with the config scalars, and
  `physics.*`/`initial.*` with `_Overridden`. What is left refuses for a reason that
  is not about jax: a path must name one finite SCALAR, so `initial.stress_state` --
  a sequence -- is out on both routes, and an unknown path is a typo.
  """
  times = np.linspace(0.0, 20e-6, 40)
  problem = imr_fast.prepare(imr_fast.SimulationConfig(R0=R0, Req=REQ, material=NHKV, backend="jax"))
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

@pytest.mark.parametrize("label,options,bound", _THERMAL_TANGENT_CASES, ids=[c[0] for c in _THERMAL_TANGENT_CASES])
def test_jax_thermal_tangents_match_the_scipy_sensitivity_path(label, options, bound, measured):
  """Every output's tangent, including the three thermal fields.

  Those three were hardcoded to None on this route. That is worse than a refusal:
  `backend="jax"` would have returned a `SensitivityResult` whose thermal tangents
  were silently absent while the scipy one filled them in -- the exact asymmetry
  `_jax_sensitivities`' own docstring argues against. They are built here by
  `vmap` over the time axis, not by the numpy path's per-timestep Python loop,
  which would unroll one wall closure per output time into the graph.
  """
  times = np.linspace(0.0, 15e-6, 60)
  paths = ("material.shear_modulus_pa",)
  reference = imr_fast.sensitivity.solve_with_sensitivities(
    imr_fast.prepare(imr_fast.SimulationConfig(R0=R0, Req=REQ, material=NHKV, **options)), times, paths
  )
  computed = imr_fast.sensitivity.solve_with_sensitivities(
    imr_fast.prepare(imr_fast.SimulationConfig(R0=R0, Req=REQ, material=NHKV, backend="jax", **options)), times, paths
  )

  worst = {}
  for field in _ALL_TANGENT_FIELDS:
    expected, actual = getattr(reference, field), getattr(computed, field)
    # Presence must agree too. A None here on one backend only is the bug this
    # test was written for, and comparing values alone would skip right past it.
    assert (expected is None) == (actual is None), f"{field} is None on only one backend"
    if expected is None: continue
    expected = np.asarray(expected)
    scale = max(float(np.max(np.abs(expected))), 1e-30)
    worst[field] = float(np.max(np.abs(expected - np.asarray(actual)))) / scale
  measured(f"jax thermal tangents {label}", "  ".join(f"{f.split('_')[0]}={w:.1e}" for f, w in worst.items()))
  assert max(worst.values()) < bound, worst

def test_jax_thermal_tangents_converge_to_the_scipy_ones(measured):
  """The property the per-case bounds above stand in for.

  At the default tolerance the spectral medium temperature disagrees by 6.1e-05,
  which is large enough to look like a defect. Tightening both integrators shows
  it is not: the disagreement falls by eight orders, so the two backends are
  computing the same tangent and differing only in how well each integrates it.
  """
  times = np.linspace(0.0, 15e-6, 60)
  paths = ("material.shear_modulus_pa",)
  options: dict[str, Any] = dict(bubtherm=1, medtherm=1, Nt=11, Mt=11, thermal="spectral")
  errors = []
  for tolerance in (1e-7, 1e-9, 1e-11):
    reference = imr_fast.sensitivity.solve_with_sensitivities(
      imr_fast.prepare(imr_fast.SimulationConfig(R0=R0, Req=REQ, material=NHKV, rtol=tolerance, atol=tolerance, **options)),
      times, paths,
    )
    computed = imr_fast.sensitivity.solve_with_sensitivities(
      imr_fast.prepare(
        imr_fast.SimulationConfig(R0=R0, Req=REQ, material=NHKV, rtol=tolerance, atol=tolerance, backend="jax", **options)
      ),
      times, paths,
    )
    expected = np.asarray(reference.medium_temperature_k)
    scale = max(float(np.max(np.abs(expected))), 1e-30)
    errors.append(float(np.max(np.abs(expected - np.asarray(computed.medium_temperature_k)))) / scale)
  measured("jax thermal tangent convergence", " -> ".join(f"{e:.1e}" for e in errors))
  # Monotone descent plus a bound on where it ARRIVES, not a fixed ratio between the
  # ends. A ratio rewards a solver whose loose-tolerance answer is bad: Tsit5 ran
  # 1.2e-01 -> 1.6e-03 -> 5.3e-07 and passed a 1e4 ratio, while Kvaerno5 runs
  # 4.3e-04 -> 7.5e-06 -> 5.8e-08 -- better at every tolerance, including the last --
  # and fails it, purely for starting closer.
  assert all(later < earlier for earlier, later in zip(errors, errors[1:], strict=False)), f"not monotone: {errors}"
  assert errors[-1] < 1e-06, f"medium temperature tangent did not converge: {errors}"


def test_collapse_initialization_needs_nothing_from_this_backend(measured):
  """Listed as a jax blocker; it is not one, and this records why.

  Collapse shooting is a `prepare`-time computation -- `solve_ivp` with a terminal
  maximum-radius event, a bracket-expansion loop and a `brentq` on top -- none of
  which a traced program ever sees. It produces `problem.initial_state`, a
  concrete array, and the traced solve starts from that like any other. So the
  three things that would be hard to trace happen before tracing begins.

  What is genuinely still out is differentiating THROUGH it, which needs `R0` and
  `Req` as traced parameters. That is a `SCALE_PATHS` limitation, not this one.
  """
  times = np.linspace(0.0, 25e-6, 150)
  options: dict[str, Any] = dict(R0=R0, Req=REQ, material=zener(), collapse=imr_fast.CollapseInitialization())
  reference = np.asarray(imr_fast.simulate(times, imr_fast.SimulationConfig(**options)).radius_ratio)
  computed = np.asarray(imr_fast.simulate(times, imr_fast.SimulationConfig(**options, backend="jax")).radius_ratio)
  worst = float(np.nanmax(np.abs(reference - computed)))
  measured("jax vs scipy collapse init", f"max={worst:.2e}")
  assert worst < _MAX_BOUND


# A knotted history with noise on it, so the interpolation is actually exercised
# between knots rather than reproducing a smooth analytic curve. Seeded: an
# unseeded history would make a failure unreproducible.
def _sampled_history():
  rng = np.random.default_rng(3)
  knots = np.linspace(0.0, 3e-5, 24)
  pressure = 6e4 * np.sin(2 * np.pi * knots / 1.5e-5) + 1e4 * rng.standard_normal(knots.size)
  return imr_fast.SampledForcing(time_s=tuple(knots), pressure_pa=tuple(pressure))

_SAMPLED_CASES: list[tuple[str, np.ndarray, dict[str, Any]]] = [
  ("mechanical", np.linspace(0.0, 25e-6, 200), {}),
  ("coupled fd", np.linspace(0.0, 25e-6, 200), dict(bubtherm=1, medtherm=1, Nt=11, Mt=11, thermal="fd")),
  # Past the last knot, where the mask has to zero the forcing rather than
  # extrapolate the clamped cubic it still evaluates.
  ("past last knot", np.linspace(0.0, 45e-6, 200), {}),
]

@pytest.mark.parametrize("label,times,options", _SAMPLED_CASES, ids=[c[0] for c in _SAMPLED_CASES])
def test_sampled_forcing_matches_scipy(label, times, options, measured):
  """The last entry `unsupported_reason` carried.

  What kept it out was three data-dependent uses of `tn`: an out-of-range early
  return, `searchsorted` on the knots, and indexing a coefficient row with the
  result. `_sampled_pressure` clamps the index and applies the range test as a
  multiplicative mask, so the traced program has a fixed shape.

  The mask is a multiply rather than an `xp.where` over the results on purpose:
  `where` is not a ufunc, so `np.where` on a `Dual` returns an object array and
  would break the tangent path silently. Verified bit-identical on the numpy
  trajectory AND on the Dual tangent across that change.
  """
  settings: dict[str, Any] = dict(R0=R0, Req=REQ, material=NHKV, sampled_forcing=_sampled_history(), **options)
  reference = np.asarray(imr_fast.simulate(times, imr_fast.SimulationConfig(**settings)).radius_ratio)
  computed = np.asarray(imr_fast.simulate(times, imr_fast.SimulationConfig(**settings, backend="jax")).radius_ratio)
  worst, typical = float(np.nanmax(np.abs(reference - computed))), float(np.nanmedian(np.abs(reference - computed)))
  measured(f"jax vs scipy sampled forcing {label}", f"max={worst:.2e} median={typical:.2e}")
  assert worst < _MAX_BOUND and typical < _MEDIAN_BOUND


# Configuration scalars as traced parameters. These needed no `SCALE_PATHS`-style
# indirection -- `params` already takes all seven positionally, and its only `np.`
# call is `sqrt(P8/rho)` on concrete physics values. What they did need was three
# things that had been reading concrete config fields where a traced value belonged,
# each of which showed up as one wrong output and the rest right:
#
#   `derived` scaled by `config.R0`; `_thermal_fields` by `config.T8` (1.11 error);
#   and `initial_state_vector` was not rebuilt at all, so `Pb`, `kv0` and `Uc`
#   contributed zero to the starting state's tangent.
#
# Plus the medium's wall weights, which are built from `chi = T8*K8/(P8*R0*Uc)`.
# Those are subtle: `_wall_theta_bw` is invariant to a COMMON scaling of the
# weights, so `R0` -- which enters only through `chi` -- had a correct tangent from
# a constant medium, while `T8` did not because `iota` multiplies `grad_Tm` alone.
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

@pytest.mark.parametrize("label,options,paths,bound", _CONFIG_TANGENT_CASES, ids=[c[0] for c in _CONFIG_TANGENT_CASES])
def test_jax_config_scalar_tangents_match_the_scipy_sensitivity_path(label, options, paths, bound, measured):
  """`R0`, `Req`, `T8` and `pA`, mixed with a material scale in one of the cases so
  the two groups' ordering in the traced vector is exercised rather than assumed.

  The `T8 w/ vapor` case carries `vapor=1` on purpose. Without vapour, `Pv = 0 *
  pvsat(T8)` and `T8` has no effect on a mechanical solve at all, so both backends
  return exactly zero and the comparison passes without testing anything.
  """
  times = np.linspace(0.0, 15e-6, 60)
  reference = imr_fast.sensitivity.solve_with_sensitivities(
    imr_fast.prepare(imr_fast.SimulationConfig(R0=R0, Req=REQ, material=NHKV, **options)), times, paths
  )
  computed = imr_fast.sensitivity.solve_with_sensitivities(
    imr_fast.prepare(imr_fast.SimulationConfig(R0=R0, Req=REQ, material=NHKV, backend="jax", **options)), times, paths
  )

  worst = {}
  for field in _ALL_TANGENT_FIELDS:
    expected, actual = getattr(reference, field), getattr(computed, field)
    assert (expected is None) == (actual is None), f"{field} is None on only one backend"
    if expected is None: continue
    expected = np.asarray(expected)
    scale = float(np.max(np.abs(expected)))
    assert scale > 0.0, f"{field} tangent is identically zero -- this case tests nothing"
    worst[field] = float(np.max(np.abs(expected - np.asarray(actual)))) / scale
  measured(f"jax config tangents {label}", "  ".join(f"{f.split('_')[0]}={w:.1e}" for f, w in worst.items()))
  assert max(worst.values()) < bound, worst

def test_the_traced_medium_is_rebuilt_for_every_consumer(measured):
  """A regression test for a defect that only `T8` exposed, and only in two of
  seven outputs.

  The medium's wall weights carry `T8`. Rebuilding them for `_rhs` but leaving
  `_thermal_fields` on the prepared ones left the two temperature output tangents
  5.7e-04 wrong -- tolerance-independent -- while all five state tangents
  converged. Convergence is what separates a missing term from integrator error,
  so that is what is asserted: an eight-order fall over six orders of tolerance.
  """
  times = np.linspace(0.0, 15e-6, 60)
  options: dict[str, Any] = dict(bubtherm=1, medtherm=1, Nt=11, Mt=11, thermal="fd")
  errors = []
  for tolerance in (1e-7, 1e-10, 1e-12):
    reference = imr_fast.sensitivity.solve_with_sensitivities(
      imr_fast.prepare(imr_fast.SimulationConfig(R0=R0, Req=REQ, material=NHKV, rtol=tolerance, atol=tolerance, **options)),
      times, ("T8",),
    )
    computed = imr_fast.sensitivity.solve_with_sensitivities(
      imr_fast.prepare(
        imr_fast.SimulationConfig(R0=R0, Req=REQ, material=NHKV, rtol=tolerance, atol=tolerance, backend="jax", **options)
      ),
      times, ("T8",),
    )
    expected = np.asarray(reference.bubble_temperature_k)
    errors.append(float(np.max(np.abs(expected - np.asarray(computed.bubble_temperature_k)))) / float(np.max(np.abs(expected))))
  measured("jax T8 temperature tangent convergence", " -> ".join(f"{e:.1e}" for e in errors))
  assert errors[-1] < errors[0] / 1e4, f"the T8 temperature tangent did not converge: {errors}"


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
  problem = imr_fast.prepare(imr_fast.SimulationConfig(R0=R0, Req=REQ, material=NHKV, backend="jax"))
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

@pytest.mark.parametrize("label,options,paths,bound", _STRUCTURE_TANGENT_CASES, ids=[c[0] for c in _STRUCTURE_TANGENT_CASES])
def test_jax_structure_field_tangents_match_the_scipy_sensitivity_path(label, options, paths, bound, measured):
  """The three thermal `physics` fields are on a mass-transfer configuration on
  purpose: on a mechanical one their tangents are identically zero, so the
  comparison would pass without testing anything. The assertion below rejects a
  zero reference for exactly that reason."""
  times = np.linspace(0.0, 15e-6, 60)
  reference = imr_fast.sensitivity.solve_with_sensitivities(
    imr_fast.prepare(imr_fast.SimulationConfig(R0=R0, Req=REQ, material=NHKV, **options)), times, paths
  )
  computed = imr_fast.sensitivity.solve_with_sensitivities(
    imr_fast.prepare(imr_fast.SimulationConfig(R0=R0, Req=REQ, material=NHKV, backend="jax", **options)), times, paths
  )
  worst = {}
  for field in _ALL_TANGENT_FIELDS:
    expected, actual = getattr(reference, field), getattr(computed, field)
    assert (expected is None) == (actual is None), f"{field} is None on only one backend"
    if expected is None: continue
    expected = np.asarray(expected)
    scale = float(np.max(np.abs(expected)))
    if scale == 0.0: continue
    worst[field] = float(np.max(np.abs(expected - np.asarray(actual)))) / scale
  assert worst, f"{label}: every tangent is identically zero -- this case tests nothing"
  measured(f"jax structure tangents {label}", "  ".join(f"{f.split('_')[0]}={w:.1e}" for f, w in worst.items()))
  assert max(worst.values()) < bound, worst

def test_the_traced_path_covers_every_path_the_scipy_route_accepts():
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
  times = np.linspace(0.0, 8e-6, 20)
  accepted = []
  for path in sorted(set(candidates)):
    try:
      imr_fast.sensitivity.solve_with_sensitivities(imr_fast.prepare(config), times, (path,))
    except Exception:  # noqa: BLE001,S112 - the scipy route's own refusals are not this test's subject
      continue
    accepted.append(path)
  assert len(accepted) > 20, f"only {len(accepted)} paths accepted; the enumeration has gone stale"
  assert not [p for p in accepted if p not in covered], f"the traced path does not cover {[p for p in accepted if p not in covered]}"


def test_jax_differentiates_through_the_collapse_shooting(measured):
  """The last blocker on deleting the numpy sensitivity route.

  A collapse precursor makes the STARTING memory state a function of the
  parameters, through a shooting solve with a terminal maximum-radius event. The
  traced path used to receive that state as a constant, so its tangent was zero --
  measured at relative 1.00, wrong outright rather than imprecise.

  What makes it tractable without a differentiable event or a differentiable
  root-find is that both conditions can be applied AFTER a fixed-endpoint solve, and
  that `dR/dt` is zero at the maximum -- which decouples a 2x2 implicit system into
  two scalar divisions. See `_jax._collapse_tangents`.

  Asserted by convergence as well as by agreement, because the Zener collapse
  amplifies integrator error: the radius tangent sits at 3.7e-02 at rtol 1e-08 and
  7.8e-07 at 1e-12, so a single loose bound would say nothing about correctness.
  """
  times = np.linspace(0.0, 25e-6, 60)
  radius, memory = [], []
  for tolerance in (1e-8, 1e-10, 1e-12):
    settings: dict[str, Any] = dict(
      R0=R0, Req=REQ, material=zener(), collapse=imr_fast.CollapseInitialization(), rtol=tolerance, atol=tolerance
    )
    reference = imr_fast.sensitivity.solve_with_sensitivities(imr_fast.prepare(imr_fast.SimulationConfig(**settings)), times, ("Req",))
    computed = imr_fast.sensitivity.solve_with_sensitivities(
      imr_fast.prepare(imr_fast.SimulationConfig(**settings, backend="jax")), times, ("Req",)
    )
    expected = np.asarray(reference.radius_ratio)
    radius.append(float(np.max(np.abs(expected - np.asarray(computed.radius_ratio)))) / float(np.max(np.abs(expected))))
    # The starting state's own tangent, which is what `_collapse_tangents` produces.
    start, got_start = np.asarray(reference.state)[0], np.asarray(computed.state)[0]
    memory.append(float(np.max(np.abs(start - got_start))) / float(np.max(np.abs(start))))
  measured("jax collapse tangent", "radius " + " -> ".join(f"{e:.1e}" for e in radius))
  measured("jax collapse memory tangent", " -> ".join(f"{e:.1e}" for e in memory))
  assert memory[0] < 1e-07, f"the starting memory tangent is wrong at the default-ish tolerance: {memory}"
  assert radius[-1] < radius[0] / 1e3, f"the collapse radius tangent did not converge: {radius}"
  assert memory[-1] < memory[0] / 1e2, f"the collapse memory tangent did not converge: {memory}"


def test_sampled_forcing_tangents_carry_the_scaling_parameters(measured):
  """A bug that shipped in #116 and was found by asking what `prepare` computes from
  a parameter and then hands over as a constant.

  `_prepare_forcing` divides the knots by `t0 = R0/Uc` and the cubic coefficients by
  `P8`, each row also carrying `t0**degree`. `_rhs_args` passed the prepared forcing
  straight through, so a tangent with respect to either lost those terms: `R0` sat at
  3.1e-02 and `physics.far_field_pressure_pa` at 6.1e-02, both tolerance-independent.
  `material.shear_modulus_pa` was unaffected, which is why the coverage audit missed
  it -- the audit asked which paths RUN, not which produce the right answer.

  Convergence is the assertion, because a plateau is what distinguishes a missing
  term from integrator error.
  """
  rng = np.random.default_rng(3)
  knots = np.linspace(0.0, 3e-5, 24)
  history = imr_fast.SampledForcing(
    time_s=tuple(knots), pressure_pa=tuple(6e4 * np.sin(2 * np.pi * knots / 1.5e-5) + 1e4 * rng.standard_normal(knots.size))
  )
  times = np.linspace(0.0, 20e-6, 60)
  for path in ("R0", "physics.far_field_pressure_pa"):
    errors = []
    for tolerance in (1e-9, 1e-12):
      settings: dict[str, Any] = dict(
        R0=R0, Req=REQ, material=NHKV, sampled_forcing=history, rtol=tolerance, atol=tolerance
      )
      reference = imr_fast.sensitivity.solve_with_sensitivities(
        imr_fast.prepare(imr_fast.SimulationConfig(**settings)), times, (path,)
      )
      computed = imr_fast.sensitivity.solve_with_sensitivities(
        imr_fast.prepare(imr_fast.SimulationConfig(**settings, backend="jax")), times, (path,)
      )
      expected = np.asarray(reference.radius_ratio)
      errors.append(float(np.max(np.abs(expected - np.asarray(computed.radius_ratio)))) / float(np.max(np.abs(expected))))
    measured(f"jax sampled-forcing tangent {path}", " -> ".join(f"{e:.1e}" for e in errors))
    assert errors[-1] < errors[0] / 10.0, f"{path} did not converge: {errors} -- a scaling parameter is missing"
