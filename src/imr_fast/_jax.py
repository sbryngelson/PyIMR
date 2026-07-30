"""The JAX/diffrax backend (W11 stage 2b).

There is no physics here. Stage 2a made `_rhs` namespace-agnostic, so this
module's whole job is to hand the SHIPPED right-hand side to diffrax with
`xp=jax.numpy` and translate the result back. A transcribed copy of the
equations would have been a second implementation that can drift -- exactly the
`_mechanical` problem this migration exists to remove.

jax and diffrax are OPTIONAL, on the `pymc_op` precedent: a core install stays
numpy/scipy/numba, and the import is deferred so nothing pays for it unbidden.

**Solver choice is static, and that is not a shortcut.** W11 measured
`|lambda_max/lambda_min|` of `df/dy` along the trajectory at 1.18 for the
mechanical problem against 6.9e+16 for coupled spectral, with median equal to
worst in every case: stiffness belongs to the configuration, not to a phase of
the solve. scipy already relies on this, picking BDF when a Jacobian sparsity
pattern is present and LSODA otherwise. The mechanical path this backend accepts
is the non-stiff one, so it takes an explicit solver.
"""

from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass

import numpy as np

from ._config import SimulationError, SolverStats

__all__ = ["SCALE_PATHS", "TracedOutputs", "available", "integrate_jax", "sensitivities_jax", "unsupported_reason"]

_MISSING = "backend='jax' requires jax and diffrax: pip install 'imr-fast[jax]'"

def _jax():
  # The ignores are local rather than baselined: `tools/pyright_baseline.py`
  # refuses reportMissingImports on purpose, because baselining it would mask a
  # genuinely broken import. These two are optional and absent by design.
  try:
    import diffrax  # pyright: ignore[reportMissingImports]
    import jax  # pyright: ignore[reportMissingImports]
    import jax.numpy as jnp  # pyright: ignore[reportMissingImports]
  except ImportError as error:  # pragma: no cover - exercised only without jax
    raise ImportError(_MISSING) from error
  # float64 before anything touches a dtype. JAX defaults to float32, which
  # would quietly pass the loose checks in this suite and fail the tight ones.
  jax.config.update("jax_enable_x64", True)
  _enable_compilation_cache(jax)
  return jax, jnp, diffrax

def _enable_compilation_cache(jax):
  """Persist XLA compilations across processes.

  A fresh process pays ~1.4 s before the first solve on the coupled thermal
  problem, and the two halves have different remedies: ~580 ms of tracing, which
  is Python and unavoidable per process, and ~800 ms of XLA compilation, which
  is not. On-disk caching takes the second from 803 ms to 95 ms, measured across
  separate interpreters.

  `IMR_FAST_JAX_CACHE` overrides the location; setting it empty disables the
  cache entirely. The size thresholds are set to zero because the defaults skip
  short compilations, and here every one of them is worth keeping.
  """
  location = os.environ.get("IMR_FAST_JAX_CACHE")
  if location == "": return
  if location is None:
    location = str(pathlib.Path(os.environ.get("XDG_CACHE_HOME", pathlib.Path.home() / ".cache")) / "imr_fast" / "jax")
  jax.config.update("jax_compilation_cache_dir", location)
  jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
  jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)

def available() -> bool:
  """Whether the optional dependencies are importable."""
  try:
    import diffrax  # noqa: F401  # pyright: ignore[reportMissingImports]
    import jax  # noqa: F401  # pyright: ignore[reportMissingImports]
  except ImportError:
    return False
  return True

def unsupported_reason(config) -> str | None:
  """Why this configuration cannot use the JAX backend yet, or None.

  Nothing, now -- kept because the shape of the check is what makes a future
  restriction a named refusal at construction rather than a tracer error three
  frames inside diffrax.

  Both entries it used to carry were the same kind of problem and neither needed a
  new algorithm. Mass transfer warm-started its wall solve from the previous call's
  answer, making the right-hand side a function of the integrator's step history;
  #111 bracketed it instead. A sampled forcing history branched on `tn` and indexed
  its knots with the result; `_rhs._sampled_pressure` clamps and masks instead.
  """
  return None


# Tracing and compiling the solve costs far more than running it -- measured at
# 796 ms against 30 ms for the same mechanical case, because every call rebuilt
# the graph. Keyed on the caller's identity plus the shapes that change the
# traced program; the problem is held so an id cannot be reused underneath the
# entry. Bounded, because a design sweep that varies the time grid would
# otherwise grow one entry per distinct length forever.
_COMPILED: dict = {}
_COMPILED_LIMIT = 32

def _cached(key, build):
  hit = _COMPILED.get(key)
  if hit is None:
    if len(_COMPILED) >= _COMPILED_LIMIT: _COMPILED.clear()
    hit = _COMPILED[key] = build()
  return hit

def integrate_jax(rhs, times, initial, *, args, rtol, atol, failure, label="", max_step=None, cache_key=None):
  """Run `rhs` through diffrax, returning `(times, states, stats)`.

  `states` comes back in scipy's `solution.y` orientation -- one row per state,
  one column per requested time -- so callers need no branch of their own.
  """
  jax, jnp, diffrax = _jax()
  from time import perf_counter

  def build():
    def solve(grid, start):
      return diffrax.diffeqsolve(
        diffrax.ODETerm(lambda t, y, _a: jnp.asarray(rhs(t, y, *args, xp=jnp))), diffrax.Tsit5(),
        t0=grid[0], t1=grid[-1], dt0=None, y0=start,
        stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol, dtmax=max_step),
        saveat=diffrax.SaveAt(ts=grid), max_steps=1_000_000, throw=False,
      )

    return jax.jit(solve)

  compiled = _cached(cache_key, build) if cache_key is not None else build()
  started = perf_counter()
  try:
    solution = compiled(jnp.asarray(np.asarray(times, dtype=float)), jnp.asarray(np.asarray(initial, dtype=float)))
  except Exception as error:  # noqa: BLE001 - any solver failure is one failure
    elapsed = perf_counter() - started
    stats = SolverStats(backend=f"jax-tsit5{label}", success=False, message=str(error), nfev=0, njev=0, nlu=0, elapsed_s=elapsed)
    raise SimulationError(f"{failure}: {error}", stats) from error

  elapsed = perf_counter() - started
  states = np.asarray(jax.block_until_ready(solution.ys), dtype=float).T
  steps = int(solution.stats.get("num_steps", 0))
  finite = bool(np.all(np.isfinite(states)))
  # diffrax reports its own outcome; `throw=False` keeps it out of the traceback
  # so it can be reported the same way scipy's is. RESULTS compares by identity
  # rather than by its repr, which is an opaque `RESULTS<>` for every member.
  ok = bool(solution.result == diffrax.RESULTS.successful)
  success = finite and ok
  message = "The solver successfully reached the end of the integration interval." if ok else str(solution.result)
  if ok and not finite: message = f"{message}; solution contains non-finite states"
  stats = SolverStats(backend=f"jax-tsit5{label}", success=success, message=message, nfev=steps, njev=0, nlu=0, elapsed_s=elapsed)
  if not success: raise SimulationError(f"{failure}: {message}", stats)
  return states, stats

# `derived`'s column order, which `sensitivity._jax_sensitivities` unpacks.
_DERIVED_ORDER = ("radius_ratio", "radius_m", "wall_velocity_m_s", "internal_pressure_pa", "stress_integral_pa")

@dataclass(frozen=True, slots=True)
class TracedOutputs:
  """What `sensitivities_jax` differentiates, returned once as values and once as
  tangents.

  A structure rather than a flat tuple, and that is not cosmetic: this was ten
  positional values by the end, callers reached in with `*_, tangent`, and the
  moment the thermal fields were appended that silently started picking up the
  vapour tangent -- None for a mechanical configuration. Naming the fields also
  lets the three optional ones BE optional to a type checker while the other two
  are not, which a dict of arrays cannot express.
  """

  states: np.ndarray
  derived: np.ndarray
  bubble_temperature: np.ndarray | None
  medium_temperature: np.ndarray | None
  vapor_fraction: np.ndarray | None

# The five scales `_material_scales` returns, by the parameter path that names
# each. Structure stays concrete and only these are traced -- a material cannot
# be built from a traced value at all, because `__post_init__` validates with
# `np.isfinite`. See PLAN.md W11 stage 3.
SCALE_PATHS = {
  "material.shear_modulus_pa": 0,
  "material.viscosity_pa_s": 1,
  "material.relaxation_time_s": 2,
  "material.retardation_time_s": 3,
  "material.stiffening": 4,
}

def sensitivities_jax(problem, times, paths):
  """`(outputs, tangents)` for the mechanical path, both from one `jacfwd`.

  The payoff over the Dual route is that nothing here derives a tangent. The
  traced function returns the primal outputs -- radius, velocity, pressure,
  stress integral -- and `jacfwd` differentiates all of them together, so
  `internal_pressure_pa`'s sensitivity costs no more code than the radius's.
  `_output_duals` and `_compiled_mechanical_outputs` exist to do exactly this by
  hand.

  Returns `(values, tangents)`, both `TracedOutputs`. `derived` is ordered
  `_DERIVED_ORDER`; tangents carry a trailing parameter axis. The thermal fields
  are None for a mechanical configuration, on both backends.
  """
  jax, jnp, diffrax = _jax()
  from ._prepare import _material_scales, params
  from ._rhs import _rhs, _rhs_args
  from ._stress import _stress
  from ._thermal import _apply_thermal_boundaries

  config = problem.config
  unknown = [path for path in paths if path not in SCALE_PATHS]
  if unknown: raise ValueError(f"jax sensitivities cover the material scales {sorted(SCALE_PATHS)}; got {unknown}")
  base = np.asarray(_material_scales(config.material), dtype=float)
  slots = [SCALE_PATHS[path] for path in paths]
  layout = problem.layout
  has_stress = layout.stress.stop > layout.stress.start
  initial = jnp.asarray(np.asarray(problem.initial_state, dtype=float))
  grid_s = np.asarray(times, dtype=float)

  def outputs(values):
    scales = jnp.asarray(base)
    for slot, value in zip(slots, values, strict=True): scales = scales.at[slot].set(value)
    p = params(
      config.R0, config.Req, config.material, config.vapor, config.T8, config.pA, config.omega, config.TW,
      config.DT, config.mn, config.wave_type, config.bubtherm, config.masstrans, config.physics,
      xp=jnp, scales=tuple(scales),
    )
    # The prepared thermal operators pass through as constants, and that is
    # correct rather than convenient: the medium's flux weights are built from
    # `chi`, `iota`, `Fom` and `L_heat_star`, and its grid powers from `Lt` --
    # all thermal or geometric, none of them derived from the five material
    # scales this traces. `Br` does carry viscosity, but it lives in `p`, which
    # is rebuilt from tracers above. Checked against the scipy tangents for
    # bubtherm, medtherm and masstrans rather than argued from the parameter
    # list alone.
    args = _rhs_args(problem, p)
    grid = jnp.asarray(grid_s) / p["t0"]
    solution = diffrax.diffeqsolve(
      diffrax.ODETerm(lambda t, y, _a: jnp.asarray(_rhs(t, y, *args, xp=jnp))), diffrax.Tsit5(),
      t0=grid[0], t1=grid[-1], dt0=None, y0=initial,
      stepsize_controller=diffrax.PIDController(rtol=config.rtol, atol=config.atol),
      saveat=diffrax.SaveAt(ts=grid), max_steps=1_000_000, adjoint=diffrax.ForwardMode(),
    )
    states = solution.ys
    radius, velocity = states[:, 0], states[:, 1]
    stress_state = states[:, layout.stress].T if has_stress else None
    # Same closed forms `_build_result` uses, vectorised over time rather than
    # looped -- verified equal to the loop at 0 and 3.5e-18 for NHKV and Zener.
    #
    # Including its BRANCH. The polytropic form holds only while the internal
    # pressure is algebraic in the radius; with `bubtherm=1` it is a state
    # variable instead, and using the closed form there left every thermal
    # pressure tangent 8.0e-01 wrong while radius, velocity and stress agreed to
    # 1e-07 -- a discrepancy in one output only, which is what pointed here.
    if layout.pressure is None:
      pressure = (p["Pb"] - p["Pv"]) * radius ** (-3.0 * p["kappa"]) + p["Pv"]
    else:
      pressure = states[:, layout.pressure]
    stress = _stress(config.material, p, radius, velocity, stress_state, problem.instantaneous_material, False, xp=jnp)[0]
    derived = jnp.stack(
      [radius, radius * config.R0, velocity * config.R0 / p["t0"], pressure * p["P8"], stress * p["P8"]], axis=1
    )
    return (states, derived, *_thermal_fields(states, p, problem, jnp, _apply_thermal_boundaries))

  values = jnp.asarray(base[slots])

  def required(item):
    return np.asarray(jax.block_until_ready(item), dtype=float)

  def optional(item):
    return None if item is None else required(item)

  def plain(group):
    # Unpacked by name rather than splatted, so the two always-present fields
    # stay non-optional to a type checker.
    states, derived, bubble, medium, vapor = group
    return TracedOutputs(required(states), required(derived), optional(bubble), optional(medium), optional(vapor))

  return plain(outputs(values)), plain(jax.jacfwd(outputs)(values))

def _thermal_fields(states, p, problem, jnp, apply_boundaries):
  """`(bubble_temperature, medium_temperature, vapor_fraction)`, or three Nones.

  The same three `_thermal_outputs` builds on the numpy side, and they have to be
  built the same way: the wall values are not in the state vector -- the closure
  supplies them -- so reading the raw slices would report the interior with a
  stale wall node.

  `vmap` rather than the numpy path's per-timestep Python loop. That loop would
  unroll into the traced graph once per output time, and with mass transfer each
  copy carries the bracketed solve's 29 residual evaluations, so a 60-point
  request would trace ~1700 of them. Mapping over the time axis traces one.
  """
  config, layout = problem.config, problem.layout
  if not config.bubtherm: return None, None, None

  def at_one_time(state):
    theta = state[layout.bubble_thermal]
    Tm = state[layout.medium_thermal] if layout.medium_thermal is not None else None
    kv = state[layout.vapor_fraction] if layout.vapor_fraction is not None else None
    _, Tm, kv, temperature, _ = apply_boundaries(
      theta, Tm, kv, state[layout.pressure], p, problem.medium, config.masstrans, xp=jnp
    )
    return temperature, Tm, kv

  import jax  # noqa: PLC0415  # pyright: ignore[reportMissingImports]

  temperature, Tm, kv = jax.vmap(at_one_time)(states)
  return config.T8 * temperature, None if Tm is None else config.T8 * Tm, kv
