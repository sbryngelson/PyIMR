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

import numpy as np

from ._config import SimulationError, SolverStats

__all__ = ["SCALE_PATHS", "available", "integrate_jax", "sensitivities_jax", "unsupported_reason"]

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

  Naming the reason here beats a tracer error three frames inside diffrax. The
  list is short now: everything except mass transfer, whose wall closure is an
  iterative solve, and a sampled forcing history, whose interpolation searches
  its knots.
  """
  # Mass transfer is the one thermal branch still out: `_wall_theta_bw_full`
  # solves the wall temperature by secant iteration with a fallback ladder, and
  # a data-dependent loop needs `lax.custom_root` rather than a namespace swap.
  # `medtherm` alone does NOT -- #57 made that wall closure closed form.
  if config.masstrans: return "masstrans=1 is not on the jax backend: its wall closure is an iterative solve -- see PLAN.md W11"
  if config.sampled_forcing is not None: return "sampled_forcing is not on the jax backend yet"
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

  Returns `(states, derived, state_tangent, derived_tangent)`. `derived` is
  ordered `radius_ratio, radius_m, wall_velocity_m_s, internal_pressure_pa,
  stress_integral_pa`; the tangents carry a trailing parameter axis.
  """
  jax, jnp, diffrax = _jax()
  from ._config import _WallState
  from ._prepare import _material_scales, params
  from ._rhs import _rhs
  from ._stress import _stress

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
    args = (p, config.material, config.radial, 0, None, None, None, 0, None, 0, _WallState(), problem.forcing,
            problem.instantaneous_material, None)
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
    pressure = (p["Pb"] - p["Pv"]) * radius ** (-3.0 * p["kappa"]) + p["Pv"]
    stress = _stress(config.material, p, radius, velocity, stress_state, problem.instantaneous_material, False, xp=jnp)[0]
    derived = jnp.stack(
      [radius, radius * config.R0, velocity * config.R0 / p["t0"], pressure * p["P8"], stress * p["P8"]], axis=1
    )
    return states, derived

  values = jnp.asarray(base[slots])
  primal, tangent = outputs(values), jax.jacfwd(outputs)(values)
  return tuple(np.asarray(jax.block_until_ready(item), dtype=float) for pair in (primal, tangent) for item in pair)
