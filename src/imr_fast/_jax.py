"""The JAX/diffrax backend (W11 stage 2b).

There is no physics here. Stage 2a made `_rhs` namespace-agnostic, so this
module's whole job is to hand the SHIPPED right-hand side to diffrax with
`xp=jax.numpy` and translate the result back. A transcribed copy of the
equations would have been a second implementation that can drift -- exactly the
`_mechanical` problem this migration exists to remove.

jax and diffrax are core dependencies (W11 stage 5). The import is still deferred,
but for start-up cost rather than optionality: `_jax()` is also where float64 and
the on-disk compilation cache get switched on, and both must happen exactly once
before anything touches a dtype.

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
import dataclasses
from dataclasses import dataclass

import numpy as np

from ._config import InitialState, PhysicalParameters, SimulationError, SolverStats

__all__ = ["CONFIG_PATHS", "INITIAL_PATHS", "PHYSICS_PATHS", "SCALE_PATHS", "TracedOutputs", "integrate_jax", "sensitivities_jax"]

def _jax():
  # No try/except around these. They are declared dependencies now, so an
  # ImportError here means a broken install and the raw error says so better than
  # anything this module could add.
  import diffrax  # pyright: ignore[reportMissingImports]
  import jax  # pyright: ignore[reportMissingImports]
  import jax.numpy as jnp  # pyright: ignore[reportMissingImports]

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

class _Overridden:
  """`base` with some attributes replaced, for reading only.

  `params` reads twenty-five fields off `physics` at scattered sites, and
  `initial_state_vector` five off `initial`. Tracing one of them cannot be done by
  building a new dataclass -- `__post_init__` validates with `np.isfinite`, which
  converts a tracer -- and rewriting twenty-five read sites to consult an override
  dict would put the same lookup in twenty-five places.

  So the substitution happens at attribute access instead. Read-only on purpose:
  nothing may write through it, and a missing name is the base's error to raise.
  """

  __slots__ = ("_base", "_values")

  def __init__(self, base, values):
    object.__setattr__(self, "_base", base)
    object.__setattr__(self, "_values", values)

  def __getattr__(self, name):
    values = object.__getattribute__(self, "_values")
    if name in values: return values[name]
    return getattr(object.__getattribute__(self, "_base"), name)

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

# The configuration scalars `params` already takes positionally, so tracing them
# needs no structure rebuilt around a tracer -- which is exactly why the material
# scales needed `SCALE_PATHS` and these do not. `params`' only `np.` call is
# `sqrt(P8/rho)` on concrete physics values, so it is traceable as it stands.
#
# `physics.*` and `initial.*` are NOT here. They are dataclass fields, so tracing
# one means constructing the dataclass from a tracer, and `__post_init__` validates
# with `np.isfinite`. The remaining nondifferentiable config fields are discrete or
# choose the discretisation; `_dual._NONDIFFERENTIABLE_FIELDS` lists them.
CONFIG_PATHS = ("R0", "Req", "T8", "pA", "omega", "TW", "DT", "mn")

# `physics.*` and `initial.*` reach `params` and `initial_state_vector` through
# `_Overridden` rather than as arguments, because they are read off a structure
# rather than passed. Enumerated from the dataclasses so a new field is covered
# without editing this list -- the alternative is a hand-written tuple that goes
# stale, which is #43's failure mode.
#
# `stress_state` is excluded because it is a sequence, not a scalar, and
# `_normalize_parameters` already requires a finite scalar for every path.
_PHYSICS_FIELDS = tuple(f.name for f in dataclasses.fields(PhysicalParameters))
_INITIAL_FIELDS = tuple(f.name for f in dataclasses.fields(InitialState) if f.name != "stress_state")
PHYSICS_PATHS = tuple(f"physics.{name}" for name in _PHYSICS_FIELDS)
INITIAL_PATHS = tuple(f"initial.{name}" for name in _INITIAL_FIELDS)

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
  from ._prepare import _material_scales, initial_state_vector, medium_with_parameters, params
  from ._rhs import _rhs, _rhs_args
  from ._stress import _stress
  from ._thermal import _apply_thermal_boundaries

  config = problem.config
  covered = set(SCALE_PATHS) | set(CONFIG_PATHS) | set(PHYSICS_PATHS) | set(INITIAL_PATHS)
  unknown = [path for path in paths if path not in covered]
  if unknown: raise ValueError(f"jax sensitivities cover {sorted(covered)}; got {unknown}")
  base = np.asarray(_material_scales(config.material), dtype=float)
  # ONE traced vector in the caller's order, whichever group each path belongs to:
  # `jacfwd` differentiates with respect to a single input, and the tangent columns
  # have to come back in the order the caller asked for.
  def started(path):
    if path in SCALE_PATHS: return base[SCALE_PATHS[path]]
    group, _, field = path.partition(".")
    return getattr(config, path) if not field else getattr(getattr(config, group), field)

  traced_base = np.asarray([started(path) for path in paths], dtype=float)
  layout = problem.layout
  has_stress = layout.stress.stop > layout.stress.start
  grid_s = np.asarray(times, dtype=float)
  collapse_state = None if problem.collapse_stats is None else np.asarray(problem.collapse_stats.stress_state, dtype=float)

  def outputs(traced, grid_s):
    # Scales override `params`' `scales=` argument; config scalars ARE its
    # positional arguments. Two destinations, one traced vector.
    scales = jnp.asarray(base)
    scalars = {path: getattr(config, path) for path in CONFIG_PATHS}
    physics_values: dict = {}
    initial_values: dict = {}
    for index, path in enumerate(paths):
      if path in SCALE_PATHS: scales = scales.at[SCALE_PATHS[path]].set(traced[index])
      elif path in CONFIG_PATHS: scalars[path] = traced[index]
      elif path in PHYSICS_PATHS: physics_values[path.split(".", 1)[1]] = traced[index]
      else: initial_values[path.split(".", 1)[1]] = traced[index]
    physics = config.physics if not physics_values else _Overridden(config.physics, physics_values)
    initial = config.initial if not initial_values else _Overridden(config.initial, initial_values)
    p = params(
      scalars["R0"], scalars["Req"], config.material, config.vapor, scalars["T8"], scalars["pA"], scalars["omega"],
      scalars["TW"], scalars["DT"], scalars["mn"], config.wave_type, config.bubtherm, config.masstrans, physics,
      xp=jnp, scales=tuple(scales),
    )
    # Rebuilt from `p`, not reused from `prepare`. `Pb`, `kv0` and `Uc` all come
    # from `p`, so with R0, Req or T8 traced the STARTING state carries a tangent
    # and a concrete `problem.initial_state` would contribute exactly zero.
    start = initial_state_vector(config, layout, p, collapse_state, xp=jnp, initial=initial)
    # The prepared thermal operators pass through as constants, and that is
    # correct rather than convenient: the medium's flux weights are built from
    # `chi`, `iota`, `Fom` and `L_heat_star`, and its grid powers from `Lt` --
    # all thermal or geometric, none of them derived from the five material
    # scales this traces. `Br` does carry viscosity, but it lives in `p`, which
    # is rebuilt from tracers above. Checked against the scipy tangents for
    # bubtherm, medtherm and masstrans rather than argued from the parameter
    # list alone.
    # Rebuilt once and used everywhere: `_rhs` through `args`, and the temperature
    # outputs below, which run the same wall closure.
    medium = medium_with_parameters(problem.medium, p, xp=jnp)
    args = _rhs_args(problem, p, medium=medium)
    grid = grid_s / p["t0"]
    solution = diffrax.diffeqsolve(
      diffrax.ODETerm(lambda t, y, _a: jnp.asarray(_rhs(t, y, *args, xp=jnp))), diffrax.Tsit5(),
      t0=grid[0], t1=grid[-1], dt0=None, y0=start,
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
    # `scalars["R0"]`, not `config.R0`: the dimensional outputs scale by it, so
    # with R0 among the parameters the concrete field would drop that term from
    # its own tangent -- a wrong derivative for the one output named after it.
    length = scalars["R0"]
    derived = jnp.stack(
      [radius, radius * length, velocity * length / p["t0"], pressure * p["P8"], stress * p["P8"]], axis=1
    )
    return (states, derived, *_thermal_fields(states, p, problem, medium, jnp, _apply_thermal_boundaries, scalars["T8"]))

  # Traced and compiled ONCE per (problem, parameter set, grid length), the same
  # treatment `integrate_jax` gives the forward solve. Without it every call
  # retraced the whole `jacfwd`, which measured 1121 ms against the compiled numba
  # route's 47 ms -- 24x slower for no reason other than the missing cache.
  def build():
    return jax.jit(outputs), jax.jit(jax.jacfwd(outputs))

  primal_fn, tangent_fn = _cached((id(problem), tuple(paths), grid_s.size, "sensitivities"), build)
  values, grid = jnp.asarray(traced_base), jnp.asarray(grid_s)

  def required(item):
    return np.asarray(jax.block_until_ready(item), dtype=float)

  def optional(item):
    return None if item is None else required(item)

  def plain(group):
    # Unpacked by name rather than splatted, so the two always-present fields
    # stay non-optional to a type checker.
    states, derived, bubble, medium, vapor = group
    return TracedOutputs(required(states), required(derived), optional(bubble), optional(medium), optional(vapor))

  return plain(primal_fn(values, grid)), plain(tangent_fn(values, grid))

def _thermal_fields(states, p, problem, medium, jnp, apply_boundaries, reference_temperature):
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
      theta, Tm, kv, state[layout.pressure], p, medium, config.masstrans, xp=jnp
    )
    return temperature, Tm, kv

  import jax  # noqa: PLC0415  # pyright: ignore[reportMissingImports]

  temperature, Tm, kv = jax.vmap(at_one_time)(states)
  # `reference_temperature`, not `config.T8`. Both outputs are a RATIO times that
  # scale, so with T8 among the parameters the concrete field drops the product
  # rule's other term -- measured as a 1.11 relative error on the bubble
  # temperature tangent, the same mistake `derived` made with `config.R0`.
  return reference_temperature * temperature, None if Tm is None else reference_temperature * Tm, kv
