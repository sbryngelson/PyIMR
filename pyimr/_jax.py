"""The JAX/diffrax backend (W11 stage 2b)."""

from __future__ import annotations

import os
import pathlib
import dataclasses
from dataclasses import dataclass

import numpy as np

from collections.abc import Mapping

from ._config import InitialState, PhysicalParameters, SimulationError, SolverStats
from ._materials import Zener

__all__ = ["CONFIG_PATHS", "INITIAL_PATHS", "PHYSICS_PATHS", "SCALE_PATHS", "TracedOutputs", "ensemble_states_jax", "integrate_jax", "sensitivities_jax", "state_tangents_jax"]

# chord scales but its reused jacobian ruins tangents (1e-3 vs a 5e-5 bound); newton is
# exact but stalls past ~80 nodes. forward answers agree to 5e-12, so split on use. #120
_CHORD_ABOVE = 80

def _solver_for(config, diffrax, *, differentiating=True):
  if config.bubtherm and config.thermal == "spectral":
    import optimistix  # pyright: ignore[reportMissingImports]

    nodes = config.Nt + (config.Mt if config.medtherm else 0)
    if not differentiating and nodes > _CHORD_ABOVE:
      return diffrax.Kvaerno5(root_finder=diffrax.VeryChord(rtol=1e-8, atol=1e-8)), "kvaerno5-verychord"
    # Forward: `optimistix.Chord` -- one Jacobian per STAGE, reused inside that stage's
    # solve, rather than one per Newton iteration. The Jacobian costs 5.4x the RHS, so that
    # is where the money is: measured 1.600x (12/12 paired, single core), step count
    # IDENTICAL at 333, trajectory agreeing to 3.6e-15.
    #
    # NOT `diffrax.VeryChord`, which reuses across STEPS and collapses step control (9574
    # steps against 333, 12x worse). Reuse within a stage leaves the controller intact,
    # which the identical step count is the evidence for.
    #
    # Differentiating: `Newton`. Chord reintroduces the #133 failure -- an off-equilibrium
    # start drives the solver through unphysical trial states, and the reused Jacobian
    # hands lineax a non-finite operator where a fresh one survives. The forward path never
    # reaches that, so the split buys the speed without the fragility.
    if differentiating:
      return diffrax.Kvaerno5(root_finder=optimistix.Newton(rtol=1e-8, atol=1e-8)), "kvaerno5"
    return diffrax.Kvaerno5(root_finder=optimistix.Chord(rtol=1e-8, atol=1e-8)), "kvaerno5-chord"
  return diffrax.Tsit5(), "tsit5"

def _jax():
  import diffrax  # pyright: ignore[reportMissingImports]
  import jax  # pyright: ignore[reportMissingImports]
  import jax.numpy as jnp  # pyright: ignore[reportMissingImports]

  jax.config.update("jax_enable_x64", True)
  _enable_compilation_cache(jax)
  return jax, jnp, diffrax

def _enable_compilation_cache(jax):
  location = os.environ.get("IMR_FAST_JAX_CACHE")
  if location == "": return
  if location is None:
    location = str(pathlib.Path(os.environ.get("XDG_CACHE_HOME", pathlib.Path.home() / ".cache")) / "pyimr" / "jax")
  jax.config.update("jax_compilation_cache_dir", location)
  jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
  jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)

_COMPILED: dict = {}
_COMPILED_LIMIT = 192

def _content_key(value):
  if isinstance(value, np.ndarray):
    return ("array", value.shape, str(value.dtype), value.tobytes())
  if isinstance(value, (str, bytes, bool, int, float, type(None))): return value
  if isinstance(value, Mapping):
    return ("map", tuple((key, _content_key(value[key])) for key in sorted(value)))
  if isinstance(value, (tuple, list)):
    return ("seq", tuple(_content_key(item) for item in value))
  if isinstance(value, slice): return ("slice", value.start, value.stop, value.step)
  if dataclasses.is_dataclass(value) and not isinstance(value, type):
    return (type(value).__name__, tuple((f.name, _content_key(getattr(value, f.name))) for f in dataclasses.fields(value)))
  return (type(value).__name__, repr(value))

def _cached(key, build):
  hit = _COMPILED.get(key)
  if hit is None:
    if len(_COMPILED) >= _COMPILED_LIMIT: del _COMPILED[next(iter(_COMPILED))]
    return _COMPILED.setdefault(key, build())
  _COMPILED[key] = _COMPILED.pop(key)
  return hit

def integrate_jax(rhs, times, initial, *, args, rtol, atol, failure, label="", max_step=None, cache_key=None, config=None):
  """Run `rhs` through diffrax, returning `(times, states, stats)`."""
  jax, jnp, diffrax = _jax()
  from time import perf_counter

  solver, solver_name = (diffrax.Tsit5(), "tsit5") if config is None else _solver_for(config, diffrax, differentiating=False)
  budget = 1_000_000 if config is None else int(config.max_steps)

  # The nondimensional groups are traced ARGUMENTS, not constants closed over. Baking them
  # in made the cache key depend on their values, so every distinct parameter set was a
  # fresh XLA compile: a 20-point sweep cost 27.3 s against 0.04 s of actual solving (#163).
  #
  # `wave_type` stays static because `_pinf` branches on it. It is the only int among the
  # 44 entries, and a branch on a tracer is an error rather than a slow path.
  from ._prepare import medium_with_parameters

  static = {"wave_type": args.p["wave_type"]}
  dynamic = {name: float(value) for name, value in args.p.items() if name not in static}
  medium = args.mt

  def program(chosen):
    def solve(grid, start, values):
      merged = {**values, **static}
      # `medium`'s wall weights are built FROM p, so they have to be rebuilt for the
      # traced values rather than reused from preparation.
      rebuilt = None if medium is None else medium_with_parameters(medium, merged, xp=jnp)
      inner = args._replace(p=merged, mt=rebuilt)
      return diffrax.diffeqsolve(
        diffrax.ODETerm(lambda t, y, _a: jnp.asarray(rhs(t, y, *inner, xp=jnp))), chosen,
        t0=grid[0], t1=grid[-1], dt0=None, y0=start,
        stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol, dtmax=max_step),
        saveat=diffrax.SaveAt(ts=grid), max_steps=budget, throw=False,
      )

    return jax.jit(solve)

  grid_values = jnp.asarray(np.asarray(times, dtype=float))
  start_values = jnp.asarray(np.asarray(initial, dtype=float))
  dynamic_values = {name: jnp.asarray(value) for name, value in dynamic.items()}

  def run(chosen, name, key):
    compiled = _cached(key, lambda: program(chosen)) if key is not None else program(chosen)
    started = perf_counter()
    try:
      solution = compiled(grid_values, start_values, dynamic_values)
    except Exception as error:  # noqa: BLE001 - any solver failure is one failure
      elapsed = perf_counter() - started
      stats = SolverStats(backend=f"jax-{name}{label}", success=False, message=str(error), nfev=0, njev=0, nlu=0, elapsed_s=elapsed)
      raise SimulationError(f"{failure}: {error}", stats) from error
    elapsed = perf_counter() - started
    states = np.asarray(jax.block_until_ready(solution.ys), dtype=float).T
    ok = bool(solution.result == diffrax.RESULTS.successful)
    finite = bool(np.all(np.isfinite(states)))
    message = "The solver successfully reached the end of the integration interval." if ok else str(solution.result)
    if ok and not finite: message = f"{message}; solution contains non-finite states"
    stats = SolverStats(
      backend=f"jax-{name}{label}", success=ok and finite, message=message,
      nfev=int(solution.stats.get("num_steps", 0)), njev=0, nlu=0, elapsed_s=elapsed,
    )
    return states, stats, solution.result

  states, stats, result = run(solver, solver_name, cache_key)

  if not stats.success and result == diffrax.RESULTS.max_steps_reached and solver_name == "tsit5":
    states, stats, _ = run(diffrax.Dopri8(), "dopri8", None if cache_key is None else (*cache_key, "dopri8"))

  if not stats.success: raise SimulationError(f"{failure}: {stats.message}", stats)
  return states, stats

class _Overridden:

  __slots__ = ("_base", "_values")

  def __init__(self, base, values):
    object.__setattr__(self, "_base", base)
    object.__setattr__(self, "_values", values)

  def __getattr__(self, name):
    values = object.__getattribute__(self, "_values")
    if name in values: return values[name]
    return getattr(object.__getattribute__(self, "_base"), name)

def _substituted(config, paths, traced, base, jnp):
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
  return scalars, tuple(scales), physics, initial

def _collapse_tangents(problem, paths, base_values, base_scales):
  jax, jnp, diffrax = _jax()
  from ._prepare import _collapse_zener_rhs, params
  from ._rhs import _rhs

  config = problem.config
  stats = problem.collapse_stats
  memory_width = problem.layout.stress.stop - problem.layout.stress.start
  upstream_zener = isinstance(config.material, Zener)
  event_time = stats.maximum_time_nondimensional

  def flow(inputs):
    scalars, scales, physics, _initial = _substituted(config, paths, inputs, base_scales, jnp)
    precursor = params(
      scalars["R0"], scalars["Req"], config.material, config.vapor, scalars["T8"], physics=physics,
      bubtherm=1, masstrans=config.masstrans, xp=jnp, scales=scales,
    )
    precursor = dict(precursor)
    precursor["kappa"] = 1.0
    start = jnp.zeros(2 + memory_width).at[0].set(precursor["req"]).at[1].set(inputs[-1])

    def rhs(_t, state, _a):
      if upstream_zener: return jnp.asarray(_collapse_zener_rhs(state, precursor))
      return jnp.asarray(_rhs(_t, state, precursor, config.material, config.radial, xp=jnp,
                              instantaneous_material=problem.instantaneous_material,
                              distributed_stress=problem.distributed_stress))

    solution = diffrax.diffeqsolve(
      diffrax.ODETerm(rhs), diffrax.Tsit5(), t0=0.0, t1=event_time, dt0=None, y0=start,
      stepsize_controller=diffrax.PIDController(rtol=min(config.rtol, 1e-9), atol=min(config.atol, 1e-11)),
      saveat=diffrax.SaveAt(t1=True), max_steps=config.max_steps, adjoint=diffrax.ForwardMode(),
    )
    end = solution.ys[-1]
    return end, rhs(event_time, end, None)

  point = jnp.asarray([*base_values, stats.initial_velocity_nondimensional])
  values, derivatives = flow(point), jax.jacfwd(flow)(point)
  end, slope = (np.asarray(item, dtype=float) for item in values)
  jacobian = np.asarray(derivatives[0], dtype=float)

  acceleration = slope[1]
  if abs(acceleration) < 1e-14: raise SimulationError("collapse precursor maximum is degenerate: zero acceleration")
  time_tangents = -jacobian[1, :] / acceleration
  memory_tangents = jacobian[2:, :] + slope[2:, None] * time_tangents[None, :]
  radius_tangents = jacobian[0, :]
  radius_velocity = radius_tangents[-1]
  if abs(radius_velocity) < 1e-14: raise SimulationError("collapse shooting root has a singular velocity derivative")
  velocity_tangents = -radius_tangents[:-1] / radius_velocity
  return end[2:], memory_tangents[:, :-1] + memory_tangents[:, [-1]] * velocity_tangents[None, :]

_DERIVED_ORDER = ("radius_ratio", "radius_m", "wall_velocity_m_s", "internal_pressure_pa", "stress_integral_pa")

@dataclass(frozen=True, slots=True)
class TracedOutputs:
  """What `sensitivities_jax` differentiates, returned once as values and once as"""

  states: np.ndarray
  derived: np.ndarray
  bubble_temperature: np.ndarray | None
  medium_temperature: np.ndarray | None
  vapor_fraction: np.ndarray | None
  # Solver steps, primal only. `jacfwd` never sees it -- an integer count has no
  # derivative, and including it in the differentiated outputs is an error, not a zero.
  steps: np.ndarray | None = None

SCALE_PATHS = {
  "material.shear_modulus_pa": 0,
  "material.viscosity_pa_s": 1,
  "material.relaxation_time_s": 2,
  "material.retardation_time_s": 3,
  "material.stiffening": 4,
  "material.mobility": 5,
  "material.extensibility": 5,
  "material.second_relaxation_time_s": 6,
  "material.second_share": 7,
  "material.thinning_time_s": 8,
  "material.power_index": 9,
  "material.cubic": 10,
}

CONFIG_PATHS = ("R0", "Req", "T8", "pA", "omega", "TW", "DT", "mn")

_PHYSICS_FIELDS = tuple(f.name for f in dataclasses.fields(PhysicalParameters))
_INITIAL_FIELDS = tuple(f.name for f in dataclasses.fields(InitialState) if f.name != "stress_state")
PHYSICS_PATHS = tuple(f"physics.{name}" for name in _PHYSICS_FIELDS)
INITIAL_PATHS = tuple(f"initial.{name}" for name in _INITIAL_FIELDS)

def _traced_flow(problem, times, *, throw=True):
  """`(jax, solve, grid)` for one prepared problem, where `solve(y0)` returns the saved states.

  Shared by the tangent operator and the ensemble: both differ only in what they map over.
  """
  jax, jnp, diffrax = _jax()
  from ._rhs import _rhs, _rhs_args

  config = problem.config
  p = problem.parameters
  args = _rhs_args(problem, p, medium=problem.medium)
  grid = jnp.asarray(np.asarray(times, dtype=float) / p["t0"])

  def solve(y0):
    return diffrax.diffeqsolve(
      diffrax.ODETerm(lambda t, y, _a: jnp.asarray(_rhs(t, y, *args, xp=jnp))), _solver_for(config, diffrax)[0],
      t0=grid[0], t1=grid[-1], dt0=None, y0=y0,
      stepsize_controller=diffrax.PIDController(rtol=config.rtol, atol=config.atol),
      saveat=diffrax.SaveAt(ts=grid), max_steps=config.max_steps, adjoint=diffrax.ForwardMode(), throw=throw,
    )

  return jax, solve, grid

def ensemble_states_jax(problem, times, members):
  """`(states, ok)` for a batch of initial states, advanced under one `vmap`.

  `throw=False` deliberately: a vmapped solve shares one program, so a single diverged
  member would otherwise take the whole batch down with it. Ensembles are drawn from a
  distribution and their tails do diverge, so the failure is reported per member instead.
  """
  jax, solve, grid = _traced_flow(problem, times, throw=False)
  _, _, diffrax = _jax()
  batch = np.asarray(members, dtype=float)
  key = (_content_key(problem), int(grid.size), batch.shape, "ensemble")
  compiled = _cached(key, lambda: jax.jit(jax.vmap(solve)))
  solution = jax.block_until_ready(compiled(batch))
  states = np.asarray(solution.ys, dtype=float)
  ok = np.asarray(solution.result == diffrax.RESULTS.successful)
  return states, ok

def state_tangents_jax(problem, times, state):
  """`(states, d states / d y0)` -- the tangent linear operator along the trajectory.

  Forward mode. That is nominally one solve per state component, but the tangent columns
  share a step controller and a factorization, so measured warm cost is far below it:
  3.6x a plain solve at width 10, 4.5x at 28, 5.5x at 36. Sublinear, not proportional.
  """
  jax, solve, grid = _traced_flow(problem, times)
  start = np.asarray(state, dtype=float)
  primal_of = (lambda y0: solve(y0).ys)

  key = (_content_key(problem), int(grid.size), int(start.size), "state-tangents")
  primal, tangent = _cached(key, lambda: (jax.jit(primal_of), jax.jit(jax.jacfwd(primal_of))))
  states = np.asarray(jax.block_until_ready(primal(start)), dtype=float)
  jacobian = np.asarray(jax.block_until_ready(tangent(start)), dtype=float)
  return states, jacobian

def sensitivities_jax(problem, times, paths, at=None, values_only=False):
  """`(outputs, tangents)` for the mechanical path, both from one `jacfwd`."""
  jax, jnp, diffrax = _jax()
  from ._prepare import _material_scales, forcing_with_parameters, initial_state_vector, medium_with_parameters, params
  from ._rhs import _rhs, _rhs_args
  from ._stress import _stress
  from ._thermal import _apply_thermal_boundaries

  config = problem.config
  covered = set(SCALE_PATHS) | set(CONFIG_PATHS) | set(PHYSICS_PATHS) | set(INITIAL_PATHS)
  unknown = [path for path in paths if path not in covered]
  if unknown: raise ValueError(f"jax sensitivities cover {sorted(covered)}; got {unknown}")
  base = np.asarray(_material_scales(config.material), dtype=float)
  def started(path):
    if path in SCALE_PATHS: return base[SCALE_PATHS[path]]
    group, _, field = path.partition(".")
    return getattr(config, path) if not field else getattr(getattr(config, group), field)

  traced_base = np.asarray([started(path) for path in paths], dtype=float)
  layout = problem.layout
  has_stress = layout.stress.stop > layout.stress.start
  grid_s = np.asarray(times, dtype=float)
  forcing_reference = (problem.parameters["t0"], problem.parameters["P8"])
  collapse_value, collapse_jacobian = None, None
  if problem.collapse_stats is not None:
    collapse_value, collapse_jacobian = _cached(
      (_content_key(problem), tuple(paths), "collapse"), lambda: _collapse_tangents(problem, paths, traced_base, base)
    )

  def outputs(traced, grid_s):
    scalars, scales, physics, initial = _substituted(config, paths, traced, base, jnp)
    p = params(
      scalars["R0"], scalars["Req"], config.material, config.vapor, scalars["T8"], scalars["pA"], scalars["omega"],
      scalars["TW"], scalars["DT"], scalars["mn"], config.wave_type, config.bubtherm, config.masstrans, physics,
      xp=jnp, scales=scales,
    )
    collapse = collapse_value
    if collapse_jacobian is not None:
      collapse = jnp.asarray(collapse_value) + collapse_jacobian @ (traced - jnp.asarray(traced_base))
    start = initial_state_vector(config, layout, p, collapse, xp=jnp, initial=initial)
    medium = medium_with_parameters(problem.medium, p, xp=jnp)
    args = _rhs_args(problem, p, medium=medium)
    args = args._replace(forcing=forcing_with_parameters(problem.forcing, p, forcing_reference, xp=jnp))
    grid = grid_s / p["t0"]
    solution = diffrax.diffeqsolve(
      diffrax.ODETerm(lambda t, y, _a: jnp.asarray(_rhs(t, y, *args, xp=jnp))), _solver_for(config, diffrax)[0],
      t0=grid[0], t1=grid[-1], dt0=None, y0=start,
      stepsize_controller=diffrax.PIDController(rtol=config.rtol, atol=config.atol),
      saveat=diffrax.SaveAt(ts=grid), max_steps=config.max_steps, adjoint=diffrax.ForwardMode(),
    )
    states = solution.ys
    radius, velocity = states[:, 0], states[:, 1]
    stress_state = states[:, layout.stress].T if has_stress else None
    if layout.pressure is None:
      pressure = (p["Pb"] - p["Pv"]) * radius ** (-3.0 * p["kappa"]) + p["Pv"]
    else:
      pressure = states[:, layout.pressure]
    stress = _stress(config.material, p, radius, velocity, stress_state, problem.instantaneous_material, False, xp=jnp)[0]
    length = scalars["R0"]
    derived = jnp.stack(
      [radius, radius * length, velocity * length / p["t0"], pressure * p["P8"], stress * p["P8"]], axis=1
    )
    fields = _thermal_fields(states, p, problem, medium, jnp, _apply_thermal_boundaries, scalars["T8"])
    return (states, derived, *fields, solution.stats["num_steps"])

  def differentiable(traced, grid_s):
    """`outputs` without the step count, which is what `jacfwd` may be applied to."""
    return outputs(traced, grid_s)[:-1]

  point = traced_base if at is None else np.asarray(at, dtype=float)
  batched = point.ndim == 2
  if point.shape[-1] != len(paths):
    raise ValueError(f"`at` must supply {len(paths)} values per point; got {point.shape}")

  def build():
    if batched:
      return jax.jit(jax.vmap(outputs, in_axes=(0, None))), jax.jit(jax.vmap(jax.jacfwd(differentiable), in_axes=(0, None)))
    return jax.jit(outputs), jax.jit(jax.jacfwd(differentiable))

  program_key = (_content_key(problem), tuple(paths), grid_s.size, point.shape, "sensitivities")
  primal_fn, tangent_fn = _cached(program_key, build)
  values, grid = jnp.asarray(point), jnp.asarray(grid_s)

  def required(item):
    return np.asarray(jax.block_until_ready(item), dtype=float)

  def optional(item):
    return None if item is None else required(item)

  def plain(group):
    """Primal groups carry a trailing step count; tangent groups do not."""
    states, derived, bubble, medium, vapor, *rest = group
    steps = required(rest[0]) if rest else None
    return TracedOutputs(required(states), required(derived), optional(bubble), optional(medium), optional(vapor), steps)

  # `jax.jit` traces lazily, so leaving `tangent_fn` uncalled skips the whole `jacfwd`
  # rather than merely discarding it.
  if values_only: return plain(primal_fn(values, grid)), None
  return plain(primal_fn(values, grid)), plain(tangent_fn(values, grid))

def _thermal_fields(states, p, problem, medium, jnp, apply_boundaries, reference_temperature):
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
  return reference_temperature * temperature, None if Tm is None else reference_temperature * Tm, kv
