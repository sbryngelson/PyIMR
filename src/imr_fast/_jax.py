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

import numpy as np

from ._config import SimulationError, SolverStats

__all__ = ["available", "integrate_jax", "unsupported_reason"]

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
  return jax, jnp, diffrax

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

  Stage 2b covers the mechanical path only. The thermal fields are assembled by
  in-place slice assignment (`thetadot[-1] = 0.0` and its neighbours), which
  needs `jnp.at[].set()` rather than a namespace swap, and distributed stress
  packs its output into a preallocated buffer for the same reason. Both are
  stage 4 work; naming them here beats a `KeyError` three frames down.
  """
  if config.bubtherm: return "bubtherm=1 (thermal PDE) is not on the jax backend yet -- see PLAN.md W11 stage 4"
  if config.sampled_forcing is not None: return "sampled_forcing is not on the jax backend yet"
  if getattr(config.material, "points", None) is not None:
    return "distributed-memory materials are not on the jax backend yet -- their output is packed into a preallocated buffer"
  return None

def integrate_jax(rhs, times, initial, *, args, rtol, atol, failure, label="", max_step=None):
  """Run `rhs` through diffrax, returning `(times, states, stats)`.

  `states` comes back in scipy's `solution.y` orientation -- one row per state,
  one column per requested time -- so callers need no branch of their own.
  """
  jax, jnp, diffrax = _jax()
  from time import perf_counter

  def field(t, y, _args): return jnp.asarray(rhs(t, y, *args, xp=jnp))

  controller = diffrax.PIDController(rtol=rtol, atol=atol, dtmax=max_step)
  started = perf_counter()
  try:
    solution = diffrax.diffeqsolve(
      diffrax.ODETerm(field), diffrax.Tsit5(), t0=float(times[0]), t1=float(times[-1]), dt0=None,
      y0=jnp.asarray(np.asarray(initial, dtype=float)), stepsize_controller=controller,
      saveat=diffrax.SaveAt(ts=jnp.asarray(np.asarray(times, dtype=float))), max_steps=1_000_000,
      throw=False,
    )
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
