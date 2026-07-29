"""Where an ODE meets a solver -- the seam a second backend plugs into.

W11 stage 1. Both integrating call sites, the forward solve and the sensitivity
solve, chose a method, invoked `solve_ivp`, and turned the outcome into a
`SolverStats` in the same twenty lines with two words different: a backend label
suffix and an error prefix. They now share this.

Nothing about a second backend is here yet, deliberately. Stage 1's only claim
is that routing the solve through one function leaves every trajectory
BIT-IDENTICAL, which is checkable now; a `backend` field on `SimulationConfig`
arrives with stage 2, when there is a second value for it to take. A public
option accepting exactly one value would be dead surface.

The method choice is already static -- BDF when the problem has a Jacobian
sparsity pattern, LSODA otherwise -- and W11 measured that stiffness is a
property of the configuration rather than of a phase of the solve
(`|lambda_max/lambda_min|` is 1.18 mechanical against 6.9e+16 coupled spectral,
with median equal to worst in every case). So this is the right shape for a
backend that must pick a solver up front rather than switch mid-flight.
"""

from __future__ import annotations

from time import perf_counter

from scipy.integrate import solve_ivp

from ._config import SimulationError, SolverStats, _solve_stats
from ._stress import _MaterialDomainError

__all__ = ["integrate"]

def integrate(rhs, times, initial, *, args, event, sparsity, rtol, atol, failure, label="", max_step=None, backend="scipy"):
  """Run `rhs` over `times`, returning `(states, stats)` and raising on failure.

  `states` is scipy's `solution.y` orientation -- one row per state variable, one
  column per requested time -- rather than a solver's own result object, so the
  two backends present the same thing and callers need no branch.

  `failure` is the sentence a `SimulationError` leads with; `label` suffixes the
  recorded backend name, which is how the sensitivity solve reports itself as
  `scipy-lsoda-forward` rather than `scipy-lsoda`.
  """
  if backend == "jax":
    from ._jax import integrate_jax

    return integrate_jax(rhs, times, initial, args=args, rtol=rtol, atol=atol, failure=failure, label=label, max_step=max_step)
  method = "BDF" if sparsity is not None else "LSODA"
  options = {"jac_sparsity": sparsity} if sparsity is not None else {}
  if max_step is not None: options["max_step"] = max_step
  backend = f"scipy-{method.lower()}{label}"
  started = perf_counter()
  try:
    solution = solve_ivp(
      rhs, (times[0], times[-1]), initial, t_eval=times, args=args, events=event, method=method, rtol=rtol, atol=atol, **options
    )
  except _MaterialDomainError as error:
    message = f"material domain failure: {error}"
    stats = SolverStats(backend=backend, success=False, message=message, nfev=0, njev=0, nlu=0, elapsed_s=perf_counter() - started)
    raise SimulationError(f"{failure}: {message}", stats) from error
  success, message, stats = _solve_stats(solution, times, backend, perf_counter() - started)
  if not success: raise SimulationError(f"{failure}: {message}", stats)
  return solution.y, stats
