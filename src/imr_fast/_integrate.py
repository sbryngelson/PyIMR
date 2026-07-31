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

import numpy as np


__all__ = ["integrate"]

def integrate(rhs, times, initial, *, args, rtol, atol, failure, label="", max_step=None, config=None):
  """Run `rhs` over `times`, returning `(states, stats)` and raising on failure.

  `states` keeps scipy's `solution.y` orientation -- one row per state variable, one
  column per requested time -- because every caller and every recorded stat was written
  against it, so the orientation outlived the solver that produced it.

  One backend. `event` and `sparsity` went with the scipy branch: both were its alone, a
  terminal radius floor for `solve_ivp` and a Jacobian sparsity pattern that chose BDF
  over LSODA. The traced path chooses its solver from the configuration instead --
  `_jax._solver_for` -- which is the same idea by a different route.
  """
  from ._jax import _content_key, integrate_jax

  # Content, not `id(args[0])`. The parameters mapping is rebuilt by every `prepare`, so
  # identity keying missed on every `simulate(times, config)` and the forward solve
  # retraced: 344 ms against 3 ms.
  #
  # `max_step` belongs in the key because it is CLOSED OVER, not passed: it becomes
  # `PIDController(dtmax=...)` inside the compiled program. The identity key omitted it
  # too and got away with it, because a fresh `prepare` gave a fresh id and so a fresh
  # program anyway. Content keying removed that accident and turned the omission into a
  # silent collision -- two configurations differing only in `max_step_s` sharing one
  # program, which `test_max_step_forces_finer_integration` caught.
  key = (_content_key(args), len(times), float(times[0]), float(times[-1]), np.shape(initial), rtol, atol, label, max_step)
  return integrate_jax(
    rhs, times, initial, args=args, rtol=rtol, atol=atol, failure=failure, label=label, max_step=max_step,
    cache_key=key, config=config
  )
