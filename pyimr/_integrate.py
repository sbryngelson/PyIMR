"""Where an ODE meets a solver -- the seam a second backend plugs into."""

from __future__ import annotations

import numpy as np


__all__ = ["integrate"]

def integrate(rhs, times, initial, *, args, rtol, atol, failure, label="", max_step=None, config=None):
  """Run `rhs` over `times`, returning `(states, stats)` and raising on failure."""
  from ._jax import _content_key, integrate_jax

  key = (_content_key(args), len(times), float(times[0]), float(times[-1]), np.shape(initial), rtol, atol, label, max_step)
  return integrate_jax(
    rhs, times, initial, args=args, rtol=rtol, atol=atol, failure=failure, label=label, max_step=max_step,
    cache_key=key, config=config
  )
