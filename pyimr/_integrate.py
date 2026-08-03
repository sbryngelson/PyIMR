"""Where an ODE meets a solver -- the seam a second backend plugs into."""

from __future__ import annotations

import numpy as np


__all__ = ["integrate"]

def integrate(rhs, times, initial, *, args, rtol, atol, failure, label="", max_step=None, config=None):
  """Run `rhs` over `times`, returning `(states, stats)` and raising on failure."""
  from ._jax import _content_key, integrate_jax

  # The nondimensional groups are passed to the compiled program rather than baked into
  # it, so their VALUES must not appear here -- that is what made every distinct parameter
  # set a fresh compile (#163). Their names and `wave_type` do appear: the first fixes the
  # pytree the program is traced against, the second is control flow inside `_pinf`.
  # The closed-form materials reach the solve ONLY through `p` -- `_stress` and
  # `_dissipation` read `Ca, Re8, De, LAM, alphax` and dispatch on type, never on a field.
  # So their numeric fields must not key the program either, or the sweep recompiles
  # anyway. `InstantaneousMaterial` and the distributed models DO read their own fields,
  # so those keep a full content key.
  from ._materials import (
    NeoHookeanKelvinVoigt,
    NoStress,
    OldroydB,
    QuadraticKelvinVoigt,
    QuadraticZener,
    LinearMaxwell,
    Zener,
  )

  through_groups = (NoStress, NeoHookeanKelvinVoigt, QuadraticKelvinVoigt, Zener, QuadraticZener, OldroydB, LinearMaxwell)
  material = args.material
  material_key = type(material).__name__ if isinstance(material, through_groups) else _content_key(material)
  groups = args.p
  key = (
    tuple(sorted(groups)), groups["wave_type"], material_key, _content_key(args._replace(p=None, material=None)),
    len(times), float(times[0]), float(times[-1]), np.shape(initial), rtol, atol, label, max_step,
    None if config is None else config.max_steps,  # static to the compiled program, so it keys it
  )
  return integrate_jax(
    rhs, times, initial, args=args, rtol=rtol, atol=atol, failure=failure, label=label, max_step=max_step,
    cache_key=key, config=config
  )
