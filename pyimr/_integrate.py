"""Where an ODE meets a solver -- the seam a second backend plugs into."""

from __future__ import annotations

import numpy as np

from ._materials import (
  Giesekus,
  InstantaneousMaterial,
  LinearMaxwell,
  LinearPTT,
  NeoHookeanKelvinVoigt,
  NoStress,
  OldroydB,
  QuadraticKelvinVoigt,
  QuadraticZener,
  Zener,
  law_values,
)


__all__ = ["THROUGH_GROUPS", "integrate", "shares_one_program"]

# Materials whose fields reach the solve ONLY through the nondimensional groups, so the
# compiled program can be keyed by type and a whole parameter sweep compiles once (#163).
# Everything else is keyed by content: one compile per distinct parameter set.
# `Giesekus` and `LinearPTT` belong here since #196 routed mobility and extensibility
# through `p` as well. Their structural fields -- points, extent, quadrature -- still get
# their own program without being named here, because they change the PREPARED arrays,
# which the argument content key already covers.
def shares_one_program(material) -> bool:
  """Whether a sweep over this material's numbers reuses one compiled program.

  True when every number the solve reads travels through `p`. False for `Ogden`, whose
  variable-length tuples a fixed-width vector cannot carry, so it is keyed by content and
  compiles once per parameter set (#196).
  """
  if isinstance(material, THROUGH_GROUPS): return True
  if not isinstance(material, InstantaneousMaterial): return False
  return law_values(material.elastic) is not None and law_values(material.viscous) is not None


THROUGH_GROUPS = (
  NoStress, NeoHookeanKelvinVoigt, QuadraticKelvinVoigt, Zener, QuadraticZener, OldroydB, LinearMaxwell,
  Giesekus, LinearPTT,
)

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
  material = args.material
  if isinstance(material, THROUGH_GROUPS): material_key = type(material).__name__
  elif isinstance(material, InstantaneousMaterial) and shares_one_program(material):
    # the laws' numbers travel in `p`; only their TYPES and the quadrature size, which
    # sets array shapes, still have to key the program (#196)
    material_key = (
      "InstantaneousMaterial", type(material.elastic).__name__, type(material.viscous).__name__,
      material.quadrature_points,
    )
  else: material_key = _content_key(material)
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
