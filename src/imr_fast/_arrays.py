"""The one array helper that has to work in both namespaces.

Was `_autodiff.py`, and held a forward-mode `Dual` scalar with a full set of
operator overloads and ufunc dispatch. Nothing constructs one any more: W11
stage 5 left a single traced backend, and jax computes its own tangents. The
class outlived its consumers by exactly one release and was found by coverage,
at 36% against 88% for the package -- unreferenced code cannot be exercised, and
that is what the number was saying.

What went with it: `seed` (built a `Dual`), `unpack` (split an array of them),
`is_real_scalar` (asked whether something was NOT one), and `primal` /
`primal_array`, which stripped tangents and are the identity without a `Dual` to
strip. Two object-dtype branches went too -- one in `_stress._viscosity_and_tangent`
and one in `_thermal._distributed_dissipation` -- because `Dual` was the only
thing that ever made a numpy array hold objects.
"""

from __future__ import annotations

def at_set(array, index, value):
  """`array[index] = value`, returning the array, for numpy AND jax alike.

  jax arrays are immutable and spell this `.at[index].set(value)`. Dispatching
  on the attribute rather than on a namespace keeps the call sites free of an
  `xp` they would otherwise need only for this -- numpy arrays have no `.at`.
  See PLAN.md W11 stage 4: the thermal fields are assembled by slice assignment,
  which is what kept them off the jax backend.
  """
  if hasattr(array, "at"): return array.at[index].set(value)
  array[index] = value
  return array
