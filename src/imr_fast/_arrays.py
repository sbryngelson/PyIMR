"""The one array helper that has to work in both namespaces."""

from __future__ import annotations

def at_set(array, index, value):
  """`array[index] = value`, returning the array, for numpy AND jax alike.

  Dispatches on `.at`, which jax arrays have and numpy arrays do not, so call
  sites need no `xp`.
  """
  if hasattr(array, "at"): return array.at[index].set(value)
  array[index] = value
  return array
