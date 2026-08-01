"""The one array helper that has to work in both namespaces."""

from __future__ import annotations

def at_set(array, index, value):
  """`array[index] = value`, returning the array, for numpy AND jax alike."""
  if hasattr(array, "at"): return array.at[index].set(value)
  array[index] = value
  return array
