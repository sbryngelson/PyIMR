"""Argument checks that were being written out at every call site.

A handful of predicates -- positive integer, positive scalar, finite array -- accounted for
some sixty hand-written checks across the package, each with its own wording, so the messages
drifted and adding one meant three lines of boilerplate. Nothing else belongs here: a check
used once is clearer where it is used.
"""

from __future__ import annotations

from numbers import Integral

import numpy as np

__all__ = ["deviation_for", "finite_array", "positive_integer", "positive_scalar", "unit_interval"]


def positive_integer(name, value, *, minimum=1):
  """`value` as an `int`, or `ValueError`. Rejects floats: `2.5` runs is not a budget."""
  if not isinstance(value, Integral) or int(value) < minimum:
    bound = {0: "a non-negative integer", 1: "a positive integer"}.get(minimum, f"an integer of at least {minimum}")
    raise ValueError(f"{name} must be {bound}; got {value!r}")
  return int(value)


def positive_scalar(name, value, *, allow_zero=False):
  number = float(value)
  if not np.isfinite(number) or number < 0.0 or (number == 0.0 and not allow_zero):
    raise ValueError(f"{name} must be finite and {'non-negative' if allow_zero else 'positive'}; got {value!r}")
  return number


def finite_array(name, values, *, non_negative=False, shape=None):
  """`values` as a float array, checked finite and optionally non-negative.

  A `shape` entry is an int to require that extent, or a name for a free one -- so
  `("samples", "p")` means "two-dimensional" and reports itself that way when it fails.
  """
  array = np.asarray(values, dtype=float)
  if shape is not None:
    if len(shape) != array.ndim or any(w != g for w, g in zip(shape, array.shape, strict=True) if isinstance(w, int)):
      wanted = ", ".join(str(w) for w in shape) + ("," if len(shape) == 1 else "")
      raise ValueError(f"{name} must have shape ({wanted}); got {array.shape}")
  if not np.all(np.isfinite(array)): raise ValueError(f"{name} must be finite")
  if non_negative and np.any(array < 0.0): raise ValueError(f"{name} must be non-negative")
  return array


def unit_interval(name, value):
  number = float(value)
  if not 0.0 <= number <= 1.0: raise ValueError(f"{name} must lie in [0, 1]; got {value!r}")
  return number


def deviation_for(deviation, samples):
  """A noise scale broadcastable over `samples`: scalar, or one value per sample.

  Real records carry the second -- the trial spread varies across the trace, and collapsing
  it to one number silently reweights which parts of the curve the fit is asked to match.
  """
  scale = finite_array("deviation", deviation)
  if np.any(scale <= 0.0): raise ValueError("deviation must be finite and positive")
  if scale.ndim > 1 or (scale.ndim == 1 and scale.size not in (1, samples)):
    raise ValueError(f"deviation must be scalar or one per sample; got {scale.size} for {samples} observations")
  return scale
