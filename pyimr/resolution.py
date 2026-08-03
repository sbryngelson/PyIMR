"""Choose thermal discretization and solver tolerance by measuring, not by folklore.

Requirements depend on record length, material stiffness and which observable is fitted.
A lookup table sees none of those, so this measures on the caller's own problem and
reports what it achieved.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

__all__ = ["NT_LADDER", "RTOL_LADDER", "Resolution"]

NT_LADDER = (5, 7, 9, 13, 17, 25, 33)
RTOL_LADDER = (1e-10, 1e-8, 1e-6, 1e-4, 1e-3)

_REFERENCE_RTOL, _REFERENCE_ATOL = 1e-10, 1e-12
_CHECK_RTOL, _CHECK_ATOL = 1e-12, 1e-14


@dataclass(frozen=True, slots=True)
class Resolution:
  """A measured setting, with the error and cost it was chosen on."""

  thermal: str
  Nt: int
  rtol: float
  atol: float
  achieved: float
  seconds: float

  def apply(self, config):
    return _at(config, self.thermal, self.Nt, self.rtol, self.atol)


def _at(config, thermal, nt, rtol, atol):
  """`config` with one resolution substituted. `Mt` follows `Nt` only if the medium runs."""
  fields: dict[str, Any] = {"thermal": thermal, "Nt": int(nt), "rtol": float(rtol), "atol": float(atol)}
  if config.medtherm: fields["Mt"] = int(nt)
  return replace(config, **fields)


def _solve(config, times, fields):
  from . import simulate

  result = simulate(times, config)
  return {name: np.asarray(getattr(result, name), dtype=float) for name in fields}


def _timed(config, times, fields):
  start = time.perf_counter()
  values = _solve(config, times, fields)
  return time.perf_counter() - start, values


def _deviation(candidate, reference):
  """Worst per-field deviation, each scaled by that field's own peak magnitude."""
  worst = 0.0
  for name, truth in reference.items():
    scale = float(np.max(np.abs(truth))) or 1.0
    worst = max(worst, float(np.max(np.abs(candidate[name] - truth))) / scale)
  return worst


def _reference(config, times, fields, target, nt_ladder):
  """A converged solve, plus proof that it is converged.

  Solves at the ladder ceiling and again at doubled `Nt`. The finer one is the reference;
  their gap must be under `target / 10` or there is nothing to measure against.
  """
  ceiling = int(nt_ladder[-1])
  coarse = _solve(_at(config, "spectral", ceiling, _REFERENCE_RTOL, _REFERENCE_ATOL), times, fields)
  finer = _solve(_at(config, "spectral", 2 * ceiling, _CHECK_RTOL, _CHECK_ATOL), times, fields)

  gap = _deviation(coarse, finer)
  if gap > target / 10.0:
    raise ValueError(
      f"reference is not converged: Nt={ceiling} and Nt={2 * ceiling} differ by {gap:.2e}, "
      f"above target/10 = {target / 10.0:.2e}. Raise the Nt ladder or loosen target."
    )
  return finer, gap


def _smallest_adequate(config, times, fields, target, reference, thermal, nt_ladder):
  """Index of the smallest ladder entry meeting `target`, or None. Bisection, since
  discretization error falls with `Nt` even though cost does not rise smoothly."""
  low, high = 0, len(nt_ladder) - 1

  def misses(index):
    candidate = _at(config, thermal, nt_ladder[index], _REFERENCE_RTOL, _REFERENCE_ATOL)
    return _deviation(_solve(candidate, times, fields), reference) > target

  if misses(high):
    return None
  while low < high:
    middle = (low + high) // 2
    if misses(middle):
      low = middle + 1
    else:
      high = middle
  return low


def _choose_discretization(config, times, fields, target, reference, nt_ladder):
  """The cheapest `(thermal, Nt)` meeting `target`, ranked by measured time."""
  best = None
  for thermal in ("spectral", "fd"):
    index = _smallest_adequate(config, times, fields, target, reference, thermal, nt_ladder)
    if index is None:
      continue
    nodes = nt_ladder[index]
    seconds, _ = _timed(_at(config, thermal, nodes, _REFERENCE_RTOL, _REFERENCE_ATOL), times, fields)
    if best is None or seconds < best[0]:
      best = (seconds, thermal, nodes)

  if best is None:
    raise ValueError(
      f"no discretization in the ladder reaches target={target:.2e}; "
      f"raise the Nt ladder or loosen target"
    )
  return best[1], best[2]
