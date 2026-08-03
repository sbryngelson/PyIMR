"""Choose thermal discretization and solver tolerance by measuring, not by folklore.

Requirements depend on record length, material stiffness and which observable is fitted.
A lookup table sees none of those, so this measures on the caller's own problem and
reports what it achieved.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

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
