"""Why did this configuration fail to integrate, and can its answer be trusted?

`SimulationError: maximum number of solver steps was reached` is true and nearly useless.
It is the same message whether the trajectory needs a bigger budget, violates a material's
domain, or is so sensitive to its inputs that no tolerance will ever be met. Those want
opposite responses, and guessing between them wastes hours -- raising the budget on a
sensitive trajectory fails at forty times the cost, and loosening tolerance "succeeds"
while returning an answer that changes with the tolerance.

This runs the ladder that distinguishes them and says which it is. It is a diagnostic:
several solves, meant for a point that already misbehaved, not for a sweep.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ._config import SimulationConfig, SimulationError
from ._solver import simulate

__all__ = ["Diagnosis", "diagnose"]

_BUDGET_FACTOR = 40
_LOOSE = (1e-4, 1e-6)
_SENSITIVE = 1e3


@dataclass(frozen=True, slots=True)
class Diagnosis:
  """What a failing -- or suspiciously easy -- configuration turned out to be."""

  outcome: str
  summary: str
  steps: int | None = None
  amplification: float | None = None
  tolerance_gap: float | None = None

  def __str__(self) -> str: return f"{self.outcome}: {self.summary}"


def _solve(config, times, **overrides):
  try:
    return simulate(times, replace(config, **overrides)), None
  except SimulationError as error:
    return None, str(error)


def _amplification(config, times, *, relative=1e-9, **overrides):
  """How much a tiny change in `R0` grows by the end. Above ~1e3 the trajectory has shed
  digits, and asking the controller for more precision than it retains cannot succeed."""
  base, _ = _solve(config, times, **overrides)
  bumped, _ = _solve(replace(config, R0=config.R0 * (1.0 + relative)), times, **overrides)
  if base is None or bumped is None: return None
  moved = float(np.max(np.abs(np.asarray(base.radius_ratio) - np.asarray(bumped.radius_ratio))))
  return moved / relative


def _domain_failure(config, floor=0.02):
  """Re-evaluate the material across a collapse with numpy, where its guards run.

  Under tracing the value checks are skipped -- a tracer has no value to test -- so a
  material refusing its state surfaces as a step-budget failure instead of saying so.
  Scanning outside the tracer recovers the real message.

  The scan covers the whole collapse rather than the part already integrated: a solve that
  dies early never reaches the radius that refused it, which is exactly the case worth
  naming.
  """
  from ._prepare import _prepare_instantaneous_material, params
  from ._stress import _MaterialDomainError, _stress

  material = config.material
  prepared = _prepare_instantaneous_material(material) if hasattr(material, "elastic") else None
  p = params(config.R0, config.Req, material)
  for radius in np.geomspace(floor, 1.0, 60):
    try:
      _stress(material, p, float(radius), 0.0, np.zeros(4), prepared, False)
    except _MaterialDomainError as error:
      return f"{error} (reached at R/R0 = {radius:.4f})"
    except Exception:  # noqa: BLE001 - any other failure is not the domain question
      return None
  return None


def _runaway(error):
  """A runaway only surfaces once a solve actually completes, so it can appear at any rung
  of the ladder -- at a tight tolerance the step budget dies first and hides it."""
  if error and "ran away" in error:
    return Diagnosis("runaway", f"the model is not physical here, so no tolerance or solver will help -- {error}")
  return None


def diagnose(config: SimulationConfig, times) -> Diagnosis:
  """Classify why `config` fails over `times`, or whether a success can be trusted.

  Outcomes: `ok`, `runaway`, `ill-conditioned`, `domain`, `budget`, `unresolved`.
  """
  result, error = _solve(config, times)
  if result is not None:
    growth = _amplification(config, times)
    if growth is not None and growth > _SENSITIVE:
      return Diagnosis(
        "ill-conditioned",
        f"integrates, but a 1e-9 change in R0 grows {growth:.1e}x -- the trajectory has shed "
        f"about {np.log10(growth):.0f} digits, so treat it as one draw rather than the answer",
        steps=result.stats.nfev, amplification=growth,
      )
    return Diagnosis("ok", f"integrates in {result.stats.nfev} steps", steps=result.stats.nfev, amplification=growth)

  if runaway := _runaway(error): return runaway

  if "domain" in (error or "").lower() or "lock-up" in (error or "").lower():
    return Diagnosis("domain", f"the material refused the state it was driven to -- {error}")

  bigger, bigger_error = _solve(config, times, max_steps=int(config.max_steps) * _BUDGET_FACTOR)
  if runaway := _runaway(bigger_error): return runaway
  if bigger is not None:
    return Diagnosis(
      "budget", f"a {_BUDGET_FACTOR}x budget succeeds in {bigger.stats.nfev} steps; it was expensive, not ill-posed",
      steps=bigger.stats.nfev,
    )

  loose, loose_error = _solve(config, times, rtol=_LOOSE[0], atol=_LOOSE[1], max_steps=int(config.max_steps) * _BUDGET_FACTOR)
  if runaway := _runaway(loose_error): return runaway
  if loose is None:
    if domain := _domain_failure(config):
      return Diagnosis("domain", f"the material refuses a state a collapse reaches -- {domain}")
    return Diagnosis("unresolved", f"fails at a {_BUDGET_FACTOR}x budget and at rtol={_LOOSE[0]:.0e} -- {error}")

  looser, _ = _solve(config, times, rtol=_LOOSE[0] * 10, atol=_LOOSE[1] * 10, max_steps=int(config.max_steps) * _BUDGET_FACTOR)
  gap = None
  if looser is not None:
    gap = float(np.max(np.abs(np.asarray(loose.radius_ratio) - np.asarray(looser.radius_ratio))))
  growth = _amplification(config, times, rtol=_LOOSE[0], atol=_LOOSE[1], max_steps=int(config.max_steps) * _BUDGET_FACTOR)

  if (growth is not None and growth > _SENSITIVE) or (gap is not None and gap > 1e-3):
    detail = f"amplification {growth:.1e}x" if growth is not None else f"tolerances disagree by {gap:.1e}"
    return Diagnosis(
      "ill-conditioned",
      f"only a loose tolerance completes, and the answer is not converged ({detail}). More steps will not "
      f"help: the trajectory no longer carries the precision being asked of it",
      amplification=growth, tolerance_gap=gap,
    )
  return Diagnosis(
    "unresolved", "a loose tolerance completes and looks converged, but the requested one does not -- "
    "worth investigating by hand", amplification=growth, tolerance_gap=gap,
  )
