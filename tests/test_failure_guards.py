"""Guards that need a state built to fail one specific way (#124).

Each test pairs the failing configuration with a **neighbour** that differs only in the
parameter the guard is about, and asserts the neighbour succeeds. Matching the message
alone is not enough: a state that fails for the wrong reason would still raise, and would
still match if two guards share phrasing. The neighbour is what pins the cause.
"""

import numpy as np
import pytest

import pyimr
from pyimr import SimulationError
from _validation_support import R0, REQ

SECTION = "11. Constructed failure states"

_MATERIAL = pyimr.Zener(2500.0, 0.1, 40e-6, 8e-6)
_TIMES = np.linspace(0.0, 2e-5, 20)


def _prepare(**collapse):
  return pyimr.prepare(
    pyimr.SimulationConfig(R0, REQ, _MATERIAL, dynamics="keller-miksis", collapse=pyimr.CollapseInitialization(**collapse))
  )


def test_a_precursor_that_never_reaches_a_maximum_radius_is_reported_as_such(measured):
  """Too short a window, and the event never fires. The neighbour is the same problem

  with the default window, which must succeed -- otherwise this would be recording that
  the case is broken rather than that the budget is too small.
  """
  _prepare()  # neighbour: identical but for the window

  with pytest.raises(SimulationError, match="did not reach a maximum radius"):
    _prepare(maximum_time_nondimensional=1e-3)

  measured("precursor window", "1e-3 raises, default succeeds")


def test_shooting_that_cannot_bracket_names_the_expansion_budget(measured):
  """One expansion is not enough to bracket; the default 24 is. Same material, same

  radii, same solver -- only the budget differs, so only the budget can be the cause.
  """
  _prepare(maximum_bracket_expansions=24)  # neighbour

  with pytest.raises(SimulationError, match="could not bracket an initial velocity after 1 expansions"):
    _prepare(maximum_bracket_expansions=1)

  measured("bracket budget", "1 raises, 24 succeeds")


def test_a_solver_failure_is_wrapped_with_the_stats_that_explain_it(monkeypatch):
  """Plumbing, not physics: this guard turns any solver exception into a `SimulationError`

  carrying `SolverStats`. Driving it with a real physical failure would test the physics
  instead of the translation, so the failure is injected and only the wrapping asserted.
  """
  from pyimr import _jax

  def exploding(key, build):
    def raise_instead(*args, **kwargs):
      raise RuntimeError("synthetic solver failure")

    return raise_instead

  monkeypatch.setattr(_jax, "_cached", exploding)

  with pytest.raises(SimulationError) as caught:
    pyimr.simulate(_TIMES, pyimr.SimulationConfig(R0, REQ, _MATERIAL))

  assert "synthetic solver failure" in str(caught.value), "the underlying cause must survive the translation"
  stats = caught.value.stats
  assert stats is not None, "a wrapped failure without stats tells the caller nothing"
  assert not stats.success
  assert "synthetic solver failure" in stats.message
  assert stats.backend.startswith("jax-")


def test_an_absurd_gas_pressure_reports_a_failed_bracket(measured):
  """Reachable, contrary to the guess that it was dead code. The gas term diverges as

  `ratio ** -3k` and the surface term only as `1/ratio`, so the gas term wins as the
  radius shrinks -- unless the gas pressure is small enough to push the crossover below
  the 1e-14 floor, which is `gas * R0 / 2 sigma < 1e-45`.
  """
  from pyimr import data

  assert data.equilibrium_radius(225e-6, 1e-40) > 0.0  # neighbour: five orders larger, still solvable

  with pytest.raises(ValueError, match="could not bracket an equilibrium radius"):
    data.equilibrium_radius(225e-6, 1e-45)

  measured("equilibrium bracket", "1e-45 Pa raises, 1e-40 Pa succeeds")


def test_an_equilibrium_at_the_observed_maximum_is_refused(measured):
  """The precursor shoots for an initial velocity that grows the bubble to the observed

  maximum. If equilibrium is already there, no velocity does, and the guard says so.
  """
  def prepare(fraction):
    return pyimr.prepare(
      pyimr.SimulationConfig(R0, R0 * fraction, _MATERIAL, dynamics="keller-miksis", collapse=pyimr.CollapseInitialization())
    )

  prepare(0.9999)  # neighbour: equilibrium a hair below the maximum still works

  with pytest.raises(SimulationError, match="equilibrium radius is not below the observed maximum"):
    prepare(1.0)

  measured("equilibrium vs maximum", "Req/R0=1 raises, 0.9999 succeeds")


def test_a_viscosity_that_overflows_is_caught_before_it_reaches_the_solver(measured):
  """Tested at the function rather than through `simulate`, deliberately.

  No public configuration was found that reaches this guard: every material extreme
  enough to overflow the viscosity also exhausts the step budget first, so a
  `simulate`-level test would assert the max-steps failure while appearing to cover this
  line. The guard is a defensive check on an internal function, and that is how it is
  tested. See the issue for the search that established it.
  """
  from pyimr._stress import _MaterialDomainError, _viscosity_and_tangent

  rate = np.array([0.0, 1e3, 1e8])

  def model(consistency):
    return pyimr.HerschelBulkley(yield_stress_pa=10.0, consistency_pa_s_n=consistency, exponent=5.0, regularization_rate_per_s=1e3)

  viscosity, tangent = _viscosity_and_tangent(model(1e2), rate)  # neighbour: finite, up to 1e34
  assert np.all(np.isfinite(viscosity)) and np.all(np.isfinite(tangent))

  # the overflow warning is left unsuppressed on purpose: the overflow is the event under
  # test, and this repo keeps `ERRSTATE_ALLOWED` empty rather than hiding such warnings
  with pytest.raises(_MaterialDomainError, match="generalized viscosity became invalid"):
    _viscosity_and_tangent(model(1e300), rate)

  measured("viscosity domain", "1e300 raises, 1e2 finite")
