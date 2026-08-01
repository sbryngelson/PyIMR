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
    pyimr.SimulationConfig(R0, REQ, _MATERIAL, radial=2, collapse=pyimr.CollapseInitialization(**collapse))
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
