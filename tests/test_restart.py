"""Restarting from a raw state, the foundation for windowed assimilation (#122)."""

import numpy as np
import pytest

import pyimr
from _validation_support import NHKV, R0, REQ

SECTION = "9. State restart"

_TIMES = np.linspace(0.0, 30e-6, 121)
_SPLIT = 60

_LAYOUTS = {
  "mechanical": {},
  "zener memory": {"material": pyimr.Zener(2500.0, 0.1, 40e-6, 8e-6)},
  "bubtherm": {"bubtherm": 1, "Nt": 15},
  "all thermal": {"bubtherm": 1, "medtherm": 1, "masstrans": 1, "vapor": 1, "Nt": 11, "Mt": 11},
}


def _problem(**overrides):
  material = overrides.pop("material", NHKV)
  return pyimr.prepare(pyimr.SimulationConfig(R0, REQ, material, rtol=1e-11, atol=1e-13, **overrides))


@pytest.mark.parametrize("label", list(_LAYOUTS))
def test_a_restart_reproduces_the_tail_of_an_uninterrupted_solve(label, measured):
  """The property the whole windowed scheme rests on. If it fails, nothing above it means anything."""
  problem = _problem(**dict(_LAYOUTS[label]))
  whole = problem.solve(_TIMES)
  states = problem.solve_states(_TIMES)

  tail = problem.solve_from(states[_SPLIT], _TIMES[_SPLIT:] - _TIMES[_SPLIT])

  error = float(np.nanmax(np.abs(np.asarray(tail.radius_ratio) - np.asarray(whole.radius_ratio)[_SPLIT:])))
  measured(f"{label} restart", f"max|dR/R0|={error:.2e}  state width={states.shape[1]}")
  assert error < 1e-8, f"{label}: restart drifted from the uninterrupted solve by {error:.3e}"


def test_solve_states_agrees_with_the_physical_result():
  """`solve_states` must be the same trajectory `solve` reports, not a second integration."""
  problem = _problem(bubtherm=1, Nt=15)
  result = problem.solve(_TIMES)
  states = problem.solve_states(_TIMES)
  np.testing.assert_array_equal(states[:, 0], result.radius_ratio)
  assert states.shape == (_TIMES.size, problem.initial_state.size)
  with pytest.raises(ValueError):
    states[0, 0] = 0.0


def test_the_configured_start_is_the_default():
  """`solve_from(problem.initial_state, ...)` is exactly `solve(...)`."""
  problem = _problem()
  np.testing.assert_array_equal(
    problem.solve(_TIMES).radius_ratio,
    problem.solve_from(problem.initial_state, _TIMES).radius_ratio,
  )


@pytest.mark.parametrize(
  ("state", "message"),
  [
    (np.zeros(3), "state must have shape"),
    (np.zeros((2, 2)), "state must have shape"),
    (np.array([1.0, np.nan]), "state must contain only finite values"),
  ],
)
def test_restart_refuses_a_state_that_does_not_match_the_layout(state, message):
  with pytest.raises(ValueError, match=message):
    _problem().solve_from(state, _TIMES)
