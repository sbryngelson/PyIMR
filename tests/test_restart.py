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


@pytest.mark.parametrize("label", ["mechanical", "zener memory", "bubtherm"])
def test_the_tangent_linear_operator_propagates_a_perturbation(label, measured):
  """`J[k] @ v` must move the trajectory the way perturbing the start by `v` does."""
  problem = _problem(**dict(_LAYOUTS[label]))
  times = np.linspace(0.0, 12e-6, 25)
  start = np.asarray(problem.initial_state, dtype=float)

  states, jacobian = problem.state_tangents(times)
  assert jacobian.shape == (times.size, start.size, start.size)

  identity = np.eye(start.size)
  assert np.array_equal(jacobian[0], identity), "the flow at t0 is the identity, exactly"

  rng = np.random.default_rng(0)
  direction = rng.normal(size=start.size)
  direction /= np.linalg.norm(direction)
  # 1e-5, not smaller. Differencing two solves that agree to `rtol` puts a roundoff floor of
  # about rtol/step on this comparison: measured 7e-08 at 1e-5 rising to 4.8e-03 at 1e-9, so a
  # tighter step measures the instrument rather than the tangent.
  step = 1e-5
  difference = (problem.solve_states(times, start + step * direction) - problem.solve_states(times, start - step * direction)) / (2.0 * step)
  predicted = np.einsum("kij,j->ki", jacobian, direction)

  error = float(np.linalg.norm(predicted - difference) / np.linalg.norm(difference))
  measured(f"{label} tangent operator", f"rel={error:.2e}")
  assert error < 1e-6, f"{label}: J @ v differs from the perturbed solve by {error:.3e}"


def test_the_tangent_operator_is_consistent_with_the_primal():
  """The primal returned beside the Jacobian must be the trajectory itself."""
  problem = _problem(bubtherm=1, Nt=7)
  times = np.linspace(0.0, 12e-6, 25)
  states, _ = problem.state_tangents(times)
  np.testing.assert_allclose(states, problem.solve_states(times), rtol=1e-9, atol=1e-12)


def test_an_ensemble_member_matches_its_individual_solve():
  """The vmap must be a batching detail, not a different integration."""
  problem = _problem(bubtherm=1, Nt=7)
  times = np.linspace(0.0, 12e-6, 21)
  start = np.asarray(problem.initial_state, dtype=float)
  rng = np.random.default_rng(0)
  members = start + 1e-4 * rng.normal(size=(6, start.size))

  ensemble, _ = problem.solve_ensemble(members, times)
  assert ensemble.shape == (6, times.size, start.size)

  worst = max(float(np.abs(ensemble[i] - problem.solve_states(times, members[i])).max()) for i in range(len(members)))
  assert worst < 1e-12, f"batched member drifted from its own solve by {worst:.3e}"
  with pytest.raises(ValueError):
    ensemble[0, 0, 0] = 0.0


def test_the_ensemble_carries_a_spread_forward():
  """A filter needs the spread to actually evolve, not merely be transported."""
  problem = _problem()
  times = np.linspace(0.0, 20e-6, 41)
  start = np.asarray(problem.initial_state, dtype=float)
  rng = np.random.default_rng(1)
  members = start + np.column_stack([rng.normal(0.0, 1e-3, 24), np.zeros(24)])

  ensemble, _ = problem.solve_ensemble(members, times)
  spread = ensemble[:, :, 0].std(axis=0)
  assert spread[0] == pytest.approx(members[:, 0].std(), rel=1e-9)
  assert spread[-1] > spread[0], "a perturbation in radius should not stay the same size through a collapse"


@pytest.mark.parametrize(
  ("members", "message"),
  [
    (np.zeros(2), "members must be a 2-D array"),
    (np.zeros((3, 2, 2)), "members must be a 2-D array"),
    (np.zeros((3, 5)), "state must have shape"),
  ],
)
def test_the_ensemble_refuses_a_malformed_batch(members, message):
  with pytest.raises(ValueError, match=message):
    _problem().solve_ensemble(members, _TIMES)


def test_chunking_an_ensemble_changes_only_the_last_bits_of_the_remainder(measured):
  """`chunk` is a performance lever, and it is NOT exactly a no-op.

  70 members at 32 splits 32/32/6. The full-width pieces come back bit-identical to the
  unchunked solve; the width-6 remainder does not, because a different batch width is a
  different XLA kernel and so a different summation order. Measured at 1.0e-15 absolute,
  3.6e-14 relative -- three orders below the solver's own rtol, but not zero, and a test
  asserting equality would fail on the remainder alone.
  """
  problem = _problem(bubtherm=1, Nt=7)
  times = np.linspace(0.0, 12e-6, 21)
  start = np.asarray(problem.initial_state, dtype=float)
  members = start + 1e-4 * np.random.default_rng(1).normal(size=(70, start.size))

  whole, whole_ok = problem.solve_ensemble(members, times)
  pieces, pieces_ok = problem.solve_ensemble(members, times, chunk=32)

  np.testing.assert_array_equal(whole_ok, pieces_ok)
  np.testing.assert_array_equal(whole[:64], pieces[:64])  # the two full-width chunks
  remainder = float(np.abs(whole[64:] - pieces[64:]).max())
  measured("chunk remainder", f"max|diff|={remainder:.1e}")
  assert remainder < 1e-12, f"the remainder chunk moved by {remainder:.2e}, far more than rounding"


@pytest.mark.parametrize("chunk", [0, -1, 2.5])
def test_a_malformed_chunk_is_refused(chunk):
  problem = _problem()
  members = np.asarray(problem.initial_state, dtype=float)[None, :].repeat(4, axis=0)
  with pytest.raises(ValueError, match="chunk must be a positive integer"):
    problem.solve_ensemble(members, _TIMES, chunk=chunk)
