"""Solving one traced program at many parameter sets -- the shape of a grid campaign."""

import numpy as np
import pytest

import pyimr
from _validation_support import NHKV, R0, REQ

SECTION = "13. Parameter sweeps"

_TIMES = np.linspace(0.0, 20e-6, 60)
_PATHS = ["material.shear_modulus_pa", "material.viscosity_pa_s"]


def _problem(**overrides):
  return pyimr.prepare(pyimr.SimulationConfig(R0, REQ, NHKV, rtol=1e-10, atol=1e-12, **overrides))


def _grid(n=5):
  return np.column_stack([np.linspace(5e3, 20e3, n), np.linspace(0.02, 0.12, n)])


def test_every_row_matches_its_own_solve(measured):
  """A sweep must be a batching detail. If a row differs from solving that point alone,

  the vmapped program is not the same computation and nothing above it can be trusted.
  """
  problem = _problem()
  values = _grid()
  sweep = problem.solve_sweep(_TIMES, _PATHS, values)

  assert sweep.radius_ratio.shape == (len(values), _TIMES.size)
  assert sweep.state.shape == (len(values), _TIMES.size, problem.initial_state.size)
  assert sweep.parameters == tuple(_PATHS)

  worst = 0.0
  for index, (modulus, viscosity) in enumerate(values):
    alone = pyimr.simulate(_TIMES, pyimr.SimulationConfig(R0, REQ, pyimr.NeoHookeanKelvinVoigt(modulus, viscosity), rtol=1e-10, atol=1e-12))
    worst = max(worst, float(np.abs(sweep.radius_ratio[index] - alone.radius_ratio).max()))
  measured("sweep vs individual solves", f"max|dR/R0|={worst:.1e}")
  assert worst < 1e-10, f"a swept row drifted from its own solve by {worst:.3e}"


def test_the_sweep_reports_per_point_solver_steps():
  """Steps come back per point, not as a single number for the batch."""
  sweep = _problem().solve_sweep(_TIMES, _PATHS, _grid())
  assert sweep.steps.shape == (5,)
  assert np.all(sweep.steps > 0)


def test_distinct_parameters_give_distinct_trajectories():
  """One compiled program is only correct if it still answers each point separately."""
  sweep = _problem().solve_sweep(_TIMES, _PATHS, _grid())
  for earlier, later in zip(sweep.radius_ratio[:-1], sweep.radius_ratio[1:], strict=True):
    assert np.abs(earlier - later).max() > 1e-6


def test_the_result_is_read_only():
  sweep = _problem().solve_sweep(_TIMES, _PATHS, _grid())
  for array in (sweep.radius_ratio, sweep.state, sweep.values, sweep.time_s):
    with pytest.raises(ValueError):
      array.flat[0] = 0.0


def test_a_single_point_is_still_a_batch():
  """`(1, k)` must work, so a caller can use one entry point for both shapes."""
  problem = _problem()
  sweep = problem.solve_sweep(_TIMES, _PATHS, np.array([[10e3, 0.05]]))
  alone = pyimr.simulate(_TIMES, pyimr.SimulationConfig(R0, REQ, pyimr.NeoHookeanKelvinVoigt(10e3, 0.05), rtol=1e-10, atol=1e-12))
  assert sweep.radius_ratio.shape == (1, _TIMES.size)
  np.testing.assert_allclose(sweep.radius_ratio[0], alone.radius_ratio, atol=1e-10)


@pytest.mark.parametrize(
  ("paths", "values", "message"),
  [
    ([], np.zeros((2, 1)), "at least one parameter path"),
    (_PATHS, np.zeros((2, 3)), "values must be"),
    (_PATHS, np.array([[1e3, np.nan]]), "values must be finite"),
  ],
)
def test_a_malformed_sweep_is_refused(paths, values, message):
  with pytest.raises(ValueError, match=message):
    _problem().solve_sweep(_TIMES, paths, values)
