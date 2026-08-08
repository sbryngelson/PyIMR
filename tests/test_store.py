"""The disk cache: does it return the same answer, and does it know when not to?

A cache that returns a stale answer is worse than no cache, so the tests here are mostly
about misses -- every input that changes the result must change the key.
"""

import numpy as np
import pytest

import pyimr
from pyimr.store import ResultStore

SECTION = "20. Result store"

_TIMES = np.linspace(0.0, 1.4e-4, 121)
_R0, _REQ = 277e-6, 277e-6 / 7.09


def _config(**overrides):
  options = {"dynamics": "keller-miksis", "rtol": 1e-6, "atol": 1e-8} | overrides
  return pyimr.SimulationConfig(_R0, _REQ, pyimr.NeoHookeanKelvinVoigt(2500.0, 0.1), **options)


def test_a_hit_returns_the_same_result_the_solver_would_have(tmp_path, measured):
  store = ResultStore(tmp_path)
  first = store.simulate(_TIMES, _config())
  second = store.simulate(_TIMES, _config())

  measured("cache", f"{store.hits} hit, {store.misses} miss")
  assert (store.hits, store.misses) == (1, 1)
  np.testing.assert_array_equal(first.radius_ratio, second.radius_ratio)
  np.testing.assert_array_equal(first.internal_pressure_pa, second.internal_pressure_pa)
  assert second.stats.nfev == first.stats.nfev
  # the result contract says these are read-only; a cached one must not be looser
  with pytest.raises(ValueError):
    second.radius_ratio.flat[0] = 2.0


@pytest.mark.parametrize(
  "overrides",
  [{"rtol": 1e-7}, {"atol": 1e-9}, {"dynamics": "rayleigh-plesset"}, {"max_steps": 999}, {"bubtherm": 1, "Nt": 9}],
)
def test_anything_that_changes_the_answer_changes_the_key(tmp_path, overrides):
  store = ResultStore(tmp_path)
  store.simulate(_TIMES, _config())
  store.simulate(_TIMES, _config(**overrides))
  assert store.misses == 2, f"{overrides} was served from cache"


def test_a_different_material_or_time_grid_misses(tmp_path):
  store = ResultStore(tmp_path)
  store.simulate(_TIMES, _config())
  store.simulate(_TIMES, pyimr.SimulationConfig(_R0, _REQ, pyimr.NeoHookeanKelvinVoigt(2600.0, 0.1), dynamics="keller-miksis"))
  store.simulate(np.linspace(0.0, 1.4e-4, 122), _config())
  assert store.misses == 3


def test_a_failure_is_cached_so_its_cost_is_paid_once(tmp_path):
  """The point of this: a grid point that exhausts its budget spends the whole budget
  before failing, and a study re-run pays that again for every one of them.
  """
  store = ResultStore(tmp_path)
  failing = _config(max_steps=40)
  for _ in range(2):
    with pytest.raises(pyimr.SimulationError):
      store.simulate(_TIMES, failing)
  assert (store.hits, store.misses) == (1, 1), "the second failure should have been cached"


def test_the_version_invalidates_a_directory(tmp_path):
  """The key cannot see a change to PyIMR itself, so this is the escape hatch."""
  ResultStore(tmp_path).simulate(_TIMES, _config())
  later = ResultStore(tmp_path, version="2")
  later.simulate(_TIMES, _config())
  assert later.misses == 1


def test_clear_removes_the_entries(tmp_path):
  store = ResultStore(tmp_path)
  store.simulate(_TIMES, _config())
  assert store.clear() == 1
  store.simulate(_TIMES, _config())
  assert store.misses == 2
