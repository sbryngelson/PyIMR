"""Worker pools must not oversubscribe the machine.

XLA sizes its CPU thread pool from the affinity mask, so an unpinned spawn worker on a
many-core host builds a pool for every core it can see -- once per worker. Nothing fails;
it just thrashes, which is why this went unnoticed through every parallel run until a
128-core host showed a load average of 647.
"""

import os

import pytest

from pyimr.parallel import cores_are_committed, default_workers, map_work, pin_worker, worker_pool

SECTION = "18. Worker pools"

_LINUX_ONLY = pytest.mark.skipif(not hasattr(os, "sched_setaffinity"), reason="affinity control is Linux-only")


def _probe(_):
  """Run in a worker: how many cores it may use, and how many threads a solve costs."""
  import numpy as np

  import pyimr

  pyimr.simulate(np.linspace(0.0, 2e-5, 40), pyimr.SimulationConfig(225e-6, 225e-6 / 6, pyimr.NeoHookeanKelvinVoigt(2500.0, 0.1)))
  return len(os.sched_getaffinity(0)), len(os.listdir(f"/proc/{os.getpid()}/task"))


@_LINUX_ONLY
def test_a_worker_pool_confines_each_worker_and_keeps_its_threads_down(measured):
  workers = 4
  with worker_pool(workers) as pool:
    results = list(pool.map(_probe, range(workers)))

  cores = [c for c, _ in results]
  threads = [t for _, t in results]
  measured("pinned worker", f"{max(cores)} cores, {max(threads)} threads each")
  assert max(cores) <= max(1, len(os.sched_getaffinity(0)) // workers), "a worker sees more cores than its share"
  # unpinned this is ~400 on a 128-core host; the point is that it no longer tracks the host
  assert max(threads) < 64, f"worker used {max(threads)} threads, which means it was not confined"


@_LINUX_ONLY
def test_pinning_does_not_stack_every_worker_on_one_core():
  """Every worker on the same core is correct for thread count and useless for throughput,
  which is exactly what keying on `_identity` produced: `ProcessPoolExecutor` reports
  `(1,)` for all of them. Assignment is by pid and best-effort, so two workers MAY collide
  -- what must not happen is all of them landing together.
  """
  workers = 4
  with worker_pool(workers) as pool:
    masks = list(pool.map(_affinity, range(workers)))
  assert len({frozenset(m) for m in masks}) > 1, f"every worker pinned to the same core: {masks}"


def _affinity(_):
  import time

  time.sleep(0.3)  # hold the worker so the pool actually starts all of them
  return sorted(os.sched_getaffinity(0))


def test_pin_worker_is_a_no_op_where_the_platform_has_no_affinity_control(monkeypatch):
  monkeypatch.delattr(os, "sched_setaffinity", raising=False)
  pin_worker(4)  # must not raise


# module level so a spawned worker can import them; a closure or a __main__ function
# cannot be pickled and the pool dies with BrokenProcessPool
_CALLS = []


def _double(value):
  return value * 2


def _committed(_):
  return cores_are_committed()


def _pid(_):
  return os.getpid()


def test_cheap_work_stays_serial_and_expensive_work_is_pooled():
  """The decision, forced through the thresholds rather than through wall time, so the
  test cannot go flaky on a loaded machine.

  Sizing by item COUNT was the tempting rule and it is wrong: 64 Fisher draws at ~10 ms
  take 0.66 s serially against 5.32 s pooled, while 2401 solves at ~1 s each go from 40
  minutes to 20. Both have many items; only the per-item cost separates them.
  """
  items = list(range(8))
  expected = [value * 2 for value in items]

  # a pool could never pay for itself: must stay serial
  assert map_work(_double, items, overhead_s=1e9) == expected
  # the pool is free by construction: must be used, and must still be correct
  assert map_work(_double, items, overhead_s=0.0, margin=0.0) == expected


def test_work_dispatched_from_inside_a_pool_stays_serial():
  """Otherwise optimize_design -> expected_information_gain -> evaluate_batch opens pools
  inside pools. That is how this project reached a load average of 240.
  """
  assert not cores_are_committed()
  with worker_pool(2) as pool:
    assert cores_are_committed(), "the parent must know its cores are committed"
    assert all(pool.map(_committed, range(2))), "each worker must know it too"
  assert not cores_are_committed(), "the flag must be restored when the pool closes"


def test_map_work_actually_obeys_the_nesting_guard():
  """Asserting the flag is not enough -- deleting the guard from `map_work` leaves every
  flag assertion passing. This watches the pids instead: work that stayed serial ran in
  this process, and work that opened a pool did not.

  The thresholds force pooling, so only the guard can keep it serial.
  """
  mine = os.getpid()
  free = map_work(_pid, range(4), overhead_s=0.0, margin=0.0)
  assert set(free) != {mine}, "the thresholds should have forced a pool here"

  with worker_pool(2):
    nested = map_work(_pid, range(4), overhead_s=0.0, margin=0.0)
  assert set(nested) == {mine}, f"nested work opened a pool: pids {sorted(set(nested))}"


def test_an_explicit_worker_count_is_honoured():
  items = list(range(6))
  assert map_work(_double, items, workers=1) == [v * 2 for v in items]
  assert map_work(_double, items, workers=2) == [v * 2 for v in items]
  with pytest.raises(ValueError, match="positive integer or None"):
    map_work(_double, items, workers=0)


def test_the_timed_first_item_is_not_recomputed():
  """`map_work` runs one item to measure it. Keeping that result is what makes the
  measurement free; recomputing it would be a silent waste and, for a stateful callable,
  wrong."""
  seen = []

  def record(value):
    seen.append(value)
    return value

  assert map_work(record, [1, 2, 3], overhead_s=1e9) == [1, 2, 3]
  assert seen == [1, 2, 3], f"items were evaluated {len(seen)} times: {seen}"


def test_the_automatic_worker_count_is_capped_and_overridable(monkeypatch):
  """Each worker holds its own JAX, so an uncapped count is a memory decision disguised as
  a core count: a 128-core host would open 128 of them."""
  monkeypatch.delenv("PYIMR_WORKERS", raising=False)
  assert 1 <= default_workers() <= 32
  monkeypatch.setenv("PYIMR_WORKERS", "3")
  assert default_workers() == 3
