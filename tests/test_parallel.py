"""Worker pools must not oversubscribe the machine.

XLA sizes its CPU thread pool from the affinity mask, so an unpinned spawn worker on a
many-core host builds a pool for every core it can see -- once per worker. Nothing fails;
it just thrashes, which is why this went unnoticed through every parallel run until a
128-core host showed a load average of 647.
"""

import os

import pytest

from pyimr.parallel import pin_worker, worker_pool

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
