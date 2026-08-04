"""Process pools that do not oversubscribe the machine.

XLA's CPU backend sizes its thread pool from the process's CPU AFFINITY MASK, not from a
thread-count environment variable. So a spawn worker on a 128-core host builds a pool for
128 cores, and sixteen of them ask for roughly 6500 threads between them. That does not
fail -- it thrashes, and it does it to every other job on the machine too. Measured
threads in one worker after a single solve: 397 unrestricted, 270 with
`XLA_FLAGS=--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1`, and 4
pinned. Only affinity works.

numpy's OpenBLAS does the same thing at ITS import, which is the larger half: measured in
one worker after a single solve, 395 threads unrestricted, 130 pinned but with OpenBLAS
already sized, and 4 with both handled. Env vars fix OpenBLAS but only if they are set
before numpy is imported -- too late inside a pool initializer, since unpickling the
initializer imports its module first. Children inherit the parent's environment at spawn,
so `worker_pool` sets them in the parent instead.

Thread count tracks the WIDTH of the mask, not the worker count: a 32-core slice still
costs ~205 threads. So each worker gets exactly one core. That trades per-solve
parallelism for worker count, which is right for solve batches -- they are embarrassingly
parallel, and XLA's intra-op threading was measured not to help on solves this size. A
caller who wants more of the machine raises `workers`.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from multiprocessing import get_context
from numbers import Integral
from time import perf_counter

__all__ = ["cores_are_committed", "default_workers", "limit_worker_threads", "map_work", "pin_worker", "worker_pool"]

POOL_OVERHEAD_S = 5.0
"""Fixed cost of a pool before it does useful work. Measured on 128 cores: 1.35 s to start
32 workers, and about 5 s once each has imported JAX and retraced the program."""

POOL_MARGIN = 4.0
"""How much the projected serial time must exceed `POOL_OVERHEAD_S` before pooling. At 4,
a job must project past ~20 s serial, which keeps 64 Fisher draws (0.66 s) serial and sends
a 2400-solve sweep (~40 min) to the pool."""

WORKER_CAP = 32
"""Ceiling on the automatic worker count. Each worker is a separate process holding its own
JAX, so the cost is memory rather than cores: a 128-core host would otherwise open 128 of
them. 32 was measured to start in 1.35 s and is what the benchmarks behind `map_work` used.
`PYIMR_WORKERS` overrides it in either direction."""

_COMMITTED = False


def cores_are_committed() -> bool:
  """True when a pool already owns the machine, so nested work must stay serial.

  Set two ways, because workers are separate processes and a parent-only flag never reaches
  them: `worker_pool` sets it in the parent for as long as the pool is open, and
  `pin_worker` sets it inside each worker.
  """
  return _COMMITTED


def default_workers() -> int:
  """Workers to use when the caller does not say.

  From the affinity mask rather than `cpu_count`: on a cluster the mask is what the
  scheduler actually granted, while `cpu_count` reports the whole node and would have this
  process claim cores belonging to someone else. `PYIMR_WORKERS` overrides.
  """
  override = os.environ.get("PYIMR_WORKERS")
  if override: return max(1, int(override))
  if hasattr(os, "sched_getaffinity"): granted = len(os.sched_getaffinity(0))
  else: granted = os.cpu_count() or 1
  return max(1, min(granted, WORKER_CAP))


def pin_worker(_workers: int = 1) -> None:
  """Confine this worker to one core.

  A pool initializer. Keyed on the pid because that is the only worker-distinct number
  available in both pool flavours: `multiprocessing.Pool` numbers its workers through
  `_identity`, but `ProcessPoolExecutor` reports `(1,)` for every one of them, which
  would stack the whole pool on a single core. Assignment is therefore best-effort --
  two workers can land on one core -- which costs a little throughput and never
  correctness.

  Does nothing where the platform has no affinity control, i.e. anywhere but Linux.
  """
  if not hasattr(os, "sched_setaffinity"): return
  available = sorted(os.sched_getaffinity(0))
  if available: os.sched_setaffinity(0, {available[os.getpid() % len(available)]})


_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def limit_worker_threads() -> None:
  """Ask the numeric libraries for one thread each, for this process and its children.

  `setdefault`, so a caller who has already chosen keeps their choice. This has to happen
  before numpy is imported in the process that matters -- for spawned workers that is the
  parent, whose environment they inherit.
  """
  for name in _THREAD_VARS: os.environ.setdefault(name, "1")


def _start_worker(workers: int) -> None:
  """Pool initializer: pin the core, and mark the cores committed inside this worker.

  Separate from `pin_worker` so that pinning stays a pure affinity call. `pin_worker` is
  called directly -- by callers and by this project's own tests -- and setting a global
  there would commit the parent's cores for the rest of the session.
  """
  global _COMMITTED
  _COMMITTED = True
  pin_worker(workers)


class _CommittingPool(ProcessPoolExecutor):
  """A pool that marks the cores as committed while it is open."""

  def __enter__(self):
    global _COMMITTED
    self._restore = _COMMITTED
    _COMMITTED = True
    return super().__enter__()

  def __exit__(self, *details):
    global _COMMITTED
    try:
      return super().__exit__(*details)
    finally:
      _COMMITTED = self._restore


def worker_pool(workers: int) -> ProcessPoolExecutor:
  """A spawn pool whose workers each hold one core and one thread per library.

  Spawn rather than fork because JAX deadlocks under fork.
  """
  limit_worker_threads()
  return _CommittingPool(
    max_workers=workers, mp_context=get_context("spawn"), initializer=_start_worker, initargs=(workers,)
  )


def _pooled(function, items, workers):
  try:
    with worker_pool(workers) as pool:
      return list(pool.map(function, items))
  except BrokenProcessPool as error:
    raise RuntimeError(
      f"a worker died with {workers} processes. Two usual causes: `function` is not "
      f"importable by a spawned worker (define it at module level, not in a closure or a "
      f"__main__ script), or memory pressure, since each worker loads its own JAX. Retry "
      f"with a smaller `workers`, or set PYIMR_WORKERS"
    ) from error


def map_work(function, items, *, workers=None, overhead_s=POOL_OVERHEAD_S, margin=POOL_MARGIN):
  """Run `function` over `items`, deciding serial or pooled by timing the first item.

  Item COUNT is the wrong criterion; per-item COST is the right one. Measured: 64 Fisher
  draws at ~10 ms each take 0.66 s serially against 5.32 s pooled, while 2401 solves at
  about a second each go from 40 minutes to 20. Both are "many items", and sizing by count
  would have made the first case eight times slower.

  So the first item runs and is timed -- its result is kept, not recomputed -- and a pool
  opens only when the projected serial time clears the pool's fixed cost by `margin`.

  `workers` given explicitly is honoured exactly, and `workers=1` forces serial.
  """
  items = list(items)
  if not items: return []

  if workers is not None:
    if not isinstance(workers, Integral) or workers < 1: raise ValueError("workers must be a positive integer or None")
    if int(workers) == 1: return [function(item) for item in items]
    return _pooled(function, items, int(workers))

  if cores_are_committed() or len(items) < 2: return [function(item) for item in items]

  started = perf_counter()
  first = function(items[0])
  projected = (perf_counter() - started) * len(items)
  chosen = min(default_workers(), len(items) - 1)
  if projected < overhead_s * margin or chosen < 2:
    return [first, *(function(item) for item in items[1:])]
  return [first, *_pooled(function, items[1:], chosen)]
