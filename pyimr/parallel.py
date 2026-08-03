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
from multiprocessing import get_context

__all__ = ["limit_worker_threads", "pin_worker", "worker_pool"]


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


def worker_pool(workers: int) -> ProcessPoolExecutor:
  """A spawn pool whose workers each hold one core and one thread per library.

  Spawn rather than fork because JAX deadlocks under fork.
  """
  limit_worker_threads()
  return ProcessPoolExecutor(
    max_workers=workers, mp_context=get_context("spawn"), initializer=pin_worker, initargs=(workers,)
  )
