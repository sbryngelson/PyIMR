"""A disk cache for solves, so a re-run of an unchanged study costs nothing.

A parameter study is usually run many times while the analysis around it changes -- new
plots, a different window, one more candidate -- and re-solves every point each time.
Caching on the content of `(times, config)` makes the second run free.

Failures are cached too, and that is most of the value: a point that exhausts its step
budget spends the whole budget before saying so, every time it is asked.

The key covers everything the answer depends on, so a changed tolerance, material
parameter or time grid is a miss rather than a stale hit. What it cannot see is a change
to PyIMR itself, so `version` exists to invalidate a directory by hand after one.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from ._config import SimulationError, SimulationResult, SolverStats
from ._jax import _content_key
from ._solver import simulate

__all__ = ["ResultStore"]

_ARRAYS = (
  "time_s", "radius_ratio", "wall_velocity_m_s", "internal_pressure_pa", "stress_integral_pa",
  "bubble_temperature_k", "medium_temperature_k", "vapor_mass_fraction", "stress_state",
  "stress_reference_radius_ratio",
)
_STATS = ("backend", "success", "message", "nfev", "njev", "nlu", "elapsed_s")  # written generically, read explicitly so the types are checkable


class ResultStore:
  """Solve through this instead of `pyimr.simulate` to reuse earlier runs.

      store = ResultStore("~/.cache/pyimr")
      result = store.simulate(times, config)   # same signature, same result
  """

  __slots__ = ("_directory", "_version", "hits", "misses")

  def __init__(self, directory, *, version: str = "1"):
    self._directory = Path(directory).expanduser()
    self._directory.mkdir(parents=True, exist_ok=True)
    self._version = str(version)
    self.hits = 0
    self.misses = 0

  def _path(self, times, config) -> Path:
    key = repr((self._version, _content_key(np.asarray(times, dtype=float)), _content_key(config)))
    return self._directory / f"{hashlib.sha256(key.encode()).hexdigest()[:32]}.npz"

  def simulate(self, times, config) -> SimulationResult:
    """`pyimr.simulate`, answered from disk when this exact call was made before.

    A cached failure re-raises `SimulationError` rather than replaying the cost of
    discovering it again.
    """
    path = self._path(times, config)
    failed = path.with_suffix(".failed")
    if failed.exists():
      self.hits += 1
      raise SimulationError(failed.read_text())
    if path.exists():
      self.hits += 1
      return self._load(path, config)
    self.misses += 1
    try:
      result = simulate(times, config)
    except SimulationError as error:
      failed.write_text(str(error))
      raise
    self._save(path, result)
    return result

  def _save(self, path, result) -> None:
    payload = {name: np.asarray(getattr(result, name)) for name in _ARRAYS if getattr(result, name) is not None}
    payload |= {f"_stat_{name}": np.array(getattr(result.stats, name)) for name in _STATS}
    # written beside then moved, so a killed run cannot leave a half-file that later reads
    # as a hit. The name must already end in `.npz`, since `savez` appends it otherwise and
    # the rename would then look for a file that was never written.
    temporary = path.with_name(path.name + ".partial.npz")
    # `payload`'s keys are ours, never numpy's own keyword arguments, which a splat cannot state
    np.savez_compressed(temporary, **payload)  # pyright: ignore[reportArgumentType]
    temporary.replace(path)

  def _load(self, path, config) -> SimulationResult:
    with np.load(path, allow_pickle=False) as blob:
      stats = SolverStats(
        backend=str(blob["_stat_backend"]),
        success=bool(blob["_stat_success"]),
        message=str(blob["_stat_message"]),
        nfev=int(blob["_stat_nfev"]),
        njev=int(blob["_stat_njev"]),
        nlu=int(blob["_stat_nlu"]),
        elapsed_s=float(blob["_stat_elapsed_s"]),
      )
      fields = {name: (np.asarray(blob[name]) if name in blob else None) for name in _ARRAYS}
    for array in fields.values():
      if array is not None: array.setflags(write=False)
    # named one at a time rather than splatted: these five are not optional on
    # `SimulationResult`, and a dict of `array | None` cannot say so
    time_s, radius, velocity = fields["time_s"], fields["radius_ratio"], fields["wall_velocity_m_s"]
    pressure, stress = fields["internal_pressure_pa"], fields["stress_integral_pa"]
    if time_s is None or radius is None or velocity is None or pressure is None or stress is None:
      raise SimulationError(f"cached result at {path} is missing a required field")
    return SimulationResult(
      time_s=time_s, radius_ratio=radius, wall_velocity_m_s=velocity, internal_pressure_pa=pressure,
      stress_integral_pa=stress, **{name: fields[name] for name in _ARRAYS[5:]}, stats=stats, config=config,
    )

  def clear(self) -> int:
    """Delete every cached entry, returning how many were removed."""
    paths = [*self._directory.glob("*.npz"), *self._directory.glob("*.failed")]
    for path in paths: path.unlink()
    return len(paths)

  def __repr__(self) -> str:
    return f"ResultStore({str(self._directory)!r}, hits={self.hits}, misses={self.misses})"
