"""Run the measured-data comparison on every dataset and record what the figures need.

Separate from plotting on purpose: this is the expensive half, and a figure should never
silently re-run a study or, worse, be drawn from numbers typed in by hand.

Run: .venv/bin/python docs/writeup/collect.py [workers]
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))

import measured_selection as study  # noqa: E402
from pyimr.noise import weighted_deviation  # noqa: E402
from pyimr.parallel import pin_worker  # noqa: E402
from pyimr.selection import compare, log_evidence, parameter_grid  # noqa: E402

OUT = Path(__file__).resolve().parent / "results.json"


def one(dataset, workers):
  trials, times, weights, solve, _, (rmax, req), bounds = study.setup(dataset, 0)
  payloads = [
    (dataset, 0, m.name, low, min(low + study._CHUNK, study.GRID_COUNT**m.dimension))
    for m in study.MODELS.values()
    for low in range(0, study.GRID_COUNT**m.dimension, study._CHUNK)
  ]
  with mp.get_context("spawn").Pool(workers, initializer=pin_worker, initargs=(workers,)) as pool:
    results = pool.map(study.solve_chunk, payloads)

  spread = trials.std(axis=0, ddof=1)
  deviations = weighted_deviation(spread, weights)
  record = {
    "times_s": times.tolist(),
    "mean": trials.mean(axis=0).tolist(),
    "spread": spread.tolist(),
    "trials": int(trials.shape[0]),
    "stretch": float(rmax / req),
    "models": {},
  }
  for candidate in study.MODELS.values():
    parts = sorted((low, r, w) for name, low, r, w in results if name == candidate.name)
    radii = np.concatenate([r for _, r, _ in parts])
    redundancies = np.concatenate([w for _, _, w in parts])
    dropped = int(np.isnan(radii).any(axis=1).sum())
    if dropped == len(radii): continue
    normalized = parameter_grid(candidate.axes, study.GRID_COUNT, bounds)[1]
    evidence, chi = log_evidence(radii, normalized, redundancies, trials, deviations, dimension=candidate.dimension)
    best = int(np.nanargmin(chi))
    record["models"][candidate.name] = {
      "evidence": float(evidence),
      "chi2_per_sample": float(np.nanmin(chi) / trials.size),
      "dropped": dropped,
      "points": int(len(radii)),
      "axes": list(candidate.axes),
      "best_theta": dict(zip(candidate.axes, parameter_grid(candidate.axes, study.GRID_COUNT, bounds)[0][best].tolist())),
      "best_trace": radii[best].tolist(),
    }
  posterior = compare({k: v["evidence"] for k, v in record["models"].items()})
  for name, value in posterior.items(): record["models"][name]["posterior"] = float(value)
  return record


def main():
  workers = int(sys.argv[1]) if len(sys.argv) > 1 else 16
  out = {}
  for dataset in study.DATASETS:
    print(f"  {dataset} ...", flush=True)
    out[dataset] = one(dataset, workers)
  OUT.write_text(json.dumps(out))
  print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
  main()
