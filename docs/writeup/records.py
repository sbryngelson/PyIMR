"""The pieces every record-fitting study here needs, in one place.

Five studies grew separately and repeated the same six things: pinning the BLAS threads,
loading a record and dropping its zero-spread samples, building a solve callback, fitting,
summing evidence over the modes the fit found, and reading the residual diagnostics back.
The duplication was not free -- `R_max` differs per record and every copy hardcoded the
15 C value, which was correct only because each study touched one record.
"""

import json
import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
  os.environ.setdefault(_name, "1")

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# R_max per record, from the paper's dataset table (see `examples/measured_selection.py`).
# They differ; `results.json` does not carry them, which is why they live here now.
DATASETS = {"gelatin_15C": 277e-6, "gelatin_23C": 298e-6, "gelatin_33C": 312e-6}


def load(dataset):
  """`(times, mean, spread, maximum_radius, stretch)` with the useless samples dropped."""
  record = json.loads((HERE / "results.json").read_text())[dataset]
  times, mean, spread = (np.array(record[k], dtype=float) for k in ("times_s", "mean", "spread"))
  keep = spread > 0
  return times[keep], mean[keep], spread[keep], DATASETS[dataset], record["stretch"]


def solver(times, maximum_radius, stretch, *, radial=2, rtol=1e-8, max_steps=400_000, **options):
  """A `solve(material)` callback of the shape `pyimr.selection` expects.

  `options` passes anything else `SimulationConfig` takes -- `bubtherm`, `medtherm`, `Nt` --
  so a study varying the thermal treatment uses this path rather than building its own.
  """
  import pyimr

  def solve(material):
    config = pyimr.SimulationConfig(maximum_radius, maximum_radius / stretch, material,
                                    radial=radial, rtol=rtol, atol=rtol * 1e-2,
                                    max_steps=max_steps, **options)
    trace = np.asarray(pyimr.simulate(times, config).radius_ratio, dtype=float)
    return trace, trace

  return solve


def score(candidate, solve, mean, spread, *, bounds=None, starts=6, evaluations=200):
  """Fit, then the evidence summed over modes and the residual diagnostics at the best.

  Summed rather than taken at the best because the expansion is about one mode and the
  integral is over all of them; modes the expansion cannot use are dropped rather than
  allowed to abort the study.
  """
  from pyimr.noise import check_residuals
  from pyimr.selection import (PARAMETER_BOUNDS, candidate_log_evidence, fit_candidate,
                               physical_from_unit)
  from scipy.special import logsumexp

  fit = fit_candidate(candidate, solve, mean, spread, bounds=bounds, starts=starts,
                      max_evaluations=evaluations)
  scored = []
  for point in fit.modes:
    try: scored.append(candidate_log_evidence(candidate, solve, mean, spread, point, bounds=bounds))
    except ValueError: continue

  values = physical_from_unit(candidate.axes, fit.unit, bounds)
  fitted = dict(zip(candidate.axes, (float(v) for v in values), strict=True))
  residual = (solve(candidate.build(fitted))[0] - mean) / spread
  check = check_residuals(np.asarray(residual, dtype=float))
  box = bounds or PARAMETER_BOUNDS
  return dict(chi2_per_n=fit.chi_squared, log_evidence=float(logsumexp(scored)) if scored else float("nan"),
              lag_one=float(check.lag_one), n_eff=float(check.effective_samples),
              failure_fraction=fit.failure_fraction, modes=len(fit.modes), fitted=fitted,
              pinned=[k for k, v in fitted.items()
                      if min(abs(np.log(v / box[k][0])), abs(np.log(v / box[k][1]))) < 1e-6])
