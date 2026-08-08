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


def pool(jobs):
  """A worker pool sized to the work and the machine, not to a number typed once.

  Every study here hardcoded its own count -- 3, 4, 6, 20 -- and they were mostly far too
  small: 54 jobs across 6 workers is nine waves on a 128-core machine, with 122 cores idle.
  One worker per job removes the waves, capped by the cores this process is actually allowed
  (`sched_getaffinity`, not `cpu_count`, so a scheduler's allocation is respected) less two
  so the parent and the OS keep a core.

  These jobs are also very unevenly sized -- a thermal fit costs many times a cold one -- and
  waves are what turns that imbalance into idle time, so removing them matters more here than
  the raw count does.
  """
  from pyimr.parallel import worker_pool

  available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 2)
  return worker_pool(max(1, min(int(jobs), available - 2)))


def load(dataset):
  """`(times, mean, spread, maximum_radius, stretch)` with the useless samples dropped."""
  record = json.loads((HERE / "results.json").read_text())[dataset]
  times, mean, spread = (np.array(record[k], dtype=float) for k in ("times_s", "mean", "spread"))
  keep = spread > 0
  return times[keep], mean[keep], spread[keep], DATASETS[dataset], record["stretch"]


def trial_count(dataset):
  """How many repeats the record carries. Not uniform: 18, 14 and 7.

  Named apart from `score`'s `trials=` argument deliberately: a module function and a
  keyword of the same name in one file is a shadow waiting for whoever calls one from
  inside the other.
  """
  return int(json.loads((HERE / "results.json").read_text())[dataset]["trials"])


def solver(times, maximum_radius, stretch, *, dynamics="keller-miksis", liquid_eos=None,
           rtol=1e-8, max_steps=400_000, **options):
  """A `solve(material)` callback of the shape `pyimr.selection` expects.

  `options` passes anything else `SimulationConfig` takes -- `bubtherm`, `medtherm`, `Nt` --
  so a study varying the thermal treatment uses this path rather than building its own.
  """
  import pyimr

  def solve(material, config_axes=None):
    # `req_scale` is a fitted axis when the candidate declares it, and 1.0 otherwise. Req is
    # inferred rather than measured, and a 1.68% error in it leaves the same residual as
    # changing the operator, so a study that pins it is asserting a precision it does not have.
    scale = float((config_axes or {}).get("req_scale", 1.0))
    config = pyimr.SimulationConfig(maximum_radius, maximum_radius / stretch * scale, material,
                                    dynamics=dynamics, liquid_eos=liquid_eos,
                                    rtol=rtol, atol=rtol * 1e-2,
                                    max_steps=max_steps, **options)
    trace = np.asarray(pyimr.simulate(times, config).radius_ratio, dtype=float)
    return trace, trace

  return solve


def score(candidate, solve, mean, spread, *, bounds=None, starts=6, evaluations=200,
          trials=None):
  """Fit, then the evidence summed over modes and the residual diagnostics at the best.

  Summed rather than taken at the best because the expansion is about one mode and the
  integral is over all of them; modes the expansion cannot use are dropped rather than
  allowed to abort the study.
  """
  from pyimr.noise import check_residuals, lack_of_fit
  from pyimr.selection import (PARAMETER_BOUNDS, candidate_log_evidence, evaluate_at,
                               fit_candidate, physical_from_unit)
  from scipy.special import logsumexp

  fit = fit_candidate(candidate, solve, mean, spread, bounds=bounds, starts=starts,
                      max_evaluations=evaluations)
  scored = []
  for point in fit.modes:
    try: scored.append(candidate_log_evidence(candidate, solve, mean, spread, point, bounds=bounds))
    except ValueError: continue

  values = physical_from_unit(candidate.axes, fit.unit, bounds)
  fitted = dict(zip(candidate.axes, (float(v) for v in values), strict=True))
  # through `evaluate_at`, or a candidate with configuration axes reports its residual at the
  # DEFAULT configuration while its evidence came from the fitted one
  model = evaluate_at(candidate, solve, fitted)[0]
  residual = (model - mean) / spread
  check = check_residuals(np.asarray(residual, dtype=float))
  # the replicate-based answer to the question `chi2_per_n` is usually mistaken for. `trials`
  # differs by record -- 18, 14 and 7 -- and it enters the statistic, so it is read from the
  # record rather than assumed
  misfit = float("nan")
  if trials is not None and int(trials) >= 2 and mean.size > candidate.dimension:
    misfit = lack_of_fit(mean, model, spread, int(trials), candidate.dimension).ratio
  box = bounds or PARAMETER_BOUNDS
  return dict(chi2_per_n=fit.chi_squared, log_evidence=float(logsumexp(scored)) if scored else float("nan"),
              lack_of_fit=misfit,
              lag_one=float(check.lag_one), n_eff=float(check.effective_samples),
              failure_fraction=fit.failure_fraction, modes=len(fit.modes), fitted=fitted,
              pinned=[k for k, v in fitted.items()
                      if min(abs(np.log(v / box[k][0])), abs(np.log(v / box[k][1]))) < 1e-6])
