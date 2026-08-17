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

TEMPSWEEPS = Path.home() / "fastscratch/imr-data-tempsweeps/data"
# The PAAm release, read from its CSVs rather than from `results.json`: 249 events against
# gelatin's 39, on the same rig.
#
# `PA05_temp_exp_data.csv` is a POOLED temperature sweep and must be split before it is a
# record. Its 117 columns are three blocks of 30, 57 and 30, which is what the per-temperature
# event counts in `PA05_t{10,21,33}_pIMR_i.csv` state and what the collapse time confirms
# independently at F = 14.1 against a permutation null of the same partition (p < 0.0005).
# `PA05003_temp_exp_data.csv` shows no block structure at ANY two-cut partition -- its best F
# of 6.6 sits below a best-of-all-partitions null of 8.6 -- so it is carried whole. If it is
# pooled after all its spread is inflated, which makes its lack-of-fit ratio conservative.
# The three files share no column: every one of the 249 events appears exactly once.
#
# R_max is the mean of `PA0503_radius.csv`, which is PAAm as a whole rather than per record;
# `radius_provenance.py` shows the row-to-trace pairing is not recoverable. It enters only
# through the dimensionless groups, so it rescales mu and lambda1 and leaves the residual
# shape alone -- `paam_sensitivity.py` measures how far that holds.
PAAM_MAXIMUM = 359.281e-6
PAAM = {
  "paam_PA05": ("PA05_exp_data.csv", (0, 52), 7.753),
  "paam_PA05_10C": ("PA05_temp_exp_data.csv", (0, 30), 7.836),
  "paam_PA05_21C": ("PA05_temp_exp_data.csv", (30, 87), 7.736),
  "paam_PA05_33C": ("PA05_temp_exp_data.csv", (87, 117), 7.451),
  "paam_PA05003": ("PA05003_temp_exp_data.csv", (0, 80), 8.007),
}


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


def trials(dataset):
  """The `(times, events)` matrix a record is built from, screened but not averaged.

  Gelatin comes from `results.json`, which carries only the mean and the spread, so the
  matrix is available for PAAm alone. Studies that need per-event traces on both use
  `trial_modes.FILES` for gelatin.
  """
  if dataset not in PAAM: raise KeyError(f"{dataset} has no released per-event matrix here")
  from pyimr.noise import characteristic_time

  filename, (low, high), _ = PAAM[dataset]
  table = np.loadtxt(TEMPSWEEPS / filename, delimiter=",", ndmin=2)
  tau, events = table[:, 0], table[:, 1 + low:1 + high].T
  # the same screen `measured_selection.screen` applies: a trace above its own maximum is a
  # tracking failure, and a sample where every event agrees exactly is the t*=0 definition
  keep = ~(events > 1.05).any(axis=0) & (events.std(axis=0, ddof=1) > 0.0)
  return tau[keep] * characteristic_time(PAAM_MAXIMUM), events[:, keep]


def load(dataset):
  """`(times, mean, spread, maximum_radius, stretch)` with the useless samples dropped."""
  if dataset in PAAM:
    times, events = trials(dataset)
    return (times, events.mean(axis=0), events.std(axis=0, ddof=1), PAAM_MAXIMUM,
            PAAM[dataset][2])
  record = json.loads((HERE / "results.json").read_text())[dataset]
  times, mean, spread = (np.array(record[k], dtype=float) for k in ("times_s", "mean", "spread"))
  keep = spread > 0
  return times[keep], mean[keep], spread[keep], DATASETS[dataset], record["stretch"]


def trial_count(dataset):
  """How many repeats the record carries. Not uniform: 18, 14 and 7 on gelatin.

  Named apart from `score`'s `trials=` argument deliberately: a module function and a
  keyword of the same name in one file is a shadow waiting for whoever calls one from
  inside the other.
  """
  if dataset in PAAM:
    low, high = PAAM[dataset][1]
    return high - low
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
          trials=None, seeds=None, correlation_time_s=None, times=None):
  """Fit, then the evidence summed over modes and the residual diagnostics at the best.

  Summed rather than taken at the best because the expansion is about one mode and the
  integral is over all of them; modes the expansion cannot use are dropped rather than
  allowed to abort the study.
  """
  from pyimr.noise import check_residuals, lack_of_fit
  from pyimr.selection import (PARAMETER_BOUNDS, candidate_log_evidence, evaluate_at,
                               fit_candidate, physical_from_unit)
  from scipy.special import logsumexp

  noise = dict(correlation_time_s=correlation_time_s, times=times)
  fit = fit_candidate(candidate, solve, mean, spread, bounds=bounds, starts=starts,
                      max_evaluations=evaluations, seeds=seeds, **noise)
  scored = []
  for point in fit.modes:
    try: scored.append(candidate_log_evidence(candidate, solve, mean, spread, point, bounds=bounds, **noise))
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
              # in the prior's unit coordinates, so a neighbouring fit can be warm-started
              # from it -- the physical `fitted` cannot be, the box may differ
              unit=[float(v) for v in fit.unit],
              lack_of_fit=misfit,
              lag_one=float(check.lag_one), n_eff=float(check.effective_samples),
              failure_fraction=fit.failure_fraction, modes=len(fit.modes), fitted=fitted,
              pinned=[k for k, v in fitted.items()
                      if min(abs(np.log(v / box[k][0])), abs(np.log(v / box[k][1]))) < 1e-6])
