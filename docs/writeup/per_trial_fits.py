r"""Fit every trial separately: is the dominant mode parameter variation or not?

Every previous attempt on #222 projected trial deviations onto LINEARISED sensitivities and
asked whether the dominant mode lay in their span. Nine directions failed. That test has a known
weakness: a linearisation about one fitted point can miss variation that a genuine refit would
absorb, because the parameters move far enough for the Jacobian to turn.

This runs the nonlinear version. Each trial is fitted on its own, in the identified coordinates
where the bound of \cref{sec:discrepancy} is computable, and two things are read off.

The SCATTER of the fitted parameters is a direct estimate of $\Sigma_\theta$, the
trial-to-trial parameter covariance the hierarchical model in #222 posits. Nobody has measured
it; it has only been inferred from projections.

The RESIDUALS after per-trial fitting are the decisive part. If the dominant mode is latent
parameter variation, giving each trial its own parameters absorbs it and the mode disappears
from the residual scatter. If the mode survives at the same share of variance, it is not
parameters -- not linearly, and not at the true optimum either, which is what the projection
tests could not establish.

A caveat that cannot be removed with this data. A single trial's noise is not known separately
from the trial spread, so the per-trial fits are weighted by the record's spread, which includes
the variation being estimated. That inflates the residuals and makes $\Sigma_\theta$ an
overestimate. It does not affect the mode-survival question, which is about SHAPE.
"""

import json

import numpy as np

import records
from identified import BOX, candidate_at_ratio
from trial_modes import DATA, FILES

RATIO = 38.5
AXES = ("mu", "galpha", "lambda1")
STARTS, EVALUATIONS = 8, 300


def _trials(dataset):
  filename, offset = FILES[dataset]
  times, mean, spread, maximum, stretch = records.load(dataset)
  raw = np.loadtxt(DATA / filename, delimiter=",")
  return times, mean, spread, maximum, stretch, raw[offset:offset + times.size, 1:]


def one(job):
  """Fit a single trial's trace in the identified coordinates."""
  from pyimr.selection import fit_candidate, physical_from_unit
  dataset, index = job
  times, mean, spread, maximum, stretch, window = _trials(dataset)
  candidate = candidate_at_ratio(RATIO)
  solve = records.solver(times, maximum, stretch)
  fit = fit_candidate(candidate, solve, window[:, index], spread, bounds=BOX, starts=STARTS,
                      max_evaluations=EVALUATIONS)
  values = physical_from_unit(AXES, fit.unit, BOX)
  fitted = dict(zip(AXES, (float(v) for v in values), strict=True))
  from pyimr.selection import evaluate_at
  model = evaluate_at(candidate, solve, fitted)[0]
  return (dataset, index), {"fitted": fitted, "chi2_per_n": float(fit.chi_squared),
                            "residual": (np.asarray(model, dtype=float)
                                         - window[:, index]).tolist()}


def main():
  jobs = [(d, i) for d in FILES for i in range(_trials(d)[5].shape[1])]
  print(f"  fitting {len(jobs)} trials across {len(FILES)} records ...", flush=True)
  with records.pool(len(jobs)) as pool:
    table = dict(pool.map(one, jobs))

  summary = {}
  print("\n  per-trial parameter scatter -- a direct estimate of Sigma_theta\n")
  print(f"  {'record':13s} {'J':>3s} " + " ".join(f"{a + ' cv':>12s}" for a in AXES)
        + f" {'median chi2/N':>14s}")
  for dataset in FILES:
    rows = [table[(dataset, i)] for i in range(_trials(dataset)[5].shape[1])]
    points = np.array([[r["fitted"][a] for a in AXES] for r in rows])
    logs = np.log(points)
    # a coefficient of variation in log coordinates, which is where the box and the bound live
    spread_cv = np.exp(logs.std(axis=0, ddof=1)) - 1.0
    summary[dataset] = {
      "trials": len(rows),
      "median": {a: float(np.exp(np.median(logs[:, k]))) for k, a in enumerate(AXES)},
      "cv": {a: float(spread_cv[k]) for k, a in enumerate(AXES)},
      "log_covariance": np.cov(logs, rowvar=False).tolist(),
      "chi2_median": float(np.median([r["chi2_per_n"] for r in rows])),
    }
    print(f"  {dataset:13s} {len(rows):3d} "
          + " ".join(f"{spread_cv[k]:11.1%}" for k in range(len(AXES)))
          + f" {summary[dataset]['chi2_median']:14.4f}")

  print("\n  does the dominant mode survive per-trial fitting?\n")
  print(f"  {'record':13s} {'mode 1 before':>14s} {'mode 1 after':>13s} {'|cos| before/after':>19s}")
  for dataset in FILES:
    _, _, _, _, _, window = _trials(dataset)
    before = window - window.mean(axis=1, keepdims=True)
    rows = [table[(dataset, i)] for i in range(window.shape[1])]
    after = np.column_stack([r["residual"] for r in rows])
    after = after - after.mean(axis=1, keepdims=True)
    shares, modes = [], []
    for matrix in (before, after):
      left, values, _ = np.linalg.svd(matrix, full_matrices=False)
      shares.append(float(values[0] ** 2 / (values**2).sum()))
      modes.append(left[:, 0])
    overlap = abs(float(modes[0] @ modes[1]))
    summary[dataset] |= {"mode_before": shares[0], "mode_after": shares[1], "overlap": overlap}
    print(f"  {dataset:13s} {shares[0]:14.3f} {shares[1]:13.3f} {overlap:19.3f}")

  print("\n  A mode that keeps its share AND its shape after every trial is given its own")
  print("  parameters is not parameter variation. One that collapses is.")
  json.dump(summary, open(records.HERE / "per_trial_fits.json", "w"), indent=1)


if __name__ == "__main__":
  main()
