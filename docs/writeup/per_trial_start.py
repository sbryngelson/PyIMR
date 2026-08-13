r"""The last axis #222 has never been given: a per-trial initial VELOCITY.

\Cref{sec:latent} has now failed to absorb the dominant trial mode three ways. The material axes
$(\mu, g\alpha, \lambda_1)$ refitted per trial leave it with its shape intact --- $\lvert\cos\rvert$
of $0.987$ and $0.943$ against the raw mode. Freeing $R_{\rm eq}$ over the $\pm11\%$ its prior
allows changes nothing at \SI{33}{\celsius} and pins at the two other records. Nine linearised
directions correlate at $0.467$ or less.

Every one of those tests varies what the bubble IS. None varies how it STARTS moving.

That gap matters for the same reason the $R_{\rm eq}$ one did, and more so. The mode is $97$ to
$98\%$ post-collapse. What a bubble carries through its collapse and into the rebound is set by
the energy it goes in with, and laser breakdown is not repeatable shot to shot --- the imaged
maximum is taken as a turning point of a free collapse, and if it is not, the error is a wall
velocity at $t=0$ that differs between trials.

\Cref{sec:deltascreen} makes this the largest candidate ever put through the discrepancy screen,
at $37.6\%$, and \texttt{initial\_condition.py} fits it on the mean traces. This asks the other
question: with each trial free to start differently, does the leading mode collapse?

THE STRETCH IS FREE HERE TOO, and the reason is \cref{sec:reqprior}. A stretch pinned at a
population mean known to five--seventeen percent would let $R_{\rm eq}$ error masquerade as
velocity scatter and vice versa. Both are configuration quantities, both are unmeasured, and the
honest test frees both and reports how far each moves.

WHAT WOULD BE CONVINCING. A mode that collapses --- share falling and $\lvert\cos\rvert$ to the
raw mode falling with it. What would NOT be convincing is $\chi^2$ improving, which five free
parameters per trial guarantee; the shape of what is left is the whole question.
"""

import json

import numpy as np

import records
from identified import BOX
from initial_condition import REQ_BOX, VELOCITY_BOXES, candidate_for, solver_with_start
from per_trial_fits import _trials

# 20, not 10: the mean-trace sweep found the (req_scale, u0) surface multimodal at 15 C, and a
# single trial is noisier than a mean over eighteen
STARTS, EVALUATIONS = 20, 500
WIDTH = "+-10%"                  # the middle of initial_condition.py's sweep
FREE = ("req_scale", "u0_shift")


def one(job):
  """One trial, with both configuration axes free alongside the material."""
  from pyimr.selection import evaluate_at, fit_candidate, physical_from_unit

  dataset, index = job
  times, mean, spread, maximum, stretch, window = _trials(dataset)
  candidate = candidate_for(FREE)
  solve = solver_with_start(times, maximum, stretch)
  box = dict(BOX) | {"req_scale": REQ_BOX, "u0_shift": VELOCITY_BOXES[WIDTH]}
  try:
    fit = fit_candidate(candidate, solve, window[:, index], spread, bounds=box, starts=STARTS,
                        max_evaluations=EVALUATIONS)
  except ValueError as error:
    return job, {"failed": str(error)}
  axes = tuple(candidate.axes)
  fitted = dict(zip(axes, (float(v) for v in physical_from_unit(axes, fit.unit, box)),
                    strict=True))
  model = np.asarray(evaluate_at(candidate, solve, fitted)[0], dtype=float)
  return job, {"fitted": fitted, "chi2_per_n": float(fit.chi_squared),
               "residual": (model - window[:, index]).tolist()}


def main():
  before = json.load(open(records.HERE / "per_trial_fits.json"))
  earlier = json.load(open(records.HERE / "per_trial_initial.json"))
  jobs = [(d, i) for d in records.DATASETS for i in range(_trials(d)[5].shape[1])]
  print(f"  refitting {len(jobs)} trials with req_scale AND u0 free ...", flush=True)
  with records.pool(len(jobs)) as pool:
    table = dict(pool.map(one, jobs))

  summary = {}
  print("\n  how far the two configuration axes move between trials\n")
  print(f"  {'record':13s} {'J':>3s} {'req_scale spread':>17s} {'u0/U spread':>12s} "
        f"{'u0/U range':>20s} {'pinned':>7s}")
  for dataset in records.DATASETS:
    rows = [table[(dataset, i)] for i in range(_trials(dataset)[5].shape[1])
            if "failed" not in table[(dataset, i)]]
    if not rows: continue
    req = np.array([r["fitted"]["req_scale"] for r in rows])
    vel = np.array([r["fitted"]["u0_shift"] - 1.0 for r in rows])
    low, high = VELOCITY_BOXES[WIDTH]
    pinned = int(np.sum((vel < low - 1.0 + 1e-3) | (vel > high - 1.0 - 1e-3)))
    summary[dataset] = {"trials": len(rows), "req_spread": float(req.std(ddof=1)),
                        "u0_spread": float(vel.std(ddof=1)),
                        "u0_range": [float(vel.min()), float(vel.max())],
                        "u0_median": float(np.median(vel)), "pinned": pinned,
                        "chi2_median": float(np.median([r["chi2_per_n"] for r in rows]))}
    print(f"  {dataset:13s} {len(rows):3d} {req.std(ddof=1):16.1%} {vel.std(ddof=1):11.3f} "
          f"  {vel.min():+.3f} to {vel.max():+.3f} {pinned:7d}")

  print("\n  does the dominant mode survive being allowed to start differently?\n")
  print(f"  {'record':13s} {'raw':>7s} {'material':>9s} {'+req':>7s} {'+req+u0':>9s} "
        f"{'|cos| to raw':>13s}")
  for dataset in records.DATASETS:
    if dataset not in summary: continue
    _, _, _, _, _, window = _trials(dataset)
    rows = [table[(dataset, i)] for i in range(window.shape[1])
            if "failed" not in table[(dataset, i)]]
    after = np.column_stack([r["residual"] for r in rows])
    after = after - after.mean(axis=1, keepdims=True)
    raw = window[:, : after.shape[1]]
    raw = raw - raw.mean(axis=1, keepdims=True)
    left_raw, values_raw, _ = np.linalg.svd(raw, full_matrices=False)
    left, values, _ = np.linalg.svd(after, full_matrices=False)
    share = float(values[0] ** 2 / (values**2).sum())
    overlap = abs(float(left_raw[:, 0] @ left[:, 0]))
    summary[dataset] |= {"mode_after": share, "overlap_with_raw": overlap}
    print(f"  {dataset:13s} {float(values_raw[0] ** 2 / (values_raw**2).sum()):7.3f} "
          f"{before[dataset]['mode_after']:9.3f} "
          f"{earlier[dataset]['mode_after_req']:7.3f} {share:9.3f} {overlap:13.3f}")

  print("\n  A mode that collapses ONLY once the start is free was latent variation in how the")
  print("  records begin, which no material axis and no equilibrium radius could absorb. One")
  print("  that survives this is not any quantity this model has, of any kind.")

  json.dump(summary, open(records.HERE / "per_trial_start.json", "w"), indent=1)


if __name__ == "__main__":
  main()
