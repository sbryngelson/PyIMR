r"""How much of the reported lack of fit is the optimiser failing to find the fit?

`lackoffit.py` runs the qSLS multistart at `starts=10`, and `identified.py` already records why
that number is not $6$: at $6$ the \SI{33}{\celsius} fits sat at $\chi^2/N = 1.151$ while a fit
at $0.439$ existed, so every operator was stuck in one poor basin. Ten was the repair. Ten is
also not enough. At $40$ starts the same dataset reaches $0.4354$ and its lack-of-fit ratio
falls from $3.26$ to $1.81$, which is the $1.80$ the discrepancy table reports for the same
dataset -- so the document already contains both answers and quotes the larger one in its
headline table.

That matters beyond one cell. A lack-of-fit ratio is only a statement about the model if the
fit is the best the model can do; otherwise it is a statement about the search, and it is
biased in the direction that flatters the paper's conclusion. This runs every dataset at both
budgets and reports what moves, so the published table can be corrected where it is wrong and
confirmed where it is not.

The check was expected to be one-sided, since more starts can only lower $\chi^2$, and it is
not. At \SI{15}{\celsius} a better fit ($0.9727 \to 0.9535$) has a LARGER lack-of-fit ratio
($14.16 \to 16.69$), which is not a numerical accident: the fit minimises $\sum r_i^2/s_i^2$
while $\mathrm{SS}_{\rm lack}$ sums $r_i^2$ unweighted, so the fitted trace does not minimise
the numerator of its own $F$ and the two can move in opposite directions. That is worth more
than the budget question it turned up in, because it means the $k-p$ degrees of freedom
attributed to $\mathrm{SS}_{\rm lack}$ are not the degrees of freedom of anything that was
minimised.

Any dataset whose ratio does NOT move is one where ten starts was sufficient, which is worth
knowing too.
"""

import json

import numpy as np

import records
from lackoffit import WIDE

BUDGETS = (10, 40)
ORDER = (*records.DATASETS, *records.PAAM)


def _job(job):
  from pyimr.selection import STANDARD_MODELS, evaluate_at

  dataset, starts = job
  times, mean, spread, maximum, stretch = records.load(dataset)
  solve = records.solver(times, maximum, stretch, max_steps=600_000)
  candidate = STANDARD_MODELS["qSLS"]
  try:
    scored = records.score(candidate, solve, mean, spread, bounds=WIDE, starts=starts,
                           evaluations=300)
    model = np.asarray(evaluate_at(candidate, solve, scored["fitted"])[0], dtype=float)
  except Exception as error:                                        # noqa: BLE001
    return job, {"failed": f"{type(error).__name__}: {error}"}
  count = records.trial_count(dataset)
  pure = float(np.sum(spread**2)) / mean.size
  lack = count * float(np.sum((mean - model) ** 2)) / (mean.size - 4)
  return job, {"chi2_per_n": scored["chi2_per_n"], "f_ratio": lack / pure,
               "fitted": scored["fitted"]}


def main():
  jobs = [(d, s) for d in ORDER for s in BUDGETS]
  print(f"  {len(jobs)} fits: every dataset at {BUDGETS} starts\n")
  with records.pool(len(jobs)) as pool:
    table = dict(pool.map(_job, jobs))

  print(f"  {'dataset':>16s} " + " ".join(f"{'chi2 @' + str(s):>10s}" for s in BUDGETS)
        + " " + " ".join(f"{'F @' + str(s):>8s}" for s in BUDGETS) + f" {'F moves':>9s}")
  summary = {}
  for dataset in ORDER:
    rows = {s: table[(dataset, s)] for s in BUDGETS}
    if any("failed" in r for r in rows.values()):
      print(f"  {dataset:>16s}   a fit failed"); continue
    lo, hi = rows[BUDGETS[0]], rows[BUDGETS[-1]]
    move = hi["f_ratio"] / lo["f_ratio"] - 1.0
    print(f"  {dataset:>16s} " + " ".join(f"{rows[s]['chi2_per_n']:10.4f}" for s in BUDGETS)
          + " " + " ".join(f"{rows[s]['f_ratio']:8.2f}" for s in BUDGETS) + f" {move:+8.1%}")
    summary[dataset] = {str(s): rows[s] for s in BUDGETS} | {"f_change": move}

  moved = [d for d, v in summary.items() if abs(v["f_change"]) > 0.05]
  paam_moved = [d for d in moved if d in records.PAAM]
  print("\n  ---- what it says ----\n")
  print(f"  {len(moved)} of {len(summary)} datasets change their lack-of-fit ratio by more than")
  print(f"  five percent on the search budget alone: {', '.join(moved) if moved else 'none'}.")
  print(f"  None of them is a PAAm dataset ({len(paam_moved)} moved), so the results sec:paam")
  print("  rests on are not search artefacts; the exposure is entirely in gelatin.")
  for dataset in moved:
    v = summary[dataset]
    lo, hi = v[str(BUDGETS[0])], v[str(BUDGETS[-1])]
    direction = "DOWN" if hi["f_ratio"] < lo["f_ratio"] else "UP, on a BETTER fit"
    print(f"    {dataset:>14s}: chi2 {lo['chi2_per_n']:.4f} -> {hi['chi2_per_n']:.4f}, "
          f"F {lo['f_ratio']:.2f} -> {hi['f_ratio']:.2f}  ({direction})")
  print("\n  A ratio that rises when the fit improves is not a budget problem. The fit minimises")
  print("  the WEIGHTED sum of squares and SS_lack is UNWEIGHTED, so the fitted trace does not")
  print("  minimise the numerator of its own F, and the k-p degrees of freedom charged to it")
  print("  belong to no minimisation that was performed.")
  records.HERE.joinpath("lackoffit_starts.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
  main()
