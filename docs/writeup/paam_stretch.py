r"""What equilibrium radius does each record actually want, and does the rejection survive it?

Two measurements arrived at the same input from different directions. `paam_sensitivity.py`
finds $F$ running $12.5$ to $23.3$ across the stretch's own uncertainty while $R_{\max}$ moves
it by \SI{4}{\percent}, and `paam_enrichment.py` finds the initial radius the largest single
candidate on six of eight records, up to \SI{36}{\percent} of the discrepancy. Both say the
equilibrium radius carries a share of what is being called model error.

THE SWEEP HAD A BRACKET EDGE IN IT, WHICH IS WHY THIS EXISTS. $\chi^2$ fell monotonically
toward the smallest stretch tried, so $7.45$ was an argmin at a wall rather than an optimum and
the record wants something below it. A wall is not a measurement, so the bracket is widened
here until the minimum is interior or the record is reported as not having one.

GELATIN IS THE CONTROL AND IT IS THE POINT. The PAAm stretches are estimated here, from a tail
median, so a PAAm-only result would be about the estimate. Gelatin's come from the source
paper's dataset table, and if those ALSO sit off their own $\chi^2$ minimum then this is not an
artefact of how PAAm was prepared -- it is a statement about a number the field pins and
publishes without an error bar, and \cref{sec:reqprior} already argues it should not be pinned.

WHAT WOULD OVERTURN THE FIRST LINK. If the rejection needs the assumed stretch, then at the
stretch each record prefers the qSLS is not rejected and \cref{sec:discrepancy} loses its first
step. The last column is that test.
"""

import json

import numpy as np

import records
from lackoffit import PARAMETER_SHARE, WIDE

STRETCHES = (5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5)
ORDER = (*records.DATASETS, *records.PAAM)


def _job(job):
  from pyimr.selection import STANDARD_MODELS, evaluate_at

  dataset, stretch = job
  times, mean, spread, maximum, _ = records.load(dataset)
  solve = records.solver(times, maximum, stretch, max_steps=600_000)
  candidate = STANDARD_MODELS["qSLS"]
  try:
    scored = records.score(candidate, solve, mean, spread, bounds=WIDE, starts=10,
                           evaluations=300)
    model = np.asarray(evaluate_at(candidate, solve, scored["fitted"])[0], dtype=float)
  except Exception as error:                                        # noqa: BLE001
    return job, {"failed": f"{type(error).__name__}: {error}"}
  count = records.trial_count(dataset)
  pure = float(np.sum(spread**2)) / mean.size
  lack = count * float(np.sum((mean - model) ** 2)) / (mean.size - 4)
  return job, {"chi2_per_n": scored["chi2_per_n"], "f_ratio": lack / pure,
               "f_corrected": lack / pure / (1.0 - PARAMETER_SHARE),
               "fitted": scored["fitted"]}


def main():
  jobs = [(d, s) for d in ORDER for s in STRETCHES]
  print(f"  {len(jobs)} fits: every record against every stretch\n")
  with records.pool(len(jobs)) as pool:
    table = dict(pool.map(_job, jobs))

  assumed = {**{d: records.load(d)[4] for d in records.DATASETS},
             **{d: records.PAAM[d][2] for d in records.PAAM}}

  print(f"  {'record':>14s} {'assumed':>8s} " + " ".join(f"{s:>7.1f}" for s in STRETCHES)
        + f" {'argmin':>8s}")
  summary = {}
  for dataset in ORDER:
    row = {s: table[(dataset, s)] for s in STRETCHES}
    good = {s: v for s, v in row.items() if "failed" not in v}
    if not good:
      print(f"  {dataset:>14s}   every fit failed"); continue
    chi = {s: v["chi2_per_n"] for s, v in good.items()}
    best = min(chi, key=chi.get)
    edge = best in (min(good), max(good))
    print(f"  {dataset:>14s} {assumed[dataset]:8.2f} "
          + " ".join(f"{chi.get(s, float('nan')):7.3f}" for s in STRETCHES)
          + f" {best:8.2f}" + ("  AT A WALL" if edge else ""))
    summary[dataset] = {"assumed": assumed[dataset], "argmin": best, "at_wall": bool(edge),
                        "chi2": {str(s): v["chi2_per_n"] for s, v in good.items()},
                        "f_ratio": {str(s): v["f_ratio"] for s, v in good.items()},
                        "fitted": {str(s): v["fitted"] for s, v in good.items()}}

  print("\n  and the lack-of-fit ratio at each stretch: does the rejection need the assumption?\n")
  print(f"  {'record':>14s} " + " ".join(f"{s:>7.1f}" for s in STRETCHES)
        + f" {'F at argmin':>12s} {'smallest F':>11s}")
  for dataset, v in summary.items():
    fs = v["f_ratio"]
    print(f"  {dataset:>14s} " + " ".join(f"{fs.get(str(s), float('nan')):7.1f}" for s in STRETCHES)
          + f" {fs[str(v['argmin'])]:12.1f} {min(fs.values()):11.1f}")

  print("\n  ---- what it says ----\n")
  walls = [d for d, v in summary.items() if v["at_wall"]]
  off = [d for d, v in summary.items() if abs(v["argmin"] - v["assumed"]) > 0.5]
  floor = min((min(v["f_ratio"].values()) for v in summary.values()), default=float("nan"))
  print(f"  {len(walls)} of {len(summary)} records put their chi2 minimum at a bracket wall"
        f"{': ' + ', '.join(walls) if walls else ''}.")
  print(f"  {len(off)} sit more than 0.5 in stretch away from the value they are carried at.")
  gel = [d for d in off if d in records.DATASETS]
  if gel:
    print(f"  Among them {', '.join(gel)}, whose stretch is TABULATED rather than estimated")
    print("  here, so this is not an artefact of how the PAAm records were prepared.")
  print(f"\n  The smallest lack-of-fit ratio anywhere in the sweep is {floor:.1f}, against a")
  print("  critical value near 1.2. The rejection therefore does not rest on the assumed")
  print("  equilibrium radius, though the SIZE of any quoted F plainly does.")
  records.HERE.joinpath("paam_stretch.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
  main()
