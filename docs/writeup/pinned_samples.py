r"""Does the pinned normalisation sample change the fit, or only the weights?

Every trace is divided by its OWN maximum, so at the sample where it attains that maximum it
equals $1$ for every trial and the spread there measures nothing at all. Measured, the smallest
per-sample spread on each record sits at index $0$ or $1$ where the mean is $0.9996$ or above,
and it is $30$ to $90$ times below the record's median. Weighted by $1/\sigma^2$ -- which is
what `fit_candidate` does with it -- the first sample alone carries $28$, $48$ and $63\%$ of the
likelihood, and the first five carry $81$ to $91\%$.

Whether that MATTERS is a different question from whether it is true, and the answer is
reassuring in one direction and not the other.

The parameters are robust. Flooring the spread at its tenth or twenty-fifth percentile, which
removes most of the pinned weight, moves the identified coordinates by at most $22\%$ and
usually by less than $5$. So the published fits are not an artifact of this.

The fit STATISTICS are not robust. The same flooring moves $\chi^2/N$ from $1.03$ to $0.40$ at
\SI{23}{\celsius} -- a factor of $2.6$ -- and every goodness-of-fit statement, lack-of-fit
ratio and evidence built on it inherits that choice.

Dropping the samples instead of flooring them is the wrong experiment, and it is included to
show why: it moves the parameters by factors of $12$ to $46$ and drives $\chi^2/N$ past $12$.
Those early samples carry the initial condition and the phase. They are load-bearing rather
than spurious, which is exactly why the fix is a floor and not an exclusion.
"""
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import numpy as np

import records
from identified import BOX, candidate_at_ratio

RATIO = 38.5
STARTS, EVALUATIONS = 16, 600


def one(job):
  from pyimr.selection import fit_candidate, physical_from_unit
  dataset, treatment = job
  times, mean, spread, maximum, stretch = records.load(dataset)
  used = np.array(spread, dtype=float)
  keep = np.ones(times.size, dtype=bool)
  if treatment == "floor p10":
    used = np.maximum(used, float(np.quantile(used, 0.10)))
  elif treatment == "drop first 10":
    keep[:10] = False
  elif treatment == "floor p25":
    used = np.maximum(used, float(np.quantile(used, 0.25)))

  candidate = candidate_at_ratio(RATIO)
  solve = records.solver(times[keep], maximum, stretch)
  fit = fit_candidate(candidate, solve, mean[keep], used[keep], bounds=BOX, starts=STARTS,
                      max_evaluations=EVALUATIONS)
  axes = tuple(candidate.axes)
  values = physical_from_unit(axes, fit.unit, BOX)
  return (dataset, treatment), {
    "fitted": dict(zip(axes, (float(v) for v in values), strict=True)),
    "chi2_per_n": float(fit.chi_squared),
  }


def main():
  treatments = ("as published", "floor p10", "floor p25", "drop first 10")
  jobs = [(d, t) for d in records.DATASETS for t in treatments]
  with records.pool(len(jobs)) as pool:
    table = dict(pool.map(one, jobs))
  axes = ("mu", "galpha", "lambda1")
  print("\n  fitted parameters with and without the pinned normalisation samples\n")
  for dataset in records.DATASETS:
    base = table[(dataset, "as published")]["fitted"]
    print(f"  {dataset}")
    print(f"    {'treatment':16s} " + " ".join(f"{a:>12s}" for a in axes)
          + f" {'chi2/N':>9s}   ratio to published")
    for t in treatments:
      got = table[(dataset, t)]
      ratio = " ".join(f"{a} x{got['fitted'][a] / base[a]:.2f}" for a in axes)
      print(f"    {t:16s} " + " ".join(f"{got['fitted'][a]:12.5g}" for a in axes)
            + f" {got['chi2_per_n']:9.4f}   {ratio}")
    print()


if __name__ == "__main__":
  main()
