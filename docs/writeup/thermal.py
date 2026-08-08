"""Does the record support a thermal treatment, scored rather than assumed?

Thermal is the largest of the three model axes by sensitivity -- it moves the trace by 16
noise units and keeps 8.5 of them after the material absorbs what it can -- and it is the
only one never scored. The one number in circulation is a chi^2/N that got slightly worse
when it was switched on, which is a fit statistic and not evidence: a model that fits a
little worse can still be preferred if it buys that with fewer effective parameters, and one
that fits better can lose.

Run in identified coordinates (`g*alpha` free, `g/alpha` fixed) because the ridge otherwise
makes any ranking a statement about the prior box, and with the Occam factor capped because
the treatments differ in what they can resolve rather than in parameter count.
"""

import json

import numpy as np

import records

RATIO = 38.5
BOX = {"mu": (1e-5, 1e1), "galpha": (1e0, 1e7), "lambda1": (1e-9, 1e-2)}
# each treatment adds physics, not parameters: the material axes are identical throughout,
# so the Occam terms cancel and the difference in log evidence is a Bayes factor
TREATMENTS = {
  "cold": {},
  "bubble": {"bubtherm": 1, "Nt": 11},
  "bubble+medium": {"bubtherm": 1, "medtherm": 1, "Nt": 11, "Mt": 11},
}


def _candidate():
  import pyimr
  from pyimr.selection import CandidateModel

  def build(t):
    product = t["galpha"]
    return pyimr.QuadraticZener(float(np.sqrt(product * RATIO)), t["mu"], t["lambda1"], 0.0,
                                float(np.sqrt(product / RATIO)))

  return CandidateModel("qSLS|identified", build, ("mu", "galpha", "lambda1"))


def _job(argument):
  dataset, name = argument
  times, mean, spread, maximum, stretch = records.load(dataset)
  solve = records.solver(times, maximum, stretch, max_steps=600_000, **TREATMENTS[name])
  try:
    # starts=24, not 10. At 10 the cold baselines still disagreed across records in the way
    # that indicates a search failing on some rows rather than physics -- 15 C reported
    # bubble+medium at -131 nats while 23 C reported it at +8 for the same treatment. Only
    # nine fits run here, so the budget is cheap; a margin read off an under-converged
    # baseline is not.
    got = records.score(_candidate(), solve, mean, spread, bounds=BOX, starts=24,
                        evaluations=600, trials=records.trial_count(dataset))
  except Exception as error:                          # noqa: BLE001
    got = {"failed": f"{type(error).__name__}: {error}"}
  return (dataset, name), got


def main():

  jobs = [(d, n) for d in records.DATASETS for n in TREATMENTS]
  with records.pool(len(jobs)) as pool:
    table = dict(pool.map(_job, jobs))

  print("thermal treatments scored by evidence, identified coordinates, capped Occam\n")
  print(f"{'record':>14} {'treatment':>15} {'chi2/N':>9} {'log Z':>11} {'vs cold':>9} {'lag-1':>8}")
  for dataset in records.DATASETS:
    cold = table[(dataset, "cold")]
    for name in TREATMENTS:
      r = table[(dataset, name)]
      if "failed" in r:
        print(f"{dataset:>14} {name:>15}   {r['failed'][:40]}")
        continue
      against = r["log_evidence"] - cold["log_evidence"] if "failed" not in cold else float("nan")
      print(f"{dataset:>14} {name:>15} {r['chi2_per_n']:9.3f} {r['log_evidence']:11.2f} "
            f"{against:+9.2f} {r['lag_one']:8.3f}")

  # A baseline that did not converge makes every margin on its row meaningless, and an
  # under-converged fit reports a bad chi^2 rather than announcing itself. A first run of this
  # study at starts=5 left two of the three cold fits at chi2/N of 12.8 and 8.9 against 0.97
  # and 1.15 for the same model elsewhere, and reported +1209 and +829 nats as though they
  # were physics. So the rows are checked against each other before anything is believed.
  print()
  for dataset in records.DATASETS:
    usable = [r["chi2_per_n"] for r in (table[(dataset, n)] for n in TREATMENTS) if "failed" not in r]
    if usable and max(usable) > 3.0 * min(usable):
      print(f"  WARNING {dataset}: chi2/N spans {min(usable):.2f} to {max(usable):.2f} across")
      print("    treatments of the same data. That is a search that failed on some rows, not")
      print("    physics; the margins on this record are not interpretable.")

  print("\n  A positive `vs cold` favours the thermal treatment. The material axes are the same")
  print("  in every row, so the Occam terms cancel and this is a Bayes factor between physics,")
  print("  not between parameter counts. Read beside chi2/N: the two can disagree, and that")
  print("  disagreement is the whole reason a fit statistic was not enough.")
  records.HERE.joinpath("thermal.json").write_text(
    json.dumps({f"{d}|{n}": v for (d, n), v in table.items()}, indent=1))


if __name__ == "__main__":
  main()
