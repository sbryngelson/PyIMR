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


def _score(dataset, name, *, starts, seeds=None):
  times, mean, spread, maximum, stretch = records.load(dataset)
  solve = records.solver(times, maximum, stretch, max_steps=600_000, **TREATMENTS[name])
  return records.score(_candidate(), solve, mean, spread, bounds=BOX, starts=starts,
                       evaluations=600, trials=records.trial_count(dataset), seeds=seeds)


def _job(dataset):
  """One record: the cold fit, then every thermal treatment warm-started from it.

  Raising the budget did not work. Across three budgets -- 5, 10 and 24 restarts -- the
  15 C and 33 C rows moved by 15 to 153 nats while 23 C was bit-stable, and at two cells the
  fit was WORSE at 24 restarts than at 10. Best-of-24 cannot lose to best-of-10 at a converged
  optimum, so the multistart was sampling basins rather than refining one, and a fourth budget
  would have been the same mistake a fourth time.

  The cold fit, by contrast, is converged everywhere: its log evidence agrees to 0.03 nats
  across those same budget changes. The treatments differ from it by added physics and not by
  parameters -- the material axes are identical, which is what makes the comparison a Bayes
  factor at all -- so the cold optimum is a far better starting point than any random one.
  Each thermal fit therefore starts there as well as from its own hypercube.
  """
  try:
    cold = _score(dataset, "cold", starts=24)
  except Exception as error:                          # noqa: BLE001
    return dataset, {n: {"failed": f"cold: {type(error).__name__}: {error}"} for n in TREATMENTS}

  got = {"cold": cold}
  for name in TREATMENTS:
    if name == "cold": continue
    try:
      got[name] = _score(dataset, name, starts=12, seeds=[cold["unit"]])
    except Exception as error:                        # noqa: BLE001
      got[name] = {"failed": f"{type(error).__name__}: {error}"}
  return dataset, got


def main():

  jobs = list(records.DATASETS)
  with records.pool(len(jobs)) as pool:
    nested = dict(pool.map(_job, jobs))
  table = {(d, n): row for d, rows in nested.items() for n, row in rows.items()}

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
