"""Does the recommended batch move when the question changes?

`design_operator.py` maximises `log det M` over the augmented (material, operator) parameter
set. The identity `log det M = log det M_thth + log det S` says that criterion is material
volume plus discrimination volume, weighted equally -- a choice nobody made. This asks the
same 226 candidate geometries three different questions and compares the twelve bubbles each
one recommends:

  material   what pins the qSLS parameters             A = the four material coordinates
  operator   what tells the two operators apart        A = the operator coordinate alone
  both       a portfolio, weighted as stated           the sum of the two

and then re-apportions the winner under a replication constraint, because none of these
criteria can see model inadequacy and the lack-of-fit test on these records rejects.

Everything is computed at the 15 C material, so it ranks geometry and inherits the limit
`design_operator.py` states: at their own fitted materials the records separate by collapse
depth, not by geometry.
"""

import json

import numpy as np

import records
from design_operator import PERFORMED, RADII, STRETCH, information

BUDGET = 12
MATERIAL = (0, 1, 2, 3)
OPERATOR = (4,)
PARAMETERS = len(MATERIAL)


def _all():
  from pyimr.gain import Question
  return [Question("material", MATERIAL), Question("operator", OPERATOR)]


def batch(stack, points, questions, *, replicates=1):
  """The apportioned batch a question recommends, and what it is worth in nats."""
  from pyimr.gain import expected_gain, gain_criterion, lack_of_fit_degrees
  from pyimr.measure import apportion, optimal_measure

  ALL = _all()                       # every batch is scored on every question, not just its own

  criterion = gain_criterion(questions)
  measure = optimal_measure(stack, criterion=criterion, iterations=400_000)
  exact = apportion(measure.weights, BUDGET, stack, criterion=criterion, replicates=replicates)
  averaged = np.tensordot(exact.counts / BUDGET, stack, axes=(0, 0)) * BUDGET
  return {
    "gap": float(measure.gap), "certified": bool(measure.certified),
    "efficiency": float(exact.efficiency),
    "nats": expected_gain(averaged, ALL).per_question,
    "degrees": lack_of_fit_degrees(exact.counts, parameters=PARAMETERS),
    "table": [(points[i][0], points[i][1], int(exact.counts[i]))
              for i in np.argsort(-exact.counts) if exact.counts[i] > 0],
  }


def show(label, got):
  settings, reps, lack_df, pure_df = got["degrees"]
  nats = ", ".join(f"{k} {v:.2f}" for k, v in sorted(got["nats"].items()))
  print(f"\n  {label}   gap {got['gap']:.1e}  certified={got['certified']}  "
        f"D-eff {got['efficiency']:.3f}")
  print(f"    {nats} nats   |   {settings} settings, min {reps} runs, "
        f"lack-of-fit df {lack_df} and {pure_df}")
  for radius, stretch, runs in got["table"]:
    print(f"      R_max {radius * 1e6:7.1f} um   stretch {stretch:5.2f}   {runs:2d} runs")


def main():
  from pyimr.gain import Question

  designs = [(float(r), float(s)) for r in RADII for s in STRETCH] + list(PERFORMED.values())
  with records.pool(len(designs)) as pool:
    out = list(pool.map(information, designs))
  points = [d for d, m in out if m is not None]
  stack = np.array([m for _, m in out if m is not None])
  print(f"{len(points)} of {len(designs)} candidates integrate under both operators")

  asked = {
    "material": [Question("material", MATERIAL)],
    "operator": [Question("operator", OPERATOR)],
    "both": [Question("material", MATERIAL), Question("operator", OPERATOR, weight=4.0)],
  }
  answers = {name: batch(stack, points, q) for name, q in asked.items()}
  for name, got in answers.items(): show(f"{name:9s}", got)

  print("\n  the same operator question, held testable at three replicates per setting:")
  held = batch(stack, points, asked["operator"], replicates=3)
  show("operator+3", held)
  answers["operator_replicated"] = held

  free, bound = answers["operator"]["nats"]["operator"], held["nats"]["operator"]
  settings, reps, lack_df, _ = held["degrees"]
  print(f"\n  replication costs {free - bound:.2f} of {free:.2f} nats about the operator and "
        f"buys {reps} repeats per setting, so pure error is now estimable.")
  print(f"  it does NOT make the batch testable: {settings} settings against {PARAMETERS} "
        f"parameters leaves lack-of-fit df {lack_df}. The binding constraint is the NUMBER of")
  print("  settings, which is a property of the measure and not of the rounding (#232).")

  json.dump({k: {**v, "degrees": list(v["degrees"])} for k, v in answers.items()},
            open(records.HERE / "gain_design.json", "w"), indent=1)


if __name__ == "__main__":
  main()
