r"""What does a testable batch cost once the noise model is the measured one?

`noise_portfolio.py` leaves the design chapter with a broken sentence. Under the constant scale
the portfolio spreads over six geometries and carries $\mathrm{df} = +2$, so
\eqref{eq:constrained} never binds and \cref{sec:design} calls it the only aggressive option
that stays testable. Under two of the three measured profiles it spreads over four and
$\mathrm{df} = 0$: the lack-of-fit test has no power at all.

So the constraint has to be imposed rather than inherited, which is what
`pyimr.measure.constrained_measure` is for (#232). That function was written against the
operator design, where the free batch takes two settings and no amount of replication moves it.
This asks the same question of the portfolio under a noise model the records support, and
prices the answer.

The price is the whole point. Under the constant, testability was free -- the portfolio already
satisfied the floor, so nobody had to decide whether it was worth paying for. Under the
measured profiles it is not free, and the number below is what \cref{sec:design} would have to
report instead of the claim that the constraint never binds.

The floor is $k > p$, more distinct settings than parameters, because that is what gives the
lack-of-fit test a numerator at all. With four material parameters that means five, and
`lack_of_fit_degrees` then reads $\mathrm{df} = +1$. Six is also run, since the constant's own
batch reached that on its own and it shows what a comfortable margin costs.
"""

import json

import numpy as np

import records
from noise_design import RELATIVE_NOISE, profile, raw_information
from noise_portfolio import BUDGET, MATERIAL, OPERATOR, PARAMETERS, score

WEIGHT = 4.0                     # the operator's weight in the portfolio, as sec:design uses


def objective(nats):
  """The quantity `gain_criterion` actually maximises, so a cost is measured on it.

  Reporting the material component alone makes a constrained batch look BETTER than the free
  one, because the constraint can trade operator nats for material nats -- which the weighted
  sum forbids but a single component does not.
  """
  return nats["material"] + WEIGHT * nats["operator"]


def _portfolio():
  from pyimr.gain import Question
  return [Question("material", MATERIAL), Question("operator", OPERATOR, weight=4.0)]


def main():
  from design_operator import PERFORMED, RADII, STRETCH
  from pyimr.gain import gain_criterion, lack_of_fit_degrees
  from pyimr.measure import apportion, constrained_measure, optimal_measure

  designs = [(float(r), float(s)) for r in RADII for s in STRETCH] + list(PERFORMED.values())
  with records.pool(len(designs)) as pool:
    got = list(pool.map(raw_information, designs))
  usable = [(d, payload) for d, payload in got if payload is not None]
  points = [d for d, _ in usable]
  print(f"\n  {len(usable)} of {len(designs)} candidates integrate")

  profiles = {name: profile(name) for name in records.DATASETS}
  models = ["constant", *profiles]

  def fisher(columns, phase, model):
    scale = (np.full(columns.shape[0], RELATIVE_NOISE) if model == "constant"
             else np.interp(phase, *profiles[model]))
    whitened = columns / scale[:, None]
    matrix = whitened.T @ whitened
    return 0.5 * (matrix + matrix.T)

  stacks = {m: np.array([fisher(c, p, m) for _, (c, p) in usable]) for m in models}
  questions = _portfolio()
  criterion = gain_criterion(questions)

  print("\n  the portfolio, free and held to a testable support\n")
  print(f"  {'noise model':13s} {'demand':>8s} {'settings':>9s} {'lack-df':>8s} "
        f"{'material':>9s} {'operator':>9s} {'objective':>10s} {'given up':>10s} "
        f"{'certified':>10s}")
  table = {}
  for model in models:
    free = optimal_measure(stacks[model], criterion=criterion, iterations=400_000)
    rows = {}
    exact = apportion(free.weights, BUDGET, stacks[model], criterion=criterion)
    settings, _, lack_df, _ = lack_of_fit_degrees(exact.counts, parameters=PARAMETERS)
    base = score(stacks[model], exact.counts, questions)
    print(f"  {model:13s} {'free':>8s} {settings:9d} {lack_df:8d} {base['material']:9.3f} "
          f"{base['operator']:9.3f} {objective(base):10.3f} {'--':>10s} "
          f"{str(free.certified):>10s}")
    rows["free"] = {"settings": settings, "lack_df": lack_df, "nats": base,
                    "table": [(points[i][0], points[i][1], int(n))
                              for i, n in enumerate(exact.counts) if n > 0]}
    for wanted in (PARAMETERS + 1, PARAMETERS + 2):
      held = constrained_measure(stacks[model], wanted, criterion=criterion, iterations=400_000)
      exact = apportion(held.measure.weights, BUDGET, stacks[model], criterion=criterion)
      settings, _, lack_df, _ = lack_of_fit_degrees(exact.counts, parameters=PARAMETERS)
      got_nats = score(stacks[model], exact.counts, questions)
      lost = objective(base) - objective(got_nats)
      print(f"  {'':13s} {wanted:8d} {settings:9d} {lack_df:8d} {got_nats['material']:9.3f} "
            f"{got_nats['operator']:9.3f} {objective(got_nats):10.3f} {lost:10.3f} "
            f"{str(held.certified_under_floor):>10s}")
      rows[str(wanted)] = {"settings": settings, "lack_df": lack_df, "nats": got_nats,
                           "given_up": float(lost),
                           "table": [(points[i][0], points[i][1], int(n))
                                     for i, n in enumerate(exact.counts) if n > 0]}
    table[model] = rows
    print()

  free_df = {m: table[m]["free"]["lack_df"] for m in models}
  untestable = [m for m in models if free_df[m] < 1]
  costs = [table[m][str(PARAMETERS + 1)]["given_up"] for m in untestable]
  print(f"  free batches that cannot be tested (df < 1): {len(untestable)} of {len(models)}"
        f" -- {', '.join(untestable) if untestable else 'none'}")
  if costs:
    scale = max(objective(table[m]["free"]["nats"]) for m in untestable)
    print(f"  restoring df >= +1 on those costs {min(costs):.3f} to {max(costs):.3f} nats of "
          f"{scale:.1f}, or {min(costs) / scale:.1%} to {max(costs) / scale:.1%}")
  print("\n  Under the constant this table is uninteresting because the free batch already")
  print("  satisfied the floor. Under the measured profiles it is the price of an experiment")
  print("  that can detect its own model being wrong.")

  json.dump(table, open(records.HERE / "noise_testable.json", "w"), indent=1)


if __name__ == "__main__":
  main()
