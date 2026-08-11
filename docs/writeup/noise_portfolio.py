r"""Does the noise model move the BATCH, or only the single best geometry?

`noise_design.py` shows the single-design argmax moves: two of three measured profiles prefer
\SI{416}{\micro\metre} at stretch $10.85$ over the constant's \SI{514}{\micro\metre} at $6.92$,
at up to fifteen percent of the gain. A single argmax is not what \cref{sec:batch} recommends,
though. The recommendation is a MEASURE over geometries, apportioned into twelve runs, and a
measure can be stable where its argmax is not: it spreads over several settings, so a
near-tie's resolution matters less.

So the question is whether the batch moves, and it is asked the same way -- exactly, not by a
bound, because the misspecification is a covariance about a shared mean.

Three things are compared under each noise model, since they can disagree. The nats each
question is worth is the criterion's own answer. The GEOMETRIES the batch visits is what an
experimentalist would actually do differently. And the lack-of-fit degrees of freedom is
whether the batch stays testable at all (#232), which no information criterion sees.

The regret column is the one that decides it: what the constant's batch gives up, judged by
each profile. A batch chosen under the wrong noise model and costing nothing under the right
one is a batch nobody needs to revisit. Small NEGATIVE regrets appear and are apportionment
rounding rather than a failure of the search -- twelve runs over a handful of settings is a
coarse grid, and a rounded batch can beat another rounded batch by a few hundredths.

WHAT IT COSTS IS NOT INFORMATION, IT IS TESTABILITY, and that is the finding. Under the
constant the portfolio spends its twelve runs over six geometries and carries $\mathrm{df} = +2$
for lack of fit; \cref{sec:design} calls it the only aggressive option that stays testable, and
says \eqref{eq:constrained} never binds because the portfolio spreads on its own. Under two of
the three measured profiles it spreads over FOUR geometries and $\mathrm{df} = 0$: the test has
no power at all. Meanwhile the nats it gives up are $0.45$ to $0.77$ of about nineteen, two to
four percent.

So the constraint of #232 stops being a formality the portfolio satisfies for free and becomes
something that has to be imposed. That is a stronger consequence than the geometry moving,
because a batch that cannot detect a wrong model is the failure mode \cref{sec:fitquality}
already found on every existing record.
"""

import json

import numpy as np

import records
from noise_design import RELATIVE_NOISE, profile, raw_information

BUDGET = 12
MATERIAL, OPERATOR = (0, 1, 2, 3), (4,)
PARAMETERS = len(MATERIAL)


def _questions():
  from pyimr.gain import Question
  return {
    "material": [Question("material", MATERIAL)],
    "operator": [Question("operator", OPERATOR)],
    "portfolio": [Question("material", MATERIAL), Question("operator", OPERATOR, weight=4.0)],
  }


def batch_for(stack, questions):
  """The apportioned batch a question set recommends under one noise model."""
  from pyimr.gain import gain_criterion
  from pyimr.measure import apportion, optimal_measure
  criterion = gain_criterion(questions)
  measure = optimal_measure(stack, criterion=criterion, iterations=400_000)
  exact = apportion(measure.weights, BUDGET, stack, criterion=criterion)
  return exact, measure


def score(stack, counts, questions):
  """What a batch is worth, in nats per question, under a given noise model."""
  from pyimr.gain import expected_gain
  averaged = np.tensordot(counts / BUDGET, stack, axes=(0, 0)) * BUDGET
  return expected_gain(averaged, questions).per_question


def main():
  from pyimr.gain import lack_of_fit_degrees

  from design_operator import PERFORMED, RADII, STRETCH
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
  asked = _questions()
  everything = [q for group in (asked["material"], asked["operator"]) for q in group]

  results = {}
  for label, questions in asked.items():
    print(f"\n  ===== {label}\n")
    print(f"  {'noise model':13s} {'settings':>9s} {'lack-df':>8s} {'material':>9s} "
          f"{'operator':>9s} {'regret vs its own best':>23s}")
    chosen = {}
    for model in models:
      exact, _ = batch_for(stacks[model], questions)
      chosen[model] = exact
    for model in models:
      own = score(stacks[model], chosen[model].counts, everything)
      under_constant = score(stacks[model], chosen["constant"].counts, everything)
      settings, _, lack_df, _ = lack_of_fit_degrees(chosen[model].counts,
                                                    parameters=PARAMETERS)
      # the criterion's own objective, so the regret is measured on what was optimised
      key = "operator" if label == "operator" else "material"
      regret = own[key] - under_constant[key]
      note = "-- (it is the choice)" if model == "constant" else f"{regret:+.4f} nats on {key}"
      print(f"  {model:13s} {settings:9d} {lack_df:8d} {own['material']:9.3f} "
            f"{own['operator']:9.3f} {note:>23s}")
    results[label] = {
      model: {"counts": chosen[model].counts.tolist(),
              "table": [(points[i][0], points[i][1], int(n))
                        for i, n in enumerate(chosen[model].counts) if n > 0]}
      for model in models}
    same = sum(np.array_equal(chosen[m].counts, chosen["constant"].counts)
               for m in models if m != "constant")
    print(f"\n    batches identical to the constant's: {same} of {len(profiles)}")
    for model in models:
      geometries = ", ".join(f"{r * 1e6:.0f}um@{s:.1f}x{n}"
                             for r, s, n in results[label][model]["table"])
      print(f"    {model:13s} {geometries}")

  json.dump(results, open(records.HERE / "noise_portfolio.json", "w"), indent=1)


if __name__ == "__main__":
  main()
