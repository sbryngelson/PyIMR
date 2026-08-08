"""Does the operator ranking survive letting the initial radius float?

`initial_state.py` measures that a 1.68% error in `Req` leaves the same residual as changing
the bubble-dynamics operator, at 58 degrees to it. Comparable in size, separable in
direction: the geometry says an experiment could tell them apart, and says nothing about
whether these three records did. Every fit in this document pins `Req` at its inferred value,
which asserts a precision the experiment does not have.

So the operator comparison is run again with `Req` fitted as a nuisance parameter. The
question is not whether the evidence changes -- adding a parameter always changes it -- but
whether the ORDERING does. An operator preferred by 74 nats with `Req` pinned, and not
preferred once it floats, was never an operator result.

The prior width is the whole argument and there is no measurement that fixes it, so it is
swept rather than chosen. Too wide and `Req` absorbs the operator difference, which would
read as "the operators are indistinguishable" while being a statement about the box; too
narrow and this reproduces the pinned answer by construction. Reporting the ranking at each
width is the only honest form: a conclusion that holds across the plausible range is a
conclusion, and one that flips inside it is a prior.

Run in identified coordinates (`g*alpha` free, `g/alpha` fixed) for the same reason as
`identified.py`: the ridge otherwise makes any ranking a statement about the prior box.
"""

import json

import numpy as np

import records

RATIO = 38.5
BASE_BOX = {"mu": (1e-5, 1e1), "galpha": (1e0, 1e7), "lambda1": (1e-9, 1e-2)}
# None pins Req at its inferred value, as every other study here does. The rest are
# half-widths on a log-symmetric prior about it.
WIDTHS = (None, 0.03, 0.10, 0.25)
# Four axes rather than three, so the search needs more than `identified.py` used. At
# starts=10 the monotonicity gate below caught five cells whose fit was worse in a WIDER box
# than a narrower one -- one of them reported -117 nats between a +13 and a +7, which reads
# as a prior effect and is a failed search.
STARTS, EVALUATIONS = 24, 600
DECISIVE = 5.0                      # nats; below this an ordering is not claimed either way
# how much worse chi2/N may get on widening before the fit is called unconverged; the true
# bound is 1.0, and the slack is for a search that lands a shade off the same optimum
TOLERANCE = 1.02


def _candidate(width):
  import pyimr
  from pyimr.selection import CandidateModel

  def build(t):
    product = t["galpha"]
    return pyimr.QuadraticZener(float(np.sqrt(product * RATIO)), t["mu"], t["lambda1"], 0.0,
                                float(np.sqrt(product / RATIO)))

  axes = ("mu", "galpha", "lambda1")
  if width is None:
    return CandidateModel("qSLS|identified", build, axes)
  return CandidateModel(f"qSLS|identified|req{width:g}", build, (*axes, "req_scale"),
                        config_axes=("req_scale",))


def _box(width):
  if width is None: return dict(BASE_BOX)
  return {**BASE_BOX, "req_scale": (1.0 - width, 1.0 / (1.0 - width))}


def _job(argument):
  dataset, operator, width = argument
  times, mean, spread, maximum, stretch = records.load(dataset)
  solve = records.solver(times, maximum, stretch, dynamics=operator[0], liquid_eos=operator[1],
                         max_steps=600_000)
  try:
    got = records.score(_candidate(width), solve, mean, spread, bounds=_box(width),
                        starts=STARTS, evaluations=EVALUATIONS,
                        trials=records.trial_count(dataset))
  except Exception as error:                          # noqa: BLE001
    got = {"failed": f"{type(error).__name__}: {error}"}
  return (dataset, operator, width), got


def main():
  from pyimr import operator_name
  from pyimr.selection import DYNAMICS_MODELS

  operators = list(DYNAMICS_MODELS)
  jobs = [(d, o, w) for d in records.DATASETS for w in WIDTHS for o in operators]
  with records.pool(len(jobs)) as pool:
    table = dict(pool.map(_job, jobs))

  # Every wider box CONTAINS every narrower one, and all of them contain `req_scale = 1`,
  # which is the pinned fit. So the best attainable chi^2/N can only fall as the width grows.
  # A row that rises has not found its own optimum, and that is a statement about the search
  # with no tuning constant in it -- unlike a threshold on the spread, which would have to be
  # chosen. Cells that fail are dropped from the ordering test rather than read as physics.
  suspect = set()
  for dataset in records.DATASETS:
    for operator in operators:
      best = float("inf")
      for width in WIDTHS:
        row = table[(dataset, operator, width)]
        if "failed" in row: continue
        if row["chi2_per_n"] > TOLERANCE * best:
          suspect.add((dataset, operator, width))
        best = min(best, row["chi2_per_n"])
  if suspect:
    print(f"  UNCONVERGED: {len(suspect)} of {len(jobs)} cells fit worse in a wider box than a\n"
          "    narrower one, which is impossible at the optimum. Excluded from the ordering test:")
    for dataset, operator, width in sorted(suspect, key=lambda k: (k[0], k[1], k[2] or 0)):
      row = table[(dataset, operator, width)]
      print(f"      {dataset} {operator_name(*operator)} at "
            f"{'pinned' if width is None else f'{width:.0%}'}: chi2/N {row['chi2_per_n']:.3f}")
    print()

  reference = ("keller-miksis", None)
  for dataset in records.DATASETS:
    print(f"\n{dataset}: operator evidence against keller-miksis, by Req prior width\n")
    header = "".join(f"{'pinned' if w is None else f'+/-{w:.0%}':>12}" for w in WIDTHS)
    print(f"{'operator':>30}{header}   chi2/N (pinned -> widest)")
    for operator in operators:
      cells, chis = [], []
      for width in WIDTHS:
        row, base = table[(dataset, operator, width)], table[(dataset, reference, width)]
        if "failed" in row or "failed" in base:
          cells.append(f"{'--':>12}"); chis.append(float("nan")); continue
        cells.append(f"{row['log_evidence'] - base['log_evidence']:+12.1f}")
        chis.append(row["chi2_per_n"])
      print(f"{operator_name(*operator):>30}{''.join(cells)}   "
            f"{chis[0]:.2f} -> {chis[-1]:.2f}")

    # What the nuisance parameter actually did: a fit that pins Req at a prior edge is not
    # measuring Req, it is being pushed there by a misfit the material could not absorb.
    print("\n    fitted Req scale (and whether it sat on the prior edge):")
    for width in WIDTHS[1:]:
      scales = []
      for operator in operators:
        row = table[(dataset, operator, width)]
        if "failed" in row: continue
        edge = "*" if "req_scale" in row["pinned"] else " "
        scales.append(f"{row['fitted']['req_scale']:.3f}{edge}")
      print(f"      +/-{width:.0%}: {'  '.join(scales)}")

  # The claim under test, stated so it can fail: does any ordering that the pinned fit calls
  # decisive reverse, or stop being decisive, once Req floats?
  print("\n\n  orderings that do not survive the nuisance parameter:\n")
  survived = broken = 0
  for dataset in records.DATASETS:
    for i, a in enumerate(operators):
      for b in operators[i + 1:]:
        rows = {w: (table[(dataset, a, w)], table[(dataset, b, w)]) for w in WIDTHS}
        if any("failed" in x or "failed" in y for x, y in rows.values()): continue
        if any((dataset, o, w) in suspect for o in (a, b) for w in WIDTHS): continue
        gaps = {w: x["log_evidence"] - y["log_evidence"] for w, (x, y) in rows.items()}
        if abs(gaps[None]) < DECISIVE: continue
        floated = [gaps[w] for w in WIDTHS[1:]]
        if all(np.sign(g) == np.sign(gaps[None]) and abs(g) > DECISIVE for g in floated):
          survived += 1
          continue
        broken += 1
        print(f"    {dataset}  {operator_name(*a)} vs {operator_name(*b)}: "
              f"{gaps[None]:+.1f} pinned -> " + ", ".join(f"{g:+.1f}" for g in floated))
  print(f"\n  {survived} orderings survive at every width, {broken} do not.")
  print("  An ordering that does not survive was a statement about the initial radius.")

  records.HERE.joinpath("req_prior.json").write_text(json.dumps(
    {f"{d}|{operator_name(*o)}|{w}": v for (d, o, w), v in table.items()}, indent=1))


if __name__ == "__main__":
  main()
