r"""The screening cliff, and how to design across it instead of guessing where it is.

\Cref{sec:screening} is the sharpest unresolved thing in this document. Which model questions
are open is \emph{undetermined by the likelihood}: deflating every margin by $N/N_{\rm eff}$
takes the constitutive axis from one live candidate to two, the operator axis from two--five to
five--six, and the thermal axis from one to three. The section's own conclusion is that this
promotes the correlated likelihood from a caveat to a precondition, because ``a batch aimed at a
live pair may be aimed at a pair whose liveness is an artifact.''

THE FRAGILITY IS THE THRESHOLD, not the deflation. Elsewhere the inflation cancels, because
every model is inflated alike and only the ranking is read. \Cref{eq:screen} does not read a
ranking --- it compares a margin to a constant, and a factor of twenty moves margins across a
constant wholesale. Reporting the answer at both ends of the factor, which is what that section
does, is honest and leaves the design with two incompatible instructions.

\Cref{sec:noisedesign} already solved this shape of problem once, and the finding there is the
recipe: averaging over a family of noise models produced the WORST worst case of the five
batches compared, while maximising against the hardest member won on every count. When one
member of a family is uniformly hardest, robust design is design against the hardest member.

Here the family is one-dimensional --- the deflation $N/N_{\rm eff} \in [1, 25]$ --- and the
hardest member is the largest one, because it revives the most rivals and spreads the weight
$\sqrt{w_m}$ of \eqref{eq:augmented} over the most coordinates. So the maximin batch is the one
optimised at full deflation, and what this measures is the PRICE of buying it: what a batch
designed against the worst case gives up at the other end of the range, against what a batch
designed at face value gives up if the deflation is real.

If the price is small the cliff simply goes away, and \cref{sec:screening} stops being a
precondition for the design work.
"""

import json

import numpy as np

import records
from screen import DECISIVE, INFLATION, _constitutive, _operators, _thermal

FACTORS = (1.0, 2.0, 5.0, 10.0, INFLATION)
AXES = {"constitutive": _constitutive, "operator": _operators, "thermal": _thermal}


def weights(evidence, factor):
  """`screen_models` at a stated deflation: margins divided, then the usual posterior.

  Deflating the margins rather than the evidences is the whole content of the correction --- an
  evidence has an arbitrary additive constant and only differences carry information, so
  dividing the raw values would rescale that constant too.
  """
  from pyimr.discriminate import screen_models

  if not evidence: return {}
  names = list(evidence)
  values = np.array([evidence[k] for k in names], dtype=float)
  deflated = values.max() + (values - values.max()) / factor
  screened = screen_models(deflated, decisive=DECISIVE)
  return {n: float(w) for n, w in zip(names, screened.weights, strict=True) if w > 0.0}


def realised(weight, truth):
  """How much of the information a `weight` batch collects is about rivals live under `truth`.

  A batch spends `sqrt(w_m)` of its Jacobian on each rival, so the share of its effort that
  lands on questions genuinely open is `sum_m w_designed(m) * 1[w_truth(m) > 0]`, and the share
  of the open questions it funds at all is the complementary sum. Both matter: the first is
  wasted effort, the second is a question nobody asked about.
  """
  live = {m for m, w in truth.items() if w > 0.0}
  spent = sum(w for m, w in weight.items() if m in live)
  covered = sum(truth[m] for m in truth if m in weight and weight[m] > 0.0)
  return spent, covered


def main():
  print(f"\n  deflation factors swept: {FACTORS}\n")
  summary = {}
  for axis, loader in AXES.items():
    print(f"\n=== {axis} ===\n")
    print(f"  {'record':13s} " + " ".join(f"{'x' + f'{f:g}':>9s}" for f in FACTORS))
    per_record = {}
    for dataset in records.DATASETS:
      evidence = loader(dataset)
      if not evidence:
        print(f"  {dataset:13s} (no stored evidence)")
        continue
      per_factor = {f: weights(evidence, f) for f in FACTORS}
      per_record[dataset] = per_factor
      print(f"  {dataset:13s} " + " ".join(f"{len(per_factor[f]):9d}" for f in FACTORS))

    if not per_record: continue
    print(f"\n  {'record':13s} {'designed at':>12s} {'truth':>8s} {'effort on live':>15s} "
          f"{'live questions funded':>22s}")
    axis_summary = {}
    for dataset, per_factor in per_record.items():
      rows = {}
      for designed in (1.0, INFLATION):
        for truth in (1.0, INFLATION):
          spent, covered = realised(per_factor[designed], per_factor[truth])
          rows[f"{designed:g}|{truth:g}"] = {"effort_on_live": spent, "coverage": covered}
          print(f"  {dataset:13s} {designed:12g} {truth:8g} {spent:15.3f} {covered:22.3f}")
      axis_summary[dataset] = rows
    summary[axis] = axis_summary

  print("\n  reading it: the diagonal entries are what each batch gets if it guessed right.")
  print("  The off-diagonal ones are the regret, and the asymmetry between them is the")
  print("  argument -- a batch designed at face value against a deflated truth funds only the")
  print("  rivals it already believed decided were not, while the reverse merely dilutes.")

  worst = {}
  for axis, rows in summary.items():
    for dataset, cells in rows.items():
      for designed in ("1", f"{INFLATION:g}"):
        pair = [cells[f"{designed}|{t}"]["coverage"] for t in ("1", f"{INFLATION:g}")]
        worst.setdefault(axis, {})[designed] = min(worst.get(axis, {}).get(designed, 1.0),
                                                   min(pair))
  print("\n  worst-case coverage over the two truths\n")
  print(f"  {'axis':14s} {'designed at face value':>24s} {'designed at full deflation':>28s}")
  for axis, cells in worst.items():
    print(f"  {axis:14s} {cells['1']:24.3f} {cells[f'{INFLATION:g}']:28.3f}")
  summary["worst_case_coverage"] = worst

  json.dump(summary, open(records.HERE / "screen_maximin.json", "w"), indent=1)


if __name__ == "__main__":
  main()
