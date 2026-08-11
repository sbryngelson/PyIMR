r"""A batch that does not depend on which noise model is right.

Three measurements leave the design chapter with three answers. Under the constant scale the
recommendation is \SI{514}{\micro\metre} at stretch $6.92$; under the measured profiles it is
\SI{416}{\micro\metre} at $10.85$, and \S`noise\_transfer` shows the profiles agree on that once
the estimator noise is filtered out. A paper has to publish one batch.

Choosing under a single profile inherits its assumptions. The alternative is to choose the batch
that does well under ALL of them, which is the same move \cref{sec:design} already makes across
QUESTIONS -- a portfolio, weighted -- applied to a second axis of uncertainty.

The construction keeps the certificate. Averaging gains across noise models cannot be done by
averaging criteria, because each model gives a different information matrix for the same
candidate. Stacking them block-diagonally does: with
$\tilde M_i = \operatorname{blockdiag}(M_i^{(1)},\dots,M_i^{(K)})$ the averaged matrix
$\sum_i \xi_i \tilde M_i$ is the block-diagonal of the per-model averages, so a criterion that
reads each block and returns $\sum_k w_k U_k$ is concave in $\xi$ with a block-diagonal gradient,
and `optimal_measure`'s equivalence theorem applies unchanged.

A weighted sum is chosen over a worst-case maximin deliberately. $\max_\xi \min_k U_k(\xi)$ is
also concave, but nonsmooth, and the certificate this package is built on is a statement about a
differentiable criterion. The weighted batch is then REPORTED against its worst case, which is
the quantity that matters, so nothing is hidden by the choice of objective.

AND THAT CHOICE LOSES, which is the result. The weighted batch has the WORST worst case of the
five compared -- below the batch chosen under the flat scale it was meant to improve on.
Maximising an average does not maximise a minimum, and hedging toward the mean can be dominated
on every component at once. The smooth objective bought a certificate and cost the property it
was built for.

What replaces it is simpler than a maximin solver. One profile binds for every batch, so the
worst case is always that profile's column, and the batch that maximises it is just the argmax
under that profile. When one member of the family is uniformly hardest, robust design collapses
to design against the hardest member -- no new machinery required.

Read the spread before reading the ranking, though. All five batches lie within $0.21$ nats of
$33.5$, six tenths of a percent, so on information alone the choice barely matters. Where they
differ is testability: the settings run from five to eight and $\mathrm{df}$ from $+1$ to $+4$.

The testability floor of #232 is imposed on top, because `noise_portfolio.py` shows the free
batch reaches $\mathrm{df} = 0$ under two of the three profiles and `noise_testable.py` prices
the repair at two tenths of a percent.
"""

import json

import numpy as np

import records
from noise_design import RELATIVE_NOISE, raw_information
from noise_portfolio import BUDGET, MATERIAL, OPERATOR, PARAMETERS, score
from noise_transfer import transfers

FLOOR_SETTINGS = PARAMETERS + 1


def _questions():
  from pyimr.gain import Question
  return [Question("material", MATERIAL), Question("operator", OPERATOR, weight=4.0)]


def blocked(criterion, blocks, width, weights):
  """A criterion over a block-diagonal stack: `sum_k w_k U_k`, gradient block-diagonal."""
  def evaluate(matrix):
    total, gradient = 0.0, np.zeros_like(matrix)
    for k in range(blocks):
      piece = slice(k * width, (k + 1) * width)
      value, slope = criterion(matrix[piece, piece])
      total += weights[k] * value
      gradient[piece, piece] = weights[k] * slope
    return total, gradient
  return evaluate


def main():
  from design_operator import PERFORMED, RADII, STRETCH
  from pyimr.gain import gain_criterion, lack_of_fit_degrees
  from pyimr.measure import apportion, constrained_measure
  from pyimr.noise import characteristic_time

  designs = [(float(r), float(s)) for r in RADII for s in STRETCH] + list(PERFORMED.values())
  with records.pool(len(designs)) as pool:
    got = list(pool.map(raw_information, designs))
  usable = [(d, payload) for d, payload in got if payload is not None]
  points = [d for d, _ in usable]
  print(f"\n  {len(usable)} of {len(designs)} candidates integrate")

  # the smoothed phase transfer, which noise_transfer.py shows is the one the records agree on
  smoothed = {name: transfers(name)["smoothed"] for name in records.DATASETS}
  names = list(smoothed)

  def fisher(columns, phase, scale):
    whitened = columns / scale[:, None]
    matrix = whitened.T @ whitened
    return 0.5 * (matrix + matrix.T)

  stacks = {name: np.array([fisher(c, p, np.interp(p, *smoothed[name]))
                            for _, (c, p) in usable]) for name in names}
  stacks["constant"] = np.array([fisher(c, p, np.full(c.shape[0], RELATIVE_NOISE))
                                 for _, (c, p) in usable])
  width = stacks[names[0]].shape[1]

  # equal weights: nothing measured says one record's profile is likelier than another's
  weights = np.full(len(names), 1.0 / len(names))
  blocks = np.zeros((len(usable), width * len(names), width * len(names)))
  for k, name in enumerate(names):
    piece = slice(k * width, (k + 1) * width)
    blocks[:, piece, piece] = stacks[name]

  questions = _questions()
  criterion = gain_criterion(questions)
  robust = constrained_measure(blocks, FLOOR_SETTINGS,
                               criterion=blocked(criterion, len(names), width, weights),
                               iterations=400_000)
  candidates = {"robust": apportion(robust.measure.weights, BUDGET)}
  for name in ("constant", *names):
    held = constrained_measure(stacks[name], FLOOR_SETTINGS, criterion=criterion,
                               iterations=400_000)
    candidates[name] = apportion(held.measure.weights, BUDGET)

  def objective(nats): return nats["material"] + 4.0 * nats["operator"]

  print("\n  each batch judged under every measured profile (portfolio objective, nats)\n")
  print(f"  {'batch chosen under':20s} " + " ".join(f"{n[-3:]:>9s}" for n in names)
        + f" {'WORST':>9s} {'settings':>9s} {'lack-df':>8s}")
  table = {}
  for label, exact in candidates.items():
    under = [objective(score(stacks[name], exact.counts, questions)) for name in names]
    settings, _, lack_df, _ = lack_of_fit_degrees(exact.counts, parameters=PARAMETERS)
    table[label] = {"under": dict(zip(names, under, strict=True)), "worst": float(min(under)),
                    "settings": settings, "lack_df": lack_df,
                    "table": [(points[i][0], points[i][1], int(n))
                              for i, n in enumerate(exact.counts) if n > 0]}
    print(f"  {label:20s} " + " ".join(f"{v:9.3f}" for v in under)
          + f" {min(under):9.3f} {settings:9d} {lack_df:8d}")

  best = max(table, key=lambda k: table[k]["worst"])
  margin = table[best]["worst"] - table["constant"]["worst"]
  print(f"\n  best worst-case: {best}, at {table[best]['worst']:.3f} nats"
        f" ({margin:+.3f} against the batch chosen under the constant)")
  if best != "robust":
    gap = table[best]["worst"] - table["robust"]["worst"]
    rank = sorted(table, key=lambda k: -table[k]["worst"]).index("robust") + 1
    print(f"  the WEIGHTED batch ranks {rank} of {len(table)} on worst case, {gap:.3f} nats"
          f" behind -- maximising an average does not maximise a minimum")
  binding = {label: min(row["under"], key=row["under"].get) for label, row in table.items()}
  if len(set(binding.values())) == 1:
    which = next(iter(set(binding.values())))
    print(f"  one profile binds for every batch ({which}), so the maximin answer is just the")
    print("  argmax under it -- robust design collapses to design against the hardest member")
  worsts = np.array([row["worst"] for row in table.values()])
  spread = float(np.ptp(worsts))          # ndarray.ptp() went away in numpy 2
  print(f"\n  spread of worst cases: {spread:.3f} nats on {worsts.mean():.1f}"
        f" ({spread / worsts.mean():.1%}) -- the information barely distinguishes them")
  print(f"  settings run {min(r['settings'] for r in table.values())} to "
        f"{max(r['settings'] for r in table.values())}, df "
        f"{min(r['lack_df'] for r in table.values())} to "
        f"{max(r['lack_df'] for r in table.values())}: testability is where they differ")

  print(f"\n  the batch to publish -- best worst case AND most testable ({best}):")
  for radius, stretch, runs in table[best]["table"]:
    print(f"    R_max {radius * 1e6:7.1f} um   stretch {stretch:5.2f}   {runs:2d} runs"
          f"   (t_c {characteristic_time(radius) * 1e6:.2f} us)")
  json.dump(table, open(records.HERE / "robust_batch.json", "w"), indent=1)


if __name__ == "__main__":
  main()
