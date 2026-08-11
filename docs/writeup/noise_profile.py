r"""What does a constant noise scale cost, when the measured one spans three decades?

Fitting already uses the per-sample spread. The design work does not: `design_operator.py`,
`design_three.py`, `identify.py` and both figure scripts take `RELATIVE_NOISE = 0.018`, a
single number, and every Fisher matrix, expected information gain and recommended batch in
\cref{sec:design} is built on it. The records disagree by two to three orders of magnitude
within a single trace.

The number that matters is not the ratio but where the information ends up. Fisher information
weights sample $i$ by $1/\sigma_i^2$, so a scale that varies by $10^3$ varies the weights by
$10^6$, and a design that assumes flat weights is optimising a quantity the data will not
deliver. The participation ratio of those weights says how many samples are actually carrying
the answer, against the $201$ the constant assumes.

Read with \S"the dominant trial mode": the scatter is concentrated in the REBOUND, at $97$ to
$98\%$ of the leading mode on two records, while the identifiable discrepancy sits $65$ to
$69\%$ in the FIRST COLLAPSE.

THE ESTIMATOR IS THE WHOLE DIFFICULTY, and taking the raw per-sample spread is a trap. A
standard deviation from $J$ trials carries relative error $1/\sqrt{2(J-1)}$ -- $17\%$ at
eighteen trials and $29\%$ at seven -- so $1/\sigma^2$ carries twice that, and weighting by it
rewards exactly the samples whose spread was underestimated by luck. Measured raw, the minimum
spread sits $30$ to $90$ times below the median and the weights collapse onto a handful of
points: a participation ratio of $2.3$ to $4.7$ samples out of two hundred, which is an
artifact of the estimator and not a property of the apparatus. Under a median filter it moves
to $7$--$21$, and under a tenth-percentile floor to $45$--$111$. Nothing that depends on those
weights should be reported from the raw estimate, so the profile here is smoothed and the
sensitivity to the window is printed beside it.
"""

import json

import numpy as np

import records
from identified import BOX, candidate_at_ratio

RATIO = 38.5
CONSTANT = 0.018                 # what the design scripts assume, from the median spread
SMOOTH = 11                      # samples pooled per variance estimate; see the docstring
STARTS, EVALUATIONS = 16, 600


def one(dataset):
  """Sensitivities at the record's own fit, and the information under both noise models."""
  import pyimr
  from pyimr.selection import fit_candidate, physical_from_unit

  times, mean, spread, maximum, stretch = records.load(dataset)
  candidate = candidate_at_ratio(RATIO)
  solve = records.solver(times, maximum, stretch)
  fit = fit_candidate(candidate, solve, mean, spread, bounds=BOX, starts=STARTS,
                      max_evaluations=EVALUATIONS)
  axes = tuple(candidate.axes)
  fitted = dict(zip(axes, (float(v) for v in physical_from_unit(axes, fit.unit, BOX)),
                    strict=True))
  point = np.array([fitted[k] for k in axes])

  def trace(scale):
    moved = dict(zip(axes, (float(v) for v in point * scale), strict=True))
    config = pyimr.SimulationConfig(maximum, maximum / stretch, candidate.build(moved),
                                    dynamics="keller-miksis", rtol=1e-9, atol=1e-11,
                                    max_steps=600_000)
    return np.asarray(pyimr.simulate(times, config).radius_ratio, dtype=float)

  step = 1e-4
  # unwhitened, so the two noise models can be applied to the same Jacobian
  jacobian = np.column_stack([
    (trace(np.where(np.arange(len(axes)) == k, 1 + step, 1.0))
     - trace(np.where(np.arange(len(axes)) == k, 1 - step, 1.0))) / (2 * step)
    for k in range(len(axes))])

  model = trace(np.ones(len(axes)))
  bottom = int(np.argmin(mean))
  half = max(1, times.size // 20)
  windows = {"pre": slice(0, max(0, bottom - half)),
             "collapse": slice(max(0, bottom - half), bottom + half),
             "rebound": slice(bottom + half, times.size)}

  # pooled over a window, so each variance rests on ~SMOOTH*J deviations rather than J. A
  # median is used rather than a mean because the quantity being suppressed is an outlier.
  from scipy.ndimage import median_filter
  smoothed = median_filter(spread, SMOOTH, mode="nearest")
  weights = 1.0 / smoothed**2
  share = weights / weights.sum()
  measured = jacobian.T @ (jacobian / smoothed[:, None] ** 2)
  constant = jacobian.T @ jacobian / CONSTANT**2

  def participation(scale):
    w = (1.0 / scale**2); w = w / w.sum()
    return float(1.0 / np.sum(w**2))
  return dataset, {
    "axes": list(axes), "fitted": fitted,
    "sigma": {k: float(np.median(spread[w])) for k, w in windows.items()},
    "participation_raw": participation(spread),
    "participation_by_window": {int(n): participation(median_filter(spread, n, mode="nearest"))
                                for n in (5, 11, 21, 41)},
    "weight_share": {k: float(share[w].sum()) for k, w in windows.items()},
    # how many samples actually carry the information, against the 201 a constant assumes
    "participation": float(1.0 / np.sum(share**2)),
    "samples": int(times.size),
    "logdet_measured": float(np.linalg.slogdet(measured)[1]),
    "logdet_constant": float(np.linalg.slogdet(constant)[1]),
    "sigma_measured": {k: float(v) for k, v in zip(
      axes, np.sqrt(np.diag(np.linalg.inv(measured))), strict=True)},
    "sigma_constant": {k: float(v) for k, v in zip(
      axes, np.sqrt(np.diag(np.linalg.inv(constant))), strict=True)},
    "chi2_per_n": float(fit.chi_squared),
    "model_min": float(model.min()),
  }


def main():
  with records.pool(len(records.DATASETS)) as pool:
    table = dict(pool.map(one, list(records.DATASETS)))

  print(f"\n  measured noise scale by phase, against the constant {CONSTANT} the design uses\n")
  print(f"  {'record':13s} {'sigma pre':>10s} {'sigma coll':>11s} {'sigma reb':>10s} "
        f"{'weight in collapse':>19s}")
  for name, got in table.items():
    print(f"  {name:13s} {got['sigma']['pre']:10.5f} {got['sigma']['collapse']:11.5f} "
          f"{got['sigma']['rebound']:10.5f} {got['weight_share']['collapse']:18.1%}")

  print("\n  how many samples carry the Fisher weight, and how much that depends on the")
  print("  estimator -- the raw column is an artifact of a variance from few trials\n")
  print(f"  {'record':13s} {'samples':>8s} {'raw':>8s} "
        + " ".join(f"{'median-' + str(n):>10s}" for n in (5, 11, 21, 41)))
  for name, got in table.items():
    cells = " ".join(f"{got['participation_by_window'][str(n)] if isinstance(next(iter(got['participation_by_window'])), str) else got['participation_by_window'][n]:10.1f}"
                     for n in (5, 11, 21, 41))
    print(f"  {name:13s} {got['samples']:8d} {got['participation_raw']:8.1f} " + cells)

  print("\n  what the constant does to the information, at the same Jacobian\n")
  print(f"  {'record':13s} {'log det F':>22s}   per-parameter sigma, measured / constant")
  for name, got in table.items():
    ratio = " ".join(f"{k} x{got['sigma_constant'][k] / got['sigma_measured'][k]:.2f}"
                     for k in got["axes"])
    print(f"  {name:13s} {got['logdet_measured']:10.2f} vs {got['logdet_constant']:8.2f}   {ratio}")

  print("\n  A constant scale is not merely wrong in size -- it is wrong in WHERE it puts the")
  print("  weight. Fisher information weights a sample by 1/sigma^2, so three decades of")
  print("  spread is six decades of weight, and the design optimises a precision the")
  print("  apparatus will not deliver.")

  json.dump(table, open(records.HERE / "noise_profile.json", "w"), indent=1)


if __name__ == "__main__":
  main()
