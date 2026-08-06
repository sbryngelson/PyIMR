"""The operator comparison, asked in coordinates the record can answer.

The likelihood depends on the product `g*alpha` alone (derived in the writeup, confirmed
numerically by an SVD at the fit: the sloppiest direction is `g^+1 alpha^-1` and its
posterior is 2.07 times wider than its prior). Fitting `g` and `alpha` separately therefore
slides along a ridge until it hits the prior box, and the operator ranking follows the box
rather than the data --- moving the bounds swung it by 50 nats and reversed the winner.

So this fixes the unidentified ratio at a stated value, fits the identified product, and
asks the operator question again in three parameters instead of four.

The point is not the ranking at one ratio; it is whether the ranking SURVIVES the ratio.
`g/alpha` is exactly what the data cannot see, so a conclusion that changes with it is a
conclusion about the prior. Each operator is therefore scored at several ratios spanning two
decades around the published fit, and what is reported is whether the ordering holds.
"""

import os, json

for _n in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(_n, "1")

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATASET, R_MAX = "gelatin_15C", 277e-6
STARTS, EVALUATIONS = 6, 200
# the published qSLS fit is g = 204.3, alpha = 5.301, so g/alpha = 38.5; two decades around it
RATIOS = (3.85, 38.5, 385.0)
BOX = {"mu": (1e-5, 1e1), "galpha": (1e0, 1e7), "lambda1": (1e-9, 1e-2)}


def _candidate(ratio):
  """qSLS with `g/alpha` fixed, so the free axes are the ones the data determines."""
  import pyimr
  from pyimr.selection import CandidateModel

  def build(t):
    product = t["galpha"]
    return pyimr.QuadraticZener(float(np.sqrt(product * ratio)), t["mu"], t["lambda1"], 0.0,
                                float(np.sqrt(product / ratio)))

  return CandidateModel(f"qSLS|g/alpha={ratio:g}", build, ("mu", "galpha", "lambda1"))


def _job(argument):
  import pyimr
  from pyimr.noise import check_residuals
  from pyimr.selection import (DYNAMICS_MODELS, candidate_log_evidence, fit_candidate,
                               physical_from_unit)
  from scipy.special import logsumexp

  name, ratio = argument
  record = json.loads((HERE / "results.json").read_text())[DATASET]
  times, mean, spread = (np.array(record[k]) for k in ("times_s", "mean", "spread"))
  keep = spread > 0
  times, mean, spread = times[keep], mean[keep], spread[keep]
  radial = DYNAMICS_MODELS[name]

  def solve(material):
    config = pyimr.SimulationConfig(R_MAX, R_MAX / record["stretch"], material, radial=radial,
                                    rtol=1e-8, atol=1e-10, max_steps=400_000)
    trace = np.asarray(pyimr.simulate(times, config).radius_ratio, dtype=float)
    return trace, trace

  candidate = _candidate(ratio)
  try:
    fit = fit_candidate(candidate, solve, mean, spread, bounds=BOX, starts=STARTS,
                        max_evaluations=EVALUATIONS)
  except ValueError as error:
    return dict(name=name, ratio=ratio, failed=str(error))

  scored = []
  for point in fit.modes:
    try: scored.append(candidate_log_evidence(candidate, solve, mean, spread, point, bounds=BOX))
    except ValueError: continue
  values = physical_from_unit(candidate.axes, fit.unit, BOX)
  residual = (solve(candidate.build(dict(zip(candidate.axes, values, strict=True))))[0] - mean) / spread
  return dict(name=name, ratio=ratio, chi2_per_n=fit.chi_squared,
              log_evidence=float(logsumexp(scored)) if scored else float("nan"),
              lag_one=float(check_residuals(np.asarray(residual, float)).lag_one),
              pinned=[k for k, v in zip(candidate.axes, values, strict=True)
                      if min(abs(np.log(v / BOX[k][0])), abs(np.log(v / BOX[k][1]))) < 1e-6],
              fitted={k: float(v) for k, v in zip(candidate.axes, values, strict=True)})


def main():
  from pyimr.parallel import worker_pool
  from pyimr.selection import DYNAMICS_MODELS

  names = list(DYNAMICS_MODELS)
  jobs = [(n, r) for r in RATIOS for n in names]
  with worker_pool(6) as pool:
    results = list(pool.map(_job, jobs))
  table = {(r["name"], r["ratio"]): r for r in results}

  print(f"{DATASET}: the operator question in identified coordinates "
        f"(g*alpha free, g/alpha fixed)\n")
  header = "".join(f"{f'g/a={r:g}':>22}" for r in RATIOS)
  print(f"{'operator':>22}{header}")
  print(f"{'':>22}" + "".join(f"{'log Z   (chi2/N)':>22}" for _ in RATIOS))
  orders = {}
  for ratio in RATIOS:
    usable = [(n, table[(n, ratio)]) for n in names if "failed" not in table[(n, ratio)]]
    orders[ratio] = [n for n, _ in sorted(usable, key=lambda kv: -kv[1]["log_evidence"])]
  for name in names:
    cells = ""
    for ratio in RATIOS:
      r = table[(name, ratio)]
      cells += f"{'   did not fit':>22}" if "failed" in r else \
               f"{r['log_evidence']:14.2f} ({r['chi2_per_n']:5.2f})"
    print(f"{name:>22}{cells}")

  print("\n  ranking at each ratio, best first:")
  for ratio in RATIOS:
    print(f"    g/alpha={ratio:<7g} " + " > ".join(orders[ratio]))
  stable = len({tuple(v) for v in orders.values()}) == 1
  print(f"\n  ordering identical across two decades of the unidentified ratio: {stable}")

  # "unstable" is too blunt to act on. A pairwise check says WHICH comparisons survive the
  # direction the data cannot see, and those are the ones worth reporting: a conclusion is
  # only about the data if it does not move when `g/alpha` does.
  print("\n  pairwise orderings that hold at EVERY ratio (these are about the data):")
  robust, fragile = [], []
  for i, a in enumerate(names):
    for b in names[i + 1:]:
      gaps = [table[(a, r)]["log_evidence"] - table[(b, r)]["log_evidence"]
              for r in RATIOS if "failed" not in table[(a, r)] and "failed" not in table[(b, r)]]
      if len(gaps) != len(RATIOS): continue
      if all(g > 0 for g in gaps): robust.append((a, b, min(abs(g) for g in gaps)))
      elif all(g < 0 for g in gaps): robust.append((b, a, min(abs(g) for g in gaps)))
      else: fragile.append((a, b))
  for a, b, margin in sorted(robust, key=lambda t: -t[2]):
    print(f"    {a:>22} > {b:<22} by at least {margin:7.2f} nats")
  print("\n  pairs whose order REVERSES with the ratio (not determined by this record):")
  for a, b in fragile: print(f"    {a} vs {b}")
  pins = {k: v["pinned"] for k, v in table.items() if v.get("pinned")}
  print(f"  fits pinned at a bound: {len(pins)} of {len(table)}")
  (HERE / "identified.json").write_text(json.dumps({f"{k[0]}@{k[1]}": v for k, v in table.items()}, indent=1))


if __name__ == "__main__":
  main()
