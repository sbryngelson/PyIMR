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
# R_max per record, from the paper's dataset table (see `examples/measured_selection.py`).
# They differ -- 277, 298, 312 um -- and reusing the 15 C value for the others would put a
# geometry error into every fit it touched.
DATASETS = {"gelatin_15C": 277e-6, "gelatin_23C": 298e-6, "gelatin_33C": 312e-6}
STARTS, EVALUATIONS = 6, 200
# the published qSLS fit is g = 204.3, alpha = 5.301, so g/alpha = 38.5; two decades around it
RATIOS = (3.85, 38.5, 385.0)
# Below this, two operators are not ordered by the record -- they are indistinguishable. A
# test that only asks "did the sign hold" calls that a contradiction, which is how a record
# with no discriminating power gets to veto a result from one that has it. On the 33 C record
# all six operators land within 0.5 nats of each other, so every ordering there is noise.
DECISIVE = 1.0
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

  dataset, name, ratio = argument
  record = json.loads((HERE / "results.json").read_text())[dataset]
  times, mean, spread = (np.array(record[k]) for k in ("times_s", "mean", "spread"))
  keep = spread > 0
  times, mean, spread = times[keep], mean[keep], spread[keep]
  radial = DYNAMICS_MODELS[name]

  def solve(material):
    maximum = DATASETS[dataset]
    config = pyimr.SimulationConfig(maximum, maximum / record["stretch"], material, radial=radial,
                                    rtol=1e-8, atol=1e-10, max_steps=400_000)
    trace = np.asarray(pyimr.simulate(times, config).radius_ratio, dtype=float)
    return trace, trace

  candidate = _candidate(ratio)
  try:
    fit = fit_candidate(candidate, solve, mean, spread, bounds=BOX, starts=STARTS,
                        max_evaluations=EVALUATIONS)
  except ValueError as error:
    return dict(dataset=dataset, name=name, ratio=ratio, failed=str(error))

  scored = []
  for point in fit.modes:
    try: scored.append(candidate_log_evidence(candidate, solve, mean, spread, point, bounds=BOX))
    except ValueError: continue
  values = physical_from_unit(candidate.axes, fit.unit, BOX)
  residual = (solve(candidate.build(dict(zip(candidate.axes, values, strict=True))))[0] - mean) / spread
  return dict(dataset=dataset, name=name, ratio=ratio, chi2_per_n=fit.chi_squared,
              log_evidence=float(logsumexp(scored)) if scored else float("nan"),
              lag_one=float(check_residuals(np.asarray(residual, float)).lag_one),
              pinned=[k for k, v in zip(candidate.axes, values, strict=True)
                      if min(abs(np.log(v / BOX[k][0])), abs(np.log(v / BOX[k][1]))) < 1e-6],
              fitted={k: float(v) for k, v in zip(candidate.axes, values, strict=True)})


def main():
  from pyimr.parallel import worker_pool
  from pyimr.selection import DYNAMICS_MODELS

  names = list(DYNAMICS_MODELS)
  jobs = [(d, n, r) for d in DATASETS for r in RATIOS for n in names]
  with worker_pool(6) as pool:
    results = list(pool.map(_job, jobs))
  table = {(r["dataset"], r["name"], r["ratio"]): r for r in results}

  def robust_pairs(dataset):
    """Orderings that hold at every ratio: the ones not set by what the data cannot see."""
    holds = {}
    for i, a in enumerate(names):
      for b in names[i + 1:]:
        gaps = [table[(dataset, a, r)].get("log_evidence") for r in RATIOS], \
               [table[(dataset, b, r)].get("log_evidence") for r in RATIOS]
        if any(x is None or not np.isfinite(x) for x in gaps[0] + gaps[1]): continue
        diff = [x - y for x, y in zip(*gaps, strict=True)]
        if all(d > DECISIVE for d in diff): holds[(a, b)] = min(diff)
        elif all(d < -DECISIVE for d in diff): holds[(b, a)] = min(-d for d in diff)
        elif max(abs(d) for d in diff) < DECISIVE: holds[(a, b)] = holds[(b, a)] = None  # a tie
    return holds

  print("the operator question in identified coordinates, on every record\n")
  per = {d: robust_pairs(d) for d in DATASETS}
  for dataset in DATASETS:
    chi = [table[(dataset, n, RATIOS[0])].get("chi2_per_n") for n in names]
    best = min((c for c in chi if c is not None), default=float("nan"))
    print(f"  {dataset}: {len(per[dataset])} of 15 pairwise orderings robust to the ratio, "
          f"best chi2/N {best:.2f}")

  # A conclusion is supported if some record orders it decisively and NO record orders it
  # the other way decisively. Records that merely cannot tell abstain rather than veto.
  decisive = {d: {k: v for k, v in per[d].items() if v is not None} for d in DATASETS}
  everywhere = {k for d in DATASETS for k in decisive[d]
                if not any((k[1], k[0]) in decisive[e] for e in DATASETS)}
  print(f"\n  per record, orderings decided at all (margin > {DECISIVE} nat): "
        + ", ".join(f"{d.replace('gelatin_','')}:{len(decisive[d])//1}" for d in DATASETS))
  print(f"  supported overall -- decided somewhere, contradicted nowhere: {len(everywhere)}")
  for a, b in sorted(everywhere, key=lambda ab: -max(decisive[d].get(ab, 0) for d in DATASETS)):
    votes = {d: decisive[d][(a, b)] for d in DATASETS if (a, b) in decisive[d]}
    quiet = [d for d in DATASETS if (a, b) not in decisive[d]]
    print(f"    {a:>22} > {b:<22} " + ", ".join(f"{d.replace('gelatin_','')}:+{v:.1f}" for d, v in votes.items())
          + (f"   (no power: {', '.join(x.replace('gelatin_','') for x in quiet)})" if quiet else ""))

  # "supported" and "replicated" are different claims and the difference is the whole point
  # of running three records: an ordering decided by ONE record with the others abstaining is
  # not confirmed by them, it is merely untested by them.
  print("\n  against the `radial=2` pressure form every candidate in this package assumes:")
  for challenger in ("keller-miksis-tait", "keller-miksis-mie"):
    key = (challenger, "keller-miksis")
    votes = [d for d in DATASETS if key in decisive[d]]
    verdict = ("REPLICATED on " + str(len(votes)) + " records" if len(votes) > 1 else
               "supported by one record, untested by the others" if len(votes) == 1 else
               "not decided anywhere")
    print(f"    {challenger:>22}: {verdict}")
  key = ("keller-miksis-tait", "keller-miksis")
  for dataset in DATASETS:
    margin, against = decisive[dataset].get(key), decisive[dataset].get((key[1], key[0]))
    print(f"    {dataset}: " + (f"+{margin:.2f} nats at worst" if margin else
                                 f"CONTRADICTED by {against:.2f}" if against else
                                 "no power to decide it"))
  (HERE / "identified.json").write_text(json.dumps(
    {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in table.items()}, indent=1))


if __name__ == "__main__":
  main()
