r"""Does the ranking mean anything when every model in it is wrong?

\Cref{eq:lackoffit} rejects every candidate on every record. This document's defence is that ``a
ranking among wrong models is still a ranking'', and that is a claim rather than a result. Under
misspecification the evidence rewards whichever model best ABSORBS the discrepancy, and
\cref{sec:discrepancy} shows absorption is exactly the mechanism that biases the parameters. So
the ranking could be measuring absorption capacity and not physical closeness, and nothing in the
document distinguishes those.

It is directly testable with machinery already here, and the test is the one \cref{sec:survives}
runs for a different purpose. Generate from a truth OUTSIDE the catalogue --- thermal transport,
which \cref{sec:axes} sizes at $16$ noise units, and a second relaxation mode at $14.4$ --- then
rank the catalogue by evidence and, separately, measure which candidate is actually closest to
the truth. If the two orderings agree, the ranking survives misspecification of this size. If
they do not, every ranking in this document is a statement about absorption.

CLOSENESS HAS TO BE DEFINED, and the definition is the honest part. A candidate's distance to the
truth is the residual its OWN best fit leaves against the noiseless truth, in noise units --- not
its distance at some nominal parameters, which would measure the parameter choice instead. That
is the quantity a modeller cares about: how well can this model, tuned as well as it can be,
reproduce what is really happening.

THREE OUTCOMES ARE POSSIBLE and all are informative. The orderings agree, and the defence holds
at this size of misspecification. They disagree on the winner, and no ranking here is safe. Or
they agree on the winner and disagree below it, which is the outcome to expect if evidence is
dominated by fit at the top and by Occam factors further down --- and would license the ordinal
reading the document already restricts itself to.
"""

import json

import numpy as np

import records

DATASET = "gelatin_15C"
STARTS, EVALUATIONS = 12, 400
SEED = 0
# the catalogue: the four nested laws tab:fits ranks, in their natural parameters
BOXES = {
  "qSLS": {"g": (1e2, 1e5), "mu": (1e-3, 1e0), "lambda1": (1e-8, 1e-5), "alpha": (1e-1, 1e2)},
  "SLS": {"g": (1e2, 1e5), "mu": (1e-3, 1e0), "lambda1": (1e-8, 1e-5)},
  "qKV": {"g": (1e2, 1e5), "mu": (1e-3, 1e0), "alpha": (1e-1, 1e2)},
  "NHKV": {"g": (1e2, 1e5), "mu": (1e-3, 1e0)},
}
# the published qSLS fit, not a value invented here: an invented one did not integrate
# under the thermal truth at all, which would have made the test about the solver
TRUTH_MATERIAL = (204.3, 0.04651, 1.964e-7, 5.301)   # g, mu, lambda1, alpha


def _candidate(name):
  import pyimr
  from pyimr.selection import CandidateModel

  def qsls(t): return pyimr.QuadraticZener(t["g"], t["mu"], t["lambda1"], 0.0, t["alpha"])
  def sls(t): return pyimr.Zener(t["g"], t["mu"], t["lambda1"], 0.0)
  def qkv(t): return pyimr.QuadraticKelvinVoigt(t["g"], t["mu"], t["alpha"])
  def nhkv(t): return pyimr.NeoHookeanKelvinVoigt(t["g"], t["mu"])

  build = {"qSLS": qsls, "SLS": sls, "qKV": qkv, "NHKV": nhkv}[name]
  return CandidateModel(name, build, tuple(BOXES[name]))


def truth_trace(kind):
  """The noiseless trace of a truth the catalogue does not contain."""
  import pyimr

  times, mean, spread, maximum, stretch = records.load(DATASET)
  g, mu, lam, alpha = TRUTH_MATERIAL
  if kind == "thermal":
    material = pyimr.QuadraticZener(g, mu, lam, 0.0, alpha)
    extra = {"bubtherm": 1, "medtherm": 1}
  elif kind == "two-mode":
    material = pyimr.TwoModeQuadraticZener(g, mu, lam, 0.0, alpha, 15.7 * lam, 0.16)
    extra = {}
  else:
    raise ValueError(kind)
  config = pyimr.SimulationConfig(maximum, maximum / stretch, material,
                                  dynamics="keller-miksis", rtol=1e-8, atol=1e-10,
                                  max_steps=2_000_000, **extra)
  return np.asarray(pyimr.simulate(times, config).radius_ratio, dtype=float)


def one(job):
  """Evidence and closeness for one candidate against one truth."""
  from pyimr.selection import evaluate_at

  kind, name, clean = job
  times, mean, spread, maximum, stretch = records.load(DATASET)
  noisy = clean + np.random.default_rng(SEED).normal(0.0, spread)
  candidate = _candidate(name)
  solve = records.solver(times, maximum, stretch)
  try:
    got = records.score(candidate, solve, noisy, spread, bounds=BOXES[name], starts=STARTS,
                        evaluations=EVALUATIONS, trials=records.trial_count(DATASET))
  except ValueError as error:
    return job[:2], {"failed": str(error)}
  fitted = got["fitted"]
  model = np.asarray(evaluate_at(candidate, solve, fitted)[0], dtype=float)
  # closeness: the residual this candidate's own best fit leaves against the NOISELESS truth
  distance = float(np.sqrt(np.mean(((model - clean) / spread) ** 2)))
  return job[:2], {"log_evidence": got["log_evidence"], "chi2_per_n": got["chi2_per_n"],
                   "distance": distance, "fitted": fitted,
                   "dimension": len(BOXES[name])}


def main():
  truths = {kind: truth_trace(kind) for kind in ("thermal", "two-mode")}
  times, mean, spread, maximum, stretch = records.load(DATASET)
  for kind, clean in truths.items():
    size = float(np.sqrt(np.mean(((clean - mean) / spread) ** 2)))
    print(f"  truth '{kind}' sits {size:.2f} noise units from the measured mean")

  jobs = [(kind, name, truths[kind]) for kind in truths for name in BOXES]
  print(f"\n  {len(jobs)} fits ...", flush=True)
  with records.pool(len(jobs)) as pool:
    table = dict(pool.map(one, jobs))

  summary = {}
  for kind in truths:
    rows = {n: table[(kind, n)] for n in BOXES if "failed" not in table[(kind, n)]}
    if len(rows) < 2:
      print(f"\n  {kind}: too few candidates fitted")
      continue
    best = max(r["log_evidence"] for r in rows.values())
    print(f"\n=== truth: {kind} ===\n")
    print(f"  {'candidate':8s} {'p':>3s} {'log Z margin':>13s} {'evidence rank':>14s} "
          f"{'distance':>10s} {'closeness rank':>15s}")
    by_evidence = sorted(rows, key=lambda n: -rows[n]["log_evidence"])
    by_distance = sorted(rows, key=lambda n: rows[n]["distance"])
    entry = {}
    for name in by_evidence:
      row = rows[name]
      entry[name] = {"margin": row["log_evidence"] - best, "distance": row["distance"],
                     "evidence_rank": by_evidence.index(name) + 1,
                     "closeness_rank": by_distance.index(name) + 1,
                     "chi2_per_n": row["chi2_per_n"]}
      print(f"  {name:8s} {row['dimension']:3d} {row['log_evidence'] - best:13.2f} "
            f"{by_evidence.index(name) + 1:14d} {row['distance']:10.3f} "
            f"{by_distance.index(name) + 1:15d}")

    agree_winner = by_evidence[0] == by_distance[0]
    order = float(np.corrcoef([entry[n]["evidence_rank"] for n in rows],
                              [entry[n]["closeness_rank"] for n in rows])[0, 1])
    summary[kind] = {"rows": entry, "winner_agrees": bool(agree_winner),
                     "evidence_winner": by_evidence[0], "closest": by_distance[0],
                     "rank_correlation": order}
    print(f"\n  evidence picks {by_evidence[0]}, closest is {by_distance[0]}: "
          f"{'AGREE' if agree_winner else 'DISAGREE'}")
    print(f"  rank correlation over the whole ordering: {order:+.3f}"
          "   (+1 is the ranking meaning what it is read to mean)")

  print("\n  A ranking that recovers the closest member under a misspecification of this size")
  print("  is a ranking; one that does not is a measurement of which model absorbs best.")
  json.dump(summary, open(records.HERE / "ranking_validity.json", "w"), indent=1)


if __name__ == "__main__":
  main()
