"""Does a richer relaxation actually help on the 15 C record?

The one-mode residual is correlated at lag one and concentrated in the first collapse. Two
enrichments answer that differently -- `qSLS2` adds a second relaxation time, `qSLSthin`
lets one time move with the shear rate -- and both reduce exactly to `qSLS`, so a comparison
between them is meaningful rather than a comparison of two unrelated fits.

The Occam factor is capped at the prior throughout: uncapped, a model is PAID for a
parameter its data cannot resolve, which is exactly the regime an enrichment lives in.

Reported beside the evidence: chi^2/N, because selection only means something where some
candidate fits, and lag-one, because that correlation is the whole reason these exist.
"""

import json

import records

DATASET = "gelatin_15C"
MODELS = ("qSLS", "qSLS2", "qSLSthin")


def _job(name):
  from pyimr.selection import EXTENDED_MODELS, STANDARD_MODELS

  times, mean, spread, maximum, stretch = records.load(DATASET)
  candidate = (STANDARD_MODELS | EXTENDED_MODELS)[name]
  solve = records.solver(times, maximum, stretch)
  return name, records.score(candidate, solve, mean, spread, starts=8, evaluations=240)


def main():
  from pyimr.parallel import worker_pool

  with worker_pool(3) as pool:
    order = dict(pool.map(_job, list(MODELS)))
  baseline = order["qSLS"]["log_evidence"]

  print(f"{DATASET}: enrichments of the one-mode law, capped Occam factor\n")
  print(f"{'model':>10} {'chi2/N':>9} {'log Z':>12} {'vs qSLS':>10} {'lag-1':>8} {'N_eff':>8}")
  for name in MODELS:
    r = order[name]
    print(f"{name:>10} {r['chi2_per_n']:9.3f} {r['log_evidence']:12.2f} "
          f"{r['log_evidence'] - baseline:+10.2f} {r['lag_one']:8.3f} {r['n_eff']:8.1f}")

  print("\n  where the extra parameters landed:")
  for name in MODELS[1:]:
    extra = {k: v for k, v in order[name]["fitted"].items() if k not in order["qSLS"]["fitted"]}
    print(f"    {name:>9}: " + ", ".join(f"{k} = {v:.4g}" for k, v in extra.items()))

  print("\n  A margin is worth its face value only if lag-1 came down with it: both log Z and")
  print("  chi2/N presume independent residuals, and a correlated one leaves the likelihood")
  print("  overstating its information by roughly N/N_eff.")
  records.HERE.joinpath("enrichment.json").write_text(json.dumps(order, indent=1))


if __name__ == "__main__":
  main()
