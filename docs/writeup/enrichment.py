"""Does a richer relaxation actually help on the 15 C record?

The one-mode residual is correlated at lag one and is not explained by variation between
bubbles, which points at model-form error. Two enrichments answer that differently --
`qSLS2` adds a second relaxation time, `qSLSthin` lets one time move with the shear rate --
and both reduce exactly to `qSLS`, so a comparison between them is meaningful rather than a
comparison of two unrelated fits.

Each is fitted by multistart in the prior's unit coordinates and scored by Laplace expansion
about the fit, with the Occam factor capped at the prior: uncapped, a model is PAID for a
parameter its data cannot resolve, which is exactly the regime an enrichment lives in.
Evidence is summed over the distinct modes the multistart found, since the expansion is
about one mode and the integral is over all of them.

Reported beside the evidence: chi^2/N, because model selection only means something where
some candidate actually fits, and the share of evaluations that landed where the material
would not integrate, because a fit that spent its budget against a wall is not a fit.
"""

import os, json

for _n in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(_n, "1")

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATASET, R_MAX = "gelatin_15C", 277e-6
MODELS = ("qSLS", "qSLS2", "qSLSthin")
STARTS, EVALUATIONS = 8, 240


def _job(name):
  import pyimr
  from pyimr.selection import (EXTENDED_MODELS, STANDARD_MODELS, candidate_log_evidence,
                               fit_candidate, physical_from_unit)
  from scipy.special import logsumexp

  record = json.loads((HERE / "results.json").read_text())[DATASET]
  times, mean, spread = (np.array(record[k]) for k in ("times_s", "mean", "spread"))
  keep = spread > 0
  times, mean, spread = times[keep], mean[keep], spread[keep]

  def solve(material):
    config = pyimr.SimulationConfig(R_MAX, R_MAX / record["stretch"], material, radial=2,
                                    rtol=1e-8, atol=1e-10, max_steps=300_000)
    trace = np.asarray(pyimr.simulate(times, config).radius_ratio, dtype=float)
    return trace, trace

  candidate = (STANDARD_MODELS | EXTENDED_MODELS)[name]
  fit = fit_candidate(candidate, solve, mean, spread, starts=STARTS, max_evaluations=EVALUATIONS)
  # the expansion is about ONE mode; the evidence is over all of them
  per_mode = []
  for point in fit.modes:
    try:
      per_mode.append(candidate_log_evidence(candidate, solve, mean, spread, point))
    except ValueError:
      continue                                        # a mode the expansion cannot use
  evidence = float(logsumexp(per_mode)) if per_mode else float("nan")

  # The diagnostic the enrichment was FOR. chi^2/N and the evidence both presume independent
  # residuals; the one-mode fit's are correlated at lag one, which is what said the defect
  # was model form rather than noise. An enrichment that improves the evidence while leaving
  # the correlation untouched has not addressed what it was reached for -- and worse, the
  # margin it won is measured under an independence assumption the residual still violates.
  from pyimr.noise import check_residuals

  residual = (solve(candidate.build(dict(zip(candidate.axes, physical_from_unit(candidate.axes, fit.unit),
                                             strict=True))))[0] - mean) / spread
  check = check_residuals(np.asarray(residual, dtype=float))
  fitted = dict(zip(candidate.axes, (float(v) for v in physical_from_unit(candidate.axes, fit.unit)), strict=True))
  return dict(name=name, dimension=candidate.dimension, chi2_per_n=fit.chi_squared, fitted=fitted,
              log_evidence=evidence, modes=len(fit.modes), scored=len(per_mode),
              lag_one=float(check.lag_one), n_eff=float(check.effective_samples), samples=int(mean.size),
              failure_fraction=fit.failure_fraction, converged=fit.converged, starts=fit.starts)


def main():
  from pyimr.parallel import worker_pool

  with worker_pool(3) as pool:
    results = list(pool.map(_job, list(MODELS)))
  order = {r["name"]: r for r in results}
  baseline = order["qSLS"]["log_evidence"]

  print(f"{DATASET}: enrichments of the one-mode law, {STARTS} starts, capped Occam factor\n")
  print(f"{'model':>10} {'p':>3} {'chi2/N':>9} {'log Z':>12} {'vs qSLS':>10} "
        f"{'lag-1':>8} {'N_eff':>8} {'modes':>6} {'failed':>8}")
  for name in MODELS:
    r = order[name]
    print(f"{name:>10} {r['dimension']:3d} {r['chi2_per_n']:9.3f} {r['log_evidence']:12.2f} "
          f"{r['log_evidence'] - baseline:+10.2f} {r['lag_one']:8.3f} {r['n_eff']:8.1f} "
          f"{r['modes']:6d} {r['failure_fraction']:7.1%}")

  print("\n  where the extra parameters landed:")
  for name in MODELS[1:]:
    extra = {k: v for k, v in order[name]["fitted"].items() if k not in order["qSLS"]["fitted"]}
    print(f"    {name:>9}: " + ", ".join(f"{k} = {v:.4g}" for k, v in extra.items()))

  print(f"\n  N_eff is out of {order['qSLS']['samples']} samples. A margin in `vs qSLS` is only")
  print("  worth its face value if lag-1 has come down with it: both the evidence and chi2/N")
  print("  presume independent residuals, and a correlated residual leaves the likelihood")
  print("  overstating its own information by roughly N/N_eff.")
  (HERE / "enrichment.json").write_text(json.dumps(order, indent=1))


if __name__ == "__main__":
  main()
