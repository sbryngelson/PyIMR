"""Is the correlated residual a property of the model, or of averaging the trials?

The one-mode fit's residual is correlated at lag one (0.918) and concentrated in the first
collapse. Two enrichments of the constitutive law left that untouched, so the defect is
probably not the material.

There is a candidate that is not constitutive at all. The fits are against the MEAN of 18
trials, and those trials do not collapse at the same time: their minima spread across a
third of the record, so the mean is shallower at the collapse than any trial by about 1.6
times the median noise. A sharp trajectory fitted to a smeared curve would be wrong exactly
where the residual is worst, smoothly (hence correlated), and no constitutive change would
fix it.

This fits the SAME model to each trial separately and asks whether the correlation survives.
If it falls, averaging was the problem and no new material model was ever going to help. If
it stays, the defect really is in the physics and this rules out the cheapest explanation.

Caveat kept in view: every trial is fitted at the same `R_MAX` and stretch as the mean is,
because per-trial values are not in the record. That is a source of per-trial error this
cannot separate, so the comparison to make is lag-one against lag-one, not fit against fit.
"""

import os, json

for _n in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(_n, "1")

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = Path.home() / "fastscratch/papers/paper_imr_windowing/data/Ga_t15_exp_data.csv"
DATASET, R_MAX = "gelatin_15C", 277e-6
STARTS, EVALUATIONS = 6, 200


def _record():
  record = json.loads((HERE / "results.json").read_text())[DATASET]
  times, mean, spread = (np.array(record[k]) for k in ("times_s", "mean", "spread"))
  keep = spread > 0
  table = np.loadtxt(DATA, delimiter=",", ndmin=2)
  raw = table[:, 1:].T                                  # (trials, samples)
  usable = ~(raw > 1.05).any(axis=0) & (raw.std(axis=0, ddof=1) > 0.0)
  return times[keep], mean[keep], spread[keep], raw[:, usable][:, : int(keep.sum())], record["stretch"]


def _job(index):
  """`index = -1` is the mean trace; otherwise trial `index`."""
  import pyimr
  from pyimr.noise import check_residuals
  from pyimr.selection import STANDARD_MODELS, fit_candidate, physical_from_unit

  times, mean, spread, trials, stretch = _record()
  observed = mean if index < 0 else trials[index]

  def solve(material):
    config = pyimr.SimulationConfig(R_MAX, R_MAX / stretch, material, radial=2,
                                    rtol=1e-8, atol=1e-10, max_steps=300_000)
    trace = np.asarray(pyimr.simulate(times, config).radius_ratio, dtype=float)
    return trace, trace

  candidate = STANDARD_MODELS["qSLS"]
  fit = fit_candidate(candidate, solve, observed, spread, starts=STARTS, max_evaluations=EVALUATIONS)
  values = physical_from_unit(candidate.axes, fit.unit)
  residual = (solve(candidate.build(dict(zip(candidate.axes, values, strict=True))))[0] - observed) / spread
  check = check_residuals(np.asarray(residual, dtype=float))
  return dict(index=index, chi2_per_n=fit.chi_squared, lag_one=float(check.lag_one),
              n_eff=float(check.effective_samples), failure_fraction=fit.failure_fraction)


def main():
  from pyimr.parallel import worker_pool

  times, mean, spread, trials, _ = _record()
  count = trials.shape[0]
  print(f"{DATASET}: the same qSLS fitted to the mean, then to each of {count} trials "
        f"({len(times)} samples)\n")

  with worker_pool(6) as pool:
    results = list(pool.map(_job, [-1, *range(count)]))
  averaged = next(r for r in results if r["index"] < 0)
  singles = sorted((r for r in results if r["index"] >= 0), key=lambda r: r["index"])

  lag = np.array([r["lag_one"] for r in singles])
  chi = np.array([r["chi2_per_n"] for r in singles])
  print(f"{'target':>12} {'chi2/N':>9} {'lag-1':>8} {'N_eff':>8}")
  print(f"{'mean trace':>12} {averaged['chi2_per_n']:9.3f} {averaged['lag_one']:8.3f} {averaged['n_eff']:8.1f}")
  print(f"{'trials: med':>12} {np.median(chi):9.3f} {np.median(lag):8.3f} "
        f"{np.median([r['n_eff'] for r in singles]):8.1f}")
  print(f"{'range':>12} {chi.min():5.3f}-{chi.max():<5.3f} {lag.min():8.3f}-{lag.max():.3f}")

  drop = averaged["lag_one"] - float(np.median(lag))
  print(f"\n  lag-one falls by {drop:+.3f} going from the mean to a typical single trial.")
  print("  A large fall means the correlation was an artefact of averaging misaligned")
  print("  collapses, and no constitutive enrichment could have removed it. A small one")
  print("  means the defect is in the physics and this rules the cheap explanation out.")
  (HERE / "pertrial.json").write_text(json.dumps({"mean": averaged, "trials": singles}, indent=1))


if __name__ == "__main__":
  main()
