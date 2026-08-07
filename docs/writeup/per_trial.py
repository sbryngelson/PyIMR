"""Is the correlated residual a property of the model, or of averaging the trials?

The fits are against the MEAN of 18 trials, and those trials do not collapse at the same
time: their minima spread across a third of the record, so the mean is shallower at the
collapse than any trial by about 1.6 times the median noise. A sharp trajectory fitted to a
smeared curve would be wrong exactly where the residual is worst, smoothly, and no
constitutive change would fix it.

This fits the same model to each trial separately and asks whether the correlation survives.

Caveat kept in view: every trial is fitted at the record's `R_max` and stretch, because
per-trial values are not available. That contaminates chi^2/N and not lag-one, so the
comparison this test supports is lag-one against lag-one.
"""

import json

import numpy as np

import records

DATASET = "gelatin_15C"
DATA = records.Path.home() / "fastscratch/papers/paper_imr_windowing/data/Ga_t15_exp_data.csv"


def _trials(samples):
  table = np.loadtxt(DATA, delimiter=",", ndmin=2)
  raw = table[:, 1:].T
  usable = ~(raw > 1.05).any(axis=0) & (raw.std(axis=0, ddof=1) > 0.0)
  return raw[:, usable][:, :samples]


def _job(index):
  """`index < 0` is the mean trace; otherwise that trial."""
  from pyimr.selection import STANDARD_MODELS

  times, mean, spread, maximum, stretch = records.load(DATASET)
  observed = mean if index < 0 else _trials(mean.size)[index]
  solve = records.solver(times, maximum, stretch)
  return index, records.score(STANDARD_MODELS["qSLS"], solve, observed, spread)


def main():
  from pyimr.parallel import worker_pool

  times, mean, *_ = records.load(DATASET)
  count = _trials(mean.size).shape[0]
  print(f"{DATASET}: the same qSLS fitted to the mean, then to each of {count} trials\n")

  with worker_pool(6) as pool:
    got = dict(pool.map(_job, [-1, *range(count)]))
  averaged, singles = got[-1], [got[i] for i in range(count)]
  lag = np.array([r["lag_one"] for r in singles])
  chi = np.array([r["chi2_per_n"] for r in singles])

  print(f"{'target':>12} {'chi2/N':>9} {'lag-1':>8} {'N_eff':>8}")
  print(f"{'mean trace':>12} {averaged['chi2_per_n']:9.3f} {averaged['lag_one']:8.3f} {averaged['n_eff']:8.1f}")
  print(f"{'trials: med':>12} {np.median(chi):9.3f} {np.median(lag):8.3f} "
        f"{np.median([r['n_eff'] for r in singles]):8.1f}")
  print(f"{'range':>12} {chi.min():5.3f}-{chi.max():<5.3f} {lag.min():8.3f}-{lag.max():.3f}")
  print(f"\n  lag-one falls by {averaged['lag_one'] - float(np.median(lag)):+.3f} from the mean to a "
        f"typical single trial.\n  A large fall would mean the correlation was an artefact of "
        f"averaging misaligned\n  collapses; a small one rules that explanation out.")
  records.HERE.joinpath("pertrial.json").write_text(
    json.dumps({"mean": averaged, "trials": singles}, indent=1))


if __name__ == "__main__":
  main()
