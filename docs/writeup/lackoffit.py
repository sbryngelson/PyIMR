"""The classical test for the claim this document argues by other means.

`selection.tex` spends several pages establishing that the qSLS residual is model-form error
rather than measurement noise, using lag-one autocorrelation and whitening by a hierarchical
trial covariance. There is a standard test for exactly that question and it needs exactly what
these records already provide: replicates.

Each record is $J$ repeated bubbles observed at $k$ times. The trial spread at each time is
*pure error* -- how much the apparatus disagrees with itself -- and it costs no model. The
residual of the mean trace splits into that plus what the model cannot follow:

    SS_pure  = (J-1) sum_i s_i^2                     df = k(J-1)
    SS_lack  = J sum_i (ybar_i - yhat_i)^2           df = k - p
    F        = (SS_lack/df_lack) / (SS_pure/df_pure)

Two things make this worth running even though the writeup reached the same conclusion.

It gets the factor of J right by construction. The fit compares the MEAN of J trials against
`s_i`, the spread BETWEEN trials, when the standard error of the mean is smaller by sqrt(J).
`selection.tex` notes this and carries it as a correction; here the `J` in `SS_lack` is simply
part of the statistic.

And it is conservative in a direction we can quantify. `trial_variation.py` finds 39.3% of the
trial variance lies in the span of the parameter sensitivities -- each bubble has its own
parameters, so the spread is not measurement error alone. Pure error is therefore inflated,
the denominator is too large, and `F` understates the model's inadequacy. The corrected row
below divides that share out.

The F distribution assumes independent errors across time and ours are correlated at 0.918, so
the p-value is not usable. The ratio is: it says how many times larger the systematic
deviation is than the scatter of the apparatus, and no distributional assumption enters that.
"""

import json

import numpy as np

import records

# wider than `PARAMETER_BOUNDS`, matching `dynamics.py`: with the package's own box three of
# six fits sat exactly on a bound, so the reported optimum was a bound rather than a fit
WIDE = {"g": (1e0, 1e6), "mu": (1e-5, 1e1), "lambda1": (1e-9, 1e-2), "alpha": (1e-4, 1e3)}
# what `trial_variation.py` measures: the share of trial variance lying in the span of
# dR/dtheta, which is bubble-to-bubble parameter variation rather than measurement error
PARAMETER_SHARE = 0.393


def _job(dataset):
  from pyimr.selection import STANDARD_MODELS, evaluate_at

  times, mean, spread, maximum, stretch = records.load(dataset)
  solve = records.solver(times, maximum, stretch, max_steps=600_000)
  candidate = STANDARD_MODELS["qSLS"]
  try:
    scored = records.score(candidate, solve, mean, spread, bounds=WIDE, starts=10, evaluations=300)
  except Exception as error:                          # noqa: BLE001
    return dataset, {"failed": f"{type(error).__name__}: {error}"}
  fitted = evaluate_at(candidate, solve, scored["fitted"])[0]
  return dataset, {"model": np.asarray(fitted, dtype=float).tolist(), **scored}


def main():
  jobs = list(records.DATASETS)
  with records.pool(len(jobs)) as pool:
    table = dict(pool.map(_job, jobs))

  print("lack of fit against pure error, qSLS at its own optimum\n")
  print(f"{'record':>14} {'chi2/N':>8} {'MS_pure':>11} {'MS_lack':>11} {'F':>9} "
        f"{'F corrected':>12}")
  summary = {}
  for dataset in records.DATASETS:
    row = table[dataset]
    if "failed" in row:
      print(f"{dataset:>14}   {row['failed'][:50]}")
      continue
    times, mean, spread, _, _ = records.load(dataset)
    trials = json.loads(records.HERE.joinpath("results.json").read_text())[dataset]["trials"]
    model = np.asarray(row["model"], dtype=float)
    samples, parameters = mean.size, 4

    pure = float((trials - 1) * np.sum(spread**2))
    lack = float(trials * np.sum((mean - model) ** 2))
    pure_df, lack_df = samples * (trials - 1), samples - parameters
    mean_pure, mean_lack = pure / pure_df, lack / lack_df
    ratio = mean_lack / mean_pure
    # only the measurement share of the trial spread is error the model should not have to
    # follow; the rest is real bubble-to-bubble variation and belongs in the numerator's world
    corrected = ratio / (1.0 - PARAMETER_SHARE)
    print(f"{dataset:>14} {row['chi2_per_n']:8.3f} {mean_pure:11.3e} {mean_lack:11.3e} "
          f"{ratio:9.1f} {corrected:12.1f}")
    summary[dataset] = dict(chi2_per_n=row["chi2_per_n"], mean_pure=mean_pure,
                            mean_lack=mean_lack, f_ratio=ratio, f_corrected=corrected,
                            pure_df=pure_df, lack_df=lack_df, trials=trials)

  if summary:
    example = next(iter(summary.values()))
    print(f"\n  degrees of freedom {example['lack_df']} against {example['pure_df']}, so the "
          f"5% critical value is near 1.2.")
    print("  The systematic deviation is larger than the apparatus's own scatter by the factor")
    print("  in the last two columns. That is the same conclusion the lag-one argument reaches,")
    print("  by a route that shares none of its machinery.")
    print(f"\n  F is close to {example['trials']} x chi2/N by construction: the fit compares the")
    print("  MEAN of the trials against the spread BETWEEN them, and this statistic puts the")
    print("  missing sqrt(J) back. The correction divides out the 39.3% of trial variance that")
    print("  `trial_variation.py` shows is bubble-to-bubble parameter spread, not measurement.")
    print("\n  The p-value is NOT usable: F assumes independent errors across time and the")
    print("  residual is correlated at 0.918. The ratio needs no such assumption.")
  records.HERE.joinpath("lackoffit.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
  main()
