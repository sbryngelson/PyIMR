r"""The rejection, on a second material at six times the sample size.

`lackoffit.py` rejects the qSLS on three gelatin records holding $18$, $14$ and $7$ events.
That is the first link in the chain \cref{sec:discrepancy} hangs everything else from, and it
is the link a referee will press on: three records of one material, one of them with seven
repeats, is a case study.

The same release carries $249$ PAAm events on the same rig. If the rejection reproduces there,
the claim stops being about gelatin and becomes about the METHOD, which is the claim worth
making. If it does not reproduce, the failure is gelatin's, and that has to be said before
publication rather than after.

Nothing about the statistic changes. Pure error is the between-event spread, which costs no
model; lack of fit is what the qSLS at its own optimum cannot follow; the ratio needs no
distributional assumption and the p-value remains unusable for the same reason (the residual
is correlated across time). Only the records are new, and `records.PAAM` records how they were
cut out of two pooled files.
"""

import json

import numpy as np

import records
from lackoffit import PARAMETER_SHARE, _job

ORDER = (*records.DATASETS, *records.PAAM)


def main():
  with records.pool(len(ORDER)) as pool:
    table = dict(pool.map(_job, ORDER))

  print("lack of fit against pure error, qSLS at its own optimum, both materials\n")
  print(f"{'record':>16} {'events':>7} {'chi2/N':>8} {'MS_pure':>11} {'MS_lack':>11} "
        f"{'F':>9} {'F corrected':>12}")
  summary = {}
  for dataset in ORDER:
    row = table[dataset]
    if "failed" in row:
      print(f"{dataset:>16}   {row['failed'][:50]}")
      continue
    _, mean, spread, _, _ = records.load(dataset)
    count = records.trial_count(dataset)
    model = np.asarray(row["model"], dtype=float)
    samples, parameters = mean.size, 4

    pure = float((count - 1) * np.sum(spread**2))
    lack = float(count * np.sum((mean - model) ** 2))
    pure_df, lack_df = samples * (count - 1), samples - parameters
    mean_pure, mean_lack = pure / pure_df, lack / lack_df
    ratio = mean_lack / mean_pure
    corrected = ratio / (1.0 - PARAMETER_SHARE)
    print(f"{dataset:>16} {count:7d} {row['chi2_per_n']:8.3f} {mean_pure:11.3e} "
          f"{mean_lack:11.3e} {ratio:9.1f} {corrected:12.1f}")
    summary[dataset] = dict(events=count, chi2_per_n=row["chi2_per_n"], mean_pure=mean_pure,
                            mean_lack=mean_lack, f_ratio=ratio, f_corrected=corrected,
                            pure_df=pure_df, lack_df=lack_df, fitted=row["fitted"],
                            bounds_hit=row["pinned"])

  paam = [v["f_ratio"] for k, v in summary.items() if k in records.PAAM]
  gel = [v["f_ratio"] for k, v in summary.items() if k in records.DATASETS]
  print("\n  ---- what it says ----\n")
  if paam and gel:
    print(f"  gelatin F {min(gel):.0f} to {max(gel):.0f} on {sum(records.trial_count(k) for k in records.DATASETS)} events, "
          f"PAAm F {min(paam):.0f} to {max(paam):.0f} on {sum(records.trial_count(k) for k in records.PAAM)} events.")
  print("  The correction divides out the 39.3% of trial variance that `trial_variation.py`")
  print("  measures as bubble-to-bubble parameter spread rather than measurement error. It was")
  print("  measured on gelatin, so on PAAm it is carried rather than established.")
  print("  PARAMETER_SHARE is a gelatin number applied to both; the uncorrected column is the")
  print("  one that assumes nothing.")
  records.HERE.joinpath("paam_lackoffit.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
  main()
