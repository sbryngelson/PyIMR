r"""Can the \SI{72}{\kilo\hertz} anomaly be attributed with the data in hand? Partly, and the
clean version needs one column that the normalisation removed.

`shape_modes.py` refutes every frequency belonging to the bubble and leaves two possibilities:
the near-constancy is the coincidence its $p = 0.016$ allows, or the frequency belongs to the
apparatus. The discriminator is $R_0$: everything intrinsic to the bubble moves with it and an
apparatus resonance does not. This asks what the released data can already say.

THE BETWEEN-DATASET TEST EXISTS AND IS CONFOUNDED. $R_0$ spans \SIrange{39.07}{45.68}{\micro\metre}
across the three datasets, a \SI{15.8}{\percent} variation, and the anomalous period moved
\SI{-3.2}{\percent} where every bubble-intrinsic scale predicts $-16$ to \SI{-24}{\percent}.
That is a real $R_0$ variation and the anomaly failed to track it. It is not decisive because
$R_0$ moves together with temperature and with the fitted modulus across these three datasets,
so a cancellation between them cannot be excluded from three points.

THE WITHIN-DATASET TEST WOULD NOT BE CONFOUNDED, AND CANNOT BE RUN. At fixed temperature the
events of one dataset differ only in their bubble, so regressing each event's anomalous period
against its own $R_{\max}$ separates the hypotheses cleanly with $18$, $14$ and $7$ points.
It needs a per-event $R_{\max}$, and that is precisely what the released form removes: every
trace is normalised to $R/R_{\max}$ with all events resampled onto one shared nondimensional
grid of $201$ points at a step of $0.025$. The per-event files that do exist carry $18$, $14$
and $7$ rows, matching the event counts, but hold fitted quantities rather than radii.

AND THE SCATTER CANNOT SUBSTITUTE FOR IT. If the shared grid carries each event's own clock,
a bubble-intrinsic period would show near-zero between-event scatter and an apparatus one
would scatter like $R_{\max}$. The measured period scatter at the anomalous bounce is $12.9$,
$15.3$ and \SI{19.6}{\percent}. The RATIO scatter at the same bounces is $10.3$, $15.0$ and
\SI{17.0}{\percent}, and a ratio carries no clock at all, so that is measurement scatter
alone. The clock-bearing quantity is no noisier than the clock-free one, which means the
period scatter is consistent with pure measurement error and carries no usable signal about
$R_{\max}$.

SO THE ASK IS A DATA ASK RATHER THAN AN EXPERIMENT. One column, $R_{\max}$ per event, makes
this decidable from measurements already taken. That is worth saying precisely, because the
alternative reading of `shape_modes.py` is that a new experiment is required, and it is not.
"""

import json

import numpy as np

import records

# 0-indexed anomalous bounce per dataset, from shape_anomaly.py
WORST = {"gelatin_15C": 3, "gelatin_23C": 2, "gelatin_33C": 2}


def main():
  freq = json.load(open(records.HERE / "frequency_space.json"))
  modes = json.load(open(records.HERE / "shape_modes.json"))

  print("  The discriminator is R0. Everything intrinsic to the bubble moves with it.\n")
  print("  ---- what the three datasets already say, confounded ----\n")
  r0 = [modes["rows"][d]["r0"] * 1e6 for d in records.DATASETS]
  anom = [modes["anomaly_khz"][d] for d in records.DATASETS]
  print("  R0 (um)          " + " ".join(f"{x:8.2f}" for x in r0)
        + f"   spread {100*(max(r0)-min(r0))/np.mean(r0):5.1f}%")
  print("  anomaly (kHz)    " + " ".join(f"{x:8.2f}" for x in anom)
        + f"   spread {100*(max(anom)-min(anom))/np.mean(anom):5.1f}%")
  print("  every bubble-intrinsic scale predicts the anomaly moving -16 to -24 percent.")
  print("  it moved -3.2. But R0, temperature and the modulus move together here, so three")
  print("  points cannot exclude a cancellation.\n")

  print("  ---- why the within-dataset test cannot substitute ----\n")
  print(f"  {'dataset':>13s} {'events':>7s} {'period scatter':>15s} {'ratio scatter':>14s}")
  rows = {}
  for dataset in records.DATASETS:
    m = freq[dataset]["measured"]
    k = WORST[dataset]
    per = 100 * m["period_spread"][k] / m["period"][k]
    rat = 100 * m["ratio_spread"][k] / m["ratio"][k]
    rows[dataset] = {"period_scatter_pct": per, "ratio_scatter_pct": rat,
                     "events": m["events"], "bounce": k + 1}
    print(f"  {dataset:>13s} {m['events']:7d} {per:14.1f}% {rat:13.1f}%")
  print("\n  A ratio carries no clock, so its scatter is measurement error alone. The")
  print("  clock-bearing period is no noisier than the clock-free ratio, so the period")
  print("  scatter is consistent with measurement error and says nothing about R_max.")

  print("\n  ---- what would settle it ----\n")
  print("  R_max per event, one column, for datasets already collected. At fixed temperature")
  print("  the events of one dataset differ only in their bubble, so regressing each event's")
  print("  anomalous period on its own R_max separates the hypotheses with 18, 14 and 7")
  print("  points and no new experiment. The released traces are normalised to R/R_max on a")
  print("  shared 201-point grid, which removes exactly that column.")
  json.dump({"r0_um": r0, "anomaly_khz": anom, "scatter": rows,
             "verdict": "confounded between datasets; within-dataset test needs per-event R_max"},
            open(records.HERE / "apparatus_test.json", "w"), indent=1)


if __name__ == "__main__":
  main()
