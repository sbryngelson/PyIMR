r"""Which traces do the $437$ radii belong to? PAAm, as a superset, and not row by row.

`measured_scatter.py` uses `PA0503_radius.csv` for a scatter LAW, which needs only the
distribution. `paam_afterbounce.py` needs more: it converts normalised bounce periods to
absolute frequencies using a single mean $R_{\max}$ of \SI{359}{\micro\metre} for all three
PAAm trace files, and the per-event regression that would settle the apparatus question needs
each trace paired with its own radius. Both rest on a correspondence nobody has checked.

THE HANDLE IS THAT EVERY NORMALISED TRACE CARRIES ITS OWN $R_{\rm eq}/R_{\max}$. A ringdown
settles toward the equilibrium radius, so the late trace estimates the same ratio the radius
file states outright. Calibrated against gelatin, whose stretch is known, the tail MEDIAN is
the best of four estimators at $+1.8$ to \SI{+6.0}{\percent}; a tail mean is corrupted by
events that never settle, and midpoint or minimum estimators are worse by $11$ to
\SI{43}{\percent}.

WHICH SETTLES THE MATERIAL AND NOT THE PAIRING. The radius file gives
$R_{\rm eq}/R_{\max} = 0.1255 \pm 0.0085$. Gelatin traces give $0.148$ for a known $0.141$, so
the estimator's bias is about \SI{+4}{\percent}; the PAAm traces give $0.132$ to $0.135$,
which corrects to about $0.127$ and matches. The file is PAAm, as its name says, and the
released PAAm traces are consistent with being drawn from it.

THE PAIRING IS NOT RECOVERABLE AND THAT IS THE ANSWER. There are $437$ radii against $249$
released PAAm events, so it is a superset, and no arrangement recovers the order. The file is
not sorted by either column, has no block structure in chunks of $40$, and a sliding-window
correlation of every offset against each trace file peaks at $|r| = 0.41$ to $0.44$ over
roughly $350$ offsets tried, with the best offset for one file ANTI-correlated. That is the
profile of noise. It could not have worked anyway: the estimator's own bias varies by
\SI{2}{\percent} between datasets against a population spread of \SI{6.8}{\percent}.

SO THE ASK CHANGES SHAPE. The apparatus test does not need new radii, and it does not need new
experiments. It needs the index mapping between these $437$ rows and the released trace
columns, which whoever produced the files has and which no amount of analysis here
reconstructs.
"""

import json
import pathlib

import numpy as np

import records

DATA = pathlib.Path.home() / "fastscratch/imr-data-tempsweeps/data"
RADIUS = "PA0503_radius.csv"
TRACES = {"gelatin 15C": "Ga_t15_exp_data.csv", "gelatin 23C": "Ga_t23_exp_data.csv",
          "gelatin 33C": "Ga_t33_exp_data.csv", "PAAm PA05": "PA05_exp_data.csv",
          "PAAm PA05 temp": "PA05_temp_exp_data.csv",
          "PAAm PA05003 temp": "PA05003_temp_exp_data.csv"}
GELATIN = ("gelatin 15C", "gelatin 23C", "gelatin 33C")


def per_event_ratio(fname):
  """`R_eq/R_max` per event from the late trace. The tail MEDIAN, per the calibration."""
  table = np.loadtxt(DATA / fname, delimiter=",", ndmin=2)
  trials = table[:, 1:].T
  keep = ~(trials > 1.05).any(axis=0) & (trials.std(axis=0, ddof=1) > 0.0)
  trials = trials[:, keep]
  return np.array([float(np.median(row[-40:])) for row in trials])


def main():
  radii = np.loadtxt(DATA / RADIUS, delimiter=",", ndmin=2)
  ref = radii[:, 1] / radii[:, 0]
  print(f"  {RADIUS}: {len(radii)} rows, R_eq/R_max {ref.mean():.4f} +- {ref.std(ddof=1):.4f}, "
        f"cv {ref.std(ddof=1)/ref.mean():.3f}")
  print(f"  sorted by R_max {bool(np.all(np.diff(radii[:, 0]) >= 0))}, "
        f"by R_eq {bool(np.all(np.diff(radii[:, 1]) >= 0))}\n")

  # the estimator's bias, from gelatin where the stretch is known
  bias = []
  for label in GELATIN:
    key = "gelatin_" + label.split()[1].replace("C", "C")
    _, _, _, _, stretch = records.load(key)
    got = float(np.median(per_event_ratio(TRACES[label])))
    bias.append(got / (1.0 / stretch))
  factor = float(np.mean(bias))
  print(f"  estimator bias calibrated on gelatin: x{factor:.3f} "
        f"(spread {100*(max(bias)-min(bias)):.1f} percentage points)\n")

  print(f"  {'dataset':>18s} {'events':>7s} {'raw':>8s} {'corrected':>10s} {'vs radius file':>15s}")
  summary = {"radius_file": {"n": len(radii), "ratio_mean": float(ref.mean()),
                             "ratio_sd": float(ref.std(ddof=1))},
             "bias_factor": factor, "datasets": {}}
  for label, fname in TRACES.items():
    v = per_event_ratio(fname)
    raw, corr = float(np.median(v)), float(np.median(v)) / factor
    z = (corr - ref.mean()) / ref.std(ddof=1)
    summary["datasets"][label] = {"events": len(v), "raw": raw, "corrected": corr,
                                  "z_vs_radius_file": float(z)}
    print(f"  {label:>18s} {len(v):7d} {raw:8.4f} {corr:10.4f} {z:14.2f} sd")

  print("\n  ---- can any ordering be recovered? ----\n")
  for label in ("PAAm PA05", "PAAm PA05 temp", "PAAm PA05003 temp"):
    x = per_event_ratio(TRACES[label])
    n = len(x)
    best = max(((abs(float(np.corrcoef(x, ref[o:o + n])[0, 1])), o)
                for o in range(len(ref) - n + 1)), default=(0.0, 0))
    allr = [abs(float(np.corrcoef(x, ref[o:o + n])[0, 1])) for o in range(len(ref) - n + 1)]
    summary["datasets"][label]["best_abs_r"] = best[0]
    summary["datasets"][label]["best_offset"] = best[1]
    print(f"  {label:>18s}: best |r| {best[0]:.3f} at offset {best[1]}, "
          f"median over {len(allr)} offsets {np.median(allr):.3f}")

  print("\n  ---- what it says ----\n")
  paam = sum(summary['datasets'][k]['events'] for k in TRACES if k.startswith('PAAm'))
  print(f"  {len(radii)} radii against {paam} released PAAm events: a superset, not a pairing.")
  print("  The material is settled and the ordering is not. The apparatus test therefore needs")
  print("  the index mapping from whoever produced these files, not new radii and not a new")
  print("  experiment. Until then a per-event PAAm frequency cannot be formed, and the mean")
  print(f"  R_max of {radii[:, 0].mean():.0f} um that paam_afterbounce.py applies to all three")
  print("  PAAm files is an assumption carried by every absolute frequency it reports.")
  json.dump(summary, open(records.HERE / "radius_provenance.json", "w"), indent=1)


if __name__ == "__main__":
  main()
