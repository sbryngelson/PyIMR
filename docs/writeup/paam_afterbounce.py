r"""The same rig, a different material, and four times the events: is the anomaly there too?

`apparatus_test.py` says the within-dataset test needs a per-event $R_{\max}$ and calls it
absent. It is not absent, it is in a directory that module did not look in, and the same
directory holds something that answers the question better: PAAm trace files with $52$, $117$
and $80$ events against gelatin's $18$, $14$ and $7$.

WHY THAT IS THE BETTER TEST. An apparatus resonance belongs to the rig, so it must appear
whatever is in the sample chamber. A bubble-intrinsic feature belongs to the bubble and moves
with the material. PAAm on the same instrument, at four to six times the sample size, is
therefore a sharper discriminator than a regression on gelatin radii would have been, and it
needs no model fit at all: the ratio sequence of a spherically symmetric viscoelastic model is
monotone increasing whatever its material, so departures can be read directly.

THE COMPARISON IS MODEL-FREE AND THE SAMPLE SIZE IS THE POINT. `shape_error.py` establishes
monotonicity as a property of the FORM rather than of any parameter choice, so a measured
sequence that rises throughout is consistent with the form and one that dips is not. Counting
dips and sizing them against the between-event scatter is then a complete test, and with $117$
events the scatter is small enough for that to mean something.

WHAT A NULL RESULT ON PAAm WOULD SAY. If PAAm passes through the same absolute frequency band
without a dip, no resonance of the apparatus lives there, and the gelatin signature is either
the material's or the small-sample artefact its $18$, $14$ and $7$ events permit. Both are
worth knowing and only one of them was on the table before.
"""

import json

import numpy as np

import records
from bounce_sweep import maxima

KEEP = 6
MAX_RATIO = 1.05
# t_c per micrometre of R_max, calibrated on gelatin 15C: window 137.454 us = 5 t_c at 277 um
TC_PER_UM = 137.454 / 5.0 / 277.0
FILES = {"gelatin 15C": ("Ga_t15_exp_data.csv", 277.0),
         "gelatin 23C": ("Ga_t23_exp_data.csv", 298.0),
         "gelatin 33C": ("Ga_t33_exp_data.csv", 312.0),
         "PAAm PA05": ("PA05_exp_data.csv", None),
         "PAAm PA05 temp": ("PA05_temp_exp_data.csv", None),
         "PAAm PA05003 temp": ("PA05003_temp_exp_data.csv", None)}
RADIUS = "PA0503_radius.csv"


def sequences(path):
  table = np.loadtxt(path, delimiter=",", ndmin=2)
  tau, trials = table[:, 0], table[:, 1:].T
  keep = ~(trials > MAX_RATIO).any(axis=0) & (trials.std(axis=0, ddof=1) > 0.0)
  tau, trials = tau[keep], trials[:, keep]
  ratios, periods = [], []
  for row in trials:
    _, amps, times = maxima(row, tau, order=2)
    if len(amps) < 4: continue
    ratios.append([amps[k + 1] / amps[k] for k in range(len(amps) - 1)])
    periods.append([times[k + 1] - times[k] for k in range(len(times) - 1)])
  n = min(KEEP, min(len(x) for x in ratios))
  r = np.array([x[:n] for x in ratios], dtype=float)
  p = np.array([x[:n] for x in periods], dtype=float)
  return len(ratios), r, p


def main():
  import pathlib
  data = pathlib.Path.home() / "fastscratch/imr-data-tempsweeps/data"
  radii = np.loadtxt(data / RADIUS, delimiter=",", ndmin=2)
  r_paam = float(np.mean(radii[:, 0]))
  print(f"  PAAm per-event radii: {len(radii)} events, R_max mean {r_paam:.1f} um, "
        f"cv {radii[:, 0].std(ddof=1)/r_paam:.3f}")
  print("  A spherical viscoelastic model has a MONOTONE ratio sequence whatever its")
  print("  material, so a dip is outside the form and needs no fit to detect.\n")

  summary = {}
  for label, (fname, r_max) in FILES.items():
    path = data / fname
    if not path.exists(): continue
    events, r, p = sequences(path)
    mean, se = r.mean(axis=0), r.std(axis=0, ddof=1) / np.sqrt(events)
    per = p.mean(axis=0)
    radius = r_max if r_max is not None else r_paam
    absolute = per * TC_PER_UM * radius            # microseconds
    dips = [k for k in range(1, len(mean)) if mean[k] < mean[k - 1]]
    worst = None
    if dips:
      k = min(dips, key=lambda j: (mean[j] - mean[j - 1]) / np.hypot(se[j], se[j - 1]))
      worst = {"bounce": k + 1, "drop": float(mean[k] - mean[k - 1]),
               "sigmas": float((mean[k] - mean[k - 1]) / np.hypot(se[k], se[k - 1])),
               "period_us": float(absolute[k]), "khz": float(1e3 / absolute[k])}
    summary[label] = {"events": events, "ratios": mean.tolist(), "se": se.tolist(),
                      "periods_us": absolute.tolist(), "r_max_um": radius,
                      "dips": [d + 1 for d in dips], "worst_dip": worst}
    print(f"  ==== {label}, {events} events, R_max {radius:.0f} um ====")
    print(f"  {'ratios':>10s} " + " ".join(f"{v:7.3f}" for v in mean))
    print(f"  {'+- se':>10s} " + " ".join(f"{v:7.3f}" for v in se))
    print(f"  {'period us':>10s} " + " ".join(f"{v:7.2f}" for v in absolute))
    if worst:
      print(f"  dips at bounce {summary[label]['dips']}; deepest {worst['drop']:+.3f} "
            f"({worst['sigmas']:.1f} sd) at {worst['period_us']:.2f} us "
            f"= {worst['khz']:.1f} kHz")
    else:
      print("  MONOTONE throughout: consistent with the spherical form")
    print()

  print("  ---- what it says ----\n")
  band = [13.49, 13.94]
  for label, v in summary.items():
    inband = [i for i, x in enumerate(v["periods_us"]) if band[0] <= x <= band[1]]
    tag = (f"bounce {[i+1 for i in inband]}" if inband else "no bounce")
    dip = "and dips there" if any(i + 1 in v["dips"] for i in inband) else "and does NOT dip there"
    print(f"  {label:>18s} ({v['events']:3d} events): {tag} in the 13.5-13.9 us band, {dip}")
  print("\n  An apparatus resonance belongs to the rig and must appear whatever is in the")
  print("  chamber. A material that passes through the same band without a dip, at four to")
  print("  six times the sample size, is the evidence against one.")
  json.dump(summary, open(records.HERE / "paam_afterbounce.json", "w"), indent=1)


if __name__ == "__main__":
  main()
