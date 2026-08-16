r"""Where does the afterbounce misfit sit, once the material has been given its best shot?

`shape_error.py` fits the qSLS directly to the afterbounce ratio sequence over a box spanning
six decades and finds the sequence unreachable on the two datasets with the most events:
$\chi^2$ per bounce falls from $36.75$ to $6.52$ at \SI{15}{\celsius} and from $11.62$ to
$5.33$ at \SI{23}{\celsius}, and stops there. The fitter is not the limitation, because the
same fitter recovers a model-generated sequence to $\chi^2 = 0.0000$.

THE FAILURE HAS A SHAPE AND IT IS THE SAME SHAPE EVERY TIME. The model's ratio sequence is
monotone increasing in all six cases -- three datasets, at the trace fit and at the best fit
-- because damping in a spherical viscoelastic model weakens as the bounces slow, and nothing
in the form allows otherwise. Every measured sequence is non-monotone. At the best achievable
fit the residual collapses onto a SINGLE bounce, at $-5.0$, $-4.4$ and $-1.4$ standard errors
of the mean, always negative: one bounce loses more amplitude than any monotone damping
allows.

AND IT IS NOT THE SAME BOUNCE. It is bounce $4$, $3$ and $3$, which is the interesting part,
because those sit at $13.94$, $13.84$ and \SI{13.49}{\micro\second} -- a common FREQUENCY near
\SI{72}{\kilo\hertz} rather than a common index. This module exists to check that against the
two things that would explain it away, since a fixed frequency, a fixed fraction of the
bubble's own timescale, and a fixed number of samples per period are easy to confuse.

BOTH CONFOUNDS PREDICT THE OPPOSITE SIGN. $R_{\max}$ grows \SI{12.6}{\percent} across these
three records and the sampling interval grows \SI{12.6}{\percent} with it, so an anomaly
pinned to the bubble's own clock, or to the grid, would have a period $12.6$ percent LONGER at
\SI{33}{\celsius} than at \SI{15}{\celsius}. The measured period is $3.2$ percent SHORTER. And
because the later periods cluster, the coincidence is priced by enumeration rather than
asserted: of the $125$ ways to pick one bounce from each record, only $2$ cluster this tightly.
"""

import itertools
import json

import numpy as np

import records

SPAN = 5.0
HERE = records.HERE


def _periods_us(dataset, normalised):
  times, _, _, _, _ = records.load(dataset)
  window = float(times[-1] - times[0])
  return [p / SPAN * window * 1e6 for p in normalised]


def main():
  shape = json.load(open(HERE / "shape_error.json"))
  freq = json.load(open(HERE / "frequency_space.json"))

  print("  Residual at the BEST achievable material, in standard errors of the mean.")
  print("  A monotone model against a non-monotone record.\n")

  worst, summary = {}, {}
  for dataset in records.DATASETS:
    v = shape.get(dataset)
    if v is None: continue
    n = v["n_bounces"]
    se = np.array(freq[dataset]["measured"]["ratio_spread"][:n]) / np.sqrt(v["events"])
    meas, best = np.array(v["measured"]), np.array(v["model_at_best"])
    resid = (meas - best) / se
    k = int(np.argmax(np.abs(resid)))
    worst[dataset] = k
    mono = {"measured": bool(np.all(np.diff(meas) > 0)),
            "model": bool(np.all(np.diff(best) > 0))}
    summary[dataset] = {"resid_se": resid.tolist(), "worst_bounce": k + 1,
                        "worst_resid_se": float(resid[k]), "monotone": mono,
                        "events": v["events"], "chi2_best": v["chi2_best"]}
    print(f"  {dataset:>13s} ({v['events']:2d} events, chi2 {v['chi2_best']:5.2f})")
    print(f"    {'measured':>10s} " + " ".join(f"{x:7.3f}" for x in meas))
    print(f"    {'best fit':>10s} " + " ".join(f"{x:7.3f}" for x in best))
    print(f"    {'resid/SE':>10s} " + " ".join(f"{x:7.1f}" for x in resid))
    print(f"    monotone: model {mono['model']}, measured {mono['measured']}; "
          f"worst at bounce {k+1} ({resid[k]:+.1f} SE)\n")

  print("  ---- is the worst bounce at a common index or a common frequency? ----\n")
  periods, rows = {}, {}
  for dataset in records.DATASETS:
    if dataset not in worst: continue
    periods[dataset] = _periods_us(dataset, freq[dataset]["measured"]["period"])
    times, _, _, maximum, _ = records.load(dataset)
    dt = float(times[-1] - times[0]) / (len(times) - 1) * 1e6
    T = periods[dataset][worst[dataset]]
    rows[dataset] = {"period_us": T, "khz": 1e3 / T, "r_max_um": maximum * 1e6,
                     "dt_us": dt, "samples_per_period": T / dt}
    print(f"  {dataset:>13s}: bounce {worst[dataset]+1}, period {T:6.2f} us "
          f"({1e3/T:5.1f} kHz), R_max {maximum*1e6:5.1f} um, dt {dt:.4f} us, "
          f"{T/dt:5.2f} samples/period")

  keys = list(rows)
  if len(keys) == 3:
    def growth(field):
      a, b = rows[keys[0]][field], rows[keys[-1]][field]
      return 100.0 * (b - a) / a
    print(f"\n  across the three records: R_max {growth('r_max_um'):+.1f}%, "
          f"sampling dt {growth('dt_us'):+.1f}%, anomalous period {growth('period_us'):+.1f}%")
    print("  both confounds predict the anomalous period GROWING with those; it shrinks.")

    obs = [rows[d]["period_us"] for d in keys]
    spread = (max(obs) - min(obs)) / float(np.mean(obs))
    hits = tot = 0
    for combo in itertools.product(*(periods[d] for d in keys)):
      tot += 1
      if (max(combo) - min(combo)) / float(np.mean(combo)) <= spread: hits += 1
    print(f"\n  the trio spans {100*spread:.1f}%; of {tot} ways to pick one bounce per record,"
          f" {hits} are at least this tight -> p = {hits/tot:.3f}")
    summary["frequency"] = {"rows": rows, "relative_spread": float(spread),
                            "p_by_enumeration": hits / tot, "combinations": tot}

  print("\n  A spherical viscoelastic model damps monotonically by construction, so a single")
  print("  anomalous bounce is outside what ANY material in that form can produce. That the")
  print("  anomaly sits at a fixed frequency rather than a fixed bounce, and moves against")
  print("  both the bubble's own clock and the sampling grid, is what a resonance would do.")
  print("  With three records and p = 0.016 it is a lead, not a result.")
  json.dump(summary, open(HERE / "shape_anomaly.json", "w"), indent=1)


if __name__ == "__main__":
  main()
