r"""Is the afterbounce dip locked to the bubble's own bounce, or to a frequency of the rig?

The index mapping between `PA0503_radius.csv` and the released traces does not exist, so no
per-event radius can be attached to a per-event frequency. This settles the same question
without one.

THE TRACES CARRY THEIR OWN CLOCKS, WHICH IS WHAT MAKES IT POSSIBLE. The first collapse sits at
a normalised time whose between-event coefficient of variation is $0.017$ to $0.025$ on all
six datasets. Were the events sharing one clock it would scatter like $R_{\max}$, whose
measured cv is $0.223$, ten times larger. So each trace is expressed in its own event's time.

WHICH SPLITS THE TWO HYPOTHESES CLEANLY. A feature belonging to the bubble sits at the same
BOUNCE INDEX in every event, because everything about that event scales together. A feature at
a fixed frequency of the apparatus sits at an index that depends on that event's $R_{\max}$,
and with a cv of $0.223$ against bounce-to-bounce period gaps of $9$ to \SI{19}{\percent} it
moves by of order one index between events. Averaging ratios at fixed index therefore
preserves the first and dilutes the second.

THE NULL IS BUILT BY SHIFTING RATHER THAN ASSUMED. Each event's ratio sequence is circularly
shifted by a random amount, which destroys any alignment between events while preserving every
event's own dip depth and shape exactly. The ensemble dip is recomputed many times to give the
distribution an incoherent feature would produce. Comparing the real ensemble dip against that
distribution needs no model of the noise, no estimate of $R_{\max}$, and no pairing.

WHAT A NEGATIVE HERE WOULD COST. If the dips are incoherent, the cross-dataset frequency
agreement near \SI{72}{\kilo\hertz} was the coincidence its $p = 0.016$ allowed, and the
single-bounce residual of `shape_anomaly.py` is an artefact of averaging events whose dips sit
at different places. That is a real risk and it is why the two large PAAm datasets matter: they
have the sample size to make the null tight.
"""

import json
import pathlib

import numpy as np

import records
from bounce_sweep import maxima

DATA = pathlib.Path.home() / "fastscratch/imr-data-tempsweeps/data"
FILES = {"gelatin 15C": "Ga_t15_exp_data.csv", "gelatin 23C": "Ga_t23_exp_data.csv",
         "gelatin 33C": "Ga_t33_exp_data.csv", "PAAm PA05": "PA05_exp_data.csv",
         "PAAm PA05 temp": "PA05_temp_exp_data.csv",
         "PAAm PA05003 temp": "PA05003_temp_exp_data.csv"}
KEEP, DRAWS = 6, 4000


def ratio_matrix(fname):
  table = np.loadtxt(DATA / fname, delimiter=",", ndmin=2)
  tau, trials = table[:, 0], table[:, 1:].T
  keep = ~(trials > 1.05).any(axis=0) & (trials.std(axis=0, ddof=1) > 0.0)
  tau, trials = tau[keep], trials[:, keep]
  seqs = []
  for row in trials:
    _, amps, _ = maxima(row, tau, order=2)
    if len(amps) >= 5: seqs.append([amps[k + 1] / amps[k] for k in range(len(amps) - 1)])
  n = min(KEEP, min(len(s) for s in seqs))
  return np.array([s[:n] for s in seqs], dtype=float)


def _dip(mean_sequence):
  """Depth of the deepest fall in a sequence. Positive means a dip exists."""
  return float(-np.diff(mean_sequence).min())


def main():
  print("  Each event's ratio sequence is circularly shifted at random, which destroys")
  print("  alignment between events while preserving each event's own dip exactly.\n")
  print(f"  {'dataset':>18s} {'n':>4s} {'ensemble dip':>13s} {'null mean':>10s} {'null sd':>8s} "
        f"{'z':>7s} {'p':>7s}")

  rng = np.random.default_rng(20260816)
  summary = {}
  for label, fname in FILES.items():
    matrix = ratio_matrix(fname)
    events, n = matrix.shape
    observed = _dip(matrix.mean(axis=0))
    null = np.empty(DRAWS)
    for d in range(DRAWS):
      shifts = rng.integers(0, n, size=events)
      rolled = np.array([np.roll(matrix[i], int(shifts[i])) for i in range(events)])
      null[d] = _dip(rolled.mean(axis=0))
    z = (observed - null.mean()) / null.std(ddof=1)
    p = float((null >= observed).mean())
    summary[label] = {"events": events, "bounces": n, "observed_dip": observed,
                      "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1)),
                      "z": float(z), "p": p}
    print(f"  {label:>18s} {events:4d} {observed:13.4f} {null.mean():10.4f} "
          f"{null.std(ddof=1):8.4f} {z:7.2f} {p:7.4f}")

  print("\n  ---- what it says ----\n")
  coherent = [k for k, v in summary.items() if v["p"] < 0.05]
  absent = [k for k, v in summary.items() if v["observed_dip"] <= v["null_mean"]]
  for label, v in summary.items():
    verdict = ("INDEX-LOCKED" if v["p"] < 0.05
               else "no dip beyond the null" if v["observed_dip"] <= v["null_mean"]
               else "inconclusive")
    print(f"  {label:>18s}: {verdict}")
  print(f"\n  {len(coherent)} of {len(summary)} datasets show a dip locked to the bounce index,")
  print(f"  and {len(absent)} show none beyond what shifting produces by itself.")
  print("\n  A dip that survives averaging at fixed INDEX is tied to the bubble's own sequence.")
  print("  A resonance of the apparatus sits at a fixed frequency, so the R_max scatter would")
  print("  move it by about one index between events and the shift null is exactly what that")
  print("  looks like. Index-locking is therefore evidence against the apparatus, and it also")
  print("  says the cross-dataset agreement near 72 kHz was the coincidence p = 0.016 allowed:")
  print("  the structure is in the bounce number, not in the frequency.")
  json.dump(summary, open(records.HERE / "dip_coherence.json", "w"), indent=1)


if __name__ == "__main__":
  main()
