r"""Is the afterbounce loss curve a frequency dependence or an amplitude dependence?

`spectroscopy.py` reads $\tan\delta = -\ln r_k/\pi$ at each bounce's own frequency and finds it
falling with a log-log slope near $-2$ over $36$ to \SI{374}{\kilo\hertz}. Along one event's
sequence, though, later bounces are both HIGHER in frequency and SMALLER in amplitude, so that
slope is a frequency dependence and a strain dependence superimposed, and a single sequence
cannot separate them. A slope of $-2$ is also steeper than one Debye arm permits above its peak,
which is a hint that something other than a single relaxation is in it.

WHAT SEPARATES THEM IS HAVING 288 EVENTS RATHER THAN ONE. At a FIXED bounce index the relative
amplitude is nearly fixed, while the frequency still varies event to event because each bubble
has its own size and its own clock. So a regression of $\tan\delta$ on frequency WITHIN a bounce
index asks the frequency question at held amplitude, and the two-predictor regression over all
points asks it with amplitude carried explicitly. Whether either is answerable is a property of
the collinearity, which is measured here first rather than assumed away.

TEMPERATURE IS THE SECOND, INDEPENDENT LEVER. A relaxation time moves with temperature and a
strain nonlinearity largely does not, so if $\tan\delta$ at fixed frequency AND fixed amplitude
shifts across \SI{10}{}, \SI{21}{} and \SI{33}{\celsius}, that shift is spectral whatever the
first test says.

ERRORS ARE CLUSTERED BY EVENT, because five bounces of one bubble are not five independent
measurements, and the naive interval would be roughly $\sqrt{5}$ too narrow.
"""

import json

import numpy as np

import records
from bounce_sweep import maxima
from spectroscopy import DENSITY, AMBIENT, KELVIN, KEEP, _events
from shape_error import seed_for

DRAWS = 2000


def collect(dataset):
  """`(event, bounce, frequency, amplitude, tan delta)` for every bounce of every event."""
  tau, events, maximum = _events(dataset)
  characteristic = maximum * np.sqrt(DENSITY / AMBIENT)
  rows = []
  for index, row in enumerate(events):
    _, amps, times = maxima(row, tau, order=2)
    if len(amps) < 3: continue
    for k in range(min(KEEP, len(amps) - 1)):
      period = (times[k + 1] - times[k]) * characteristic
      ratio = amps[k + 1] / amps[k]
      if period <= 0 or not (0 < ratio < 1): continue
      rows.append((index, k, 1.0 / period, amps[k], -np.log(ratio) / np.pi))
  return np.array(rows, dtype=float)


def _fit(design, target):
  return np.linalg.lstsq(design, target, rcond=None)[0]


def _cluster_bootstrap(table, build, seed=0):
  """Resample EVENTS with replacement, not bounces: one bubble is one observation."""
  rng = np.random.default_rng(seed)
  events = np.unique(table[:, 0])
  out = []
  for _ in range(DRAWS):
    pick = rng.choice(events, size=events.size, replace=True)
    rows = np.vstack([table[table[:, 0] == e] for e in pick])
    try: out.append(build(rows))
    except np.linalg.LinAlgError: continue
  return np.array(out)


def main():
  print("  Frequency and amplitude are confounded along a sequence. 288 events break it.\n")
  tables = {d: collect(d) for d in (*records.DATASETS, *records.PAAM)}
  summary = {}

  print("  ---- can they be separated at all? ----\n")
  print(f"  {'dataset':>16s} {'points':>7s} {'events':>7s} {'corr(log f, log A)':>19s} "
        f"{'within-bounce corr':>19s}")
  for dataset, t in tables.items():
    lf, la = np.log(t[:, 2]), np.log(t[:, 3])
    overall = float(np.corrcoef(lf, la)[0, 1])
    within = []
    for k in np.unique(t[:, 1]):
      m = t[:, 1] == k
      if m.sum() > 10: within.append(float(np.corrcoef(lf[m], la[m])[0, 1]))
    print(f"  {dataset:>16s} {len(t):7d} {len(np.unique(t[:,0])):7d} {overall:19.3f} "
          f"{np.mean(within):19.3f}")
    summary[dataset] = {"points": len(t), "events": int(len(np.unique(t[:, 0]))),
                        "corr_overall": overall, "corr_within_bounce": float(np.mean(within))}

  print("\n  ---- the two-predictor regression, errors clustered by event ----\n")
  print(f"  {'dataset':>16s} {'d ln(tand)/d ln f':>19s} {'95% CI':>20s} "
        f"{'d ln(tand)/d ln A':>19s} {'95% CI':>20s}")
  for dataset, t in tables.items():
    def build(rows):
      design = np.column_stack([np.ones(len(rows)), np.log(rows[:, 2]), np.log(rows[:, 3])])
      return _fit(design, np.log(np.maximum(rows[:, 4], 1e-6)))
    point = build(t)
    boot = _cluster_bootstrap(t, build, seed=seed_for(dataset))
    ci_f = np.percentile(boot[:, 1], [2.5, 97.5])
    ci_a = np.percentile(boot[:, 2], [2.5, 97.5])
    print(f"  {dataset:>16s} {point[1]:19.3f} [{ci_f[0]:7.2f},{ci_f[1]:7.2f}] "
          f"{point[2]:19.3f} [{ci_a[0]:7.2f},{ci_a[1]:7.2f}]")
    summary[dataset] |= {"slope_frequency": float(point[1]),
                         "ci_frequency": [float(x) for x in ci_f],
                         "slope_amplitude": float(point[2]),
                         "ci_amplitude": [float(x) for x in ci_a],
                         "frequency_resolved": bool(ci_f[0] * ci_f[1] > 0),
                         "amplitude_resolved": bool(ci_a[0] * ci_a[1] > 0)}

  print("\n  ---- within a fixed bounce index: frequency alone, amplitude held ----\n")
  print(f"  {'dataset':>16s} " + " ".join(f"{'b' + str(k + 1):>10s}" for k in range(KEEP)))
  for dataset, t in tables.items():
    cells = []
    for k in range(KEEP):
      m = t[:, 1] == k
      if m.sum() < 15: cells.append("         -"); continue
      rows = t[m]
      slope = _fit(np.column_stack([np.ones(m.sum()), np.log(rows[:, 2])]),
                   np.log(np.maximum(rows[:, 4], 1e-6)))[1]
      cells.append(f"{slope:10.2f}")
    print(f"  {dataset:>16s} " + " ".join(cells))
    summary[dataset]["within_bounce_slopes"] = [
      float(_fit(np.column_stack([np.ones((t[:, 1] == k).sum()),
                                  np.log(t[t[:, 1] == k][:, 2])]),
                 np.log(np.maximum(t[t[:, 1] == k][:, 4], 1e-6)))[1])
      if (t[:, 1] == k).sum() >= 15 else None for k in range(KEEP)]

  print("\n  ---- the temperature lever, at matched bounce index ----\n")
  for family, group in (("gelatin", ("gelatin_15C", "gelatin_23C", "gelatin_33C")),
                        ("PAAm PA05", ("paam_PA05_10C", "paam_PA05_21C", "paam_PA05_33C"))):
    print(f"  {family}: mean tan(delta) at each bounce, by temperature")
    print(f"    {'T (K)':>7s} " + " ".join(f"{'b' + str(k + 1):>9s}" for k in range(KEEP)))
    for dataset in group:
      t = tables[dataset]
      cells = []
      for k in range(KEEP):
        m = t[:, 1] == k
        cells.append(f"{t[m, 4].mean():9.4f}" if m.sum() > 5 else "        -")
      print(f"    {KELVIN[dataset]:7.1f} " + " ".join(cells))
    print()

  print("  ---- what it says ----\n")
  resolved = [d for d, v in summary.items() if v.get("frequency_resolved")]
  both = [d for d in resolved if summary[d].get("amplitude_resolved")]
  print(f"  The frequency coefficient is resolved away from zero on {len(resolved)} of "
        f"{len(summary)} datasets,")
  print(f"  with amplitude carried explicitly, and both coefficients are resolved on {len(both)}.")
  print("  Where both are resolved the sequence is a genuine two-variable sweep and the loss")
  print("  curve is not a strain artefact. Where only amplitude is resolved it is the reverse.")
  records.HERE.joinpath("disentangle.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
  main()
