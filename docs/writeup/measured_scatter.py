r"""The setpoint scatter, measured on 437 events instead of assumed from three rows.

\Cref{sec:setpoint} prices what it costs to design as though a geometry were a knob, and
\cref{sec:control} prices buying the knob. Both rest on two numbers --- a coefficient of
variation of $0.25$ in $R_{\max}$ and $0.06$ in the stretch --- taken from the dataset table of
\citet{chu2026limits}, which reports a mean and a spread over a handful of records. Both also
assume a SHAPE, log-normal, and a STRUCTURE, that the two coordinates are drawn independently.
None of the three was measured here.

ALL THREE ARE MEASURABLE, AND THE FILE WAS ALREADY ON DISK. `PA0503_radius.csv` carries $437$
per-event pairs of $R_{\max}$ and $R_{\rm eq}$ in micrometres --- absolute radii, before the
normalisation that removes them from every trace this document fits. That is a direct sample of
the joint law the design was guessing at, two orders of magnitude larger than the table.

THREE MODELS, WHICH SEPARATE THE THREE ASSUMPTIONS. Independent log-normals at the assumed
coefficients are what the chapter used. Independent log-normals at the MEASURED coefficients
change only the width. Resampling the measured pairs changes the shape and the joint structure
together, and resampling them SHUFFLED separates the two: shuffling keeps each measured marginal
exactly and destroys only the correlation between them.

WHAT WOULD OVERTURN THE CHAPTER AND WHAT WOULD CONFIRM IT. If the recommended setpoint and the
price of ignoring scatter survive all four, then \cref{sec:setpoint}'s conclusion rests on a
measured law rather than on a convenient one, and the assumptions were harmless. If they move,
the design recommendation was an artifact of the distribution nobody had checked, and the
measured version is the one to report.

THE DEVIATIONS ARE MULTIPLICATIVE AND THAT IS NOT AN ASSUMPTION HERE. A setpoint is a laser
energy; what scales with it is the size, not an additive offset. So each event contributes
$(\log R_i - \overline{\log R},\ \log s_i - \overline{\log s})$, applied to whatever setpoint is
being evaluated. No density is fitted and no tail is extrapolated.
"""

import json

import numpy as np

import records
from noise_design import profile, raw_information

SOURCE = "gelatin_15C"
RADIUS_FILE = "PA0503_radius.csv"
# what sec:setpoint and sec:control assumed, from the paper's dataset table
ASSUMED_RADIUS_CV = 0.25
ASSUMED_STRETCH_CV = 0.06
DRAWS = 437
SEED = 0
LADDER = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)


def measured_pairs():
  """`(R_max in metres, stretch)` for every event in the acquisition's radius file."""
  from trial_modes import DATA

  table = np.loadtxt(DATA / RADIUS_FILE, delimiter=",")
  maximum, equilibrium = table[:, 0] * 1e-6, table[:, 1] * 1e-6
  good = np.isfinite(maximum) & np.isfinite(equilibrium) & (equilibrium > 0) & (maximum > 0)
  return maximum[good], maximum[good] / equilibrium[good]


def main():
  from scipy.interpolate import RegularGridInterpolator

  from design_operator import RADII, STRETCH
  from pyimr.measure import optimal_measure

  maximum, stretch = measured_pairs()
  log_radius = np.log(maximum) - np.log(maximum).mean()
  log_stretch = np.log(stretch) - np.log(stretch).mean()
  measured_radius_cv = float(maximum.std(ddof=1) / maximum.mean())
  measured_stretch_cv = float(stretch.std(ddof=1) / stretch.mean())
  print(f"  {len(maximum)} events from {RADIUS_FILE}\n")
  print(f"  {'quantity':>22s} {'assumed':>9s} {'measured':>9s}")
  print(f"  {'R_max cv':>22s} {ASSUMED_RADIUS_CV:9.3f} {measured_radius_cv:9.3f}")
  print(f"  {'stretch cv':>22s} {ASSUMED_STRETCH_CV:9.3f} {measured_stretch_cv:9.3f}")
  print(f"  {'corr(logR, log s)':>22s} {0.0:9.3f} "
        f"{float(np.corrcoef(log_radius, log_stretch)[0, 1]):9.3f}")
  print(f"  {'R_max mean (um)':>22s} {'277-312':>9s} {maximum.mean() * 1e6:9.1f}")
  print("  The design assumed independence; the measured correlation says whether that")
  print("  assumption was doing any work.")

  designs = [(float(r), float(s)) for r in RADII for s in STRETCH]
  print(f"\n  building information at {len(designs)} geometries ...", flush=True)
  with records.pool(len(designs)) as pool:
    got = list(pool.map(raw_information, designs))
  usable = {d: payload for d, payload in got if payload is not None}
  if len(usable) < len(designs):
    print(f"  {len(usable)} of {len(designs)} integrate; the grid must be complete")
    return

  tau, spread = profile(SOURCE)
  prior = 1.0 / np.sqrt(12.0)
  cube, size = None, None
  for i, radius in enumerate(RADII):
    for j, value in enumerate(STRETCH):
      columns, phase = usable[(float(radius), float(value))]
      whitened = columns / np.interp(phase, tau, spread)[:, None]
      fisher = whitened.T @ whitened
      fisher = 0.5 * (fisher + fisher.T) * prior**2
      if cube is None:
        size = fisher.shape[0]
        cube = np.zeros((len(RADII), len(STRETCH), size, size))
      cube[i, j] = fisher

  logr = np.log(np.asarray(RADII, dtype=float))
  interpolate = RegularGridInterpolator(
    (logr, np.asarray(STRETCH, dtype=float)), cube, bounds_error=False, fill_value=None)
  identity = np.eye(size)
  rng = np.random.default_rng(SEED)

  # four laws over the SAME deviations-from-setpoint question, isolating one assumption each
  shuffled = rng.permutation(log_stretch)
  laws = {
    "assumed log-normal": (rng.normal(0.0, np.log1p(ASSUMED_RADIUS_CV), DRAWS),
                           rng.normal(0.0, np.log1p(ASSUMED_STRETCH_CV), DRAWS)),
    "log-normal, measured cv": (rng.normal(0.0, np.log1p(measured_radius_cv), DRAWS),
                                rng.normal(0.0, np.log1p(measured_stretch_cv), DRAWS)),
    "measured, shuffled": (log_radius, shuffled),
    "measured, joint": (log_radius, log_stretch),
  }

  def delivered(setpoint, offsets, factor=1.0):
    """`0.5 log det (I + E[M])` at a setpoint under a law's multiplicative deviations."""
    target_r, target_s = setpoint
    if factor <= 0.0 or offsets is None:
      return 0.5 * float(np.linalg.slogdet(
        identity + interpolate(np.array([[np.log(target_r), target_s]]))[0])[1])
    shift_r, shift_s = offsets
    drawn_r = np.clip(np.log(target_r) + factor * shift_r, logr.min(), logr.max())
    drawn_s = np.clip(target_s * np.exp(factor * shift_s), min(STRETCH), max(STRETCH))
    averaged = interpolate(np.column_stack([drawn_r, drawn_s])).mean(axis=0)
    return 0.5 * float(np.linalg.slogdet(identity + averaged)[1])

  fine_r = np.exp(np.linspace(logr.min(), logr.max(), 40))
  fine_s = np.linspace(min(STRETCH), max(STRETCH), 24)
  candidates = [(float(r), float(s)) for r in fine_r for s in fine_s]
  nominal = np.array([delivered(c, None, factor=0.0) for c in candidates])
  best_nominal = int(np.argmax(nominal))

  summary = {"events": int(len(maximum)), "measured_radius_cv": measured_radius_cv,
             "measured_stretch_cv": measured_stretch_cv,
             "correlation": float(np.corrcoef(log_radius, log_stretch)[0, 1]),
             "nominal_setpoint": list(candidates[best_nominal])}

  print("\n  the scatter-aware optimum, and the regret of ignoring the scatter, under each law\n")
  print(f"  {'law':>24s} {'best setpoint':>22s} {'nats':>8s} {'regret':>8s} {'moved':>6s}")
  single = {}
  for label, offsets in laws.items():
    values = np.array([delivered(c, offsets) for c in candidates])
    top = int(np.argmax(values))
    regret = float(values[top] - values[best_nominal])
    single[label] = {"setpoint": list(candidates[top]), "nats": float(values[top]),
                     "regret": regret, "moved": bool(top != best_nominal)}
    print(f"  {label:>24s} {candidates[top][0] * 1e6:8.0f} um at {candidates[top][1]:5.2f} "
          f"{values[top]:8.3f} {regret:8.3f} {str(top != best_nominal):>6s}")
  summary["single"] = single
  span = max(v["regret"] for v in single.values()) - min(v["regret"] for v in single.values())
  print(f"\n  The regret spans {span:.3f} nats across all four laws. A span small against the")
  print("  regret itself means sec:setpoint's number did not depend on the assumed law.")
  summary["regret_span"] = float(span)

  print("\n  and the same question for a certified MEASURE, which is what the chapter designs\n")
  print(f"  {'law':>24s} {'settings':>9s} {'gap':>9s} {'nats':>8s}")
  coarse = [(float(r), float(s)) for r in fine_r[::3] for s in fine_s[::3]]
  batch = {}
  for label, offsets in laws.items():
    built = np.array([
      interpolate(np.column_stack([
        np.clip(np.log(r) + offsets[0], logr.min(), logr.max()),
        np.clip(s * np.exp(offsets[1]), min(STRETCH), max(STRETCH))])).mean(axis=0)
      for r, s in coarse])
    held = optimal_measure(built, iterations=200_000)
    averaged = np.tensordot(held.weights, built, axes=(0, 0))
    value = 0.5 * float(np.linalg.slogdet(identity + averaged)[1])
    batch[label] = {"settings": int(held.support.size), "gap": held.gap, "nats": value,
                    "support": [[coarse[i][0], coarse[i][1], float(held.weights[i])]
                                for i in held.support]}
    print(f"  {label:>24s} {held.support.size:9d} {held.gap:9.1e} {value:8.3f}")
  summary["batch"] = batch
  supports = {label: {(round(r * 1e6), round(s, 1)) for r, s, _ in v["support"]}
              for label, v in batch.items()}
  reference = supports["measured, joint"]
  print("\n  overlap of each law's certified support with the measured one\n")
  for label, chosen in supports.items():
    shared = len(chosen & reference)
    print(f"  {label:>24s} {shared:2d} of {len(reference):2d} settings shared")
  summary["support_overlap"] = {label: len(chosen & reference) for label, chosen in
                                supports.items()}

  print("\n  what tightening the measured scatter is worth, as a multiple of what is delivered\n")
  print(f"  {'x measured':>11s} {'R_max cv':>9s} {'stretch cv':>11s} {'nats':>8s} "
        f"{'vs measured':>12s}")
  control = {}
  base = None
  for factor in LADDER:
    values = [delivered(c, laws["measured, joint"], factor=factor) for c in candidates]
    value = float(max(values))
    if abs(factor - 1.0) < 1e-9: base = value
    control[f"{factor:g}"] = {"nats": value,
                              "radius_cv": measured_radius_cv * factor,
                              "stretch_cv": measured_stretch_cv * factor}
  for factor, row in control.items():
    row["gain_vs_measured"] = row["nats"] - base
    print(f"  {float(factor):11.2f} {row['radius_cv']:9.3f} {row['stretch_cv']:11.3f} "
          f"{row['nats']:8.3f} {row['gain_vs_measured']:12.3f}")
  summary["control"] = control
  print(f"\n  Perfect control of both, against the measured law, is worth "
        f"{control['0']['gain_vs_measured']:+.3f} nats at a single setpoint.")
  print("  sec:control found the sign of this flips between a setpoint and a measure, because a")
  print("  measure can emulate any scatter and beat it; the ladder above is the setpoint case.")

  json.dump(summary, open(records.HERE / "measured_scatter.json", "w"), indent=1)


if __name__ == "__main__":
  main()
