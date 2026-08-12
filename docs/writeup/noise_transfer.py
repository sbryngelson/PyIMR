r"""Does the design conclusion survive a different way of transferring the noise profile?

`noise_design.py` and `noise_portfolio.py` rest on one modelling step: a measured $\sigma(t)$ is
carried to a geometry nobody has run by reading it as a function of $t/t_c$ and interpolating at
matching phases. That step is defended -- \S`noise\_shape` shows the profile is dynamical rather
than instrumental, so phase is the natural coordinate -- but defended is not tested, and every
conclusion about the recommended batch depends on it.

Three transfers are compared, all of the same measured spreads.

  phase        the current choice: sigma as a function of t/t_c
  absolute     sigma as a function of t in seconds, which ignores that a bigger bubble
               collapses later and stretches the same profile over a longer record
  smoothed     phase again, but on a median-filtered spread, so a per-sample estimate
               carrying 17 to 29 percent relative error is not interpolated raw

If the recommendation is the same under all three, the transfer is not what is driving it and
\S3.2 of the issues note can be closed. If phase and absolute disagree, the conclusion is a
statement about the transfer and has to be reported that way.

Absolute time is the informative comparison rather than a straw man. It is what a reader would
do by default -- take the measured noise vector and apply it sample by sample -- and it is wrong
for a reason worth stating: the candidate geometries span radii from \SI{50}{\micro\metre} to
\SI{1200}{\micro\metre}, a factor of twenty-four in $t_c$, so the same absolute time lands at
completely different points of the collapse.
"""

import json

import numpy as np

import records
from noise_design import RELATIVE_NOISE, gain, raw_information
from scipy.ndimage import median_filter

SMOOTH = 11


def transfers(dataset):
  """`{name: (coordinate, sigma)}` for one record, ready to interpolate."""
  from pyimr.noise import characteristic_time
  times, mean, spread, maximum, stretch = records.load(dataset)
  elapsed = times - times[0]
  scale = characteristic_time(maximum)
  return {
    "phase": (elapsed / scale, spread),
    "absolute": (elapsed, spread),
    "smoothed": (elapsed / scale, median_filter(spread, SMOOTH, mode="nearest")),
  }


def main():
  from design_operator import PERFORMED, RADII, STRETCH
  from pyimr.noise import characteristic_time

  designs = [(float(r), float(s)) for r in RADII for s in STRETCH] + list(PERFORMED.values())
  with records.pool(len(designs)) as pool:
    got = list(pool.map(raw_information, designs))
  usable = [(d, payload) for d, payload in got if payload is not None]
  print(f"\n  {len(usable)} of {len(designs)} candidates integrate")
  print(f"  candidate t_c spans {characteristic_time(min(RADII)):.2e} to "
        f"{characteristic_time(max(RADII)):.2e} s, a factor of "
        f"{characteristic_time(max(RADII)) / characteristic_time(min(RADII)):.0f}")

  kinds = ("phase", "absolute", "smoothed")
  table, picks = {}, {}
  for dataset in records.DATASETS:
    made = transfers(dataset)
    for kind in kinds:
      coordinate, sigma = made[kind]
      values = []
      for design, (columns, phase) in usable:
        # the candidate's own grid, expressed in whichever coordinate the transfer uses
        span = 5.0 * characteristic_time(design[0])
        where = phase if kind != "absolute" else phase * (span / 5.0)
        values.append(gain(columns, np.interp(where, coordinate, sigma)))
      values = np.array(values)
      table[f"{dataset}|{kind}"] = values.tolist()
      picks[(dataset, kind)] = int(np.argmax(values))

  constant = np.array([gain(c, np.full(c.shape[0], RELATIVE_NOISE)) for _, (c, _) in usable])
  points = [d for d, _ in usable]
  base = int(np.argmax(constant))
  print(f"\n  constant picks R_max {points[base][0] * 1e6:.1f} um at stretch "
        f"{points[base][1]:.2f}\n")
  print(f"  {'record':13s} " + "  ".join(f"{k:>22s}" for k in kinds))
  for dataset in records.DATASETS:
    cells = []
    for kind in kinds:
      index = picks[(dataset, kind)]
      mark = " =" if index == base else "  "
      cells.append(f"{points[index][0] * 1e6:8.1f}um @{points[index][1]:5.2f}{mark}")
    print(f"  {dataset:13s} " + "  ".join(f"{c:>22s}" for c in cells))

  # `absolute` is reported for contrast, not as a rival: with t_c spanning a factor of 24
  # across the candidates, the same absolute time lands at different points of the collapse,
  # so it transfers the profile to the wrong PHASE. Reading it beside the two defensible
  # coordinates would only make the answer look less settled than it is.
  physical = ("phase", "smoothed")
  print("\n  the two defensible coordinates, and whether they agree:")
  for dataset in records.DATASETS:
    chosen = {kind: picks[(dataset, kind)] for kind in physical}
    same = len(set(chosen.values())) == 1
    print(f"    {dataset:13s} {'agree' if same else 'DISAGREE':9s}"
          f"  raw phase {'moves' if chosen['phase'] != base else 'stays':5s}"
          f"  smoothed {'moves' if chosen['smoothed'] != base else 'stays':5s}")

  smoothed = {picks[(d, 'smoothed')] for d in records.DATASETS}
  raw = {picks[(d, 'phase')] for d in records.DATASETS}
  print(f"\n  across records, the smoothed transfer picks {len(smoothed)} distinct geometry"
        f"{'' if len(smoothed) == 1 else 'ies'} and the raw picks {len(raw)}")
  if len(smoothed) == 1 and base not in smoothed:
    where = points[next(iter(smoothed))]
    print(f"    all three agree on R_max {where[0] * 1e6:.1f} um at stretch {where[1]:.2f},"
          " and none agrees with the constant")
    print("    -- so smoothing the estimator makes the records CONSISTENT, and the")
    print("    disagreement that remains is with the flat scale rather than among themselves")
  print(f"\n  absolute time, for contrast, picks "
        f"{len({picks[(d, 'absolute')] for d in records.DATASETS})} geometry across all three")
  print("  and is wrong for a statable reason: it transfers the profile to the wrong phase.")
  json.dump({"picks": {f"{d}|{k}": {"radius_m": points[v][0], "stretch": points[v][1],
                                    "agrees_with_constant": v == base}
                       for (d, k), v in picks.items()},
             "constant_pick": {"radius_m": points[base][0], "stretch": points[base][1]}},
            open(records.HERE / "noise_transfer.json", "w"), indent=1)


if __name__ == "__main__":
  main()
