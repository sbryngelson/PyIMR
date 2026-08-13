r"""Does a two-timescale alignment remove what one cannot?

\Cref{sec:latent} leaves \#222 at a specific place. The dominant trial mode is a clock error ---
it overlaps $-t\dot R$ at $0.87$ to $0.95$ under every clock tried --- and no single scale factor
removes it, because the collapse and the rebound do not scale together. A bubble has two
independent timescales: the collapse, set by $R_{\max}$ and the ambient pressure, and the
afterbounce, set by the gas it is left with. The pipeline normalises by one of them.

Both are measurable per event. The peak sits at $t_{\rm norm} = 0$ by construction, the first
minimum near $0.94$, the second near $1.73$, so the collapse time and the first afterbounce
period can be read off each trace directly. A piecewise-linear warp taking
%
    (peak, first minimum, second minimum)  ->  (0, 1, 2)

then aligns both, and every event is compared at the same PHASE of the same event in its own
history rather than at the same multiple of one estimated scale.

WHAT EACH OUTCOME MEANS, and both are worth having. If the mode collapses, \#222 is explained
outright: it is a clock effect, the clock has two hands, and the remedy is a processing change
rather than a model change. If it survives, what is left has been stripped of every timing
degree of freedom the data contains, which makes it a far sharper object than the mode this
issue started with --- and the first thing in this whole investigation that is neither a
parameter, nor a configuration quantity, nor a clock.

THE WARP IS NOT FREE AND THE COST IS REPORTED. Two knots per event is two fitted quantities, so
some variance reduction is guaranteed by construction. The comparison that controls for it is a
SHAM warp: the same two knots, drawn from the population rather than from each event's own
trace. What a real alignment buys is the difference between them, not its own number.
"""

import json
import pathlib

import numpy as np

import records

SOURCE = pathlib.Path("~/data_pa5_tempsweeps_master_20260210.mat").expanduser()
BINS = {"12-18C": (12.0, 18.0), "20-26C": (20.0, 26.0), "30-36C": (30.0, 36.0)}
PHASE = np.linspace(0.05, 1.95, 200)        # peak to second minimum, in warped units
SEED = 0


def _scalar(entry, field):
  try:
    return float(getattr(entry, field))
  except Exception:                                                          # noqa: BLE001
    return float("nan")


def _cos(a, b):
  return abs(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))


def knots(clock, trace):
  """`(first minimum, second minimum)` in the event's own clock, or None if not resolved."""
  from scipy.signal import argrelextrema

  after = clock > 0.2
  if after.sum() < 20: return None
  minima = argrelextrema(trace, np.less, order=3)[0]
  minima = [i for i in minima if clock[i] > 0.2]
  if len(minima) < 2: return None
  first, second = clock[minima[0]], clock[minima[1]]
  if not (0.3 < first < 2.0 and first < second < 4.0): return None
  return float(first), float(second)


def warped(clock, trace, first, second):
  """Resample on the phase grid under the warp `(0, first, second) -> (0, 1, 2)`."""
  phase = np.interp(clock, [0.0, first, second], [0.0, 1.0, 2.0])
  order = np.argsort(phase)
  return np.interp(PHASE, phase[order], trace[order])


def mode_of(stack):
  """Leading deviation mode of resampled traces, its dilation overlap, and the total variance."""
  matrix = np.column_stack(stack)
  deviation = matrix - matrix.mean(axis=1, keepdims=True)
  left, values, _ = np.linalg.svd(deviation, full_matrices=False)
  mean = matrix.mean(axis=1)
  dilation = -PHASE * np.gradient(mean, PHASE)
  unit = dilation / np.linalg.norm(dilation)
  fraction = (unit @ deviation) / np.linalg.norm(dilation)
  fitted = np.outer(dilation, fraction)
  return {
    "events": matrix.shape[1],
    "share": float(values[0] ** 2 / (values**2).sum()),
    "cos_dilation": _cos(left[:, 0], dilation),
    "dilation_variance": 1.0 - float(np.sum((deviation - fitted) ** 2) / np.sum(deviation**2)),
    "total_variance": float(np.sum(deviation**2) / deviation.shape[1]),
  }


def main():
  import scipy.io as sio

  if not SOURCE.exists():
    print(f"  {SOURCE} not present; nothing to do")
    return
  loaded = sio.loadmat(SOURCE, squeeze_me=True, struct_as_record=False)
  entries = loaded["dataPa5TempSweep"]
  temperature = np.array([_scalar(e, "T") for e in entries])
  flag = np.array([_scalar(e, "errorFlag") for e in entries])
  usable = (flag == 0) & np.isfinite(temperature)
  rng = np.random.default_rng(SEED)

  print("\n  the two timescales, per event, in units of the inertial one\n")
  print(f"  {'bin':>8s} {'n':>4s} {'collapse':>16s} {'cv':>7s} {'afterbounce':>17s} {'cv':>7s} "
        f"{'corr':>7s}")
  summary = {}
  built = {}
  for name, (low, high) in BINS.items():
    chosen = [e for e, ok, t in zip(entries, usable, temperature, strict=True)
              if ok and low <= t < high]
    rows = []
    for entry in chosen:
      trace = np.asarray(getattr(entry, "R_norm"), dtype=float)
      clock = np.asarray(getattr(entry, "t_norm"), dtype=float)
      good = np.isfinite(trace) & np.isfinite(clock)
      if good.sum() < 60: continue
      trace, clock = trace[good], clock[good]
      pair = knots(clock, trace)
      if pair is None: continue
      rows.append((clock, trace, pair[0], pair[1]))
    if len(rows) < 8: continue
    first = np.array([r[2] for r in rows]); second = np.array([r[3] for r in rows])
    rebound = second - first
    summary[name] = {
      "events": len(rows), "collapse_mean": float(first.mean()),
      "collapse_cv": float(first.std(ddof=1) / first.mean()),
      "rebound_mean": float(rebound.mean()),
      "rebound_cv": float(rebound.std(ddof=1) / rebound.mean()),
      "correlation": float(np.corrcoef(first, rebound)[0, 1])}
    built[name] = rows
    row = summary[name]
    print(f"  {name:>8s} {len(rows):4d} {first.mean():9.4f} +- {first.std(ddof=1):.4f} "
          f"{row['collapse_cv']:7.4f} {rebound.mean():10.4f} +- {rebound.std(ddof=1):.4f} "
          f"{row['rebound_cv']:7.4f} {row['correlation']:+7.3f}")
  print("  A correlation near zero is the whole argument: two timescales that vary")
  print("  independently cannot both be fixed by one scale factor.")

  print("\n  the mode under each alignment\n")
  print(f"  {'bin':>8s} {'alignment':>16s} {'n':>4s} {'share':>7s} {'|cos| dilation':>15s} "
        f"{'dilation var':>13s} {'total var':>11s}")
  table = {}
  for name, rows in built.items():
    first = np.array([r[2] for r in rows]); second = np.array([r[3] for r in rows])
    variants = {
      "inertial only": [(1.0, 2.0)] * len(rows),
      "collapse only": [(f, 2.0 * f) for f in first],
      "two timescales": list(zip(first, second, strict=True)),
      # the control: two knots per event, but drawn from the population rather than measured,
      # so any reduction it shows is what warping buys for free
      "sham (population)": [(a, b) for a, b in zip(
        rng.permutation(first), rng.permutation(second), strict=True)],
    }
    entry = {}
    for label, pairs in variants.items():
      stack = []
      for (clock, trace, _, _), (a, b) in zip(rows, pairs, strict=True):
        if not (0.0 < a < b): continue
        stack.append(warped(clock, trace, a, b))
      if len(stack) < 8: continue
      entry[label] = mode_of(stack)
      got = entry[label]
      print(f"  {name:>8s} {label:>16s} {got['events']:4d} {got['share']:7.3f} "
            f"{got['cos_dilation']:15.3f} {got['dilation_variance']:13.3f} "
            f"{got['total_variance']:11.3e}")
    table[name] = entry
    print()
  summary_out = {"timescales": summary, "modes": table}

  print(f"  {'bin':>8s} {'two-scale vs inertial':>23s} {'two-scale vs sham':>20s} "
        f"{'dilation share':>16s}")
  for name, entry in table.items():
    if not {"inertial only", "two timescales", "sham (population)"} <= set(entry): continue
    base, real, sham = (entry["inertial only"], entry["two timescales"],
                        entry["sham (population)"])
    print(f"  {name:>8s} {'x' + format(real['total_variance'] / base['total_variance'], '.3f'):>23s} "
          f"{'x' + format(real['total_variance'] / sham['total_variance'], '.3f'):>20s} "
          f"{format(base['dilation_variance'], '.3f') + ' -> ' + format(real['dilation_variance'], '.3f'):>16s}")
    summary_out.setdefault("reductions", {})[name] = {
      "vs_inertial": real["total_variance"] / base["total_variance"],
      "vs_sham": real["total_variance"] / sham["total_variance"]}

  print("\n  The middle column is the one that means anything: a warp with two knots removes")
  print("  variance whatever the knots are, and only the ratio against the sham says how much")
  print("  of that came from the timescales being MEASURED rather than merely fitted.")

  json.dump(summary_out, open(records.HERE / "two_timescale.json", "w"), indent=1)


if __name__ == "__main__":
  main()
