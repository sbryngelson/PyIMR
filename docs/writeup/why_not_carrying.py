r"""Why the two-timescale alignment works on one dataset and not the other.

\Cref{sec:latent} leaves a specific discrepancy. On the acquisition sweep, warping each event onto
its own collapse and afterbounce removes four fifths of the trial variance and destroys the
dilation signature --- $\lvert\cos\rvert$ from $0.85$--$0.90$ down to $0.09$--$0.30$. On the three
records this document fits, the same warp halves the variance and leaves the mode MORE
dilation-like at two records of three.

Three candidate explanations were named and none was tested: a different material at a different
stretch, $7$--$18$ trials against $84$--$253$ events, and time resolution. Two of the three can be
settled with the data already in hand.

RESOLUTION GOES FIRST, because it is arithmetic. The records carry $201$ samples over
$5\,t_c$, a spacing of $0.025\,t_c$. The sweep traces span up to $9.5$ inertial times in $253$
samples, a spacing of $0.037$. The records are FINER, and the warp works on the coarser data. So
resolution cannot be the explanation, and saying so costs one division.

TRIAL COUNT IS THE REAL TEST. If eighteen events cannot support the warp, then drawing eighteen
at random from a bin where it demonstrably works should reproduce the failure --- the same
material, the same instrument, the same processing, and only the count changed. Many draws, so
the answer is a distribution rather than one unlucky sample.

WHAT EACH OUTCOME MEANS. If subsampling reproduces the failure, the discrepancy is statistical
and the mechanism stands for both datasets; what the records lack is replicates, which is a
statement \cref{sec:allocation} already prices. If subsampling does NOT reproduce it --- if
eighteen sweep events still give $\lvert\cos\rvert$ near $0.1$ --- then the count is not the
reason and the material difference is what is left, which would be a physical result rather than
a bookkeeping one.
"""

import json
import pathlib

import numpy as np

import records

SOURCE = pathlib.Path("~/data_pa5_tempsweeps_master_20260210.mat").expanduser()
BINS = {"12-18C": (12.0, 18.0), "20-26C": (20.0, 26.0), "30-36C": (30.0, 36.0)}
PHASE = np.linspace(0.05, 1.95, 200)
COUNTS = (7, 14, 18, 40, 80)
DRAWS = 200
SEED = 0


def _scalar(entry, field):
  try:
    return float(getattr(entry, field))
  except Exception:                                                          # noqa: BLE001
    return float("nan")


def _cos(a, b):
  return abs(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))


def knots(clock, trace):
  from scipy.signal import argrelextrema

  minima = [i for i in argrelextrema(trace, np.less, order=3)[0] if clock[i] > 0.2]
  if len(minima) < 2: return None
  first, second = clock[minima[0]], clock[minima[1]]
  if not (0.3 < first < 2.0 and first < second < 4.0): return None
  return float(first), float(second)


def warped(clock, trace, first, second):
  phase = np.interp(clock, [0.0, first, second], [0.0, 1.0, 2.0])
  order = np.argsort(phase)
  return np.interp(PHASE, phase[order], trace[order])


def mode_of(stack):
  matrix = np.column_stack(stack)
  deviation = matrix - matrix.mean(axis=1, keepdims=True)
  left, values, _ = np.linalg.svd(deviation, full_matrices=False)
  mean = matrix.mean(axis=1)
  dilation = -PHASE * np.gradient(mean, PHASE)
  return (_cos(left[:, 0], dilation),
          float(np.sum(deviation**2) / deviation.shape[1]))


def main():
  import scipy.io as sio

  print("\n  resolution first, because it is arithmetic\n")
  from pyimr.noise import characteristic_time

  for dataset in records.DATASETS:
    times, mean, spread, maximum, stretch = records.load(dataset)
    scale = characteristic_time(maximum)
    span = (times[-1] - times[0]) / scale
    print(f"  {dataset:13s} {len(times):4d} samples over {span:5.2f} t_c "
          f"-> spacing {span / (len(times) - 1):.4f} t_c")

  if not SOURCE.exists():
    print(f"\n  {SOURCE} not present; the subsampling half needs it")
    return
  loaded = sio.loadmat(SOURCE, squeeze_me=True, struct_as_record=False)
  entries = loaded["dataPa5TempSweep"]
  temperature = np.array([_scalar(e, "T") for e in entries])
  flag = np.array([_scalar(e, "errorFlag") for e in entries])
  usable = (flag == 0) & np.isfinite(temperature)

  spacings = []
  for entry, ok in zip(entries, usable, strict=True):
    if not ok: continue
    clock = np.asarray(getattr(entry, "t_norm"), dtype=float)
    clock = clock[np.isfinite(clock)]
    if clock.size > 10: spacings.append(float(np.median(np.diff(np.sort(clock)))))
  print(f"  {'sweep events':13s} median spacing {np.median(spacings):.4f} inertial times")
  print("  The records are FINER than the data the warp works on, so resolution is not it.")

  print("\n  now the count: draw n events from a bin where the warp demonstrably works\n")
  rng = np.random.default_rng(SEED)
  summary = {"record_spacing": {}, "sweep_spacing": float(np.median(spacings)), "counts": {}}
  for dataset in records.DATASETS:
    times, mean, spread, maximum, stretch = records.load(dataset)
    span = (times[-1] - times[0]) / characteristic_time(maximum)
    summary["record_spacing"][dataset] = float(span / (len(times) - 1))

  print(f"  {'bin':>8s} {'n':>4s} {'|cos| before':>26s} {'|cos| after':>26s} "
        f"{'P(after > before)':>18s}")
  for name, (low, high) in BINS.items():
    rows = []
    for entry, ok, t in zip(entries, usable, temperature, strict=True):
      if not ok or not (low <= t < high): continue
      trace = np.asarray(getattr(entry, "R_norm"), dtype=float)
      clock = np.asarray(getattr(entry, "t_norm"), dtype=float)
      good = np.isfinite(trace) & np.isfinite(clock)
      if good.sum() < 60: continue
      pair = knots(clock[good], trace[good])
      if pair is None: continue
      rows.append((clock[good], trace[good], pair[0], pair[1]))
    if len(rows) < max(COUNTS): continue
    entry_out = {}
    for count in COUNTS:
      before, after = [], []
      for _ in range(DRAWS):
        pick = rng.choice(len(rows), size=count, replace=False)
        chosen = [rows[i] for i in pick]
        flat = [warped(c, tr, 1.0, 2.0) for c, tr, _, _ in chosen]
        real = [warped(c, tr, a, b) for c, tr, a, b in chosen]
        before.append(mode_of(flat)[0])
        after.append(mode_of(real)[0])
      before, after = np.array(before), np.array(after)
      worse = float(np.mean(after > before))
      entry_out[count] = {"before_median": float(np.median(before)),
                          "before_iqr": [float(np.percentile(before, 25)),
                                         float(np.percentile(before, 75))],
                          "after_median": float(np.median(after)),
                          "after_iqr": [float(np.percentile(after, 25)),
                                        float(np.percentile(after, 75))],
                          "prob_worse": worse}
      print(f"  {name:>8s} {count:4d} {np.median(before):8.3f} "
            f"[{np.percentile(before, 25):.3f}, {np.percentile(before, 75):.3f}]"
            f"{np.median(after):11.3f} "
            f"[{np.percentile(after, 25):.3f}, {np.percentile(after, 75):.3f}]"
            f"{worse:18.2f}")
    summary["counts"][name] = entry_out
    print()

  print("  'P(after > before)' is the probability the warp makes the dilation signature WORSE,")
  print("  which is what happens on two of this document's three records. If that probability")
  print("  is high at n = 7 to 18 and low at n = 80, the discrepancy is the replicate count and")
  print("  the mechanism stands for both datasets.")

  json.dump(summary, open(records.HERE / "why_not_carrying.json", "w"), indent=1)


if __name__ == "__main__":
  main()
