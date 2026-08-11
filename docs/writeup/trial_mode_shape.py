"""Where does the dominant trial mode live, and is it the same curve on every record?

`trial_modes.py` leaves the dominant mode unexplained: it carries $54$ to $72\\%$ of the
trial-to-trial scatter and no sensitivity this model has reaches it -- not timing, not `Req`,
not `R_max` through the nondimensional groups, not the four material axes, and not the
normalisation each trace carries. That is a statement about what it is NOT. This asks two
questions that do not need a model at all.

WHERE IN TIME. If the deviation concentrates where the radius moves fastest, it is a timing or
edge-detection artifact wearing a shape. If it concentrates at the collapse minimum, it is
about the depth each bubble reaches. If it is spread evenly it is neither.

WHETHER IT IS THE SAME CURVE. Three records at three temperatures share an apparatus, an
operator and a segmentation pipeline; they do not share a material state. A mode that repeats
across all three is a property of the measurement, and no constitutive law will absorb it. A
mode that differs is a property of each experiment. The records have different sample counts
and different time bases, so the comparison is made on the common nondimensional grid.

Deliberately model-free. Every previous attempt asked "which of my sensitivities is this?" and
the answer was none of them; this asks what the curve looks like before assuming it is
anything.
"""

import json

import numpy as np

import records
from trial_modes import DATA, FILES


def _deviations(name):
  """`(times, mean trace, per-trial deviations)` on the record's own grid."""
  filename, offset = FILES[name]
  times, mean, spread, maximum, stretch = records.load(name)
  raw = np.loadtxt(DATA / filename, delimiter=",")
  window = raw[offset:offset + times.size, 1:]
  return times, window.mean(axis=1), window - window.mean(axis=1, keepdims=True)


def main():
  from pyimr.noise import characteristic_time

  table, shapes = {}, {}
  print("\n  where the dominant mode's weight sits, as a share of its squared length\n")
  print(f"  {'record':13s} {'share':>7s} {'pre-collapse':>13s} {'at collapse':>12s} "
        f"{'rebound':>9s} {'peak at t/t_c':>14s}")
  for name in FILES:
    times, mean, deviations = _deviations(name)
    left, sv, _ = np.linalg.svd(deviations, full_matrices=False)
    mode = left[:, 0]
    # sign is arbitrary in an SVD; fix it so the records can be compared at all
    if mode[np.argmax(np.abs(mode))] < 0: mode = -mode
    bottom = int(np.argmin(mean))
    weight = mode**2 / float(mode @ mode)
    # the collapse window is the tenth of the record centred on the minimum
    half = max(1, times.size // 20)
    at = slice(max(0, bottom - half), bottom + half)
    scale = characteristic_time(records.DATASETS[name])
    grid = (times - times[0]) / scale
    table[name] = {
      "share": float(sv[0] ** 2 / (sv**2).sum()),
      "pre": float(weight[: at.start].sum()), "at": float(weight[at].sum()),
      "post": float(weight[at.stop :].sum()),
      "peak_tc": float(grid[int(np.argmax(np.abs(mode)))]),
      "minimum_tc": float(grid[bottom]),
    }
    shapes[name] = (grid, mode)
    row = table[name]
    print(f"  {name:13s} {row['share']:7.3f} {row['pre']:13.1%} {row['at']:12.1%} "
          f"{row['post']:9.1%} {row['peak_tc']:14.3f}")

  print("\n  is it the same curve? |cos| between records, on a common t/t_c grid\n")
  common = np.linspace(0.0, min(float(g.max()) for g, _ in shapes.values()), 400)
  resampled = {}
  for name, (grid, mode) in shapes.items():
    v = np.interp(common, grid, mode)
    resampled[name] = v / np.linalg.norm(v)
  names = list(FILES)
  print("  " + " " * 13 + " ".join(f"{n[-3:]:>8s}" for n in names))
  for a in names:
    cells = [f"{abs(float(resampled[a] @ resampled[b])):8.3f}" for b in names]
    print(f"  {a:13s} " + " ".join(cells))
  pairs = [abs(float(resampled[a] @ resampled[b]))
           for i, a in enumerate(names) for b in names[i + 1:]]
  print(f"\n  off-diagonal mean {np.mean(pairs):.3f}; two unrelated curves of this length "
        f"would sit near {1 / np.sqrt(common.size):.3f}")
  print("  a shared curve is the apparatus, not the material -- no constitutive law absorbs it")

  json.dump({"where": table, "cos": {f"{a}|{b}": abs(float(resampled[a] @ resampled[b]))
                                     for a in names for b in names}},
            open(records.HERE / "trial_mode_shape.json", "w"), indent=1)


if __name__ == "__main__":
  main()
