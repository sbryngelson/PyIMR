r"""The clock mechanism, on both materials, from one file format.

\Cref{sec:latent} identified the dominant trial mode as a clock error and \cref{sec:twotime}
removed four fifths of it with a two-knot warp --- on the `pa5` acquisition sweep. The mechanism
did not carry to the gelatin records this document fits, and that was left open on the stated
grounds that resolving it needed raw per-event traces for the other experiment.

IT DID NOT. The processed per-trial traces for BOTH materials sit in the same directory, in the
same format: column zero a normalised clock measured from the peak, the remaining columns one
$R/R_{\max}$ trace per event. Gelatin at three temperatures, and polyacrylamide at three
concentrations and sweeps -- $39$ gelatin events against roughly $250$ PAAm ones.

WHY THAT MAKES IT A CLEAN COMPARISON RATHER THAN ANOTHER HYPOTHESIS. The two materials arrive
sampled at the SAME step, $\Delta t = 0.025$ in the normalised clock. \Cref{sec:twotime} had to
eliminate resolution and replicate count by hand as rival explanations; here resolution is
identical by construction, so a difference between materials cannot be one. The warp, the sham
control and the mode diagnostic are the ones \cref{sec:twotime} already used, applied unchanged
to both.

WHAT EACH OUTCOME MEANS. If the warp removes the mode in PAAm and not in gelatin at equal
resolution, then the difference is the material, and gelatin's phase transition near
\SI{30}{\celsius} becomes the thing to explain rather than the pipeline. If it removes the mode
in both, then \#222 is a clock effect everywhere and the earlier gelatin result was a
resolution artifact after all. If it removes it in neither, the `pa5` result was the outlier.

THE SHAM CONTROL IS WHAT MAKES THE NUMBER MEAN ANYTHING. Two fitted knots per event guarantee
some variance reduction. The control draws the same two knots from the population instead of
from each event's own trace, so what a real alignment buys is the difference between them.
"""

import json

import numpy as np

import records
from trial_modes import DATA
from two_timescale import knots, mode_of, warped

SEED = 0
# every record in the directory that carries one trace per column against a shared normalised
# clock. The two materials are sampled at the same step, which is what makes them comparable.
RECORDS = {
  "gelatin 15C": ("Ga_t15_exp_data.csv", "gelatin"),
  "gelatin 23C": ("Ga_t23_exp_data.csv", "gelatin"),
  "gelatin 33C": ("Ga_t33_exp_data.csv", "gelatin"),
  "PAAm 0.5%": ("PA05_exp_data.csv", "PAAm"),
  "PAAm 0.5% sweep": ("PA05_temp_exp_data.csv", "PAAm"),
  "PAAm 0.5/0.03% sweep": ("PA05003_temp_exp_data.csv", "PAAm"),
}


def events(filename):
  """`[(clock, trace, first minimum, second minimum)]` for every resolvable column."""
  table = np.loadtxt(DATA / filename, delimiter=",")
  clock = table[:, 0]
  rows = []
  for column in range(1, table.shape[1]):
    trace = table[:, column]
    good = np.isfinite(trace) & np.isfinite(clock)
    if good.sum() < 60: continue
    pair = knots(clock[good], trace[good])
    if pair is None: continue
    rows.append((clock[good], trace[good], pair[0], pair[1]))
  return rows


def main():
  rng = np.random.default_rng(SEED)
  summary = {}
  gathered = {}

  print("\n  the two timescales per event, in each record's own normalised clock\n")
  print(f"  {'record':>21s} {'material':>9s} {'n':>4s} {'collapse':>9s} {'cv':>7s} "
        f"{'rebound':>9s} {'cv':>7s} {'corr':>7s}")
  for label, (filename, material) in RECORDS.items():
    if not (DATA / filename).exists():
      print(f"  {label:>21s} missing")
      continue
    rows = events(filename)
    if len(rows) < 6:
      print(f"  {label:>21s} {material:>9s} {len(rows):4d}  too few resolvable events")
      continue
    first = np.array([r[2] for r in rows])
    rebound = np.array([r[3] for r in rows]) - first
    gathered[label] = rows
    summary[label] = {
      "material": material, "events": len(rows),
      "collapse_mean": float(first.mean()),
      "collapse_cv": float(first.std(ddof=1) / first.mean()),
      "rebound_mean": float(rebound.mean()),
      "rebound_cv": float(rebound.std(ddof=1) / rebound.mean()),
      "correlation": float(np.corrcoef(first, rebound)[0, 1])}
    got = summary[label]
    print(f"  {label:>21s} {material:>9s} {len(rows):4d} {first.mean():9.4f} "
          f"{got['collapse_cv']:7.4f} {rebound.mean():9.4f} {got['rebound_cv']:7.4f} "
          f"{got['correlation']:+7.3f}")
  print("  A correlation near zero says the two hands move independently, so no single scale")
  print("  factor can align both -- the premise the two-knot warp rests on.")

  print("\n  the dominant mode under each alignment, and what the warp removes\n")
  print(f"  {'record':>21s} {'alignment':>18s} {'share':>7s} {'|cos| dilation':>15s} "
        f"{'total var':>11s}")
  for label, rows in gathered.items():
    first = np.array([r[2] for r in rows])
    second = np.array([r[3] for r in rows])
    variants = {
      "inertial only": [(1.0, 2.0)] * len(rows),
      "collapse only": [(f, 2.0 * f) for f in first],
      "two timescales": list(zip(first, second, strict=True)),
      # the control: each event warped with ANOTHER event's knot PAIR. Permuting the two knots
      # independently would let the first land past the second and break the warp's
      # monotonicity, which inflates the sham's variance for a reason that has nothing to do
      # with alignment; moving the pair together keeps the knot geometry realistic and breaks
      # only the match between an event and its own timing.
      "sham (population)": [(first[j], second[j]) for j in rng.permutation(len(rows))],
    }
    entry = {}
    for name, pairs in variants.items():
      stack = [warped(clock, trace, a, b)
               for (clock, trace, _, _), (a, b) in zip(rows, pairs, strict=True)
               if 0.0 < a < b]
      if len(stack) < 6: continue
      entry[name] = mode_of(stack)
      got = entry[name]
      print(f"  {label if name == 'inertial only' else '':>21s} {name:>18s} "
            f"{got['share']:7.3f} {got['cos_dilation']:15.3f} {got['total_variance']:11.4f}")
    summary[label]["modes"] = entry
    if "inertial only" in entry and "two timescales" in entry:
      base = entry["inertial only"]
      real = entry["two timescales"]
      sham = entry.get("sham (population)", base)
      removed = 1.0 - real["total_variance"] / base["total_variance"]
      free = 1.0 - sham["total_variance"] / base["total_variance"]
      summary[label]["removed"] = float(removed)
      summary[label]["free"] = float(free)
      # `removed - free` is only a sensible net when the sham reduces anything. When a random
      # warp INCREASES the variance the two are not on one scale, and the honest statement is
      # the pair: what the real warp removes, and whether the sham removes any of it.
      summary[label]["beats_sham"] = bool(real["total_variance"] < sham["total_variance"])
      verdict = "none of it is free" if free <= 0 else f"{free:.1%} of it is free"
      print(f"  {'':>21s} {'-> removed':>18s} {removed:7.1%} of the variance, and {verdict}")

  print("\n  by material, at identical sampling -- which is the whole point of running both\n")
  print(f"  {'material':>9s} {'records':>8s} {'events':>7s} {'|cos| before':>13s} "
        f"{'|cos| after':>12s} {'variance removed':>17s}")
  by_material = {}
  for label, got in summary.items():
    if "modes" not in got or "two timescales" not in got["modes"]: continue
    by_material.setdefault(got["material"], []).append(got)
  for material, rows in by_material.items():
    weight = np.array([r["events"] for r in rows], dtype=float)
    before = np.average([r["modes"]["inertial only"]["cos_dilation"] for r in rows],
                        weights=weight)
    after = np.average([r["modes"]["two timescales"]["cos_dilation"] for r in rows],
                       weights=weight)
    net = np.average([r["removed"] for r in rows], weights=weight)
    by_material[material] = {"records": len(rows), "events": int(weight.sum()),
                             "cos_before": float(before), "cos_after": float(after),
                             "removed": float(net),
                             "beats_sham": all(r["beats_sham"] for r in rows)}
    print(f"  {material:>9s} {len(rows):8d} {int(weight.sum()):7d} {before:13.3f} "
          f"{after:12.3f} {net:16.1%}")
  summary["by_material"] = by_material
  print("\n  Resolution is identical across these records (dt = 0.025 in the normalised clock),")
  print("  so a difference between materials here cannot be the resolution explanation that")
  print("  sec:twotime had to eliminate by hand on the acquisition sweep.")

  json.dump(summary, open(records.HERE / "material_clock.json", "w"), indent=1)


if __name__ == "__main__":
  main()
