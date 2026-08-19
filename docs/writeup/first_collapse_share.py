r"""Reconcile ``65 to 69 percent sits in the first collapse'' against the mode's 47 percent.

Two passages of `selection.tex` state incompatible things about where $\hat\delta$ lives.
\Cref{sec:discrepancy} says between $65$ and \SI{69}{\percent} of each of the three gelatin
curves sits in the first collapse; \cref{sec:universal} says over half of the common mode is
POST-collapse, and `universal_curve.json` records \SI{47.5}{\percent} inside it. A referee can
quote both sentences, so the difference is measured here rather than assumed to be definitional.

Three things differ between the two and each is priced separately: the SET (three gelatin
records against all eight), the CENTRING (the older number is a share of raw squared norm, the
mode is mean-removed before the decomposition), and the WINDOW ($t/t_c \le 1.2$ against
whatever the older passage meant, which is why the share is reported as a function of the cut).
"""

import json

import numpy as np

import records
from enrichment_screen import directions

GRID = np.linspace(0.0, 3.9, 300)   # the universal-curve grid
CUTS = (1.0, 1.2, 1.5)
ORDER = (*records.DATASETS, *records.PAAM)


def one(dataset):
  _, split, _, _ = directions(dataset)
  curve = np.asarray(split.identifiable, dtype=float)
  times, _, _, maximum, _ = records.load(dataset)
  tau = times / (maximum * np.sqrt(1064.0 / 101325.0))
  return dataset, np.interp(GRID, tau[: curve.size], curve)


def share(curve, cut):
  return float(np.sum(curve[GRID <= cut] ** 2) / np.sum(curve**2))


def main():
  with records.pool(len(ORDER)) as pool:
    raw = dict(pool.map(one, list(ORDER)))

  print("  share of each curve's squared norm inside the first collapse\n")
  print(f"  {'record':>16s} " + "  ".join(f"{'raw t/tc<' + str(c):>14s}" for c in CUTS)
        + "   " + "  ".join(f"{'centred <' + str(c):>14s}" for c in CUTS))
  out = {}
  for dataset in ORDER:
    curve = raw[dataset]
    centred = curve - curve.mean()
    out[dataset] = {f"raw_{c}": share(curve, c) for c in CUTS} | {
      f"centred_{c}": share(centred, c) for c in CUTS}
    print(f"  {dataset:>16s} " + "  ".join(f"{share(curve, c):13.1%}" for c in CUTS)
          + "   " + "  ".join(f"{share(centred, c):13.1%}" for c in CUTS))

  gel = [raw[d] for d in records.DATASETS]
  print("\n  gelatin only, RAW, t/tc <= 1.2: "
        + ", ".join(f"{share(c, 1.2):.0%}" for c in gel)
        + "   <- compare with the text's 65 to 69 percent")
  print("  gelatin only, CENTRED, t/tc <= 1.2: "
        + ", ".join(f"{share(c - c.mean(), 1.2):.0%}" for c in gel))

  stored = json.load(open(records.HERE / "universal_curve.json"))
  mode = np.asarray(stored["common_mode"], dtype=float)
  print("\n  the common mode of all eight, for comparison: "
        + ", ".join(f"t/tc<{c} {share(mode, c):.1%}" for c in CUTS))
  print(f"  as recorded in universal_curve.json: {stored['share_in_first_collapse']:.1%}")

  out["common_mode"] = {f"raw_{c}": share(mode, c) for c in CUTS}
  json.dump(out, open(records.HERE / "first_collapse_share.json", "w"), indent=1)


if __name__ == "__main__":
  main()
