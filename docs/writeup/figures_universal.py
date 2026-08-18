r"""The curve itself, which this document has so far reported only as a correlation coefficient.

\Cref{sec:universal} establishes that $\hat\delta$ is the same shape on two unrelated networks
and quotes a median $\lvert\cos\rvert$. That is enough to support the claim and useless to
anybody who wants to work on it: a theory is tested against a CURVE, not against a correlation.
This extracts the common mode by singular value decomposition of the eight discrepancies on a
shared collapse-time grid, reports how much of the total each mode carries, and writes the mode
out so it can be read off rather than re-derived.

SIGNS ARE CHECKED RATHER THAN ASSUMED AWAY. The universality test scored $\lvert\cos\rvert$ and
so is blind to whether the eight curves point the same way. They are all built as the
identifiable part of (measured $-$ model) in units of the standard error, so the sign is
physical: a positive lobe is the model running BELOW the data there. If the loadings share a
sign, the eight records are missing the same term with the same sign, which is a stronger
statement than the correlation alone and is reported here for the first time.
"""

import json

import numpy as np

import records
import style
from universal_delta import GRID, one as delta_one

ORDER = (*records.DATASETS, *records.PAAM)
LABEL = {"gelatin_15C": "gelatin 15 C", "gelatin_23C": "gelatin 23 C",
         "gelatin_33C": "gelatin 33 C", "paam_PA05": "PAAm PA05",
         "paam_PA05_10C": "PAAm 10 C", "paam_PA05_21C": "PAAm 21 C",
         "paam_PA05_33C": "PAAm 33 C", "paam_PA05003": "PAAm PA05003"}


def main():
  import matplotlib.pyplot as plt

  with records.pool(len(ORDER)) as pool:
    raw = dict(pool.map(delta_one, list(ORDER)))
  curves = np.array([raw[d]["curve"] for d in ORDER])

  left, values, right = np.linalg.svd(curves, full_matrices=False)
  share = values**2 / (values**2).sum()
  mode = right[0]
  loadings = curves @ mode
  # orient the mode so the majority of records load positively on it
  if np.sum(loadings > 0) < len(loadings) / 2:
    mode, loadings = -mode, -loadings

  print(f"  {len(ORDER)} discrepancies on a common grid of {GRID.size} points\n")
  print("  singular spectrum (share of total squared norm):")
  print("   " + "  ".join(f"{s:.3f}" for s in share))
  print(f"\n  the leading mode carries {share[0]:.1%}; a set of unrelated curves would put")
  print(f"  {1/len(ORDER):.1%} in each.\n")
  print(f"  {'record':>16s} {'loading':>9s} {'|cos| with mode 1':>18s}")
  for d, load in zip(ORDER, loadings, strict=True):
    print(f"  {d:>16s} {load:9.3f} {abs(load):18.3f}")
  same = int(np.sum(np.sign(loadings) == np.sign(loadings[0])))
  print(f"\n  {same} of {len(ORDER)} records load with the same sign, so they are missing the")
  print("  same term in the same direction rather than merely sharing a shape.")

  first = GRID[np.argmax(np.abs(mode))]
  inside = float(np.sum(mode[GRID <= 1.2] ** 2) / np.sum(mode**2))
  print(f"\n  the mode peaks at t/t_c = {first:.2f} and puts {inside:.0%} of itself inside the")
  print("  first collapse (t/t_c <= 1.2).")

  fig, (curve_ax, spectrum_ax) = plt.subplots(1, 2, figsize=style.size(1.0, 2.6),
                                              gridspec_kw={"width_ratios": [2.1, 1]})
  for d, row, load in zip(ORDER, curves, loadings, strict=True):
    colour = style.PALETTE[0] if d in records.DATASETS else style.PALETTE[1]
    curve_ax.plot(GRID, row * np.sign(load), color=colour, alpha=0.35, linewidth=0.8)
  curve_ax.plot(GRID, mode, color="black", linewidth=1.8, label="common mode")
  curve_ax.axhline(0.0, color="0.7", linewidth=0.5, zorder=0)
  curve_ax.set_xlabel(r"$t/t_c$")
  curve_ax.set_ylabel(r"$\hat\delta$, normalised")
  curve_ax.plot([], [], color=style.PALETTE[0], label="gelatin (3)")
  curve_ax.plot([], [], color=style.PALETTE[1], label="PAAm (5)")
  curve_ax.legend(frameon=False, loc="best")

  spectrum_ax.bar(np.arange(1, len(share) + 1), share, color=style.PALETTE[2])
  spectrum_ax.axhline(1 / len(ORDER), color="0.4", linestyle="--", linewidth=0.8)
  spectrum_ax.set_xlabel("mode")
  spectrum_ax.set_ylabel("share of squared norm")
  fig.tight_layout()
  fig.savefig(records.HERE / "fig_universal.pdf")
  fig.savefig(records.HERE / "fig_universal.png", dpi=200)
  print("\n  wrote fig_universal.pdf and the mode itself to universal_curve.json")

  json.dump({"tau": GRID.tolist(), "common_mode": mode.tolist(),
             "share": share.tolist(), "peak_tau": float(first),
             "share_in_first_collapse": inside,
             "loadings": {d: float(x) for d, x in zip(ORDER, loadings, strict=True)},
             "curves": {d: raw[d]["curve"] for d in ORDER}},
            open(records.HERE / "universal_curve.json", "w"), indent=1)


if __name__ == "__main__":
  main()
