"""The design trade-off, drawn two ways.

Left: the design plane. Every design the search evaluated, with the undominated ones
picked out and the three single-criterion optima labelled. The point is that they are
not in the same place.

Right: parallel coordinates over the three criteria. Each line is one front design. A
line that is high everywhere would mean no conflict; lines that cross mean improving one
criterion costs another, and where they cross says which pairs actually fight.
"""

import json
import os
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
SHORT = ("identify\n" r"$g$ vs $\alpha$", "separate\nqSLS/SLS", "separate\nqSLS/qKV")


def main(source=None):
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  path = Path(source or os.environ.get("TRADEOFF_JSON", OUT / "tradeoff.json"))
  record = json.loads(path.read_text())
  points = np.array(record["points"], dtype=float)
  values = np.array(record["values"], dtype=float)
  front = np.array(record["front"], dtype=int)
  feasible = np.all(np.isfinite(values), axis=1)

  plt.rcParams.update({"font.size": 9, "figure.dpi": 160, "savefig.bbox": "tight"})
  fig, (plane, coords) = plt.subplots(1, 2, figsize=(8.4, 3.6))

  plane.scatter(points[feasible, 0] * 1e6, points[feasible, 1], s=16, c="0.75",
                edgecolors="none", label="evaluated", zorder=1)
  if (~feasible).any():
    plane.scatter(points[~feasible, 0] * 1e6, points[~feasible, 1], s=22, c="none",
                  edgecolors="0.55", marker="x", linewidths=0.8, label="will not integrate", zorder=1)
  plane.scatter(points[front, 0] * 1e6, points[front, 1], s=34, c="tab:blue",
                edgecolors="k", linewidths=0.4, label="Pareto front", zorder=3)

  colours = ("tab:red", "tab:green", "tab:purple")
  for index, colour in enumerate(colours):
    column = np.where(feasible, values[:, index], -np.inf)
    best = points[int(np.argmax(column))]
    plane.scatter([best[0] * 1e6], [best[1]], s=95, marker="*", c=colour,
                  edgecolors="k", linewidths=0.5, zorder=4,
                  label=SHORT[index].replace("\n", " "))
  plane.set_xscale("log")
  plane.set_xticks([50, 100, 200, 400, 800, 1200])
  plane.set_xticklabels(["50", "100", "200", "400", "800", "1200"])
  plane.set_xticks([], minor=True)
  plane.set_xlabel(r"$R_{\max}$ ($\mu$m)")
  plane.set_ylabel(r"stretch $R_{\max}/R_{\rm eq}$")
  plane.set_title("Where the criteria want to be", fontsize=9)
  plane.legend(fontsize=6, loc="upper right", framealpha=0.9)

  # parallel coordinates, each criterion scaled to its own range over the front
  band = values[front]
  spread = np.ptp(band, axis=0)
  spread[spread <= 0.0] = 1.0
  scaled = (band - band.min(axis=0)) / spread
  axis = np.arange(3)
  order = np.argsort(points[front, 0])
  shades = plt.cm.viridis(np.linspace(0, 1, len(order)))
  for row, colour in zip(scaled[order], shades):
    coords.plot(axis, row, color=colour, lw=1.0, alpha=0.85)
  for position in axis:
    coords.axvline(position, color="0.8", lw=0.8, zorder=0)
  coords.set_xticks(axis)
  coords.set_xticklabels(SHORT, fontsize=7.5)
  coords.set_ylabel("share of the range achieved")
  coords.set_ylim(-0.05, 1.05)
  coords.set_title("Every front design gives something up", fontsize=9)

  bar = fig.colorbar(plt.cm.ScalarMappable(
    norm=plt.Normalize(points[front, 0].min() * 1e6, points[front, 0].max() * 1e6),
    cmap="viridis"), ax=coords, pad=0.02)
  bar.set_label(r"$R_{\max}$ ($\mu$m)", fontsize=7.5)
  bar.ax.tick_params(labelsize=7)

  fig.savefig(OUT / "fig_tradeoff.pdf")
  fig.savefig(OUT / "fig_tradeoff.png", dpi=110)
  print(f"wrote fig_tradeoff from {path} ({len(front)} front designs of {feasible.sum()} feasible)")


if __name__ == "__main__":
  main()
