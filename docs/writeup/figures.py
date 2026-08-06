"""Figures for the measured-data model comparison, drawn only from `results.json`.

Run `collect.py` first. Nothing here re-runs a study or hard-codes a number, so a figure
cannot drift from the run it claims to describe.

Run: .venv/bin/python docs/writeup/figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = json.loads((HERE / "results.json").read_text())
LABEL = {"gelatin_15C": "15 $^\\circ$C", "gelatin_23C": "23 $^\\circ$C", "gelatin_33C": "33 $^\\circ$C"}
ORDER = list(LABEL)

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 200,
                     "axes.spines.top": False, "axes.spines.right": False})


def records():
  """The three measured traces, mean and trial spread."""
  fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3), sharey=True)
  for ax, name in zip(axes, ORDER, strict=True):
    d = RESULTS[name]
    t = np.array(d["times_s"]) * 1e6
    mean, spread = np.array(d["mean"]), np.array(d["spread"])
    ax.fill_between(t, mean - spread, mean + spread, alpha=0.3, lw=0, color="C0")
    ax.plot(t, mean, lw=1.0, color="C0")
    ax.set_title(f"{LABEL[name]}  ({d['trials']} trials)")
    ax.set_xlabel(r"$t$ [$\mu$s]")
  axes[0].set_ylabel(r"$R/R_{\max}$")
  fig.tight_layout()
  fig.savefig(HERE / "fig_records.pdf")


def fits():
  """Best chi-squared per sample for every candidate, all three datasets.

  This is the best GRID NODE, not the best fit: refitting qSLS at 15 C by Gauss-Newton
  reaches 0.962 against the grid's 1.35. The axis says so, because the document reports
  both numbers and a reader must know which is which.
  """
  names = [n for n in RESULTS[ORDER[0]]["models"]]
  names.sort(key=lambda n: RESULTS[ORDER[0]]["models"][n]["chi2_per_sample"])
  fig, ax = plt.subplots(figsize=(7.2, 3.0))
  width = 0.27
  for i, dataset in enumerate(ORDER):
    vals = [RESULTS[dataset]["models"].get(n, {}).get("chi2_per_sample", np.nan) for n in names]
    ax.bar(np.arange(len(names)) + (i - 1) * width, vals, width, label=LABEL[dataset])
  ax.axhline(1.0, color="k", lw=0.8, ls="--")
  ax.text(len(names) - 0.4, 1.05, "measurement noise", ha="right", fontsize=7)
  ax.set_yscale("log")
  ax.set_xticks(np.arange(len(names)), names, rotation=45, ha="right")
  ax.set_ylabel(r"best $\chi^2/N$ on the selection grid")
  ax.legend(frameon=False, ncols=3)
  fig.tight_layout()
  fig.savefig(HERE / "fig_fits.pdf")


def trend():
  """How much the stiffening term buys, against temperature."""
  fig, ax = plt.subplots(figsize=(3.5, 2.6))
  temps = [15, 23, 33]
  q = [RESULTS[n]["models"]["qSLS"]["chi2_per_sample"] for n in ORDER]
  s = [RESULTS[n]["models"]["SLS"]["chi2_per_sample"] for n in ORDER]
  ax.plot(temps, s, "o-", label="SLS (relaxation only)")
  ax.plot(temps, q, "s-", label="qSLS (+ stiffening)")
  ax.axhline(1.0, color="k", lw=0.8, ls="--")
  ax.set_xlabel(r"temperature [$^\circ$C]")
  ax.set_ylabel(r"best $\chi^2/N$ on the selection grid")
  ax.set_xticks(temps)
  ax.legend(frameon=False, fontsize=8)
  fig.tight_layout()
  fig.savefig(HERE / "fig_trend.pdf")


def winner():
  """The winning model against the data it was selected on."""
  fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3), sharey=True)
  for ax, name in zip(axes, ORDER, strict=True):
    d = RESULTS[name]
    t = np.array(d["times_s"]) * 1e6
    mean, spread = np.array(d["mean"]), np.array(d["spread"])
    ax.fill_between(t, mean - spread, mean + spread, alpha=0.3, lw=0, color="C0", label="measured")
    ax.plot(t, np.array(d["models"]["qSLS"]["best_trace"]), lw=1.0, color="C3", label="qSLS")
    ax.set_title(f"{LABEL[name]}  grid $\\chi^2/N={d['models']['qSLS']['chi2_per_sample']:.2f}$")
    ax.set_xlabel(r"$t$ [$\mu$s]")
  axes[0].set_ylabel(r"$R/R_{\max}$")
  axes[0].legend(frameon=False, fontsize=8)
  fig.tight_layout()
  fig.savefig(HERE / "fig_winner.pdf")


def coverage():
  """What fraction of each model's grid could not be integrated."""
  names = sorted(RESULTS[ORDER[0]]["models"], key=lambda n: -RESULTS[ORDER[0]]["models"][n]["dropped"])
  names = [n for n in names if any(RESULTS[d]["models"].get(n, {}).get("dropped", 0) for d in ORDER)]
  fig, ax = plt.subplots(figsize=(3.5, 2.6))
  width = 0.27
  for i, dataset in enumerate(ORDER):
    frac = [100.0 * RESULTS[dataset]["models"].get(n, {}).get("dropped", 0)
            / max(1, RESULTS[dataset]["models"].get(n, {}).get("points", 1)) for n in names]
    ax.bar(np.arange(len(names)) + (i - 1) * width, frac, width, label=LABEL[dataset])
  ax.set_xticks(np.arange(len(names)), names, rotation=45, ha="right")
  ax.set_ylabel("grid points dropped [%]")
  ax.legend(frameon=False, fontsize=8)
  fig.tight_layout()
  fig.savefig(HERE / "fig_coverage.pdf")


if __name__ == "__main__":
  for build in (records, fits, trend, winner, coverage):
    build()
    print(f"  {build.__name__}", flush=True)
