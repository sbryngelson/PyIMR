"""What the trial-to-trial spread is, and what it does not explain.

Two panels, making one argument each.

  LEFT. The singular spectrum of the trial deviations against what independent measurement
  error would give. If the eighteen trials differed only by noise the spectrum would be
  flat at 1/rank; instead one component carries more than half the variance, and projecting
  onto the span of dR/dtheta recovers a share far above chance. The spread is largely the
  parameters differing between bubbles.

  RIGHT. The autocorrelation of the qSLS fit residual, before and after whitening by the
  hierarchical covariance that the left panel justifies. If the correlation were caused by
  parameter variation, whitening would flatten it into the white-noise band. It does not:
  about a third goes and the rest stays. What remains is structure the model lacks, which
  is a different problem with a different remedy, and the panel is here because that
  distinction is easy to assert and hard to believe without seeing it.
"""

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
  os.environ.setdefault(_name, "1")

import json
from pathlib import Path

import numpy as np

import style

HERE = Path(__file__).resolve().parent
DATA = Path.home() / "fastscratch/papers/paper_imr_windowing/data/Ga_t15_exp_data.csv"
R_MAX, STRETCH = 277e-6, 7.09
FIT = (204.3, 0.04651, 1.964e-7, 5.301)
PATHS = ("R0", "Req", "material.shear_modulus_pa", "material.viscosity_pa_s",
         "material.relaxation_time_s", "material.stiffening")
VALUES = np.array([R_MAX, R_MAX / STRETCH, FIT[0], FIT[1], FIT[2], FIT[3]])
LAGS = 40


def autocorrelation(series, lags=LAGS):
  centred = series - series.mean()
  denominator = float(centred @ centred)
  return np.array([float(centred[:len(centred) - k] @ centred[k:]) / denominator
                   for k in range(lags + 1)])


def main():
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  import pyimr

  style.use()
  record = json.loads((HERE / "results.json").read_text())["gelatin_15C"]
  times = np.array(record["times_s"])
  mean, spread = np.array(record["mean"]), np.array(record["spread"])
  keep = spread > 0
  times, mean, spread = times[keep], mean[keep], spread[keep]
  trials = record["trials"]

  table = np.loadtxt(DATA, delimiter=",", ndmin=2)
  raw = table[:, 1:].T
  good = ~(raw > 1.05).any(axis=0) & (raw.std(axis=0, ddof=1) > 0.0)
  selected = raw[:, good]
  deviations = (selected - selected.mean(axis=0))[:, :len(times)]

  config = pyimr.SimulationConfig(R_MAX, R_MAX / STRETCH,
                                  pyimr.QuadraticZener(FIT[0], FIT[1], FIT[2], 0.0, FIT[3]),
                                  dynamics="keller-miksis", rtol=1e-9, atol=1e-11, max_steps=400_000)
  problem = pyimr.prepare(config)
  model = np.asarray(problem.solve(times).radius_ratio, dtype=float)
  sensitivity = np.asarray(problem.solve_with_sensitivities(times, PATHS).radius_ratio,
                           dtype=float) * VALUES

  basis, _ = np.linalg.qr(sensitivity)
  explained = deviations @ basis @ basis.T
  leftover = deviations - explained
  share = 1.0 - (leftover**2).sum() / (deviations**2).sum()
  singular = np.linalg.svd(deviations, compute_uv=False)
  energy = singular**2 / np.sum(singular**2)
  flat = 1.0 / min(deviations.shape[0] - 1, deviations.shape[1])

  low_rank = explained.T @ explained / (deviations.shape[0] - 1)
  noise = float((leftover**2).sum() / leftover.size)
  residual = model - mean
  covariance = (low_rank + noise * np.eye(len(times))) / trials + 1e-14 * np.eye(len(times))
  whitened = np.linalg.solve(np.linalg.cholesky(covariance), residual)

  fig, (spectrum, correlation) = plt.subplots(1, 2, figsize=style.size(1.0, 2.6))

  count = min(10, energy.size)
  spectrum.bar(np.arange(1, count + 1), energy[:count], color=style.PALETTE[0], width=0.68)
  spectrum.axhline(flat, color=style.PALETTE[1], ls="--", lw=1.0)
  spectrum.text(count * 0.55, flat * 1.5, "independent noise", color=style.PALETTE[1])
  spectrum.annotate(f"{energy[0]:.0%}", (1, energy[0]), textcoords="offset points",
                    xytext=(0, 3), ha="center", color=style.PALETTE[0])
  spectrum.set_yscale("log")
  spectrum.set_xticks(np.arange(1, count + 1))
  spectrum.set_xlabel("component")
  spectrum.set_ylabel("share of trial-to-trial variance")
  spectrum.set_title(f"the spread is low rank, and {share:.0%} of it\n"
                     r"lies in span $\partial R/\partial\theta$")

  band = 2.0 / np.sqrt(len(times))
  lags = np.arange(LAGS + 1)
  correlation.axhspan(-band, band, color="0.85", zorder=0)
  correlation.axhline(0.0, color="0.4", lw=0.6)
  correlation.plot(lags, autocorrelation(residual), color=style.PALETTE[1],
                   label=f"as fitted ($\\rho_1 = {autocorrelation(residual)[1]:.2f}$)")
  correlation.plot(lags, autocorrelation(whitened), color=style.PALETTE[0],
                   label=f"hierarchical ($\\rho_1 = {autocorrelation(whitened)[1]:.2f}$)")
  correlation.set_xlabel("lag (samples)")
  correlation.set_ylabel("residual autocorrelation")
  correlation.set_title("whitening removes a third of it,\nnot the structure")
  correlation.legend(loc="upper right")
  correlation.set_xlim(0, LAGS)

  fig.tight_layout()
  fig.savefig(HERE / "fig_trial.pdf")
  fig.savefig(HERE / "fig_trial.png", dpi=200)
  print(f"wrote fig_trial: leading component {energy[0]:.1%} against {flat:.1%} for noise, "
        f"{share:.1%} in the sensitivity span, lag-one {autocorrelation(residual)[1]:.3f} "
        f"-> {autocorrelation(whitened)[1]:.3f}")


if __name__ == "__main__":
  main()
