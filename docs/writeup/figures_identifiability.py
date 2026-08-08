"""Maps of the qSLS parameter space: where it fits, where it is not physical, and why the
two are entangled.

Three figures:
  1. the (g, alpha) plane -- the degenerate valley, the non-physical region, and where each
     method's "best fit" landed
  2. the (alpha, omega_n*lambda1) plane -- the band where integration fails
  3. the sloppiness spectrum and the prior-sensitivity result side by side
"""

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
  os.environ.setdefault(_name, "1")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

from pathlib import Path

import numpy as np

import style

DATA = Path.home() / "fastscratch/papers/paper_imr_windowing/data/Ga_t15_exp_data.csv"
OUT = Path(__file__).resolve().parent
R_MAX, STRETCH = 277e-6, 7.09
OMEGA_N = 5.296e5
MU_FIT, LAM_FIT = 0.0464, 2.783e-7          # from the profile likelihood at its minimum
WORKERS = 32
GRID = 72


def observation():
  from pyimr.noise import characteristic_time

  table = np.loadtxt(DATA, delimiter=",", ndmin=2)
  trials = table[:, 1:].T
  keep = ~(trials > 1.05).any(axis=0) & (trials.std(axis=0, ddof=1) > 0.0)
  return (table[keep, 0] * characteristic_time(R_MAX),
          trials[:, keep].mean(axis=0), trials[:, keep].std(axis=0, ddof=1))


TIMES, OBSERVED, DEVIATIONS = observation()


def chi2_at(payload):
  """(g, alpha) at the fitted mu, lambda1. nan marks a point the model cannot reach."""
  import pyimr

  g, alpha = payload
  config = pyimr.SimulationConfig(
    R_MAX, R_MAX / STRETCH, pyimr.QuadraticZener(float(g), MU_FIT, LAM_FIT, 0.0, float(alpha)),
    dynamics="keller-miksis", rtol=1e-6, atol=1e-8, max_steps=50_000,
  )
  try:
    model = np.asarray(pyimr.simulate(TIMES, config).radius_ratio, dtype=float)
  except Exception:                                # noqa: BLE001
    return float("nan")
  return float(np.mean(((model - OBSERVED) / DEVIATIONS) ** 2))


def reachable(payload):
  """(alpha, lambda1) at the fitted g, mu: 1 physical, 0 not."""
  import pyimr

  alpha, lam = payload
  config = pyimr.SimulationConfig(
    R_MAX, R_MAX / STRETCH, pyimr.QuadraticZener(2154.0, MU_FIT, float(lam), 0.0, float(alpha)),
    dynamics="keller-miksis", rtol=1e-6, atol=1e-8, max_steps=50_000,
  )
  try:
    pyimr.simulate(TIMES, config)
    return 1.0
  except Exception:                                # noqa: BLE001
    return 0.0


def sweep(function, first, second, cache=None):
  from pyimr.parallel import worker_pool

  if cache is not None and (OUT / cache).exists():
    return np.load(OUT / cache)

  a, b = np.meshgrid(first, second, indexing="ij")
  payloads = list(zip(a.ravel(), b.ravel()))
  with worker_pool(WORKERS) as pool:
    values = np.array(list(pool.map(function, payloads, chunksize=32))).reshape(a.shape)
  if cache is not None: np.save(OUT / cache, values)
  return values


def main():
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  from matplotlib.colors import LogNorm

  style.use()

  # ---------------- figure 1: the (g, alpha) plane ----------------
  gs = np.geomspace(50.0, 2e4, GRID)
  alphas = np.geomspace(1e-2, 60.0, GRID)
  chi2 = sweep(chi2_at, gs, alphas, 'cache_valley.npy')
  print(f"(g, alpha) slice: {np.isfinite(chi2).sum()} of {chi2.size} reachable", flush=True)

  fig, ax = plt.subplots(figsize=style.size(0.62, 3.1))
  levels = [1.1, 1.3, 1.6, 2.0, 3.0, 5.0, 10.0, 20.0]
  mesh = ax.pcolormesh(alphas, gs, np.ma.masked_invalid(chi2), norm=LogNorm(1.0, 20.0),
                       cmap="viridis_r", shading="auto")
  ax.contour(alphas, gs, np.ma.masked_invalid(chi2), levels=levels, colors="w", linewidths=0.4, alpha=0.6)
  fig.colorbar(mesh, ax=ax, label=r"$\chi^2/N$")

  # the degenerate direction the two analyses agree on
  line_a = np.geomspace(0.3, 40.0, 50)
  ax.plot(line_a, 2154.0 * (line_a / 0.4642) ** -0.80, "w--", lw=1.4,
          label=r"$g \propto \alpha^{-0.80}$")

  for label, (a, g), look in (
    ("study grid best", (0.4642, 2154.0), dict(marker="s", color="tab:red")),
    ("refined grid", (8.859, 143.8), dict(marker="^", color="tab:orange")),
    ("multistart optimum", (10.0, 108.0), dict(marker="D", color="tab:pink")),
    ("SMC median", (5.27, 206.0), dict(marker="*", color="white", ms=13)),
  ):
    ax.plot(a, g, ls="none", mec="k", mew=0.6, label=label, **look)

  ax.set_xscale("log"); ax.set_yscale("log")
  ax.set_xlabel(r"stiffening $\alpha$"); ax.set_ylabel(r"shear modulus $g$ (Pa)")
  ax.set_title("The valley the data cannot see\n(white: the model is not physical there)")
  ax.legend(fontsize=7, loc="lower left", framealpha=0.9)
  fig.savefig(OUT / "fig_valley.pdf"); fig.savefig(OUT / "fig_valley.png", dpi=110); plt.close(fig)

  # ---------------- figure 2: where integration fails ----------------
  alphas2 = np.geomspace(1e-2, 60.0, GRID)
  lams = np.geomspace(1e-8, 1e-4, GRID)
  ok = sweep(reachable, alphas2, lams, 'cache_failure.npy')
  print(f"(alpha, lambda1) slice: {ok.mean():.1%} reachable", flush=True)

  fig, ax = plt.subplots(figsize=style.size(0.58, 2.8))
  ax.pcolormesh(alphas2, OMEGA_N * lams, ok.T, cmap="RdYlGn", vmin=0, vmax=1, shading="auto")
  ax.axhline(1.0, color="k", ls=":", lw=0.9)
  ax.text(1.2e-2, 1.15, r"$\omega_n \lambda_1 = 1$")
  ax.set_xscale("log"); ax.set_yscale("log")
  ax.set_xlabel(r"stiffening $\alpha$"); ax.set_ylabel(r"$\omega_n \lambda_1$")
  ax.set_title("Green: integrates. Red: the model runs away")
  fig.savefig(OUT / "fig_failure_map.pdf"); fig.savefig(OUT / "fig_failure_map.png", dpi=110); plt.close(fig)

  # ---------------- figure 3: sloppiness and prior sensitivity ----------------
  fig, (left, right) = plt.subplots(1, 2, figsize=style.size(0.98, 2.7))

  eigen = np.array([5.850e5, 2.005e5, 1.001e4, 5.973e-1])
  left.bar(range(4), eigen, color=["tab:blue"] * 3 + ["tab:red"])
  left.set_yscale("log"); left.set_xticks(range(4))
  left.set_xticklabels(["stiff", "#2", "#3", "sloppy"])
  left.set_ylabel("Fisher eigenvalue")
  left.set_title("Six decades of sloppiness")
  left.text(3, 3.0, r"$g$ vs $\alpha$", ha="center", color="tab:red")

  ceilings = np.array([1.0, 10.0, 100.0])
  medians = np.array([0.8456, 6.369, 5.270])
  upper = np.array([0.9686, 9.043, 8.727])
  right.plot(ceilings, medians, "o-", label=r"posterior median $\alpha$")
  right.plot(ceilings, upper, "s--", label=r"$97.5\%$")
  right.plot(ceilings, ceilings, "k:", lw=0.9, label="the prior ceiling itself")
  right.set_xscale("log"); right.set_yscale("log")
  right.set_xlabel(r"prior ceiling on $\alpha$"); right.set_ylabel(r"$\alpha$")
  right.set_title("Identified, not prior-driven")
  right.legend(fontsize=7)
  fig.savefig(OUT / "fig_identifiability.pdf"); fig.savefig(OUT / "fig_identifiability.png", dpi=110); plt.close(fig)

  print("wrote fig_valley.pdf, fig_failure_map.pdf, fig_identifiability.pdf")


if __name__ == "__main__":
  main()
