"""The equivalence-theorem certificate, drawn.

The claim that a design measure is optimal is not a report that a search stopped improving:
it is the Kiefer-Wolfowitz condition, that the sensitivity

    d(x, xi) = tr[M(xi)^-1 M(x)]

never exceeds the parameter count p anywhere in the design space, and attains p exactly on
the support. Both halves are visible in one picture, which is the point of drawing it: the
surface is capped by a level the optimum touches and never crosses.
"""

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
  os.environ.setdefault(_name, "1")

from pathlib import Path

import numpy as np

import style

OUT = Path(__file__).resolve().parent
FIT = (204.3, 0.04651, 1.964e-7, 5.301)
LOW = np.array([124.4, 0.04284, 1.482e-7, 2.424])
HIGH = np.array([431.9, 0.05005, 2.325e-7, 9.432])
RADII = np.geomspace(50e-6, 1200e-6, 20)
STRETCH = np.linspace(3.0, 20.0, 24)
RELATIVE_NOISE, SAMPLES = 0.018, 201


def information(design):
  """`J^T J` for the four qSLS parameters at one design, or `None` if it will not run."""
  import pyimr
  from pyimr.inference import InferenceParameter, RadiusObservation, prepare_inference
  from pyimr.noise import characteristic_time

  radius, stretch = design
  times = np.linspace(0.0, 5.0 * characteristic_time(radius), SAMPLES)
  material = pyimr.QuadraticZener(FIT[0], FIT[1], FIT[2], 0.0, FIT[3])
  config = pyimr.SimulationConfig(radius, radius / stretch, material, dynamics="keller-miksis",
                                  rtol=1e-7, atol=1e-9, max_steps=200_000)
  try:
    truth = np.asarray(pyimr.simulate(times, config).radius_ratio, dtype=float)
  except Exception:                                  # noqa: BLE001
    return design, None
  inference = prepare_inference(
    config, RadiusObservation(times, truth * radius, RELATIVE_NOISE * radius),
    tuple(InferenceParameter(f"material.{p}", lo, hi, "log") for p, lo, hi in zip(
      ("shear_modulus_pa", "viscosity_pa_s", "relaxation_time_s", "stiffening"), LOW, HIGH)))
  unit = np.clip((np.log(FIT) - np.log(LOW)) / (np.log(HIGH) - np.log(LOW)), 0, 1)
  try:
    jacobian = np.asarray(inference.jacobian(unit), dtype=float)
  except Exception:                                  # noqa: BLE001
    return design, None
  if not np.all(np.isfinite(jacobian)): return design, None
  matrix = jacobian.T @ jacobian
  return design, 0.5 * (matrix + matrix.T)


def main():
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.patheffects as pe
  import matplotlib.pyplot as plt

  from pyimr.measure import optimal_measure
  from pyimr.parallel import worker_pool

  designs = [(float(r), float(s)) for r in RADII for s in STRETCH]
  with worker_pool(20) as pool:
    out = list(pool.map(information, designs))
  points = np.array([d for d, m in out if m is not None])
  stack = np.array([m for _, m in out if m is not None])
  result = optimal_measure(stack, iterations=400_000)

  count = stack.shape[1]
  variance = np.einsum("jk,ikj->i", np.linalg.inv(np.tensordot(result.weights, stack, axes=(0, 0))), stack)
  style.use()
  fig, (plane, profile) = plt.subplots(1, 2, figsize=style.size(1.0, 2.7))

  # the candidates are a regular grid, so draw them as one; triangulated contours of a
  # regular grid produce interpolation speckle that reads as structure and is not.
  shape = (len(RADII), len(STRETCH))
  mesh = variance.reshape(shape) if variance.size == shape[0] * shape[1] else None
  if mesh is not None:
    grid = plane.pcolormesh(RADII * 1e6, STRETCH, mesh.T, cmap="viridis", shading="gouraud")
    plane.contour(RADII * 1e6, STRETCH, mesh.T, levels=[count - 0.02], colors="w",
                  linewidths=1.0, linestyles="--")
  else:
    grid = plane.tricontourf(points[:, 0] * 1e6, points[:, 1], variance, levels=24, cmap="viridis")
  support = result.support
  plane.scatter(points[support, 0] * 1e6, points[support, 1],
                s=40 + 900 * result.weights[support], facecolors="none",
                edgecolors="r", linewidths=1.6, zorder=5)
  # offset each weight away from whichever edge it sits near, or it is clipped by the axes
  span_x = (np.log10(points[:, 0]).min(), np.log10(points[:, 0]).max())
  span_y = (points[:, 1].min(), points[:, 1].max())
  for index in support:
    across = (np.log10(points[index, 0]) - span_x[0]) / (span_x[1] - span_x[0])
    up = (points[index, 1] - span_y[0]) / (span_y[1] - span_y[0])
    plane.annotate(f"{result.weights[index]:.2f}", (points[index, 0] * 1e6, points[index, 1]),
                   textcoords="offset points", color="r",
                   xytext=(-22 if across > 0.7 else 8, -11 if up > 0.8 else 7),
                   path_effects=[pe.withStroke(linewidth=1.8, foreground="w")])
  plane.set_xscale("log")
  plane.set_xticks([50, 100, 200, 400, 800, 1200])
  plane.set_xticklabels(["50", "100", "200", "400", "800", "1200"])
  plane.set_xticks([], minor=True)
  plane.set_xlabel(r"$R_{\max}$ ($\mu$m)")
  plane.set_ylabel(r"stretch $R_{\max}/R_{\rm eq}$")
  plane.set_title(r"sensitivity $d(x,\xi^\star)$, support circled")
  fig.colorbar(grid, ax=plane, pad=0.02)

  order = np.argsort(variance)
  profile.plot(np.arange(len(order)), variance[order], "k", lw=1.2)
  profile.axhline(count, color="r", ls="--", lw=1.0)
  profile.set_ylim(top=count * 1.13)
  profile.text(0.02, 0.965, rf"$p = {count}$: the bound", color="r",
               transform=profile.transAxes, va="top")
  rank = {int(i): k for k, i in enumerate(order)}
  profile.scatter([rank[int(i)] for i in support], variance[support], s=26, c="r", zorder=4,
                  label="support: attains the bound")
  profile.set_xlabel("designs, ordered by sensitivity")
  profile.set_ylabel(r"$d(x,\xi^\star)$")
  profile.set_title("no design exceeds the bound")
  profile.legend(fontsize=7, loc="lower right")

  fig.savefig(OUT / "fig_certificate.pdf")
  fig.savefig(OUT / "fig_certificate.png", dpi=110)
  print(f"wrote fig_certificate: gap {result.gap:.2e}, max d = {variance.max():.8f} against p = {count}, "
        f"{len(support)} support points, certified={result.certified}")


if __name__ == "__main__":
  main()
