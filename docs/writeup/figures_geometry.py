"""Pictures for the abstract parts: what the Fisher information is, what an eigenvector
means, and why a gentler collapse separates the parameters.

Three figures.

  1. What a sloppy direction DOES. Move the parameters by the same distance along the
     sloppy and the stiff eigenvector and plot the trajectories. Along the sloppy direction
     the curves lie on top of each other inside the measurement noise; along the stiff one
     they separate immediately. That is the whole meaning of the eigenvalue spectrum, shown
     rather than asserted.

  2. The confidence ellipse. The inverse Fisher information IS the covariance, so its
     ellipse in (log g, log alpha) is the degeneracy drawn to scale, with the eigenvectors
     as its axes and the analytic slope over it.

  3. Why gentler helps. The two terms of Ze = g*alpha*A + g*B against collapse depth. The
     separating term B dies as lambda^-4, which is the design result as a picture.
"""

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
  os.environ.setdefault(_name, "1")

from pathlib import Path

import numpy as np

import style

OUT = Path(__file__).resolve().parent
R_MAX, STRETCH = 277e-6, 7.09
FIT = np.array([204.3, 0.04651, 1.964e-7, 5.301])
LOW = np.array([124.4, 0.04284, 1.482e-7, 2.424])
HIGH = np.array([431.9, 0.05005, 2.325e-7, 9.432])
NAMES = ("g", r"\mu", r"\lambda_1", r"\alpha")
NOISE = 0.018


def trajectory(values, times):
  import pyimr

  g, mu, lam, alpha = (float(v) for v in values)
  config = pyimr.SimulationConfig(R_MAX, R_MAX / STRETCH,
                                  pyimr.QuadraticZener(g, mu, lam, 0.0, alpha),
                                  dynamics="keller-miksis", rtol=1e-7, atol=1e-9, max_steps=200_000)
  try:
    return np.asarray(pyimr.simulate(times, config).radius_ratio, dtype=float)
  except Exception:                                  # noqa: BLE001
    return None


def information(times):
  import pyimr
  from pyimr.inference import InferenceParameter, RadiusObservation, prepare_inference

  config = pyimr.SimulationConfig(R_MAX, R_MAX / STRETCH,
                                  pyimr.QuadraticZener(*FIT[[0, 1, 2]], 0.0, FIT[3]),
                                  dynamics="keller-miksis", rtol=1e-7, atol=1e-9, max_steps=200_000)
  truth = np.asarray(pyimr.simulate(times, config).radius_ratio) * R_MAX
  inference = prepare_inference(
    config, RadiusObservation(times, truth, NOISE * R_MAX),
    tuple(InferenceParameter(f"material.{p}", lo, hi, "log") for p, lo, hi in zip(
      ("shear_modulus_pa", "viscosity_pa_s", "relaxation_time_s", "stiffening"), LOW, HIGH)),
  )
  unit = (np.log(FIT) - np.log(LOW)) / (np.log(HIGH) - np.log(LOW))
  jacobian = np.asarray(inference.jacobian(np.clip(unit, 0, 1)), dtype=float)
  width = np.log(HIGH) - np.log(LOW)
  return (jacobian / width) .T @ (jacobian / width)     # in log-parameter coordinates


def P(x): return 5.0 - x**4 - 4.0 * x
def Q(x): return 0.675 + 0.125 * x**8 + 0.2 * x**5 + x**2 - 2.0 / x


def main():
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  style.use()
  from pyimr.noise import characteristic_time

  times = np.linspace(0.0, 5.0 * characteristic_time(R_MAX), 400)
  matrix = information(times)
  values, vectors = np.linalg.eigh(matrix)
  order = np.argsort(values)
  sloppy, stiff = vectors[:, order[0]], vectors[:, order[-1]]
  base = trajectory(FIT, times)

  # ---- 1. what a sloppy direction does to the data ----
  step = 0.45                                          # the same distance in log-parameters
  fig, axes = plt.subplots(1, 2, figsize=style.size(0.98, 2.6), sharey=True)
  for axis, direction, title in ((axes[0], sloppy, "along the SLOPPY direction"),
                                 (axes[1], stiff, "along the STIFF direction")):
    axis.fill_between(times * 1e6, base - NOISE, base + NOISE, color="0.85",
                      label="measurement noise", zorder=0)
    axis.plot(times * 1e6, base, "k", lw=1.0, label="fitted")
    for sign, colour in ((+1, "tab:red"), (-1, "tab:blue")):
      moved = trajectory(np.exp(np.log(FIT) + sign * step * direction), times)
      if moved is None: continue
      factor = np.exp(sign * step * direction)
      axis.plot(times * 1e6, moved, colour, lw=0.9, alpha=0.9,
                label=rf"$g\times{factor[0]:.2f}$, $\alpha\times{factor[3]:.2f}$")
    axis.set_title(title)
    axis.set_xlabel(r"$t$ ($\mu$s)")
    axis.legend(fontsize=6.5, loc="upper right")
  axes[0].set_ylabel(r"$R/R_{\max}$")
  fig.suptitle("The same change in the parameters, seen by the data two ways", fontsize=10)
  fig.savefig(OUT / "fig_eigenvectors.pdf")
  fig.savefig(OUT / "fig_eigenvectors.png", dpi=110)
  plt.close(fig)

  # ---- 2. the confidence ellipse in (log g, log alpha) ----
  keep = np.array([0, 3])
  covariance = np.linalg.inv(matrix)[np.ix_(keep, keep)]
  w, v = np.linalg.eigh(covariance)
  angle = np.linspace(0, 2 * np.pi, 400)
  circle = np.stack([np.cos(angle), np.sin(angle)])
  ellipse = (v * np.sqrt(w * 5.991)) @ circle          # 95% for two parameters

  long, short = float(np.sqrt(w.max() * 5.991)), float(np.sqrt(w.min() * 5.991))
  fig, (wide, curve) = plt.subplots(1, 2, figsize=style.size(0.98, 2.8))

  # Equal axes are the honest impression and hide everything: a 700:1 ellipse is a line in
  # any projection. So the left panel shows that, and the right measures the same fact along
  # each direction instead of trying to draw it.
  wide.plot(ellipse[1], ellipse[0], "k", lw=1.2)
  span = np.linspace(-long, long, 20)
  wide.plot(span, -0.981 * span, "r--", lw=1.0, label=r"analytic slope $-0.981$")
  wide.set_aspect("equal")
  wide.set_xlabel(r"$\Delta \log \alpha$"); wide.set_ylabel(r"$\Delta \log g$")
  wide.set_title("95% confidence region, true shape")
  wide.text(0.03, 0.03, f"{long:.1f} long, {short:.3f} wide\naxis ratio {long / short:.0f} : 1",
            transform=wide.transAxes)
  wide.legend(fontsize=7, loc="upper right")

  # measured, not the quadratic approximation: solve at perturbed parameters
  steps = np.linspace(-1.0, 1.0, 41)
  for direction, label, colour in ((sloppy, "sloppy direction", "tab:red"),
                                   (stiff, "stiff direction", "tab:blue")):
    misfit = []
    for step_size in steps:
      moved = trajectory(np.exp(np.log(FIT) + step_size * direction), times)
      misfit.append(np.nan if moved is None
                    else float(np.sum(((moved - base) / NOISE) ** 2)))
    curve.plot(steps, misfit, colour, lw=1.3, label=label)
  curve.axhline(3.84, color="0.4", ls=":", lw=0.9)
  curve.text(-0.97, 5.0, r"$\Delta\chi^2 = 3.84$ (95%)", color="0.4")
  curve.set_yscale("symlog", linthresh=1.0)
  curve.set_ylim(bottom=0.0)          # Delta chi^2 cannot be negative; symlog would imply it can
  curve.set_xlabel(r"distance moved in $\log$-parameters")
  curve.set_ylabel(r"$\Delta\chi^2$ against the fit")
  curve.set_title("what the data notice, along each direction")
  curve.legend(fontsize=7)
  fig.savefig(OUT / "fig_ellipse.pdf"); fig.savefig(OUT / "fig_ellipse.png", dpi=110)
  plt.close(fig)

  # ---- 3. why a gentler collapse separates them ----
  lam = np.geomspace(1.05, 6.0, 400)
  alpha = FIT[3]
  A = 1.5 * P(lam) + 2.0 * Q(lam)
  B = -0.5 * P(lam)
  fig, (left, right) = plt.subplots(1, 2, figsize=style.size(0.98, 2.6))
  left.loglog(lam, np.abs(alpha * A), label=r"$\alpha A$  (depends on $g\alpha$)")
  left.loglog(lam, np.abs(B), label=r"$B$  (depends on $g$ alone)")
  left.set_xlabel(r"stretch $\lambda = R_{\rm eq}/R$"); left.set_ylabel("magnitude")
  left.set_title("The two terms of $Z_e$"); left.legend(fontsize=7)
  for axis in (left,):
    axis.set_xticks([1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    axis.set_xticklabels(["1", "1.5", "2", "3", "4", "6"])
    axis.set_xticks([], minor=True)

  right.loglog(lam, np.abs(B) / (alpha * np.abs(A)), "k", label=r"$|B|/(\alpha|A|)$")
  right.loglog(lam, 2.0 / (alpha * lam**4), "r--", lw=1.0, label=r"$2/(\alpha\lambda^4)$")
  for depth, name, colour in ((2.215, "present", "tab:red"), (1.5, "gentler", "tab:green")):
    right.axvline(depth, color=colour, ls=":", lw=1.0)
    right.text(depth * 1.03, 8.0, name, color=colour, rotation=90,
               va="top", ha="left")
  right.set_xlabel(r"stretch $\lambda$"); right.set_ylabel("separating share of the signal")
  right.set_title(r"Separability dies as $\lambda^{-4}$")
  right.legend(fontsize=7, loc="lower left")
  right.set_xticks([1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
  right.set_xticklabels(["1", "1.5", "2", "3", "4", "6"])
  right.set_xticks([], minor=True)
  fig.savefig(OUT / "fig_separability.pdf"); fig.savefig(OUT / "fig_separability.png", dpi=110)
  plt.close(fig)

  print("wrote fig_eigenvectors, fig_ellipse, fig_separability")
  print(f"  eigenvalues {np.array2string(values[order][::-1], precision=3)}")
  print("  sloppy direction: " + "  ".join(f"{n}={sloppy[i]:+.3f}" for i, n in
                                            enumerate(("g", "mu", "lam", "alpha"))))
  print(f"  implied exponent {sloppy[0] / sloppy[3]:+.3f}")


if __name__ == "__main__":
  main()
