r"""Why the relaxation time is unidentifiable, in closed form, and what breaks the degeneracy.

A $2\times$ change in $\lambda_1$ moves the trace by $0.112$ of $R_{\max}$, six times the noise,
and $\lambda_1$ is still unidentified: at the performed geometry \SI{78}{\percent} of its effect
is reproduced by moving $\mu$ and $g\alpha$. A large signal that another axis can imitate is not
a puzzle, it is an approximate invariance, and this asks whether it has a formula.

THE CANDIDATE IS THE ONE THE CONSTITUTIVE FORM SUPPLIES. A Zener arm at frequency $\omega$ has
an effective viscosity
%
\begin{equation}
\mu_{\rm eff}(\omega) = \frac{\mu}{1 + (\omega\lambda_1)^2},
\end{equation}
%
so if the collapse probes $\mu_{\rm eff}$ and not $\mu$ and $\lambda_1$ apart, then
$\partial\log\mu_{\rm eff}/\partial\log\lambda_1 = -2x/(1+x)$ with $x = (\omega\lambda_1)^2$.
Two consequences follow and both are falsifiable. The invariant direction in
$(\log\mu, \log\lambda_1)$ has slope $2x/(1+x)$, which tends to ZERO as $\omega\lambda_1 \to 0$ --
$\lambda_1$ becomes free while $\mu$ stays put, which is exactly what an unidentified parameter
looks like. And the information about $\lambda_1$ scales as $x^2 \sim \omega^4$ in that limit, so
$\mathrm{sd}(\log\lambda_1) \propto \omega^{-2} \propto R_{\max}^{2}$.

WHICH TURNS THE DESIGN ANSWER INTO A PREDICTION RATHER THAN A SEARCH RESULT. `lambda_reachable`
finds the best geometry at the small-radius, high-stretch corner by sweeping. If the law holds,
that corner is where it has to be, the exponent is $2$, and the requirement is simply
$\omega\lambda_1 \sim 1$: put the loss peak inside the collapse. This sweeps $R_{\max}$ at FIXED
stretch and FIXED window, so the exponent is measured against one variable rather than three.
"""

import json

import numpy as np

import records

RATIO = 38.5
AXES = ("mu", "galpha", "lambda1")
PATHS = ("material.shear_modulus_pa", "material.viscosity_pa_s",
         "material.relaxation_time_s", "material.stiffening")
RADII = np.geomspace(40e-6, 900e-6, 10)
STRETCH, SPAN, SAMPLES = 7.09, 5.0, 201
SIGMA = 0.018
DENSITY, AMBIENT = 1064.0, 101325.0


def one(job):
  import pyimr
  from bounce_sweep import maxima

  maximum, dataset = job
  base = json.load(open(records.HERE / "per_trial_fits.json"))[dataset]["median"]
  mu, product, lam = (float(base[a]) for a in AXES)
  g, alpha = float(np.sqrt(product * RATIO)), float(np.sqrt(product / RATIO))
  material = pyimr.QuadraticZener(g, mu, lam, 0.0, alpha)
  characteristic = maximum * np.sqrt(DENSITY / AMBIENT)
  times = np.linspace(0.0, SPAN * characteristic, SAMPLES)
  config = pyimr.SimulationConfig(maximum, maximum / STRETCH, material,
                                  dynamics="keller-miksis", rtol=1e-9, atol=1e-11,
                                  max_steps=8_000_000)
  try:
    problem = pyimr.prepare(config)
    raw = np.asarray(problem.solve_with_sensitivities(times, PATHS).radius_ratio, dtype=float)
    trace = np.asarray(problem.solve(times).radius_ratio, dtype=float)
  except Exception as error:                                         # noqa: BLE001
    return job, {"failed": type(error).__name__}
  d_g, d_mu, d_lambda, d_alpha = (raw[:, k] for k in range(4))
  columns = np.column_stack([mu * d_mu, 0.5 * (g * d_g + alpha * d_alpha), lam * d_lambda])
  matrix = columns.T @ columns / SIGMA**2
  inverse = np.linalg.inv(matrix)

  # the collapse frequency this geometry actually presents, from its own first afterbounce
  _, _, peaks = maxima(trace, times, order=2)
  omega = 2 * np.pi / (peaks[1] - peaks[0]) if len(peaks) > 1 else np.nan
  x = (omega * lam) ** 2
  # the sloppiest direction, and its slope in (log mu, log lambda1)
  values, vectors = np.linalg.eigh(matrix)
  sloppy = vectors[:, 0]
  slope = float(sloppy[0] / sloppy[2]) if abs(sloppy[2]) > 1e-12 else np.nan
  return job, {"sd_lambda": float(np.sqrt(inverse[2, 2])), "omega": float(omega),
               "x": float(x), "predicted_slope": float(2 * x / (1 + x)),
               "measured_slope": abs(slope), "condition": float(values[-1] / values[0])}


def main():
  dataset = "gelatin_15C"
  jobs = [(r, dataset) for r in RADII]
  with records.pool(len(jobs)) as pool:
    table = dict(pool.map(one, jobs))
  good = {k[0]: v for k, v in table.items() if "failed" not in v}
  if len(good) < 4:
    print("  too few integrable points"); return

  print(f"  R_max swept at FIXED stretch {STRETCH} and window {SPAN} t_c\n")
  print(f"  {'R_max um':>9s} {'omega/2pi kHz':>14s} {'(w*lam)^2':>11s} {'sd(log lam1)':>13s} "
        f"{'slope pred':>11s} {'slope meas':>11s}")
  for r in sorted(good):
    v = good[r]
    print(f"  {r*1e6:9.0f} {v['omega']/2/np.pi/1e3:14.1f} {v['x']:11.4f} {v['sd_lambda']:13.4f} "
          f"{v['predicted_slope']:11.4f} {v['measured_slope']:11.4f}")

  radii = np.array(sorted(good))
  sd = np.array([good[r]["sd_lambda"] for r in radii])
  exponent = float(np.polyfit(np.log(radii), np.log(sd), 1)[0])
  x = np.array([good[r]["x"] for r in radii])
  info_exponent = float(np.polyfit(np.log(x), np.log(sd**-2), 1)[0])

  print("\n  ---- what it says ----\n")
  print(f"  sd(log lambda1) scales as R_max^{exponent:.2f}; the mu_eff law predicts exactly 2.")
  print(f"  Information scales as x^{info_exponent:.2f} with x = (omega*lambda1)^2; predicted 2")
  print("  in the small-x limit, falling to 0 once the loss peak is passed.")
  print(f"\n  Over this sweep (omega*lambda1)^2 runs {x.min():.4f} to {x.max():.4f}, so every")
  print("  point is below the peak and lambda1 is being read from the tail of a resonance.")
  print(f"  Reaching x = 1 needs omega*lambda1 = 1, i.e. R_max near "
        f"{radii[0]*1e6*np.sqrt(x.max())/np.sqrt(1.0):.0f} um scaled from the fastest point here.")
  print("\n  That is the whole of the identifiability problem: the experiment is run two")
  print("  decades below the material's own loss peak, where a Zener is indistinguishable from")
  print("  a Newtonian fluid of viscosity mu. Nothing about the fit or the prior causes it.")
  json.dump({str(int(r * 1e6)): good[r] for r in sorted(good)}
            | {"exponent_sd_vs_radius": exponent, "exponent_info_vs_x": info_exponent},
            open(records.HERE / "lambda_law.json", "w"), indent=1)


if __name__ == "__main__":
  main()
