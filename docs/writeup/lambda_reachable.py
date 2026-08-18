r"""Is the relaxation time measurable by ANY inertial microcavitation experiment?

Two observables have now failed on it from opposite ends. `glassy_and_spectrum.py` finds
$\lambda_1$ has no effect on collapse timing across a sixteenfold change, so the trace does not
constrain it. `disentangle.py` and `nonlinear_loss.py` find the afterbounce loss curve is an
amplitude dependence that a radiating bubble with a LINEAR medium reproduces, so the
afterbounces do not constrain it either. Both are statements about the experiments that happen
to have been run.

THE QUESTION IS WHETHER ANY REACHABLE EXPERIMENT DOES, and it is answerable rather than
rhetorical. The Fisher information for $\log\lambda_1$, profiled over the other identified
coordinates so that no part of its effect is one another axis can absorb, is a function of the
geometry, the stretch and the observation window. Sweeping those gives the best posterior width
this technique can reach on $\lambda_1$ at a stated budget, and the sweep is over a space
containing every setting these datasets used and a great deal they did not.

WHAT THE ANSWER LOOKS LIKE EITHER WAY. If some corner of the space resolves $\lambda_1$, this
names it, and the design chapter's machinery turns that corner into a certified batch. If none
does, then $\lambda_1$ is not measurable by inertial microcavitation at any geometry in the
reachable set, and every relaxation time reported from an IMR fit is a statement about the prior
box rather than about the material. That is a stronger and more useful claim than another
rejection, and it is falsifiable by anyone who finds a geometry this grid missed.

THE PROFILED WIDTH IS THE ONLY HONEST ONE. A marginal variance that ignores $\mu$ and $g\alpha$
would report $\lambda_1$ as well determined wherever it merely moves the trace; the Schur
complement asks what is left after the other axes have absorbed everything they can, which is
the quantity that decides whether a number can be reported.
"""

import json

import numpy as np

import records

RATIO = 38.5
AXES = ("mu", "galpha", "lambda1")
PATHS = ("material.shear_modulus_pa", "material.viscosity_pa_s",
         "material.relaxation_time_s", "material.stiffening")
RADII = np.geomspace(50e-6, 1200e-6, 8)
STRETCH = np.linspace(3.0, 20.0, 6)
SPANS = (2.0, 5.0, 10.0, 20.0)          # observation window, in collapse times
SAMPLES = 201                            # a fixed frame budget, which is the real constraint
SIGMA = 0.018                            # the design chapter's noise level, in R_max
DENSITY, AMBIENT = 1064.0, 101325.0
TARGET = np.log(2.0)                     # "resolved" means a posterior narrower than a factor 2
# the record's own effective sample size: M built on N independent samples overstates its
# information by N/N_eff, so every width below is inflated by the square root of that before
# it is quoted. `fisher_calibration.py` shows the corrected width predicts measured scatter.
INFLATE = np.sqrt(201.0 / 10.25)


def jacobian(base, maximum, stretch, span):
  """`d(R/R_max) / d log(mu, galpha, lambda1)` at one design point."""
  import pyimr

  product = base["galpha"]
  g, alpha = float(np.sqrt(product * RATIO)), float(np.sqrt(product / RATIO))
  material = pyimr.QuadraticZener(g, base["mu"], base["lambda1"], 0.0, alpha)
  characteristic = maximum * np.sqrt(DENSITY / AMBIENT)
  times = np.linspace(0.0, span * characteristic, SAMPLES)
  config = pyimr.SimulationConfig(maximum, maximum / stretch, material,
                                  dynamics="keller-miksis", rtol=1e-9, atol=1e-11,
                                  max_steps=2_000_000)
  problem = pyimr.prepare(config)
  raw = np.asarray(problem.solve_with_sensitivities(times, PATHS).radius_ratio, dtype=float)
  d_g, d_mu, d_lambda, d_alpha = (raw[:, k] for k in range(4))
  # g and alpha are not separately identified; the product is. Chain through it.
  return np.column_stack([base["mu"] * d_mu,
                          0.5 * (g * d_g + alpha * d_alpha),
                          base["lambda1"] * d_lambda])


def one(job):
  maximum, stretch, span, dataset = job
  base = json.load(open(records.HERE / "per_trial_fits.json"))[dataset]["median"]
  base = {a: float(base[a]) for a in AXES}
  try:
    columns = jacobian(base, maximum, stretch, span)
  except Exception as error:                                         # noqa: BLE001
    return job, {"failed": type(error).__name__}
  if not np.all(np.isfinite(columns)): return job, {"failed": "non-finite"}
  information = columns.T @ columns / SIGMA**2
  # profiled: what is left for lambda1 after mu and g*alpha absorb everything they can
  try:
    profiled = float(1.0 / np.linalg.inv(information)[2, 2])
  except np.linalg.LinAlgError:
    return job, {"failed": "singular"}
  marginal = float(information[2, 2])
  return job, {"profiled_sd": float(INFLATE / np.sqrt(profiled)) if profiled > 0 else float("inf"),
               "marginal_sd": float(INFLATE / np.sqrt(marginal)) if marginal > 0 else float("inf"),
               "absorbed": float(1.0 - profiled / marginal) if marginal > 0 else float("nan"),
               "condition": float(np.linalg.cond(information))}


def main():
  dataset = "gelatin_15C"
  jobs = [(r, s, span, dataset) for r in RADII for s in STRETCH for span in SPANS]
  print(f"  {len(jobs)} design points: R_max {RADII[0]*1e6:.0f}-{RADII[-1]*1e6:.0f} um, "
        f"stretch {STRETCH[0]:.0f}-{STRETCH[-1]:.0f}, window {SPANS[0]:.0f}-{SPANS[-1]:.0f} t_c\n")
  with records.pool(min(len(jobs), 64)) as pool:
    table = dict(pool.map(one, jobs))

  good = {k: v for k, v in table.items() if "failed" not in v}
  print(f"  {len(good)} of {len(jobs)} integrable\n")
  if not good:
    print("  nothing integrable; the grid or the base material is wrong"); return

  print("  best posterior sd on log(lambda1) from ONE bubble, by window\n")
  print(f"  {'window':>8s} {'best sd':>10s} {'at R_max':>10s} {'stretch':>8s} "
        f"{'absorbed':>9s} {'bubbles for a factor of 2':>26s}")
  summary = {}
  for span in SPANS:
    here = {k: v for k, v in good.items() if k[2] == span}
    if not here: continue
    best = min(here, key=lambda k: here[k]["profiled_sd"])
    sd = here[best]["profiled_sd"]
    needed = max(1.0, np.ceil((sd / TARGET) ** 2))
    print(f"  {span:8.0f} {sd:10.3f} {best[0]*1e6:10.0f} {best[1]:8.1f} "
          f"{here[best]['absorbed']:9.1%} {needed:26.0f}")
    summary[f"span_{span:g}"] = {"best_sd": sd, "r_max_um": best[0] * 1e6,
                                 "stretch": best[1], "absorbed": here[best]["absorbed"],
                                 "bubbles_for_factor_two": float(needed)}

  overall = min(good, key=lambda k: good[k]["profiled_sd"])
  sd = good[overall]["profiled_sd"]
  print("\n  ---- what it says ----\n")
  print(f"  The best design anywhere on this grid gives sd(log lambda1) = {sd:.3f} from one")
  print(f"  bubble, at R_max {overall[0]*1e6:.0f} um, stretch {overall[1]:.1f}, "
        f"window {overall[2]:.0f} t_c.")
  print(f"  Other axes absorb {good[overall]['absorbed']:.1%} of what lambda1 does there.")
  print(f"  Reaching a factor of two needs {max(1, np.ceil((sd/TARGET)**2)):.0f} bubble(s) there;")
  print(f"  reaching ten percent needs {max(1, np.ceil((sd/0.1)**2)):.0f}.")
  worst = max(good, key=lambda k: good[k]["profiled_sd"])
  print(f"\n  For contrast the worst integrable point gives {good[worst]['profiled_sd']:.1f},")
  print("  and the performed geometry sits at R_max 277 um, stretch 7.1, window 5.")
  summary["best"] = {"sd": sd, "r_max_um": overall[0] * 1e6, "stretch": overall[1],
                     "span": overall[2], "bubbles_for_factor_two": float((sd / TARGET) ** 2),
                     "bubbles_for_ten_percent": float((sd / 0.1) ** 2)}
  records.HERE.joinpath("lambda_reachable.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
  main()
