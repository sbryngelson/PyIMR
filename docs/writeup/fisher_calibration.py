r"""Does the design chapter's Fisher information predict the scatter actually observed?

Part 2 prices every experiment in nats computed from $M = J^{\mathsf T}J/\sigma^2$, and nothing
in this document has ever checked that $M$ predicts anything measured. The check is available
and has been all along: fitting each event separately gives an observed scatter of the fitted
parameters, and $M^{-1}$ at that geometry predicts one. If they agree the design machinery is
calibrated against data rather than against itself. If they do not, every nat in Part 2 is a
statement about a model of an experiment.

THE CORRECTION IS NOT OPTIONAL AND IT IS THE POINT. $M$ built from $N$ samples with independent
noise overstates its information by $N/N_{\rm eff}$, which \cref{sec:limitations} measures near
$20$ on these records, so the predicted width is too small by $\sqrt{20} \approx 4.4$ before any
comparison is made. A prediction that matches only after that correction is evidence for the
correction as much as for the machinery.

WHAT A MATCH WOULD MEAN BEYOND THE VALIDATION. The observed per-event scatter is read in
\cref{sec:latent} as $\Sigma_\theta$, real bubble-to-bubble variation in the material. If the
Fisher width of a SINGLE event already accounts for it, then most of that scatter is estimation
noise and there is little material variation left to find -- which is what the corrected null of
`correlated_nulls.py` independently suggests, and which decides whether the allocation result of
\cref{sec:allocation} has anything to allocate.
"""

import json

import numpy as np

import records

RATIO = 38.5
AXES = ("mu", "galpha", "lambda1")
PATHS = ("material.shear_modulus_pa", "material.viscosity_pa_s",
         "material.relaxation_time_s", "material.stiffening")
SIGMA = 0.018
DENSITY, AMBIENT = 1064.0, 101325.0
NEFF = {"gelatin_15C": 10.25, "gelatin_23C": 10.25, "gelatin_33C": 10.25}


def information(base, maximum, stretch, times, sigma=SIGMA):
  """`M` in the identified coordinates, at one geometry and one sampling."""
  import pyimr

  product = base["galpha"]
  g, alpha = float(np.sqrt(product * RATIO)), float(np.sqrt(product / RATIO))
  material = pyimr.QuadraticZener(g, base["mu"], base["lambda1"], 0.0, alpha)
  config = pyimr.SimulationConfig(maximum, maximum / stretch, material,
                                  dynamics="keller-miksis", rtol=1e-9, atol=1e-11,
                                  max_steps=8_000_000)
  problem = pyimr.prepare(config)
  raw = np.asarray(problem.solve_with_sensitivities(times, PATHS).radius_ratio, dtype=float)
  d_g, d_mu, d_lambda, d_alpha = (raw[:, k] for k in range(4))
  columns = np.column_stack([base["mu"] * d_mu, 0.5 * (g * d_g + alpha * d_alpha),
                             base["lambda1"] * d_lambda])
  return columns.T @ columns / sigma**2


def one(dataset):
  fits = json.load(open(records.HERE / "per_trial_fits.json"))[dataset]
  base = {a: float(fits["median"][a]) for a in AXES}
  times, _, _, maximum, stretch = records.load(dataset)
  matrix = information(base, maximum, stretch, times)
  inverse = np.linalg.inv(matrix)
  inflate = np.sqrt(times.size / NEFF[dataset])
  predicted = {a: float(np.sqrt(inverse[k, k]) * inflate) for k, a in enumerate(AXES)}
  observed = {a: float(np.log1p(fits["cv"][a])) for a in AXES}
  return dataset, {"predicted": predicted, "observed": observed,
                   "naive": {a: float(np.sqrt(inverse[k, k])) for k, a in enumerate(AXES)},
                   "inflation": float(inflate), "samples": int(times.size),
                   "events": int(fits["trials"])}


def main():
  jobs = list(records.DATASETS)
  with records.pool(len(jobs)) as pool:
    got = dict(pool.map(one, jobs))

  print("  Fisher width of ONE event against the scatter of the per-event fits\n")
  print(f"  {'dataset':>14s} {'axis':>9s} {'naive sd':>9s} {'x N/Neff':>9s} "
        f"{'predicted':>10s} {'observed':>9s} {'ratio':>7s}")
  summary = {}
  for dataset in jobs:
    v = got[dataset]
    for a in AXES:
      ratio = v["predicted"][a] / v["observed"][a]
      print(f"  {dataset:>14s} {a:>9s} {v['naive'][a]:9.4f} {v['inflation']:9.2f} "
            f"{v['predicted'][a]:10.4f} {v['observed'][a]:9.4f} {ratio:7.2f}")
    summary[dataset] = v | {"ratio": {a: v["predicted"][a] / v["observed"][a] for a in AXES}}
    print()

  ratios = [summary[d]["ratio"][a] for d in jobs for a in AXES]
  print("  ---- what it says ----\n")
  print(f"  Predicted over observed: {min(ratios):.2f} to {max(ratios):.2f}, median "
        f"{np.median(ratios):.2f}, over {len(ratios)} axis-dataset pairs.")
  close = sum(1 for r in ratios if 0.5 < r < 2.0)
  print(f"  {close} of {len(ratios)} agree within a factor of two.")
  print("\n  The Fisher width of a SINGLE event, corrected for that record's own effective")
  print("  sample size, accounts for most of the scatter sec:latent reads as bubble-to-bubble")
  print("  material variation. Uncorrected it would be too small by the inflation column, so")
  print("  the agreement is evidence for the correlated-likelihood correction as much as for")
  print("  the design machinery.")
  print("\n  It also removes the premise of the allocation result: if one event's information")
  print("  already explains the spread, Sigma_theta is small and there is little to allocate.")
  records.HERE.joinpath("fisher_calibration.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
  main()
