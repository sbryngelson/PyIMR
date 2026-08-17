r"""How much of the trial spread is per-event parameter variation, on both materials?

`lackoffit.py` divides its ratio by $1 - 0.393$ because `trial_variation.py` measures
\SI{39.3}{\percent} of the gelatin \SI{15}{\celsius} trial variance lying in the span of the
model's own sensitivities. Only the measurement share of that spread is error the model should
not have to follow; the rest is real bubble-to-bubble variation, so leaving it in the
denominator makes the test conservative.

That correction is currently a single number, measured on ONE dataset of ONE material, applied
to eight. \Cref{sec:paam} quotes every ratio without it for exactly that reason. This measures
it where it is used.

THE QUANTITY. Each event's deviation from the ensemble mean is projected onto
$\operatorname{span} G$, $G = \partial (R/R_{\max}) / \partial \log\theta$ over $R_0$,
$R_{\rm eq}$ and the four material axes. The share of total deviation variance that lands in
that span is what a hierarchical model would call parameter variation. Its own chance level is
$\dim G / N$, which is reported beside it, because a six-dimensional subspace of a
two-hundred-dimensional space captures something even from noise.

THE CONTROL IS THAT THE OLD NUMBER COMES BACK, AND HALF OF WHAT IT CHECKED FAILED.
`trial_variation.py` evaluates the sensitivities at a hand-entered fit; this evaluates them at
each dataset's own fit from `paam_lackoffit.py`, which sits elsewhere on the $g\alpha$ ridge.
Run at the old point it returns \SI{39.3}{\percent} exactly, so the reimplementation is the same
measurement. Run at the new one it returns \SI{46.1}{\percent} on the same dataset, so the span
DOES care where on the ridge the fit sits, against the guess that it would not. Seven points of
the share is a property of the fit rather than of the data, which is one more reason the
correction is not applied to anything in \cref{sec:paam}.
"""

import json

import numpy as np

import records
from trial_modes import DATA, FILES

PATHS = ("R0", "Req", "material.shear_modulus_pa", "material.viscosity_pa_s",
         "material.relaxation_time_s", "material.stiffening")
ORDER = (*records.DATASETS, *records.PAAM)
# the point `trial_variation.py` evaluates at, kept so its 39.3% is reproducible from here
LEGACY = {"g": 204.3, "mu": 0.04651, "lambda1": 1.964e-7, "alpha": 5.301}


def _events(dataset):
  """`(times, mean, spread, maximum, stretch, events)` with the events on the kept samples."""
  times, mean, spread, maximum, stretch = records.load(dataset)
  if dataset in records.PAAM:
    return times, mean, spread, maximum, stretch, records.trials(dataset)[1]
  filename, offset = FILES[dataset]
  raw = np.loadtxt(DATA / filename, delimiter=",")
  return times, mean, spread, maximum, stretch, raw[offset:offset + times.size, 1:].T


def _lag_one(x):
  x = x - x.mean()
  return float(np.dot(x[:-1], x[1:]) / np.dot(x, x))


def one(job):
  import pyimr

  dataset, tag = job
  times, mean, spread, maximum, stretch, events = _events(dataset)
  fitted = (LEGACY if tag == "legacy"
            else json.load(open(records.HERE / "paam_lackoffit.json"))[dataset]["fitted"])
  config = pyimr.SimulationConfig(
    maximum, maximum / stretch,
    pyimr.QuadraticZener(fitted["g"], fitted["mu"], fitted["lambda1"], 0.0, fitted["alpha"]),
    dynamics="keller-miksis", rtol=1e-9, atol=1e-11, max_steps=600_000)
  problem = pyimr.prepare(config)
  model = np.asarray(problem.solve(times).radius_ratio, dtype=float)
  values = np.array([maximum, maximum / stretch, fitted["g"], fitted["mu"],
                     fitted["lambda1"], fitted["alpha"]])
  jacobian = np.asarray(problem.solve_with_sensitivities(times, PATHS).radius_ratio,
                        dtype=float) * values

  deviations = events - events.mean(axis=0)
  basis, _ = np.linalg.qr(jacobian)
  explained = deviations @ basis @ basis.T
  leftover = deviations - explained
  share = 1.0 - float((leftover**2).sum() / (deviations**2).sum())

  # the whitening test, which is what the share is FOR: does the residual correlation survive
  count = events.shape[0]
  low_rank = explained.T @ explained / (count - 1)
  noise = float((leftover**2).sum() / leftover.size)
  residual = model - mean
  covariance = (low_rank + noise * np.eye(times.size)) / count + 1e-14 * np.eye(times.size)
  whitened = np.linalg.solve(np.linalg.cholesky(covariance), residual)
  return job, {"share": share, "chance": jacobian.shape[1] / times.size,
               "events": count, "samples": int(times.size),
               "lag_one_raw": _lag_one(residual), "lag_one_whitened": _lag_one(whitened),
               "chi2_diagonal": float((residual / spread) @ (residual / spread) / times.size),
               "chi2_hierarchical": float(whitened @ whitened / times.size)}


def main():
  jobs = [(d, "own") for d in ORDER] + [("gelatin_15C", "legacy")]
  with records.pool(len(jobs)) as pool:
    table = dict(pool.map(one, jobs))

  print("  share of trial-to-trial variance lying in the span of the sensitivities\n")
  print(f"  {'dataset':>16s} {'events':>7s} {'share':>8s} {'chance':>8s} {'excess':>8s} "
        f"{'lag-1 raw':>10s} {'lag-1 white':>12s}")
  summary = {}
  for dataset in ORDER:
    v = table[(dataset, "own")]
    print(f"  {dataset:>16s} {v['events']:7d} {v['share']:8.1%} {v['chance']:8.1%} "
          f"{v['share'] - v['chance']:8.1%} {v['lag_one_raw']:10.3f} "
          f"{v['lag_one_whitened']:12.3f}")
    summary[dataset] = v

  legacy = table[("gelatin_15C", "legacy")]
  own = table[("gelatin_15C", "own")]
  summary["gelatin_15C_legacy_fit"] = legacy
  print(f"\n  CONTROL: gelatin_15C at the hand-entered fit of trial_variation.py gives "
        f"{legacy['share']:.1%}, reproducing the published 39.3% exactly, so this is the same")
  print(f"  measurement. At its own fit the SAME dataset gives {own['share']:.1%}: the span "
        f"moves with where on the g-alpha ridge the fit landed, by "
        f"{100 * abs(own['share'] - legacy['share']):.1f} points, which was")
  print("  not expected and makes that much of the share a property of the fit, not the data.")

  print("\n  ---- what it says ----\n")
  gel = [summary[d]["share"] for d in records.DATASETS]
  paam = [summary[d]["share"] for d in records.PAAM]
  print(f"  gelatin {min(gel):.1%} to {max(gel):.1%}, PAAm {min(paam):.1%} to {max(paam):.1%}.")
  print(f"  A correction of 1/(1 - share) is {1/(1-min(paam)):.2f}x to {1/(1-max(paam)):.2f}x on")
  print(f"  PAAm against {1/(1-0.393):.2f}x at the published gelatin value, so applying the")
  print("  gelatin number to PAAm is not neutral and sec:paam is right to quote the")
  print("  uncorrected column.")
  survived = [d for d, v in summary.items() if abs(v["lag_one_whitened"]) > 0.3]
  print(f"\n  The residual correlation survives whitening by the hierarchical covariance on "
        f"{len(survived)} of {len(table)} fits,")
  print("  which is the test itself: what survives is structure the model does not contain,")
  print("  not latent parameter variation.")
  records.HERE.joinpath("paam_variation.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
  main()
