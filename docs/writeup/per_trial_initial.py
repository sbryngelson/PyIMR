r"""The gap in the last test: `Req` was never free.

`per_trial_fits.py` fitted every trial on its own and found the dominant mode surviving with
its shape intact, and \cref{sec:latent} now says the scatter is not the parameters of this
model. That statement is too strong, and the reason is in the axes it fitted:
$(\mu, g\alpha, \lambda_1)$. Those are MATERIAL parameters. The equilibrium radius was held at
$R_{\max}/\text{stretch}$, one number for every trial in a record, so no amount of refitting
could have absorbed variation in it.

That matters more than a missing axis usually would, because of WHERE the mode lives. It is
$97$ to $98\%$ post-collapse, and the afterbounce period is set by the equilibrium radius --
the gas content the bubble is left with. Nucleation is not repeatable to better than a few
percent, so $R_{\rm eq}$ is exactly the quantity one would expect to vary between bubbles, and
exactly the one whose variation shows up after the first collapse rather than during it.

So this refits each trial with `req_scale` free alongside the material axes, over the
$\pm 11\%$ its prior box already allows. The question is unchanged: does giving each trial its
own value absorb the leading mode.

It also tests the reading in the other direction. `trial_modes.py` puts $R_{\rm eq}$ in its
nine linearised directions and finds the mode correlating at $0.467$, $0.109$ and $0.181$ --
which is a statement about the derivative at one point, and says nothing about what a genuine
refit reaches when the parameter moves by ten percent.
"""

import json

import numpy as np

import records
from identified import BOX
from per_trial_fits import _trials

RATIO = 38.5
AXES = ("mu", "galpha", "lambda1")
STARTS, EVALUATIONS = 8, 300


def candidate_with_req(ratio=RATIO):
  """The identified qSLS, with the equilibrium radius as a fitted configuration axis."""
  import pyimr
  from pyimr.selection import CandidateModel

  def build(t):
    product = t["galpha"]
    return pyimr.QuadraticZener(float(np.sqrt(product * ratio)), t["mu"], t["lambda1"], 0.0,
                                float(np.sqrt(product / ratio)))

  # config_axes must be a SUBSET of axes: `axes` is everything fitted, `config_axes` names
  # which of those reach the configuration rather than the material
  return CandidateModel(f"qSLS|g/alpha={ratio:g}+req", build, (*AXES, "req_scale"), (),
                        ("req_scale",))


def one(job):
  from pyimr.selection import PARAMETER_BOUNDS, fit_candidate, physical_from_unit
  dataset, index = job
  times, mean, spread, maximum, stretch, window = _trials(dataset)
  candidate = candidate_with_req()
  box = dict(BOX) | {"req_scale": PARAMETER_BOUNDS["req_scale"]}
  solve = records.solver(times, maximum, stretch)
  axes = tuple(candidate.axes)
  fit = fit_candidate(candidate, solve, window[:, index], spread, bounds=box, starts=STARTS,
                      max_evaluations=EVALUATIONS)
  values = physical_from_unit(axes, fit.unit, box)
  fitted = dict(zip(axes, (float(v) for v in values), strict=True))
  from pyimr.selection import evaluate_at
  model = evaluate_at(candidate, solve, fitted)[0]
  return (dataset, index), {"fitted": fitted, "chi2_per_n": float(fit.chi_squared),
                            "residual": (np.asarray(model, dtype=float)
                                         - window[:, index]).tolist()}


def main():
  before = json.load(open(records.HERE / "per_trial_fits.json"))
  jobs = [(d, i) for d in records.DATASETS for i in range(_trials(d)[5].shape[1])]
  print(f"  refitting {len(jobs)} trials with req_scale free ...", flush=True)
  with records.pool(len(jobs)) as pool:
    table = dict(pool.map(one, jobs))

  summary = {}
  print("\n  how far the equilibrium radius moves between trials\n")
  print(f"  {'record':13s} {'J':>3s} {'req_scale spread':>17s} {'range':>18s} "
        f"{'median chi2/N':>14s}")
  for dataset in records.DATASETS:
    rows = [table[(dataset, i)] for i in range(_trials(dataset)[5].shape[1])]
    scales = np.array([r["fitted"]["req_scale"] for r in rows])
    summary[dataset] = {"req_cv": float(scales.std(ddof=1)),
                        "req_range": [float(scales.min()), float(scales.max())],
                        "chi2_median": float(np.median([r["chi2_per_n"] for r in rows])),
                        "pinned": int(np.sum((scales < 0.905) | (scales > 1.105)))}
    print(f"  {dataset:13s} {len(rows):3d} {scales.std(ddof=1):16.1%} "
          f"  {scales.min():.3f} to {scales.max():.3f} {summary[dataset]['chi2_median']:14.4f}")

  print("\n  does the dominant mode survive NOW?\n")
  print(f"  {'record':13s} {'raw':>7s} {'material only':>14s} {'+ req_scale':>12s} "
        f"{'|cos| to raw':>13s}")
  for dataset in records.DATASETS:
    _, _, _, _, _, window = _trials(dataset)
    rows = [table[(dataset, i)] for i in range(window.shape[1])]
    after = np.column_stack([r["residual"] for r in rows])
    after = after - after.mean(axis=1, keepdims=True)
    raw = window - window.mean(axis=1, keepdims=True)
    left_raw, values_raw, _ = np.linalg.svd(raw, full_matrices=False)
    left, values, _ = np.linalg.svd(after, full_matrices=False)
    share = float(values[0] ** 2 / (values**2).sum())
    overlap = abs(float(left_raw[:, 0] @ left[:, 0]))
    summary[dataset] |= {"mode_after_req": share, "overlap_with_raw": overlap}
    print(f"  {dataset:13s} {float(values_raw[0]**2/(values_raw**2).sum()):7.3f} "
          f"{before[dataset]['mode_after']:14.3f} {share:12.3f} {overlap:13.3f}")

  print("\n  A mode that collapses only once the equilibrium radius is free was latent")
  print("  variation in the INITIAL CONDITION, which the material axes could never absorb.")
  print("  One that survives this too is not any parameter the model has.")
  json.dump(summary, open(records.HERE / "per_trial_initial.json", "w"), indent=1)


if __name__ == "__main__":
  main()
