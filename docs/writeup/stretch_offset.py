r"""Where does the equilibrium radius actually want to sit?

`per_trial_initial.py` freed $R_{\rm eq}$ over the $\pm 11\%$ its prior allows and found the
fits pressed against the ceiling: seven of eighteen at \SI{15}{\celsius} and four of fourteen at
\SI{23}{\celsius}, with every \SI{15}{\celsius} trial above $1.062$. Pinning that is ONE-SIDED
is not scatter between bubbles. Scatter would straddle the stored value; this says the stored
value itself is off, and the prior was too narrow to show how far.

$R_{\rm eq}$ is not measured. It comes from a stretch ratio carried in the dataset table --
$7.09$, $7.37$, $6.83$ -- and \cref{sec:reqprior} already establishes that a $1.68\%$ error in
it is worth as much as the operator difference that section's margins measure. So the size of
the offset matters directly to a conclusion this document draws.

This widens the box to $\pm 40\%$ and asks two separate questions the narrow fit could not.

The MEAN trace answers the systematic one: one fit per record, and the $R_{\rm eq}$ it prefers
against the stored value. A record whose mean wants a different equilibrium radius has a
calibration offset, not noise.

The per-trial fits answer the scatter one: with the box wide enough that nothing is cut off, how
far do individual bubbles differ around whatever the mean prefers. That is the number
\cref{sec:latent} wanted and could not have, because a pinned distribution reports the bound's
position rather than the data's spread.

The two are worth separating because they have different remedies. An offset is fixed by
re-deriving the stretch; scatter is a term in $\Sigma$.
"""

import json

import numpy as np

import records
from identified import BOX
from per_trial_fits import _trials
from per_trial_initial import candidate_with_req

WIDE = (0.6, 1.6)                # far wider than the +-11% that pinned
STARTS, EVALUATIONS = 12, 400


def _fit(dataset, observed, box):
  from pyimr.selection import fit_candidate, physical_from_unit
  times, mean, spread, maximum, stretch = records.load(dataset)
  candidate = candidate_with_req()
  solve = records.solver(times, maximum, stretch)
  axes = tuple(candidate.axes)
  fit = fit_candidate(candidate, solve, observed, spread, bounds=box, starts=STARTS,
                      max_evaluations=EVALUATIONS)
  values = physical_from_unit(axes, fit.unit, box)
  return dict(zip(axes, (float(v) for v in values), strict=True)), float(fit.chi_squared)


def one(job):
  dataset, index = job
  times, mean, spread, maximum, stretch, window = _trials(dataset)
  box = dict(BOX) | {"req_scale": WIDE}
  observed = mean if index < 0 else window[:, index]
  fitted, chi2 = _fit(dataset, observed, box)
  return (dataset, index), {"fitted": fitted, "chi2_per_n": chi2}


def main():
  jobs = [(d, i) for d in records.DATASETS
          for i in range(-1, _trials(d)[5].shape[1])]      # -1 is the mean trace
  print(f"  {len(jobs)} fits with req_scale free over {WIDE} ...", flush=True)
  with records.pool(len(jobs)) as pool:
    table = dict(pool.map(one, jobs))

  summary = {}
  print("\n  the systematic part: what the MEAN trace prefers\n")
  print(f"  {'record':13s} {'stored stretch':>15s} {'req_scale':>10s} {'implied stretch':>16s} "
        f"{'offset':>8s} {'chi2/N':>8s}")
  for dataset in records.DATASETS:
    times, mean, spread, maximum, stretch = records.load(dataset)
    got = table[(dataset, -1)]
    scale = got["fitted"]["req_scale"]
    summary[dataset] = {"stored_stretch": float(stretch), "mean_req_scale": scale,
                        "implied_stretch": float(stretch) / scale,
                        "mean_chi2": got["chi2_per_n"]}
    print(f"  {dataset:13s} {stretch:15.4f} {scale:10.4f} {stretch / scale:16.4f} "
          f"{scale - 1:+8.1%} {got['chi2_per_n']:8.4f}")

  print("\n  the scatter part: per-trial spread once nothing is cut off\n")
  print(f"  {'record':13s} {'J':>3s} {'median':>8s} {'spread':>8s} {'range':>16s} "
        f"{'still pinned':>13s}")
  for dataset in records.DATASETS:
    count = _trials(dataset)[5].shape[1]
    scales = np.array([table[(dataset, i)]["fitted"]["req_scale"] for i in range(count)])
    pinned = int(np.sum((scales < WIDE[0] * 1.005) | (scales > WIDE[1] * 0.995)))
    summary[dataset] |= {"trial_median": float(np.median(scales)),
                         "trial_spread": float(scales.std(ddof=1)),
                         "trial_range": [float(scales.min()), float(scales.max())],
                         "still_pinned": pinned}
    print(f"  {dataset:13s} {count:3d} {np.median(scales):8.4f} {scales.std(ddof=1):8.1%} "
          f"  {scales.min():.3f} to {scales.max():.3f} {pinned:13d}")

  print("\n  an offset is re-derived; a spread is a term in Sigma. Reading one as the other")
  print("  either invents scatter that is a miscalibration, or hides a miscalibration in it.")
  worst = max(summary, key=lambda k: abs(summary[k]["mean_req_scale"] - 1))
  size = summary[worst]["mean_req_scale"] - 1
  print(f"\n  largest offset: {worst} at {size:+.1%}, against the {1.68:.2f}% that "
        "sec:reqprior says is worth an operator change")
  json.dump(summary, open(records.HERE / "stretch_offset.json", "w"), indent=1)


if __name__ == "__main__":
  main()
