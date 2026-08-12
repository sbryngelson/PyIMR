"""Is the Req offset a calibration error, or the model absorbing its own inadequacy?

Freeing Req improves chi2 a lot. Two readings, opposite remedies:

  calibration  the stored stretch is wrong; the model was being asked to hit the wrong
               equilibrium radius, and correcting it should reduce STRUCTURE in the residual
  absorption   the model is wrong (lack_of_fit rejects every record), and Req is one more
               parameter for it to hide error in, so chi2 falls while the residual stays
               structured

The discriminator is lack-of-fit, not chi2. Replicates give pure error that costs no model, so
lack_of_fit measures systematic misfit against it. A genuine calibration fix reduces that ratio;
absorption moves the parameter and leaves it.

Also reported: how far the material parameters travel. Req trading against g*alpha along the
sloppy direction would show as a large move there, which is a third reading -- neither
calibration nor absorption, just the valley.
"""
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import numpy as np

import records
from identified import BOX, candidate_at_ratio
from per_trial_initial import candidate_with_req

WIDE = (0.6, 1.6)
STARTS, EVALUATIONS = 12, 400
AXES = ("mu", "galpha", "lambda1")


def one(job):
  from pyimr.noise import lack_of_fit
  from pyimr.selection import evaluate_at, fit_candidate, physical_from_unit

  dataset, free = job
  times, mean, spread, maximum, stretch = records.load(dataset)
  trials = records.trial_count(dataset)
  candidate = candidate_with_req() if free else candidate_at_ratio(38.5)
  box = (dict(BOX) | {"req_scale": WIDE}) if free else dict(BOX)
  solve = records.solver(times, maximum, stretch)
  axes = tuple(candidate.axes)
  fit = fit_candidate(candidate, solve, mean, spread, bounds=box, starts=STARTS,
                      max_evaluations=EVALUATIONS)
  fitted = dict(zip(axes, (float(v) for v in physical_from_unit(axes, fit.unit, box)),
                    strict=True))
  model = evaluate_at(candidate, solve, fitted)[0]
  ratio = lack_of_fit(mean, np.asarray(model, dtype=float), spread, trials,
                      candidate.dimension).ratio
  return (dataset, free), {"fitted": fitted, "chi2": float(fit.chi_squared),
                           "lack_of_fit": float(ratio)}


def main():
  jobs = [(d, f) for d in records.DATASETS for f in (False, True)]
  with records.pool(len(jobs)) as pool:
    table = dict(pool.map(one, jobs))
  print(f"\n  {'record':13s} {'Req':>6s} {'chi2/N':>8s} {'lack-of-fit':>12s} "
        f"{'mu':>10s} {'g*alpha':>10s} {'lambda1':>11s} {'req_scale':>10s}")
  for dataset in records.DATASETS:
    for free in (False, True):
      g = table[(dataset, free)]
      f = g["fitted"]
      print(f"  {dataset if not free else '':13s} {'free' if free else 'pinned':>6s} "
            f"{g['chi2']:8.4f} {g['lack_of_fit']:12.2f} {f['mu']:10.5f} {f['galpha']:10.1f} "
            f"{f['lambda1']:11.4e} {f.get('req_scale', 1.0):10.4f}")
    a, b = table[(dataset, False)], table[(dataset, True)]
    moves = " ".join(f"{k} x{b['fitted'][k] / a['fitted'][k]:.2f}" for k in AXES)
    print(f"  {'':13s} {'->':>6s} chi2 x{b['chi2'] / a['chi2']:.2f}, "
          f"lack-of-fit x{b['lack_of_fit'] / a['lack_of_fit']:.2f};  material moves: {moves}")
  print("\n  lack-of-fit falling with chi2 is a calibration fix; lack-of-fit staying put while")
  print("  chi2 falls is the model absorbing its own error into one more free parameter.")


if __name__ == "__main__":
  main()
