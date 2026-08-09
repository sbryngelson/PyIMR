"""How wrong is the model, and what could that be doing to the parameters?

`lack_of_fit` rejects every record: the model misses structure the apparatus repeats. It does
not say what that costs the parameters, and the honest answer is that the cost cannot be
measured -- only bounded.

Writing the truth as `y = m(theta) + delta + noise`, the residual splits into the part no
parameter choice reproduces and the part some choice does. At a converged fit the second is
EXACTLY zero, because the normal equations put the residual orthogonal to every Jacobian
column: whatever component of `delta` the model could imitate has already been absorbed into
`theta_hat`, leaving nothing behind to detect. So the visible discrepancy is the whole
residual, and the invisible one is bounded rather than estimated.

The bound is a single number per record. Over all absorbed discrepancies of norm `B`, the
worst bias in coordinate `k` is `B` times that coordinate's own standard deviation -- the same
multiple for every parameter. Taking `B` equal to the visible part, which assumes the unseen
component is no larger than the seen one and is not conservative, gives the figure reported
here.

The fits are done here rather than read from `results.json`, whose `best_theta` is a grid
argmax on the bounds and fails the orthogonality precondition outright (#235).

AND THE PRECONDITION FAILS HERE TOO, which is the finding. A continuous refit of all three
records puts the shear modulus exactly on its lower bound of 100 Pa. At a
CONSTRAINED optimum the gradient along the pinned direction need not vanish, so the residual
keeps a component the model could still absorb -- between $40%$ and $59%$ of it -- and the
bound above does not apply. The number this script exists to report cannot be computed until
the fits are free, which makes the shear-modulus bound a blocker rather than a detail (#216).
"""

import json

import numpy as np

import records

AXES = ("g", "mu", "lambda1", "alpha")
STARTS, EVALUATIONS = 16, 600


def fit_and_split(dataset):
  """Refit the record, then split its residual into the seen and unseen halves."""
  import pyimr
  from pyimr.noise import discrepancy, lack_of_fit
  from pyimr.selection import (PARAMETER_BOUNDS, STANDARD_MODELS, fit_candidate,
                               physical_from_unit)

  times, mean, spread, maximum, stretch = records.load(dataset)
  trials = records.trial_count(dataset)
  candidate = STANDARD_MODELS["qSLS"]
  solve = records.solver(times, maximum, stretch)
  fit = fit_candidate(candidate, solve, mean, spread, starts=STARTS, max_evaluations=EVALUATIONS)
  values = physical_from_unit(candidate.axes, fit.unit, None)
  point = np.array([dict(zip(candidate.axes, (float(v) for v in values), strict=True))[k]
                    for k in AXES])

  def trace(scale):
    scaled = point * scale
    material = pyimr.QuadraticZener(scaled[0], scaled[1], scaled[2], 0.0, scaled[3])
    config = pyimr.SimulationConfig(maximum, maximum / stretch, material,
                                    dynamics="keller-miksis", rtol=1e-9, atol=1e-11,
                                    max_steps=400_000)
    return np.asarray(pyimr.simulate(times, config).radius_ratio, dtype=float)

  # the standard error of the MEAN: y is an average over `trials` repeats, and the sqrt(trials)
  # is exactly what a chi-squared against `spread` forgets
  error = spread / np.sqrt(trials)
  model = trace(np.ones(len(AXES)))
  residual = (mean - model) / error

  step = 1e-4
  jacobian = np.column_stack([
    (trace(np.where(np.arange(len(AXES)) == k, 1 + step, 1.0)) -
     trace(np.where(np.arange(len(AXES)) == k, 1 - step, 1.0))) / (2 * step * error)
    for k in range(len(AXES))])

  split = discrepancy(residual, jacobian)
  misfit = lack_of_fit(mean, model, spread, trials, candidate.dimension)
  sigma = np.sqrt(np.diag(np.linalg.pinv(jacobian.T @ jacobian)))
  # which axes the optimiser could not move: a pinned axis is why the residual is not
  # orthogonal to the model's span, and so why the bound does not apply
  box = PARAMETER_BOUNDS
  pinned = {k: float(v) for k, v in zip(AXES, point, strict=True)
            if min(abs(np.log(v / box[k][0])), abs(np.log(v / box[k][1]))) < 1e-6}
  return {
    "chi2_per_n": float(fit.chi_squared), "lack_of_fit": float(misfit.ratio),
    "identifiable": split.size, "absorbed": split.absorbed,
    "at_optimum": split.at_optimum, "bias_sigmas": split.bias_sigmas,
    "sigma": {k: float(v) for k, v in zip(AXES, sigma, strict=True)},
    "pinned": pinned,
    "fitted": {k: float(v) for k, v in zip(AXES, point, strict=True)},
  }


def main():
  with records.pool(len(records.DATASETS)) as pool:
    table = dict(zip(records.DATASETS, pool.map(fit_and_split, list(records.DATASETS)),
                     strict=True))

  print(f"  {'record':13s} {'chi2/N':>7s} {'lack-of-fit':>12s} {'|delta|':>9s} "
        f"{'still absorbable':>17s} {'at optimum':>11s}")
  for name, got in table.items():
    total = np.hypot(got["identifiable"], got["absorbed"])
    print(f"  {name:13s} {got['chi2_per_n']:7.3f} {got['lack_of_fit']:12.1f} "
          f"{got['identifiable']:9.1f} {got['absorbed'] / total:16.1%} "
          f"{str(got['at_optimum']):>11s}")

  blocked = [name for name, got in table.items() if not got["at_optimum"]]
  if blocked:
    print(f"\n  the bias bound is NOT reported for {len(blocked)} of {len(table)} records, "
          "because it does not hold away from an\n  interior optimum. What pins them:")
    for name in blocked:
      pinned = ", ".join(f"{k} at {v:g}" for k, v in table[name]["pinned"].items())
      print(f"    {name:13s} {pinned or 'no bound active -- the fit has not converged'}")
    print("\n  a constrained optimum leaves a gradient along the pinned axis, so part of the"
          "\n  residual is still absorbable and delta_hat is contaminated by it (#216).")
  for name, got in table.items():
    if got["at_optimum"]:
      print(f"\n  {name}: every parameter could be biased by up to "
            f"{got['bias_sigmas']:.0f} of its own standard deviations.")

  json.dump(table, open(records.HERE / "discrepancy.json", "w"), indent=1)


if __name__ == "__main__":
  main()
