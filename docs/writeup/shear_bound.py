"""Is the shear modulus estimated, or is it just resting on its bound?

Every record fits `g` to exactly \\SI{100}{\\pascal}, the lower end of the prior box, so the
value in the tables is the bound rather than a measurement (#216). That has two possible
causes and they call for opposite fixes. Either the box is simply too tight and `g` settles
somewhere interior once released -- in which case widen it and move on -- or `g` runs to
whatever floor it is given, which means it is absorbing something the model is missing and no
bound makes it identifiable.

The two are told apart by lowering the floor and watching. A parameter that settles stops
moving; a parameter that compensates keeps going, and its chi-squared barely improves for the
decades it travels.

Gelatin is the reason this matters rather than being bookkeeping: its shear modulus is
kilopascals, so a fit driving below \\SI{100}{\\pascal} -- let alone below \\SI{1}{\\pascal} --
is not reporting a material property.
"""

import json

import numpy as np

import records

AXES = ("g", "mu", "lambda1", "alpha")
FLOORS = (1e2, 1e1, 1e0, 1e-1, 1e-2)
STARTS, EVALUATIONS = 16, 600


def fit_at(job):
  """Fit one record with the shear-modulus floor moved to `floor`."""
  from pyimr.noise import discrepancy, lack_of_fit
  from pyimr.selection import (PARAMETER_BOUNDS, STANDARD_MODELS, fit_candidate,
                               physical_from_unit)
  import pyimr

  dataset, floor = job
  times, mean, spread, maximum, stretch = records.load(dataset)
  trials = records.trial_count(dataset)
  candidate = STANDARD_MODELS["qSLS"]
  bounds = dict(PARAMETER_BOUNDS) | {"g": (float(floor), PARAMETER_BOUNDS["g"][1])}
  solve = records.solver(times, maximum, stretch)
  fit = fit_candidate(candidate, solve, mean, spread, bounds=bounds, starts=STARTS,
                      max_evaluations=EVALUATIONS)
  values = physical_from_unit(candidate.axes, fit.unit, bounds)
  fitted = dict(zip(candidate.axes, (float(v) for v in values), strict=True))
  point = np.array([fitted[k] for k in AXES])

  def trace(scale):
    scaled = point * scale
    material = pyimr.QuadraticZener(scaled[0], scaled[1], scaled[2], 0.0, scaled[3])
    config = pyimr.SimulationConfig(maximum, maximum / stretch, material,
                                    dynamics="keller-miksis", rtol=1e-9, atol=1e-11,
                                    max_steps=400_000)
    return np.asarray(pyimr.simulate(times, config).radius_ratio, dtype=float)

  error = spread / np.sqrt(trials)
  model = trace(np.ones(len(AXES)))
  step = 1e-4
  jacobian = np.column_stack([
    (trace(np.where(np.arange(len(AXES)) == k, 1 + step, 1.0)) -
     trace(np.where(np.arange(len(AXES)) == k, 1 - step, 1.0))) / (2 * step * error)
    for k in range(len(AXES))])
  split = discrepancy((mean - model) / error, jacobian)

  return (dataset, floor), {
    "g": fitted["g"], "on_floor": bool(abs(fitted["g"] / floor - 1.0) < 1e-6),
    "chi2_per_n": float(fit.chi_squared),
    "lack_of_fit": float(lack_of_fit(mean, model, spread, trials, candidate.dimension).ratio),
    "at_optimum": split.at_optimum, "absorbable": split.absorbed,
    "identifiable": split.size,
    "fitted": fitted,
  }


def main():
  jobs = [(name, floor) for name in records.DATASETS for floor in FLOORS]
  with records.pool(len(jobs)) as pool:
    table = dict(pool.map(fit_at, jobs))

  for name in records.DATASETS:
    print(f"\n  {name}")
    print(f"    {'floor':>8s} {'fitted g':>10s} {'on floor':>9s} {'chi2/N':>8s} "
          f"{'lack-of-fit':>12s} {'absorbable':>11s} {'at opt':>7s}")
    for floor in FLOORS:
      got = table[(name, floor)]
      total = float(np.hypot(got["absorbable"], got["identifiable"]))
      print(f"    {floor:8.3g} {got['g']:10.4g} {str(got['on_floor']):>9s} "
            f"{got['chi2_per_n']:8.4f} {got['lack_of_fit']:12.2f} "
            f"{got['absorbable'] / total:10.1%} {str(got['at_optimum']):>7s}")

  print("\n  verdict per record:")
  for name in records.DATASETS:
    stuck = [f for f in FLOORS if table[(name, f)]["on_floor"]]
    span = table[(name, FLOORS[0])]["chi2_per_n"] - table[(name, FLOORS[-1])]["chi2_per_n"]
    if len(stuck) == len(FLOORS):
      verdict = (f"rides every floor from {FLOORS[0]:g} to {FLOORS[-1]:g} Pa for "
                 f"{span:+.4f} of chi2/N: compensating, not estimated")
    elif not stuck:
      verdict = f"interior at every floor -- settles at {table[(name, FLOORS[-1])]['g']:.4g} Pa"
    else:
      free = min(f for f in FLOORS if not table[(name, f)]["on_floor"])
      verdict = (f"released at floor {free:g} Pa, settles at "
                 f"{table[(name, free)]['g']:.4g} Pa")
    print(f"    {name:13s} {verdict}")

  json.dump({f"{n}|{f:g}": v for (n, f), v in table.items()},
            open(records.HERE / "shear_bound.json", "w"), indent=1)


if __name__ == "__main__":
  main()
