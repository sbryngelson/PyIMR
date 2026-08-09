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
component is no larger than the seen one and is not conservative, gives the figure here.

WHICH COORDINATES, AND WHY THAT DECIDES EVERYTHING. In the natural axes the precondition
cannot be met, ever. The likelihood depends on the product `g*alpha` alone, which the
identifiability section derives and this measures independently: `g` and `alpha` come out
anticorrelated at `-1.00000`, the Fisher matrix conditioned near `1e8`, and the null direction
`(-0.707, 0, 0, +0.708)` in log coordinates. A fit slides along that valley until it meets a
box wall, and at a constrained optimum the gradient along the wall does not vanish, so the
residual keeps a component an interior optimum would not have. Widening the wall does not
help: released from `g >= 100 Pa` the pair slid on until `alpha` reached its own upper bound,
so the pin moved rather than lifted.

Both parametrisations are therefore run. The natural one shows the precondition failing and
why; the identified one, `(mu, g*alpha, lambda1)` taken from `identified.py` so there is one
definition of it, has no null direction and is where the bound can actually be computed.
"""

import json

import numpy as np

import records
from identified import BOX, candidate_at_ratio

NATURAL = ("mu", "g", "lambda1", "alpha")
IDENTIFIED = ("mu", "galpha", "lambda1")
RATIO = 38.5                      # the published fit's g/alpha; identified.py brackets it
STARTS, EVALUATIONS = 16, 600


def one(dataset):
  """Both parametrisations on one record, each fitted in its own coordinates."""
  import pyimr
  from pyimr.noise import discrepancy, lack_of_fit
  from pyimr.selection import (PARAMETER_BOUNDS, STANDARD_MODELS, fit_candidate,
                               physical_from_unit)

  times, mean, spread, maximum, stretch = records.load(dataset)
  trials = records.trial_count(dataset)
  error = spread / np.sqrt(trials)
  out = {}

  for label, candidate, box in (("natural", STANDARD_MODELS["qSLS"], None),
                                ("identified", candidate_at_ratio(RATIO), BOX)):
    solve = records.solver(times, maximum, stretch)
    fit = fit_candidate(candidate, solve, mean, spread, bounds=box, starts=STARTS,
                        max_evaluations=EVALUATIONS)
    values = physical_from_unit(candidate.axes, fit.unit, box)
    fitted = dict(zip(candidate.axes, (float(v) for v in values), strict=True))
    axes = tuple(candidate.axes)
    point = np.array([fitted[k] for k in axes])

    def build(scale, point=point, axes=axes, candidate=candidate):
      moved = dict(zip(axes, (float(v) for v in point * scale), strict=True))
      config = pyimr.SimulationConfig(maximum, maximum / stretch, candidate.build(moved),
                                      dynamics="keller-miksis", rtol=1e-9, atol=1e-11,
                                      max_steps=400_000)
      return np.asarray(pyimr.simulate(times, config).radius_ratio, dtype=float)

    model = build(np.ones(len(axes)))
    step = 1e-4
    jacobian = np.column_stack([
      (build(np.where(np.arange(len(axes)) == k, 1 + step, 1.0)) -
       build(np.where(np.arange(len(axes)) == k, 1 - step, 1.0))) / (2 * step * error)
      for k in range(len(axes))])

    split = discrepancy((mean - model) / error, jacobian)
    fisher = jacobian.T @ jacobian
    eigenvalues = np.linalg.eigvalsh(fisher)
    deviation = np.sqrt(np.diag(np.linalg.pinv(fisher)))
    walls = box or PARAMETER_BOUNDS
    out[label] = {
      "axes": list(axes), "chi2_per_n": float(fit.chi_squared),
      "lack_of_fit": float(lack_of_fit(mean, model, spread, trials, candidate.dimension).ratio),
      "identifiable": split.size, "absorbable": split.absorbed,
      "at_optimum": split.at_optimum, "bias_sigmas": split.bias_sigmas,
      "condition": float(eigenvalues[-1] / max(eigenvalues[0], 1e-300)),
      "sigma": {k: float(v) for k, v in zip(axes, deviation, strict=True)},
      "fitted": fitted,
      "pinned": {k: float(v) for k, v in fitted.items()
                 if min(abs(np.log(v / walls[k][0])), abs(np.log(v / walls[k][1]))) < 1e-6},
    }
  return dataset, out


def main():
  with records.pool(len(records.DATASETS)) as pool:
    table = dict(pool.map(one, list(records.DATASETS)))

  for label in ("natural", "identified"):
    print(f"\n  {label} coordinates {tuple(next(iter(table.values()))[label]['axes'])}")
    print(f"    {'record':13s} {'chi2/N':>7s} {'lack-of-fit':>11s} {'cond':>9s} "
          f"{'absorbable':>11s} {'at opt':>7s} {'bias (sigmas)':>14s}  pinned")
    for name, got in table.items():
      here = got[label]
      total = float(np.hypot(here["absorbable"], here["identifiable"]))
      bias = (f"{here['bias_sigmas']:14.1f}" if here["at_optimum"]
              else f"{'not applicable':>14s}")
      print(f"    {name:13s} {here['chi2_per_n']:7.3f} {here['lack_of_fit']:11.2f} "
            f"{here['condition']:9.1e} {here['absorbable'] / total:10.1%} "
            f"{str(here['at_optimum']):>7s} {bias}  {','.join(here['pinned']) or '-'}")

  print("\n  where the bound applies, what it allows each parameter")
  for name, got in table.items():
    here = got["identified"]
    if not here["at_optimum"]: continue
    inside = "  ".join(f"{k} x{np.exp(here['bias_sigmas'] * v):.3g}"
                       for k, v in here["sigma"].items())
    print(f"    {name:13s} up to {here['bias_sigmas']:.1f} sigmas -- {inside}")

  json.dump(table, open(records.HERE / "discrepancy.json", "w"), indent=1)


if __name__ == "__main__":
  main()
