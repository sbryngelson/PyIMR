"""Which bubble-dynamics equation does the record prefer?

Every model comparison in this package varies the material and holds the forward operator
at Keller-Miksis. That is an assumption, not a result. This asks it the other way round:
the candidate is fixed and the `solve` callback varies, which the machinery already supports
because `solve` was always a free argument. The parameter space is then identical across the
set, so the Occam terms cancel and the difference in log evidence is a Bayes factor between
operators.

Superseded for conclusions by `identified.py`, which repeats this in coordinates the record
can answer -- fitting `g` and `alpha` separately slides along a ridge until it hits the prior
box, and the ranking here follows the box. Kept because it is what demonstrated that.
"""

import json


import records

DATASET = "gelatin_15C"
# wider than `PARAMETER_BOUNDS`: with the package's own box, three of six fits sat exactly on
# `g = 1e2` or `lambda1 = 1e-7`, so the reported optimum was a bound rather than a fit
WIDE = {"g": (1e0, 1e6), "mu": (1e-5, 1e1), "lambda1": (1e-9, 1e-2), "alpha": (1e-4, 1e3)}


def _job(name):
  from pyimr.selection import STANDARD_MODELS

  times, mean, spread, maximum, stretch = records.load(DATASET)
  solve = records.solver(times, maximum, stretch, radial=name)
  try:
    return name, records.score(STANDARD_MODELS["qSLS"], solve, mean, spread, bounds=WIDE)
  except ValueError as error:
    return name, {"failed": str(error)}


def main():
  from pyimr.selection import DYNAMICS_MODELS

  names = list(DYNAMICS_MODELS)
  with records.pool(len(names)) as pool:
    order = dict(pool.map(_job, names))
  usable = {k: v for k, v in order.items() if "failed" not in v}
  baseline = usable["keller-miksis"]["log_evidence"]

  print(f"{DATASET}: qSLS held fixed, the bubble-dynamics operator varied\n")
  print(f"{'operator':>20} {'radial':>7} {'chi2/N':>9} {'log Z':>12} {'vs KM':>9} {'lag-1':>8}  pinned")
  for name in names:
    r = order[name]
    if "failed" in r: print(f"{name:>20} {DYNAMICS_MODELS[name]:7d}   did not fit"); continue
    print(f"{name:>20} {DYNAMICS_MODELS[name]:7d} {r['chi2_per_n']:9.3f} {r['log_evidence']:12.2f} "
          f"{r['log_evidence'] - baseline:+9.2f} {r['lag_one']:8.3f}  {','.join(r['pinned'])}")

  moduli = [r["fitted"]["g"] for r in usable.values()]
  lags = [r["lag_one"] for r in usable.values()]
  print(f"\n  shear modulus across operators: {min(moduli):.1f} to {max(moduli):.1f} Pa "
        f"({max(moduli) / min(moduli):.0f}x) -- set by where the prior cuts the g-alpha ridge.")
  print(f"  lag-one: {min(lags):.3f} to {max(lags):.3f}. The operator changes the fit and the")
  print("  evidence enormously and the residual correlation not at all.")
  records.HERE.joinpath("dynamics.json").write_text(json.dumps(order, indent=1))


if __name__ == "__main__":
  main()
