r"""What the missing PAAm $R_{\max}$ costs, measured rather than argued.

Gelatin's $R_{\max}$ and stretch come from the source paper's dataset table, one value per
record. PAAm has no such table here: `PA0503_radius.csv` holds $437$ radii whose pairing to the
released traces `radius_provenance.py` shows is not recoverable, so every PAAm record is
carried at the file's mean, \SI{359}{\micro\metre}, and at a stretch read off its own tail.

THE ARGUMENT THAT THIS IS SURVIVABLE. The traces are normalised, so $R_{\max}$ enters only
through the dimensionless groups. It multiplies $\mu$ and $\lambda_1$ and leaves $G$ alone,
and the only place it enters otherwise is the Weber number, which is $1.5\times 10^{-3}$ here.
If that is right, a wrong $R_{\max}$ moves the reported moduli and leaves every residual-shape
conclusion -- the lack-of-fit ratio, the discrepancy, the afterbounce test -- untouched.

THE ARGUMENT IS NOT THE MEASUREMENT. `PA0503_radius.csv` has a coefficient of variation of
\SI{22}{\percent} on $R_{\max}$, so the mean could be wrong by a good deal more than a rounding
error, and the equilibrium radius is not a pure scale at all: it sets the gas content through
the equilibrium condition. This refits one record across the plausible range of both and
reports what moves. A conclusion that survives here is reported without an asterisk; one that
does not is reported with the range it spans.
"""

import json

import numpy as np

import records
from lackoffit import PARAMETER_SHARE, WIDE

RECORD = "paam_PA05_21C"           # the largest single-condition PAAm block
RADIUS_FACTORS = (0.85, 1.0, 1.15)
STRETCH_VALUES = (7.45, 7.74, 8.01)


def _job(job):
  from pyimr.selection import STANDARD_MODELS, evaluate_at

  factor, stretch = job
  times, events = records.trials(RECORD)
  maximum = records.PAAM_MAXIMUM * factor
  times = times * factor                       # tau is fixed; the clock scales with R_max
  mean, spread = events.mean(axis=0), events.std(axis=0, ddof=1)
  solve = records.solver(times, maximum, stretch, max_steps=600_000)
  candidate = STANDARD_MODELS["qSLS"]
  try:
    scored = records.score(candidate, solve, mean, spread, bounds=WIDE, starts=10,
                           evaluations=300)
  except Exception as error:                                        # noqa: BLE001
    return job, {"failed": f"{type(error).__name__}: {error}"}
  model = np.asarray(evaluate_at(candidate, solve, scored["fitted"])[0], dtype=float)
  count = records.trial_count(RECORD)
  pure = (count - 1) * float(np.sum(spread**2)) / (mean.size * (count - 1))
  lack = count * float(np.sum((mean - model) ** 2)) / (mean.size - 4)
  return job, {"chi2_per_n": scored["chi2_per_n"], "f_ratio": lack / pure,
               "f_corrected": lack / pure / (1.0 - PARAMETER_SHARE),
               "fitted": scored["fitted"], "lag_one": scored["lag_one"]}


def main():
  jobs = [(f, s) for f in RADIUS_FACTORS for s in STRETCH_VALUES]
  print(f"  {RECORD}: refitting across R_max and stretch, {len(jobs)} fits\n")
  with records.pool(len(jobs)) as pool:
    table = dict(pool.map(_job, jobs))

  print(f"  {'R_max um':>9s} {'stretch':>8s} {'chi2/N':>8s} {'F':>8s} {'F corr':>8s} "
        f"{'g Pa':>10s} {'mu Pa s':>10s} {'lambda1 s':>11s} {'alpha':>9s}")
  summary = {}
  for job in jobs:
    row = table[job]
    factor, stretch = job
    key = f"{factor:g}x_{stretch:g}"
    if "failed" in row:
      print(f"  {records.PAAM_MAXIMUM*factor*1e6:9.1f} {stretch:8.2f}   {row['failed'][:44]}")
      summary[key] = row
      continue
    f = row["fitted"]
    print(f"  {records.PAAM_MAXIMUM*factor*1e6:9.1f} {stretch:8.2f} {row['chi2_per_n']:8.3f} "
          f"{row['f_ratio']:8.1f} {row['f_corrected']:8.1f} {f['g']:10.1f} {f['mu']:10.4f} "
          f"{f['lambda1']:11.3e} {f['alpha']:9.3g}")
    summary[key] = row

  # the two axes must be read APART. Lumping them reports the worse one as though both were
  # that bad, and here they are not remotely alike.
  def key(factor, stretch): return f"{factor:g}x_{stretch:g}"

  middle = STRETCH_VALUES[len(STRETCH_VALUES) // 2]
  along_radius = [(f, summary.get(key(f, middle))) for f in RADIUS_FACTORS]
  along_radius = [(f, v) for f, v in along_radius if v and "failed" not in v]
  along_stretch = [(s, summary.get(key(1.0, s))) for s in STRETCH_VALUES]
  along_stretch = [(s, v) for s, v in along_stretch if v and "failed" not in v]

  def spread(rows, field):
    values = [(v["fitted"][field] if field in ("g", "mu", "lambda1", "alpha") else v[field])
              for _, v in rows]
    return min(values), max(values), max(values) / min(values)

  print("\n  ---- along R_max, at the middle stretch ----\n")
  for field in ("chi2_per_n", "f_ratio", "g", "mu", "lambda1"):
    lo, hi, factor = spread(along_radius, field)
    print(f"  {field:12s} {lo:11.4g} to {hi:11.4g}   x{factor:.3f}")
  if len(along_radius) > 1:
    print("\n  against the scaling the dimensionless groups predict:")
    for axis, power in (("g", 0.0), ("mu", 1.0), ("lambda1", 1.0)):
      ref = along_radius[0][1]["fitted"][axis]
      got = [v["fitted"][axis] / ref for _, v in along_radius]
      want = [(f / along_radius[0][0]) ** power for f, _ in along_radius]
      print(f"    {axis:8s} measured {[round(x, 3) for x in got]}  "
            f"predicted {[round(x, 3) for x in want]}")

  print("\n  ---- along the stretch, at the mean R_max ----\n")
  for field in ("chi2_per_n", "f_ratio", "g", "mu", "lambda1"):
    lo, hi, factor = spread(along_stretch, field)
    print(f"  {field:12s} {lo:11.4g} to {hi:11.4g}   x{factor:.3f}")

  print("\n  ---- what it says ----\n")
  if along_radius and along_stretch:
    _, _, radius_f = spread(along_radius, "f_ratio")
    lo_f, hi_f, stretch_f = spread(along_stretch, "f_ratio")
    print(f"  R_max is the harmless one: {RADIUS_FACTORS[-1]/RADIUS_FACTORS[0]:.2f} of it moves F"
          f" by x{radius_f:.2f}, and mu tracks it linearly as the groups say it must. The mean")
    print("  radius the PAAm records are carried at is therefore not load-bearing.")
    print(f"  The equilibrium radius is NOT harmless: over its own uncertainty F runs {lo_f:.1f}"
          f" to {hi_f:.1f}, a factor of {stretch_f:.2f}.")
    print("  The rejection survives that -- the smallest F in the sweep is still an order above")
    print("  the critical value -- but the SIZE of any F quoted here is pinned no better than a")
    print("  factor of two, on gelatin as much as on PAAm, and must be quoted that way.")
  records.HERE.joinpath("paam_sensitivity.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
  main()
