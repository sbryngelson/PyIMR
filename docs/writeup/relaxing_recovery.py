"""Can `yeohSLS` be fitted at all, or only reported?

The relaxing-law screen (#241) rejected five of six candidates on identifiability grounds and
left `yeohSLS` with a different verdict: not degenerate, but ill-conditioned at $3$--$6\\times
10^6$, with its sloppiest direction moving the trace by $7.9\\times10^{-2}$ against trial noise
of $1.8\\times10^{-2}$. Sloppy is not the same as unidentifiable, and the difference is
decidable: generate data from a KNOWN parameter set, fit it back at a stated budget, and see
whether the answer returns.

This is the only honest test available. A fit to the real records cannot be graded -- there is
no true value to compare against, and `lack_of_fit` rejects every model on them, so a small
chi-squared would say nothing about whether the parameters mean anything. Synthetic data has
neither problem: the truth is known and the model is exactly right, so any failure to recover
is the optimizer or the conditioning and nothing else.

The bar is recovery of the PARAMETERS, not of the trace. A sloppy model reproduces its own data
from the wrong point by construction -- that is what sloppiness means -- so a matched trace is
the null result here, not the positive one.

Noise is drawn at the measured trial spread rather than at some clean fraction of it. The whole
question is whether the conditioning survives contact with the noise this apparatus actually
delivers, and a quieter test would answer an easier question.

THE BOX IS THE ADMISSIBLE ONE, NOT `PARAMETER_BOUNDS`, and this is the finding rather than a
convenience. Over the raw box only $7\%$ of log-uniform draws integrate at all, so a multistart
spends its budget failing and reports "did not fit from any of 8 starts" -- which grades the
prior, not the conditioning. Raising the step cap twentyfold admits not one extra point and
makes each failure five times more expensive, so this is not a budget the fit was denied.

Swept one axis at a time, the obstruction is a strain and rate limit, not a coordinate error:
`yeoh_c3` dies above $\approx 18$, `yeoh_c2` above $\approx 320$, `g` above $\approx 7.5$ kPa, and
`mu` BELOW $0.1$ -- too little damping, not too much. `lambda1` is the one that decides what
this test can mean: it is admissible fast and admissible slow and dead in between, over
$6.8\times10^{-6}$ to $1.5\times10^{-4}$, which in units of the collapse time is $0.1$ to $2.3$.
The model diverges when its relaxation time is comparable to the collapse -- the one regime in
which a relaxation time is observable at all. That band survives twenty times the steps,
tolerances two decades either way, an implicit solver, and quadrature from $8$ to $512$ points,
so it is the model and not the integrator.

What remains fittable is therefore the QUASI-ELASTIC branch, where the arm relaxes far faster
than the bubble collapses. A recovery run there is still worth doing -- it is the branch any
fit to these records would land in -- but a success on it does not license the law generally,
because the branch is chosen for being integrable rather than for being where the physics is.
"""

import json

import numpy as np

import records

DATASET = "gelatin_15C"
BUDGETS = ((8, 240), (16, 600), (32, 1200))     # (starts, evaluations per start)
REPEATS = 4                                     # independent noise draws, so one lucky seed cannot pass it
# the truth has to integrate at the budget every other study here uses, and a strongly
# stiffening one does not: c2 = 4e2 with c3 = 6e1 exhausts two million solver steps at the
# truth itself. This one converges in under 200k, so raising the cap would only let hopeless
# candidates burn five times longer before failing -- one fit went from minutes to over
# a quarter of an hour on that alone
TRUTH = {"g": 3.0e3, "yeoh_c2": 1.0e2, "yeoh_c3": 1.0e1,
         "mu": 1.5e-1, "lambda1": 2.0e-6}
AXES = tuple(TRUTH)
# measured, not chosen: the largest box whose axes were admissible one at a time, with
# `lambda1` held to the fast branch below the dead band
ADMISSIBLE = {"g": (1e2, 7.5e3), "yeoh_c2": (1.0, 3.16e2), "yeoh_c3": (1.0, 1.78e1),
              "mu": (1e-1, 1.0), "lambda1": (1e-7, 4.64e-6)}
# recovery within a factor of two is generous -- these are decades-wide priors -- and is the
# loosest bar under which a fitted value could still be reported as a measurement
TOLERANCE = 2.0


def _material(values):
  import pyimr
  return pyimr.RelaxingMaterial(
    elastic=pyimr.Yeoh(values["g"], values["yeoh_c2"], values["yeoh_c3"]),
    viscosity_pa_s=values["mu"], relaxation_time_s=values["lambda1"], retardation_time_s=0.0)


def _candidate():
  from pyimr.selection import CandidateModel
  return CandidateModel("yeohSLS", _material, AXES)


def _box():
  """The admissible box, and the truth had better be inside it.

  Not `bounds_for_invariant`: that places `gent_jm` and `fung_b` against the measured strain
  because those two diverge, and Yeoh has neither axis. The limit here is on the stiffening
  coefficients and the relaxation rate themselves, which no existing placement covers.
  """
  outside = {k: (v, ADMISSIBLE[k]) for k, v in TRUTH.items()
             if not ADMISSIBLE[k][0] <= v <= ADMISSIBLE[k][1]}
  if outside: raise ValueError(f"the truth is outside the admissible box: {outside}")
  return dict(ADMISSIBLE)


def one(job):
  """One noise draw at one budget: fit the truth back and report how far off it lands."""
  from pyimr.selection import fit_candidate, physical_from_unit

  repeat, starts, evaluations = job
  times, mean, spread, maximum, stretch = records.load(DATASET)
  solve = records.solver(times, maximum, stretch)
  # a stiff candidate RAISES rather than returning non-finite, and the truth itself must
  # survive: a recovery test whose data never existed grades the optimizer on nothing
  try:
    clean = solve(_material(TRUTH))[0]
  except Exception as failure:                                   # noqa: BLE001
    return (repeat, starts), {"integrated": False, "why": type(failure).__name__}
  if not np.all(np.isfinite(clean)):
    return (repeat, starts), {"integrated": False, "why": "non-finite"}

  noisy = clean + np.random.default_rng(repeat).normal(0.0, spread)
  candidate = _candidate()
  bounds = _box()
  fit = fit_candidate(candidate, solve, noisy, spread, bounds=bounds, starts=starts,
                      max_evaluations=evaluations, seed=repeat)
  values = physical_from_unit(AXES, fit.unit, bounds)
  fitted = dict(zip(AXES, (float(v) for v in values), strict=True))
  ratios = {k: fitted[k] / TRUTH[k] for k in AXES}
  return (repeat, starts), {
    "integrated": True, "chi2_per_n": float(fit.chi_squared), "fitted": fitted,
    "ratios": ratios,
    "recovered": all(1 / TOLERANCE <= r <= TOLERANCE for r in ratios.values()),
    # the null result: a sloppy model matching the trace from the wrong point
    "trace_error": float(np.max(np.abs(solve(_material(fitted))[0] - clean))),
    "noise": float(np.median(spread)),
  }


def main():
  jobs = [(r, s, e) for s, e in BUDGETS for r in range(REPEATS)]
  with records.pool(len(jobs)) as pool:
    table = dict(pool.map(one, jobs))

  print(f"  yeohSLS recovering its own data at {DATASET}'s noise, truth "
        + ", ".join(f"{k}={v:.4g}" for k, v in TRUTH.items()) + "\n")
  print(f"  {'budget':>12s} {'recovered':>10s} {'chi2/N':>9s} {'trace err':>10s}   worst axis")
  for starts, evaluations in BUDGETS:
    got = [table[(r, starts)] for r in range(REPEATS) if table[(r, starts)]["integrated"]]
    if not got:
      why = {table[(r, starts)].get("why") for r in range(REPEATS)}
      print(f"  {starts:4d}x{evaluations:<7d} did not integrate ({', '.join(sorted(map(str, why)))})")
      continue
    passed = sum(g["recovered"] for g in got)
    worst = max(got, key=lambda g: max(abs(np.log(v)) for v in g["ratios"].values()))
    axis = max(worst["ratios"], key=lambda k: abs(np.log(worst["ratios"][k])))
    print(f"  {starts:4d}x{evaluations:<7d} {f'{passed}/{len(got)}':>10s} "
          f"{np.median([g['chi2_per_n'] for g in got]):9.3f} "
          f"{np.median([g['trace_error'] for g in got]):10.2e}   "
          f"{axis} off by x{worst['ratios'][axis]:.3g}")

  best = max(BUDGETS, key=lambda b: sum(table[(r, b[0])].get("recovered", False)
                                        for r in range(REPEATS)))
  hits = sum(table[(r, best[0])].get("recovered", False) for r in range(REPEATS))
  noise = np.median([g["noise"] for g in table.values() if g["integrated"]])
  print(f"\n  A trace error below the noise ({noise:.2e}) with the parameters off is the SLOPPY")
  print("  outcome, not the passing one: the model reproduces its own data from the wrong")
  print("  point, which is exactly what the conditioning predicted and what makes a fitted")
  print("  value unreportable.")
  print(f"\n  verdict: {hits}/{REPEATS} recovered at the best budget tried ({best[0]}x{best[1]}). "
        + ("register yeohSLS" if hits == REPEATS else
           "do NOT register yeohSLS as a fitted candidate"))

  json.dump({f"{r}|{s}": v for (r, s), v in table.items()},
            open(records.HERE / "relaxing_recovery.json", "w"), indent=1)


if __name__ == "__main__":
  main()
