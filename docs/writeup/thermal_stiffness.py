r"""WHICH term makes the corrected forward model un-integrable, and is the solver to blame?

`sec:universal` reports that refitting under `bubtherm`/`masstrans`/`vapor` raises on six of
eight records, and a probe localised that to the forward solve rather than the search. It did
NOT localise it further, and the obvious remedy was never checked: `_solver_for` already
switches to an IMPLICIT Kvaerno5 whenever `bubtherm` is on with the spectral scheme, so the
stalls happened WITH a stiff solver, and "use a stiff solver" is not available as a fix.

So the question is which term, at what resolution. The flag constraints make the split clean:
`masstrans` requires `vapor`, and `bubtherm` with `vapor` requires `masstrans`, but `vapor`
ALONE (no `bubtherm`) is legal and is just the vapour pressure entering the gas law. That
separates the two candidate causes the paper has been treating as one:

  vapor only        -- vapour pressure, explicit solver, no thermal PDE
  bubtherm only     -- the thermal PDE, implicit solver, no vapour
  bubtherm + both   -- the configuration that fails

`Nt` is swept because the failing runs used 7 against a default of 25, and a spectral
operator's conditioning is a function of its node count. A budget well below the production
one is used so a failure costs seconds rather than minutes; anything that survives here is
re-run at full budget by `thermal_vapour_fit.py`.
"""

import json
import time

import numpy as np

import records

# far below production, so a stall is cheap to observe; survivors get the real budget later
BUDGET = 60_000
RECORDS = ("gelatin_15C", "paam_PA05")
CONFIGS = (
  ("isothermal", {}),
  ("vapour only", {"vapor": 1}),
  ("thermal Nt=7", {"bubtherm": 1, "Nt": 7}),
  ("thermal Nt=15", {"bubtherm": 1, "Nt": 15}),
  ("thermal Nt=25", {"bubtherm": 1, "Nt": 25}),
  ("thermal+vapour Nt=7", {"bubtherm": 1, "Nt": 7, "masstrans": 1, "vapor": 1}),
  ("thermal+vapour Nt=15", {"bubtherm": 1, "Nt": 15, "masstrans": 1, "vapor": 1}),
  ("thermal+vapour Nt=25", {"bubtherm": 1, "Nt": 25, "masstrans": 1, "vapor": 1}),
)


def one(job):
  """Evaluate the record's converged ISOTHERMAL optimum under `extra`, and time it."""
  from pyimr.selection import STANDARD_MODELS, evaluate_at

  dataset, label, extra = job
  times, mean, spread, maximum, _ = records.load(dataset)
  stretch = json.load(open(records.HERE / "paam_stretch.json"))[dataset]["argmin"]
  warm = json.load(open(records.HERE / "paam_lackoffit.json"))[dataset]["fitted"]
  solve = records.solver(times, maximum, stretch, max_steps=BUDGET, **extra)
  clock = time.perf_counter()
  try:
    trace = np.asarray(evaluate_at(STANDARD_MODELS["qSLS"], solve, warm)[0], dtype=float)
    error = spread / np.sqrt(records.trial_count(dataset))
    return (dataset, label), {"ok": True, "seconds": time.perf_counter() - clock,
                              "chi2": float(np.sum(((mean - trace) / error) ** 2))}
  except Exception as failure:                                               # noqa: BLE001
    return (dataset, label), {"ok": False, "seconds": time.perf_counter() - clock,
                              "error": f"{type(failure).__name__}: {failure}"[:120]}


def main():
  jobs = [(d, label, extra) for d in RECORDS for label, extra in CONFIGS]
  print(f"  evaluating at each record's isothermal optimum, max_steps={BUDGET:,}\n")
  with records.pool(len(jobs)) as pool:
    got = dict(pool.map(one, jobs))

  out = {}
  for dataset in RECORDS:
    print(f"  {dataset}")
    baseline = got[(dataset, "isothermal")].get("chi2")
    for label, _ in CONFIGS:
      value = got[(dataset, label)]
      out[f"{dataset}|{label}"] = value
      if value["ok"]:
        shift = "" if baseline is None else f"   ({value['chi2'] / baseline - 1.0:+6.2%})"
        print(f"    {label:24s} {value['seconds']:7.1f}s  chi2 {value['chi2']:10.1f}{shift}")
      else:
        print(f"    {label:24s} {value['seconds']:7.1f}s  FAILED  {value['error']}")
    print()

  print("  The solver is already implicit (Kvaerno5) wherever bubtherm is on, so a stall here")
  print("  is the model's stiffness and not the integrator's order.")
  json.dump(out, open(records.HERE / "thermal_stiffness.json", "w"), indent=1)


if __name__ == "__main__":
  main()
