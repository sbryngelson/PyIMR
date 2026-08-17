r"""How much of the afterbounce $\chi^2$ was the seed, and what does the control actually cover?

`shape_error.py` reports a best achievable $\chi^2$ per bounce of $6.52$ and $5.33$ and
concludes no fixed qSLS reaches the gelatin afterbounces. `paam_shape.py` asks the same of PAAm
and gets $0.73$ to $26.32$. Both numbers came out of a multistart seeded with
`abs(hash(dataset))`, and `hash()` of a string in Python is salted PER PROCESS: the same script
drew different starts on every run, so neither number was reproducible and the spread between
runs had never been looked at. The seed is stable now. This measures what it was worth.

WHY IT MATTERS MORE THAN A TIDINESS FIX. The claim is a threshold claim -- $\chi^2$ below about
one means a fixed material reaches the sequence and above it means none does -- and three of the
PAAm records land at $1.32$, $2.43$ and $0.73$, close enough to the line that seed scatter alone
could move them across it. A verdict that depends on which side of a threshold a noisy statistic
fell on is not a verdict.

AND THE CONTROL HAS TO BE REPORTED WHERE IT IS ABSENT. On `paam_PA05003` the synthetic target
could not be generated at all, so the fitter was never validated there and its $26.32$ means
nothing; the first run simply printed no control line, which reads exactly like a control that
passed. That is the failure mode `enrichment_screen.py` already carries a comment about. Here a
record with no passing control is reported as UNINTERPRETABLE and takes no verdict.

WHICH DIRECTION THE STATISTIC BOUNDS, BECAUSE IT IS THE AWKWARD ONE. A fit returns an UPPER
bound on the achievable $\chi^2$: some better point may exist that the multistart did not find.
"Reachable" therefore needs nothing -- a small $\chi^2$ exhibits the material that reaches it --
but "NOT reachable" needs the optimum to be genuine, which is a claim about the optimiser rather
than about the data. The sham control is what licenses it, and the record where it could not run
is also the only one seen to land in two different basins between runs, at $26.3$ and $39.1$.
That is the failure the control exists to catch, caught by the control's own absence.
"""

import json

import numpy as np

import records
import paam_shape
from frequency_space import KEEP, measured as gelatin_measured
from shape_error import (AXES, BOX, DISPLACED, EDGE, MAX_STEPS_REPORT, _fit, _objective,
                         _ratios, seed_for)

ORDER = (*records.DATASETS, *records.PAAM)
REPEATS = 4
CONTROL_PASSES = 1.0          # chi2 the fitter must reach on a target the model itself made


def _target(dataset):
  """`(target, error, base material)` for either material, from its own fit."""
  if dataset in records.PAAM:
    obs = paam_shape.measured(dataset)
    base = paam_shape.base_material(dataset)
  else:
    obs = gelatin_measured(dataset)
    median = json.load(open(records.HERE / "per_trial_fits.json"))[dataset]["median"]
    base = {a: float(median[a]) for a in AXES}
  n = min(KEEP, len(obs["ratio"]))
  return (np.array(obs["ratio"][:n], dtype=float),
          np.array(obs["ratio_spread"][:n], dtype=float) / np.sqrt(obs["events"]),
          base, n, obs["events"])


def one(job):
  dataset, repeat = job
  target, err, base, n, events = _target(dataset)
  objective = _objective(dataset, target, err, base, n)
  best, best_x = _fit(objective, seed=seed_for(f"{dataset}|{repeat}"))
  fitted = {a: float(base[a] * np.exp(best_x[k])) for k, a in enumerate(AXES)}
  at_edge = [a for a in AXES if not (BOX[a][0] * EDGE < fitted[a] / base[a] < BOX[a][1] / EDGE)]
  flat = {}
  for a in at_edge:
    unit = np.array([np.log(fitted[b] / base[b]) for b in AXES])
    unit[AXES.index(a)] = np.log(BOX[a][0] if fitted[a] / base[a] < 1.0 else BOX[a][1])
    pushed = objective(unit)
    flat[a] = bool(pushed < 1e5 and abs(pushed - best) <= 0.01 * max(best, 1e-12))

  control = None
  if repeat == 0:
    sham_values = dict(base, lambda1=base["lambda1"] * DISPLACED)
    sham_target = _ratios(dataset, sham_values, n, MAX_STEPS_REPORT)
    if sham_target is None:
      control = {"ran": False, "why": "the model could not produce the synthetic target"}
    else:
      value, raw = _fit(_objective(dataset, sham_target, err, base, n),
                        seed=seed_for(f"{dataset}|sham"))
      control = {"ran": True, "chi2": value,
                 "lambda1_recovered": float(base["lambda1"] * np.exp(raw[2])
                                            / sham_values["lambda1"])}
  return job, {"chi2": best, "fitted": fitted, "at_edge": at_edge, "edge_is_flat": flat,
               "events": events, "n_bounces": n, "control": control}


def main():
  jobs = [(d, r) for d in ORDER for r in range(REPEATS)]
  print(f"  {len(jobs)} fits: every record at {REPEATS} seeds, both materials\n")
  with records.pool(len(jobs)) as pool:
    table = dict(pool.map(one, jobs))

  print(f"  {'record':>14s} {'events':>7s} " + " ".join(f"{'seed ' + str(r):>9s}"
                                                        for r in range(REPEATS))
        + f" {'spread':>8s} {'control':>22s}")
  summary = {}
  for dataset in ORDER:
    rows = [table[(dataset, r)] for r in range(REPEATS)]
    values = [r["chi2"] for r in rows]
    control = rows[0]["control"]
    if not control["ran"]: note = "DID NOT RUN"
    elif control["chi2"] > CONTROL_PASSES: note = f"FAILED at {control['chi2']:.2f}"
    else: note = f"passed, lam1 x{control['lambda1_recovered']:.3f}"
    print(f"  {dataset:>14s} {rows[0]['events']:7d} " + " ".join(f"{v:9.3f}" for v in values)
          + f" {max(values) - min(values):8.3f} {note:>22s}")
    summary[dataset] = {"events": rows[0]["events"], "n_bounces": rows[0]["n_bounces"],
                        "chi2_by_seed": values, "chi2_min": min(values),
                        "chi2_median": float(np.median(values)), "chi2_max": max(values),
                        "control": control,
                        "at_edge": rows[int(np.argmin(values))]["at_edge"],
                        "edge_is_flat": rows[int(np.argmin(values))]["edge_is_flat"]}

  print("\n  ---- verdicts, at the best seed, and only where the control covers them ----\n")
  for dataset, v in summary.items():
    truncated = [a for a in v["at_edge"] if not v["edge_is_flat"].get(a)]
    control = v["control"]
    if not control["ran"] or control["chi2"] > CONTROL_PASSES:
      verdict = "UNINTERPRETABLE: the fitter was never shown to work here"
    elif truncated:
      verdict = "DISCARDED (still descending at a wall)"
    elif v["chi2_max"] < 1.0: verdict = "REACHABLE by a fixed qSLS"
    elif v["chi2_min"] > 1.0: verdict = "NOT reachable"
    else: verdict = "STRADDLES the threshold across seeds: no verdict"
    print(f"  {dataset:>14s}: chi2 {v['chi2_min']:6.2f} to {v['chi2_max']:6.2f} -> {verdict}")

  gel = [v for d, v in summary.items() if d in records.DATASETS and v["control"]["ran"]]
  paam = [v for d, v in summary.items() if d in records.PAAM and v["control"]["ran"]]
  print("\n  ---- what it says ----\n")
  if gel and paam:
    print(f"  gelatin best achievable chi2 per bounce {min(v['chi2_min'] for v in gel):.2f} to "
          f"{max(v['chi2_min'] for v in gel):.2f} over {len(gel)} records")
    print(f"  PAAm    best achievable chi2 per bounce {min(v['chi2_min'] for v in paam):.2f} to "
          f"{max(v['chi2_min'] for v in paam):.2f} over {len(paam)} records")
  print("\n  The afterbounce test is the third of the three load-bearing measurements. Unlike")
  print("  the lack-of-fit test and the enrichment screen, whether it reproduces on PAAm is")
  print("  what these numbers decide, and a threshold verdict is only as good as the seed")
  print("  scatter beside it.")
  records.HERE.joinpath("shape_seeds.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
  main()
