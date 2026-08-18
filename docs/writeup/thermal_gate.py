r"""Does the corrected model integrate on all eight records, and at what resolution?

`thermal_stiffness.py` found that `paam_PA05` fails under thermal+vapour at `Nt = 7` and
SUCCEEDS at `Nt = 15` and `Nt = 25`. That inverts the standing conclusion. The stalls
reported in `sec:universal` were run at `Nt = 7` -- a number chosen for speed against a
package default of 25 -- so the model was never shown to be un-integrable; an
under-resolved spectral grid was.

This is the gate that settles it: every record, at production budget, at three resolutions.
It also prices the shortcut, because the one gelatin record that DID complete at `Nt = 7` is
the source of the "vapour barely moves the objective" number now in the text, and if that
record's chi-squared moves with resolution then the number is an artifact of the grid rather
than a property of the physics.
"""

import json
import time

import numpy as np

import records

BUDGET = 400_000  # the production budget the failing runs used
ORDER = (*records.DATASETS, *records.PAAM)
RESOLUTIONS = (7, 15, 25)


def one(job):
  from pyimr.selection import STANDARD_MODELS, evaluate_at

  dataset, nodes = job
  times, mean, spread, maximum, _ = records.load(dataset)
  stretch = json.load(open(records.HERE / "paam_stretch.json"))[dataset]["argmin"]
  warm = json.load(open(records.HERE / "paam_lackoffit.json"))[dataset]["fitted"]
  extra = {"bubtherm": 1, "Nt": nodes, "masstrans": 1, "vapor": 1}
  solve = records.solver(times, maximum, stretch, max_steps=BUDGET, **extra)
  clock = time.perf_counter()
  try:
    trace = np.asarray(evaluate_at(STANDARD_MODELS["qSLS"], solve, warm)[0], dtype=float)
    error = spread / np.sqrt(records.trial_count(dataset))
    return (dataset, nodes), {"ok": True, "seconds": time.perf_counter() - clock,
                              "chi2": float(np.sum(((mean - trace) / error) ** 2))}
  except Exception as failure:                                               # noqa: BLE001
    return (dataset, nodes), {"ok": False, "seconds": time.perf_counter() - clock,
                              "error": f"{type(failure).__name__}: {failure}"[:110]}


def main():
  jobs = [(d, n) for d in ORDER for n in RESOLUTIONS]
  print(f"  thermal+vapour at each record's isothermal optimum, max_steps={BUDGET:,}\n")
  print(f"  {'record':>16s} " + "  ".join(f"{'Nt=' + str(n):>18s}" for n in RESOLUTIONS))
  with records.pool(len(jobs)) as pool:
    got = dict(pool.map(one, jobs))

  out, integrates = {}, {n: 0 for n in RESOLUTIONS}
  for dataset in ORDER:
    cells = []
    for nodes in RESOLUTIONS:
      value = got[(dataset, nodes)]
      out[f"{dataset}|{nodes}"] = value
      if value["ok"]:
        integrates[nodes] += 1
        cells.append(f"{value['chi2']:12.1f} {value['seconds']:5.0f}s")
      else:
        cells.append(f"{'STALLED':>12s} {value['seconds']:5.0f}s")
    print(f"  {dataset:>16s} " + "  ".join(cells))

  print("\n  records that integrate: "
        + ", ".join(f"Nt={n} {integrates[n]}/{len(ORDER)}" for n in RESOLUTIONS))
  moved = [d for d in ORDER
           if got[(d, 7)]["ok"] and got[(d, 25)]["ok"]
           and abs(got[(d, 25)]["chi2"] / got[(d, 7)]["chi2"] - 1.0) > 0.05]
  print(f"  records whose chi-squared moves more than 5% between Nt=7 and Nt=25: "
        f"{len(moved)} ({', '.join(moved) if moved else 'none'})")
  json.dump(out, open(records.HERE / "thermal_gate.json", "w"), indent=1)


if __name__ == "__main__":
  main()
