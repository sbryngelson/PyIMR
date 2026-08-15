r"""With the clock held at the data's own convention, which medium density do the records want?

\#273 read `WATER_DENSITY = 998` in `pyimr/noise.py` and `RHO = 1064` in `pyimr/_config.py` as
two values of one quantity. They are not. The acquisition pipeline stores
$\mathrm{d}t_{\rm norm}/\mathrm{d}t = 10/R_{\max}$ with
$\sqrt{p_\infty/\rho} = \SI{10.07}{\metre\per\second}$ (\cref{sec:latent}), which implies
$\rho = \SI{999.2}{\kilo\gram\per\metre\cubed}$; `characteristic_time` at $998$ returns
$10.0761$. The clock's job is to UNDO the experimenters' normalisation, so it has to carry
their density, and it does. Only the equation of motion should carry the medium's.

WHAT THAT LEAVES OPEN, AND WHY IT IS NOT THE ORIGINAL QUESTION. `density_clock.py` swept
consistent pairs and found the published mismatched pair fits best, which read as "the bug is
load-bearing". Holding the clock fixed instead shows both halves are doing work: at clock $998$,
moving the dynamics from $1064$ to $998$ costs $4.6$ in lack of fit at \SI{15}{\celsius}, and at
dynamics $1064$, moving the clock from $998$ to $1064$ costs $4.66$. The clock is explained. The
dynamics density is not: $1064$ is the polyacrylamide value inherited from IMRv2, and these
records are $5\%$ gelatin at roughly $\SI{1015}{\kilo\gram\per\metre\cubed}$.

THE TEST DISTINGUISHES A MATERIAL PROPERTY FROM A SINK. A denser medium collapses more slowly,
$t_c \propto \sqrt{\rho}$, so a fit that is short of collapse time can buy it by inflating
$\rho$. Sweeping the dynamics density with the clock pinned separates the two readings:

  - an interior optimum near a physically admissible value is a medium density the records
    measure, and the answer is to set it and move on;
  - a lack of fit that keeps falling as $\rho$ grows past anything a gel can be means the
    density is absorbing a timing discrepancy, $1064$ is arbitrary-but-lucky, and the number to
    report is the discrepancy rather than the density.

The sweep therefore runs well past gelatin on both sides, because an optimum that is only
visible against admissible values is not an optimum, and \emph{nothing free} is read first for
the reason \cref{sec:reqprior} gives.
"""

import json

import numpy as np

import records
from density_clock import CONFIGURATIONS, _box, candidate_for, solver_with_start

CLOCK_DENSITY = 998.0                    # the acquisition convention; held fixed throughout
# spans gelatin (1015) and the inherited polyacrylamide value (1064), and runs past both so an
# interior optimum can be told from a monotone trend
GELATIN = 1015.0                         # 5% gelatin
GELATIN_MAX = 1060.0                     # generous upper end for a dilute gel
DENSITIES = (940.0, 998.0, 1015.0, 1064.0, 1120.0, 1200.0, 1320.0, 1500.0)


def one(job):
  """One fit at one dynamics density, with the clock pinned to the data's convention."""
  from pyimr.noise import lack_of_fit
  from pyimr.selection import evaluate_at, fit_candidate, physical_from_unit

  dataset, free, density = job
  times, mean, spread, maximum, stretch = records.load(dataset)
  candidate = candidate_for(free)
  solve = solver_with_start(times, maximum, stretch, CLOCK_DENSITY, density)
  box = _box(free)
  try:
    fit = fit_candidate(candidate, solve, mean, spread, bounds=box, starts=32,
                        max_evaluations=800)
  except ValueError as error:
    return job, {"failed": str(error)}
  axes = tuple(candidate.axes)
  fitted = dict(zip(axes, (float(v) for v in physical_from_unit(axes, fit.unit, box)), strict=True))
  model = np.asarray(evaluate_at(candidate, solve, fitted)[0], dtype=float)
  ratio = lack_of_fit(mean, model, spread, records.trial_count(dataset), candidate.dimension).ratio
  return job, {"chi2_per_n": float(fit.chi_squared), "lack_of_fit": float(ratio),
               "fitted": fitted, "u0": float(fitted.get("u0_shift", 1.0) - 1.0)}


def main():
  configurations = [c for c in CONFIGURATIONS if c in ((), ("u0_shift",))]
  jobs = [(d, free, rho) for d in records.DATASETS for free in configurations
          for rho in DENSITIES]
  print(f"  clock pinned at rho = {CLOCK_DENSITY:.0f} (the acquisition convention, "
        f"sqrt(p/rho) = {(101325.0 / CLOCK_DENSITY) ** 0.5:.4f} against the file's 10.07)")
  print(f"  sweeping the DYNAMICS density over {DENSITIES}")
  print("  gelatin is about 1015; 1064 is the inherited polyacrylamide value\n")
  print(f"  {len(jobs)} fits ...", flush=True)

  with records.pool(len(jobs)) as pool:
    got = dict(pool.map(one, jobs))

  summary = {"clock_density": CLOCK_DENSITY, "densities": list(DENSITIES), "records": {}}
  for free in configurations:
    label = "+".join(free) or "nothing"
    print(f"\n  ==== lack of fit, {label} free ====\n")
    print(f"  {'record':>14s}  " + " ".join(f"{r:8.0f}" for r in DENSITIES))
    for dataset in records.DATASETS:
      cells = []
      for rho in DENSITIES:
        v = got[(dataset, free, rho)]
        cells.append(f"{'failed':>8s}" if "failed" in v else f"{v['lack_of_fit']:8.2f}")
      print(f"  {dataset:>14s}  " + " ".join(cells))
      best = min((r for r in DENSITIES if "failed" not in got[(dataset, free, r)]),
                 key=lambda r: got[(dataset, free, r)]["lack_of_fit"])
      summary["records"].setdefault(dataset, {})[label] = {
        "by_density": {str(int(r)): got[(dataset, free, r)] for r in DENSITIES},
        "best_density": float(best)}
      print(f"  {'':>14s}  best at rho = {best:.0f}"
            + ("   <-- AT THE EDGE OF THE SWEEP" if best in (DENSITIES[0], DENSITIES[-1]) else ""))

  print("\n  ---- what the sweep says ----\n")
  interior, admissible = [], []
  for dataset in records.DATASETS:
    best = summary["records"][dataset]["nothing"]["best_density"]
    interior.append(DENSITIES[0] < best < DENSITIES[-1])
    admissible.append(best <= GELATIN_MAX)
    print(f"  {dataset:>14s}: nothing free prefers rho = {best:.0f}"
          f"  ({100 * (best / GELATIN - 1):+.0f}% on gelatin, a collapse time "
          f"{100 * ((best / GELATIN) ** 0.5 - 1):+.1f}% slower)")
  summary["all_interior"] = bool(all(interior))
  summary["all_admissible"] = bool(all(admissible))
  # An interior optimum says the density is IDENTIFIED, not that it is a density. Both have to
  # be checked, and separating them is the whole point: a parameter with a well-posed optimum at
  # an impossible value is the textbook signature of a sink for model error, and reading only
  # the first test would have called this a measurement.
  if all(interior) and all(admissible):
    print("\n  Interior and admissible. The records measure a medium density; set it.")
  elif all(interior):
    print("\n  Interior but NOT admissible: every record wants a medium denser than a gel can be")
    print(f"  ({GELATIN:.0f} for 5% gelatin, {GELATIN_MAX:.0f} at the generous end). The density is")
    print("  identified and is not a density -- it is a bounded sink absorbing a collapse-time")
    print("  deficit, and the number to report is that deficit rather than the density.")
  else:
    print("\n  At least one record pushes to the edge of the sweep, so the optimum is not even")
    print("  bounded and nothing here is a measurement of anything.")
  json.dump(summary, open(records.HERE / "dynamics_density.json", "w"), indent=1)


if __name__ == "__main__":
  main()
