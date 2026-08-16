r"""Is the thermal model resolved at the grid this package uses?

\Cref{sec:thermal} scores the thermal treatment at $N_t = 11$ spectral nodes and never varies it.
\Cref{sec:axes} sizes the whole thermal axis at $16$ noise units, $8.5$ after the material
absorbs what it can. Nobody has asked whether $11$ nodes resolve the answer, and the question
turns out to decide whether the comparison means anything.

WHY IT CAME UP. An attempt to score the thermal treatment WITH the vapour and mass transfer the
validator requires alongside it produced fits that failed their own nesting gate: full physics
fitting worse than the \texttt{bubtherm} model it contains and was seeded from, returning a
viscosity five hundred times water's and reversing the sign of the $g\alpha$ trend. That reads
as a search failure on a multimodal surface. It is not. The objective is unresolved.

WHAT SEPARATES THE TWO CANDIDATES FOR THAT. A fit through an under-converged forward model and a
fit through a rough-but-converged one look alike from the outside, and the difference is a
convergence study rather than an argument. Two knobs move independently here: the ODE tolerance,
which controls the time integration, and $N_t$, which controls the spatial discretisation of the
bubble's temperature and vapour fields. Only one of them turns out to matter, and it is not the
one the solver reports on.

HOW THE ERROR IS MEASURED. In noise units, against the finest grid affordable, so it is directly
comparable to the $16$ and $8.5$ that \cref{sec:axes} attributes to the physics. A discretisation
error of the same order as the effect means the comparison is measuring the grid.

TIMING IS NOT READ FROM THIS SCRIPT. Each configuration is warmed before it is timed, because
the first call to a new configuration compiles and an earlier version of this probe reported
compile time as solve time --- which inverted the ordering and made the coarsest grid look the
slowest.
"""

import json

import numpy as np

import records

DATASET, RATIO = "gelatin_15C", 38.5
RTOLS = (1e-6, 1e-8, 1e-10)
GRIDS = {"bubtherm": (5, 7, 9, 11, 15, 21, 27), "full physics": (5, 7, 9, 11, 15)}
EXTRA = {"bubtherm": {}, "full physics": {"vapor": 1, "masstrans": 1}}
# what sec:axes attributes to the thermal axis itself, for scale
AXIS_NOISE_UNITS, AXIS_AFTER_REFIT = 16.0, 8.5


def _material():
  import pyimr

  m = json.load(open(records.HERE / "per_trial_fits.json"))[DATASET]["median"]
  product = m["galpha"]
  return pyimr.QuadraticZener(float(np.sqrt(product * RATIO)), m["mu"], m["lambda1"], 0.0,
                              float(np.sqrt(product / RATIO)))


def trace(nt, rtol=1e-8, warm=True, **extra):
  """`(trace, seconds)` at one grid and tolerance, timed only after a warming call."""
  import time

  import pyimr

  times, _, _, maximum, stretch = records.load(DATASET)
  config = pyimr.SimulationConfig(maximum, maximum / stretch, _material(),
                                  dynamics="keller-miksis", rtol=rtol, atol=rtol * 1e-2,
                                  max_steps=4_000_000, T8=288.15, bubtherm=1, Nt=nt, **extra)
  if warm: pyimr.simulate(times, config)
  start = time.time()
  result = pyimr.simulate(times, config)
  return np.asarray(result.radius_ratio, dtype=float), time.time() - start


def main():
  _, _, spread, _, _ = records.load(DATASET)
  summary = {"dataset": DATASET, "axis_noise_units": AXIS_NOISE_UNITS, "tolerance": {}, "grid": {}}

  print("  ---- is the TIME integration converged? min R/Rmax against rtol, at fixed Nt ----\n")
  print(f"  {'config':>13s} {'Nt':>4s} " + " ".join(f"{f'rtol {r:.0e}':>12s}" for r in RTOLS))
  for label, extra in EXTRA.items():
    for nt in (7, 11):
      row = []
      for rtol in RTOLS:
        tr, _ = trace(nt, rtol, warm=False, **extra)
        row.append(float(tr.min()))
      summary["tolerance"][f"{label}|Nt={nt}"] = row
      spanned = max(row) / min(row) - 1.0
      print(f"  {label:>13s} {nt:4d} " + " ".join(f"{v:12.5f}" for v in row)
            + f"   spans {100 * spanned:.2f}%")

  print("\n  ---- is the SPATIAL discretisation converged? error in noise units ----\n")
  for label, extra in EXTRA.items():
    grids = GRIDS[label]
    reference, ref_seconds = trace(grids[-1], **extra)
    print(f"  {label}, reference Nt={grids[-1]} ({ref_seconds:.1f} s a solve)")
    print(f"  {'Nt':>4s} {'||dR||/sigma':>13s} {'max|dR|/sigma':>14s} {'solve':>9s}")
    rows = {}
    for nt in grids[:-1]:
      tr, seconds = trace(nt, **extra)
      error = (tr - reference) / spread
      rows[str(nt)] = {"l2": float(np.sqrt(np.sum(error**2))),
                       "linf": float(np.max(np.abs(error))), "seconds": seconds}
      print(f"  {nt:4d} {rows[str(nt)]['l2']:13.1f} {rows[str(nt)]['linf']:14.1f} "
            f"{seconds:8.2f}s")
    summary["grid"][label] = {"reference_nt": grids[-1], "rows": rows}
    print()

  print("  ---- what it says ----\n")
  for label in EXTRA:
    rows = summary["grid"][label]["rows"]
    at11 = rows.get("11", {}).get("l2")
    if at11 is None: continue
    summary["grid"][label]["at_Nt11_noise_units"] = at11
    verdict = ("resolved" if at11 < 0.1 * AXIS_AFTER_REFIT else
               "NOT resolved: comparable to the effect")
    print(f"  {label:>13s}: {at11:5.1f} noise units of discretisation error at the Nt=11 this "
          f"package uses,")
    print(f"  {'':>13s}  against {AXIS_AFTER_REFIT} for the whole thermal axis after refitting. {verdict}.")
  print("\n  A forward model carrying discretisation error of the same order as the physics it")
  print("  would measure cannot be fitted, and no restart budget repairs that. The tolerance")
  print("  table above is what rules out the time integration as the cause.")
  json.dump(summary, open(records.HERE / "thermal_resolution.json", "w"), indent=1)


if __name__ == "__main__":
  main()
