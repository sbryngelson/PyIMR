r"""Do the afterbounces want the same density the first collapse wants?

`dynamics_density.py` pins the clock at the acquisition convention, sweeps the dynamics
density, refits the material at each, and finds an interior optimum at $1120$--$1200$ on all
six configurations. A two-component mixing rule puts that at $33$ to \SI{73}{\percent} w/w
gelatin for any solid density between $1300$ and $1500$, so it is not a density, and the
script says so: a bounded sink absorbing a collapse-time deficit.

THAT ARGUMENT IS CURRENTLY ONE OF PLAUSIBILITY, AND IT CAN BE MADE ONE OF MEASUREMENT. A
$\chi^2$ over the trace is dominated by the first collapse, where the trace moves fastest and
the residual is largest. The afterbounces are a nearly independent stretch of the same
signal, and they are scored here by an observable the trace fit never sees. If $\rho$ were a
medium density it would be the same number for both, because a density does not know which
part of the record it is acting on. If it is a sink for a collapse-time deficit, it is
localised where that deficit lives and the afterbounces need not want it at all.

NO REFITTING, WHICH IS WHAT MAKES IT A TEST. At each density the material is the one the
TRACE fit already chose there, read from `dynamics_density.json`. The afterbounces only
score it. Refitting the material to the bounces would hand the observable three parameters
to absorb the disagreement with, and `bounce_identify.py` measures $\mu$ and $\rho$ as $96$
to \SI{99}{\percent} degenerate in exactly that observable, so a refit would report the
degeneracy rather than the density.

WHAT WOULD MAKE THE ANSWER UNINTERESTING, AND HOW IT IS CHECKED. A minimum at either end of
the swept bracket is not an optimum, so both channels are read through `resolved()` and
discarded when they sit at an edge. And the amplitude ratios are clock-free while the
periods are not, so the two are scored separately: agreement between two channels with
different clock exposure is a much stronger statement than either alone.
"""

import json

import numpy as np

import records
from bounce_sweep import RATIO, sequence
from frequency_space import KEEP, SPAN, _families, measured

CLOCK_DENSITY = 998.0
CHANNELS = ("ratio", "period")


def resolved(values):
  """The argmin, or `None` when it sits at a bracket edge and is therefore not an optimum."""
  if not values: return None
  order = sorted(values)
  best = min(values, key=values.get)
  return None if best in (order[0], order[-1]) else best


def _bounces(dataset, density, material_values, n):
  import pyimr

  times, _, _, maximum, stretch = records.load(dataset)
  product = material_values["galpha"]
  material = pyimr.QuadraticZener(float(np.sqrt(product * RATIO)), material_values["mu"],
                                  material_values["lambda1"], 0.0,
                                  float(np.sqrt(product / RATIO)))
  physics = pyimr.PhysicalParameters(medium_density_kg_m3=density)
  fine = np.linspace(times[0], times[-1], 20001)
  config = pyimr.SimulationConfig(maximum, maximum / stretch, material,
                                  dynamics="keller-miksis", rtol=1e-9, atol=1e-11,
                                  max_steps=2_000_000, physics=physics)
  trace = np.asarray(pyimr.simulate(fine, config).radius_ratio, dtype=float)
  phase = (fine - fine[0]) / (fine[-1] - fine[0]) * SPAN
  s = sequence(trace, phase)
  if s is None: return None
  fam = _families(s["amplitudes"], s["times"])
  return None if len(fam[1]) < n else {name: np.array(fam[j][:n], dtype=float)
                                       for j, name in enumerate(("ratio", "period"))}


def one(job):
  dataset, free = job
  swept = json.load(open(records.HERE / "dynamics_density.json"))
  densities = [float(x) for x in swept["densities"]]
  fits = swept["records"][dataset][free]["by_density"]

  obs = measured(dataset)
  n = min(KEEP, len(obs["ratio"]), len(obs["period"]))
  events = obs["events"]
  out = {}
  for density in densities:
    entry = fits.get(str(int(density)))
    if entry is None: continue
    got = _bounces(dataset, density, entry["fitted"], n)
    if got is None: continue
    row = {"trace_chi2": entry["chi2_per_n"], "trace_lof": entry["lack_of_fit"]}
    for name in CHANNELS:
      err = np.array(obs[name + "_spread"][:n], dtype=float) / np.sqrt(events)
      gap = (got[name] - np.array(obs[name][:n], dtype=float)) / err
      row[name + "_chi2"] = float(np.mean(gap**2))
    out[str(int(density))] = row
  return (dataset, free), out


def main():
  print("  At each density the material is the one the TRACE fit chose there. The afterbounces")
  print("  only score it. A real medium density is the same number for both halves of the")
  print("  record; a sink for a collapse-time deficit need not be.\n")

  jobs = [(d, f) for d in records.DATASETS for f in ("nothing", "u0_shift")]
  with records.pool(min(len(jobs), 6)) as pool:
    got = dict(pool.map(one, jobs))

  summary = {}
  for (dataset, free), rows in got.items():
    if not rows: continue
    keys = sorted(rows, key=lambda s: float(s))
    print(f"  ==== {dataset}, {free} ====")
    print(f"  {'rho':>12s} " + " ".join(f"{k:>8s}" for k in keys))
    for label, field in (("trace chi2/n", "trace_chi2"), ("bounce ratios", "ratio_chi2"),
                         ("bounce periods", "period_chi2")):
      print(f"  {label:>12s} " + " ".join(f"{rows[k][field]:8.2f}" for k in keys))
    picks = {}
    for label, field in (("trace", "trace_chi2"), ("ratio", "ratio_chi2"),
                         ("period", "period_chi2")):
      picks[label] = resolved({float(k): rows[k][field] for k in keys})
    summary[f"{dataset}/{free}"] = {"rows": rows, "argmin": picks}
    shown = {k: (f"{v:.0f}" if v else "edge (discarded)") for k, v in picks.items()}
    print(f"  preferred rho: trace {shown['trace']}, bounce ratios {shown['ratio']}, "
          f"bounce periods {shown['period']}\n")

  print("  ---- what it says ----\n")
  agree = disagree = 0
  for key, s in summary.items():
    t, r, p = (s["argmin"][k] for k in ("trace", "ratio", "period"))
    for name, v in (("ratios", r), ("periods", p)):
      if t is None or v is None: continue
      if v == t: agree += 1
      else: disagree += 1
      print(f"  {key:>26s}: trace wants {t:.0f}, bounce {name} want {v:.0f}"
            + ("   AGREE" if v == t else "   DISAGREE"))
  print(f"\n  {agree} channels agree with the trace, {disagree} disagree.")
  print("  A density acts on the whole record. If the halves want different numbers, the")
  print("  quantity being fitted is not a property of the medium.")
  json.dump(summary, open(records.HERE / "bounce_density.json", "w"), indent=1)


if __name__ == "__main__":
  main()
