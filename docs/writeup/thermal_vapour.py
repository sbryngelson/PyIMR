r"""Does the thermal treatment, run with the vapour it needs, explain what $\mu$ is absorbing?

Three unexplained results point the same way. The datasets want a medium density $10$--$18\%$
above gelatin's, a collapse $5$--$9\%$ slower (\cref{sec:limitations}). The identified $\mu$ sits
at $44$, $52$ and $66$ times water's viscosity and \emph{rises} as the gel warms and softens, an
activation energy of \SI{-19.3}{\kilo\joule\per\mole} that no liquid has. And switching on the
vapour pressure this package leaves off improved every fit and halved the $g\alpha$ trend. A
denser medium, a larger viscosity and a higher internal pressure all slow a collapse, so the
hypothesis is that the model collapses too fast and the fit buys the slowing wherever it can.

THE DIRECTION WAS CHECKED BEFORE THIS WAS RUN, because the hypothesis is only coherent if the
candidate physics moves the collapse the right way. On a grid twenty times finer than the
record's, full physics moves the first minimum LATER by $1.14$, $1.97$ and \SI{3.31}{\percent} at
$15$, $23$ and \SI{33}{\celsius}: the right sign, graded with temperature as the $\mu$ anomaly
requires, and between a quarter and a half of the $5$--$9\%$ wanted. \texttt{bubtherm} alone
contributes about \SI{0.2}{\percent}, so the effect is essentially all vapour and mass transfer,
which is what \cref{sec:thermal} leaves out.

THE CANDIDATE IS THE AXIS THIS DOCUMENT SIZED AND NEVER RAN PROPERLY. \Cref{sec:axes} makes the
thermal treatment the largest of the three model axes. \Cref{sec:thermal} then scores it with
\texttt{thermal.py}, which sets \texttt{bubtherm} and \texttt{medtherm} and neither \texttt{vapor}
nor \texttt{masstrans}; the validator requires all three together, so that comparison carries gas
conduction with no latent heat and no condensation.

WHAT THIS RUN ESTABLISHED, WHICH IS NOT WHAT IT SET OUT TO. The fit-based half FAILED, and the
failure is diagnosable rather than mysterious. Profiled, one solve costs \SI{15}{\milli\second}
cold, \SI{372}{\milli\second} with \texttt{bubtherm} and \SI{3356}{\milli\second} with vapour
and mass transfer: a factor of $223$, which at $20\times600$ starts and evaluations makes a single
full-physics cell an eleven-hour job. Cutting the budget to $6\times250$ to afford it broke the
search on a surface \texttt{thermal.py} already documents as multimodal, and three independent
tells say so. Full physics fits WORSE than the \texttt{bubtherm} model it contains ($14.06$,
$6.05$, $3.21$ against $11.38$, $3.09$, $1.83$), which is impossible for a superset seeded from
that model's own optimum. It returns $\mu = \SI{0.53}{\pascal\second}$ at \SI{33}{\celsius},
five hundred times water. And the $g\alpha$ slope reverses, making a warming gel stiffen. Even
the cheap cold row fails to reproduce \texttt{vapour\_trend.py} at a larger budget, $0.0858$
against $0.0723$ at \SI{33}{\celsius}.

So the hypothesis is UNTESTED, not refuted, and the useful finding is the price: this question
costs about eleven hours per cell and is not answerable at a discount. The nesting gate below now
detects the failure rather than leaving it to be noticed.

TWO CONTROLS THAT AN EARLIER VERSION OF THIS STUDY GOT WRONG, recorded because both were silent
failures rather than errors. The published thermal fit runs at $T_8 = \SI{298.15}{\kelvin}$ on
every dataset, so a row claiming to reproduce it must too; setting $T_8$ correctly everywhere
made the baseline something \cref{sec:thermal} never ran, and with \texttt{bubtherm} on $T_8$
reaches the conductivities, so the difference is real. Both rows are kept, which also isolates
what $T_8$ alone is worth once it stops being inert. And the density question cannot be answered
by comparing two values: \texttt{dynamics\_density.py} puts the optimum at $1120$--$1200$, so
$1064$ and $1015$ sit on the SAME side of it and $1064$ wins whatever the physics does. It needs
a sweep, and a sweep at full physics is eighteen eleven-hour cells, so it is deferred to a second
stage that is only worth paying for if the trend below moves.
"""

import json

import numpy as np

import records

RATIO = 38.5
BOX = {"mu": (1e-5, 1e1), "galpha": (1e0, 1e7), "lambda1": (1e-9, 1e-2)}
# Profiled rather than guessed: one solve costs 15 ms cold, 372 ms with bubtherm and
# 3356 ms with vapour and mass transfer, a factor of 223. A fit is STARTS x EVALUATIONS
# solves, so a uniform 20 x 600 makes a full-physics cell an ELEVEN HOUR job and the whole
# wave with it. The budget is therefore set per configuration against its own solve cost,
# and the full-physics cells are seeded from their own bubtherm optimum rather than the
# cold one, which is a much shorter distance for a smaller search to cover.
BUDGET = {"cold": (20, 600), "bubtherm": (12, 400), "full": (6, 250)}
KELVIN = {"gelatin_15C": 288.15, "gelatin_23C": 296.15, "gelatin_33C": 306.15}
PUBLISHED_T8, INHERITED, GELATIN = 298.15, 1064.0, 1015.0
GAS_CONSTANT = 8.314462618

_THERMAL = {"bubtherm": 1, "Nt": 11}
_FULL = {"bubtherm": 1, "Nt": 11, "vapor": 1, "masstrans": 1}
# (label, options, correct T8, density, budget key)
CASES = [
  ("cold", {}, False, INHERITED, "cold"),
  ("bubble, as published", _THERMAL, False, INHERITED, "bubtherm"),
  ("bubble, T8 correct", _THERMAL, True, INHERITED, "bubtherm"),
  ("bubble+vapour+mass", _FULL, True, INHERITED, "full"),
]
# The density sweep is DEFERRED. Six densities at full physics is eighteen eleven-hour cells,
# and it is only worth paying for if the trend below moves. Staging it costs nothing, since
# the trend answer does not depend on it.
DENSITIES = ()


def candidate():
  import pyimr
  from pyimr.selection import CandidateModel

  def build(t):
    product = t["galpha"]
    return pyimr.QuadraticZener(float(np.sqrt(product * RATIO)), t["mu"], t["lambda1"], 0.0,
                                float(np.sqrt(product / RATIO)))

  return CandidateModel("qSLS|identified", build, ("mu", "galpha", "lambda1"))


def _solver(dataset, options, correct_t8, density):
  import pyimr

  times, _, _, maximum, stretch = records.load(dataset)
  return records.solver(times, maximum, stretch, max_steps=800_000,
                        T8=KELVIN[dataset] if correct_t8 else PUBLISHED_T8,
                        physics=pyimr.PhysicalParameters(medium_density_kg_m3=density),
                        **options)


def fit(dataset, label, seed):
  """One cell, at the budget its own solve cost affords."""
  _, options, correct_t8, density, key = next(c for c in CASES if c[0] == label)
  starts, evaluations = BUDGET[key]
  _, mean, spread, _, _ = records.load(dataset)
  try:
    return records.score(candidate(), _solver(dataset, options, correct_t8, density),
                         mean, spread, bounds=BOX, starts=starts, evaluations=evaluations,
                         trials=records.trial_count(dataset),
                         seeds=None if seed is None else [seed])
  except Exception as error:                                                 # noqa: BLE001
    return {"failed": f"{type(error).__name__}: {error}"}


def stage(job):
  """A dataset's whole chain, so each fit seeds the next and the expensive one starts close."""
  dataset = job
  out, seed = {}, None
  for label, *_ in CASES:
    result = fit(dataset, label, seed)
    out[label] = result
    if isinstance(result, dict) and "unit" in result:
      seed = result["unit"]          # cold seeds bubtherm, bubtherm seeds full physics
  return dataset, out


def arrhenius(values):
  t = np.array([KELVIN[d] for d in records.DATASETS], dtype=float)
  x = 1.0 / t - 1.0 / t.mean()
  slope = float(np.polyfit(x, np.log(np.array(values, dtype=float)), 1)[0])
  return slope, slope * GAS_CONSTANT / 1000.0


def main():
  for key, (a, b) in BUDGET.items():
    cost = {"cold": 0.015, "bubtherm": 0.372, "full": 3.356}[key]
    print(f"  {key:>9s}: {a:2d} starts x {b:3d} evals = {a*b:5d} solves at {cost*1000:.0f} ms "
          f"= {a*b*cost/3600:.2f} h per fit")
  print(f"\n  {len(records.DATASETS)} chains of {len(CASES)}, each fit seeding the next ...",
        flush=True)

  with records.pool(len(records.DATASETS)) as pool:
    got = dict(pool.map(stage, list(records.DATASETS)))

  summary = {"kelvin": KELVIN, "budget": BUDGET, "cells": {}, "trend": {}}
  for d, chain in got.items():
    for label, v in chain.items():
      if isinstance(v, dict) and "fitted" in v:
        summary["cells"][f"{d}|{label}"] = {
          "fitted": v["fitted"], "chi2_per_n": v.get("chi2_per_n"),
          "lack_of_fit": v.get("lack_of_fit"), "log_evidence": v.get("log_evidence")}

  print("\n  ==== what mu does as the physics is completed ====\n")
  print(f"  {'configuration':>22s} {'mu 15C':>8s} {'mu 23C':>8s} {'mu 33C':>8s} "
        f"{'E_a kJ/mol':>11s} {'ga slope':>9s} {'mean l.o.f.':>11s}")
  for label, *_ in CASES:
    mus, gas, lof = [], [], []
    for d in records.DATASETS:
      v = got.get(d, {}).get(label) or {}
      if "fitted" not in v: continue
      mus.append(v["fitted"]["mu"]); gas.append(v["fitted"]["galpha"])
      lof.append(v.get("lack_of_fit", np.nan))
    if len(mus) != 3:
      print(f"  {label:>22s}   incomplete ({len(mus)} of 3)"); continue
    _, ea = arrhenius(mus); b_ga, _ = arrhenius(gas)
    summary["trend"][label] = {"mu": mus, "galpha": gas, "activation_kj_mol": ea,
                               "b_galpha": b_ga, "lack_of_fit": lof}
    print(f"  {label:>22s} {mus[0]:8.4f} {mus[1]:8.4f} {mus[2]:8.4f} {ea:+11.1f} "
          f"{b_ga:+9.0f} {np.nanmean(lof):11.2f}")

  # A superset seeded from its own subset's optimum cannot fit worse at a converged search,
  # so this is the gate that separates "the physics does not help" from "the search failed".
  print("\n  ---- nesting gate ----\n")
  sub, sup = summary["trend"].get("bubble, T8 correct"), summary["trend"].get("bubble+vapour+mass")
  if sub and sup:
    worse = [i for i in range(3) if sup["lack_of_fit"][i] > sub["lack_of_fit"][i] * 1.001]
    summary["nesting_gate_failed"] = bool(worse)
    print(f"  bubtherm  {[round(x, 2) for x in sub['lack_of_fit']]}")
    print(f"  full      {[round(x, 2) for x in sup['lack_of_fit']]}")
    print(f"  full physics fits WORSE on {len(worse)} of 3 datasets"
          f"{'   *** SEARCH FAILED, the rows below are not readable ***' if worse else ''}")

  print("\n  ---- what it says ----\n")
  cold_t, full_t = summary["trend"].get("cold"), summary["trend"].get("bubble+vapour+mass")
  if cold_t and full_t:
    print(f"  activation energy for mu: {cold_t['activation_kj_mol']:+.1f} kJ/mol cold, "
          f"{full_t['activation_kj_mol']:+.1f} at full physics")
    summary["activation_turns_physical"] = bool(full_t["activation_kj_mol"] > 0)
    print(f"  it {'TURNS POSITIVE' if full_t['activation_kj_mol'] > 0 else 'stays negative'}")
    print(f"  g*alpha slope: {cold_t['b_galpha']:+.0f} K cold, {full_t['b_galpha']:+.0f} K full")
  if summary.get("nesting_gate_failed"):
    print("\n  The gate failed, so neither number above is evidence about the physics. What the")
    print("  run establishes is the price: 3356 ms a solve, about eleven hours a cell at a")
    print("  budget this surface needs, and no cheaper search that holds.")
  else:
    print("\n  If it turns positive the density sweep is worth its eighteen cells.")
  json.dump(summary, open(records.HERE / "thermal_vapour.json", "w"), indent=1)


if __name__ == "__main__":
  main()
