"""Reproducible timings. `python benchmarks/run.py [--json out] [--baseline old] [--budget S]`

Median of repeats, never one shot. Reports `prepared` (re-solving an already
prepared problem, which is what prepare() actually claims) and, for sensitivity
cases, the ratio to one forward solve -- the figure that matters for #25.

Each case carries its own grid: a shared one would be wrong for at least one,
and #44's timings were originally taken over a window that stopped before the
collapse. Cases over budget are listed as skipped, never dropped silently.
Not for CI -- shared-runner timings are noise. See #28.
"""

import argparse, json, platform, statistics, sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

import pyimr as F
from pyimr import sensitivity

R0, REQ = 225e-6, 225e-6 / 6
T0 = R0 / np.sqrt(101325 / 1064)
NHKV = F.NeoHookeanKelvinVoigt(2500.0, 0.1)
LONG = np.linspace(0.0, 120e-6, 300)  # a collapse and several rebounds
SHORT = np.linspace(0.0, 20e-6, 80)  # through the first collapse only
PARAMS = ("material.shear_modulus_pa", "material.viscosity_pa_s", "R0", "Req")

# name, group, times, measured seconds/solve, kwargs, sensitivity directions.
# Costs are measured, not guessed. Most cases use SHORT: the composable path
# re-evaluates quadrature every RHS call, and over LONG that case exceeds 195s
# for a single solve while taking 0.015s over SHORT. A benchmark that takes
# longer than the work it measures does not get run.
CASES = (
  ("nhkv", "forward", LONG, 0.06, dict(), 0),
  ("nhkv-keller-miksis", "forward", LONG, 0.05, dict(dynamics="keller-miksis"), 0),
  ("zener", "forward", LONG, 0.09, dict(material=F.Zener(2500.0, 0.1, 2 * T0, 0.4 * T0)), 0),
  (
    "gent-carreau",
    "forward",
    SHORT,
    0.02,
    dict(material=F.InstantaneousMaterial(elastic=F.Gent(2500.0, 250.0), viscous=F.CarreauYasuda(0.5, 0.02, 20e-6, 2.0, 0.45))),
    0,
  ),
  ("ogden-3-term", "forward", SHORT, 0.13, dict(material=F.InstantaneousMaterial(elastic=F.Ogden((1800.0, 600.0, -300.0), (1.3, 4.0, -2.0)))), 0),
  ("giesekus-gauss-240", "distributed", SHORT, 0.02, dict(material=F.Giesekus(0.1, 2 * T0, 0.4 * T0, 0.2, points=240)), 0),
  ("giesekus-trapz-480", "distributed", SHORT, 0.04, dict(material=F.Giesekus(0.1, 2 * T0, 0.4 * T0, 0.2, points=480, quadrature="trapezoid")), 0),
  ("bubtherm-fd-25", "thermal", SHORT, 0.06, dict(bubtherm=1, Nt=25, thermal="fd"), 0),
  ("bubtherm-spectral-25", "thermal", SHORT, 0.08, dict(bubtherm=1, Nt=25, thermal="spectral"), 0),
  ("coupled-fd", "thermal", SHORT, 0.30, dict(bubtherm=1, medtherm=1, masstrans=1, vapor=1, Nt=25, Mt=25, thermal="fd"), 0),
  ("sens-nhkv-1", "sensitivity", SHORT, 0.2, dict(dynamics="keller-miksis"), 1),
  ("sens-nhkv-4", "sensitivity", SHORT, 0.2, dict(dynamics="keller-miksis"), 4),
  # Off the compiled path (#44): ~90% of runtime is Dual dispatch. Opt-in only.
  ("sens-coupled-fd-1", "sensitivity", SHORT, 900.0, dict(bubtherm=1, medtherm=1, masstrans=1, vapor=1, Nt=7, Mt=7, thermal="fd"), 1),
)


def median(action, repeats):
  action()  # warm-up discarded: numba compiles on the first compiled-path call
  return statistics.median(_timed(action) for _ in range(repeats))


def _timed(action):
  start = perf_counter()
  action()
  return perf_counter() - start


def measure(name, group, times, kwargs, directions, repeats):
  config = F.SimulationConfig(R0=R0, Req=REQ, material=kwargs.pop("material", NHKV), **kwargs)
  problem = F.prepare(config)
  forward = median(lambda: F.simulate(times, config), repeats)
  prepared = median(lambda: problem.solve(times), repeats)
  row = dict(name=name, group=group, state=int(problem.layout.size), forward_s=forward, prepared_s=prepared, prepare_x=forward / prepared)
  if directions:
    sens = median(lambda: sensitivity.solve_with_sensitivities(problem, times, list(PARAMS[:directions])), repeats)
    row.update(directions=directions, sensitivity_s=sens, sens_over_forward=sens / prepared)
  return row


def compare(baseline, results):
  before = {r["name"]: r for r in baseline["results"]}
  print("\nagainst baseline (prepared solve):")
  for r in results:
    old = before.get(r["name"])
    if not old:
      print(f"  {r['name']:<22} {'--':>9}  {r['prepared_s'] * 1e3:>9.2f}ms   new")
      continue
    x = r["prepared_s"] / old["prepared_s"]
    mark = "" if 0.9 <= x <= 1.1 else ("  faster" if x < 0.9 else "  SLOWER")
    print(f"  {r['name']:<22} {old['prepared_s'] * 1e3:>9.2f}ms {r['prepared_s'] * 1e3:>9.2f}ms {x:>6.2f}x{mark}")
  for name in sorted(set(before) - {r["name"] for r in results}):
    print(f"  {name:<22} in baseline, not run now")


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--repeats", type=int, default=3)
  p.add_argument("--budget", type=float, default=1.0, help="skip cases estimated above this (seconds)")
  p.add_argument("--group")
  p.add_argument("--json", type=Path)
  p.add_argument("--baseline", type=Path)
  a = p.parse_args()

  chosen = [c for c in CASES if not a.group or c[1] == a.group]
  run, skip = [c for c in chosen if c[3] <= a.budget], [c for c in chosen if c[3] > a.budget]
  print(
    f"python {platform.python_version()} {platform.machine()}, numpy {np.__version__}; "
    f"{len(run)} case(s), median of {a.repeats}, budget {a.budget:g}s\n"
  )
  head = f"{'case':<22} {'group':<12} {'forward':>10} {'prepared':>10} {'prep x':>7} {'sens':>10} {'sens/fwd':>9}"
  print(head + "\n" + "-" * len(head))

  results = []
  for name, group, times, _cost, kwargs, directions in run:
    r = measure(name, group, times, dict(kwargs), directions, a.repeats)
    results.append(r)
    s, x = r.get("sensitivity_s"), r.get("sens_over_forward")
    print(
      f"{name:<22} {group:<12} {r['forward_s'] * 1e3:>9.2f}ms {r['prepared_s'] * 1e3:>9.2f}ms "
      f"{r['prepare_x']:>7.2f} {(f'{s * 1e3:.1f}ms' if s else '--'):>10} {(f'{x:.1f}x' if x else '--'):>9}",
      flush=True,
    )

  if skip:
    print(f"\nSKIPPED {len(skip)} over budget -- raise --budget to include:")
    for name, _g, _t, cost, *_ in skip:
      print(f"  {name:<22} est. {cost:>6.0f}s")
  if a.json:
    a.json.write_text(
      json.dumps(
        dict(python=platform.python_version(), numpy=np.__version__, repeats=a.repeats, skipped=[c[0] for c in skip], results=results), indent=2
      )
      + "\n"
    )
    print(f"\nwrote {a.json}")
  if a.baseline:
    compare(json.loads(a.baseline.read_text()), results)


if __name__ == "__main__":
  main()
