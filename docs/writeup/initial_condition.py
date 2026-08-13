r"""Fit how the records BEGIN, with the stretch treated as unknown.

\Cref{sec:deltascreen} screens a non-zero initial wall velocity at $37.6\%$ of $\hat\delta$ ---
the largest candidate ever put through that screen, against $23.7\%$ for the initial radius and
$6.5\%$ for the best genuine physics on the constitutive and transport list. A screen is an
upper bound with the amplitude free. This fits it.

THE STRETCH IS NOT KNOWN AND IS NOT ASSUMED HERE. \Cref{sec:reqprior} establishes that
$R_{\max}/R_{\rm eq}$ is a population mean applied to every trial, tabulated at $7.09\pm0.36$,
$7.37\pm1.28$ and $6.83\pm0.49$ --- five to seventeen percent. Fitting $u_0$ against a stretch
pinned at its stored value would condition a new claim on an old one that is known to be
uncertain by more than the effect being measured. So `req_scale` is free in every fit below, over
a box wide enough that nothing pins, and the question asked of $u_0$ is what it adds ON TOP of
that freedom rather than what it can do while the stretch absorbs the difference.

$u_0$ IS SIGNED AND THE COORDINATES ARE NOT. Every axis in `physical_from_unit` is log-spaced and
therefore positive, so the velocity enters as $u_0 = (s - 1)\,U$ with $s$ a positive axis and $U$
the Rayleigh scale $R_{\max}/t_c$. A box $(w, 1/w)$ is then symmetric in the fitted coordinate
and puts $u_0 = 0$ exactly at its centre, which is what makes ``consistent with zero'' a
statement the fit can actually make.

THE DISCRIMINATOR IS LACK OF FIT, NOT $\chi^2$. A free parameter always buys $\chi^2$; that is
what free parameters do, and reading it as evidence is the mistake \cref{sec:reqprior} was
written to avoid. \Cref{eq:lackoffit} compares the misfit against PURE ERROR, which costs no
model, so it is the statistic that distinguishes a genuine correction from one more place to hide
the residual.

AND THE TWO MAY NOT BE SEPARATELY IDENTIFIED. A bubble released slightly early and a bubble with
slightly more gas both reach the collapse differently, so `req_scale` and $u_0$ could easily
trade. That is the $g$-against-$\alpha$ failure of \cref{sec:identifiability} waiting to happen
again, and it is measured here rather than hoped away: the correlation between the two is
reported alongside every fit that frees both.
"""

import json

import numpy as np

import records
from identified import BOX

RATIO = 38.5
AXES = ("mu", "galpha", "lambda1")
# 32, not 16. At 16 the nesting gate below failed on two records -- a wider box scoring WORSE
# than a narrower one it contains, which is arithmetically impossible at a converged optimum
# and so a search that stopped early. sec:reqprior met the same thing and answered it the same
# way, at 10 restarts against 24.
STARTS, EVALUATIONS = 32, 800
# half-widths swept rather than chosen, exactly as sec:reqprior sweeps the Req prior
VELOCITY_BOXES = {"+-2%": (0.98, 1.0 / 0.98), "+-10%": (0.90, 1.0 / 0.90),
                  "+-30%": (0.70, 1.0 / 0.70)}
REQ_BOX = (0.6, 1.6)                     # wide enough that nothing pins; see sec:reqprior


def solver_with_start(times, maximum, stretch):
  """`solve(material, config_axes)` that reads BOTH `req_scale` and `u0_shift`."""
  import pyimr
  from pyimr.noise import characteristic_time

  velocity_scale = maximum / characteristic_time(maximum)

  def solve(material, config_axes=None):
    axes = config_axes or {}
    scale = float(axes.get("req_scale", 1.0))
    shift = float(axes.get("u0_shift", 1.0))
    config = pyimr.SimulationConfig(
      maximum, maximum / stretch * scale, material, dynamics="keller-miksis",
      rtol=1e-8, atol=1e-10, max_steps=400_000,
      initial=pyimr.InitialState(wall_velocity_m_s=(shift - 1.0) * velocity_scale))
    result = pyimr.simulate(times, config)
    return np.asarray(result.radius_ratio, dtype=float), None

  return solve


def candidate_for(free):
  """The identified qSLS with whichever configuration axes `free` names."""
  import pyimr
  from pyimr.selection import CandidateModel

  def build(t):
    product = t["galpha"]
    return pyimr.QuadraticZener(float(np.sqrt(product * RATIO)), t["mu"], t["lambda1"], 0.0,
                                float(np.sqrt(product / RATIO)))

  extra = tuple(free)
  return CandidateModel(f"qSLS+{'+'.join(extra) or 'none'}", build, (*AXES, *extra), (), extra)


def _box(free, width):
  box = dict(BOX)
  if "req_scale" in free: box["req_scale"] = REQ_BOX
  if "u0_shift" in free: box["u0_shift"] = VELOCITY_BOXES[width]
  return box


def one(job):
  """One fit: the named configuration axes free, at one velocity-prior width."""
  from pyimr.noise import lack_of_fit
  from pyimr.selection import evaluate_at, fit_candidate, physical_from_unit

  dataset, free, width = job
  times, mean, spread, maximum, stretch = records.load(dataset)
  candidate = candidate_for(free)
  solve = solver_with_start(times, maximum, stretch)
  box = _box(free, width)
  try:
    fit = fit_candidate(candidate, solve, mean, spread, bounds=box, starts=STARTS,
                        max_evaluations=EVALUATIONS)
  except ValueError as error:
    return job, {"failed": str(error)}
  axes = tuple(candidate.axes)
  fitted = dict(zip(axes, (float(v) for v in physical_from_unit(axes, fit.unit, box)),
                    strict=True))
  model = np.asarray(evaluate_at(candidate, solve, fitted)[0], dtype=float)
  ratio = lack_of_fit(mean, model, spread, records.trial_count(dataset),
                      candidate.dimension).ratio
  pinned = [k for k in free
            if min(abs(np.log(fitted[k] / box[k][0])), abs(np.log(fitted[k] / box[k][1]))) < 1e-3]
  return job, {"chi2_per_n": float(fit.chi_squared), "lack_of_fit": float(ratio),
               "fitted": fitted, "pinned": pinned, "unit": [float(v) for v in fit.unit]}


def correlation(dataset, width, unit_point):
  """`corr(req_scale, u0_shift)` from the Fisher matrix at an already-fitted point.

  If the two trade, freeing both measures neither -- which is the g-against-alpha failure of
  sec:identifiability, and the reason this is computed rather than assumed away. The fitted
  point is passed in rather than refitted: doing it again was a serial pass costing more than
  the parallel sweep it was reporting on.
  """
  from pyimr.selection import physical_from_unit

  free = ("req_scale", "u0_shift")
  times, mean, spread, maximum, stretch = records.load(dataset)
  candidate = candidate_for(free)
  solve = solver_with_start(times, maximum, stretch)
  box = _box(free, width)
  axes = tuple(candidate.axes)
  unit = np.asarray(unit_point, dtype=float)
  step = 1e-3
  columns = []
  from pyimr.selection import evaluate_at
  for k in range(len(axes)):
    traces = []
    for direction in (step, -step):
      moved = np.clip(unit + direction * np.eye(len(axes))[k], 0.0, 1.0)
      values = physical_from_unit(axes, moved, box)
      traces.append(np.asarray(evaluate_at(
        candidate, solve, dict(zip(axes, (float(v) for v in values), strict=True)))[0],
        dtype=float))
    columns.append((traces[0] - traces[1]) / (2.0 * step) / spread)
  jacobian = np.column_stack(columns)
  covariance = np.linalg.inv(jacobian.T @ jacobian)
  sd = np.sqrt(np.diag(covariance))
  index = {a: i for i, a in enumerate(axes)}
  i, j = index["req_scale"], index["u0_shift"]
  return float(covariance[i, j] / (sd[i] * sd[j])), float(np.linalg.cond(jacobian.T @ jacobian))


def main():
  configurations = [(), ("req_scale",), ("u0_shift",), ("req_scale", "u0_shift")]
  jobs = [(d, free, width) for d in records.DATASETS for free in configurations
          for width in (VELOCITY_BOXES if "u0_shift" in free else ["+-2%"])]
  print(f"  {len(jobs)} mean-trace fits ...", flush=True)
  with records.pool(len(jobs)) as pool:
    table = dict(pool.map(one, jobs))

  summary = {}
  for dataset in records.DATASETS:
    print(f"\n=== {dataset} ===\n")
    print(f"  {'free':24s} {'u0 width':>9s} {'chi2/N':>8s} {'lack of fit':>12s} "
          f"{'req_scale':>10s} {'u0 / U':>9s} {'pinned':>16s}")
    entry = {}
    for free in configurations:
      for width in (VELOCITY_BOXES if "u0_shift" in free else ["+-2%"]):
        got = table[(dataset, free, width)]
        if "failed" in got:
          print(f"  {'+'.join(free) or 'nothing':24s} {width:>9s}  failed")
          continue
        f = got["fitted"]
        label = "+".join(free) or "nothing"
        entry[f"{label}|{width}"] = got
        print(f"  {label:24s} {width if free and 'u0_shift' in free else '--':>9s} "
              f"{got['chi2_per_n']:8.4f} {got['lack_of_fit']:12.2f} "
              f"{f.get('req_scale', float('nan')):10.4f} "
              f"{f.get('u0_shift', 1.0) - 1.0:+9.4f} "
              f"{','.join(got['pinned']) or '-':>16s}")
    summary[dataset] = entry

  print("\n  are the two separately identified, or do they trade?\n")
  print(f"  {'record':13s} {'u0 width':>9s} {'corr(req, u0)':>15s} {'cond(J^T J)':>13s}")
  for dataset in records.DATASETS:
    for width in VELOCITY_BOXES:
      joint = table[(dataset, ("req_scale", "u0_shift"), width)]
      if "failed" in joint: continue
      try:
        rho, condition = correlation(dataset, width, joint["unit"])
      except Exception as error:                                             # noqa: BLE001
        print(f"  {dataset:13s} {width:>9s}  failed: {str(error)[:40]}")
        continue
      summary[dataset][f"correlation|{width}"] = {"corr": rho, "condition": condition}
      print(f"  {dataset:13s} {width:>9s} {rho:+15.4f} {condition:13.3e}")

  print("\n  the convergence gate: every wider box CONTAINS every narrower one, so the best")
  print("  attainable chi2 can only fall as the box grows. A cell that rises has not found its\n"
        "  own optimum and is excluded rather than read.\n")
  print(f"  {'record':13s} {'+-2%':>9s} {'+-10%':>9s} {'+-30%':>9s} {'verdict':>28s}")
  order = ["+-2%", "+-10%", "+-30%"]
  for dataset in records.DATASETS:
    values = [summary[dataset].get(f"req_scale+u0_shift|{w}", {}).get("chi2_per_n")
              for w in order]
    if any(v is None for v in values): continue
    ok = all(values[i + 1] <= values[i] + 1e-6 for i in range(len(values) - 1))
    summary[dataset]["nesting_ok"] = bool(ok)
    print(f"  {dataset:13s} " + " ".join(f"{v:9.4f}" for v in values)
          + f" {'converged' if ok else 'NOT CONVERGED -- excluded':>28s}")

  print("\n  Read the lack-of-fit column, not chi2/N: a free parameter always buys chi2, and")
  print("  only the replicate-based statistic says whether the STRUCTURE went away. A")
  print("  correlation near -1 would mean the pair is one direction and neither is measured.")

  json.dump(summary, open(records.HERE / "initial_condition.json", "w"), indent=1)


if __name__ == "__main__":
  main()
