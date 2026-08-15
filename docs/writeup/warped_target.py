r"""Does aligning the fit target remove the collapse-time deficit, or only the depth bias?

Two verified defects sit next to each other and might be one thing. \#274: the fit target is
`record["mean"]`, a sample mean over trials at fixed lab time, and the trials' first-collapse
times scatter by about $2\%$, so averaging across that jitter inflates the observed minimum by
$10$--$20\%$ --- the mean curve is not a solution of the equations and no spherical model passes
through it. \#273, as it now stands: with the clock pinned to the acquisition convention, every
record prefers a medium density of $1120$--$1200$ against gelatin's $1015$, which is a collapse
time $5$--$9\%$ too slow, and \texttt{RHO}$=1064$ has been absorbing part of it.

WHY THEY MIGHT BE ONE DEFECT. Jitter does not only fill in the trough, it \emph{widens} it: the
mean of traces whose minima are spread over $2\%$ of $t_c$ has a broader, shallower collapse than
any single trace. A model fitted to a widened trough can buy that width by collapsing more
slowly, and inflating $\rho$ is exactly how this parameterisation buys a slower collapse.

WHY THEY MIGHT NOT BE. One is a depth error and the other is a timing error, and the jitter is
symmetric about the median, so to first order it should not move the trough's \emph{centre} at
all. Asserting the connection because both are large and both live in the first collapse is the
mistake \#270 was renamed for.

SO IT IS MEASURED. Each trial is warped onto common knots, the mean and the trial spread are
rebuilt from the warped traces, and the density sweep of `dynamics_density.py` is re-run against
each target. The comparable quantity is the LOCATION of the density optimum within each
condition, not the lack of fit across conditions: warping shrinks the trial spread near the
collapse, which is the denominator of \eqref{eq:lackoffit}, so the absolute statistic is not on
one scale across the three targets and is not read that way here.

THE WARP IS CHECKED BEFORE IT IS USED. \#274 reports the one-knot rescale taking the depth bias
from $+19.6/+9.9/+14.0\%$ to $+16.7/+0.0/+0.0$. If that does not reproduce, nothing below is
worth running, so it is computed and printed first.
"""

import json
import pathlib

import numpy as np

import records
from density_clock import _box, candidate_for, solver_with_start

DATA = pathlib.Path.home() / "fastscratch/papers/paper_imr_windowing/data"
FILES = {"gelatin_15C": "Ga_t15_exp_data.csv", "gelatin_23C": "Ga_t23_exp_data.csv",
         "gelatin_33C": "Ga_t33_exp_data.csv"}
MAX_RATIO = 1.05
CLOCK_DENSITY = 998.0
GELATIN = 1015.0
# Widened after the first run put three argmins on the 1320 end. An argmin at a bracket edge is
# not an optimum -- it says the bracket was too narrow -- so the range is extended until every
# cell either turns over inside it or is reported unresolved.
#
# It cannot be extended indefinitely, and the limit is physical rather than chosen: the NASG
# liquid has a covolume `b`, so `b * rho < 1` is required for the equation of state to admit the
# density at all, capping it at 1/b = 1513. A cell still wanting more than that is not asking for
# a denser liquid, it is asking for one this equation of state cannot represent -- which is a
# result, and is reported as unresolved rather than as a number.
DENSITIES = (998.0, 1015.0, 1064.0, 1120.0, 1200.0, 1320.0, 1450.0)
TARGETS = ("raw", "one_knot", "two_knot", "sham")
CONFIGURATIONS = [(), ("u0_shift",)]


def resolved(values):
  """The argmin, or `None` when it sits at either end of the bracket.

  An argmin on a boundary is not a located optimum: it reports that the search range stopped
  before the function turned over. Returning a number there invites it to be read, differenced
  and quoted as though it were a measurement, which is exactly what happened to the first run
  of this sweep -- three cells landed on the top end and the direction of the effect was then
  read off them. Callers get `None` and drop the cell.
  """
  if not values: return None
  order = sorted(values)
  best = min(values, key=values.get)
  return None if best in (order[0], order[-1]) else best


def local_minima(trace, floor=3):
  """Indices of local minima, earliest first, ignoring the first few samples."""
  return [i for i in range(floor, len(trace) - 1)
          if trace[i] <= trace[i - 1] and trace[i] < trace[i + 1]]


def load_trials(dataset):
  """`(tau, trials, keep)` on the same screened sample set `collect.py` uses.

  The screen is taken on the RAW trials so the kept sample set -- and therefore the time axis
  -- is identical across every target. Warping then rebuilds the mean and spread on that fixed
  grid, which is what makes the three conditions comparable at all.
  """
  table = np.loadtxt(DATA / FILES[dataset], delimiter=",", ndmin=2)
  tau, trials = table[:, 0], table[:, 1:].T
  keep = ~(trials > MAX_RATIO).any(axis=0) & (trials.std(axis=0, ddof=1) > 0.0)
  return tau[keep], trials[:, keep]


def warp(tau, trials, knots, sham=False):
  """Resample each trial so its own knots land on the median ones.

  `knots=1` maps (peak, first minimum) linearly; `knots=2` adds the second minimum and is
  piecewise linear between, extrapolating past the last knot with the final segment's slope.
  Returns `None` when a trial lacks the minima the warp needs, so the caller can say so rather
  than silently averaging a different set of trials.
  """
  found = []
  for trace in trials:
    m = local_minima(trace)
    if len(m) < knots: return None
    found.append([tau[i] for i in m[:knots]])
  reference = np.median(np.array(found, dtype=float), axis=0)
  # The sham hands each trace its NEIGHBOUR's knots. The knot geometry stays realistic and the
  # interpolation is identical, so whatever the resampling does by itself it does here too --
  # only the match to the event is broken. sec:latent uses exactly this device, and without it
  # a lag-one that rises after warping cannot be told from linear interpolation low-pass
  # filtering the target.
  if sham: found = found[1:] + found[:1]

  out = []
  for trace, own in zip(trials, found, strict=True):
    if knots == 1:
      source = tau * (own[0] / reference[0])
    else:
      a_j, b_j = own
      a_m, b_m = reference
      slope = (b_j - a_j) / (b_m - a_m)
      source = np.where(tau <= a_m, tau * (a_j / a_m), a_j + (tau - a_m) * slope)
    out.append(np.interp(source, tau, trace))
  return np.asarray(out, dtype=float), reference


def build_targets(dataset):
  """`{name: (mean, spread)}` for the raw target and each warp, plus the depth diagnostics."""
  tau, trials = load_trials(dataset)
  per_trial_depth = float(np.mean([t[local_minima(t)[0]] for t in trials]))

  built, report = {}, {}
  for name, knots in (("raw", 0), ("one_knot", 1), ("two_knot", 2), ("sham", 1)):
    if knots == 0:
      warped = trials
      reference = None
    else:
      got = warp(tau, trials, knots, sham=(name == "sham"))
      if got is None:
        report[name] = {"failed": f"a trial has fewer than {knots} resolved minima"}
        continue
      warped, reference = got
    mean = warped.mean(axis=0)
    spread = warped.std(axis=0, ddof=1)
    depth = float(mean[local_minima(mean)[0]])
    built[name] = (mean, np.where(spread > 0.0, spread, np.nan))
    report[name] = {"min_of_mean": depth, "mean_of_mins": per_trial_depth,
                    "bias": depth / per_trial_depth - 1.0,
                    "median_spread": float(np.nanmedian(spread)),
                    "knots": None if reference is None else [float(v) for v in reference]}
  return tau, built, report


def one(job):
  """One fit: a record, a target, free axes, and a dynamics density."""
  from pyimr.noise import lack_of_fit
  from pyimr.selection import evaluate_at, fit_candidate, physical_from_unit

  dataset, target, free, density = job
  times, _, _, maximum, stretch = records.load(dataset)
  cache = json.load(open(records.HERE / "warped_target_cache.json"))[dataset][target]
  mean = np.asarray(cache["mean"], dtype=float)
  spread = np.asarray(cache["spread"], dtype=float)
  candidate = candidate_for(free)
  try:
    solve = solver_with_start(times, maximum, stretch, CLOCK_DENSITY, density)
  except ValueError as error:                                    # e.g. b * rho >= 1
    return job, {"failed": f"inadmissible density: {error}"}
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
  print("  ---- step 1: does the warp remove the depth bias? ----\n")
  print(f"  {'record':>14s} {'target':>10s} {'min of mean':>12s} {'mean of mins':>13s} "
        f"{'bias':>8s} {'median spread':>14s}")
  cache, reports = {}, {}
  for dataset in records.DATASETS:
    _, built, report = build_targets(dataset)
    cache[dataset] = {k: {"mean": v[0].tolist(),
                          "spread": np.nan_to_num(v[1], nan=float(np.nanmedian(v[1]))).tolist()}
                      for k, v in built.items()}
    reports[dataset] = report
    for name in TARGETS:
      r = report.get(name, {})
      if "failed" in r:
        print(f"  {dataset:>14s} {name:>10s}  {r['failed']}")
        continue
      print(f"  {dataset:>14s} {name:>10s} {r['min_of_mean']:12.4f} {r['mean_of_mins']:13.4f} "
            f"{100 * r['bias']:7.1f}% {r['median_spread']:14.5f}")
  json.dump(cache, open(records.HERE / "warped_target_cache.json", "w"))

  print("\n  #274 reports the one-knot rescale taking +19.6/+9.9/+14.0% to +16.7/+0.0/+0.0.")
  available = [t for t in TARGETS
               if all("failed" not in reports[d].get(t, {"failed": 1}) for d in records.DATASETS)]
  print(f"  targets usable on all three records: {available}\n")

  print("  ---- step 2: where does each target put the density optimum? ----\n")
  # Reuse whatever a previous run already computed. Widening the bracket must not cost a full
  # recomputation of the cells that were already inside it, and the fits are deterministic
  # given (record, target, free, density), so a cached lack of fit is the same number.
  held = {}
  previous = records.HERE / "warped_target.json"
  if previous.exists():
    for dataset, labels in json.load(open(previous)).get("sweep", {}).items():
      for label, targets in labels.items():
        free = () if label == "nothing" else tuple(label.split("+"))
        for target, payload in targets.items():
          for rho, value in payload.get("by_density", {}).items():
            held[(dataset, target, free, float(rho))] = {"lack_of_fit": float(value)}

  jobs = [(d, t, free, rho) for d in records.DATASETS for t in available
          for free in CONFIGURATIONS for rho in DENSITIES
          if (d, t, free, rho) not in held]
  print(f"  {len(held)} cells reused, {len(jobs)} to compute, "
        f"clock pinned at {CLOCK_DENSITY:.0f} ...", flush=True)
  got = dict(held)
  if jobs:
    with records.pool(len(jobs)) as pool:
      got |= dict(pool.map(one, jobs))

  summary = {"depth": reports, "densities": list(DENSITIES), "clock": CLOCK_DENSITY,
             "sweep": {}}
  for free in CONFIGURATIONS:
    label = "+".join(free) or "nothing"
    print(f"\n  ==== lack of fit, {label} free  (compare ACROSS a row, never down a column) ====\n")
    for dataset in records.DATASETS:
      print(f"  {dataset}")
      print(f"  {'target':>10s}  " + " ".join(f"{r:8.0f}" for r in DENSITIES) + "     best")
      for target in available:
        cells, values = [], {}
        for rho in DENSITIES:
          v = got[(dataset, target, free, rho)]
          cells.append(f"{'failed':>8s}" if "failed" in v else f"{v['lack_of_fit']:8.2f}")
          if "failed" not in v: values[rho] = v["lack_of_fit"]
        best = resolved(values)
        summary["sweep"].setdefault(dataset, {}).setdefault(label, {})[target] = {
          "by_density": {str(int(k)): float(v) for k, v in values.items()},
          "best_density": None if best is None else float(best)}
        shown = "UNRESOLVED" if best is None else f"{best:7.0f}"
        print(f"  {target:>10s}  " + " ".join(cells) + f"  {shown:>10s}")
      print()

  print("  ---- what it says ----\n")
  print("  A cell whose argmin sits at either end of the bracket is reported UNRESOLVED, not as")
  print("  a best: it says the bracket was too narrow and nothing more. Such a cell is excluded")
  print("  from every comparison below rather than carried with a caveat.\n")
  for dataset in records.DATASETS:
    row = summary["sweep"][dataset]["nothing"]
    line = "  ".join(
      f"{t}: " + ("unresolved" if row[t]["best_density"] is None else f"{row[t]['best_density']:.0f}")
      for t in available if t in row)
    print(f"  {dataset:>14s}  {line}")

  toward, comparable, unresolved = [], [], 0
  for dataset in records.DATASETS:
    row = summary["sweep"][dataset]["nothing"]
    for target in available[1:]:
      raw, warped = row.get("raw", {}), row.get(target, {})
      if raw.get("best_density") is None or warped.get("best_density") is None:
        unresolved += 1
        continue
      comparable.append((dataset, target))
      toward.append(abs(warped["best_density"] - GELATIN) < abs(raw["best_density"] - GELATIN))
  summary["comparable_cells"] = [f"{d}/{t}" for d, t in comparable]
  summary["unresolved_cells"] = unresolved
  summary["optimum_moves_toward_gelatin"] = bool(all(toward)) if toward else None
  print(f"\n  {len(comparable)} comparable cells, {unresolved} dropped as unresolved")
  print(f"  the optimum moves toward gelatin in {sum(toward)} of {len(comparable)}")
  if comparable:
    print("  comparable cells: " + ", ".join(f"{d.replace('gelatin_', '')}/{t}"
                                             for d, t in comparable))
  print("\n  If the optimum reaches ~1015, the two defects are one and both close together.")
  print("  If it stays well above, the depth bias and the collapse-time deficit are separate.")
  json.dump(summary, open(records.HERE / "warped_target.json", "w"), indent=1)


if __name__ == "__main__":
  main()
