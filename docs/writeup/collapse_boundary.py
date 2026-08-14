r"""What sets the solvable boundary, and what the sensitivity ridge actually is.

Two things about this geometry space have been asserted repeatedly in this document and
established nowhere. This measures both, and each measurement can refute the story attached
to it.

FIRST, THE BOUNDARY. \#270 records that Keller--Miksis will not integrate at large $R_{\max}$ and
high stretch, and attributes it to deep-collapse stiffness on the strength of \emph{one} failing
geometry, where the bubble reached $R/R_{\rm eq} = 0.157$. One point is an anecdote. If the
mechanism is compression, then the collapse depth of the geometries that DO integrate must
predict which ones do not: there should be a critical $R_{\min}/R_{\rm eq}$ with successes above
it and failures below, and the boundary in $(R_{\max}, s)$ should be a contour of that quantity
rather than a corner of the box. If successes and failures interleave in depth, the compression
story is wrong and \#270 needs rewriting.

SECOND, THE RIDGE. The grid refinement located a sensitivity spike near \SI{937}{\micro\metre} at
stretch $17.6$ --- $d$ rising from $0.5$ to $64$ across roughly $0.6\%$ in stretch --- and put it
in the certified support. It has since been flagged three times and examined none. A spike that
narrow means something about the trajectory changes qualitatively across it. The candidates are
countable: the collapse depth jumps, an afterbounce appears or disappears within the observation
window, or the first minimum crosses the edge of the sampled record. Each leaves a different
signature, and scanning stretch finely while recording all three separates them.

WHY THIS MATTERS RATHER THAN BEING TIDINESS. If the ridge is a real feature of the information
--- a geometry where the data genuinely become much more informative --- it belongs in the
support and the design should exploit it. If it is the observation window clipping a bounce, it
is an artifact of how the record is sampled, it will move when the window changes, and a design
that puts weight there is chasing the grid rather than the physics.
"""

import json

import numpy as np

import records
from identified import candidate_at_ratio
from noise_design import profile, raw_information
from offgrid_and_compound import SOURCE, pieces

RATIO = 38.5
AXES = ("mu", "galpha", "lambda1")
SPAN = 5.0                      # observation window, in characteristic times
SAMPLES = 201
RIDGE_R = 937.2e-6
RIDGE_S = np.linspace(17.0, 18.2, 49)


def trajectory(job):
  """`(depth, minima, first minimum phase, solvable)` at one geometry."""
  import pyimr
  from pyimr.noise import characteristic_time
  from scipy.signal import argrelextrema

  radius, stretch = job
  measured = json.load(open(records.HERE / "per_trial_fits.json"))[SOURCE]
  values = {a: float(measured["median"][a]) for a in AXES}
  times = np.linspace(0.0, SPAN * characteristic_time(radius), SAMPLES)
  config = pyimr.SimulationConfig(radius, radius / stretch, candidate_at_ratio(RATIO).build(values),
                                  dynamics="keller-miksis", rtol=1e-8, atol=1e-10,
                                  max_steps=200_000)
  try:
    trace = np.asarray(pyimr.simulate(times, config).radius_ratio, dtype=float)
  except Exception:                                                          # noqa: BLE001
    return job, None
  if not np.all(np.isfinite(trace)): return job, None
  phase = np.linspace(0.0, SPAN, SAMPLES)
  minima = [i for i in argrelextrema(trace, np.less, order=3)[0] if phase[i] > 0.2]
  return job, {
    # depth against R_eq is the compression the gas term actually sees; against R_max is what
    # the trace shows. sec:270 quotes the first, so it is the one the boundary story must predict
    "depth_req": float(trace.min() * stretch),
    "depth_rmax": float(trace.min()),
    "minima": len(minima),
    "first_min_phase": float(phase[minima[0]]) if minima else float("nan"),
    "last_min_phase": float(phase[minima[-1]]) if minima else float("nan")}


def main():
  summary = {}

  # ---- does collapse depth predict the failure boundary? ----
  radii = np.geomspace(200e-6, 2685e-6, 14)
  stretches = np.geomspace(4.0, 25.9, 10)
  jobs = [(float(r), float(s)) for r in radii for s in stretches]
  print(f"  probing {len(jobs)} geometries for collapse depth ...", flush=True)
  with records.pool(len(jobs)) as pool:
    got = dict(pool.map(trajectory, jobs))

  solved = {d: v for d, v in got.items() if v is not None}
  failed = [d for d, v in got.items() if v is None]
  print(f"  {len(solved)} integrate, {len(failed)} do not\n")
  print("  collapse depth R_min/R_eq of the geometries that integrate, by radius and stretch")
  print("  ('X' did not integrate; the question is whether X sits beyond a depth contour)\n")
  print(f"  {'stretch':>8s}  " + " ".join(f"{r * 1e6:6.0f}" for r in radii))
  for s in stretches:
    cells = []
    for r in radii:
      v = got[(float(r), float(s))]
      cells.append("     X" if v is None else f"{v['depth_req']:6.3f}")
    print(f"  {s:8.2f}  " + " ".join(cells))

  depths = np.array([v["depth_req"] for v in solved.values()])
  print(f"\n  solvable depths span {depths.min():.4f} to {depths.max():.4f} of R_eq")
  # the sharp test: is there a threshold with all failures on one side?
  neighbours = []
  for (r, s), v in solved.items():
    ri = int(np.argmin(np.abs(radii - r)))
    if ri + 1 < len(radii) and got[(float(radii[ri + 1]), float(s))] is None:
      neighbours.append(v["depth_req"])
  if neighbours:
    print(f"  the {len(neighbours)} solvable geometries that sit immediately BELOW a failure "
          f"reach\n  a depth of {min(neighbours):.4f} to {max(neighbours):.4f} of R_eq")
    deepest_safe = max(d for d in depths)
    print(f"  and the deepest ANY solvable geometry reaches is {min(depths):.4f}")
    clean = max(neighbours) < 1.0 and min(neighbours) - min(depths) < 0.05
    print(f"\n  a compression threshold {'IS' if clean else 'is NOT'} a clean predictor here.")
    summary["boundary"] = {"edge_depths": [float(min(neighbours)), float(max(neighbours))],
                           "deepest_solvable": float(min(depths)),
                           "threshold_clean": bool(clean)}
    _ = deepest_safe

  # ---- what changes across the ridge? ----
  print(f"\n  the ridge at {RIDGE_R * 1e6:.0f} um: scanning stretch {RIDGE_S[0]:.2f} "
        f"to {RIDGE_S[-1]:.2f}\n")
  ridge_jobs = [(float(RIDGE_R), float(s)) for s in RIDGE_S]
  with records.pool(len(ridge_jobs)) as pool:
    ridge = dict(pool.map(trajectory, ridge_jobs))
    info = dict(pool.map(raw_information, ridge_jobs))

  # the sensitivity that made this a ridge, against the refined certified measure
  held = json.load(open(records.HERE / "grid_refinement.json"))
  support = held["rounds"][-1]["support"]
  cache = np.load(records.HERE / "grid_refinement_cache.npz", allow_pickle=True)
  built = {tuple(k): np.asarray(v, dtype=float)
           for k, v in zip(cache["keys"], cache["values"], strict=True) if v is not None}
  tau, spread = profile(SOURCE)
  average = np.zeros_like(next(iter(built.values())))
  for r, s, w in support:
    key = (round(r, 12), round(s, 9))
    if key in built: average = average + w * built[key]
  inverse = np.linalg.inv(average)

  print(f"  {'stretch':>8s} {'d(x,xi*)':>10s} {'R_min/R_eq':>11s} {'R_min/R_max':>12s} "
        f"{'minima':>7s} {'1st min':>8s} {'last min':>9s}")
  rows = {}
  for s in RIDGE_S:
    v = ridge[(float(RIDGE_R), float(s))]
    payload = info.get((float(RIDGE_R), float(s)))
    d = float("nan")
    if payload is not None:
      d = float(np.trace(inverse @ pieces(payload, tau, spread)[0]))
    if v is None:
      print(f"  {s:8.3f} {d:10.3f} {'did not integrate':>41s}")
      continue
    rows[f"{s:.3f}"] = dict(v, d=d)
    print(f"  {s:8.3f} {d:10.3f} {v['depth_req']:11.4f} {v['depth_rmax']:12.5f} "
          f"{v['minima']:7d} {v['first_min_phase']:8.3f} {v['last_min_phase']:9.3f}")
  summary["ridge"] = rows

  if rows:
    ds = np.array([r["d"] for r in rows.values()])
    counts = np.array([r["minima"] for r in rows.values()])
    lasts = np.array([r["last_min_phase"] for r in rows.values()])
    peak = int(np.nanargmax(ds))
    keys = list(rows)
    print(f"\n  d peaks at stretch {keys[peak]} with d = {ds[peak]:.3f}")
    print(f"  minima count over the scan: {sorted(set(counts.tolist()))}")
    changed = len(set(counts.tolist())) > 1
    print(f"  the number of resolved minima {'CHANGES' if changed else 'does not change'} "
          f"across the ridge")
    near = np.nanmax(lasts)
    print(f"  the last resolved minimum sits at phase {np.nanmin(lasts):.3f} to {near:.3f} "
          f"of a window that ends at {SPAN:.1f}")
    summary["ridge_verdict"] = {
      "peak_stretch": keys[peak], "peak_d": float(ds[peak]),
      "minima_changes": bool(changed),
      "last_min_max_phase": float(near), "window": SPAN}
    print("\n  A ridge that coincides with the minima count changing is the observation window")
    print("  clipping a bounce, which is an artifact of sampling. One that does not is a")
    print("  genuine feature of the information and belongs in the support.")

  json.dump(summary, open(records.HERE / "collapse_boundary.json", "w"), indent=1)


if __name__ == "__main__":
  main()
