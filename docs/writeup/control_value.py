r"""What is it worth to make the bubbles more alike?

\Cref{sec:setpoint} prices one consequence of the measured scatter: designing as though
$R_{\max}$ were settable, when per event it varies by a quarter, costs about a nat. That
immediately raises the question a collaborator can act on. \emph{Tightening the scatter is an
experimental investment --- a steadier laser, a better-controlled nucleation site --- so how many
nats does it buy, and is it worth more than choosing the geometry better?}

This prices it, in the same currency as everything else in the chapter.

WHY THE LOSS IS REAL EVEN THOUGH $R_{\max}$ IS MEASURED. The pipeline records each event's own
$R_{\max}$ and normalises by it, so nothing is misattributed. That removes a bias; it does not
remove the loss. Information from $n$ runs is $\sum_i M(x_i)$ with $x_i$ drawn, and $\log\det$ is
concave, so
%
    E[ log det sum_i M(x_i) ]  <=  log det ( n E[M] ) ,

which is the quantity \eqref{eq:setpoint} computes. Knowing which bubble you got tells you what
you learned; it does not give you back the bubble you wanted. The scatter costs information
whether or not it is measured.

BOTH KNOBS ARE PRICED, because they are separate investments. $R_{\max}$ scatter is set by the
laser and the nucleation; the stretch scatter is set by how repeatably the gas content comes out,
which is a different part of the apparatus. Sweeping each with the other held at its measured
value says which one to spend on.

THE GRID HAD TO BE INTERPOLATED, and the reason is worth recording. The candidate radii are
geometrically spaced at $23\%$, which is the same size as the scatter being resolved --- snapping
draws to the nearest grid point cannot see a coefficient of variation below the spacing, and
would report zero cost for any control tighter than the grid. The information is interpolated in
$(\log R_{\max}, \text{stretch})$ instead, so the sweep resolves what it claims to.
"""

import json

import numpy as np

import records
from noise_design import profile, raw_information

SOURCE = "gelatin_15C"
RADIUS_CV = 0.25                 # measured per event
STRETCH_CV = 0.06
LADDER = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50)
DRAWS = 400
SEED = 0


def main():
  from scipy.interpolate import RegularGridInterpolator

  from design_operator import RADII, STRETCH

  designs = [(float(r), float(s)) for r in RADII for s in STRETCH]
  print(f"  building information at {len(designs)} geometries ...", flush=True)
  with records.pool(len(designs)) as pool:
    got = list(pool.map(raw_information, designs))
  usable = {d: payload for d, payload in got if payload is not None}
  if len(usable) < len(designs):
    print(f"  {len(usable)} of {len(designs)} integrate; the grid must be complete")
    return

  tau, spread = profile(SOURCE)
  prior = 1.0 / np.sqrt(12.0)
  size = None
  cube = None
  for i, radius in enumerate(RADII):
    for j, stretch in enumerate(STRETCH):
      columns, phase = usable[(float(radius), float(stretch))]
      scale = np.interp(phase, tau, spread)
      whitened = columns / scale[:, None]
      fisher = 0.5 * (whitened.T @ whitened + (whitened.T @ whitened).T) * prior**2
      if cube is None:
        size = fisher.shape[0]
        cube = np.zeros((len(RADII), len(STRETCH), size, size))
      cube[i, j] = fisher

  # interpolated, not snapped: the radii are spaced at 23%, the same size as the scatter, so a
  # nearest-grid-point draw would report zero cost for any control tighter than the grid
  logr = np.log(np.asarray(RADII, dtype=float))
  interpolate = RegularGridInterpolator(
    (logr, np.asarray(STRETCH, dtype=float)), cube, bounds_error=False, fill_value=None)
  rng = np.random.default_rng(SEED)
  identity = np.eye(size)

  def delivered(setpoint, radius_cv, stretch_cv):
    """`0.5 log det (I + E[M])` at a setpoint, over log-normal draws in both coordinates."""
    target_r, target_s = setpoint
    if radius_cv <= 0.0 and stretch_cv <= 0.0:
      averaged = interpolate(np.array([[np.log(target_r), target_s]]))[0]
    else:
      drawn_r = np.log(target_r) + rng.normal(0.0, np.log1p(radius_cv), DRAWS)
      drawn_s = target_s * np.exp(rng.normal(0.0, np.log1p(stretch_cv), DRAWS))
      drawn_r = np.clip(drawn_r, logr.min(), logr.max())
      drawn_s = np.clip(drawn_s, min(STRETCH), max(STRETCH))
      averaged = interpolate(np.column_stack([drawn_r, drawn_s])).mean(axis=0)
    return 0.5 * float(np.linalg.slogdet(identity + averaged)[1])

  # a fine setpoint grid, since the optimum moves with the scatter
  fine_r = np.exp(np.linspace(logr.min(), logr.max(), 40))
  fine_s = np.linspace(min(STRETCH), max(STRETCH), 24)
  candidates = [(float(r), float(s)) for r in fine_r for s in fine_s]

  def best(radius_cv, stretch_cv):
    values = [delivered(c, radius_cv, stretch_cv) for c in candidates]
    top = int(np.argmax(values))
    return candidates[top], float(values[top])

  summary = {"measured": {"radius_cv": RADIUS_CV, "stretch_cv": STRETCH_CV}}

  print(f"\n  tightening the R_max scatter, stretch held at its measured {STRETCH_CV}\n")
  print(f"  {'R_max cv':>9s} {'best setpoint':>22s} {'nats':>8s} {'gained vs measured':>19s}")
  base = None
  radius_ladder = {}
  for cv in LADDER:
    point, value = best(cv, STRETCH_CV)
    if abs(cv - RADIUS_CV) < 1e-9: base = value
    radius_ladder[f"{cv:g}"] = {"setpoint": list(point), "nats": value}
    print(f"  {cv:9.2f} {point[0] * 1e6:8.0f} um at {point[1]:5.2f} {value:8.3f}", end="")
    print("" if base is None else f" {value - base:19.3f}")
  for cv, row in radius_ladder.items():
    row["gain_vs_measured"] = row["nats"] - base
  summary["radius_ladder"] = radius_ladder

  print(f"\n  tightening the stretch scatter, R_max held at its measured {RADIUS_CV}\n")
  print(f"  {'stretch cv':>10s} {'best setpoint':>22s} {'nats':>8s} {'gained vs measured':>19s}")
  stretch_base = None
  stretch_ladder = {}
  for cv in LADDER:
    point, value = best(RADIUS_CV, cv)
    if abs(cv - STRETCH_CV) < 1e-9: stretch_base = value
    stretch_ladder[f"{cv:g}"] = {"setpoint": list(point), "nats": value}
  if stretch_base is None:
    point, stretch_base = best(RADIUS_CV, STRETCH_CV)
  for cv, row in stretch_ladder.items():
    row["gain_vs_measured"] = row["nats"] - stretch_base
    print(f"  {float(cv):10.2f} {row['setpoint'][0] * 1e6:8.0f} um at {row['setpoint'][1]:5.2f} "
          f"{row['nats']:8.3f} {row['gain_vs_measured']:19.3f}")
  summary["stretch_ladder"] = stretch_ladder

  perfect_r = radius_ladder["0"]["nats"] - base
  perfect_s = stretch_ladder["0"]["nats"] - stretch_base
  print(f"\n  perfect control of R_max is worth {perfect_r:+.3f} nats; of the stretch, "
        f"{perfect_s:+.3f}.")
  print(f"  halving the R_max scatter (0.25 -> 0.125) is worth about "
        f"{np.interp(0.125, LADDER, [radius_ladder[f'{c:g}']['nats'] for c in LADDER]) - base:+.3f}.")
  summary["perfect_radius"] = float(perfect_r)
  summary["perfect_stretch"] = float(perfect_s)

  # The single-setpoint sweep above answers "what if I run one setting", and there scatter
  # HELPS -- it accidentally spreads a design that would otherwise be a point. That is not the
  # question a chapter built on measures should ask. The set of E[M] reachable under scatter is
  # a subset of the convex hull of {M(x)}, so an optimal MEASURE can emulate any scatter and
  # beat it; scatter can only blur a support the measure chose deliberately.
  from pyimr.measure import optimal_measure

  print("\n  the same ladder, but optimising a MEASURE rather than one setpoint\n")
  print(f"  {'R_max cv':>9s} {'settings':>9s} {'gap':>9s} {'nats':>8s} {'gained vs measured':>19s}")
  coarse = [(float(r), float(s)) for r in fine_r[::3] for s in fine_s[::3]]
  measure_ladder = {}
  measured_value = None
  for cv in LADDER:
    built = []
    for point in coarse:
      target_r, target_s = point
      if cv <= 0.0 and STRETCH_CV <= 0.0:
        built.append(interpolate(np.array([[np.log(target_r), target_s]]))[0])
        continue
      drawn_r = np.clip(np.log(target_r) + rng.normal(0.0, np.log1p(cv), DRAWS),
                        logr.min(), logr.max())
      drawn_s = np.clip(target_s * np.exp(rng.normal(0.0, np.log1p(STRETCH_CV), DRAWS)),
                        min(STRETCH), max(STRETCH))
      built.append(interpolate(np.column_stack([drawn_r, drawn_s])).mean(axis=0))
    stack_cv = np.array(built)
    held = optimal_measure(stack_cv, iterations=100_000)
    averaged = np.tensordot(held.weights, stack_cv, axes=(0, 0))
    value = 0.5 * float(np.linalg.slogdet(identity + averaged)[1])
    if abs(cv - RADIUS_CV) < 1e-9: measured_value = value
    measure_ladder[f"{cv:g}"] = {"settings": int(held.support.size), "gap": held.gap,
                                 "nats": value}
  for cv, row in measure_ladder.items():
    row["gain_vs_measured"] = row["nats"] - measured_value
    print(f"  {float(cv):9.2f} {row['settings']:9d} {row['gap']:9.1e} {row['nats']:8.3f} "
          f"{row['gain_vs_measured']:19.3f}")
  summary["measure_ladder"] = measure_ladder
  perfect_measure = measure_ladder["0"]["nats"] - measured_value
  summary["perfect_radius_measure"] = float(perfect_measure)
  print(f"\n  Under a measure, perfect R_max control is worth {perfect_measure:+.3f} nats --")
  print("  the opposite sign to the single-setpoint answer, and the one that applies here.")
  print("  Scatter helps a design that was going to be ONE setting and hurts a design that")
  print("  chooses its own spread, because a measure can emulate any scatter and beat it.")

  print("\n  against the chapter's other levers, in the same units\n")
  for label, value in (("bubbles vs frames (sec:allocation)", 3.93),
                       ("perfect R_max control, under a measure", perfect_measure),
                       ("perfect R_max control, one setpoint", perfect_r),
                       ("geometry under the wrong noise model", 1.34),
                       ("ignoring setpoint scatter (sec:setpoint)", 1.118),
                       ("perfect stretch control", perfect_s),
                       ("restoring testability", 0.24)):
    print(f"  {label:42s} {value:+7.3f}")

  json.dump(summary, open(records.HERE / "control_value.json", "w"), indent=1)


if __name__ == "__main__":
  main()
