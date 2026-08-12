r"""Make the model error a coordinate, so a design can be scored against it.

\Cref{sec:gain} stops at a wall: ``is every model in the catalogue wrong?'' is not a coordinate,
no column of the Jacobian reaches model inadequacy, and so testability has to enter as a
CARDINALITY CONSTRAINT --- keep the batch on enough distinct settings to leave lack-of-fit
degrees of freedom --- rather than as a term anything is optimised for. \Cref{sec:survives} then
finds the certified geometries losing to the geometry already performed once the model is wrong,
and says plainly that no criterion in the design chapter takes model error as an argument.

IT CAN BE A COORDINATE, because $\hat\delta$ was measured. \Cref{sec:discrepancy} extracts it as
a vector, not a magnitude: the three records' $\hat\delta$ correlate at $0.49$ to $0.84$ where
unrelated curves of this length sit near $0.08$, and $65$ to $69$ percent of each lies in the
first collapse. That is a SHAPE, and \cref{sec:noisedesign} already establishes how to carry a
shape to an unrun geometry --- by phase $t/t_c$, tested rather than asserted, against transfer by
absolute time which is wrong for a statable reason.

So write $m(\theta) + \eta\,\hat\delta$ and treat $\eta$ exactly as \cref{sec:batch} treats the
model label: one more column of the Jacobian. Then

    Var(eta) = 1 / ( ||d||^2 - d' P d ),    P the projector onto span(J),

is the Schur complement, and its inverse is the part of the discrepancy a design can SEE rather
than absorb into the material.

THE TWO USES POINT THE SAME WAY, which is what makes this worth doing. To test the model you
want $\hat\delta$ visible; to keep the parameters clean you want it unabsorbed. Both are the
same angle, so one number serves the criterion and the robustness question that
\cref{sec:survives} leaves unanswered.

WHETHER IT IS TRUE IS THE POINT OF THE SECOND HALF. \Cref{sec:survives} scored four geometries
against two synthetic mismatches and reported what each cost the parameters. A single measured
$\hat\delta$ cannot know which mismatch, so it can only predict what the two have in common; if
it does not even do that, the transfer is refuted and the criterion below is not usable.
"""

import json

import numpy as np

import records
from identified import BOX, candidate_at_ratio
from noise_design import profile

RATIO = 38.5
AXES = ("mu", "galpha", "lambda1")
STARTS, EVALUATIONS = 16, 600
SAMPLES = 201
RADII = np.geomspace(50e-6, 1200e-6, 12)
STRETCH = np.linspace(3.0, 20.0, 9)
# the four sec:survives scored, as (label, R_max, stretch, gA err under thermal, under two-mode)
SURVIVES = (("performed", 277e-6, 7.09, 3.8, 5.0),
            ("E-optimal", 199e-6, 3.55, 5.9, 4.9),
            ("discrimination-optimal", 100e-6, 20.0, 37.1, 60.7),
            ("measure support", 60e-6, 18.0, 5.2, 43.5))


def measured_discrepancy(dataset):
  """`(phase, whitened residual)` for a record: `sec:discrepancy`'s `delta_hat` as a curve."""
  import pyimr
  from pyimr.noise import characteristic_time
  from pyimr.selection import fit_candidate, physical_from_unit
  from scipy.ndimage import median_filter

  times, mean, spread, maximum, stretch = records.load(dataset)
  candidate = candidate_at_ratio(RATIO)
  solve = records.solver(times, maximum, stretch)
  fit = fit_candidate(candidate, solve, mean, spread, bounds=BOX, starts=STARTS,
                      max_evaluations=EVALUATIONS)
  fitted = dict(zip(AXES, (float(v) for v in physical_from_unit(AXES, fit.unit, BOX)),
                    strict=True))
  config = pyimr.SimulationConfig(maximum, maximum / stretch, candidate.build(fitted),
                                  dynamics="keller-miksis", rtol=1e-9, atol=1e-11,
                                  max_steps=600_000)
  model = np.asarray(pyimr.simulate(times, config).radius_ratio, dtype=float)
  # whitened by the standard error of the MEAN, which is the scale sec:discrepancy uses
  sigma = median_filter(spread, 11, mode="nearest") / np.sqrt(records.trial_count(dataset))
  phase = (times - times[0]) / characteristic_time(maximum)
  return phase, (model - mean) / sigma


def one(job):
  """`(visible fraction, |delta|, Var(eta))` at a candidate geometry, plus the material block."""
  import pyimr
  from pyimr.noise import characteristic_time

  radius, stretch, source_phase, source_delta, dataset = job
  measured = json.load(open(records.HERE / "per_trial_fits.json"))[dataset]
  point = np.array([float(measured["median"][a]) for a in AXES])
  candidate = candidate_at_ratio(RATIO)
  times = np.linspace(0.0, 5.0 * characteristic_time(radius), SAMPLES)

  def trace(scale):
    moved = dict(zip(AXES, (float(v) for v in point * scale), strict=True))
    config = pyimr.SimulationConfig(radius, radius / stretch, candidate.build(moved),
                                    dynamics="keller-miksis", rtol=1e-8, atol=1e-10,
                                    max_steps=200_000)
    return np.asarray(pyimr.simulate(times, config).radius_ratio, dtype=float)

  step = 1e-4
  try:
    jacobian = np.column_stack([
      (trace(np.where(np.arange(len(AXES)) == k, 1 + step, 1.0))
       - trace(np.where(np.arange(len(AXES)) == k, 1 - step, 1.0))) / (2.0 * step)
      for k in range(len(AXES))])
  except Exception:                                                          # noqa: BLE001
    return (radius, stretch, dataset), None
  if not np.all(np.isfinite(jacobian)): return (radius, stretch, dataset), None

  tau, spread = profile(dataset)
  phase = (times - times[0]) / characteristic_time(radius)
  scale = np.interp(phase, tau, spread)
  whitened = jacobian / scale[:, None]
  # the discrepancy carried by PHASE, in noise units, exactly as sec:noisedesign carries sigma
  delta = np.interp(phase, source_phase, source_delta)

  gram = whitened.T @ whitened
  cross = whitened.T @ delta
  absorbed = float(cross @ np.linalg.solve(gram, cross))
  total = float(delta @ delta)
  visible = total - absorbed
  return (radius, stretch, dataset), {
    "visible": visible, "total": total, "share": visible / total if total > 0 else 0.0,
    "material_logdet": float(np.linalg.slogdet(gram)[1]),
    "variance": float("inf") if visible <= 0 else 1.0 / visible,
  }


def main():
  print("  fitting each record and extracting delta_hat ...", flush=True)
  curves = {d: measured_discrepancy(d) for d in records.DATASETS}

  print("\n  do the three delta_hat agree on a common phase grid? that is the transfer's premise\n")
  common = np.linspace(0.0, 5.0, 400)
  resampled = {d: np.interp(common, *curves[d]) for d in records.DATASETS}
  names = list(records.DATASETS)
  overlaps = {}
  for i, a in enumerate(names):
    for b in names[i + 1:]:
      u, v = resampled[a], resampled[b]
      cosine = float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))
      overlaps[f"{a}|{b}"] = cosine
      print(f"  {a:13s} vs {b:13s}  cos = {cosine:+.3f}")
  print(f"  unrelated curves of this length would sit near {1.0 / np.sqrt(len(common)):.3f}")

  source = "gelatin_15C"
  phase, delta = curves[source]
  designs = [(float(r), float(s), phase, delta, source) for r in RADII for s in STRETCH]
  designs += [(r, s, phase, delta, source) for _, r, s, _, _ in SURVIVES]
  print(f"\n  scoring {len(designs)} geometries ...", flush=True)
  with records.pool(len(designs)) as pool:
    got = dict(pool.map(one, designs))
  usable = {k: v for k, v in got.items() if v is not None}
  print(f"  {len(usable)} of {len(designs)} integrate")

  print("\n  the sec:survives geometries, scored by how much of delta_hat they can SEE\n")
  print(f"  {'geometry':24s} {'R (um)':>7s} {'stretch':>8s} {'visible share':>14s} "
        f"{'Var(eta)':>10s} {'g*alpha err: thermal':>21s} {'two-mode':>9s}")
  checks = {}
  for label, radius, stretch, thermal, twomode in SURVIVES:
    got_one = usable.get((radius, stretch, source))
    if got_one is None:
      print(f"  {label:24s} did not integrate")
      continue
    checks[label] = {"share": got_one["share"], "variance": got_one["variance"],
                     "thermal_err": thermal, "twomode_err": twomode}
    print(f"  {label:24s} {radius * 1e6:7.0f} {stretch:8.2f} {got_one['share']:14.3f} "
          f"{got_one['variance']:10.3e} {thermal:21.1f} {twomode:9.1f}")

  ranks = {}
  if len(checks) >= 3:
    order = list(checks)
    shares = np.array([checks[k]["share"] for k in order])
    for column in ("thermal_err", "twomode_err"):
      errs = np.array([checks[k][column] for k in order])
      ranks[column] = float(np.corrcoef(np.argsort(np.argsort(shares)),
                                        np.argsort(np.argsort(errs)))[0, 1])
      print(f"  rank correlation of visible share against {column}: {ranks[column]:+.3f}"
            "   (negative is the prediction)")

  print("\n  where each criterion sends the experiment\n")
  grid = [(k, v) for k, v in usable.items() if (k[0], k[1]) not in
          {(r, s) for _, r, s, _, _ in SURVIVES}]
  best_material = max(grid, key=lambda kv: kv[1]["material_logdet"])
  best_visible = max(grid, key=lambda kv: kv[1]["visible"])
  worst_visible = min(grid, key=lambda kv: kv[1]["visible"])
  for name, (key, value) in (("material log det", best_material),
                             ("discrepancy visibility", best_visible),
                             ("worst for the discrepancy", worst_visible)):
    print(f"  {name:26s} R_max {key[0] * 1e6:7.1f} um, stretch {key[1]:5.2f}, "
          f"visible share {value['share']:.3f}, material {value['material_logdet']:8.3f}")

  json.dump({"overlaps": overlaps, "survives": checks, "rank_correlations": ranks,
             "grid": {f"{k[0]:.6e}|{k[1]:.6e}": v for k, v in usable.items()}},
            open(records.HERE / "discrepancy_design.json", "w"), indent=1)


if __name__ == "__main__":
  main()
