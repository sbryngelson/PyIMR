"""A batch of experiments designed to settle the model question that is still open.

This replaces a sampled expected log Bayes factor with the derivative route, for reasons the
sampled version demonstrated itself. Scoring 48 geometries by a double-loop evidence estimator
hit every documented failure of that estimator: the first run put ten of twelve batch runs on
the design whose evidence integral rested on 1.15 effective draws, and once the reliability
gate moved into the candidate set, 22 of 48 designs dropped out entirely. It also forced a
prior on the rival, which if centred on the incumbent's fit measures prior placement rather
than discriminability -- the error `selection.tex` names.

THE DERIVATIVE ROUTE. Write the model choice as a coordinate, `R(theta, eps) = (1-eps) R_A +
eps R_B`, so `dR/deps = R_B - R_A` is one more Jacobian column. Material sensitivities and the
model question then live in ONE information matrix, linear in the design measure, so
`optimal_measure` certifies the answer exactly as it does for the material alone. No evidence
integral, no prior on the rival, no effective sample size, and every candidate scorable.

WHY D-OPTIMALITY ON THE AUGMENTED MATRIX RATHER THAN A BLEND. `identification_front` needs a
utility LINEAR in the measure, because that is what keeps the compound criterion concave. The
post-material variance of `eps` is not linear -- it is a nonlinear function of the averaged
matrix -- so blending it would forfeit the certificate. Putting `eps` into the matrix instead
keeps one concave criterion that balances estimating the material against determining the
model, which is what `design_three.py` established and what the equivalence theorem covers.

WHICH RIVALS. `screen.py` first. Designing to separate models the records have already
separated spends runs on a closed question, and a local criterion is only right for rivals
that are close -- which is what survives screening. The conservative reading is used: margins
are deflated by `N/N_eff` before screening, because every evidence assumes independent
residuals and the correlation makes them roughly twentyfold too confident. Designing for one
rival too many costs efficiency; designing for one too few answers the wrong question.
"""

import json

import numpy as np

import records

RADII = np.geomspace(50e-6, 1200e-6, 8)
STRETCHES = np.linspace(3.0, 20.0, 6)
DESIGNS = [(float(r), float(s)) for r in RADII for s in STRETCHES]
SAMPLES = 121
RELATIVE_NOISE = 0.018
RUNS = 12
INFLATION = 201.0 / 10.0
FIT = {"g": 204.3, "mu": 0.04651, "lambda1": 1.964e-7, "alpha": 5.301}
PATHS = ("material.shear_modulus_pa", "material.viscosity_pa_s",
         "material.relaxation_time_s", "material.stiffening")


def _live_operators():
  """The operators the records have not settled, with their posterior weights."""
  from pyimr.discriminate import screen_models

  stored = json.loads((records.HERE / "req_prior.json").read_text())
  per_record = {}
  for key, row in stored.items():
    record, operator, width = key.split("|")
    if width != "None" or "failed" in row: continue
    per_record.setdefault(record, {})[operator] = row["log_evidence"]

  # Averaged over records, not taken from one. Selecting rivals by "live anywhere" while
  # weighting them from a single record is inconsistent, and it fails loudly: rayleigh-plesset
  # is live at 23 and 33 C but decided at 15 C, so it entered the rival list at weight zero,
  # contributed no information, and left the augmented matrix singular. The design should
  # serve the posterior the records jointly support.
  names = sorted({n for table in per_record.values() for n in table})
  stacked = []
  for table in per_record.values():
    if not all(n in table for n in names): continue
    stacked.append(screen_models(np.array([table[n] for n in names]) / INFLATION).weights)
  averaged = np.mean(np.vstack(stacked), axis=0)
  averaged = averaged / averaged.sum()

  reference = names[int(np.argmax(averaged))]
  keep = [(n, w) for n, w in zip(names, averaged, strict=True) if n != reference and w > 1e-6]
  rivals = [n for n, _ in keep]
  weights = np.array([w for _, w in keep])
  return reference, rivals, weights / weights.sum()


def _material():
  import pyimr

  return pyimr.QuadraticZener(FIT["g"], FIT["mu"], FIT["lambda1"], 0.0, FIT["alpha"])


def _job(argument):
  design, reference, rivals = argument
  import pyimr
  from pyimr.noise import characteristic_time

  radius, stretch = design
  times = np.linspace(0.0, 5.0 * characteristic_time(radius), SAMPLES)

  def configured(operator):
    dynamics, _, eos = operator.partition("/")
    return pyimr.SimulationConfig(radius, radius / stretch, _material(), dynamics=dynamics,
                                  liquid_eos=eos or None, rtol=1e-9, atol=1e-11,
                                  max_steps=600_000)

  def trace(operator):
    return np.asarray(pyimr.simulate(times, configured(operator)).radius_ratio, dtype=float)

  try:
    jacobian = np.asarray(pyimr.prepare(configured(reference)).solve_with_sensitivities(
      times, PATHS).radius_ratio, dtype=float)
    base = trace(reference)
    columns = np.vstack([trace(rival) - base for rival in rivals])
  except Exception as error:                          # noqa: BLE001
    return design, {"failed": f"{type(error).__name__}: {error}"}
  if not (np.all(np.isfinite(jacobian)) and np.all(np.isfinite(columns))):
    return design, {"failed": "not finite"}

  scales = np.array([FIT[k] for k in ("g", "mu", "lambda1", "alpha")])
  return design, {"jacobian": (jacobian * scales / RELATIVE_NOISE).tolist(),
                  "columns": (columns / RELATIVE_NOISE).tolist()}


def main():
  from pyimr.measure import apportion, augmented_information, optimal_measure, separability

  reference, rivals, weights = _live_operators()
  print(f"reference operator: {reference}")
  print(f"live rivals ({len(rivals)}), weighted by posterior after deflating by N/N_eff = "
        f"{INFLATION:.0f}:")
  for rival, weight in zip(rivals, weights, strict=True):
    print(f"    {rival:34} {weight:.3f}")

  jobs = [(d, reference, rivals) for d in DESIGNS]
  with records.pool(len(jobs)) as pool:
    table = dict(pool.map(_job, jobs))

  usable = [d for d in DESIGNS if "failed" not in table[d]]
  print(f"\n{len(usable)} of {len(DESIGNS)} designs scored. Every design that integrates is "
        "scorable here,\n  where the sampled criterion lost 22 of 48 to its own estimator.")
  if len(usable) < 8:
    print("  too few designs to build a measure"); return

  matrices = np.array([augmented_information(np.array(table[d]["jacobian"]),
                                             np.array(table[d]["columns"]), weights=weights)
                       for d in usable])
  measure = optimal_measure(matrices, iterations=400_000)
  batch = apportion(measure.weights, RUNS, matrices)
  print(f"\nmeasure certified at a gap of {measure.gap:.1e} on {measure.support.size} support "
        f"points\n  ({'certified' if measure.certified else 'NOT CERTIFIED'}); the {RUNS}-run "
        f"batch keeps {batch.efficiency:.1%} of it\n")
  print(f"{'R_max/um':>9} {'stretch':>8} {'runs':>5}")
  for index, count in batch.table:
    print(f"{usable[index][0]*1e6:9.0f} {usable[index][1]:8.1f} {count:5d}")

  def variance_at(counts):
    stacked = np.vstack([np.array(table[usable[i]]["jacobian"])
                         for i, n in enumerate(counts) for _ in range(int(n))])
    columns = np.hstack([np.array(table[usable[i]]["columns"])
                         for i, n in enumerate(counts) for _ in range(int(n))])
    return separability(stacked, columns, weights=weights)[0]

  performed = min(range(len(usable)),
                  key=lambda i: abs(np.log(usable[i][0] / 277e-6)) + abs(np.log(usable[i][1] / 7.09)))
  same = np.zeros(len(usable), dtype=int)
  same[performed] = RUNS
  designed, flat = variance_at(batch.counts), variance_at(same)
  print(f"\n  post-material variance of each model coordinate, {RUNS} runs, against spending")
  print(f"  all {RUNS} at the performed geometry "
        f"({usable[performed][0]*1e6:.0f} um, stretch {usable[performed][1]:.1f}):\n")
  print(f"{'rival':>34} {'designed':>11} {'all performed':>14} {'gain':>8}")
  for rival, a, b in zip(rivals, designed, flat, strict=True):
    print(f"{rival:>34} {a:11.3e} {b:14.3e} {b / max(a, 1e-300):7.2f}x")
  print("\n  Smaller is better: it is the variance of the coordinate saying which operator")
  print("  produced the trace, after the material has absorbed everything it can. Infinite")
  print("  would mean no design separates that pair at all.")
  records.HERE.joinpath("identify.json").write_text(json.dumps({
    "reference": reference, "rivals": rivals, "weights": weights.tolist(),
    "gap": measure.gap, "certified": bool(measure.certified), "efficiency": batch.efficiency,
    "batch": [[usable[i][0], usable[i][1], int(n)] for i, n in batch.table],
    "variance_designed": designed.tolist(), "variance_performed": flat.tolist()}, indent=1))


if __name__ == "__main__":
  main()
