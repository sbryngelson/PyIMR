"""A design that discriminates all three model axes, averaged over the material.

Each axis becomes a parameter: with `R(theta, eps) = (1 - eps) R_A + eps R_B`, the
sensitivity to `eps` is `R_B - R_A`, so three axes are three more Jacobian columns and
D-optimality over the augmented set balances estimating the material against determining
which dynamics operator, which constitutive law and which thermal treatment produced the
trace. Concave in `M`, so `optimal_measure` and its certificate apply unchanged.

TWO THINGS THE EVIDENCE FORCED, both absent from the single-axis version.

Averaged over materials, not computed at one. Operator information is governed by collapse
depth, which the material sets: at their own fits the operators separate by 4.17, 3.06 and
0.00005 noise units on the three records. A measure certified at one material ranks
geometries for that material and misranks the others, which is exactly what the single-axis
version did -- its efficiency ordering came out backwards from the empirical one. Averaging
the information over a spread of `g*alpha`, the identified stiffness, makes the design
answer what it is asked.

Per-axis variance reported, not only the determinant. The axes are far from equally
visible: after refitting the material absorbs what it can, thermal leaves 8.46 noise units,
constitutive 8.14, and dynamics 1.98. A design maximising the augmented determinant can buy
its whole score on the easy axes and leave the operator undetermined, and the determinant
would not say so. The binding constraint is dynamics, so its variance is what the design
should be read against.
"""

import json

import numpy as np

import records

DATASET = "gelatin_15C"
RATIO = 38.5                                       # g/alpha, the unidentified direction, held fixed
STIFFNESS = (3.6e2, 1.08e3, 3.2e3)                 # g*alpha, the identified one, spanning a decade
RADII = np.geomspace(60e-6, 1200e-6, 9)
STRETCH = np.linspace(4.0, 18.0, 7)
RELATIVE_NOISE, SAMPLES = 0.018, 121
AXES = ("dynamics", "constitutive", "thermal")


def _material(product, **over):
  import pyimr

  shear, stiffening = float(np.sqrt(product * RATIO)), float(np.sqrt(product / RATIO))
  if over:
    return pyimr.TwoModeQuadraticZener(shear, 0.04651, 1.964e-7, 0.0, stiffening,
                                       over["tau"], over["share"])
  return pyimr.QuadraticZener(shear, 0.04651, 1.964e-7, 0.0, stiffening)


def information(job):
  """`J^T J` over (material, three model axes) at one design and one stiffness."""
  import pyimr
  from pyimr.inference import InferenceParameter, RadiusObservation, prepare_inference
  from pyimr.noise import characteristic_time

  (radius, stretch), product = job
  times = np.linspace(0.0, 5.0 * characteristic_time(radius), SAMPLES)
  common = dict(rtol=1e-7, max_steps=300_000)
  base_material = _material(product)

  def run(material, **options):
    solve = records.solver(times, radius, stretch, **common, **options)
    return solve(material)[0]

  try:
    base = run(base_material)
    perturbed = {
      "dynamics": run(base_material, dynamics="keller-enthalpy", liquid_eos="mie-gruneisen"),
      "constitutive": run(_material(product, tau=2e-6, share=0.2)),
      "thermal": run(base_material, bubtherm=1, medtherm=1, Nt=11, Mt=11),
    }
  except Exception:                                  # noqa: BLE001
    return job, None
  traces = [base, *perturbed.values()]
  if not all(np.all(np.isfinite(t)) for t in traces): return job, None

  config = pyimr.SimulationConfig(radius, radius / stretch, base_material, dynamics="keller-miksis", **common)
  low, high = np.array([1e2, 1e-3, 1e-8, 1e-1]), np.array([1e4, 1e0, 1e-5, 1e2])
  centre = np.array([np.sqrt(product * RATIO), 0.04651, 1.964e-7, np.sqrt(product / RATIO)])
  inference = prepare_inference(
    config, RadiusObservation(times, base * radius, RELATIVE_NOISE * radius),
    tuple(InferenceParameter(f"material.{p}", a, b, "log") for p, a, b in zip(
      ("shear_modulus_pa", "viscosity_pa_s", "relaxation_time_s", "stiffening"), low, high, strict=True)))
  try:
    jacobian = np.asarray(inference.jacobian(np.clip((np.log(centre) - np.log(low))
                                                     / (np.log(high) - np.log(low)), 0, 1)), dtype=float)
  except Exception:                                  # noqa: BLE001
    return job, None
  if not np.all(np.isfinite(jacobian)): return job, None

  columns = [jacobian] + [((perturbed[a] - base) / RELATIVE_NOISE)[:, None] for a in AXES]
  augmented = np.column_stack(columns)
  matrix = augmented.T @ augmented
  return job, 0.5 * (matrix + matrix.T)


def main():
  from pyimr.measure import optimal_measure

  times, mean, spread, maximum, stretch = records.load(DATASET)
  designs = [(float(r), float(s)) for r in RADII for s in STRETCH] + [(maximum, stretch)]
  jobs = [(d, p) for d in designs for p in STIFFNESS]
  with records.pool(len(jobs)) as pool:
    out = list(pool.map(information, jobs))

  # average over stiffness: a design is kept only where every material integrates, so the
  # averaged matrix is over the same set everywhere and the comparison stays honest
  gathered = {}
  for (design, _), matrix in out:
    gathered.setdefault(design, []).append(matrix)
  usable = [d for d, ms in gathered.items() if len(ms) == len(STIFFNESS) and all(m is not None for m in ms)]
  stack = np.array([np.mean(gathered[d], axis=0) for d in usable])
  print(f"{len(usable)} of {len(designs)} designs integrate at every stiffness "
        f"({len(STIFFNESS)} values of g*alpha spanning a decade)\n")

  result = optimal_measure(stack, iterations=400_000)
  print(f"D-optimal measure over (4 material + 3 model axes): gap {result.gap:.2e}, "
        f"certified={result.certified}, {len(result.support)} support points")
  for index in result.support:
    radius, stretch_here = usable[index]
    print(f"    R_max {radius * 1e6:7.1f} um, stretch {stretch_here:5.2f}   weight {result.weights[index]:.3f}")

  def variances(matrix):
    try: inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError: return {a: float("inf") for a in AXES}
    return {a: float(inverse[-len(AXES) + i, -len(AXES) + i]) for i, a in enumerate(AXES)}

  optimal = variances(np.tensordot(result.weights, stack, axes=(0, 0)))
  print("\n  variance of each model axis, per run. The determinant alone would hide this:")
  for axis in AXES:
    print(f"    {axis:>14}  {optimal[axis]:12.4g}")
  worst = max(AXES, key=lambda a: optimal[a])
  print(f"  binding axis under this design: {worst}")

  here = variances(stack[usable.index((maximum, stretch))])
  print(f"\n  the {DATASET} geometry, at the same averaging:")
  for axis in AXES:
    print(f"    {axis:>14}  {here[axis]:12.4g}   efficiency {optimal[axis] / here[axis]:6.1%}")
  records.HERE.joinpath("design_three.json").write_text(json.dumps(
    {"support": [[usable[i][0], usable[i][1], float(result.weights[i])] for i in result.support],
     "certified": bool(result.certified), "gap": float(result.gap),
     "optimal": optimal, "performed_15C": here}, indent=1))


if __name__ == "__main__":
  main()
