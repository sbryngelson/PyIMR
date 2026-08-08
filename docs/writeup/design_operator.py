"""What experiment would tell the bubble-dynamics operators apart?

Three records were fitted and their power to distinguish operators ranged from about 170
nats to about half a nat. That is a design question, and this package has the apparatus for
it: a design measure over candidate experiments, maximised by the multiplicative algorithm
and certified by the Kiefer-Wolfowitz equivalence theorem.

The operator choice is made a PARAMETER so the existing machinery applies unchanged. Writing

    R(theta, eps) = (1 - eps) R_A(theta) + eps R_B(theta),

the sensitivity to `eps` at `eps = 0` is exactly `R_B - R_A`, so discrimination becomes one
more column of the Jacobian and D-optimality over the augmented parameter set balances
estimating the material against determining which operator produced the trace. The criterion
stays concave in the information matrix, so `optimal_measure` and its certificate are valid
as they stand -- no new theory, and the same proof of optimality.

The number worth having at the end is not the optimal design itself but the ratio: how much
of the attainable information about `eps` the experiments actually performed already carry.

ONE MATERIAL, AND WHY THAT LIMITS THE ANSWER. The measure is computed with the material held
at the 15 C fit, so it ranks GEOMETRIES at that material and nothing else. That is not what
separates the three records. Measured at each record's own fitted material and window, the
two operators differ by 4.17 noise units at 15 C, 3.06 at 23 C and 0.00005 -- five parts in
a hundred thousand -- at 33 C, because the 33 C fit barely collapses: it reaches
`R/Rmax = 0.146` against `0.068` at 15 C, and a wall that slow never makes compressibility
matter. So operator information is governed by collapse DEPTH, which the material sets, and
a measure computed at one material misranks experiments performed on others. The efficiency
column below inherits that and should be read as a statement about geometry alone.
"""

import json

import numpy as np

import records

FIT = (204.3, 0.04651, 1.964e-7, 5.301)          # the published qSLS fit
LOW = np.array([124.4, 0.04284, 1.482e-7, 2.424])
HIGH = np.array([431.9, 0.05005, 2.325e-7, 9.432])
RADII = np.geomspace(50e-6, 1200e-6, 16)
STRETCH = np.linspace(3.0, 20.0, 14)
RELATIVE_NOISE, SAMPLES = 0.018, 201
# the package default against what replicated, as (dynamics, liquid_eos) pairs
RIVALS = (("keller-miksis", None), ("keller-enthalpy", "mie-gruneisen"))
# the three experiments actually performed, from the shared record table
PERFORMED = {name: (records.DATASETS[name], records.load(name)[4]) for name in records.DATASETS}


def information(design):
  """`J^T J` over (four material parameters, operator), or `None` where it will not run."""
  import pyimr
  from pyimr.inference import InferenceParameter, RadiusObservation, prepare_inference
  from pyimr.noise import characteristic_time

  radius, stretch = design
  times = np.linspace(0.0, 5.0 * characteristic_time(radius), SAMPLES)
  material = pyimr.QuadraticZener(FIT[0], FIT[1], FIT[2], 0.0, FIT[3])

  traces = {}
  for operator in RIVALS:
    config = pyimr.SimulationConfig(radius, radius / stretch, material, dynamics=operator[0],
                                    liquid_eos=operator[1],
                                    rtol=1e-7, atol=1e-9, max_steps=200_000)
    try:
      traces[operator] = np.asarray(pyimr.simulate(times, config).radius_ratio, dtype=float)
    except Exception:                                # noqa: BLE001
      return design, None
  if not all(np.all(np.isfinite(t)) for t in traces.values()): return design, None

  base = pyimr.SimulationConfig(radius, radius / stretch, material, dynamics=RIVALS[0][0],
                                liquid_eos=RIVALS[0][1],
                                rtol=1e-7, atol=1e-9, max_steps=200_000)
  inference = prepare_inference(
    base, RadiusObservation(times, traces[RIVALS[0]] * radius, RELATIVE_NOISE * radius),
    tuple(InferenceParameter(f"material.{p}", lo, hi, "log") for p, lo, hi in zip(
      ("shear_modulus_pa", "viscosity_pa_s", "relaxation_time_s", "stiffening"), LOW, HIGH, strict=True)))
  unit = np.clip((np.log(FIT) - np.log(LOW)) / (np.log(HIGH) - np.log(LOW)), 0, 1)
  try:
    jacobian = np.asarray(inference.jacobian(unit), dtype=float)
  except Exception:                                  # noqa: BLE001
    return design, None
  if not np.all(np.isfinite(jacobian)): return design, None

  # the operator column: dR/d eps = R_B - R_A, whitened like the others
  gap = (traces[RIVALS[1]] - traces[RIVALS[0]]) * radius / (RELATIVE_NOISE * radius)
  augmented = np.column_stack([jacobian, gap])
  matrix = augmented.T @ augmented
  return design, 0.5 * (matrix + matrix.T)


def main():
  from pyimr.measure import optimal_measure

  designs = [(float(r), float(s)) for r in RADII for s in STRETCH] + list(PERFORMED.values())
  with records.pool(len(designs)) as pool:
    out = list(pool.map(information, designs))
  points = [d for d, m in out if m is not None]
  stack = np.array([m for _, m in out if m is not None])
  print(f"{len(points)} of {len(designs)} candidate designs integrate under both operators\n")

  result = optimal_measure(stack, iterations=400_000)
  print(f"D-optimal measure over (material, operator): gap {result.gap:.2e}, "
        f"certified={result.certified}, {len(result.support)} support points")
  for index in result.support:
    radius, stretch = points[index]
    print(f"    R_max {radius * 1e6:7.1f} um, stretch {stretch:5.2f}   weight {result.weights[index]:.3f}")

  # what matters: information about the operator alone, which is the last diagonal entry of
  # the inverse -- the variance of `eps` after the other parameters are fitted out
  def operator_variance(matrix):
    try: return float(np.linalg.inv(matrix)[-1, -1])
    except np.linalg.LinAlgError: return float("inf")

  best = operator_variance(np.tensordot(result.weights, stack, axes=(0, 0)))
  print(f"\n  variance of the operator coordinate under the optimal measure: {best:.4g}")
  print("\n  the experiments actually performed, each as a single-design measure.")
  print("  NOTE: computed at the 15 C material, so this ranks geometry, not the records --")
  print("  at their own fitted materials the operators separate by 4.17, 3.06 and 0.00 noise")
  print("  units, which is collapse depth rather than geometry and runs the other way.")
  for name, design in PERFORMED.items():
    index = points.index(design)
    here = operator_variance(stack[index])
    print(f"    {name:12s} R_max {design[0] * 1e6:5.1f} um, stretch {design[1]:.2f}   "
          f"variance {here:12.4g}   efficiency {best / here:6.1%}")
  json.dump({"support": [[points[i][0], points[i][1], float(result.weights[i])] for i in result.support],
             "gap": float(result.gap), "certified": bool(result.certified),
             "optimal_variance": best,
             "performed": {k: best / operator_variance(stack[points.index(v)]) for k, v in PERFORMED.items()}},
            open(records.HERE / "design_operator.json", "w"), indent=1)


if __name__ == "__main__":
  main()
