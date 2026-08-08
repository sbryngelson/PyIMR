"""How many frames does the record actually need?

`selection.tex` reports 201 samples carrying the information of about ten independent ones.
That is a statement about redundancy, and redundancy is a design variable: a high-speed camera
trades frame rate against exposure, against how long a window it can hold, and against how
many bubbles fit in a session. Nobody has asked what the frames are worth.

Two questions, both answered from the Fisher information of the fitted model and neither
needing a new experiment.

FIRST, how much information do frames actually add? Take the whitened sensitivity `J` on the
full grid and on subsets of it, and compare `log det J^T J`. If dropping half the frames costs
almost nothing, the camera is oversampled for the purpose of estimating these parameters.

SECOND, does it matter WHICH frames? A uniform subsample is what a slower camera gives. An
optimal subsample is what the design literature would choose -- and here it is computable
exactly, because with the parameters fixed the choice of `n` rows from `J` maximizing
`log det` is a D-optimal design on a finite candidate set, which `pyimr.measure` already
solves and certifies. The gap between uniform and optimal is what a smarter trigger buys.

The caveat that matters: `J^T J` presumes independent samples. With the residual correlated
at 0.918 the true information from 201 frames is far below what `log det J^T J` reports, so
every number here OVERSTATES what dense sampling is worth. That direction is the useful one --
it makes the conclusion that frames are cheap a conservative one.
"""

import json

import numpy as np

import records

DATASET = "gelatin_15C"
FIT = {"g": 204.3, "mu": 0.04651, "lambda1": 1.964e-7, "alpha": 5.301}
PATHS = ("material.shear_modulus_pa", "material.viscosity_pa_s",
         "material.relaxation_time_s", "material.stiffening")
KEEP = (201, 100, 50, 25, 12, 8, 6, 5, 4)


def _whitened(times, mean, spread, maximum, stretch):
  import pyimr

  material = pyimr.QuadraticZener(FIT["g"], FIT["mu"], FIT["lambda1"], 0.0, FIT["alpha"])
  config = pyimr.SimulationConfig(maximum, maximum / stretch, material,
                                  dynamics="keller-miksis", rtol=1e-10, atol=1e-12,
                                  max_steps=800_000)
  jacobian = np.asarray(pyimr.prepare(config).solve_with_sensitivities(times, PATHS).radius_ratio,
                        dtype=float)
  # scaled to fractional changes, so the columns are comparable across ten orders of magnitude,
  # and whitened by the measurement deviation
  scales = np.array([FIT["g"], FIT["mu"], FIT["lambda1"], FIT["alpha"]])
  return jacobian * scales / spread[:, None]


def _logdet(rows):
  matrix = rows.T @ rows
  sign, value = np.linalg.slogdet(matrix)
  return float(value) if sign > 0 else float("-inf")


def main():
  from pyimr.measure import apportion, optimal_measure

  times, mean, spread, maximum, stretch = records.load(DATASET)
  jacobian = _whitened(times, mean, spread, maximum, stretch)
  samples, parameters = jacobian.shape
  full = _logdet(jacobian)

  print(f"{DATASET}: what the frames are worth, at the fitted qSLS\n")
  print(f"  {samples} samples, {parameters} parameters, full log det J^T J = {full:.3f}")
  print("  N_eff from the residual is about 10, so the information below is overstated\n")
  print("  Both columns spend the SAME number of observations. `uniform` is what a slower")
  print("  camera gives; `placed` is the same count put where the information is, repeats")
  print("  allowed -- so `placed` at 201 is not a subset of the grid and may beat it.\n")
  print(f"{'observations':>13} {'uniform':>11} {'placed':>11} {'D-gain':>9} {'vs full grid':>13}")

  # one D-optimal measure over all 201 candidate frames; each frame's information is the
  # rank-one outer product of its own sensitivity row
  stack = np.einsum("ij,ik->ijk", jacobian, jacobian)
  measure = optimal_measure(stack, tolerance=1e-10, iterations=400_000)
  summary = {"full_logdet": full, "samples": samples,
             "measure_gap": measure.gap, "measure_certified": bool(measure.certified),
             "measure_support": int(measure.support.size), "rows": {}}

  for keep in KEEP:
    if keep < parameters: continue
    uniform = jacobian[np.unique(np.linspace(0, samples - 1, keep).round().astype(int))]
    uniform_value = _logdet(uniform)
    # the optimal `keep` frames: apportion the certified measure to that many runs, which is
    # exactly the "which frames, and how many repeats of each" question
    try:
      exact = apportion(measure.weights, keep, stack)
      counts = exact.counts
      chosen = np.repeat(np.arange(samples), counts)
      optimal_value = _logdet(jacobian[chosen])
    except ValueError:
      optimal_value, exact = float("-inf"), None
    # D-efficiency per parameter against the full grid
    uniform_eff = float(np.exp((uniform_value - full) / parameters))
    optimal_eff = float(np.exp((optimal_value - full) / parameters))
    gain = optimal_eff / uniform_eff if uniform_eff > 0 else float("nan")
    print(f"{keep:13d} {uniform_value:11.3f} {optimal_value:11.3f} "
          f"{gain:8.2f}x {optimal_eff:12.3f}x")
    summary["rows"][str(keep)] = dict(uniform=uniform_value, uniform_efficiency=uniform_eff,
                                      optimal=optimal_value, optimal_efficiency=optimal_eff,
                                      gain=gain)

  print(f"\n  The D-optimal measure over frames is certified at a gap of {measure.gap:.1e} "
        f"on {measure.support.size} support points")
  print(f"  out of {samples}. A design that puts every run on {measure.support.size} distinct "
        "times is what the")
  print("  information alone wants; it is not what anyone should run, because it leaves no")
  print("  degrees of freedom to notice the model is wrong -- which is the whole finding of")
  print("  the lack-of-fit and residual-correlation work. Read the `gain` column as the size")
  print("  of the prize a smarter trigger could win, not as a recommendation.")
  records.HERE.joinpath("sampling.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
  main()
