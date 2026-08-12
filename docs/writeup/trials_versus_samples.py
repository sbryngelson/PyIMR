r"""How should a budget be split between bubbles and frames?

#222 names this and the design chapter never asks it: every batch here is chosen over
GEOMETRIES, with the number of trials and the number of samples per trial fixed at whatever the
records happened to use. Under i.i.d. noise the question is uninteresting -- information is
proportional to the total sample count, so the split does not matter. Under trial-to-trial
parameter variation it does, and the algebra says which way.

Write $\theta_j \sim \mathcal{N}(\theta_{\rm pop}, \Sigma_\theta)$ for the $j$th bubble and
$\Sigma_\varepsilon(N)$ for the measurement noise of an $N$-sample trace. Averaging $J$ trials,
the mean trace has covariance $[J_m \Sigma_\theta J_m^{\mathsf T} + \Sigma_\varepsilon]/J$, so
the information about the POPULATION parameter is, by Woodbury,
%
    F(J, N) = J \left( \Sigma_\theta + F_N^{-1} \right)^{-1},
    \qquad F_N = J_m^{\mathsf T}\Sigma_\varepsilon(N)^{-1} J_m ,

with $F_N$ the Fisher matrix a single $N$-sample trace supplies. The two limits are not
symmetric. Trials enter linearly and never saturate. Samples enter only through $F_N$, and as
$N \to \infty$ the bracket tends to $\Sigma_\theta$ -- so $F \to J\Sigma_\theta^{-1}$ and further
frames buy nothing. Beyond a point, more samples measure one bubble ever more precisely rather
than the population.

At a FIXED total of frames $S = JN$ this has a sharp consequence: $F = (S/N)(\Sigma_\theta +
F_N^{-1})^{-1}$, and once $F_N^{-1}$ is small against $\Sigma_\theta$ the prefactor $S/N$ is all
that is left, which FALLS with $N$. Fewer frames on more bubbles, until $N$ drops far enough
that $F_N^{-1}$ matters again. The optimum is where those two effects meet, and it depends on
$\Sigma_\theta$ -- which `per_trial_fits.py` measures rather than assumes.

The records sit at $N = 201$ and $J$ of $18$, $14$ and $7$. Whether that is a good split is a
question nobody has asked of them, and the answer is that it is not close.

READ THE TWO COLUMNS DIFFERENTLY. Doubling the trials is worth exactly $p\log 2 = 2.079$ nats
by construction -- $F$ is linear in $J$, so $\log\det$ gains $p\log 2$ whatever the matrices
are. That number is arithmetic and measures nothing. The measurement is the other column:
doubling the SAMPLES buys $0.035$ to $0.188$ nats, which is where the saturation shows.

TWO CAVEATS, BOTH POINTING THE SAME WAY. $F_N$ is scaled linearly in $N$, as though samples
within a trace were independent. They are not -- the residuals correlate at lag one above
$0.9$ and \cref{sec:limitations} measures $N_{\rm eff}$ near $10$ of $201$ -- so the true
$F_N$ is far flatter in $N$ than assumed here, and cutting $N$ costs even less than this says.
The optimum below is therefore conservative in its direction, though it should not be pushed
past $N \approx N_{\rm eff}$, where the collapse stops being resolved at all.

And $N$ is not free in practice. It is set by the frame rate and the record length, so a
smaller $N$ means a slower camera or a shorter window, not a dial. The calculation says where
the information is, not what the apparatus can do.
"""

import json

import numpy as np

import records
from identified import BOX, candidate_at_ratio

RATIO = 38.5
AXES = ("mu", "galpha", "lambda1")
STARTS, EVALUATIONS = 16, 600
SMOOTH = 11


def per_sample_information(dataset):
  """`(F_N at the record's own N, N)` in the identified log coordinates."""
  import pyimr
  from pyimr.selection import fit_candidate, physical_from_unit
  from scipy.ndimage import median_filter

  times, mean, spread, maximum, stretch = records.load(dataset)
  candidate = candidate_at_ratio(RATIO)
  solve = records.solver(times, maximum, stretch)
  fit = fit_candidate(candidate, solve, mean, spread, bounds=BOX, starts=STARTS,
                      max_evaluations=EVALUATIONS)
  fitted = dict(zip(AXES, (float(v) for v in physical_from_unit(AXES, fit.unit, BOX)),
                    strict=True))
  point = np.array([fitted[a] for a in AXES])

  def trace(scale):
    moved = dict(zip(AXES, (float(v) for v in point * scale), strict=True))
    config = pyimr.SimulationConfig(maximum, maximum / stretch, candidate.build(moved),
                                    dynamics="keller-miksis", rtol=1e-9, atol=1e-11,
                                    max_steps=600_000)
    return np.asarray(pyimr.simulate(times, config).radius_ratio, dtype=float)

  step = 1e-4
  # log coordinates, to match the covariance per_trial_fits.py reports
  jacobian = np.column_stack([
    point[k] * (trace(np.where(np.arange(len(AXES)) == k, 1 + step, 1.0))
                - trace(np.where(np.arange(len(AXES)) == k, 1 - step, 1.0))) / (2 * step * point[k])
    for k in range(len(AXES))])
  sigma = median_filter(spread, SMOOTH, mode="nearest")
  whitened = jacobian / sigma[:, None]
  matrix = whitened.T @ whitened
  return 0.5 * (matrix + matrix.T), times.size


def main():
  measured = json.load(open(records.HERE / "per_trial_fits.json"))
  print("\n  measured Sigma_theta (log coordinates), from per-trial fits\n")

  table = {}
  for dataset in records.DATASETS:
    covariance = np.array(measured[dataset]["log_covariance"])
    fisher, samples = per_sample_information(dataset)
    trials = measured[dataset]["trials"]

    def information(J, N, covariance=covariance, fisher=fisher, samples=samples):
      """log det F(J, N), with F_N scaled linearly in the sample count."""
      scaled = fisher * (N / samples)
      bracket = covariance + np.linalg.inv(scaled)
      return float(np.linalg.slogdet(J * np.linalg.inv(bracket))[1])

    # what another frame buys against what another bubble buys, at the record's own split
    here = information(trials, samples)
    per_frame = information(trials, samples * 2) - here
    per_trial = information(trials * 2, samples) - here
    # at a fixed frame budget, where should N sit?
    budget = trials * samples
    grid = np.unique(np.clip((budget / np.arange(2, trials * 4 + 1)).astype(int), 5, budget))
    best, best_value = None, -np.inf
    for count in grid:
      J = max(1, budget // int(count))
      value = information(J, int(count))
      if value > best_value: best, best_value = (J, int(count)), value
    table[dataset] = {"trials": trials, "samples": samples, "logdet": here,
                      "doubling_samples": per_frame, "doubling_trials": per_trial,
                      "best_split": list(best), "best_logdet": best_value,
                      "gain_over_current": best_value - here}
    print(f"  {dataset:13s} J={trials:2d} N={samples:3d}   log det F = {here:8.3f}")
    print(f"    doubling the SAMPLES  {per_frame:+7.3f} nats")
    print(f"    doubling the TRIALS   {per_trial:+7.3f} nats")
    print(f"    at a fixed {budget} frames the best split is J={best[0]}, N={best[1]}"
          f"  ({best_value - here:+.3f} nats)")

  print("\n  Trials enter linearly and never saturate; samples enter through F_N and do.")
  print("  A split that buys more from a bubble than from a frame is a design variable the")
  print("  batch optimisation has never been allowed to move.")
  json.dump(table, open(records.HERE / "trials_versus_samples.json", "w"), indent=1)


if __name__ == "__main__":
  main()
