r"""Is the noise scale predicted by the normalisation, rather than measured sample by sample?

Each trace is $y_j(t) = R_j(t) / R_{\max,j}$, a ratio of two measured lengths. If each carries
independent error $s$, then
%
    \delta y = (\delta R - y\, \delta R_{\max}) / R_{\max},
    \qquad \operatorname{var}(y) = (s/R_{\max})^2 (1 + y^2),

so the spread is not flat and not free: it is affine in $1 + y^2$ with slope set by the
measurement error alone. At the peak sample the two measurements are THE SAME NUMBER, the
errors cancel identically, and the variance is structurally zero -- which is exactly what the
records show, a minimum $30$ to $90$ times below the median sitting at index $0$ or $1$ where
the mean is $0.9996$ or above.

That matters because the per-sample spread is currently used two ways it should not be. The
design work replaces it with a single constant, which throws away a shape the algebra
predicts. And `fit_candidate` weights by $1/\sigma^2$ using the raw estimate, which hands
$28$ to $63\%$ of the likelihood to a sample whose spread is zero by construction rather than
by precision.

Fitting $\sigma^2 = a^2 (1 + y^2) + b^2$ would separate the two contributions -- $a$ length
measurement, $b$ whatever varies between bubbles -- and a closed form is what the design work
needs, since a per-sample estimate cannot be evaluated at a geometry nobody has run.

MEASURED, THE PREDICTION IS WRONG, and the sign is what kills it. Away from the peak the
correlation between $\sigma^2$ and $1 + y^2$ is $-0.269$, $-0.088$ and $-0.052$: the spread is
LARGEST where $y$ is SMALLEST. The medians by phase say the same thing plainly --
$0.008$--$0.013$ before the collapse against $0.022$--$0.036$ at it. Independent length errors
on $R$ and $R_{\max}$ predict the opposite, so they are not what dominates this spread.

What remains is that the cancellation at the peak is real and local, and the rest of the
profile is dynamical: bubbles differ most where the dynamics are most sensitive, which is the
collapse and the rebound after it. That has a consequence for the design work. No closed-form
noise model follows from the measurement geometry, so a scale for an unrun geometry has to
come from the dynamics -- propagating a spread in the initial condition through the model --
rather than from a formula in $y$.

Reported as a refutation rather than deleted: the algebra is correct and the conclusion drawn
from it is not, which is worth more on the record than a script that fits nothing.
"""

import json

import numpy as np

import records


def one(dataset):
  times, mean, spread, maximum, stretch = records.load(dataset)
  trials = records.trial_count(dataset)
  design = np.column_stack([1.0 + mean**2, np.ones(mean.size)])
  variance = spread**2
  # weighted by the precision of a variance estimate from J trials, which is proportional to
  # the variance itself -- an unweighted fit lets the largest samples set the whole line
  weights = 1.0 / np.maximum(variance, 1e-30)
  solution, *_ = np.linalg.lstsq(design * weights[:, None], variance * weights, rcond=None)
  a2, b2 = float(solution[0]), float(solution[1])
  predicted = design @ solution
  ok = predicted > 0
  residual = np.log(np.maximum(variance[ok], 1e-30)) - np.log(predicted[ok])
  total = np.log(variance[ok]) - np.log(variance[ok]).mean()
  r2 = 1.0 - float(residual @ residual) / float(total @ total)
  away = mean < 0.9                       # away from the peak, where the cancellation lives
  return dataset, {
    "a": float(np.sqrt(max(a2, 0.0))), "b": float(np.sqrt(max(b2, 0.0))),
    "r2_log": r2,
    "corr_away": float(np.corrcoef(variance[away], (1.0 + mean**2)[away])[0, 1]),
    "sigma_median": float(np.median(spread)),
    "predicted_at_peak": float(np.sqrt(max(a2 * 2.0 + b2, 0.0))),
    "observed_at_peak": float(spread[int(np.argmax(mean))]),
    "trials": trials,
  }


def main():
  table = dict(one(name) for name in records.DATASETS)
  print("\n  sigma^2 = a^2 (1 + y^2) + b^2, fitted to the per-sample spread\n")
  print(f"  {'record':13s} {'J':>3s} {'a (length)':>11s} {'b (residual)':>13s} "
        f"{'R^2 in log':>11s} {'corr away from peak':>20s}")
  for name, got in table.items():
    print(f"  {name:13s} {got['trials']:3d} {got['a']:11.5f} {got['b']:13.5f} "
          f"{got['r2_log']:11.3f} {got['corr_away']:20.3f}")

  print("\n  a negative correlation refutes the form: independent errors on R and R_max give")
  print("  a POSITIVE slope in 1 + y^2, and the records give the opposite on all three.\n")
  print(f"  {'record':13s} {'sigma at peak':>14s} {'sigma median':>13s} {'ratio':>8s}"
        "   the cancellation, which is real")
  for name, got in table.items():
    print(f"  {name:13s} {got['observed_at_peak']:14.5f} {got['sigma_median']:13.5f} "
          f"{got['observed_at_peak'] / got['sigma_median']:8.4f}")
  print("\n  So the peak sample is special for the reason the algebra gives -- the numerator")
  print("  and denominator are one measurement there -- but the rest of the profile is")
  print("  dynamical, and a noise scale for an unrun geometry has to be propagated through")
  print("  the model rather than read off a formula in y.")

  json.dump(table, open(records.HERE / "noise_shape.json", "w"), indent=1)


if __name__ == "__main__":
  main()
