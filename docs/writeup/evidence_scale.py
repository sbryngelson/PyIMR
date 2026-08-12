r"""What a nat means in this document, and why three sections disagree by a factor of four.

The same model on the same record is reported with three different log evidences:

    -2289.86   the grid quadrature in `results.json`
     +529.26   the capped Laplace expansion of \cref{sec:enrich}
    +2173.55   the sequential Monte Carlo of \cref{sec:limitations}

A reader cannot tell whether a nat in \cref{sec:thermal} means what a nat in
\cref{sec:operator} means, and every threshold in this document --- the five nats
\cref{eq:screen} screens on, the ``substantial'' of Jeffreys' scale --- is stated in nats.

An evidence is only defined up to the normalisation of its likelihood, and these three make
different choices. This decomposes one fit into its terms so the offsets can be attributed
rather than guessed, and so the appendix can say which convention each reported number uses.

THE TERMS. For a Gaussian likelihood with the prior uniform on the unit cube,

    log Z = -chi^2 / 2  -  (1/2) sum_i log(2 pi sigma_i^2)  +  occam ,

and the second term is the whole argument: it is not a constant across conventions, because
`sigma` is the trial spread in one place and the standard error of the mean --- smaller by
`sqrt(J)` --- in another. \Cref{sec:latent} already measures what that choice does to
`chi^2/N`, moving it from $0.974$ to $17.54$ on one residual. It does the same to `log Z`,
by `N log sqrt(J)`, and nothing in the document says which sections took which.
"""

import json

import numpy as np

import records
from identified import BOX, candidate_at_ratio

RATIO = 38.5
AXES = ("mu", "galpha", "lambda1")
STARTS, EVALUATIONS = 16, 600
DATASET = "gelatin_15C"
# what the document reports for qSLS on this record, and where
REPORTED = {"grid quadrature (results.json)": -2289.86,
            "capped Laplace (sec:enrich)": 529.26,
            "sequential Monte Carlo (sec:limitations)": 2173.55}


def main():
  from pyimr.discriminate import laplace_log_evidence
  from pyimr.selection import evaluate_at, fit_candidate, physical_from_unit

  times, mean, spread, maximum, stretch = records.load(DATASET)
  trials = records.trial_count(DATASET)
  candidate = candidate_at_ratio(RATIO)
  solve = records.solver(times, maximum, stretch)
  fit = fit_candidate(candidate, solve, mean, spread, bounds=BOX, starts=STARTS,
                      max_evaluations=EVALUATIONS)
  fitted = dict(zip(AXES, (float(v) for v in physical_from_unit(AXES, fit.unit, BOX)),
                    strict=True))
  model = np.asarray(evaluate_at(candidate, solve, fitted)[0], dtype=float)

  # the Jacobian in the UNIT coordinates the prior is flat on, which is the space the Occam
  # factor has to be measured in
  step = 1e-4
  columns = []
  for k in range(len(AXES)):
    up, down = np.array(fit.unit, dtype=float), np.array(fit.unit, dtype=float)
    up[k], down[k] = min(up[k] + step, 1.0), max(down[k] - step, 0.0)
    traces = []
    for point in (up, down):
      values = physical_from_unit(AXES, point, BOX)
      traces.append(np.asarray(evaluate_at(
        candidate, solve, dict(zip(AXES, values, strict=True)))[0], dtype=float))
    columns.append((traces[0] - traces[1]) / (up[k] - down[k]))
  raw_jacobian = np.column_stack(columns)

  print(f"\n  {DATASET}: N = {mean.size} samples, J = {trials} trials, "
        f"median spread {np.median(spread):.5f}\n")
  print(f"  {'sigma convention':28s} {'chi2/N':>9s} {'-chi2/2':>11s} "
        f"{'-(1/2)sum log 2 pi s^2':>24s} {'occam':>9s} {'log Z':>11s}")

  rows = {}
  for name, scale in (("trial spread", spread),
                      (f"standard error (/sqrt {trials})", spread / np.sqrt(trials))):
    residual = (model - mean) / scale
    jacobian = raw_jacobian / scale[:, None]
    chi2 = float(residual @ residual)
    normalization = float(np.sum(np.log(2.0 * np.pi * scale**2)))
    plain = laplace_log_evidence(residual, jacobian, scale)
    occam = plain + 0.5 * chi2 + 0.5 * normalization
    rows[name] = {"chi2_per_n": chi2 / mean.size, "chi2": chi2,
                  "normalization": normalization, "occam": float(occam),
                  "log_evidence": float(plain),
                  "capped": float(laplace_log_evidence(residual, jacobian, scale,
                                                       cap_at_prior=True))}
    print(f"  {name:28s} {chi2 / mean.size:9.4f} {-0.5 * chi2:11.2f} "
          f"{-0.5 * normalization:24.2f} {occam:9.2f} {plain:11.2f}")

  shift = mean.size * np.log(np.sqrt(trials))
  gap = rows["trial spread"]["log_evidence"] - rows[f"standard error (/sqrt {trials})"]["log_evidence"]
  print(f"\n  the two differ by {gap:.2f}; N log sqrt(J) alone accounts for {shift:.2f}, and the")
  print(f"  remainder {gap - shift:+.2f} is the chi-squared, which scales by J rather than shifting.")

  print("\n  against what the document reports for this model on this record\n")
  print(f"  {'reported as':44s} {'value':>10s} {'offset from trial-spread Laplace':>34s}")
  base = rows["trial spread"]["log_evidence"]
  for label, value in REPORTED.items():
    print(f"  {label:44s} {value:10.2f} {value - base:34.2f}")

  print("\n  The grid quadrature is not this integral: sec:records adds a marginalised noise")
  print("  scale, a redundancy weight and a BIC-style penalty to it, and quadratures over a")
  print("  box rather than expanding about a mode. It is not on this scale and should not be")
  print("  differenced against numbers that are.")

  json.dump({"dataset": DATASET, "samples": int(mean.size), "trials": int(trials),
             "conventions": rows, "reported": REPORTED,
             "n_log_sqrt_j": float(shift)},
            open(records.HERE / "evidence_scale.json", "w"), indent=1)


if __name__ == "__main__":
  main()
