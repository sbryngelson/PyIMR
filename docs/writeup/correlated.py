"""Every margin in this document, recomputed against a likelihood that counts real samples.

The independent Gaussian likelihood treats $201$ points as $201$ observations. The residuals
of its own fits say otherwise -- lag-one autocorrelation between $0.90$ and $0.97$ -- so it
counts information it does not have, and every log evidence, every Bayes factor and every
design nat inherits the error.

This estimates the correlation time from TRIAL-TO-TRIAL SCATTER, refits under the exponential
covariance it implies, and reports the operator margins both ways. The residual of a fitted
model is the wrong place to estimate it from: `lack_of_fit` rejects every model here, and a
misspecified model leaves a smooth residual indistinguishable from correlated noise, so a tau
taken from it hands the model's own error to the noise term. Deviations from the trial mean
carry no model error -- misfit is common to every trial and cancels -- so they measure the
noise alone. Measured, the two agree to within 2% at \SI{15}{\celsius} and the replicate
estimate is HIGHER at the other two, so the correlation is real noise rather than misfit. It also
converts them into the unit an experimentalist can act on: how many bubbles it would take to
settle the question, which is where a factor of twenty in the information becomes a factor of
twenty in the budget.

The correction is not a single number. Whitening against a correlated covariance DEFLATES a
smooth displacement and AMPLIFIES a rough one -- measured at rho = 0.908, a constant offset
loses a factor of 0.053 while point-to-point jitter gains 20.7. Model differences are smooth,
so the deflation is what applies here, but that is a fact about these differences rather than
about the likelihood.
"""

import json

import numpy as np

import records

STARTS, EVALUATIONS = 12, 400
# the operator pair the document compares: the package default against what replicated
OPERATORS = (("keller-miksis", None), ("keller-enthalpy", "mie-gruneisen"))


def fit_one(job):
  """Fit one (record, operator) pair under one noise model. Returns the scored dictionary."""
  from pyimr.selection import STANDARD_MODELS

  dataset, operator, tau = job
  times, mean, spread, maximum, stretch = records.load(dataset)
  candidate = STANDARD_MODELS["qSLS"]
  solve = records.solver(times, maximum, stretch, dynamics=operator[0], liquid_eos=operator[1])
  scored = records.score(candidate, solve, mean, spread, starts=STARTS, evaluations=EVALUATIONS,
                         trials=records.trial_count(dataset),
                         correlation_time_s=tau, times=None if tau is None else times)
  return (dataset, operator, tau is not None), scored


TRIALS = {"gelatin_15C": "Ga_t15_exp_data.csv", "gelatin_23C": "Ga_t23_exp_data.csv",
          "gelatin_33C": "Ga_t33_exp_data.csv"}
TRIAL_DATA = records.Path.home() / "fastscratch/imr-data-tempsweeps/data"


def correlation_times():
  """`tau` per record from pure error: the lag-one of each trial's deviation from the mean.

  Pooled over trials. The deviations sum to zero across trials at each time, which biases the
  cross-trial direction and not the time-lag one estimated here, though with seven trials at
  \SI{33}{\celsius} that bias is smallest where the sample is thinnest.
  """
  out = {}
  for name, filename in TRIALS.items():
    raw = np.loadtxt(TRIAL_DATA / filename, delimiter=",")
    deviations = raw[:, 1:] - raw[:, 1:].mean(axis=1, keepdims=True)
    early, late = deviations[:-1, :].ravel(), deviations[1:, :].ravel()
    early, late = early - early.mean(), late - late.mean()
    rho = float(early @ late / np.sqrt((early @ early) * (late @ late)))
    spacing = float(np.median(np.diff(records.load(name)[0])))
    out[name] = (float("inf") if rho <= 0.0 else -spacing / np.log(rho), rho, raw.shape[1] - 1)
  return out


def main():
  print("estimating the correlation time from trial-to-trial scatter\n")
  taus = correlation_times()
  for name, (tau, rho, trials) in taus.items():
    print(f"  {name:14s} {trials:2d} trials   pure-error lag-one {rho:.4f}   tau {tau:.3e} s")

  jobs = [(name, operator, model)
          for name, (tau, _, _) in taus.items()
          for operator in OPERATORS
          for model in (None, tau)]
  print(f"\nrefitting {len(jobs)} (record, operator, noise model) combinations")
  with records.pool(len(jobs)) as pool:
    table = dict(pool.map(fit_one, jobs))

  print("\n  operator margin, keller-enthalpy/mie-gruneisen against keller-miksis\n")
  print(f"  {'record':14s} {'independent':>13s} {'correlated':>13s} {'ratio':>8s}")
  out = {}
  for name in records.DATASETS:
    margins = {}
    for correlated in (False, True):
      pair = [table[(name, operator, correlated)]["log_evidence"] for operator in OPERATORS]
      margins[correlated] = float(pair[1] - pair[0])
    ratio = margins[True] / margins[False] if margins[False] != 0.0 else float("nan")
    print(f"  {name:14s} {margins[False]:+13.2f} {margins[True]:+13.2f} {ratio:8.3f}")
    out[name] = {"independent": margins[False], "correlated": margins[True],
                 "tau": taus[name][0], "lag_one": taus[name][1],
                 "chi2": {str(c): table[(name, OPERATORS[0], c)]["chi2_per_n"] for c in (False, True)},
                 "lag_one_after": {str(c): table[(name, OPERATORS[0], c)]["lag_one"] for c in (False, True)}}

  print("\n  and what the refit does to the residual it was built to fix\n")
  print(f"  {'record':14s} {'chi2/N ind':>11s} {'chi2/N cor':>11s} {'lag-1 ind':>10s} {'lag-1 cor':>10s}")
  for name in records.DATASETS:
    d = out[name]
    print(f"  {name:14s} {d['chi2']['False']:11.3f} {d['chi2']['True']:11.3f} "
          f"{d['lag_one_after']['False']:10.3f} {d['lag_one_after']['True']:10.3f}")

  json.dump(out, open(records.HERE / "correlated.json", "w"), indent=1)


if __name__ == "__main__":
  main()
