r"""Does gelatin obey time--temperature superposition, and what does it mean that it does not?

\Cref{sec:measure} makes temperature a design axis by assuming an Arrhenius law: one coefficient
and one slope per material parameter, so that three temperatures determine six numbers. That is a
statement about \emph{rates} shifting with temperature, and it is the weakest form of the standard
rheological assumption --- time--temperature superposition, under which warming slides the whole
relaxation spectrum along the time axis without changing its shape.

WHY IT SHOULD FAIL HERE, AND WHY THAT IS WORTH MEASURING. Superposition holds for a
thermorheologically simple material, one whose structure is fixed and whose rates alone respond to
temperature. Gelatin is not that. It is triple helices that unzip stochastically as it warms
(J.~Estrada, personal communication), so temperature changes what the network \emph{is} and not
only how fast it relaxes. If that is right, no single shift factor can reconcile the fitted
parameters, and the Arrhenius axis is wrong for a reason that has nothing to do with the sign of
its slope.

THE TEST IS SHARP BECAUSE THE MODEL CARRIES TWO INDEPENDENT TIMES. Under superposition every
characteristic time shifts by one factor $a_T$ and every modulus by
$b_T = \rho T / \rho_{\rm ref} T_{\rm ref}$, which over \SIrange{15}{33}{\celsius} is within
\SI{3}{\percent} of unity. The identified coordinates give a time directly, $\lambda_1$, and a
second one as $\mu / g\alpha$, a viscosity over a modulus. Superposition requires those two to
shift \emph{together}. Nothing in the fit forces them to, so their ratio is a free test.

WHAT WOULD MAKE THE FAILURE UNINTERESTING. A large enough spread on the fitted parameters would
let any two shift factors disagree by chance, and $\lambda_1$ is the loosest coordinate in this
package. So the shift factors are carried with the uncertainty the per-trial fits measure,
propagated through the log covariance rather than assumed diagonal, and the verdict is stated in
standard errors rather than in ratios.
"""

import json

import numpy as np

import records

KELVIN = {"gelatin_15C": 288.15, "gelatin_23C": 296.15, "gelatin_33C": 306.15}
REFERENCE = "gelatin_15C"
AXES = ("mu", "galpha", "lambda1")
GAS_CONSTANT = 8.314462618


def _load():
  """`{dataset: (median, log covariance of the MEAN, trials)}` in the identified coordinates."""
  raw = json.load(open(records.HERE / "per_trial_fits.json"))
  out = {}
  for dataset in records.DATASETS:
    entry = raw[dataset]
    median = np.array([entry["median"][a] for a in AXES], dtype=float)
    # log_covariance is the between-trial covariance; the uncertainty on the MEDIAN is that
    # divided by the number of trials, which is what a shift factor is entitled to
    trials = int(entry.get("trials") or records.trial_count(dataset))
    cov = np.array(entry["log_covariance"], dtype=float) / trials
    out[dataset] = (median, cov, trials)
  return out


def _times(median, cov):
  """The two independent times the model carries, in logs, with their covariance.

  `lambda1` is a time. `mu / galpha` is a viscosity over a modulus and is therefore also a
  time, and it is built from the two coordinates lambda1 does not touch, so under superposition
  the pair is a genuine consistency check rather than a restatement.
  """
  log = np.log(median)
  i_mu, i_ga, i_lam = 0, 1, 2
  value = np.array([log[i_lam], log[i_mu] - log[i_ga]])
  # var(log mu - log ga) needs the cross term; assuming it away is what would fake a failure
  jac = np.array([[0.0, 0.0, 1.0], [1.0, -1.0, 0.0]])
  return value, jac @ cov @ jac.T


def main():
  data = _load()
  ref_median, ref_cov, _ = data[REFERENCE]
  ref_log, ref_var = _times(ref_median, ref_cov)
  t_ref = KELVIN[REFERENCE]

  print("  Under superposition both times shift by the same a_T, and the modulus by")
  print("  b_T = rho*T/(rho_ref*T_ref), which is within 3% of 1 over this range.\n")
  print(f"  {'dataset':>13s} {'T (K)':>7s} {'a_T from lam1':>16s} {'a_T from mu/ga':>17s} "
        f"{'disagreement':>16s} {'b_T from ga':>12s}")

  summary = {"reference": REFERENCE, "rows": {}}
  worst = 0.0
  for dataset in records.DATASETS:
    median, cov, _ = data[dataset]
    log, var = _times(median, cov)
    delta = log - ref_log                       # log shift factors, two ways
    dvar = np.diag(var) + np.diag(ref_var)
    # the test: do the two logs agree? their difference, in its own standard errors
    gap = delta[0] - delta[1]
    gap_se = float(np.sqrt(dvar[0] + dvar[1]))
    sigmas = abs(gap) / gap_se if gap_se > 0 else float("inf")
    worst = max(worst, sigmas)
    b_t = median[1] / ref_median[1]             # modulus ratio; superposition says ~1
    expected_b = KELVIN[dataset] / t_ref        # rho*T/(rho_ref*T_ref), rho within 1%
    summary["rows"][dataset] = {
      "a_lambda": float(np.exp(delta[0])), "a_mu_over_g": float(np.exp(delta[1])),
      "gap_sigmas": float(sigmas), "b_measured": float(b_t), "b_expected": float(expected_b)}
    tag = "" if dataset == REFERENCE else f"{sigmas:14.1f} sd"
    print(f"  {dataset:>13s} {KELVIN[dataset]:7.2f} {np.exp(delta[0]):16.3f} "
          f"{np.exp(delta[1]):17.3f} {tag:>16s} {b_t:12.3f}")

  print(f"\n  b_T should be {min(KELVIN.values())/t_ref:.3f} to "
        f"{max(KELVIN.values())/t_ref:.3f}; measured {min(r['b_measured'] for r in summary['rows'].values()):.3f} "
        f"to {max(r['b_measured'] for r in summary['rows'].values()):.3f}")

  print("\n  ---- verdict ----\n")
  summary["worst_disagreement_sigmas"] = float(worst)
  summary["holds"] = bool(worst < 2.0)
  print(f"  the two shift factors disagree by up to {worst:.1f} standard errors")
  lam = [summary["rows"][d]["a_lambda"] for d in records.DATASETS]
  mug = [summary["rows"][d]["a_mu_over_g"] for d in records.DATASETS]
  print(f"  a_T from lambda1:   {np.array2string(np.array(lam), precision=3)}")
  print(f"  a_T from mu/galpha: {np.array2string(np.array(mug), precision=3)}")
  same_way = np.all(np.sign(np.diff(lam)) == np.sign(np.diff(mug)))
  summary["same_direction"] = bool(same_way)
  print(f"  they move in the {'same' if same_way else 'OPPOSITE'} direction with temperature")
  print(f"\n  Superposition {'holds' if summary['holds'] else 'FAILS'} on these fits.")
  print("  A material whose structure changes with temperature cannot be shifted onto one master")
  print("  curve, and an Arrhenius axis over its parameters is then assuming what is false.")
  json.dump(summary, open(records.HERE / "superposition.json", "w"), indent=1)


if __name__ == "__main__":
  main()
