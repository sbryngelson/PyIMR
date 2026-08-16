r"""What can the bounce sweep identify that the trace cannot?

`bounce_sweep.py` establishes that the ratio of successive bounce maxima is clock-invariant,
derivative-free and computable per trial, and that the fitted model gets its shape wrong the same
way on all three datasets. That says the observable SEES something. It does not say what the
observable can MEASURE, and those are different questions: an observable can carry a discrepancy
and still be unable to identify the parameter responsible for it.

THE CLAIM WORTH TESTING. Two facts from earlier make a sharp prediction. `clock_screen.py`
measures a density perturbation and a pure clock removing the same share of $\hat\delta$ to within
$0.4$ percentage points, so $\rho$ acts here as a clock. And a ratio of amplitudes is invariant
under any rescaling of time. Together those say the bounce ratios should be nearly BLIND to
$\rho$, which is exactly the coordinate that confounds the trace fit and absorbs the $5$--$9\%$
deficit. An observable blind to the confounder cannot be corrupted by it.

AND A SECOND PREDICTION THAT COULD FAIL. `glassy_and_spectrum.py` finds $\lambda_1$ has no effect
on the collapse timing at all, across a sixteenfold change in $\mu/\lambda_1$. That is the obvious
explanation for $\lambda_1$ being the loosest coordinate here, at $51$ to \SI{117}{\percent}
between-trial spread. If the bounce sequence is genuinely a rate sweep, $\lambda_1$ should move
the ratios DIFFERENTLY at different bounces, because each bounce sits at a different rate. If it
moves them all alike, or not at all, the sweep is not resolving a relaxation time and the
spectrum reading is wrong.

WHAT IT FOUND, AND THE FIRST PREDICTION IS WRONG. The ratios are NOT blind to $\rho$: it carries
$52$ to \SI{65}{\percent} of the largest sensitivity, and $\mu$ against $\rho$ has a cosine of
$-0.971$, $-0.960$ and $-0.994$ in this observable. They are as confounded here as in the trace.

The error in the reasoning above is worth stating because it is not obvious. A ratio of
amplitudes IS invariant under a pure rescaling of time. `clock_screen.py` measured that $\rho$ and
a pure clock remove the same share of $\hat\delta$, and that was read as ``$\rho$ acts as a
clock''. It does not follow. Removing the same share of one residual is a statement about a
projection, not about the map: $\rho$ enters the Weber and Mach groups too, so it changes the
trajectory and with it the amplitudes, and no invariance of the observable protects against that.
Clock-invariance buys immunity to a clock, which $\rho$ is not.

The second prediction survives. $\lambda_1$ moves the ratios by $6.8$ to $14.7$ times as much at
one bounce as at another, so the sweep does resolve rate dependence differentially where the
trace resolves none at all. It is still degenerate with $g\alpha$ and with $\rho$, so this buys a
diagnostic and not an identification, which is the distinction the section this feeds should
make.

HOW IT IS MEASURED. The sensitivity of each ratio to each parameter in logs, whitened by the
between-trial spread of that ratio, which `bounce_sweep.py` measures rather than assumes. That
makes the columns comparable and the resulting Gram matrix an information matrix in the same
sense as \cref{sec:identifiability}'s, so its condition number and null direction can be read the
same way. The trace is put through the identical construction for contrast.
"""

import json

import numpy as np

import records
from bounce_sweep import KEEP, RATIO, measured, sequence

AXES = ("mu", "galpha", "lambda1", "rho")
STEP = 0.02                                  # fractional, in logs
BASE_RHO = 1064.0


def _trace_and_ratios(dataset, scales):
  import pyimr

  times, _, _, maximum, stretch = records.load(dataset)
  m = json.load(open(records.HERE / "per_trial_fits.json"))[dataset]["median"]
  product = m["galpha"] * scales.get("galpha", 1.0)
  material = pyimr.QuadraticZener(float(np.sqrt(product * RATIO)),
                                  m["mu"] * scales.get("mu", 1.0),
                                  m["lambda1"] * scales.get("lambda1", 1.0), 0.0,
                                  float(np.sqrt(product / RATIO)))
  physics = pyimr.PhysicalParameters(medium_density_kg_m3=BASE_RHO * scales.get("rho", 1.0))
  fine = np.linspace(times[0], times[-1], 20001)
  config = pyimr.SimulationConfig(maximum, maximum / stretch, material,
                                  dynamics="keller-miksis", rtol=1e-9, atol=1e-11,
                                  max_steps=2_000_000, physics=physics)
  trace = np.asarray(pyimr.simulate(fine, config).radius_ratio, dtype=float)
  phase = (fine - fine[0]) / (fine[-1] - fine[0]) * 5.0
  s = sequence(trace, phase)
  # the trace is also returned on the record's own grid, so the contrast is like for like
  on_record = np.asarray(pyimr.simulate(times, config).radius_ratio, dtype=float)
  return (None if s is None else s["ratios"][:KEEP]), on_record


def one(dataset):
  obs = measured(dataset)
  spread = np.array(obs["ratio_spread"], dtype=float)
  n = len(spread)
  base_ratios, base_trace = _trace_and_ratios(dataset, {})
  if base_ratios is None or len(base_ratios) < n: return dataset, None

  _, _, trace_spread, _, _ = records.load(dataset)
  ratio_cols, trace_cols = [], []
  for axis in AXES:
    up, up_t = _trace_and_ratios(dataset, {axis: 1 + STEP})
    dn, dn_t = _trace_and_ratios(dataset, {axis: 1 - STEP})
    if up is None or dn is None or len(up) < n or len(dn) < n: return dataset, None
    ratio_cols.append((np.array(up[:n]) - np.array(dn[:n])) / (2 * STEP) / spread)
    trace_cols.append((up_t - dn_t) / (2 * STEP) / trace_spread)
  return dataset, {
    "ratio_jacobian": np.column_stack(ratio_cols).tolist(),
    "trace_jacobian_norm": [float(np.linalg.norm(c)) for c in trace_cols],
    "ratio_norm": [float(np.linalg.norm(c)) for c in ratio_cols],
    "base_ratios": [float(v) for v in base_ratios[:n]],
    "ratio_spread": spread.tolist()}


def _report(name, jac):
  fisher = jac.T @ jac
  values, vectors = np.linalg.eigh(fisher)
  return {"condition": float(values[-1] / max(values[0], 1e-300)),
          "null": [float(v) for v in vectors[:, 0]],
          "eigenvalues": [float(v) for v in values]}


def main():
  print("  Sensitivity of each observable to each parameter, whitened by that observable's own")
  print("  measured between-trial spread. rho is the coordinate that confounds the trace fit.\n")

  summary = {}
  with records.pool(len(records.DATASETS)) as pool:
    got = dict(pool.map(one, list(records.DATASETS)))

  for dataset in records.DATASETS:
    v = got[dataset]
    if v is None:
      print(f"  {dataset}: not enough resolved bounces"); continue
    jac = np.array(v["ratio_jacobian"], dtype=float)
    print(f"  ==== {dataset} ====\n")
    print(f"  {'parameter':>10s} {'|d ratios|':>11s} {'|d trace|':>11s} {'ratio/trace':>12s}")
    for k, axis in enumerate(AXES):
      r, t = v["ratio_norm"][k], v["trace_jacobian_norm"][k]
      print(f"  {axis:>10s} {r:11.2f} {t:11.2f} {r/t if t > 0 else float('nan'):12.4f}")
    # per-bounce, to see whether lambda1 acts differently at different rates
    print(f"\n  {'bounce':>10s} " + " ".join(f"{k+1:>9d}" for k in range(jac.shape[0])))
    for k, axis in enumerate(AXES):
      print(f"  {axis:>10s} " + " ".join(f"{jac[b, k]:9.2f}" for b in range(jac.shape[0])))
    rep = _report(dataset, jac)
    summary[dataset] = {**v, **rep}
    print(f"\n  condition number {rep['condition']:.3g}; weakest direction "
          + ", ".join(f"{a} {c:+.2f}" for a, c in zip(AXES, rep["null"], strict=True)))
    print()

  print("  ---- what it says ----\n")
  for dataset in records.DATASETS:
    v = summary.get(dataset)
    if not v: continue
    i = AXES.index("rho")
    share = v["ratio_norm"][i] / max(v["ratio_norm"])
    lam = np.array(v["ratio_jacobian"], dtype=float)[:, AXES.index("lambda1")]
    varies = float(np.max(np.abs(lam)) / max(np.min(np.abs(lam)), 1e-12))
    summary[dataset]["rho_share"] = share
    summary[dataset]["lambda1_variation"] = varies
    print(f"  {dataset:>13s}: rho carries {100*share:5.1f} percent of the largest sensitivity; "
          f"lambda1 varies {varies:.1f}x across bounces")
  print("\n  A small rho share means the ratios cannot be corrupted by the coordinate that")
  print("  absorbs the deficit. A large lambda1 variation across bounces means the sweep")
  print("  resolves a relaxation time, which the trace demonstrably does not.")
  json.dump(summary, open(records.HERE / "bounce_identify.json", "w"), indent=1)


if __name__ == "__main__":
  main()
