r"""Is the afterbounce shape error reachable by any material, or is it structural?

Three datasets agree on a signature: the fitted model damps LESS than the record at the
first afterbounce and MORE at the second, at $-9.3/+7.5$, $-4.2/+6.0$ and $-3.9/+3.1$
standard errors of the mean. It survives reading the model on the record's own grid, it
survives the change of normalisation between `bounce_sweep.py` and `frequency_space.py`, and
`bounce_density.py` finds no density in the swept bracket that removes it.

TWO FAMILIES EXPLAIN IT AND THEY ARE NOT THE SAME KIND OF CLAIM. The relaxation time could
simply be wrong: a qSLS with too long a $\lambda_1$ spreads its dissipation over too many
bounces, where the record concentrates it at the fastest one. That is a fixed material with a
badly determined coordinate, and it is the live possibility because `glassy_and_spectrum.py`
finds $\lambda_1$ has NO effect on collapse timing across a sixteenfold change, so the trace
never constrained it and the afterbounces are the first observable that could. Or the
material changed during the first collapse -- triple helices unzipping irreversibly
(J.~Estrada, personal communication) -- in which case no single material describes both
halves and the model is structurally wrong rather than mis-parameterised.

THE TEST SEPARATES THEM WITHOUT ASSUMING EITHER. Fit the material directly to the afterbounce
ratio sequence, over a range four orders of magnitude wide in $\lambda_1$, and ask what
$\chi^2$ is ACHIEVABLE. If a fixed qSLS can reproduce the sequence, the shape error is a
mis-fitted relaxation time and the afterbounces are the observable that fixes it. If the best
achievable fit is still far outside the record's own scatter, no fixed material reproduces
both halves and the signature is structural.

THE CONTROL IS NOT OPTIONAL HERE, BECAUSE A LARGE $\chi^2$ IS ALSO WHAT A FAILED OPTIMISER
RETURNS. So the identical fitter is run against a SYNTHETIC sequence generated from the model
itself at a displaced $\lambda_1$, carrying the record's own error bars. If it recovers that
sequence to $\chi^2 \approx 0$, the fitter works and a large $\chi^2$ on the real data means
what it says. If it does not, the real number is uninterpretable and nothing is concluded.
"""

import json

import numpy as np
from scipy.optimize import minimize

import records
from bounce_sweep import RATIO, sequence
from frequency_space import KEEP, SPAN, _families, measured

BASE_RHO = 1064.0
STARTS = 8
# generous, so a failure cannot be blamed on the box. The first pass put galpha on its lower
# wall for two datasets, which is not an optimum, so the elastic axis runs to 1e-6 of the fit.
BOX = {"mu": (1e-3, 1e3), "galpha": (1e-6, 1e3), "lambda1": (1e-4, 1e4)}
EDGE = 1.05                      # within this factor of a wall is not an interior optimum
# a stiff point exhausts the step budget before it raises, so a generous budget makes every
# WASTED evaluation the most expensive one. Fitting fails fast; the reported solves do not.
MAX_STEPS_FIT, MAX_STEPS_REPORT = 150_000, 2_000_000
AXES = ("mu", "galpha", "lambda1")
DISPLACED = 4.0            # the control's lambda1 multiplier


def _ratios(dataset, values, n, max_steps=MAX_STEPS_FIT, cache={}):
  import pyimr

  key = (dataset,)
  if key not in cache:
    cache[key] = records.load(dataset)
  times, _, _, maximum, stretch = cache[key]
  product = max(values["galpha"], 1e-12)
  material = pyimr.QuadraticZener(float(np.sqrt(product * RATIO)), values["mu"],
                                  values["lambda1"], 0.0, float(np.sqrt(product / RATIO)))
  physics = pyimr.PhysicalParameters(medium_density_kg_m3=BASE_RHO)
  fine = np.linspace(times[0], times[-1], 20001)
  config = pyimr.SimulationConfig(maximum, maximum / stretch, material,
                                  dynamics="keller-miksis", rtol=1e-9, atol=1e-11,
                                  max_steps=max_steps, physics=physics)
  try:
    trace = np.asarray(pyimr.simulate(fine, config).radius_ratio, dtype=float)
  except pyimr.SimulationError:
    return None                    # a point the solver cannot reach is infeasible, not a crash
  phase = (fine - fine[0]) / (fine[-1] - fine[0]) * SPAN
  s = sequence(trace, phase)
  if s is None: return None
  got = _families(s["amplitudes"], s["times"])[0]
  return None if len(got) < n else np.array(got[:n], dtype=float)


def _objective(dataset, target, err, base, n):
  """Chi-squared per bounce as a function of the three material coordinates, in logs."""
  def chi2(unit):
    values = {a: float(base[a] * np.exp(unit[k])) for k, a in enumerate(AXES)}
    for a in AXES:
      lo, hi = BOX[a]
      if not (lo * base[a] <= values[a] <= hi * base[a]): return 1e6
    got = _ratios(dataset, values, n)
    if got is None: return 1e6
    return float(np.mean(((got - target) / err) ** 2))
  return chi2


def _fit(chi2, seed):
  """Multistart Nelder-Mead in log coordinates; the starts span the box, not a neighbourhood."""
  rng = np.random.default_rng(seed)
  best, best_x = np.inf, None
  spans = np.array([np.log(BOX[a][1] / BOX[a][0]) for a in AXES])
  for k in range(STARTS):
    x0 = np.zeros(3) if k == 0 else (rng.random(3) - 0.5) * spans * 0.5
    out = minimize(chi2, x0, method="Nelder-Mead",
                   options={"maxiter": 600, "xatol": 1e-3, "fatol": 1e-4})
    if out.fun < best: best, best_x = float(out.fun), out.x
  return best, best_x


def one(dataset):
  obs = measured(dataset)
  n = min(KEEP, len(obs["ratio"]))
  target = np.array(obs["ratio"][:n], dtype=float)
  err = np.array(obs["ratio_spread"][:n], dtype=float) / np.sqrt(obs["events"])
  base = json.load(open(records.HERE / "per_trial_fits.json"))[dataset]["median"]
  base = {a: float(base[a]) for a in AXES}

  start = _ratios(dataset, base, n, MAX_STEPS_REPORT)
  if start is None: return dataset, None
  at_fit = float(np.mean(((start - target) / err) ** 2))

  best, best_x = _fit(_objective(dataset, target, err, base, n), seed=abs(hash(dataset)) % 2**31)
  fitted = {a: float(base[a] * np.exp(best_x[k])) for k, a in enumerate(AXES)}

  # the control: the same fitter against a sequence the model itself can produce
  sham_values = dict(base, lambda1=base["lambda1"] * DISPLACED)
  sham_target = _ratios(dataset, sham_values, n, MAX_STEPS_REPORT)
  sham_best = sham_x = None
  if sham_target is not None:
    sham_best, sham_x = _fit(_objective(dataset, sham_target, err, base, n),
                             seed=abs(hash(dataset + "sham")) % 2**31)
    sham_x = {a: float(base[a] * np.exp(sham_x[k])) for k, a in enumerate(AXES)}

  at_best = _ratios(dataset, fitted, n, MAX_STEPS_REPORT)
  # an argmin on a wall is not an optimum, so it is reported as one rather than read as one
  at_edge = [a for a in AXES
             if not (BOX[a][0] * EDGE < fitted[a] / base[a] < BOX[a][1] / EDGE)]
  # but a wall reached because the objective went FLAT is not a truncated optimum: the chi2 is
  # converged and only that coordinate is unidentified. Pushing to the far bound tells which.
  chi2 = _objective(dataset, target, err, base, n)
  flat = {}
  for a in at_edge:
    unit = np.array([np.log(fitted[b] / base[b]) for b in AXES])
    unit[AXES.index(a)] = np.log(BOX[a][0] if fitted[a] / base[a] < 1.0 else BOX[a][1])
    pushed = chi2(unit)
    flat[a] = bool(pushed < 1e5 and abs(pushed - best) <= 0.01 * max(best, 1e-12))
  return dataset, {
    "at_edge": at_edge, "edge_is_flat": flat,
    "n_bounces": n, "events": obs["events"], "chi2_at_trace_fit": at_fit,
    "chi2_best": best, "fitted": fitted, "base": base,
    "measured": target.tolist(), "model_at_trace_fit": start.tolist(),
    "model_at_best": (at_best if at_best is not None else np.full(n, np.nan)).tolist(),
    "control_chi2": sham_best, "control_recovered": sham_x,
    "control_truth": {a: float(sham_values[a]) for a in AXES}}


def main():
  print("  Can ANY fixed qSLS reproduce the afterbounce ratio sequence? The box spans four")
  print("  decades in lambda1, so a failure is not a failure of the bounds.\n")

  with records.pool(len(records.DATASETS)) as pool:
    got = dict(pool.map(one, list(records.DATASETS)))

  summary = {}
  for dataset in records.DATASETS:
    v = got[dataset]
    if v is None:
      print(f"  {dataset}: not enough resolved bounces"); continue
    summary[dataset] = v
    n = v["n_bounces"]
    print(f"  ==== {dataset}, {v['events']} events, {n} bounces ====")
    print(f"  {'bounce':>14s} " + " ".join(f"{k+1:>8d}" for k in range(n)))
    print(f"  {'measured':>14s} " + " ".join(f"{x:8.3f}" for x in v["measured"]))
    print(f"  {'at trace fit':>14s} " + " ".join(f"{x:8.3f}" for x in v["model_at_trace_fit"]))
    print(f"  {'best possible':>14s} " + " ".join(f"{x:8.3f}" for x in v["model_at_best"]))
    print(f"  chi2 per bounce: {v['chi2_at_trace_fit']:.2f} at the trace fit, "
          f"{v['chi2_best']:.2f} at best")
    moves = ", ".join(f"{a} x{v['fitted'][a]/v['base'][a]:.3g}" for a in AXES)
    print(f"  best material moves the trace fit by: {moves}")
    for a in v["at_edge"]:
      if v["edge_is_flat"].get(a):
        print(f"  on the {a} wall, but chi2 is FLAT there: the observable cannot see {a} at "
              f"this scale, so the chi2 is converged and only {a} is unidentified")
      else:
        print(f"  ON A {a.upper()} WALL and still descending: not an optimum, chi2 is an "
              f"upper bound only")
    if v["control_chi2"] is not None:
      rec = v["control_recovered"]["lambda1"] / v["control_truth"]["lambda1"]
      print(f"  CONTROL (model-generated target, lambda1 x{DISPLACED:.0f}): "
            f"chi2 {v['control_chi2']:.4f}, lambda1 recovered to x{rec:.3f}")
    print()

  print("  ---- what it says ----\n")
  ok = [d for d, v in summary.items() if v["control_chi2"] is not None and v["control_chi2"] < 1.0]
  print(f"  the fitter reproduces a model-generated sequence on {len(ok)} of {len(summary)} "
        f"datasets, so a large chi2 below is the data's and not the optimiser's")
  for dataset, v in summary.items():
    truncated = [a for a in v["at_edge"] if not v["edge_is_flat"].get(a)]
    verdict = ("DISCARDED (still descending at a wall)" if truncated
               else "REACHABLE" if v["chi2_best"] < 1.0 else "NOT reachable")
    print(f"  {dataset:>13s}: best achievable chi2 per bounce {v['chi2_best']:6.2f}  -> {verdict}")
  print("\n  A fixed material that cannot reach the sequence at any point in a four-decade box")
  print("  is not a mis-fitted relaxation time. It is the wrong constitutive form, and the")
  print("  afterbounces are where it shows because that is where the rates differ.")
  json.dump(summary, open(records.HERE / "shape_error.json", "w"), indent=1)


if __name__ == "__main__":
  main()
