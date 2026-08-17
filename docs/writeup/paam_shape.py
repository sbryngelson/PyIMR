r"""Can a fixed qSLS reach the PAAm afterbounce sequence, where it cannot reach gelatin's?

`shape_error.py` fits the material directly to the ratio sequence over a box six decades wide
and stops at $\chi^2$ per bounce of $6.52$ and $5.33$ on gelatin, against $0.0000$ for the same
fitter run on a sequence the model itself produced. No fixed material in this form reaches
those afterbounces.

`paam_blocks.py` shows the PAAm sequences are MONOTONE, once the pooled temperature sweep is
taken apart, which is necessary for the spherical form and not sufficient for it: a monotone
sequence can still be one no choice of $(\mu, g\alpha, \lambda_1)$ produces. This runs the same
fit and the same control on the PAAm records and reads the achievable $\chi^2$.

The base material each search starts from is that record's own qSLS trace fit from
`paam_lackoffit.py`, which is the PAAm analogue of the per-trial median gelatin uses. The
search spans the same six decades either way, so where it lands is not a property of where it
started.
"""

import json

import numpy as np

import records
from frequency_space import KEEP, SPAN, _families
from bounce_sweep import sequence
from shape_error import AXES, DISPLACED, EDGE, MAX_STEPS_REPORT, _fit, _objective, _ratios


def measured(dataset):
  """The PAAm analogue of `frequency_space.measured`, off the record's own event matrix."""
  times, events = records.trials(dataset)
  phase = (times - times[0]) / (times[-1] - times[0]) * SPAN
  got = []
  for row in events:
    s = sequence(row, phase)
    if s is not None: got.append(_families(s["amplitudes"], s["times"])[0])
  n = min(KEEP, min(len(g) for g in got))
  stack = np.array([g[:n] for g in got], dtype=float)
  return {"events": len(got), "ratio": stack.mean(axis=0).tolist(),
          "ratio_spread": stack.std(axis=0, ddof=1).tolist()}


def base_material(dataset):
  """`(mu, galpha, lambda1)` from the record's qSLS trace fit. `galpha` is the product."""
  fitted = json.load(open(records.HERE / "paam_lackoffit.json"))[dataset]["fitted"]
  return {"mu": float(fitted["mu"]), "galpha": float(fitted["g"] * fitted["alpha"]),
          "lambda1": float(fitted["lambda1"])}


def one(dataset):
  obs = measured(dataset)
  n = min(KEEP, len(obs["ratio"]))
  target = np.array(obs["ratio"][:n], dtype=float)
  err = np.array(obs["ratio_spread"][:n], dtype=float) / np.sqrt(obs["events"])
  base = base_material(dataset)

  start = _ratios(dataset, base, n, MAX_STEPS_REPORT)
  if start is None: return dataset, None
  at_fit = float(np.mean(((start - target) / err) ** 2))

  best, best_x = _fit(_objective(dataset, target, err, base, n),
                      seed=abs(hash(dataset)) % 2**31)
  fitted = {a: float(base[a] * np.exp(best_x[k])) for k, a in enumerate(AXES)}

  sham_values = dict(base, lambda1=base["lambda1"] * DISPLACED)
  sham_target = _ratios(dataset, sham_values, n, MAX_STEPS_REPORT)
  sham_best = sham_x = None
  if sham_target is not None:
    sham_best, raw = _fit(_objective(dataset, sham_target, err, base, n),
                          seed=abs(hash(dataset + "sham")) % 2**31)
    sham_x = {a: float(base[a] * np.exp(raw[k])) for k, a in enumerate(AXES)}

  at_best = _ratios(dataset, fitted, n, MAX_STEPS_REPORT)
  from shape_error import BOX
  at_edge = [a for a in AXES
             if not (BOX[a][0] * EDGE < fitted[a] / base[a] < BOX[a][1] / EDGE)]
  chi2 = _objective(dataset, target, err, base, n)
  flat = {}
  for a in at_edge:
    unit = np.array([np.log(fitted[b] / base[b]) for b in AXES])
    unit[AXES.index(a)] = np.log(BOX[a][0] if fitted[a] / base[a] < 1.0 else BOX[a][1])
    pushed = chi2(unit)
    flat[a] = bool(pushed < 1e5 and abs(pushed - best) <= 0.01 * max(best, 1e-12))
  return dataset, {
    "at_edge": at_edge, "edge_is_flat": flat, "n_bounces": n, "events": obs["events"],
    "chi2_at_trace_fit": at_fit, "chi2_best": best, "fitted": fitted, "base": base,
    "measured": target.tolist(), "error": err.tolist(),
    "model_at_trace_fit": start.tolist(),
    "model_at_best": (at_best if at_best is not None else np.full(n, np.nan)).tolist(),
    "control_chi2": sham_best, "control_recovered": sham_x,
    "control_truth": {a: float(sham_values[a]) for a in AXES}}


def main():
  print("  The same six-decade box gelatin was refused by, on the PAAm records.\n")
  jobs = list(records.PAAM)
  with records.pool(len(jobs)) as pool:
    got = dict(pool.map(one, jobs))

  summary = {}
  for dataset in jobs:
    v = got[dataset]
    if v is None:
      print(f"  {dataset}: not enough resolved bounces"); continue
    summary[dataset] = v
    n = v["n_bounces"]
    print(f"  ==== {dataset}, {v['events']} events, {n} bounces ====")
    print(f"  {'bounce':>14s} " + " ".join(f"{k+1:>8d}" for k in range(n)))
    print(f"  {'measured':>14s} " + " ".join(f"{x:8.3f}" for x in v["measured"]))
    print(f"  {'+- se':>14s} " + " ".join(f"{x:8.3f}" for x in v["error"]))
    print(f"  {'at trace fit':>14s} " + " ".join(f"{x:8.3f}" for x in v["model_at_trace_fit"]))
    print(f"  {'best possible':>14s} " + " ".join(f"{x:8.3f}" for x in v["model_at_best"]))
    print(f"  chi2 per bounce: {v['chi2_at_trace_fit']:.2f} at the trace fit, "
          f"{v['chi2_best']:.2f} at best")
    for a in v["at_edge"]:
      note = ("chi2 FLAT there, so converged and only that axis unidentified"
              if v["edge_is_flat"].get(a) else "STILL DESCENDING: chi2 is an upper bound only")
      print(f"  on the {a} wall: {note}")
    if v["control_chi2"] is not None:
      rec = v["control_recovered"]["lambda1"] / v["control_truth"]["lambda1"]
      print(f"  CONTROL (model-generated target, lambda1 x{DISPLACED:.0f}): "
            f"chi2 {v['control_chi2']:.4f}, lambda1 recovered to x{rec:.3f}")
    print()

  print("  ---- what it says ----\n")
  for dataset, v in summary.items():
    truncated = [a for a in v["at_edge"] if not v["edge_is_flat"].get(a)]
    verdict = ("DISCARDED (still descending at a wall)" if truncated
               else "REACHABLE" if v["chi2_best"] < 1.0 else "NOT reachable")
    print(f"  {dataset:>14s}: best achievable chi2 per bounce {v['chi2_best']:6.2f} -> {verdict}")
  print("\n  Gelatin's best achievable was 6.52 and 5.33 with a fitter that reaches 0.0000 on a")
  print("  model-generated target. Whatever these numbers are, they are the same question asked")
  print("  the same way of a material six times better sampled.")
  json.dump(summary, open(records.HERE / "paam_shape.json", "w"), indent=1)


if __name__ == "__main__":
  main()
