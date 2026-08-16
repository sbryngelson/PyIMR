r"""If no fixed material reaches the whole sequence, which part of it is the material failing?

`shape_error.py` asks whether a fixed qSLS can reproduce the afterbounce ratio sequence at
any point in a four-decade box. This asks the follow-up that any answer there leaves open:
WHERE along the sequence the failure lives.

THE TWO HYPOTHESES MAKE OPPOSITE PREDICTIONS ABOUT SUBSETS. If the constitutive form is
simply wrong, the model should fail on every subset, because a wrong form is wrong at every
rate. If instead the material CHANGED during the first collapse -- helices unzipping
irreversibly, so the network that rings down is not the one that was compressed -- then the
bounces after the first should be described perfectly well by SOME fixed material, just not
the same one the first collapse wants. Damage is a statement about a boundary in time, and a
boundary shows up as a subset that fits when the whole does not.

SO EACH SUBSET IS FITTED INDEPENDENTLY. The full sequence, the sequence with the first
afterbounce dropped, and the first two alone. Reachability of the tail with unreachability of
the whole localises a change to the first collapse. Unreachability everywhere says the form
is wrong and no boundary will rescue it. Both fitting well while the union does not is the
signature of two different materials rather than one bad one.

A SUBSET IS EASIER TO FIT FOR A REASON THAT IS NOT PHYSICS, AND IT IS CONTROLLED HERE. Three
free parameters against three bounces will fit almost anything, so the comparison is not
between raw $\chi^2$ values at different lengths. Each subset is reported with its degrees of
freedom, and the tail is additionally scored on a HELD-OUT bounce it was not fitted to, which
is the only version of the question that a shrinking sample cannot answer by itself.
"""

import json

import numpy as np

import records
from frequency_space import KEEP, measured
from shape_error import AXES, BOX, _fit, _objective, _ratios

SUBSETS = (("full", None), ("tail (drop bounce 1)", 1), ("head (first two)", -2))


def _slice(n, spec):
  """Index array for a subset spec: `None` all, positive drop from the front, negative keep."""
  if spec is None: return np.arange(n)
  return np.arange(spec, n) if spec > 0 else np.arange(0, -spec)


def one(dataset):
  obs = measured(dataset)
  n = min(KEEP, len(obs["ratio"]))
  target = np.array(obs["ratio"][:n], dtype=float)
  err = np.array(obs["ratio_spread"][:n], dtype=float) / np.sqrt(obs["events"])
  base = json.load(open(records.HERE / "per_trial_fits.json"))[dataset]["median"]
  base = {a: float(base[a]) for a in AXES}

  out = {"n_bounces": n, "events": obs["events"], "base": base, "subsets": {}}
  for label, spec in SUBSETS:
    idx = _slice(n, spec)
    if len(idx) < 2: continue

    def chi2(unit, idx=idx):
      full = _objective(dataset, target, err, base, n)
      return full(unit) if len(idx) == n else _subset_chi2(dataset, base, unit, target, err, idx, n)

    best, best_x = _fit(chi2, seed=abs(hash(dataset + label)) % 2**31)
    fitted = {a: float(base[a] * np.exp(best_x[k])) for k, a in enumerate(AXES)}
    got = _ratios(dataset, fitted, n)
    held = [k for k in range(n) if k not in set(idx.tolist())]
    heldout = (float(np.mean(((got[held] - target[held]) / err[held]) ** 2))
               if got is not None and held else None)
    out["subsets"][label] = {
      "bounces": [int(k) + 1 for k in idx], "chi2": best, "dof": max(len(idx) - 3, 0),
      "fitted": fitted, "moves": {a: fitted[a] / base[a] for a in AXES},
      "heldout_bounces": [k + 1 for k in held], "heldout_chi2": heldout}
  return dataset, out


def _subset_chi2(dataset, base, unit, target, err, idx, n):
  values = {a: float(base[a] * np.exp(unit[k])) for k, a in enumerate(AXES)}
  for a in AXES:                                   # the same box the full objective enforces
    lo, hi = BOX[a]
    if not (lo * base[a] <= values[a] <= hi * base[a]): return 1e6
  got = _ratios(dataset, values, n)
  if got is None: return 1e6
  return float(np.mean(((got[idx] - target[idx]) / err[idx]) ** 2))


def main():
  print("  A wrong constitutive form fails on every subset. A material that CHANGED at the")
  print("  first collapse leaves a tail that some fixed material describes perfectly well.\n")

  with records.pool(len(records.DATASETS)) as pool:
    got = dict(pool.map(one, list(records.DATASETS)))

  summary = {}
  for dataset in records.DATASETS:
    v = got[dataset]
    if v is None or not v["subsets"]:
      print(f"  {dataset}: not enough resolved bounces"); continue
    summary[dataset] = v
    print(f"  ==== {dataset}, {v['events']} events ====")
    print(f"  {'subset':>22s} {'bounces':>12s} {'chi2':>8s} {'dof':>5s} {'held-out chi2':>14s}")
    for label, s in v["subsets"].items():
      held = "--" if s["heldout_chi2"] is None else f"{s['heldout_chi2']:.2f}"
      bounces = ",".join(str(b) for b in s["bounces"])
      print(f"  {label:>22s} {bounces:>12s} {s['chi2']:8.2f} {s['dof']:5d} {held:>14s}")
    for label, s in v["subsets"].items():
      moves = ", ".join(f"{a} x{s['moves'][a]:.2f}" for a in AXES)
      print(f"  {label:>22s} wants: {moves}")
    print()

  print("  ---- what it says ----\n")
  for dataset, v in summary.items():
    full = v["subsets"].get("full", {}).get("chi2")
    tail = v["subsets"].get("tail (drop bounce 1)", {})
    if full is None or not tail: continue
    verdict = ("a boundary at the first collapse" if tail["chi2"] < 1.0 <= full
               else "the form is wrong everywhere" if tail["chi2"] >= 1.0
               else "no failure to localise")
    print(f"  {dataset:>13s}: full {full:6.2f}, tail {tail['chi2']:6.2f}  -> {verdict}")
  print("\n  A tail that fits while the whole does not means one material cannot span the first")
  print("  collapse, which is a statement about damage rather than about a relaxation time.")
  print("  The held-out column is what keeps that from being three parameters and three points.")
  json.dump(summary, open(records.HERE / "shape_localise.json", "w"), indent=1)


if __name__ == "__main__":
  main()
