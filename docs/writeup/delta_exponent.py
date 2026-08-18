r"""Is it surface tension, or is it anything that peaks at the collapse?

`delta_dictionary.py` finds $2\sigma/R$ the only term clearing its own floor on all eight
records, at $41$ to \SI{70}{\percent}. Read directly that names the missing term as interfacial,
and it would agree with three other measurements: the screen's one surviving reachable candidate
is the initial radius, every record prefers an equilibrium radius it is not carried at, and the
sensitivity sweep makes $R_{\rm eq}$ the input the ratio is most sensitive to. Those are the same
physics, since $R_{\rm eq}$ is fixed by $p_{\rm gas} = p_\infty + 2\sigma/R_{\rm eq} - p_v$.

THE ALTERNATIVE EXPLANATION IS GEOMETRIC AND HAS TO BE KILLED FIRST. $1/R$ diverges at the
collapse and $\hat\delta$ puts $65$ to \SI{69}{\percent} of itself there, so the two agree
wherever both are large. The phase-randomised null preserves each curve's spectrum but not its
LOCALISATION, so any function spiked at the same instant clears it for free.

TWO TESTS SEPARATE THEM. If the term is really $2\sigma/R$ the exponent is resolved: $R^{-1}$
should beat $R^{-1/2}$, $R^{-2}$, $R^{-3}$ and $R^{-4}$, which are equally spiked and differ only
in how fast. And the null is rebuilt by CIRCULARLY SHIFTING $\hat\delta$ against the dictionary
term, which preserves both curves entirely and destroys only their alignment in time, so a match
that is merely "both peak somewhere" does not survive it.
"""

import json

import numpy as np

import records
from shape_error import seed_for
from universal_delta import GRID, one as delta_one

DRAWS = 4000
ORDER = (*records.DATASETS, *records.PAAM)
# The first sweep started at 0.5 and every record's argmin landed there: a bracket edge, not an
# optimum. The share must vanish as p -> 0, since (Rmax/R)^0 is a constant and the mean is
# removed, so a maximum exists strictly inside (0, 0.5) and the bracket has to contain it.
POWERS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.0)


def one(dataset):
  import pyimr

  _, payload = delta_one(dataset)
  curve = np.array(payload["curve"])
  basis, _ = np.linalg.qr(np.array(payload["span"]).T)

  times, _, _, maximum, stretch = records.load(dataset)
  fitted = json.load(open(records.HERE / "paam_lackoffit.json"))[dataset]["fitted"]
  material = pyimr.QuadraticZener(fitted["g"], fitted["mu"], fitted["lambda1"], 0.0,
                                  fitted["alpha"])
  config = pyimr.SimulationConfig(maximum, maximum / stretch, material,
                                  dynamics="keller-miksis", rtol=1e-10, atol=1e-12,
                                  max_steps=4_000_000)
  radius = np.asarray(pyimr.simulate(times, config).radius_m, dtype=float)
  tau = times / (maximum * np.sqrt(1064.0 / 101325.0))

  scores, rng = {}, np.random.default_rng(seed_for(dataset + "exp"))
  for power in POWERS:
    series = (maximum / np.maximum(radius, 1e-12)) ** power
    got = np.interp(GRID, tau, series)
    got = got - got.mean()
    got = got - basis @ (basis.T @ got)
    norm = np.linalg.norm(got)
    if norm < 1e-12: continue
    got = got / norm
    share = float(np.dot(curve, got) ** 2)
    # a null that keeps BOTH curves intact and destroys only their alignment in time
    null = np.array([float(np.dot(np.roll(curve, int(s)), got) ** 2)
                     for s in rng.integers(1, curve.size - 1, DRAWS)])
    scores[power] = {"share": share, "shift_null_95": float(np.percentile(null, 95)),
                     "p": float((null >= share).mean())}
  return dataset, scores


def main():
  with records.pool(len(ORDER)) as pool:
    got = dict(pool.map(one, list(ORDER)))

  print("  share of delta-hat explained by (Rmax/R)^p, and a null that shifts delta-hat in time\n")
  print(f"  {'record':>16s} " + " ".join(f"{'p=' + str(p):>10s}" for p in POWERS)
        + f" {'best p':>8s}")
  summary = {}
  for dataset in ORDER:
    row = got[dataset]
    best = max(row, key=lambda p: row[p]["share"])
    print(f"  {dataset:>16s} " + " ".join(
      f"{row[p]['share']:9.1%}" + ("*" if row[p]["p"] < 0.05 else " ") for p in POWERS)
      + f" {best:8.1f}")
    summary[dataset] = {str(p): row[p] for p in row} | {"best_power": best}
  print(f"  {'shift null (95%)':>16s} " + " ".join(
    f"{np.mean([got[d][p]['shift_null_95'] for d in ORDER]):9.1%} " for p in POWERS))
  print("\n  * beats a null that shifts delta-hat against the same term, at 5 percent")

  print("\n  ---- what it says ----\n")
  bests = [summary[d]["best_power"] for d in ORDER]
  survivors = {p: sum(1 for d in ORDER if got[d][p]["p"] < 0.05) for p in POWERS}
  print(f"  best exponent per record: {bests}")
  print("  records where each exponent beats the shift null: "
        + ", ".join(f"p={p}: {n}/{len(ORDER)}" for p, n in survivors.items()))
  interior = [x for x in bests if min(POWERS) < x < max(POWERS)]
  near_one = sum(1 for x in bests if 0.75 <= x <= 1.5)
  separates = len({q for q in POWERS if survivors[q] >= 6}) < len(POWERS)
  print(f"\n  argmins strictly inside the bracket: {len(interior)} of {len(bests)}")
  print(f"  argmins consistent with p = 1 (0.75 to 1.5): {near_one} of {len(bests)}")
  print(f"  does the shift null separate the exponents: {separates}")
  if len(interior) < len(bests):
    print("\n  An argmin still sits on a bracket wall, so the exponent is NOT measured and")
    print("  nothing here says which power of R is matched.")
  elif near_one >= 6 and separates:
    print("\n  The exponent is resolved near 1 and the null separates the powers, so the match")
    print("  is to 2 sigma / R specifically and the missing term is interfacial.")
  else:
    print("\n  The exponent is resolved and it is NOT 1. What delta-hat matches is a weak")
    print("  negative power of R rather than the 1/R surface tension requires, so the")
    print("  interfacial reading is refused. A real component does grow as the bubble shrinks,")
    print("  aligned in time; its power is measured here and it is not surface tension's.")
  json.dump(summary, open("delta_exponent.json", "w"), indent=1)


if __name__ == "__main__":
  main()
