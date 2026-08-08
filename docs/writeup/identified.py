"""The operator comparison, asked in coordinates the record can answer.

The likelihood depends on the product `g*alpha` alone -- derived in the writeup, confirmed by
an SVD at the fit whose sloppiest direction is `g^+1 alpha^-1`. Fitting `g` and `alpha`
separately therefore slides along a ridge until it hits the prior box, and the operator
ranking follows the box: moving the bounds swung it by 50 nats and reversed the winner.

So the unidentified ratio is fixed at a stated value and the identified product is fitted.
The point is not the ranking at one ratio but whether it SURVIVES the ratio, and whether it
replicates across records. Both filters are applied.
"""

import json

import numpy as np

import records

RATIOS = (3.85, 38.5, 385.0)            # the published fit has g/alpha = 38.5; two decades around it
BOX = {"mu": (1e-5, 1e1), "galpha": (1e0, 1e7), "lambda1": (1e-9, 1e-2)}
# Below this, two operators are not ordered by a record -- they are indistinguishable. Asking
# only "did the sign hold" lets a record with no power veto one that has some: at 33 C all six
# operators land within 0.5 nats, so every ordering there is noise.
DECISIVE = 1.0


def _candidate(ratio):
  import pyimr
  from pyimr.selection import CandidateModel

  def build(t):
    product = t["galpha"]
    return pyimr.QuadraticZener(float(np.sqrt(product * ratio)), t["mu"], t["lambda1"], 0.0,
                                float(np.sqrt(product / ratio)))

  return CandidateModel(f"qSLS|g/alpha={ratio:g}", build, ("mu", "galpha", "lambda1"))


def _job(argument):
  dataset, name, ratio = argument
  times, mean, spread, maximum, stretch = records.load(dataset)
  solve = records.solver(times, maximum, stretch, dynamics=name[0], liquid_eos=name[1])
  try:
    # starts=10, not the default 6. At 6 the 33 C fits sat at chi2/N = 1.151 while a fit at
    # 0.439 exists, so all six operators were stuck in the same poor basin -- which looks
    # exactly like six operators that cannot be told apart, and was read that way.
    got = records.score(_candidate(ratio), solve, mean, spread, bounds=BOX, starts=10,
                        evaluations=260, trials=records.trial_count(dataset))
  except ValueError as error:
    got = {"failed": str(error)}
  return (dataset, name, ratio), got


def _decided(table, dataset, names):
  """Orderings this record settles: same sign at every ratio, by more than `DECISIVE`."""
  holds = {}
  for i, a in enumerate(names):
    for b in names[i + 1:]:
      try:
        gaps = [table[(dataset, a, r)]["log_evidence"] - table[(dataset, b, r)]["log_evidence"]
                for r in RATIOS]
      except KeyError:
        continue
      if not all(np.isfinite(gaps)): continue
      if all(g > DECISIVE for g in gaps): holds[(a, b)] = min(gaps)
      elif all(g < -DECISIVE for g in gaps): holds[(b, a)] = min(-g for g in gaps)
  return holds


def main():
  from pyimr.selection import DYNAMICS_MODELS

  names = list(DYNAMICS_MODELS)
  jobs = [(d, n, r) for d in records.DATASETS for r in RATIOS for n in names]
  with records.pool(len(jobs)) as pool:
    table = dict(pool.map(_job, jobs))
  decisive = {d: _decided(table, d, names) for d in records.DATASETS}

  print("the operator question in identified coordinates, on every record\n")
  for dataset in records.DATASETS:
    chi = [table[(dataset, n, RATIOS[0])].get("chi2_per_n") for n in names]
    best = min((c for c in chi if c is not None), default=float("nan"))
    print(f"  {dataset}: decides {len(decisive[dataset]):2d} of 15 orderings, best chi2/N {best:.2f}")

  # supported = decided somewhere, contradicted nowhere. Records without power abstain.
  supported = {k for d in records.DATASETS for k in decisive[d]
               if not any((k[1], k[0]) in decisive[e] for e in records.DATASETS)}
  print(f"\n  supported overall (decided somewhere, contradicted nowhere): {len(supported)}")
  for a, b in sorted(supported, key=lambda ab: -max(decisive[d].get(ab, 0) for d in records.DATASETS)):
    votes = {d: decisive[d][(a, b)] for d in records.DATASETS if (a, b) in decisive[d]}
    quiet = [d for d in records.DATASETS if (a, b) not in decisive[d]]
    print(f"    {a:>22} > {b:<22} "
          + ", ".join(f"{d.replace('gelatin_', '')}:+{v:.1f}" for d, v in votes.items())
          + (f"   (no power: {', '.join(x.replace('gelatin_', '') for x in quiet)})" if quiet else ""))

  # "supported by one record" and "replicated" are different claims, and telling them apart
  # is the whole reason for running three records rather than the one with the best margin
  print("\n  against the `keller-miksis` pressure form every candidate in this package assumes:")
  for challenger in ("keller-miksis-tait", "keller-miksis-mie"):
    votes = [d for d in records.DATASETS if (challenger, "keller-miksis") in decisive[d]]
    print(f"    {challenger:>22}: " + ("REPLICATED on %d records" % len(votes) if len(votes) > 1
                                       else "supported by one record, untested by the others" if votes
                                       else "not decided anywhere"))
  records.HERE.joinpath("identified.json").write_text(
    json.dumps({f"{d}|{n}|{r}": v for (d, n, r), v in table.items()}, indent=1))


if __name__ == "__main__":
  main()
