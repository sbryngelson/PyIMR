"""Which missing physics could account for the discrepancy?

`discrepancy` leaves `delta_hat`, the curve no choice of parameters reproduces, and it is the
same curve on every record -- the three shapes correlate at $0.49$ to $0.84$, where unrelated
curves of this length would sit near $0.08$. So there is one target and it is measured rather
than guessed.

Each candidate enrichment supplies a direction: the trace it would add, differenced against
the current model. Only the part orthogonal to the existing sensitivities can remove any of
`delta_hat`; the rest is absorbed by refitting `mu`, `g*alpha` and `lambda_1`, which improves
chi-squared and explains nothing. `enrichment_overlap` scores that cosine, and its square is
the share of the discrepancy a candidate could remove.

Deliberately not a fit-and-compare. Fitting each candidate costs a multistart and an evidence
integral to answer "which of these wrong models fits best", which among models the lack-of-fit
test rejects is the wrong question. One solve per candidate answers "could this be the missing
physics at all", and anything near zero cannot be, whatever its Bayes factor does.

Everything is evaluated at each record's own fit in the identified coordinates, the only place
the split means anything -- see `discrepancy.py`.

This is the screen; `enrichment.py` is the fit-and-compare it is meant to precede, and the two
answer different questions about the same candidates.
"""

import json

import numpy as np

import records
from identified import BOX, candidate_at_ratio

RATIO = 38.5
STARTS, EVALUATIONS = 16, 600
# a second relaxation arm is a continuous axis, so its direction is a derivative in the share
# rather than a difference; the rest is physics that is either present or absent
SECOND_ARM = {"second_relaxation_time_s": 1e-5, "second_share": 0.02}
CANDIDATES = (
  ("thermal (bubble)", {"bubtherm": 1, "Nt": 11}),
  ("thermal (bubble+medium)", {"bubtherm": 1, "medtherm": 1, "Nt": 11, "Mt": 11}),
  ("mass transfer", {"bubtherm": 1, "Nt": 11, "masstrans": 1, "vapor": 1}),
  ("operator: gilmore/tait", {"dynamics": "gilmore", "liquid_eos": "tait"}),
  ("operator: keller-enth/MG", {"dynamics": "keller-enthalpy", "liquid_eos": "mie-gruneisen"}),
)


def directions(dataset):
  """`(delta_hat, jacobian, {name: direction})` for one record, all whitened alike."""
  import pyimr
  from pyimr.noise import discrepancy
  from pyimr.selection import fit_candidate, physical_from_unit

  times, mean, spread, maximum, stretch = records.load(dataset)
  trials = records.trial_count(dataset)
  error = spread / np.sqrt(trials)
  candidate = candidate_at_ratio(RATIO)
  solve = records.solver(times, maximum, stretch)
  fit = fit_candidate(candidate, solve, mean, spread, bounds=BOX, starts=STARTS,
                      max_evaluations=EVALUATIONS)
  axes = tuple(candidate.axes)
  fitted = dict(zip(axes, (float(v) for v in physical_from_unit(axes, fit.unit, BOX)),
                    strict=True))
  point = np.array([fitted[k] for k in axes])

  def trace(scale=None, material=None, equilibrium=None, **options):
    moved = dict(zip(axes, (float(v) for v in point * (np.ones(len(axes)) if scale is None
                                                       else scale)), strict=True))
    config = pyimr.SimulationConfig(
      maximum, maximum / stretch if equilibrium is None else equilibrium,
      candidate.build(moved) if material is None else material,
      dynamics=options.pop("dynamics", "keller-miksis"),
      liquid_eos=options.pop("liquid_eos", None),
      rtol=1e-9, atol=1e-11, max_steps=600_000, **options)
    return np.asarray(pyimr.simulate(times, config).radius_ratio, dtype=float)

  base = trace()
  step = 1e-4
  jacobian = np.column_stack([
    (trace(np.where(np.arange(len(axes)) == k, 1 + step, 1.0)) -
     trace(np.where(np.arange(len(axes)) == k, 1 - step, 1.0))) / (2 * step * error)
    for k in range(len(axes))])
  split = discrepancy((mean - base) / error, jacobian)

  arm = pyimr.TwoModeQuadraticZener(
    float(np.sqrt(fitted["galpha"] * RATIO)), fitted["mu"], fitted["lambda1"], 0.0,
    float(np.sqrt(fitted["galpha"] / RATIO)), **SECOND_ARM)
  jobs = [*CANDIDATES, ("second relaxation arm", {"material": arm}),
          # the initial radius is configuration rather than new physics, included because
          # sec:reqprior shows it moves the trace by as much as the operator does
          ("initial radius", {"equilibrium": maximum / stretch * (1 + step)})]

  found, dropped = {}, {}
  for name, kwargs in jobs:
    # a candidate that will not integrate must be REPORTED, not skipped: silently omitting it
    # makes the table read as complete when the law was never tested at all, which is how
    # gent, fung and ogden went missing from the constitutive screen
    try:
      other = trace(**kwargs)
    except Exception as failure:                                 # noqa: BLE001
      dropped[name] = type(failure).__name__
      continue
    if not np.all(np.isfinite(other)):
      dropped[name] = "non-finite trace"
      continue
    scale = (SECOND_ARM["second_share"] if name == "second relaxation arm"
             else step if name == "initial radius" else 1.0)
    found[name] = (other - base) / (error * scale)
  if dropped:
    print(f"    {dataset}: not screened -- "
          + ", ".join(f"{k} ({v})" for k, v in dropped.items()), flush=True)
  return dataset, split, jacobian, found


def main():
  from pyimr.noise import enrichment_overlap

  with records.pool(len(records.DATASETS)) as pool:
    got = list(pool.map(directions, list(records.DATASETS)))
  screens = {d: enrichment_overlap(s.identifiable, j, c) for d, s, j, c in got}
  # the same joint read as a SUBSPACE, which is what scoring |cos| amounts to. Reported so the
  # cost of the sign convention is a measured number rather than an assertion about it.
  spans = {d: enrichment_overlap(s.identifiable, j, c, one_sided=False)
           for d, s, j, c in got}
  names = sorted({k for _, _, _, c in got for k in c})

  print("\n  share of the discrepancy each candidate could remove, at its own fit\n")
  print(f"  {'candidate':28s} " + " ".join(f"{d[-3:]:>9s}" for d, *_ in got))
  for name in names:
    cells = []
    for dataset, *_ in got:
      value = screens[dataset].removable.get(name)
      if value is None: cells.append("        -")
      else: cells.append(f"{value:8.1%}" + ("" if screens[dataset].reachable[name] else "*"))
    print(f"  {name:28s} " + " ".join(cells))
  print(f"  {'ALL TOGETHER (cone)':28s} " + " ".join(f"{screens[d].joint:9.1%}" for d, *_ in got))
  print(f"  {'ALL TOGETHER (span)':28s} " + " ".join(f"{spans[d].joint:9.1%}" for d, *_ in got))
  print("\n  * anti-aligned: the amplitude is one-sided, so no admissible size of it removes")
  print("    anything -- the magnitude is what it would remove if the sign could be flipped")

  print("\n  a candidate near zero cannot be the missing physics, whatever its evidence does")
  for dataset, split, _, _ in got:
    name, value = screens[dataset].best
    print(f"    {dataset:13s} |delta| {split.size:6.1f}   best single: {name} at {value:.1%}")

  # `reachable` MUST be persisted beside `removable`. The magnitude alone is what made the
  # 23 C second arm read as the best candidate on any record at 58%, when its cosine is
  # -0.762 and no admissible amplitude of it reduces anything (#242).
  json.dump({d: {"removable": screens[d].removable, "reachable": screens[d].reachable,
                 "overlap": screens[d].overlap, "joint": screens[d].joint,
                 "joint_span": spans[d].joint,
                 "identifiable": s.size} for d, s, _, _ in got},
            open(records.HERE / "enrichment_screen.json", "w"), indent=1)


if __name__ == "__main__":
  main()
