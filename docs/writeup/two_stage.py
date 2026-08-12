r"""Spend the budget twice, and price the option to change your mind halfway.

\Cref{sec:batch} commits all twelve runs before any of them is taken. That is the right thing to
do when the only uncertainty is in the parameters, and the wrong thing here: \cref{sec:screening}
shows that WHICH rivals are live is itself undetermined, so a one-shot batch spends its whole
budget on a rival list it may have wrong.

The machinery to price the alternative is already derived. \Cref{eq:lambdalaw} gives the
distribution of the log Bayes factor before anyone books the laser,

    Lambda_m ~ N( S_m / 2 , S_m ) ,   S_m = N s_m ,

so the probability that a stage of $N_1$ runs decides rival $m$ is
$\Phi\big((N_1 s_m/2 - \delta)/\sqrt{N_1 s_m}\big)$ --- computable, not simulated from data
nobody has. What a second stage buys is the chance to move the runs it would have spent on a
rival stage one already settled onto one it did not.

WHAT IS MEASURED. Both arms spend the same total. The one-shot arm certifies one batch against
the prior rival weights and runs it. The two-stage arm certifies a batch for $N_1$, draws
$\Lambda$ from \eqref{eq:lambdalaw}, re-screens at the same five-nat threshold, re-certifies for
the survivors, and runs the rest. The score is how many rivals are decided at the end, averaged
over the stage-one draw, and the split $N_1/N$ is swept rather than chosen.

TWO WAYS THIS COULD COME OUT NEGATIVE, and both are worth knowing. A second stage cannot help
if every rival needs the whole budget --- there is nothing to reallocate. And it costs something
real: the first stage is optimised for a rival list that includes the ones it is about to
decide, so its own runs are less concentrated than the one-shot batch's. If the sweep shows no
split beating $N_1 = N$, one-shot was right and \cref{sec:batch} needs no change.
"""

import json

import numpy as np

import records

DECISIVE = 5.0
# swept: at twelve runs the one-shot batch already decides all five rivals, so the
# comparison has no room in it and says nothing about staging
BUDGETS = (2, 3, 4, 6, 9, 12)
DRAWS = 20_000
SEED = 0
SAMPLES = 201
# Two axes, because the first one turns out to be settled by a single bubble and a comparison
# with no room in it says nothing about staging. The operator axis carries the five rivals
# sec:batch screens; the constitutive axis carries the three nested laws of tab:fits, which
# sec:design measures at T = 0.67 against qSLS -- inside the measurement scatter, and the
# hardest question in the document.
OPERATOR_INCUMBENT = ("gilmore", "mie-gruneisen")
OPERATOR_RIVALS = (("rayleigh-plesset", None), ("keller-miksis", None),
                   ("keller-enthalpy", "tait"), ("gilmore", "tait"),
                   ("keller-enthalpy", "mie-gruneisen"))
CONSTITUTIVE_RIVALS = ("SLS", "qKV", "NHKV")


def _materials(fit):
  """The incumbent qSLS and its three nested rivals, all at the same fitted numbers.

  Not each rival's own best imitation: that is the inner minimisation sec:integral shows is
  unreliable, and projecting the difference off the incumbent's material span -- which is what
  happens below -- is the linearised form sec:batch already uses in its place.
  """
  import pyimr

  g, mu, lam, alpha = fit
  return pyimr.QuadraticZener(g, mu, lam, 0.0, alpha), {
    "SLS": pyimr.Zener(g, mu, lam, 0.0),
    "qKV": pyimr.QuadraticKelvinVoigt(g, mu, alpha),
    "NHKV": pyimr.NeoHookeanKelvinVoigt(g, mu),
  }


def one(job):
  """Per-run separations `s_m` for every rival at one geometry, after the material is fitted out."""
  import pyimr
  from design_operator import FIT
  from noise_design import profile
  from pyimr.noise import characteristic_time

  radius, stretch, axis = job
  times = np.linspace(0.0, 5.0 * characteristic_time(radius), SAMPLES)
  incumbent, rivals = _materials(FIT)

  def solve(material, operator):
    config = pyimr.SimulationConfig(radius, radius / stretch, material, dynamics=operator[0],
                                    liquid_eos=operator[1], rtol=1e-7, atol=1e-9,
                                    max_steps=200_000)
    return np.asarray(pyimr.simulate(times, config).radius_ratio, dtype=float)

  operator = OPERATOR_INCUMBENT if axis == "operator" else ("keller-miksis", None)
  try:
    base = solve(incumbent, operator)
    if axis == "operator":
      gaps = [solve(incumbent, rival) - base for rival in OPERATOR_RIVALS]
    else:
      gaps = [solve(rivals[name], operator) - base for name in CONSTITUTIVE_RIVALS]
  except Exception:                                                          # noqa: BLE001
    return job, None
  if not np.all(np.isfinite(base)) or not all(np.all(np.isfinite(g)) for g in gaps):
    return job, None

  step = 1e-4
  columns = []
  for k in range(4):
    moved = []
    for direction in (1.0 + step, 1.0 - step):
      scale = np.ones(4)
      scale[k] = direction
      try:
        moved.append(solve(pyimr.QuadraticZener(*(np.array(FIT)[:3] * scale[:3]), 0.0,
                                                FIT[3] * scale[3]), operator))
      except Exception:                                                      # noqa: BLE001
        return job, None
    columns.append((moved[0] - moved[1]) / (2.0 * step))
  jacobian = np.column_stack(columns)
  if not np.all(np.isfinite(jacobian)): return job, None

  tau, spread = profile("gelatin_15C")
  phase = (times - times[0]) / characteristic_time(radius)
  scale = np.interp(phase, tau, spread)
  whitened = jacobian / scale[:, None]
  gram = whitened.T @ whitened
  # s_m is the squared length of the model difference ORTHOGONAL to the material span; that is
  # exactly sec:gain's `d`, and it is what eq:runs and eq:lambdalaw both take
  separations = []
  for gap in gaps:
    column = gap / scale
    cross = whitened.T @ column
    separations.append(max(float(column @ column - cross @ np.linalg.solve(gram, cross)), 0.0))
  return job, np.array(separations)


def _batch(matrix, weights, runs):
  """`s_m` per rival for a batch of `runs` allocated over geometries at rival weights `w`.

  The batch maximises the weighted total separation, which is linear in the measure, so its
  optimum is a single geometry -- the one with the largest `w . s(x)`. That is the honest
  statement of this criterion rather than a shortcut: linearity is why sec:gain has to impose
  testability as a constraint instead of getting spread for free.
  """
  scores = matrix @ weights
  best = int(np.argmax(scores))
  return runs * matrix[best], best


def _live(axis, every):
  """The rivals the records have NOT decided, at the deflated margins of sec:screening.

  Deflated rather than face value because sec:screening shows which reading is right is itself
  undetermined, and screen_maximin.py measures that designing against the deflated list is the
  maximin choice: it covers every live question under either truth.
  """
  from screen import _constitutive, _operators
  from screen_maximin import INFLATION, weights

  if axis == "operator":
    evidence = _operators("gelatin_15C")
    weighted = weights(evidence, INFLATION)
    return {r for r in every if (r[0] if r[1] is None else f"{r[0]}/{r[1]}") in weighted}
  evidence = _constitutive("gelatin_15C")
  weighted = weights(evidence, INFLATION)
  # exact names, not substrings: "sls" is a substring of "qsls", so a substring test called the
  # incumbent its own live rival and put a settled question back on the list
  return {r for r in every if r in weighted}


def main():
  from design_operator import PERFORMED, RADII, STRETCH

  geometries = [(float(r), float(s)) for r in RADII for s in STRETCH] + list(PERFORMED.values())
  summary = {}
  for axis, every in (("operator", OPERATOR_RIVALS), ("constitutive", CONSTITUTIVE_RIVALS)):
    # Step 0 of sec:batch, and skipping it is what made the first run of this script
    # meaningless: with Rayleigh-Plesset in the list the separations span eleven decades and
    # every budget settles everything, because compressibility is not in doubt (+74 to +182
    # nats on every record). Screening removes the rivals the records already decided, which
    # is the regime the linearised criterion is valid in anyway.
    live = _live(axis, every)
    rivals = [r for r in every if r in live]
    if not rivals:
      print(f"\n=== {axis}: no live rivals ===")
      continue
    keep = [i for i, r in enumerate(every) if r in live]
    jobs = [(r, s, axis) for r, s in geometries]
    print(f"\n=== {axis}: separations at {len(jobs)} geometries ===", flush=True)
    with records.pool(len(jobs)) as pool:
      got = list(pool.map(one, jobs))
    usable = [(d, v) for d, v in got if v is not None]
    matrix = np.array([v for _, v in usable])[:, keep]
    labels = [(d[0], d[1]) for d, _ in usable]
    print(f"  {len(usable)} of {len(jobs)} integrate; "
          f"per-run separation ranges {matrix.min():.4g} to {matrix.max():.4g}")

    prior = np.full(len(rivals), 1.0 / len(rivals))
    rng = np.random.default_rng(SEED)

    def one_shot(runs, matrix=matrix, prior=prior, rng=rng, rivals=rivals):
      full, where = _batch(matrix, prior, runs)
      draws = rng.normal(full[None, :] / 2.0, np.sqrt(np.maximum(full, 1e-300))[None, :],
                         size=(DRAWS, len(rivals)))
      return float(np.mean(np.sum(draws > DECISIVE, axis=1))), where

    def two_stage(runs, first, matrix=matrix, prior=prior, rng=rng, rivals=rivals):
      stage_one, _ = _batch(matrix, prior, first)
      draws = rng.normal(stage_one[None, :] / 2.0,
                         np.sqrt(np.maximum(stage_one, 1e-300))[None, :],
                         size=(DRAWS, len(rivals)))
      settled = draws > DECISIVE
      total, moved = 0.0, {}
      for row, done in zip(draws, settled, strict=True):
        live = ~done
        if not live.any():
          total += len(rivals)
          continue
        more, where = _batch(matrix, live / live.sum(), runs - first)
        moved[where] = moved.get(where, 0) + 1
        carried = row + rng.normal(more / 2.0, np.sqrt(np.maximum(more, 1e-300)))
        total += float(np.sum(done | (carried > DECISIVE)))
      return total / DRAWS, moved

    print(f"\n  {'budget':>7s} {'one-shot':>9s} {'best two-stage':>15s} {'at N1':>6s} "
          f"{'gain':>8s} {'stage-2 geometry':>24s}")
    table = {}
    for runs in BUDGETS:
      base, where = one_shot(runs)
      best = None
      for first in range(1, runs):
        value, moved = two_stage(runs, first)
        if best is None or value > best[1]: best = (first, value, moved)
      first, value, moved = best
      common = max(moved, key=moved.get) if moved else where
      table[runs] = {"one_shot": base, "best": value, "at": first, "gain": value - base,
                     "rivals": len(rivals), "one_shot_geometry": list(labels[where]),
                     "stage2_geometry": list(labels[common])}
      print(f"  {runs:7d} {base:9.3f} {value:15.3f} {first:6d} {value - base:+8.3f} "
            f"{labels[common][0] * 1e6:11.0f} um at {labels[common][1]:5.2f}")

    gains = [table[r]["gain"] for r in BUDGETS]
    saturated = all(table[r]["one_shot"] >= len(rivals) - 1e-9 for r in BUDGETS)
    print(f"  best gain anywhere: {max(gains):+.3f} of {len(rivals)} rivals"
          + ("   (one-shot already settles every rival at every budget)" if saturated else ""))
    summary[axis] = {"budgets": {str(k): v for k, v in table.items()},
                     "saturated": bool(saturated), "best_gain": float(max(gains))}

  print("\n  A non-positive gain is the finding that one shot was right, and where the")
  print("  one-shot column is already at its ceiling the comparison had no room in it --")
  print("  the question is settled by the first bubble, not by how the budget is staged.")

  json.dump(summary, open(records.HERE / "two_stage.json", "w"), indent=1)


if __name__ == "__main__":
  main()
