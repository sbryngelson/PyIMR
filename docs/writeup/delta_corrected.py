r"""Does the shared curve survive correcting the fits the document says are misconfigured?

\Cref{sec:limitations} lists defects that were measured after the rest was written and are not
corrected in it: a far-field density that is water's rather than the medium's, vapour pressure
switched off, and an equilibrium radius pinned at a value \cref{sec:paam} shows every record
disagrees with. \Cref{sec:universal} finds $\hat\delta$ the same curve on two materials -- and
all eight fits carry all three defects, so a shared curve is what shared defects would produce
whether or not there is any model-form error at all. That is the objection this answers.

THE TWO CORRECTIONS THE DATA ITSELF ASKS FOR ARE APPLIED. Each record is refitted at the stretch
its own $\chi^2$ prefers, taken from `paam_stretch.json` rather than assumed, and with vapour and
mass transfer switched on so the driving pressure carries $p_v(T)$. The density is left alone and
said so: gelatin's is known and polyacrylamide's is not, and substituting a guess for one
material and a measurement for the other would put a new asymmetry into exactly the comparison
being tested.

WHAT EACH OUTCOME MEANS. If the corrected $\hat\delta$ is the same curve as the uncorrected one,
the defects are not what the two materials share and \cref{sec:universal} stands as written. If
it is a different curve but still shared ACROSS materials, the claim survives in corrected form
and the published curve must be the corrected one. If correcting the fits destroys the
cross-material agreement, then what was measured was the shape of three shared mistakes, which
is a real answer to what the curve is and a considerably less pleasant one.
"""

import json

import numpy as np

import records
from shape_error import seed_for
from universal_delta import GRID, _surrogate, one as delta_one

# All eight, because they run in PARALLEL and one worker per record costs the same wall time as
# one worker per pair. A thermal solve is ~35 s against ~3 s isothermal, so the budget below and
# not the record count is what sets the runtime.
ORDER = (*records.DATASETS, *records.PAAM)
DRAWS = 2000
STARTS, EVALUATIONS = 10, 300
# Switching the thermal physics on made a COLD multistart over the six-decade WIDE box
# intractable: one record ran thirty minutes without converging and others raised inside the
# search. The isothermal fit is already close, so the thermal run is given a box of one third to
# three times that answer per axis and a short search. That is a local refinement of a known
# point, not a fresh global fit, and it is the honest way to describe it.
WARM = 3.0
THERMAL_STARTS, THERMAL_EVALUATIONS = 3, 100
# `Nt = 25` is the package default and it is not negotiable here. At `Nt = 7`, chosen once for
# speed, this configuration stalls on six of eight records: `thermal_gate.py` measures 2/8
# integrating at 7 against 8/8 at both 15 and 25, and the stall is SLOWER than the success --
# 250 s of stepsize collapse against 35 s of solve -- so the coarse grid bought nothing even
# before it bought a wrong answer. The earlier reading of those stalls as the corrected model
# being un-integrable was a statement about the grid and not about the physics.
THERMAL = {"bubtherm": 1, "Nt": 25, "masstrans": 1, "vapor": 1}


def corrected(dataset):
  """`delta_hat` at the preferred stretch with vapour on, on the common grid."""
  import numpy as np
  from pyimr.noise import discrepancy
  from pyimr.selection import STANDARD_MODELS, evaluate_at, fit_candidate, physical_from_unit

  from lackoffit import WIDE

  times, mean, spread, maximum, _ = records.load(dataset)
  stretch = json.load(open(records.HERE / "paam_stretch.json"))[dataset]["argmin"]
  solve = records.solver(times, maximum, stretch, max_steps=400_000, **THERMAL)
  candidate = STANDARD_MODELS["qSLS"]
  box, starts, evaluations = WIDE, STARTS, EVALUATIONS
  if THERMAL:
    warm = json.load(open(records.HERE / "paam_lackoffit.json"))[dataset]["fitted"]
    box = {a: (warm[a] / WARM, warm[a] * WARM) for a in candidate.axes}
    starts, evaluations = THERMAL_STARTS, THERMAL_EVALUATIONS
  try:
    fit = fit_candidate(candidate, solve, mean, spread, bounds=box, starts=starts,
                        max_evaluations=evaluations)
    values = physical_from_unit(candidate.axes, fit.unit, box)
    fitted = dict(zip(candidate.axes, (float(v) for v in values), strict=True))
    base = np.asarray(evaluate_at(candidate, solve, fitted)[0], dtype=float)
  except Exception as error:                                          # noqa: BLE001
    return dataset, {"failed": f"{type(error).__name__}: {error}"}
  error_bar = spread / np.sqrt(records.trial_count(dataset))
  step, columns = 1e-4, []
  for axis in candidate.axes:
    up = dict(fitted, **{axis: fitted[axis] * (1 + step)})
    dn = dict(fitted, **{axis: fitted[axis] * (1 - step)})
    try:
      hi = np.asarray(evaluate_at(candidate, solve, up)[0], dtype=float)
      lo = np.asarray(evaluate_at(candidate, solve, dn)[0], dtype=float)
    except Exception as error:                                        # noqa: BLE001
      return dataset, {"failed": f"jacobian: {type(error).__name__}"}
    columns.append((hi - lo) / (2 * step * error_bar))
  jacobian = np.column_stack(columns)
  split = discrepancy((mean - base) / error_bar, jacobian)
  curve = np.asarray(split.identifiable, dtype=float)
  tau = times / (maximum * np.sqrt(1064.0 / 101325.0))
  got = np.interp(GRID, tau[: curve.size], curve)
  got = got - got.mean()
  span = []
  for column in jacobian.T:
    resampled = np.interp(GRID, tau[: column.size], column)
    resampled = resampled - resampled.mean()
    span.append(resampled / (np.linalg.norm(resampled) + 1e-30))
  return dataset, {"curve": (got / np.linalg.norm(got)).tolist(), "span": np.array(span).tolist(),
                   "chi2": float(fit.chi_squared), "stretch": stretch}


def main():
  with records.pool(len(ORDER)) as pool:
    fixed = dict(pool.map(corrected, list(ORDER)))
    plain = dict(pool.map(delta_one, list(ORDER)))
  good = [d for d in ORDER if "failed" not in fixed[d]]
  label = "with vapour on" if THERMAL else "isothermal, vapour NOT corrected"
  print(f"  {len(good)} of {len(ORDER)} refitted at the preferred stretch, {label}")
  for d in ORDER:
    if "failed" in fixed[d]: print(f"    {d}: {fixed[d]['failed']}")
  if len(good) < 4: return

  fixed_c = {d: np.array(fixed[d]["curve"]) for d in good}
  plain_c = {d: np.array(plain[d]["curve"]) for d in good}
  spans = {d: np.array(fixed[d]["span"]) for d in good}

  rng = np.random.default_rng(seed_for("corrected"))

  def constrained(d):
    out = _surrogate(rng, fixed_c[d])
    basis, _ = np.linalg.qr(spans[d].T)
    out = out - basis @ (basis.T @ out)
    return out / (np.linalg.norm(out) + 1e-30)

  null = np.array([abs(np.dot(constrained(good[a]), constrained(good[b])))
                   for a, b in rng.choice(len(good), size=(DRAWS, 2)) if a != b])
  cut = float(np.percentile(null, 95))

  print(f"\n  {'record':>16s} {'chi2 corrected':>15s} {'stretch':>8s} {'|cos| vs uncorrected':>21s}")
  same = []
  for d in good:
    r = float(abs(np.dot(fixed_c[d], plain_c[d])))
    same.append(r)
    print(f"  {d:>16s} {fixed[d]['chi2']:15.4f} {fixed[d]['stretch']:8.1f} {r:21.3f}")

  across, within = [], []
  for i, a in enumerate(good):
    for b in good[i + 1:]:
      r = float(abs(np.dot(fixed_c[a], fixed_c[b])))
      ga, gb = a in records.DATASETS, b in records.DATASETS
      (across if ga != gb else within).append(r)

  print(f"\n  constrained null: mean {null.mean():.3f}, 95th percentile {cut:.3f}\n")
  print(f"  corrected vs uncorrected, same record: median {np.median(same):.3f}")
  if within:
    print(f"  corrected, same material:              median {np.median(within):.3f}, "
          f"{sum(1 for r in within if r > cut)} of {len(within)} above the null")
  if across:
    print(f"  corrected, ACROSS materials:           median {np.median(across):.3f}, "
          f"{sum(1 for r in across if r > cut)} of {len(across)} above the null")

  print("\n  ---- what it says ----\n")
  if across and np.median(across) > cut:
    what = "the fits" if THERMAL else "the equilibrium radius"
    print(f"  The cross-material agreement survives correcting {what}, so the shared curve is")
    print("  not the shape of THAT defect and sec:universal stands against it.")
    if not THERMAL:
      print("  Vapour is not corrected here and this says nothing about it.")
    if np.median(same) < cut:
      print("  But the corrected curve is NOT the uncorrected one, so the curve that gets")
      print("  published has to be this one.")
  else:
    print("  Correcting the fits destroys the cross-material agreement. What sec:universal")
    print("  measured was substantially the shape of a pinned radius and an absent vapour")
    print("  pressure, shared by every fit, and the claim must be withdrawn in that form.")
  json.dump({"same_record": same, "within": within, "across": across, "null_95": cut,
             "chi2": {d: fixed[d]["chi2"] for d in good}},
            open(records.HERE / "delta_corrected_vapour.json", "w"), indent=1)


if __name__ == "__main__":
  main()
