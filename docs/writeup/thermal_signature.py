r"""Is the universal curve the isothermal approximation's own error?

\Cref{sec:vapour} names this as the leading candidate identity for the shared discrepancy: the
isothermal, vapour-free gas is a property of the MODEL rather than of either material, so its
error would appear on both alike, and it is largest in the ringdown where most of the common
mode's norm sits. The refit in `delta_corrected.py` answers the question indirectly, by asking
whether $\hat\delta$ SURVIVES the correction. This asks it directly and far more cheaply.

The thermal correction has a shape of its own, and that shape can be computed without fitting
anything. At each record's converged isothermal parameters, integrate the model twice -- once as
the document fits it, once with `bubtherm`/`masstrans`/`vapor` on -- and difference the two in
noise units. That difference IS the term the isothermal fits are missing, to first order and at
this point in parameter space. Projecting out the isothermal sensitivity span puts it in exactly
the coordinates $\hat\delta$ lives in, because a component the parameters could absorb is not a
discrepancy anyone could measure.

Then correlate it against the measured common mode. A high $\lvert\cos\rvert$ says the universal
curve is substantially the thermal approximation's error and the claim must be restated in those
terms. A low one is the stronger result for the paper, and it is the one the design of this test
would most easily have found otherwise: nothing here is tuned, and the comparison is against a
curve measured before this test existed.
"""

import json

import numpy as np

import records
from shape_error import seed_for
from universal_delta import GRID, _surrogate

THERMAL = {"bubtherm": 1, "Nt": 25, "masstrans": 1, "vapor": 1}
ORDER = (*records.DATASETS, *records.PAAM)
DRAWS = 4000


def one(dataset):
  """The thermal correction's identifiable signature, on the common grid."""
  from pyimr.noise import discrepancy
  from pyimr.selection import STANDARD_MODELS, evaluate_at

  times, mean, spread, maximum, _ = records.load(dataset)
  stretch = json.load(open(records.HERE / "paam_stretch.json"))[dataset]["argmin"]
  fitted = json.load(open(records.HERE / "paam_lackoffit.json"))[dataset]["fitted"]
  candidate = STANDARD_MODELS["qSLS"]
  cold = records.solver(times, maximum, stretch, max_steps=400_000)
  hot = records.solver(times, maximum, stretch, max_steps=400_000, **THERMAL)
  error_bar = spread / np.sqrt(records.trial_count(dataset))
  try:
    base = np.asarray(evaluate_at(candidate, cold, fitted)[0], dtype=float)
    warm = np.asarray(evaluate_at(candidate, hot, fitted)[0], dtype=float)
  except Exception as failure:                                               # noqa: BLE001
    return dataset, {"failed": f"{type(failure).__name__}: {failure}"[:110]}

  # the ISOTHERMAL sensitivities, the span delta_hat is orthogonal to on this record
  step, columns = 1e-4, []
  for axis in candidate.axes:
    up = dict(fitted, **{axis: fitted[axis] * (1 + step)})
    dn = dict(fitted, **{axis: fitted[axis] * (1 - step)})
    hi = np.asarray(evaluate_at(candidate, cold, up)[0], dtype=float)
    lo = np.asarray(evaluate_at(candidate, cold, dn)[0], dtype=float)
    columns.append((hi - lo) / (2 * step * error_bar))
  jacobian = np.column_stack(columns)

  signature = np.asarray(discrepancy((warm - base) / error_bar, jacobian).identifiable, float)
  measured = np.asarray(discrepancy((mean - base) / error_bar, jacobian).identifiable, float)
  tau = times / (maximum * np.sqrt(1064.0 / 101325.0))

  def onto(curve):
    got = np.interp(GRID, tau[: curve.size], curve)
    got = got - got.mean()
    return got / (np.linalg.norm(got) + 1e-30)

  # how much of the record's OWN discrepancy the correction could account for
  return dataset, {"signature": onto(signature).tolist(), "own": onto(measured).tolist(),
                   "relative_size": float(np.linalg.norm(signature) / np.linalg.norm(measured))}


def main():
  with records.pool(len(ORDER)) as pool:
    got = dict(pool.map(one, list(ORDER)))
  good = [d for d in ORDER if "failed" not in got[d]]
  for d in ORDER:
    if "failed" in got[d]: print(f"    {d}: {got[d]['failed']}")

  stored = json.load(open(records.HERE / "universal_curve.json"))
  mode = np.asarray(stored["common_mode"], dtype=float)
  mode = (mode - mode.mean()) / np.linalg.norm(mode - mode.mean())

  # the null pairs a surrogate of THIS record's signature with a surrogate of the mode, so each
  # side carries its own smoothness. Using the mode's spectrum for both sides would be a
  # different question, and on these curves a slightly easier one.
  rng = np.random.default_rng(seed_for("thermal_signature"))

  def null_for(curve):
    draws = np.array([abs(_surrogate(rng, curve) @ _surrogate(rng, mode)) for _ in range(DRAWS)])
    return float(np.percentile(draws, 95))

  print("\n  the thermal correction's own shape, against the measured discrepancy\n")
  print(f"  {'record':>16s} {'|cos| vs common mode':>21s} {'|cos| vs own delta':>19s}"
        f" {'size vs own':>12s} {'null 95%':>11s}")
  out = {}
  for d in good:
    signature = np.asarray(got[d]["signature"], dtype=float)
    versus_mode = float(abs(signature @ mode))
    versus_own = float(abs(signature @ np.asarray(got[d]["own"], dtype=float)))
    cut = null_for(signature)
    out[d] = {"vs_mode": versus_mode, "vs_own": versus_own, "null_95": cut,
              "relative_size": got[d]["relative_size"]}
    print(f"  {d:>16s} {versus_mode:21.3f} {versus_own:19.3f}"
          f" {got[d]['relative_size']:11.2f}x {cut:11.3f}")

  mode_scores = [out[d]["vs_mode"] for d in good]
  beat = sum(1 for d in good if out[d]["vs_mode"] > out[d]["null_95"])
  cut = float(np.median([out[d]["null_95"] for d in good]))
  print(f"\n  median |cos| against the common mode: {np.median(mode_scores):.3f}, "
        f"{beat} of {len(mode_scores)} above each record's OWN null (median {cut:.3f})")
  print(f"  the correction is {min(out[d]['relative_size'] for d in good):.2f}x to "
        f"{max(out[d]['relative_size'] for d in good):.2f}x the size of the discrepancy it "
        "would have to explain,\n  so this is not a null result for want of a large enough "
        "correction.")

  # A median below the null is NOT the whole verdict: a minority of records aligning above
  # their own null is a real partial signal, and reporting only the median would hide it in
  # the direction this test would prefer to go. So both are stated and the count decides how
  # strongly the negative is phrased.
  expected = 0.05 * len(good)
  print("\n  ---- what it says ----\n")
  if np.median(mode_scores) > cut:
    print("  The thermal correction's shape IS substantially the universal curve. The shared")
    print("  discrepancy is then largely the isothermal approximation's own error, a model")
    print("  defect common to every fit rather than a missing constitutive physics, and")
    print("  sec:universal has to be restated in those terms.")
  elif beat > 2 * expected:
    print("  The thermal correction does not have the universal curve's shape -- median")
    print(f"  {np.median(mode_scores):.3f} against a null of {cut:.3f}, and nowhere near the")
    print(f"  {0.726:.3f} the two materials agree at. But {beat} of {len(good)} records align")
    print(f"  above their OWN null where {expected:.1f} would be expected, so a MINORITY")
    print("  component of the correction does resemble the curve. The candidate is reduced")
    print("  rather than eliminated, and that is the honest reading.")
  else:
    print("  The thermal correction does NOT have the shape of the universal curve. Whatever")
    print("  the shared discrepancy is, correcting the gas thermodynamics does not supply it,")
    print("  and the leading mundane candidate is answered rather than deferred.")
  json.dump({"per_record": out, "null_95": cut,
             "median_vs_mode": float(np.median(mode_scores))},
            open(records.HERE / "thermal_signature.json", "w"), indent=1)


if __name__ == "__main__":
  main()
