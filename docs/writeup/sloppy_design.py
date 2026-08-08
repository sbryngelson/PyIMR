"""Does designing against the ridge buy identifiability at the cost of adequacy?

The literature on sloppy systems reports that a model can fit WORSE on data from its own
optimal experiment, with less predictive power after optimal selection than before, because
the optimal design pushes into the regime where the model's inadequacy shows. Our situation is
the setup for that failure: the `g`--`alpha` ridge is textbook sloppiness, the design here
wants a far more violent collapse than anything performed, and the records already say the
model is inadequate at the gentler geometry we have.

That is testable before a collaborator spends a bubble, and this is the test.

THE TRAP THIS AVOIDS. Generating data from the fitted model and refitting it would be the
same mistake the synthetic BOED study makes: with the truth inside the candidate set there is
no model error to expose, and every geometry would fit perfectly. So the truth here carries
physics the fitted model does not have -- the thermal treatment, which `confounding.py`
measures as the largest of the three model axes, moving the trace by 16 noise units and
keeping 8.5 after the material absorbs what it can. The fitted model is cold qSLS, exactly
what `selection.tex` uses. The mismatch is therefore real and of a size we have measured.

WHAT WOULD SETTLE IT. Two numbers per geometry, and they are expected to move in opposite
directions. `chi2/N` and the residual's lag-one say whether the model still describes the
data; the recovered `g*alpha` says whether the experiment did its job. The warning is
confirmed if the aggressive designs recover the parameters better while fitting worse.
"""

import json

import numpy as np

import records

TRUTH = {"g": 204.3, "mu": 0.04651, "lambda1": 1.964e-7, "alpha": 5.301}
WIDE = {"g": (1e0, 1e6), "mu": (1e-5, 1e1), "lambda1": (1e-9, 1e-2), "alpha": (1e-4, 1e3)}
SAMPLES = 201
RELATIVE_NOISE = 0.02          # of R_max, matching `design_operator.py`
# from `selection.tex`: the geometry performed, the E-optimal one, and the one the
# discrimination criterion wants, which is near the opposite corner of the design space
GEOMETRIES = {
  "performed": (277e-6, 7.09),
  "E-optimal": (199e-6, 3.55),
  "discrimination-optimal": (100e-6, 20.0),
  "measure support (small)": (60e-6, 18.0),
}


# Two mismatches, so the finding is a pattern rather than one anecdote. Both are sized by
# `confounding.py`: thermal moves the trace by 16 noise units, the second relaxation mode by
# 14.4. Neither is in the fitted model, and the fitted model is cold one-mode qSLS throughout
# -- exactly what `selection.tex` uses.
MISMATCHES = ("thermal", "two-mode")


def _material(mismatch=None):
  import pyimr

  if mismatch == "two-mode":
    return pyimr.TwoModeQuadraticZener(TRUTH["g"], TRUTH["mu"], TRUTH["lambda1"], 0.0,
                                       TRUTH["alpha"], 2e-6, 0.2)
  return pyimr.QuadraticZener(TRUTH["g"], TRUTH["mu"], TRUTH["lambda1"], 0.0, TRUTH["alpha"])


def _job(argument):
  mismatch, name = argument
  import pyimr
  from pyimr.noise import characteristic_time
  from pyimr.selection import STANDARD_MODELS, evaluate_at

  radius, stretch = GEOMETRIES[name]
  times = np.linspace(0.0, 5.0 * characteristic_time(radius), SAMPLES)

  def solve_with(**options):
    def solve(material, _config=None):
      config = pyimr.SimulationConfig(radius, radius / stretch, material,
                                      dynamics="keller-miksis", rtol=1e-9, atol=1e-11,
                                      max_steps=1_000_000, **options)
      trace = np.asarray(pyimr.simulate(times, config).radius_ratio, dtype=float)
      return trace, trace
    return solve

  # the truth carries physics the fitted model does not
  options = dict(bubtherm=1, medtherm=1, Nt=11, Mt=11) if mismatch == "thermal" else {}
  try:
    truth = solve_with(**options)(_material(mismatch))[0]
  except Exception as error:                          # noqa: BLE001
    return (mismatch, name), {"failed": f"truth: {type(error).__name__}: {error}"}
  if not np.all(np.isfinite(truth)): return (mismatch, name), {"failed": "truth did not integrate"}

  spread = np.full(SAMPLES, RELATIVE_NOISE)
  observed = truth + np.random.default_rng(0).normal(0.0, RELATIVE_NOISE, SAMPLES)
  cold = solve_with()
  candidate = STANDARD_MODELS["qSLS"]
  try:
    scored = records.score(candidate, cold, observed, spread, bounds=WIDE, starts=12,
                           evaluations=400)
  except Exception as error:                          # noqa: BLE001
    return (mismatch, name), {"failed": f"fit: {type(error).__name__}: {error}"}

  fitted = scored["fitted"]
  # the identified combination, which is what the ridge leaves determined
  product = fitted["g"] * fitted["alpha"]
  reference = TRUTH["g"] * TRUTH["alpha"]
  # how much of the residual is the thermal physics rather than the noise we added
  model_error = float(np.sqrt(np.mean(((evaluate_at(candidate, cold, fitted)[0] - truth)
                                       / RELATIVE_NOISE) ** 2)))
  return (mismatch, name), dict(chi2_per_n=scored["chi2_per_n"], lag_one=scored["lag_one"],
                                galpha=product, galpha_error=abs(product / reference - 1.0),
                                model_error=model_error, fitted=fitted)


def main():
  jobs = [(mismatch, name) for mismatch in MISMATCHES for name in GEOMETRIES]
  with records.pool(len(jobs)) as pool:
    table = dict(pool.map(_job, jobs))

  for mismatch in MISMATCHES:
    print(f"\ncold one-mode qSLS fitted to a {mismatch} truth, at four geometries\n")
    print(f"{'geometry':>26} {'R_max/um':>9} {'stretch':>8} {'chi2/N':>8} {'lag-1':>7} "
          f"{'|g.alpha| err':>14} {'model err':>10}")
    for name in GEOMETRIES:
      row = table[(mismatch, name)]
      radius, stretch = GEOMETRIES[name]
      if "failed" in row:
        print(f"{name:>26} {radius * 1e6:9.0f} {stretch:8.2f}   {row['failed'][:44]}")
        continue
      print(f"{name:>26} {radius * 1e6:9.0f} {stretch:8.2f} {row['chi2_per_n']:8.3f} "
            f"{row['lag_one']:7.3f} {row['galpha_error']:13.1%} {row['model_error']:10.2f}")

    base = table[(mismatch, "performed")]
    if "failed" in base: continue
    print("\n  against the performed geometry:")
    for name in GEOMETRIES:
      row = table[(mismatch, name)]
      if name == "performed" or "failed" in row: continue
      fit = row["chi2_per_n"] / base["chi2_per_n"]
      recovery = row["galpha_error"] / max(base["galpha_error"], 1e-12)
      verdict = ("the literature's trade: recovers better, fits worse" if fit > 1.1 and recovery < 0.9
                 else "WORSE ON BOTH" if fit > 1.1
                 else "better on both" if recovery < 0.9 and fit < 0.9
                 else "no clear trade")
      print(f"    {name:>26}: chi2/N x{fit:5.2f}, g.alpha error x{recovery:5.2f} -- {verdict}")

  # The claim is only worth making if it does not depend on which physics is missing. Stated
  # so it can fail: does the performed geometry beat the recommended ones under BOTH mismatches?
  print("\n\n  does the finding survive a change of mismatch?\n")
  for name in GEOMETRIES:
    if name == "performed": continue
    verdicts = []
    for mismatch in MISMATCHES:
      row, base = table[(mismatch, name)], table[(mismatch, "performed")]
      if "failed" in row or "failed" in base: verdicts.append("--"); continue
      worse_fit = row["chi2_per_n"] > base["chi2_per_n"]
      worse_recovery = row["galpha_error"] > base["galpha_error"]
      verdicts.append("both worse" if worse_fit and worse_recovery else
                      "fit worse" if worse_fit else
                      "recovery worse" if worse_recovery else "better")
    agree = len(set(verdicts)) == 1 and verdicts[0] != "--"
    print(f"    {name:>26}: " + ", ".join(f"{m}: {v}" for m, v in zip(MISMATCHES, verdicts, strict=True))
          + ("   [agrees]" if agree else "   [depends on the mismatch]"))

  print("\n  The truth carries physics the fitted model lacks -- thermal transport in one set,")
  print("  a second relaxation mode in the other -- both of a size `confounding.py` measured.")
  print("  `model err` is that mismatch in noise units with the added noise removed, which is")
  print("  what a study generating from its own model would report as 0.")
  records.HERE.joinpath("sloppy_design.json").write_text(
    json.dumps({f"{m}|{n}": v for (m, n), v in table.items()}, indent=1))


if __name__ == "__main__":
  main()
