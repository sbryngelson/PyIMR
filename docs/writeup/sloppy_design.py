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


def _material():
  import pyimr

  return pyimr.QuadraticZener(TRUTH["g"], TRUTH["mu"], TRUTH["lambda1"], 0.0, TRUTH["alpha"])


def _job(name):
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

  # truth carries thermal transport; the fitted model does not
  try:
    truth = solve_with(bubtherm=1, medtherm=1, Nt=11, Mt=11)(_material())[0]
  except Exception as error:                          # noqa: BLE001
    return name, {"failed": f"truth: {type(error).__name__}: {error}"}
  if not np.all(np.isfinite(truth)): return name, {"failed": "truth did not integrate"}

  spread = np.full(SAMPLES, RELATIVE_NOISE)
  observed = truth + np.random.default_rng(0).normal(0.0, RELATIVE_NOISE, SAMPLES)
  cold = solve_with()
  candidate = STANDARD_MODELS["qSLS"]
  try:
    scored = records.score(candidate, cold, observed, spread, bounds=WIDE, starts=12,
                           evaluations=400)
  except Exception as error:                          # noqa: BLE001
    return name, {"failed": f"fit: {type(error).__name__}: {error}"}

  fitted = scored["fitted"]
  # the identified combination, which is what the ridge leaves determined
  product = fitted["g"] * fitted["alpha"]
  reference = TRUTH["g"] * TRUTH["alpha"]
  # how much of the residual is the thermal physics rather than the noise we added
  model_error = float(np.sqrt(np.mean(((evaluate_at(candidate, cold, fitted)[0] - truth)
                                       / RELATIVE_NOISE) ** 2)))
  return name, dict(chi2_per_n=scored["chi2_per_n"], lag_one=scored["lag_one"],
                    galpha=product, galpha_error=abs(product / reference - 1.0),
                    model_error=model_error, fitted=fitted)


def main():
  names = list(GEOMETRIES)
  with records.pool(len(names)) as pool:
    table = dict(pool.map(_job, names))

  print("cold qSLS fitted to thermal truth, at four geometries\n")
  print(f"{'geometry':>26} {'R_max/um':>9} {'stretch':>8} {'chi2/N':>8} {'lag-1':>7} "
        f"{'|g.alpha| err':>14} {'model err':>10}")
  for name in names:
    row = table[name]
    radius, stretch = GEOMETRIES[name]
    if "failed" in row:
      print(f"{name:>26} {radius * 1e6:9.0f} {stretch:8.2f}   {row['failed'][:44]}")
      continue
    print(f"{name:>26} {radius * 1e6:9.0f} {stretch:8.2f} {row['chi2_per_n']:8.3f} "
          f"{row['lag_one']:7.3f} {row['galpha_error']:13.1%} {row['model_error']:10.2f}")

  usable = {k: v for k, v in table.items() if "failed" not in v}
  if "performed" in usable and len(usable) > 1:
    base = usable["performed"]
    print("\n  against the performed geometry:")
    for name, row in usable.items():
      if name == "performed": continue
      fit = row["chi2_per_n"] / base["chi2_per_n"]
      recovery = row["galpha_error"] / max(base["galpha_error"], 1e-12)
      verdict = ("WARNING CONFIRMED: recovers better, fits worse" if fit > 1.1 and recovery < 0.9
                 else "fits worse and recovers no better" if fit > 1.1
                 else "fits better and recovers better" if recovery < 0.9
                 else "no clear trade")
      print(f"    {name:>26}: chi2/N x{fit:5.2f}, g.alpha error x{recovery:5.2f} -- {verdict}")

  print("\n  The truth carries thermal transport the fitted model lacks, so the residual is")
  print("  model-form error of a size `confounding.py` measured, not noise drawn from the")
  print("  model itself. `model err` is that mismatch in noise units, with the added noise")
  print("  removed -- it is what a synthetic study generating from its own model reports as 0.")
  records.HERE.joinpath("sloppy_design.json").write_text(json.dumps(table, indent=1))


if __name__ == "__main__":
  main()
