"""What shape does the trial-to-trial scatter have, and can the model account for it?

The likelihood treats the trial spread as independent noise. Two things say otherwise. The
scatter is LOW RANK -- one mode carries between $54\%$ and $72\%$ of it -- and its
autocorrelation decays faster than any exponential and turns negative, which no stationary
AR(1) process does. So it is structured, and the question is what the structure is.

If it lay in the span of the model's own sensitivities it would be per-trial PARAMETER
variation, and the fix would be to marginalise those parameters rather than to widen the
noise. Measured here, the dominant mode does not: regressed on timing, `Req` and all four
material axes it reaches $R^2$ of only $0.12$ to $0.43$. The SECOND mode largely does, at
$0.50$ to $0.72$, and reads as timing jitter and viscosity.

So the spread is neither independent noise nor latent variation in the fitted parameters, and
what its dominant mode is remains open (#222). Two caveats travel with that. The sensitivities
are evaluated at each record's own fit, and a wrong fit points them the wrong way. And the
replicate deviations sum to zero across trials, which biases a seven-trial record more than an
eighteen-trial one.
"""
import sys; sys.path.insert(0, ".")

import numpy as np

import records

DATA = records.Path.home() / "fastscratch/imr-data-tempsweeps/data"
# offsets verified by matching the record mean against the trial mean to 0.00e+00; the record
# is a prefix of the trial file, and at two of three records it does not start at row zero
FILES = {"gelatin_15C": ("Ga_t15_exp_data.csv", 0), "gelatin_23C": ("Ga_t23_exp_data.csv", 1),
         "gelatin_33C": ("Ga_t33_exp_data.csv", 1)}
DATA_NOTE = "per-trial traces, not in the public repository"
import json as _json
_RES = _json.load(open("results.json"))


def fitted(name):
  """Each record's OWN qSLS fit. Using one record's material for all three would evaluate the
  sensitivities in the wrong place and understate how well they explain the scatter."""
  best = _RES[name]["models"]["qSLS"]["best_theta"]
  return (best["g"], best["mu"], best["lambda1"], best["alpha"])


def sensitivities(name):
  """Candidate causes of a per-trial deviation, on the record's own grid."""
  import pyimr
  times, mean, spread, maximum, stretch = records.load(name)

  point = fitted(name)

  def trace(req=1.0, shift=0.0, **scale):
    values = [point[0] * scale.get("g", 1.0), point[1] * scale.get("mu", 1.0),
              point[2] * scale.get("lam", 1.0), 0.0, point[3] * scale.get("alpha", 1.0)]
    material = pyimr.QuadraticZener(*values)
    config = pyimr.SimulationConfig(maximum, maximum / stretch * req, material,
                                    dynamics="keller-miksis", rtol=1e-9, atol=1e-11,
                                    max_steps=400_000)
    return np.asarray(pyimr.simulate(times + shift, config).radius_ratio, dtype=float)

  h = 1e-4
  base = trace()
  columns = {"timing": np.gradient(base, times),
             "Req": (trace(req=1 + h) - trace(req=1 - h)) / (2 * h)}
  for key in ("g", "mu", "lam", "alpha"):
    columns[key] = (trace(**{key: 1 + h}) - trace(**{key: 1 - h})) / (2 * h)
  return times, columns


def main():
  print(f"  {'record':13s} {'mode':>5s} {'share':>7s}  {'R^2 on all sensitivities':>24s}  best single")
  for name, (filename, offset) in FILES.items():
    raw = np.loadtxt(DATA / filename, delimiter=",")
    times, columns = sensitivities(name)
    window = raw[offset:offset + times.size, 1:]                 # exact alignment, verified
    deviations = window - window.mean(axis=1, keepdims=True)
    left, sv, _ = np.linalg.svd(deviations, full_matrices=False)
    share = sv**2 / (sv**2).sum()

    design = np.column_stack([v / np.linalg.norm(v) for v in columns.values()])
    for index in (0, 1):
      mode = left[:, index]
      fitted, *_ = np.linalg.lstsq(design, mode, rcond=None)
      residual = mode - design @ fitted
      r2 = 1.0 - float(residual @ residual) / float(mode @ mode)
      singles = {k: abs(float(np.corrcoef(mode, v)[0, 1])) for k, v in columns.items()}
      best = max(singles.items(), key=lambda kv: kv[1])
      print(f"  {name:13s} {index + 1:5d} {share[index]:7.3f}  {r2:24.3f}  {best[0]} {best[1]:.3f}")
    print()


if __name__ == "__main__":
  main()
