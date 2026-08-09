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

`R_max` is included because each trace is divided by its OWN maximum, so bubble-size variation
does not simply scale the curve away: what survives normalisation enters through the Reynolds,
Weber and Deborah groups, all of which carry `R_max`. It does not explain the dominant mode
either, correlating $0.30$, $0.09$ and $0.28$ -- indistinguishable from the other axes.

So the spread is neither independent noise nor latent variation in any parameter this model
has, and what its dominant mode is remains open (#222).

Read the joint $R^2$ and not the single correlations. The sensitivities here are near
collinear -- the sloppiness of the identifiability section -- so at \SI{23}{\celsius} the
second mode correlates above $0.7$ with four different axes at once, which singles out none of
them. Two further caveats: the sensitivities are evaluated at each record's own fit, and a
wrong fit points them the wrong way; and the replicate deviations sum to zero across trials,
which biases a seven-trial record more than an eighteen-trial one.
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
  """Each record's OWN qSLS point. Using one record's material for all three would evaluate
  the sensitivities in the wrong place and understate how well they explain the scatter.

  This is the stored GRID ARGMAX, not a fitted optimum: ten points per axis, adjacent nodes a
  factor of two to three apart, and on the bounds for `g` and `lambda1` on two of the three
  records (#235). Sensitivities evaluated there point in roughly, not exactly, the right
  direction. Substituting the published continuous fit moves the joint $R^2$ of the dominant
  mode by $0.02$, so the conclusion here does not turn on it -- but the caveat is real and the
  right fix is #235's, which is to store an optimiser's answer beside the grid's.
  """
  best = _RES[name]["models"]["qSLS"]["best_theta"]
  return (best["g"], best["mu"], best["lambda1"], best["alpha"])


def sensitivities(name):
  """Candidate causes of a per-trial deviation, on the record's own grid."""
  import pyimr
  times, mean, spread, maximum, stretch = records.load(name)

  point = fitted(name)
  from pyimr.noise import characteristic_time
  # the trial grid is nondimensional, so R_max variation is compared at fixed t/t_c
  grid = times / characteristic_time(maximum)

  def trace(req=1.0, shift=0.0, rmax=1.0, **scale):
    values = [point[0] * scale.get("g", 1.0), point[1] * scale.get("mu", 1.0),
              point[2] * scale.get("lam", 1.0), 0.0, point[3] * scale.get("alpha", 1.0)]
    material = pyimr.QuadraticZener(*values)
    # Each trace is divided by its OWN R_max, so a bigger bubble does not simply scale the
    # curve. What survives normalisation is the nondimensional groups -- Reynolds, Weber,
    # Deborah -- every one of which carries R_max, so the residual signature is real and is
    # the one axis none of the others reaches.
    scaled = maximum * rmax
    when = grid * characteristic_time(scaled) + shift
    config = pyimr.SimulationConfig(scaled, scaled / stretch * req, material,
                                    dynamics="keller-miksis", rtol=1e-9, atol=1e-11,
                                    max_steps=400_000)
    return np.asarray(pyimr.simulate(when, config).radius_ratio, dtype=float)

  h = 1e-4
  base = trace()
  columns = {"timing": np.gradient(base, times),
             "Req": (trace(req=1 + h) - trace(req=1 - h)) / (2 * h),
             "Rmax": (trace(rmax=1 + h) - trace(rmax=1 - h)) / (2 * h)}
  for key in ("g", "mu", "lam", "alpha"):
    columns[key] = (trace(**{key: 1 + h}) - trace(**{key: 1 - h})) / (2 * h)
  return times, columns


def main():
  labels = ("timing", "Req", "Rmax", "g", "mu", "lam", "alpha")
  print(f"  {'record':13s} {'mode':>5s} {'share':>6s} {'R^2':>6s}  "
        + " ".join(f"{k:>7s}" for k in labels))
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
      print(f"  {name:13s} {index + 1:5d} {share[index]:6.3f} {r2:6.3f}  "
            + " ".join(f"{singles[k]:7.3f}" for k in labels))
    print()


if __name__ == "__main__":
  main()
