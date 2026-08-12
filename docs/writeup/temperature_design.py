r"""The one design variable that rotates the eigenvectors instead of amplifying the signal.

\Cref{sec:criteria} proves the thing that should govern this chapter and then nothing in it
uses: a design change that merely amplifies signal sends $F \to sF$, so $\det F$ and
$\lambda_{\min}$ both improve while $\kappa = \lambda_{\max}/\lambda_{\min}$ is UNCHANGED.
Degeneracy is a property of the eigen\emph{vectors}. Only a design that rotates them removes it.

Every candidate in this document is $(R_{\max}, \text{stretch})$, and both are drive settings.
Temperature is not: it changes the material's own $g\alpha$ --- \cref{sec:trend} measures the
stiffening advantage shrinking monotonically as the gel warms, the most robust physical result
here --- and it costs nothing but a water bath. It has never been a design variable.

THE CATCH IS REAL AND IS THE POINT OF THIS SCRIPT. Each temperature has its own parameters, so
three records at three temperatures are three separate fits and there is nothing to design
FOR. Temperature becomes a design axis only under a model that shares parameters across it. The
weakest such model that is still physics is Arrhenius,

    log theta_a(T) = c_a + b_a (1/T - 1/T_ref) ,

two coefficients per axis, six in total, and the chain rule turns a trace's sensitivity to
$\log\theta_a$ into sensitivities to $(c_a, b_a)$ with the second scaled by $(1/T - 1/T_{\rm
ref})$. That is a bigger claim than anything else in the design chapter rests on --- three
temperatures fitting six coefficients leaves one residual degree of freedom per axis --- so the
Arrhenius fit's own quality is reported first and the design is conditional on it.

WHAT WOULD MAKE IT WORTH THE CLAIM. If temperature only amplified, $\kappa$ would be flat
across it and the axis would be a more expensive way to buy what $R_{\max}$ already buys. The
measurement is $\kappa$ of the six-parameter information at a batch that varies temperature
against one that does not, and $\log\det$ beside it so the two effects are not confused.
"""

import json

import numpy as np

import records
from identified import candidate_at_ratio
from noise_design import profile

RATIO = 38.5
AXES = ("mu", "galpha", "lambda1")
SAMPLES = 201
KELVIN = 273.15
RECORD_T = {"gelatin_15C": 15.0, "gelatin_23C": 23.0, "gelatin_33C": 33.0}
REFERENCE_T = 23.0
# inside the range the records span, plus a little: extrapolating an Arrhenius fit made from
# three points is exactly the move this script is meant to make visible rather than hide
TEMPERATURES = (12.0, 18.0, 23.0, 28.0, 36.0)
RADII = np.geomspace(50e-6, 1200e-6, 8)
STRETCH = np.linspace(3.0, 20.0, 6)


def arrhenius():
  """`(intercepts, slopes, residuals)` of `log theta = c + b (1/T - 1/T_ref)` per axis."""
  measured = json.load(open(records.HERE / "per_trial_fits.json"))
  inverse = np.array([1.0 / (KELVIN + RECORD_T[d]) - 1.0 / (KELVIN + REFERENCE_T)
                      for d in records.DATASETS])
  design = np.column_stack([np.ones_like(inverse), inverse])
  intercepts, slopes, residuals = {}, {}, {}
  for axis in AXES:
    values = np.log([float(measured[d]["median"][axis]) for d in records.DATASETS])
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    intercepts[axis], slopes[axis] = float(coefficients[0]), float(coefficients[1])
    residuals[axis] = (values - design @ coefficients).tolist()
  return intercepts, slopes, residuals


def parameters_at(intercepts, slopes, celsius):
  offset = 1.0 / (KELVIN + celsius) - 1.0 / (KELVIN + REFERENCE_T)
  return {a: float(np.exp(intercepts[a] + slopes[a] * offset)) for a in AXES}, offset


def one(job):
  """The six-parameter information at one `(R_max, stretch, T)`."""
  import pyimr
  from pyimr.noise import characteristic_time

  radius, stretch, celsius, intercepts, slopes = job
  fitted, offset = parameters_at(intercepts, slopes, celsius)
  point = np.array([fitted[a] for a in AXES])
  candidate = candidate_at_ratio(RATIO)
  times = np.linspace(0.0, 5.0 * characteristic_time(radius), SAMPLES)

  def trace(scale):
    moved = dict(zip(AXES, (float(v) for v in point * scale), strict=True))
    config = pyimr.SimulationConfig(radius, radius / stretch, candidate.build(moved),
                                    dynamics="keller-miksis", rtol=1e-8, atol=1e-10,
                                    max_steps=200_000)
    return np.asarray(pyimr.simulate(times, config).radius_ratio, dtype=float)

  step = 1e-4
  try:
    columns = [(trace(np.where(np.arange(len(AXES)) == k, 1 + step, 1.0))
                - trace(np.where(np.arange(len(AXES)) == k, 1 - step, 1.0))) / (2.0 * step)
               for k in range(len(AXES))]
  except Exception:                                                          # noqa: BLE001
    return (radius, stretch, celsius), None
  material = np.column_stack(columns)
  if not np.all(np.isfinite(material)): return (radius, stretch, celsius), None

  tau, spread = profile("gelatin_15C")
  phase = (times - times[0]) / characteristic_time(radius)
  whitened = material / np.interp(phase, tau, spread)[:, None]
  # the chain rule: d/dc_a is d/dlog(theta_a), and d/db_a is that times (1/T - 1/T_ref)
  augmented = np.column_stack([whitened, offset * whitened])
  gram = augmented.T @ augmented
  return (radius, stretch, celsius), 0.5 * (gram + gram.T)


def _spectrum(matrix):
  values = np.linalg.eigvalsh(matrix)
  positive = values[values > 0]
  if positive.size < matrix.shape[0]: return float("inf"), -np.inf
  return float(positive.max() / positive.min()), float(np.sum(np.log(positive)))


def main():
  from pyimr.measure import optimal_measure

  intercepts, slopes, residuals = arrhenius()
  print("\n  the Arrhenius fit the design axis rests on\n")
  print(f"  {'axis':10s} {'log theta at Tref':>18s} {'slope b (K)':>13s} "
        f"{'residuals (log units)':>34s}")
  for axis in AXES:
    marks = " ".join(f"{r:+.4f}" for r in residuals[axis])
    print(f"  {axis:10s} {intercepts[axis]:18.4f} {slopes[axis]:13.1f} {marks:>34s}")
  print("  three points, two coefficients: one residual degree of freedom per axis, so these")
  print("  residuals are a consistency note and not a test of the Arrhenius form.")

  designs = [(float(r), float(s), float(t), intercepts, slopes)
             for r in RADII for s in STRETCH for t in TEMPERATURES]
  print(f"\n  scoring {len(designs)} (geometry, temperature) candidates ...", flush=True)
  with records.pool(len(designs)) as pool:
    got = dict(pool.map(one, designs))
  usable = {k: v for k, v in got.items() if v is not None}
  print(f"  {len(usable)} of {len(designs)} integrate")

  labels = list(usable)
  stack = np.array([usable[k] for k in labels])
  summary = {"arrhenius": {"intercepts": intercepts, "slopes": slopes, "residuals": residuals}}

  print("\n  what a batch can reach, with temperature held and with it free\n")
  print(f"  {'design space':28s} {'log det':>9s} {'kappa':>11s} {'settings':>9s} {'gap':>9s}")
  rows = {}
  for name, keep in [("temperature held at 23 C", [i for i, k in enumerate(labels)
                                                   if k[2] == REFERENCE_T]),
                     ("temperature free", list(range(len(labels))))]:
    subset = stack[keep]
    try:
      held = optimal_measure(subset, iterations=200_000)
    except ValueError as error:
      print(f"  {name:28s} refused: {error}")
      rows[name] = {"refused": str(error)}
      continue
    averaged = np.tensordot(held.weights, subset, axes=(0, 0))
    condition, logdet = _spectrum(averaged)
    support = [labels[keep[i]] for i in held.support]
    rows[name] = {"log_det": logdet, "kappa": condition, "gap": held.gap,
                  "support": [[s[0], s[1], s[2], float(held.weights[i])]
                              for s, i in zip(support, held.support, strict=True)]}
    print(f"  {name:28s} {logdet:9.3f} {condition:11.3e} {len(support):9d} {held.gap:9.1e}")
  summary["batches"] = rows

  for name in rows:
    if "support" not in rows[name]: continue
    print(f"\n  support of the '{name}' batch\n")
    print(f"  {'R_max (um)':>11s} {'stretch':>8s} {'T (C)':>7s} {'weight':>8s}")
    for radius, stretch, celsius, weight in sorted(rows[name]["support"], key=lambda r: -r[3]):
      print(f"  {radius * 1e6:11.0f} {stretch:8.2f} {celsius:7.1f} {weight:8.3f}")

  if all("kappa" in rows[n] for n in rows):
    held_k = rows["temperature held at 23 C"]["kappa"]
    free_k = rows["temperature free"]["kappa"]
    held_d = rows["temperature held at 23 C"]["log_det"]
    free_d = rows["temperature free"]["log_det"]
    print(f"\n  freeing temperature multiplies log det by exp({free_d - held_d:+.3f}) and "
          f"divides kappa by {held_k / free_k:.3f}")
    print("  Amplification alone would leave kappa exactly unchanged (sec:criteria), so a")
    print("  ratio near one says temperature is buying signal rather than rotation.")
    summary["kappa_ratio"] = float(held_k / free_k)
    summary["logdet_gain"] = float(free_d - held_d)

  json.dump(summary, open(records.HERE / "temperature_design.json", "w"), indent=1)


if __name__ == "__main__":
  main()
