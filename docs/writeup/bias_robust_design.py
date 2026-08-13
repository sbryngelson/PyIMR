r"""Designing when the parameters the design is computed at are known to be wrong.

\Cref{sec:measure} certifies a measure globally optimal in $\xi$ and then states the limitation
plainly: the information matrices are evaluated at a point estimate of $\theta$, so the design is
only \emph{locally} optimal in the parameters. The standard repair is offered --- average over a
prior --- and never taken, because there was no prior to average over.

\Cref{sec:discrepancy} supplies something stronger than a prior. \Cref{eq:biasbound} bounds how
far model-form error alone can move each parameter: $B\sigma_k$ with the same multiple for every
coordinate, which on the \SI{15}{\celsius} record is $\times1.58$ in $\mu$, $\times1.30$ in
$g\alpha$ and $\times3.11$ in $\lambda_1$. That is not a guess about where $\theta$ might be. It
is a computed set containing the truth unless the unseen half of the discrepancy is larger than
the seen half.

SO THE DESIGN CAN BE MADE ROBUST TO IT, AND THE CERTIFICATE SURVIVES TWICE OVER. Averaging over
the set keeps $M(\xi)$ linear in $\xi$, so \eqref{eq:equivalence} holds unchanged. Taking the
worst case over the set is a pointwise minimum of concave functions, which is concave, so it
holds there too. Neither robustification costs the proof --- the same fact that let
\cref{sec:setpoint} absorb the setpoint scatter.

WHAT IS MEASURED. The nominal design --- certified at the fit, which is what this chapter
recommends --- evaluated across the bias set: its worst case and its average. Against a design
built on the average and one built on the worst case, evaluated the same way. The difference is
what ignoring \eqref{eq:biasbound} costs a design, and it is the last of the three limitations
\cref{sec:measure} lists that can be answered without new data.

THE SET IS THE BOUND, NOT A DISTRIBUTION. \Cref{eq:biasbound} gives a ball, not a density, so its
corners are what a maximin sees and its uniform average is the least committal thing to average
over. Both are reported, because they are different questions and the chapter has been careful
elsewhere to say which one it is asking.
"""

import json
import itertools

import numpy as np

import records
from identified import candidate_at_ratio
from noise_design import profile

RATIO = 38.5
AXES = ("mu", "galpha", "lambda1")
SOURCE = "gelatin_15C"
SAMPLES = 201
# eq:biasbound at 15 C, as sec:discrepancy tabulates it
BIAS = {"mu": 1.58, "galpha": 1.30, "lambda1": 3.11}
RADII = np.geomspace(50e-6, 1200e-6, 8)
STRETCH = np.linspace(3.0, 20.0, 6)


def corners():
  """The fit, plus the corners of the bias box: what a worst case actually sees."""
  points = [dict.fromkeys(AXES, 1.0)]
  for signs in itertools.product((-1.0, 1.0), repeat=len(AXES)):
    points.append({a: BIAS[a] ** sign for a, sign in zip(AXES, signs, strict=True)})
  return points


def one(job):
  """Whitened material information at one geometry and one point of the bias box."""
  import pyimr
  from pyimr.noise import characteristic_time

  radius, stretch, offsets = job
  measured = json.load(open(records.HERE / "per_trial_fits.json"))[SOURCE]
  point = np.array([float(measured["median"][a]) * offsets[a] for a in AXES])
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
    return job[:2] + (tuple(sorted(offsets.items())),), None
  jacobian = np.column_stack(columns)
  if not np.all(np.isfinite(jacobian)): return job[:2] + (tuple(sorted(offsets.items())),), None

  tau, spread = profile(SOURCE)
  phase = (times - times[0]) / characteristic_time(radius)
  whitened = jacobian / np.interp(phase, tau, spread)[:, None]
  prior = 1.0 / np.sqrt(12.0)
  matrix = whitened.T @ whitened * prior**2
  return job[:2] + (tuple(sorted(offsets.items())),), 0.5 * (matrix + matrix.T)


def main():
  from pyimr.measure import optimal_measure

  box = corners()
  geometries = [(float(r), float(s)) for r in RADII for s in STRETCH]
  jobs = [(r, s, offsets) for r, s in geometries for offsets in box]
  cache = records.HERE / "bias_robust_jacobians.npz"
  if cache.exists():
    stored = np.load(cache, allow_pickle=True)
    got = {tuple(k): (None if v is None else np.asarray(v))
           for k, v in zip(stored["keys"], stored["values"], strict=True)}
    print(f"  {len(got)} Jacobians read from cache")
  else:
    print(f"  {len(geometries)} geometries x {len(box)} points of the bias box "
          f"= {len(jobs)} Jacobians ...", flush=True)
    with records.pool(len(jobs)) as pool:
      got = dict(pool.map(one, jobs))
    np.savez(cache, keys=np.array(list(got), dtype=object),
             values=np.array(list(got.values()), dtype=object))

  keys = [tuple(sorted(o.items())) for o in box]
  usable = [g for g in geometries
            if all(got.get((g[0], g[1], k)) is not None for k in keys)]
  print(f"  {len(usable)} of {len(geometries)} geometries integrate at every corner")
  if len(usable) < 8:
    print("  too few to design over")
    return
  cube = np.array([[got[(g[0], g[1], k)] for k in keys] for g in usable])   # (geom, corner, p, p)
  identity = np.eye(cube.shape[-1])

  def value(weights, corner):
    averaged = np.tensordot(weights, cube[:, corner], axes=(0, 0))
    return 0.5 * float(np.linalg.slogdet(identity + averaged)[1])

  # The maximin fits the existing machinery through a block trick: stack each candidate's
  # matrices at all corners block-diagonally, so M(xi) block c is sum_i xi_i cube[i, c], and let
  # the criterion return the minimum over blocks. A pointwise minimum of concave functions is
  # concave, so `optimal_measure`'s certificate applies unchanged.
  corners_count, size = cube.shape[1], cube.shape[-1]
  blocked = np.zeros((cube.shape[0], corners_count * size, corners_count * size))
  for c in range(corners_count):
    lo, hi = c * size, (c + 1) * size
    blocked[:, lo:hi, lo:hi] = cube[:, c]

  def blockwise(reduce_):
    """A criterion over the blocks: `reduce_` is `min` for maximin or `mean` for Bayesian D."""
    def criterion(information):
      values, inverses = [], []
      for c in range(corners_count):
        lo, hi = c * size, (c + 1) * size
        block = identity + information[lo:hi, lo:hi]
        values.append(0.5 * float(np.linalg.slogdet(block)[1]))
        inverses.append(0.5 * np.linalg.inv(block))
      gradient = np.zeros_like(information)
      if reduce_ == "min":
        pick = int(np.argmin(values))
        lo, hi = pick * size, (pick + 1) * size
        gradient[lo:hi, lo:hi] = inverses[pick]
        return values[pick], gradient
      for c in range(corners_count):
        lo, hi = c * size, (c + 1) * size
        gradient[lo:hi, lo:hi] = inverses[c] / corners_count
      return float(np.mean(values)), gradient
    return criterion

  # `log det E[M]` and `E[log det M]` are not the same criterion -- Jensen -- and the first
  # version of this optimised the former while scoring the latter, which is why the averaged
  # design came out WORSE on the average than the nominal one. Both reductions go through the
  # block stack instead, so each is optimising the column it is reported in.
  builds = {
    "nominal (at the fit)": (cube[:, 0], None),
    "Bayesian, max E[log det]": (blocked, blockwise("mean")),
    "maximin, max min log det": (blocked, blockwise("min")),
  }

  summary = {}
  print(f"\n  {'design built on':>26s} {'settings':>9s} {'gap':>9s} {'at the fit':>11s} "
        f"{'average':>9s} {'worst corner':>13s}")
  for label, (built, criterion) in builds.items():
    held = optimal_measure(built, criterion=criterion, iterations=200_000)
    across = [value(held.weights, c) for c in range(corners_count)]
    summary[label] = {"settings": int(held.support.size), "gap": held.gap,
                      "at_fit": across[0], "average": float(np.mean(across)),
                      "worst": float(np.min(across)),
                      "support": [[usable[i][0], usable[i][1], float(held.weights[i])]
                                  for i in held.support]}
    print(f"  {label:>26s} {held.support.size:9d} {held.gap:9.1e} {across[0]:11.3f} "
          f"{np.mean(across):9.3f} {np.min(across):13.3f}")

  print("\n  a design at the fit, run against a truth anywhere in the bias box, loses\n")
  nominal = summary["nominal (at the fit)"]
  averaged = summary["Bayesian, max E[log det]"]
  robust = summary["maximin, max min log det"]
  print(f"  worst case: {nominal['worst']:.3f} at the fit, {averaged['worst']:.3f} averaged, "
        f"{robust['worst']:.3f} maximin -> {robust['worst'] - nominal['worst']:+.3f} nats")
  print(f"  on average: {nominal['average']:.3f} at the fit, {averaged['average']:.3f} averaged "
        f"-> {averaged['average'] - nominal['average']:+.3f} nats")
  summary["regret_worst"] = robust["worst"] - nominal["worst"]
  summary["regret_average"] = averaged["average"] - nominal["average"]

  print("\n  and how far the bias box moves the answer at all\n")
  print(f"  {'corner':>34s} {'best single geometry':>24s} {'nats':>8s}")
  spread = {}
  for corner, key in enumerate(keys):
    singles = [0.5 * float(np.linalg.slogdet(identity + cube[i, corner])[1])
               for i in range(len(usable))]
    top = int(np.argmax(singles))
    label = ",".join(f"{a}{'+' if v > 1 else ('-' if v < 1 else '0')}" for a, v in key)
    spread[label] = {"geometry": list(usable[top]), "nats": singles[top]}
    print(f"  {label:>34s} {usable[top][0] * 1e6:8.0f} um at {usable[top][1]:5.2f} "
          f"{singles[top]:8.3f}")
  summary["per_corner_best"] = spread
  distinct = {tuple(v["geometry"]) for v in spread.values()}
  print(f"\n  {len(distinct)} distinct geometries preferred across {len(keys)} corners of the box.")
  print("  One means the bias bound does not reach the design; several mean it does, and the")
  print("  averaged or maximin form is the one to report rather than the point estimate.")

  json.dump(summary, open(records.HERE / "bias_robust_design.json", "w"), indent=1)


if __name__ == "__main__":
  main()
