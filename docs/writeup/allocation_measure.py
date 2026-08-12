r"""The bubbles-against-frames split, inside the certified optimisation instead of beside it.

\Cref{sec:allocation} measures the largest design effect in this document --- $+3.4$ to $+4.0$
nats for re-splitting a fixed frame budget, against $\le 1.34$ for choosing the geometry --- and
then declines to fold it in, on the grounds that $\Sigma_\theta$ at an unrun geometry is
unknown. So the one variable worth optimising is the one the optimisation is not allowed to
move.

IT FOLDS IN EXACTLY, and the certificate survives. Let a candidate be a PAIR $(x, N)$: a
geometry and a number of samples per trace. Let $\xi$ allocate TRIALS rather than runs. Then

    F = sum_i J_i (Sigma + F_{N_i}(x_i)^{-1})^{-1} = J sum_i xi_i M_i ,

still linear in $\xi$, so $\log\det$ is still concave and Kiefer--Wolfowitz still applies. What
changes is that the candidates are no longer equally expensive: a trace of $N$ samples costs $N$
frames, and a measure counting runs would be answering how to spend a fixed number of traces
rather than how to spend a camera. `pyimr.measure.budgeted_measure` is the repair --- normalise
by cost rather than by count, which is the same simplex seen through $\eta_i = \xi_i c_i$, so the
certificate reads `tr[M^-1 M(x)] / c_x <= p` and is inherited rather than re-derived.

TWO PRICES ARE SWEPT RATHER THAN CHOSEN, because neither is knowable from here. The cost of a
candidate is `bubble + N` in frame-equivalents, and `bubble` is what one laser shot, one
alignment and one nucleation are worth against one frame. A conclusion that holds across three
decades of it is a conclusion; one that moves inside it is a statement about the price list.

$\Sigma_\theta$ IS THE ASSUMPTION, and it is weaker than \cref{sec:allocation} implies. It is a
population covariance of MATERIAL parameters --- a property of the gel and of how repeatably the
laser nucleates, not of how big the bubble is --- and the three records that measure it differ in
TEMPERATURE, not in geometry: they sit at $277$, $298$ and $312$ \si{\micro\metre}. So there is
no evidence of geometry dependence because geometry was never varied. That makes it an
assumption to state and sweep, not an extrapolation to refuse; the sweep is $\times\tfrac12$,
$\times 1$, $\times 2$ and the question is whether the recommended $N$ moves.
"""

import json

import numpy as np

import records
from identified import candidate_at_ratio
from noise_design import profile

RATIO = 38.5
AXES = ("mu", "galpha", "lambda1")
SOURCE = "gelatin_15C"
# the geometries of `design_operator`, thinned: the pairing multiplies the candidate count
RADII = np.geomspace(50e-6, 1200e-6, 10)
STRETCH = np.linspace(3.0, 20.0, 7)
# spans BELOW the record's own 201 far enough to see whether the optimum is interior. It is
# not, which is the finding: N_eff is near ten of 201 (sec:limitations), so the bottom of this
# grid is a validity limit rather than a maximum
COUNTS = (6, 8, 12, 25, 50, 100, 201, 400)         # samples per trace
SETTINGS = 5                                       # the floor sec:gain's testability needs
BUBBLE_PRICES = (0.0, 25.0, 100.0, 400.0)          # one bubble, in frames
SIGMA_SCALES = (0.5, 1.0, 2.0)
REFERENCE = 201                                    # where the Jacobian is computed


def _material():
  """The 15 C fit in the identified coordinates, as `per_trial_fits.py` left it."""
  measured = json.load(open(records.HERE / "per_trial_fits.json"))[SOURCE]
  return {a: float(measured["median"][a]) for a in AXES}, np.array(measured["log_covariance"])


def one(job):
  """`F_N` at one geometry, in the log-identified coordinates and at `REFERENCE` samples."""
  import pyimr
  from pyimr.noise import characteristic_time

  radius, stretch = job
  fitted, _ = _material()
  point = np.array([fitted[a] for a in AXES])
  candidate = candidate_at_ratio(RATIO)
  times = np.linspace(0.0, 5.0 * characteristic_time(radius), REFERENCE)

  def trace(scale):
    moved = dict(zip(AXES, (float(v) for v in point * scale), strict=True))
    config = pyimr.SimulationConfig(radius, radius / stretch, candidate.build(moved),
                                    dynamics="keller-miksis", rtol=1e-8, atol=1e-10,
                                    max_steps=200_000)
    return np.asarray(pyimr.simulate(times, config).radius_ratio, dtype=float)

  step = 1e-4
  try:
    columns = []
    for k in range(len(AXES)):
      up = trace(np.where(np.arange(len(AXES)) == k, 1 + step, 1.0))
      down = trace(np.where(np.arange(len(AXES)) == k, 1 - step, 1.0))
      columns.append((up - down) / (2.0 * step))   # d/dlog(theta_k), the log coordinate
    jacobian = np.column_stack(columns)
  except Exception:                                                          # noqa: BLE001
    return job, None
  if not np.all(np.isfinite(jacobian)): return job, None

  # the measured spread transferred by phase, as sec:noisedesign justifies -- not the flat
  # 0.018, which that section shows the records contradict by three orders of magnitude
  tau, spread = profile(SOURCE)
  phase = (times - times[0]) / characteristic_time(radius)
  whitened = jacobian / np.interp(phase, tau, spread)[:, None]
  matrix = whitened.T @ whitened
  return job, 0.5 * (matrix + matrix.T)


def _pairs(fisher, covariance, counts):
  """`(labels, matrices, sample counts)` for every (geometry, N) pair that is usable."""
  labels, matrices, samples = [], [], []
  for (radius, stretch), matrix in fisher.items():
    for count in counts:
      scaled = matrix * (count / REFERENCE)        # F_N linear in N; see the caveats below
      try:
        bracket = covariance + np.linalg.inv(scaled)
        per_trial = np.linalg.inv(bracket)
      except np.linalg.LinAlgError:
        continue
      labels.append((radius, stretch, count))
      matrices.append(0.5 * (per_trial + per_trial.T))
      samples.append(count)
  return labels, np.array(matrices), np.array(samples, dtype=float)


def _solve(labels, matrices, samples, bubble, budget, settings=None):
  """The certified allocation at one price, reported per unit of frame budget."""
  from pyimr.measure import budgeted_measure

  costs = bubble + samples
  held = budgeted_measure(matrices, costs, settings=settings, iterations=200_000)
  # log det of the information a whole budget buys: J scales the matrix, so p log(budget) is
  # the only thing the unit choice contributes
  value = held.value + matrices.shape[1] * np.log(budget)
  chosen = [(labels[i], held.weights[i], held.budget_share[i]) for i in held.support]
  return held, value, sorted(chosen, key=lambda row: -row[1])


def main():
  fitted, covariance = _material()
  designs = [(float(r), float(s)) for r in RADII for s in STRETCH]
  print(f"  {len(designs)} geometries x {len(COUNTS)} sample counts ...", flush=True)
  with records.pool(len(designs)) as pool:
    got = dict(pool.map(one, designs))
  fisher = {k: v for k, v in got.items() if v is not None}
  print(f"  {len(fisher)} of {len(designs)} geometries integrate")

  budget = 18 * 201                                # the 15 C record's own frame count
  summary = {"budget_frames": budget, "source": SOURCE}

  print(f"\n  joint (geometry, N) allocation at a budget of {budget} frames\n")
  print(f"  {'bubble price':>13s} {'nats':>8s} {'gap':>9s} {'settings':>9s} {'N used':>18s} "
        f"{'trials':>8s}")
  joint = {}
  labels, matrices, samples = _pairs(fisher, covariance, COUNTS)
  for bubble in BUBBLE_PRICES:
    held, value, chosen = _solve(labels, matrices, samples, bubble, budget)
    used = sorted({int(row[0][2]) for row in chosen})
    trials = held.runs_per_unit_cost * budget       # sum_i xi_i c_i = 1, so runs scale with it
    joint[bubble] = {"nats": value, "gap": held.gap, "certified": held.certified,
                     "settings": len(chosen), "counts": used, "trials": trials,
                     "support": [[row[0][0], row[0][1], row[0][2], row[1]] for row in chosen]}
    print(f"  {bubble:13.0f} {value:8.3f} {held.gap:9.1e} {len(chosen):9d} "
          f"{str(used):>18s} {trials:8.1f}")
  summary["joint"] = {str(k): v for k, v in joint.items()}

  print("\n  what the document's formulation gets: geometry only, N pinned at 201\n")
  print(f"  {'bubble price':>13s} {'nats':>8s} {'gap':>9s} {'settings':>9s} {'nats given up':>14s}")
  pinned = {}
  plabels, pmatrices, psamples = _pairs(fisher, covariance, (REFERENCE,))
  for bubble in BUBBLE_PRICES:
    held, value, chosen = _solve(plabels, pmatrices, psamples, bubble, budget)
    pinned[bubble] = {"nats": value, "gap": held.gap, "settings": len(chosen)}
    print(f"  {bubble:13.0f} {value:8.3f} {held.gap:9.1e} {len(chosen):9d} "
          f"{joint[bubble]['nats'] - value:14.3f}")
  summary["pinned"] = {str(k): v for k, v in pinned.items()}

  print("\n  is the answer a statement about Sigma? sweep it and see whether N moves\n")
  print(f"  {'Sigma x':>8s} {'bubble price':>13s} {'nats':>8s} {'N used':>18s}")
  sweep = {}
  for scale in SIGMA_SCALES:
    slabels, smatrices, ssamples = _pairs(fisher, scale * covariance, COUNTS)
    for bubble in (0.0, 100.0):
      held, value, chosen = _solve(slabels, smatrices, ssamples, bubble, budget)
      used = sorted({int(row[0][2]) for row in chosen})
      sweep[f"{scale}|{bubble}"] = {"nats": value, "counts": used}
      print(f"  {scale:8.2f} {bubble:13.0f} {value:8.3f} {str(used):>18s}")
  summary["sigma_sweep"] = sweep

  print(f"\n  held to {SETTINGS} settings, so the batch can still detect that the model is wrong\n")
  print(f"  {'bubble price':>13s} {'nats':>8s} {'gap':>9s} {'settings':>9s} {'N used':>18s} "
        f"{'nats lost':>10s}")
  floored = {}
  for bubble in BUBBLE_PRICES:
    held, value, chosen = _solve(labels, matrices, samples, bubble, budget, settings=SETTINGS)
    used = sorted({int(row[0][2]) for row in chosen})
    floored[bubble] = {"nats": value, "gap": held.gap, "certified_under_floor": held.certified,
                       "settings": len(chosen), "counts": used,
                       "support": [[row[0][0], row[0][1], row[0][2], row[1]] for row in chosen]}
    print(f"  {bubble:13.0f} {value:8.3f} {held.gap:9.1e} {len(chosen):9d} "
          f"{str(used):>18s} {joint[bubble]['nats'] - value:10.3f}")
  summary["floored"] = {str(k): v for k, v in floored.items()}

  print(f"\n  the batch, at bubble = 100 and {SETTINGS} settings\n")
  print(f"  {'R_max (um)':>11s} {'stretch':>8s} {'N':>5s} {'run share':>10s} {'budget share':>13s}")
  rows = floored[100.0]["support"]
  spend = sum(r[3] * (100.0 + r[2]) for r in rows)
  for row in rows:
    print(f"  {row[0] * 1e6:11.0f} {row[1]:8.2f} {int(row[2]):5d} {row[3]:10.3f} "
          f"{row[3] * (100.0 + row[2]) / spend:13.3f}")

  low = min(COUNTS)
  pinned_low = [b for b in BUBBLE_PRICES if min(joint[b]["counts"]) <= low]
  print(f"\n  N pins at the bottom of the grid ({low}) at {len(pinned_low)} of "
        f"{len(BUBBLE_PRICES)} prices. Where it does, the binding limit is validity, not")
  print("  information: F_N is scaled LINEARLY in N here and N_eff is near ten of 201, so")
  print("  below that the linear model is claiming resolution the record does not have.")
  summary["pinned_at_grid_floor"] = pinned_low

  json.dump(summary, open(records.HERE / "allocation_measure.json", "w"), indent=1)


if __name__ == "__main__":
  main()
