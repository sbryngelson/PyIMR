r"""Screening the discrepancy against first-collapse physics, which it has never been screened against.

\Cref{sec:discrepancy} screens seven candidates and finds none removes more than a quarter of
$\hat\delta$, then says plainly that ``a screen cannot propose what it was not given''. Look at
what it was given: a second relaxation arm, two operators, two thermal treatments, mass transfer,
and the initial radius. Every one is a BULK constitutive, transport or configuration effect.

And look at where $\hat\delta$ lives: $65$ to $69$ percent of it is in the FIRST COLLAPSE.
\Cref{sec:enrich} even names the two things least secure there --- ``sphericity and the far-field
boundary'' --- and the word sphericity then appears nowhere else in this document. So the largest
part of the discrepancy has never been screened against the physics most likely to produce it.

Two candidates are added here, both first-collapse and neither on the existing list.

ASPHERICITY, through the shape-mode equation of \citet{plesset1954stability}. A perturbation
$R(\theta,t) = R(t)[1 + a_n(t) P_n(\cos\theta)]$ on a collapsing bubble obeys, to leading order
in an inviscid liquid with surface tension,
%
    a_n'' + 3 (R'/R) a_n' - A_n a_n = 0 ,
    A_n = (n-1) R''/R - (n-1)(n+1)(n+2) sigma / (rho R^3) ,

which is integrated along the fitted trajectory. What it does to the MEASURED radius depends on
what the imaging reports, and that is an assumption rather than a derivation, so both conventions
are screened: an equatorial or silhouette radius picks up $-a_2/2$ at first order, while a
volume-equivalent radius is unchanged at first order and picks up $O(a_2^2)$. They are different
curves and a screen that quoted only one would be hiding the choice.

AN INITIAL CONDITION THAT IS NOT AT REST. Every fit here starts the bubble at $R_{\max}$ with
zero wall velocity, which asserts that the imaged maximum is a turning point of a free collapse.
A laser deposits energy; the trace near the maximum need not be free. A small non-zero initial
wall velocity is a one-parameter direction and costs one solve.

WHAT THIS CANNOT DO. It is a screen, so it bounds what a candidate COULD remove with its
amplitude free --- an upper bound, not a forecast, exactly as \cref{eq:enrich} says of the
existing list. And a large number here would not establish that bubbles in these records were
aspherical; it would establish that asphericity is the first thing worth measuring.
"""

import json

import numpy as np

import records
from identified import BOX, candidate_at_ratio

RATIO = 38.5
AXES = ("mu", "galpha", "lambda1")
STARTS, EVALUATIONS = 16, 600
MODES = (2, 3)
SEED_AMPLITUDE = 1e-3            # the screen is scale free; this only sets the integration start
VELOCITIES = (-0.02, 0.02)       # of the Rayleigh velocity scale


def fitted_trace(dataset):
  """`(times, mean, spread, model, maximum, stretch, fitted)` for the incumbent at its optimum."""
  from pyimr.selection import evaluate_at, fit_candidate, physical_from_unit

  times, mean, spread, maximum, stretch = records.load(dataset)
  candidate = candidate_at_ratio(RATIO)
  solve = records.solver(times, maximum, stretch)
  fit = fit_candidate(candidate, solve, mean, spread, bounds=BOX, starts=STARTS,
                      max_evaluations=EVALUATIONS)
  fitted = dict(zip(AXES, (float(v) for v in physical_from_unit(AXES, fit.unit, BOX)),
                    strict=True))
  model = np.asarray(evaluate_at(candidate, solve, fitted)[0], dtype=float)
  return times, mean, spread, model, maximum, stretch, fitted, candidate


def shape_mode(times, radius_ratio, maximum, mode):
  """`a_n(t)` along a given trajectory, by Plesset's linear shape-mode equation.

  Integrated on a refined grid because the coefficients carry `R''`, which a 201-point
  record cannot resolve through the collapse; the result is interpolated back afterwards.
  """
  import pyimr
  from scipy.integrate import solve_ivp
  from scipy.interpolate import CubicSpline

  physics = pyimr.PhysicalParameters()
  rho, sigma = physics.medium_density_kg_m3, physics.surface_tension_n_m
  spline = CubicSpline(times, radius_ratio * maximum)
  first, second = spline.derivative(1), spline.derivative(2)

  def rhs(t, y):
    R = max(float(spline(t)), 1e-9 * maximum)
    growth = (mode - 1) * float(second(t)) / R \
        - (mode - 1) * (mode + 1) * (mode + 2) * sigma / (rho * R**3)
    return [y[1], growth * y[0] - 3.0 * float(first(t)) / R * y[1]]

  solved = solve_ivp(rhs, (times[0], times[-1]), [SEED_AMPLITUDE, 0.0], t_eval=times,
                     rtol=1e-8, atol=1e-14, method="LSODA")
  if not solved.success or not np.all(np.isfinite(solved.y[0])): return None
  return np.asarray(solved.y[0], dtype=float)


def initial_velocity_direction(dataset, fitted, candidate, speed):
  """The trace difference from starting the collapse with a small non-zero wall velocity."""
  import pyimr
  from pyimr.noise import characteristic_time

  times, mean, spread, maximum, stretch = records.load(dataset)
  scale = maximum / characteristic_time(maximum)

  def run(velocity):
    config = pyimr.SimulationConfig(
      maximum, maximum / stretch, candidate.build(fitted), dynamics="keller-miksis",
      rtol=1e-9, atol=1e-11, max_steps=600_000,
      initial=pyimr.InitialState(wall_velocity_m_s=velocity * scale))
    return np.asarray(pyimr.simulate(times, config).radius_ratio, dtype=float)

  try:
    return run(speed) - run(0.0)
  except Exception:                                                          # noqa: BLE001
    return None


def one(dataset):
  """Every first-collapse direction on one record, screened against that record's delta_hat."""
  import pyimr
  from pyimr.noise import discrepancy, enrichment_overlap
  from scipy.ndimage import median_filter

  times, mean, spread, model, maximum, stretch, fitted, candidate = fitted_trace(dataset)
  trials = records.trial_count(dataset)
  sigma = median_filter(spread, 11, mode="nearest") / np.sqrt(trials)
  residual = (model - mean) / sigma

  step = 1e-4
  point = np.array([fitted[a] for a in AXES])
  columns = []
  for k in range(len(AXES)):
    traces = []
    for direction in (1.0 + step, 1.0 - step):
      moved = dict(zip(AXES, (float(v) for v in point * np.where(
        np.arange(len(AXES)) == k, direction, 1.0)), strict=True))
      config = pyimr.SimulationConfig(maximum, maximum / stretch, candidate.build(moved),
                                      dynamics="keller-miksis", rtol=1e-9, atol=1e-11,
                                      max_steps=600_000)
      traces.append(np.asarray(pyimr.simulate(times, config).radius_ratio, dtype=float))
    columns.append((traces[0] - traces[1]) / (2.0 * step) / sigma)
  jacobian = np.column_stack(columns)

  got = discrepancy(residual, jacobian)
  directions, labels = [], []
  for mode in MODES:
    amplitude = shape_mode(times, model, maximum, mode)
    if amplitude is None: continue
    # silhouette: R(pi/2) = R (1 + a_n P_n(0)); P_2(0) = -1/2, P_3(0) = 0, so mode 3 has no
    # first-order equatorial signature and only the quadratic convention applies to it
    legendre = {2: -0.5, 3: 0.0}[mode]
    if legendre != 0.0:
      directions.append(legendre * amplitude * model / sigma)
      labels.append(f"asphericity n={mode} (silhouette)")
    directions.append(amplitude**2 * model / sigma)
    labels.append(f"asphericity n={mode} (volume equivalent)")
  for speed in VELOCITIES:
    delta = initial_velocity_direction(dataset, fitted, candidate, speed)
    if delta is None: continue
    directions.append(delta / sigma)
    labels.append(f"initial wall velocity {speed:+.2f}")

  candidates = {label: vector for label, vector in zip(labels, directions, strict=True)
                if np.all(np.isfinite(vector)) and np.any(vector)}
  if not candidates:
    return dataset, {"size": float(got.size), "scored": {}, "joint": 0.0, "blocked": []}
  # `directions` is a mapping and the discrepancy is the IDENTIFIABLE vector, not the raw
  # residual: the screen asks what a candidate can remove from what the fit could not absorb
  overlap = enrichment_overlap(got.identifiable, jacobian, candidates)
  scored = {k: float(v) for k, v in overlap.removable.items()}
  blocked = [k for k, ok in overlap.reachable.items() if not ok]
  return dataset, {"size": float(got.size), "scored": scored, "joint": float(overlap.joint),
                   "blocked": blocked, "first_collapse_share": float(
                     np.sum(residual[: int(0.35 * residual.size)] ** 2) / np.sum(residual**2))}


def main():
  summary = {}
  for dataset in records.DATASETS:
    name, got = one(dataset)
    summary[name] = got
  print("\n  what each first-collapse candidate could remove from delta_hat\n")
  every = sorted({k for v in summary.values() for k in v["scored"]})
  print(f"  {'candidate':40s} " + " ".join(f"{d:>13s}" for d in summary))
  for label in every:
    cells = " ".join(f"{summary[d]['scored'].get(label, float('nan')):13.1%}" for d in summary)
    print(f"  {label:40s} {cells}")
  print(f"\n  {'all together, on the cone':40s} "
        + " ".join(f"{summary[d]['joint']:13.1%}" for d in summary))
  print(f"  {'||delta_hat||':40s} " + " ".join(f"{summary[d]['size']:13.1f}" for d in summary))
  for d in summary:
    if summary[d]["blocked"]:
      print(f"  anti-aligned on {d}: {', '.join(summary[d]['blocked'])}")
  print(f"  {'share in the first collapse':40s} "
        + " ".join(f"{summary[d]['first_collapse_share']:13.1%}" for d in summary))
  print("\n  Against the existing list, whose best genuine-physics entry is thermal transport at")
  print("  6.5 percent and whose largest entry of any kind is the initial radius at 23.7.")
  json.dump(summary, open(records.HERE / "first_collapse_screen.json", "w"), indent=1)


if __name__ == "__main__":
  main()
