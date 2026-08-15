r"""Is the fitted inward $u_0$ the density mismatch, or the physics?

\#273 records two liquid densities in one calculation. `pyimr/noise.py` builds the record's
time axis with $\rho = \SI{998}{\kilo\gram\per\metre\cubed}$ through `characteristic_time`,
while `pyimr/_config.py` integrates the equation of motion at $1064$. The sampling times handed
to the solver are therefore $\sqrt{1064/998} = 1.0325$ times shorter than the solver's own
inertial scale --- the data's collapse is asked for $3.25\%$ before the model's own.

WHAT THAT PREDICTS, AND WHY IT IS FALSIFIABLE. The fit has no clock parameter, so it can only
repay a timing deficit through the initial condition: an inward wall velocity makes the model
collapse earlier. Near the maximum the nondimensional dynamics give $\dot R^* \simeq -t^*$, so
over a collapse at $0.915\,t_c$ a $3.25\%$ deficit is worth about $0.030$ in Rayleigh units.
\Cref{sec:discrepancy} fits $u_0$ at $-0.022$ to $-0.033$, always inward, on every record, and
unchanged at $\pm10\%$ and $\pm30\%$ prior width --- and rules out frame sampling because half a
frame of the model's own deceleration is $-0.007$ to $-0.008$, three to four times too small.
The unexplained factor of three to four is the size of this mismatch, which is what makes the
identification worth testing rather than asserting.

THERE ARE TWO WAYS TO UNIFY AND THEY ARE NOT THE SAME EXPERIMENT. The mismatch can be closed by
moving the clock to the dynamics' density, or by moving the dynamics to the clock's. Only the
first is a pure rescale of the sampling times; the second changes the equation of motion, since
$\rho$ sets the Weber and Mach groups while $p_\infty$, $\sigma$ and $c_\infty$ stay fixed. Both
are run here, against the published mismatched pair, because they answer different questions:
the first asks whether the mismatch produced the inward velocity, and the second asks which
density the records actually prefer.

WHICH DENSITY IS RIGHT IS PART OF THE SECOND QUESTION. $1064$ is the polyacrylamide value
inherited from IMRv2; $5\%$ gelatin is nearer $1015$; $998$ is water, and is what the record's
own clock already uses. All three are swept as consistent pairs, so the comparison is between
liquids rather than between a liquid and a bookkeeping error.

THE STRONGEST ROW IS THE ONE WITH NOTHING FREE. A free parameter always buys $\chi^2$, which is
why \cref{sec:reqprior} reads lack of fit instead. A density change is not a free parameter: if
unifying improves \eqref{eq:lackoffit} with no new freedom at all, it is a correction rather than
a reparameterisation. That row is the one to read first, and the published pair reproduces
`initial_condition.json` exactly, which is the check that the harness is measuring what it says.
"""

import json

import numpy as np

import records
from identified import BOX

RATIO = 38.5
AXES = ("mu", "galpha", "lambda1")
CLOCK_DENSITY = 998.0                    # pyimr/noise.py:49, builds the record's time axis
DYNAMICS_DENSITY = 1064.0                # pyimr/_config.py:60, integrates the motion
GELATIN_DENSITY = 1015.0                 # 5% gelatin, the material these records are
STARTS, EVALUATIONS = 32, 800            # as sec:discrepancy's own fit, for comparability
REQ_BOX = (0.6, 1.6)
VELOCITY_BOX = (0.70, 1.0 / 0.70)        # the widest sec:discrepancy swept; nothing pinned there
CONFIGURATIONS = [(), ("u0_shift",), ("req_scale",), ("req_scale", "u0_shift")]

# (label, clock density, dynamics density). The first is the mismatch as shipped; the rest
# close it, one by moving the clock and two by moving the liquid.
CASES = [
  ("published", CLOCK_DENSITY, DYNAMICS_DENSITY),
  ("clock->1064", DYNAMICS_DENSITY, DYNAMICS_DENSITY),
  ("both@1015", GELATIN_DENSITY, GELATIN_DENSITY),
  ("both@998", CLOCK_DENSITY, CLOCK_DENSITY),
]


def solver_with_start(times, maximum, stretch, clock_density, dynamics_density):
  """`solve(material, config_axes)` at one clock and one liquid.

  The Rayleigh velocity scale is taken at the CLOCK density, since that is the scale the
  record's own nondimensional time is expressed in and therefore the one `u0_shift` has to be
  read against for the columns to be comparable.
  """
  import pyimr
  from pyimr.noise import characteristic_time

  velocity_scale = maximum / characteristic_time(maximum, density=clock_density)
  physics = pyimr.PhysicalParameters(medium_density_kg_m3=dynamics_density)

  def solve(material, config_axes=None):
    axes = config_axes or {}
    scale = float(axes.get("req_scale", 1.0))
    shift = float(axes.get("u0_shift", 1.0))
    config = pyimr.SimulationConfig(
      maximum, maximum / stretch * scale, material, dynamics="keller-miksis",
      rtol=1e-8, atol=1e-10, max_steps=400_000, physics=physics,
      initial=pyimr.InitialState(wall_velocity_m_s=(shift - 1.0) * velocity_scale))
    return np.asarray(pyimr.simulate(times, config).radius_ratio, dtype=float), None

  return solve


def candidate_for(free):
  """The identified qSLS with whichever configuration axes `free` names."""
  import pyimr
  from pyimr.selection import CandidateModel

  def build(t):
    product = t["galpha"]
    return pyimr.QuadraticZener(float(np.sqrt(product * RATIO)), t["mu"], t["lambda1"], 0.0,
                                float(np.sqrt(product / RATIO)))

  extra = tuple(free)
  return CandidateModel(f"qSLS+{'+'.join(extra) or 'none'}", build, (*AXES, *extra), (), extra)


def _box(free):
  box = dict(BOX)
  if "req_scale" in free: box["req_scale"] = REQ_BOX
  if "u0_shift" in free: box["u0_shift"] = VELOCITY_BOX
  return box


def one(job):
  """One fit: a record, a set of free configuration axes, and a (clock, liquid) pair."""
  from pyimr.noise import lack_of_fit
  from pyimr.selection import evaluate_at, fit_candidate, physical_from_unit

  dataset, free, label = job
  clock_density, dynamics_density = next((c, d) for name, c, d in CASES if name == label)
  times, mean, spread, maximum, stretch = records.load(dataset)
  times = times * float(np.sqrt(clock_density / CLOCK_DENSITY))
  candidate = candidate_for(free)
  solve = solver_with_start(times, maximum, stretch, clock_density, dynamics_density)
  box = _box(free)
  try:
    fit = fit_candidate(candidate, solve, mean, spread, bounds=box, starts=STARTS,
                        max_evaluations=EVALUATIONS)
  except ValueError as error:
    return job, {"failed": str(error)}
  axes = tuple(candidate.axes)
  fitted = dict(zip(axes, (float(v) for v in physical_from_unit(axes, fit.unit, box)), strict=True))
  model = np.asarray(evaluate_at(candidate, solve, fitted)[0], dtype=float)
  ratio = lack_of_fit(mean, model, spread, records.trial_count(dataset), candidate.dimension).ratio
  pinned = [k for k in free
            if min(abs(np.log(fitted[k] / box[k][0])), abs(np.log(fitted[k] / box[k][1]))) < 1e-3]
  return job, {"chi2_per_n": float(fit.chi_squared), "lack_of_fit": float(ratio),
               "fitted": fitted, "pinned": pinned,
               "u0": float(fitted.get("u0_shift", 1.0) - 1.0)}


def main():
  labels = [name for name, _, _ in CASES]
  jobs = [(d, free, label) for d in records.DATASETS for free in CONFIGURATIONS
          for label in labels]
  print(f"  the clock uses rho = {CLOCK_DENSITY:.0f} and the dynamics rho = {DYNAMICS_DENSITY:.0f}, "
        f"a factor of {np.sqrt(DYNAMICS_DENSITY / CLOCK_DENSITY):.4f} "
        f"({100 * (np.sqrt(DYNAMICS_DENSITY / CLOCK_DENSITY) - 1):.2f}%).")
  print(f"  if that produced the inward velocity, closing it should return u0 to zero from "
        f"{-0.915 * (np.sqrt(DYNAMICS_DENSITY / CLOCK_DENSITY) - 1):.4f}.\n")
  print(f"  {len(jobs)} fits at {STARTS} restarts ...", flush=True)

  with records.pool(len(jobs)) as pool:
    got = dict(pool.map(one, jobs))

  summary = {"cases": {n: {"clock": c, "dynamics": d} for n, c, d in CASES}, "records": {}}
  for metric, digits in (("lack_of_fit", 2), ("chi2_per_n", 3), ("u0", 4)):
    print(f"\n  ==== {metric} ====")
    for dataset in records.DATASETS:
      print(f"\n  {dataset}")
      print(f"  {'free':>22s}  " + " ".join(f"{n:>11s}" for n in labels))
      for free in CONFIGURATIONS:
        cells = []
        for label in labels:
          v = got[(dataset, free, label)]
          cells.append(f"{'failed':>11s}" if "failed" in v else f"{v[metric]:11.{digits}f}")
        print(f"  {'+'.join(free) or 'nothing':>22s}  " + " ".join(cells))

  for dataset in records.DATASETS:
    summary["records"][dataset] = {
      "+".join(free) or "nothing": {label: got[(dataset, free, label)] for label in labels}
      for free in CONFIGURATIONS}

  print("\n  ---- what the rows say ----\n")
  verdicts = {}
  base = "published"
  for label in labels[1:]:
    nothing = [(summary["records"][d]["nothing"][base], summary["records"][d]["nothing"][label])
               for d in records.DATASETS]
    better = sum(u["lack_of_fit"] < p["lack_of_fit"] for p, u in nothing)
    velocity = [(summary["records"][d]["u0_shift"][base], summary["records"][d]["u0_shift"][label])
                for d in records.DATASETS]
    shrunk = sum(abs(u["u0"]) < 0.5 * abs(p["u0"]) for p, u in velocity)
    verdicts[label] = {"lack_of_fit_better": int(better), "u0_halved": int(shrunk),
                       "u0": [float(u["u0"]) for _, u in velocity]}
    print(f"  {label:>12s}: lack of fit improves on {better} of 3 records with nothing free; "
          f"u0 halves on {shrunk} of 3")
    print(f"  {'':>12s}  u0 = " + np.array2string(np.array([u["u0"] for _, u in velocity]),
                                                  precision=4))
  summary["verdicts"] = verdicts
  print("\n  published u0 = " + np.array2string(
    np.array([summary["records"][d]["u0_shift"][base]["u0"] for d in records.DATASETS]),
    precision=4))
  print("\n  A mechanism that produced the inward velocity would both remove it and improve the")
  print("  fit with nothing free. A case that improves neither is not the cause; a case that")
  print("  improves the fit is the liquid the records prefer, whatever u0 does.")

  json.dump(summary, open(records.HERE / "density_clock.json", "w"), indent=1)


if __name__ == "__main__":
  main()
