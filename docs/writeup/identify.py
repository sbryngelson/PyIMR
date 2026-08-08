"""A batch of experiments designed to identify the constitutive model.

`selection.tex` derives the compound criterion of Atkinson and of Cook and Wong -- estimation
and discrimination in one concave objective, certified by the same equivalence theorem -- and
never runs it. This runs it, on the model pair that actually binds.

WHICH PAIR. The T-optimality table puts qSLS and SLS closest: at the geometry performed, SLS
imitates qSLS to within 0.67 noise units, comfortably inside the scatter. Stiffening and
relaxation are the two ingredients the whole comparison turns on, and this is the pair that
asks whether the stiffening is there at all. Every other rival is easier.

WHY THE RECORD IS SUBSAMPLED AND THE PRIOR IS TIGHT. `selection.tex` shows prior sampling of
the evidence fails outright on a 201-frame record: useful draws go as `S N^(-p/2)`, giving
fewer than one, and the effective sample size is 1.0 at every design with reported values
between 793 and 10979 nats that are pure noise. `sampling.py` independently finds the record
about twenty times redundant, so shortening it costs little. Measured here, at this design:

    samples  log-spread  bank   S_eff
      24        1.20       96    1.00   <- the wide box, unusable, as documented
      24        0.25       96    2.17
       8        0.25       96    7.61
      16        0.35      768    7.55   <- used

The prior is therefore a lognormal about the fitted material rather than a decades-wide box,
which also makes this the question a collaborator actually has: given the records already in
hand, where should the NEXT batch go? A design scored under a prior nobody holds is answering
a different question, and answering it unreliably.

WHAT THIS DOES NOT DO. The utility is an expected log Bayes factor between two models we have
already measured to be inadequate -- `lackoffit.py` rejects qSLS on every record. A criterion
computed under a model the data reject prices information about a process that did not
generate the data, and `sloppy_design.py` shows which geometry that flatters depends on which
physics is missing. So this is the best available answer to "where should the batch go", not a
claim that the batch will settle the question.
"""

import json

import numpy as np

import records

# the design plane the measure work uses, log-spaced in radius because the physics is
RADII = np.geomspace(50e-6, 1200e-6, 8)
STRETCHES = np.linspace(3.0, 20.0, 6)
DESIGNS = [(float(r), float(s)) for r in RADII for s in STRETCHES]
SAMPLES = 16
RELATIVE_NOISE = 0.02
BANK = 768
RUNS = 12
# a lognormal prior about the fit rather than a decades-wide box: see the module docstring
PRIOR_SPREAD = 0.35
FIT = {"g": 204.3, "mu": 0.04651, "lambda1": 1.964e-7, "alpha": 5.301}
PATHS = ("material.shear_modulus_pa", "material.viscosity_pa_s",
         "material.relaxation_time_s", "material.stiffening")


def _times(radius):
  from pyimr.noise import characteristic_time

  return np.linspace(0.0, 5.0 * characteristic_time(radius), SAMPLES)


def _bank(design, stiffening, seed):
  """Trajectories from one model's prior at one design. `stiffening=False` is SLS."""
  import pyimr

  radius, stretch = design
  times, rng, rows = _times(radius), np.random.default_rng(seed), []
  for _ in range(BANK):
    draw = {k: v * float(np.exp(rng.normal(0.0, PRIOR_SPREAD))) for k, v in FIT.items()}
    material = (pyimr.QuadraticZener(draw["g"], draw["mu"], draw["lambda1"], 0.0, draw["alpha"])
                if stiffening else pyimr.Zener(draw["g"], draw["mu"], draw["lambda1"], 0.0))
    config = pyimr.SimulationConfig(radius, radius / stretch, material,
                                    dynamics="keller-miksis", rtol=1e-7, atol=1e-9,
                                    max_steps=300_000)
    try:
      trace = np.asarray(pyimr.simulate(times, config).radius_ratio, dtype=float)
    except Exception:                                   # noqa: BLE001 -- a corner of the prior
      continue
    if np.all(np.isfinite(trace)): rows.append(trace)
  return np.asarray(rows, dtype=float)


def _job(design):
  import pyimr
  from pyimr.discriminate import expected_log_bayes_factor

  radius, stretch = design
  times = _times(radius)
  material = pyimr.QuadraticZener(FIT["g"], FIT["mu"], FIT["lambda1"], 0.0, FIT["alpha"])
  config = pyimr.SimulationConfig(radius, radius / stretch, material, dynamics="keller-miksis",
                                  rtol=1e-9, atol=1e-11, max_steps=600_000)
  try:
    jacobian = np.asarray(pyimr.prepare(config).solve_with_sensitivities(times, PATHS).radius_ratio,
                          dtype=float)
  except Exception as error:                            # noqa: BLE001
    return design, {"failed": f"information: {type(error).__name__}: {error}"}
  # fractional sensitivities, whitened: the columns then span ten orders of magnitude equally
  jacobian = jacobian * np.array([FIT[k] for k in ("g", "mu", "lambda1", "alpha")]) / RELATIVE_NOISE
  if not np.all(np.isfinite(jacobian)): return design, {"failed": "information not finite"}

  truth, rival = _bank(design, True, 0), _bank(design, False, 1)
  if truth.shape[0] < 16 or rival.shape[0] < 16:
    return design, {"failed": f"banks too small: {truth.shape[0]}, {rival.shape[0]}"}
  scored = expected_log_bayes_factor(truth, rival, RELATIVE_NOISE)
  return design, dict(information=(jacobian.T @ jacobian).tolist(),
                      utility=scored.expected_log_bayes_factor,
                      standard_error=scored.standard_error,
                      reliable=bool(scored.reliable),
                      effective=min(scored.effective_draws_true, scored.effective_draws_rival),
                      bank=(truth.shape[0], rival.shape[0]))


def main():
  from pyimr.measure import identification_front

  with records.pool(len(DESIGNS)) as pool:
    table = dict(pool.map(_job, DESIGNS))

  usable = [d for d in DESIGNS if "failed" not in table[d]]
  unreliable = [d for d in usable if not table[d]["reliable"]]
  print(f"{len(usable)} of {len(DESIGNS)} designs scored; {len(unreliable)} have an unreliable "
        "evidence integral")
  if len(usable) < 8:
    print("  too few designs survived to build a front")
    for d in DESIGNS[:5]:
      if "failed" in table[d]: print(f"    {d}: {table[d]['failed'][:70]}")
    return

  information = np.array([table[d]["information"] for d in usable], dtype=float)
  utility = np.array([table[d]["utility"] for d in usable], dtype=float)
  print(f"  expected log Bayes factor per run spans {utility.min():.2f} to {utility.max():.2f} nats")
  best = usable[int(np.argmax(utility))]
  print(f"  most discriminating single setting: R_max {best[0]*1e6:.0f} um at stretch {best[1]:.1f}")

  front = identification_front(information, utility, RUNS)
  print(f"\nbatches of {RUNS}, on the qSLS-vs-SLS question\n")
  print(f"{'blend':>6} {'settings':>9} {'D-eff':>7} {'nats':>8} {'certified':>10}  allocation")
  summary = []
  for batch in front:
    allocation = [(f"{usable[i][0]*1e6:.0f}um/{usable[i][1]:.1f}", n) for i, n in batch.table]
    print(f"{batch.blend:6.2f} {batch.settings:9d} {batch.efficiency:7.3f} "
          f"{batch.discrimination:8.2f} {str(batch.certified):>10}  "
          + ", ".join(f"{k}x{n}" for k, n in allocation))
    summary.append(dict(blend=batch.blend, settings=batch.settings, efficiency=batch.efficiency,
                        discrimination=batch.discrimination, certified=batch.certified,
                        allocation=allocation))

  # what the geometry already performed would buy, for comparison: the closest candidate
  performed = min(usable, key=lambda d: abs(np.log(d[0] / 277e-6)) + abs(np.log(d[1] / 7.09)))
  index = usable.index(performed)
  print(f"\n  the performed geometry is nearest {performed[0]*1e6:.0f} um at stretch "
        f"{performed[1]:.1f}, worth {utility[index]:.2f} nats per run")
  print(f"  so {RUNS} runs there give {RUNS * utility[index]:.1f} nats against "
        f"{front[2].discrimination:.1f} for the balanced batch")
  print("\n  Read `settings` before anything else: a batch on one setting cannot detect that")
  print("  BOTH models are wrong, which `lackoffit.py` says is the case on every record.")
  records.HERE.joinpath("identify.json").write_text(json.dumps(
    {"front": summary,
     "designs": [{"radius": d[0], "stretch": d[1], **{k: v for k, v in table[d].items()
                                                      if k != "information"}} for d in DESIGNS]},
    indent=1))


if __name__ == "__main__":
  main()
