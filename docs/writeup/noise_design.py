r"""Does the constant noise scale change which experiment the design recommends?

The design chapter assumes $\sigma = 0.018$ at every sample. The records say the spread varies
by two to three orders of magnitude within one trace, and \S`noise\_shape` says no closed form
in $y$ predicts it -- the profile is dynamical, largest through the collapse and the rebound
after it. So the honest question is not how wrong the constant is, which is settled, but
whether it changes the ANSWER.

Because the misspecification is a noise covariance and not a mean, the exact difference is
computable and needs no bound. Both likelihoods are Gaussian about the same forward model, so
each design's Fisher matrix is $J^{\mathsf T}\Sigma^{-1}J$ under either $\Sigma$ and the gain
follows in closed form.

TRANSFERRING A MEASURED PROFILE TO A GEOMETRY NOBODY HAS RUN is the one modelling step here,
and it is the step \S`noise\_shape` argues has to be dynamical. Each record's spread is read as
a function of $t/t_c$ rather than of $t$, and interpolated onto the candidate's own grid at the
same phases. That assumes what the measurement supports -- that the spread tracks where the
bubble is in its collapse -- and nothing more. Running all three records' profiles separately
shows how much the conclusion depends on which one is borrowed.

The pinned normalisation sample is floored, for the reason `pinned\_samples.py` gives: its
spread is zero by construction rather than by precision, and $1/\sigma^2$ would hand it most of
the Fisher weight. The floor is a percentile of the record's own spread.

THE ANSWER IS THAT IT DOES CHANGE. Two of the three profiles move the recommended geometry off
the constant's choice of \SI{514}{\micro\metre} at stretch $6.92$ and onto
\SI{416}{\micro\metre} at stretch $10.85$, and judged by those profiles the constant's pick
gives up $1.34$ and $0.50$ nats of $8.89$ and $8.63$ -- fifteen percent in the worse case. Only
\SI{15}{\celsius}'s profile agrees with the constant.

Which way the disagreement runs is the part worth noticing. The two profiled models agree with
EACH OTHER and not with the constant, so the flat scale is the outlier rather than one record
being eccentric. \Cref{sec:screening} warns that a batch aimed at a live pair may be aimed at
one whose liveness is an artifact of the likelihood; this is the same warning for the geometry,
with a number attached.
"""

import json

import numpy as np

import records

RELATIVE_NOISE = 0.018           # what the design scripts assume
FLOOR = 0.10                     # percentile the pinned samples are held at
SAMPLES = 201


def profile(dataset, floor=FLOOR):
  """A record's measured spread as a function of `t/t_c`, floored and ready to interpolate."""
  from pyimr.noise import characteristic_time
  times, mean, spread, maximum, stretch = records.load(dataset)
  held = np.maximum(spread, float(np.quantile(spread, floor)))
  return (times - times[0]) / characteristic_time(maximum), held


def raw_information(design):
  """`(design, jacobian)` unwhitened, so either noise model can be applied to the same one."""
  import pyimr
  from pyimr.inference import InferenceParameter, RadiusObservation, prepare_inference
  from pyimr.noise import characteristic_time
  from design_operator import FIT, HIGH, LOW, RIVALS

  radius, stretch = design
  times = np.linspace(0.0, 5.0 * characteristic_time(radius), SAMPLES)
  material = pyimr.QuadraticZener(FIT[0], FIT[1], FIT[2], 0.0, FIT[3])
  traces = {}
  for operator in RIVALS:
    config = pyimr.SimulationConfig(radius, radius / stretch, material, dynamics=operator[0],
                                    liquid_eos=operator[1], rtol=1e-7, atol=1e-9,
                                    max_steps=200_000)
    try:
      traces[operator] = np.asarray(pyimr.simulate(times, config).radius_ratio, dtype=float)
    except Exception:                                                        # noqa: BLE001
      return design, None
  if not all(np.all(np.isfinite(t)) for t in traces.values()): return design, None

  base = pyimr.SimulationConfig(radius, radius / stretch, material, dynamics=RIVALS[0][0],
                                liquid_eos=RIVALS[0][1], rtol=1e-7, atol=1e-9,
                                max_steps=200_000)
  # unit noise here: the whitening is applied afterwards, once per noise model
  inference = prepare_inference(
    base, RadiusObservation(times, traces[RIVALS[0]] * radius, radius),
    tuple(InferenceParameter(f"material.{p}", lo, hi, "log") for p, lo, hi in zip(
      ("shear_modulus_pa", "viscosity_pa_s", "relaxation_time_s", "stiffening"),
      LOW, HIGH, strict=True)))
  unit = np.clip((np.log(FIT) - np.log(LOW)) / (np.log(HIGH) - np.log(LOW)), 0, 1)
  try:
    jacobian = np.asarray(inference.jacobian(unit), dtype=float)
  except Exception:                                                          # noqa: BLE001
    return design, None
  if not np.all(np.isfinite(jacobian)): return design, None
  gap = traces[RIVALS[1]] - traces[RIVALS[0]]
  return design, (np.column_stack([jacobian, gap]), times / (times[-1] / 5.0))


def gain(columns, scale, prior=1.0 / np.sqrt(12.0)):
  """`0.5 log det(I + s^2 J^T Sigma^-1 J)` for a per-sample noise scale."""
  whitened = columns / np.asarray(scale)[:, None]
  fisher = whitened.T @ whitened
  fisher = 0.5 * (fisher + fisher.T)
  return 0.5 * float(np.linalg.slogdet(np.eye(fisher.shape[0]) + prior**2 * fisher)[1])


def main():
  from design_operator import PERFORMED, RADII, STRETCH

  designs = [(float(r), float(s)) for r in RADII for s in STRETCH] + list(PERFORMED.values())
  with records.pool(len(designs)) as pool:
    got = list(pool.map(raw_information, designs))
  usable = [(d, payload) for d, payload in got if payload is not None]
  print(f"\n  {len(usable)} of {len(designs)} candidates integrate")

  profiles = {name: profile(name) for name in records.DATASETS}
  models = ["constant", *profiles]
  table = {}
  for design, (columns, phase) in usable:
    entry = {"radius_m": design[0], "stretch": design[1],
             "constant": gain(columns, np.full(columns.shape[0], RELATIVE_NOISE))}
    for name, (tau, spread) in profiles.items():
      entry[name] = gain(columns, np.interp(phase, tau, spread))
    table[f"{design[0]:.6e}|{design[1]:.6e}"] = entry
  values = list(table.values())

  print("\n  expected gain in nats, and the design each noise model recommends\n")
  print(f"  {'noise model':16s} {'best U':>8s} {'R_max (um)':>11s} {'stretch':>8s} "
        f"{'regret of the constant choice':>30s}")
  arrays = {m: np.array([v[m] for v in values]) for m in models}
  picked = {m: int(np.argmax(arrays[m])) for m in models}
  for m in models:
    best = values[picked[m]]
    # what the constant's pick costs, judged by THIS model
    regret = arrays[m][picked[m]] - arrays[m][picked["constant"]]
    label = "-- (it is the choice)" if m == "constant" else f"{regret:.4f} nats"
    print(f"  {m:16s} {arrays[m][picked[m]]:8.3f} {best['radius_m'] * 1e6:11.1f} "
          f"{best['stretch']:8.2f} {label:>30s}")

  agree = sum(picked[m] == picked["constant"] for m in models if m != "constant")
  worst = max(arrays[m][picked[m]] - arrays[m][picked["constant"]]
              for m in models if m != "constant")
  share = worst / max(arrays[m][picked[m]] for m in models if m != "constant")
  print(f"\n  profiles agreeing with the constant's design: {agree} of {len(profiles)};"
        f" worst regret {worst:.3f} nats ({share:.0%})")
  others = [picked[m] for m in models if m != "constant"]
  print(f"  the profiled models agree with each other on "
        f"{max(others.count(o) for o in set(others))} of {len(others)} -- so the flat scale is")
  print("  the outlier, not one eccentric record. The noise model changes the recommendation.")
  json.dump(table, open(records.HERE / "noise_design.json", "w"), indent=1)


if __name__ == "__main__":
  main()
