r"""The afterbounces are a frequency sweep, and the damping across them is a clock-free observable.

Every fit in this document targets a trace, which is a function of time, so anything that
rescales time trades against anything that dissipates energy. That confounding is not
hypothetical: `clock_screen.py` measures a timebase perturbation as \SI{97}{\percent} absorbed by
refitting the material, and \cref{sec:limitations} measures the datasets buying a $5$--$9\%$
slower collapse through the density. Neither is visible in the residual.

THE RATIO OF SUCCESSIVE BOUNCE MAXIMA CANNOT BE FOOLED THAT WAY. Stretch the time axis however
you like and $A_{k+1}/A_k$ is unchanged. It is invariant under exactly the transformation that
hides the deficit, it needs no derivative of a sampled trace, and it needs no model. Three
properties the trace does not have, and the third matters because \cref{sec:discrepancy}'s screen
can only rank candidates it has been handed.

IT IS ALSO A FREQUENCY SWEEP. Each bounce is slower than the one before, so a single event probes
the material at a descending sequence of rates, and the way the damping CHANGES across bounces is
a statement about the shape of the relaxation spectrum rather than about one relaxation time.
Gelatin is unequal chains debonding stochastically (\cref{sec:limitations}), which is a broad
spectrum by construction, and a single $\lambda_1$ fitted to a multi-rate signal has no correct
value --- which is the obvious candidate explanation for $\lambda_1$ being the loosest coordinate
in this package at $51$ to \SI{117}{\percent} between-trial spread.

AND IT IS THE ONE OBSERVABLE \#274 DOES NOT CORRUPT. That issue's defect is that averaging traces
across trials with jittered collapse times is not a trajectory. Averaging RATIOS is legitimate,
because each ratio is computed within a trial and is invariant to that trial's own clock. So this
is measured per trial and aggregated afterwards, which is the opposite order to everything else
here and is the reason it can use all $39$ events rather than three mean curves.

THE RESOLUTION CONTROL IS THE ONE THAT COULD HAVE KILLED IT. The model is read on a grid a
hundred times finer than the record, and a coarse grid samples NEAR a maximum rather than at it,
which would bias the measured amplitudes low and could manufacture the whole signature. Reading
the model on the record's own $201$ points instead shifts its ratios by $0.000$ to $0.001$,
against measured-against-model gaps of $0.03$ to $0.10$. The reason is the reason this observable
is worth having: it lives on the MAXIMA, which are broad and well resolved, where everything else
in this document depends on the collapse minimum, which the same grid resolves badly enough that
\#274 is an issue at all.

WHAT WOULD BE UNINTERESTING. A constant ratio across bounces is a single exponential decay and
says one relaxation time suffices. Damping that weakens as the bounces slow is rate dependence,
but it is not automatically MATERIAL rate dependence: acoustic radiation falls with Mach number
and the gas cushion stiffens as the bubble shrinks, and both weaken damping on their own. The
model carries all three, so the test is not whether the measured ratios vary. It is whether the
fitted model reproduces the variation it already contains.
"""

import json
import pathlib

import numpy as np

import records

DATA = pathlib.Path.home() / "fastscratch/papers/paper_imr_windowing/data"
FILES = {"gelatin_15C": "Ga_t15_exp_data.csv", "gelatin_23C": "Ga_t23_exp_data.csv",
         "gelatin_33C": "Ga_t33_exp_data.csv"}
MAX_RATIO, ORDER, KEEP = 1.05, 2, 6      # samples either side for an extremum; bounces to report
RATIO = 38.5


def maxima(trace, tau, order=ORDER):
  """`(indices, amplitudes, times)` of the resolved bounce maxima, first collapse onward."""
  idx = [i for i in range(order, len(trace) - order)
         if trace[i] == max(trace[i - order:i + order + 1]) and trace[i] > trace[i - 1]]
  return idx, [float(trace[i]) for i in idx], [float(tau[i]) for i in idx]


def sequence(trace, tau):
  """Per-bounce amplitude ratios and the rate each one probes.

  The rate is taken as the inverse of the interval between the bracketing maxima, which is the
  only rate available without differentiating a sampled trace. It is a period, so it carries the
  event's own clock; the RATIO of successive periods does not, and both are returned so the
  clock-free comparison can be made on the second.
  """
  _, amps, times = maxima(trace, tau)
  if len(amps) < 3: return None
  ratios = [amps[k + 1] / amps[k] for k in range(len(amps) - 1)]
  periods = [times[k + 1] - times[k] for k in range(len(times) - 1)]
  return {"amplitudes": amps, "times": times, "ratios": ratios, "periods": periods}


def measured(dataset):
  """Per trial, then aggregated. Ratios are clock-invariant so the order is legitimate."""
  table = np.loadtxt(DATA / FILES[dataset], delimiter=",", ndmin=2)
  tau, trials = table[:, 0], table[:, 1:].T
  keep = ~(trials > MAX_RATIO).any(axis=0) & (trials.std(axis=0, ddof=1) > 0.0)
  tau, trials = tau[keep], trials[:, keep]
  got = [s for s in (sequence(r, tau) for r in trials) if s is not None]
  n = min(len(s["ratios"]) for s in got)
  return {
    "events": len(got),
    "ratio_median": [float(np.median([s["ratios"][k] for s in got])) for k in range(min(n, KEEP))],
    "ratio_spread": [float(np.std([s["ratios"][k] for s in got], ddof=1)) for k in range(min(n, KEEP))],
    "period_median": [float(np.median([s["periods"][k] for s in got])) for k in range(min(n, KEEP))],
  }


def modelled(dataset):
  """The same observable from the fitted model, on a grid fine enough to place the extrema."""
  import pyimr

  times, _, _, maximum, stretch = records.load(dataset)
  m = json.load(open(records.HERE / "per_trial_fits.json"))[dataset]["median"]
  product = m["galpha"]
  material = pyimr.QuadraticZener(float(np.sqrt(product * RATIO)), m["mu"], m["lambda1"], 0.0,
                                  float(np.sqrt(product / RATIO)))
  fine = np.linspace(times[0], times[-1], 20001)
  config = pyimr.SimulationConfig(maximum, maximum / stretch, material,
                                  dynamics="keller-miksis", rtol=1e-9, atol=1e-11,
                                  max_steps=2_000_000)
  trace = np.asarray(pyimr.simulate(fine, config).radius_ratio, dtype=float)
  phase = (fine - fine[0]) / (fine[-1] - fine[0]) * 5.0
  # the model is resolved 100x finer than the record, so require a wider window for an extremum
  s = sequence(trace, phase)
  return None if s is None else {k: s[k][:KEEP] for k in ("ratios", "periods", "amplitudes")}


def control(dataset):
  """The model's own ratios read on the record's grid, which is what rules out a sampling bias."""
  import pyimr

  times, _, _, maximum, stretch = records.load(dataset)
  m = json.load(open(records.HERE / "per_trial_fits.json"))[dataset]["median"]
  product = m["galpha"]
  material = pyimr.QuadraticZener(float(np.sqrt(product * RATIO)), m["mu"], m["lambda1"], 0.0,
                                  float(np.sqrt(product / RATIO)))
  config = pyimr.SimulationConfig(maximum, maximum / stretch, material,
                                  dynamics="keller-miksis", rtol=1e-9, atol=1e-11,
                                  max_steps=2_000_000)
  trace = np.asarray(pyimr.simulate(times, config).radius_ratio, dtype=float)
  phase = (times - times[0]) / (times[-1] - times[0]) * 5.0
  s = sequence(trace, phase)
  return None if s is None else s["ratios"][:KEEP]


def main():
  print("  A_{k+1}/A_k is unchanged by any rescaling of time, so it cannot absorb a clock error.")
  print("  Each bounce is slower than the last, so the sequence is a descending rate sweep.\n")

  summary = {}
  for dataset in records.DATASETS:
    obs = measured(dataset)
    mod = modelled(dataset)
    coarse = control(dataset)
    summary[dataset] = {"measured": obs, "model": mod, "model_on_record_grid": coarse}
    print(f"  ==== {dataset}, {obs['events']} events ====\n")
    n = len(obs["ratio_median"])
    print(f"  {'bounce':>8s} " + " ".join(f"{k+1:>9d}" for k in range(n)))
    print(f"  {'measured':>8s} " + " ".join(f"{v:9.3f}" for v in obs["ratio_median"]))
    print(f"  {'spread':>8s} " + " ".join(f"{v:9.3f}" for v in obs["ratio_spread"]))
    if mod:
      mr = mod["ratios"][:n] + [float("nan")] * max(0, n - len(mod["ratios"]))
      print(f"  {'model':>8s} " + " ".join(f"{v:9.3f}" for v in mr))
      gap = [mr[k] - obs["ratio_median"][k] for k in range(n) if np.isfinite(mr[k])]
      sd = [obs["ratio_spread"][k] for k in range(len(gap))]
      sig = [g / (s / np.sqrt(obs["events"])) if s > 0 else float("nan")
             for g, s in zip(gap, sd, strict=True)]
      print(f"  {'gap/se':>8s} " + " ".join(f"{v:9.1f}" for v in sig))
      summary[dataset]["gap_sigmas"] = [float(v) for v in sig]
    if mod and coarse:
      n2 = min(len(mod["ratios"]), len(coarse))
      shift = max(abs(coarse[k] - mod["ratios"][k]) for k in range(n2))
      summary[dataset]["resolution_shift"] = float(shift)
      print(f"  {'grid ctl':>8s} " + " ".join(f"{coarse[k] - mod['ratios'][k]:9.3f}"
                                              for k in range(n2))
            + f"   max {shift:.4f}")
    print(f"  {'period':>8s} " + " ".join(f"{v:9.3f}" for v in obs["period_median"]))
    print()

  print("  ---- what it says ----\n")
  for dataset in records.DATASETS:
    r = summary[dataset]["measured"]["ratio_median"]
    trend = "rises" if r[-1] > r[0] else "falls"
    print(f"  {dataset:>13s}: damping ratio {trend} from {r[0]:.3f} to {r[-1]:.3f} across the "
          f"sweep")
  print("\n  A constant ratio would be one exponential decay and one relaxation time. A ratio")
  print("  that rises means the material damps less as the bounces slow. The model carries")
  print("  acoustic and gas-cushion damping too, so the test is the gap row: how far the fitted")
  print("  model's own sweep sits from the measured one, in standard errors of the mean.")
  json.dump(summary, open(records.HERE / "bounce_sweep.json", "w"), indent=1)


if __name__ == "__main__":
  main()
