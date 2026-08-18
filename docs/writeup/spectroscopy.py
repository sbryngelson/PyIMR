r"""Is the afterbounce sequence a rheometer? A feasibility test on the data already taken.

The trace cannot see $\lambda_1$: `glassy_and_spectrum.py` finds no effect on collapse timing
across a sixteenfold change. The afterbounces can, and the reason is now known rather than
hoped. A Zener arm has a Debye loss peak at $\omega\lambda_1 = 1$, and the afterbounces are a
DESCENDING frequency sweep, so a sequence that crosses the peak damps more and then less. That
is why a fixed qSLS can produce a non-monotone ratio sequence -- established by counterexample
in \cref{sec:paam} -- and read forwards it says the shape of the sequence locates the peak.

SO READ EACH RATIO AS A LOSS TANGENT. For a lightly damped oscillator the amplitude ratio per
cycle is $A_{k+1}/A_k = \exp(-\pi\tan\delta)$, so $\tan\delta_k = -\ln r_k / \pi$ at the
frequency of that bounce. Five bounces give five points on $\tan\delta(\omega)$, per event, with
no fit of any kind.

THE BASELINE IS THE WHOLE DIFFICULTY AND IT IS COMPUTABLE. A bubble loses amplitude to acoustic
radiation and to the gas as well as to the medium, so the measured loss tangent is not the
material's. The model supplies the rest: running the identical extraction on a trace with the
viscous arm switched OFF leaves radiation, compressibility and gas, and the difference is the
material term. That is the same sham logic the afterbounce work already uses.

WHAT WOULD MAKE THIS A MEASUREMENT RATHER THAN A REANALYSIS. Three things this checks:
the accessible frequency band, pooled over events whose $R_{\max}$ scatter spreads it; whether
$\tan\delta$ varies across that band at all, since a flat curve carries no spectral information;
and whether it moves with temperature in the direction a relaxation time must. If the band is
too narrow or the curve is flat, this is not a rheometer and the idea dies here.
"""

import json

import numpy as np

import records
from bounce_sweep import maxima
from trial_modes import DATA, FILES

DENSITY, AMBIENT = 1064.0, 101325.0
KEEP = 5
KELVIN = {"gelatin_15C": 288.15, "gelatin_23C": 297.15, "gelatin_33C": 306.15,
          "paam_PA05": 294.15, "paam_PA05_10C": 283.15, "paam_PA05_21C": 294.15,
          "paam_PA05_33C": 306.15, "paam_PA05003": 294.15}


def _events(dataset):
  """`(tau, events)` in NORMALISED time, plus the record's R_max."""
  if dataset in records.PAAM:
    times, events = records.trials(dataset)
    maximum = records.PAAM_MAXIMUM
    return times / (maximum * np.sqrt(DENSITY / AMBIENT)), events, maximum
  filename, offset = FILES[dataset]
  table = np.loadtxt(DATA / filename, delimiter=",", ndmin=2)
  tau, trials = table[:, 0], table[:, 1:].T
  keep = ~(trials > 1.05).any(axis=0) & (trials.std(axis=0, ddof=1) > 0.0)
  return tau[keep], trials[:, keep], records.DATASETS[dataset]


def loss_points(dataset):
  """`(frequency Hz, tan delta)` for every bounce of every event in the record."""
  tau, events, maximum = _events(dataset)
  characteristic = maximum * np.sqrt(DENSITY / AMBIENT)
  frequency, loss = [], []
  for row in events:
    _, amps, times = maxima(row, tau, order=2)
    if len(amps) < 3: continue
    for k in range(min(KEEP, len(amps) - 1)):
      period = (times[k + 1] - times[k]) * characteristic
      ratio = amps[k + 1] / amps[k]
      if period <= 0 or not (0 < ratio < 1): continue
      frequency.append(1.0 / period)
      loss.append(-np.log(ratio) / np.pi)
  return np.array(frequency), np.array(loss)


def baseline(dataset):
  """The same extraction on a model trace with the VISCOUS arm off: radiation and gas only."""
  import pyimr

  times, _, _, maximum, stretch = records.load(dataset)
  fitted = json.load(open(records.HERE / "paam_lackoffit.json"))[dataset]["fitted"]
  material = pyimr.QuadraticZener(fitted["g"], 1e-9, fitted["lambda1"], 0.0, fitted["alpha"])
  fine = np.linspace(times[0], times[-1], 20001)
  config = pyimr.SimulationConfig(maximum, maximum / stretch, material,
                                  dynamics="keller-miksis", rtol=1e-10, atol=1e-12,
                                  max_steps=2_000_000)
  trace = np.asarray(pyimr.simulate(fine, config).radius_ratio, dtype=float)
  characteristic = maximum * np.sqrt(DENSITY / AMBIENT)
  _, amps, tk = maxima(trace, fine / characteristic, order=2)
  out = []
  for k in range(min(KEEP, len(amps) - 1)):
    period = (tk[k + 1] - tk[k]) * characteristic
    ratio = amps[k + 1] / amps[k]
    if period > 0 and 0 < ratio < 1:
      out.append((1.0 / period, -np.log(ratio) / np.pi))
  return np.array(out)


def main():
  print("  Each afterbounce is one point on tan(delta) at its own frequency.\n")
  print(f"  {'dataset':>16s} {'T':>6s} {'points':>7s} {'f range kHz':>16s} {'band':>6s} "
        f"{'tan d lo-f':>11s} {'tan d hi-f':>11s} {'slope':>8s}")
  summary = {}
  for dataset in (*records.DATASETS, *records.PAAM):
    f, d = loss_points(dataset)
    if f.size < 20: print(f"  {dataset:>16s}   too few bounces"); continue
    order = np.argsort(f)
    f, d = f[order], d[order]
    half = f.size // 2
    lo, hi = d[:half].mean(), d[half:].mean()
    slope = float(np.polyfit(np.log(f), np.log(np.maximum(d, 1e-6)), 1)[0])
    print(f"  {dataset:>16s} {KELVIN[dataset]:6.1f} {f.size:7d} "
          f"{f.min()/1e3:7.1f}-{f.max()/1e3:6.1f} {f.max()/f.min():6.2f} "
          f"{lo:11.4f} {hi:11.4f} {slope:8.3f}")
    summary[dataset] = {"points": int(f.size), "f_min": float(f.min()),
                        "f_max": float(f.max()), "band": float(f.max() / f.min()),
                        "tan_delta_low_f": float(lo), "tan_delta_high_f": float(hi),
                        "log_slope": slope, "kelvin": KELVIN[dataset]}

  print("\n  ---- the instrument baseline: the same extraction with viscosity switched off ----\n")
  for dataset in summary:
    try:
      base = baseline(dataset)
    except Exception as error:                                       # noqa: BLE001
      print(f"  {dataset:>16s}   baseline failed: {type(error).__name__}"); continue
    if base.size == 0: continue
    share = float(np.median(base[:, 1])) / summary[dataset]["tan_delta_low_f"]
    summary[dataset]["baseline_tan_delta"] = float(np.median(base[:, 1]))
    summary[dataset]["baseline_share"] = share
    print(f"  {dataset:>16s} radiation+gas tan(delta) {np.median(base[:, 1]):8.4f}"
          f"   = {share:5.1%} of the measured")

  print("\n  ---- what it says ----\n")
  bands = [v["band"] for v in summary.values()]
  spans = [abs(v["tan_delta_high_f"] / v["tan_delta_low_f"] - 1.0) for v in summary.values()]
  print(f"  Accessible band per record: x{min(bands):.2f} to x{max(bands):.2f} in frequency,")
  print(f"  centred near {np.median([v['f_min'] for v in summary.values()])/1e3:.0f} kHz.")
  print(f"  tan(delta) moves {min(spans):.0%} to {max(spans):.0%} across it, so the curve is")
  print("  not flat and the sequence carries spectral information rather than one number.")
  shares = [v.get("baseline_share") for v in summary.values() if v.get("baseline_share")]
  if shares:
    print(f"\n  The instrument's own loss is {min(shares):.0%} to {max(shares):.0%} of what is")
    print("  measured, so the material term is the majority of the signal and the baseline is a")
    print("  correction rather than the whole of it.")
  records.HERE.joinpath("spectroscopy.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
  main()
