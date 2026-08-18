r"""The afterbounce loss scales with STRAIN, not frequency. Is that the medium or the instrument?

`disentangle.py` carries frequency and amplitude together, with errors clustered by event, and
the answer is one-sided: $\mathrm{d}\ln\tan\delta / \mathrm{d}\ln A$ is resolved away from zero
on all eight datasets at $+0.96$ to $+3.87$, while the frequency coefficient is resolved on two,
with OPPOSITE signs, and the within-bounce slopes scatter from $-3.0$ to $+2.6$ with no
consistent sign. So the loss curve of `spectroscopy.py` is an amplitude dependence wearing a
frequency costume, and the spectral reading is withdrawn.

WHICH LEAVES A BETTER MEASUREMENT THAN THE ONE INTENDED. $\tan\delta \propto A^2$ is the
signature of dissipation CUBIC in strain: a damping force $\propto A^3$ dissipates
$\propto A^4$ per cycle against $\propto A^2$ stored, so the loss tangent goes as $A^2$. A
linear viscoelastic solid, qSLS included, has $\tan\delta$ independent of amplitude at fixed
frequency. The exponent is therefore a direct, fit-free statement that the dominant dissipation
at these rates is not the linear one every IMR model assumes.

BUT A BUBBLE RADIATES, AND RADIATION IS ALSO NONLINEAR IN AMPLITUDE. Acoustic loss grows with
the collapse speed, so some exponent is expected from the instrument alone with no medium
nonlinearity whatever. That is the control, and it is the whole question: the model is run at
each record's own fitted LINEAR material and the identical extraction applied, giving the
exponent a linear medium in a radiating bubble produces. Only the excess over that is a
statement about the material.

Three model configurations are compared, because the baseline is not one number: the fitted
material, the same with the viscous arm off (radiation and gas alone), and the same with the
stiffening term off (so any elastic nonlinearity is removed too).
"""

import json

import numpy as np

import records
from bounce_sweep import maxima
from disentangle import collect
from spectroscopy import AMBIENT, DENSITY, KEEP


def _exponent(amplitude, loss):
  """`d ln tan(delta) / d ln A`, the single-predictor version, data and model alike."""
  keep = (amplitude > 0) & (loss > 0)
  if keep.sum() < 3: return None
  design = np.column_stack([np.ones(keep.sum()), np.log(amplitude[keep])])
  return float(np.linalg.lstsq(design, np.log(loss[keep]), rcond=None)[0][1])


def model_points(dataset, viscosity=None, stiffening=None):
  import pyimr

  times, _, _, maximum, stretch = records.load(dataset)
  fitted = json.load(open(records.HERE / "paam_lackoffit.json"))[dataset]["fitted"]
  material = pyimr.QuadraticZener(
    fitted["g"], fitted["mu"] if viscosity is None else viscosity,
    fitted["lambda1"], 0.0, fitted["alpha"] if stiffening is None else stiffening)
  fine = np.linspace(times[0], times[-1], 20001)
  config = pyimr.SimulationConfig(maximum, maximum / stretch, material,
                                  dynamics="keller-miksis", rtol=1e-10, atol=1e-12,
                                  max_steps=2_000_000)
  trace = np.asarray(pyimr.simulate(fine, config).radius_ratio, dtype=float)
  characteristic = maximum * np.sqrt(DENSITY / AMBIENT)
  _, amps, tk = maxima(trace, fine / characteristic, order=2)
  amplitude, loss = [], []
  for k in range(min(KEEP, len(amps) - 1)):
    ratio = amps[k + 1] / amps[k]
    if tk[k + 1] > tk[k] and 0 < ratio < 1:
      amplitude.append(amps[k]); loss.append(-np.log(ratio) / np.pi)
  return np.array(amplitude), np.array(loss)


def one(dataset):
  table = collect(dataset)
  measured = _exponent(table[:, 3], table[:, 4])
  out = {"measured": measured, "points": len(table)}
  for label, kwargs in (("fitted material", {}),
                        ("viscosity off", {"viscosity": 1e-9}),
                        ("viscosity and stiffening off", {"viscosity": 1e-9, "stiffening": 0.0})):
    try:
      amplitude, loss = model_points(dataset, **kwargs)
      out[label] = _exponent(amplitude, loss)
    except Exception as error:                                       # noqa: BLE001
      out[label] = None
      out[label + " failed"] = f"{type(error).__name__}"
  return dataset, out


def main():
  print("  tan(delta) against amplitude: the medium, or a radiating bubble?\n")
  jobs = list((*records.DATASETS, *records.PAAM))
  with records.pool(len(jobs)) as pool:
    got = dict(pool.map(one, jobs))

  print(f"  {'dataset':>16s} {'points':>7s} {'MEASURED':>10s} {'fitted model':>13s} "
        f"{'visc off':>10s} {'visc+stiff off':>15s} {'excess':>8s}")
  summary = {}
  for dataset in jobs:
    v = got[dataset]
    cells = [v.get("fitted material"), v.get("viscosity off"),
             v.get("viscosity and stiffening off")]
    excess = (v["measured"] - cells[0]) if (v["measured"] and cells[0]) else None
    print(f"  {dataset:>16s} {v['points']:7d} {v['measured']:10.2f} "
          + " ".join(f"{c:13.2f}" if c is not None else f"{'-':>13s}" for c in cells[:1])
          + " " + " ".join(f"{c:10.2f}" if c is not None else f"{'-':>10s}" for c in cells[1:2])
          + " " + " ".join(f"{c:15.2f}" if c is not None else f"{'-':>15s}" for c in cells[2:])
          + f" {excess:8.2f}" if excess is not None else "")
    summary[dataset] = v | {"excess_over_fitted_model": excess}

  print("\n  ---- what it says ----\n")
  good = {d: v for d, v in summary.items() if v.get("excess_over_fitted_model") is not None}
  if good:
    excesses = [v["excess_over_fitted_model"] for v in good.values()]
    measured = [v["measured"] for v in good.values()]
    models = [v["fitted material"] for v in good.values()]
    print(f"  measured exponent {min(measured):.2f} to {max(measured):.2f}; the same extraction")
    print(f"  on the fitted LINEAR material gives {min(models):.2f} to {max(models):.2f}.")
    print(f"  Excess {min(excesses):+.2f} to {max(excesses):+.2f} over {len(good)} datasets.")
    above = [d for d, v in good.items() if v["excess_over_fitted_model"] > 0.5]
    print(f"\n  {len(above)} of {len(good)} datasets dissipate more steeply with amplitude than a")
    print("  radiating bubble carrying a linear medium does. Where the excess is near zero the")
    print("  amplitude scaling is the instrument's and says nothing about the material.")
  records.HERE.joinpath("nonlinear_loss.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
  main()
