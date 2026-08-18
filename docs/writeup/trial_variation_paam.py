r"""How much of the trial-to-trial spread is parameter variation, on BOTH materials?

`trial_variation.py` answers this for one record, `gelatin_15C`, and the number it produced --
`PARAMETER_SHARE = 0.393` -- is used as though it described the experiment. It does not: it is
a gelatin number from a single record, and \cref{sec:allocation} leans on it while the second
material was not yet in the document. Five polyacrylamide records now carry released per-event
matrices, so the number can be measured where it is being applied instead of transferred.

The construction is the one that script establishes. Each event is a draw from
$\theta_j \sim N(\theta_{\rm pop}, \Sigma)$, so its deviation from the event mean lies in the
span of $G = \partial (R/R_{\max})/\partial\log\theta$ up to measurement noise, and the share of
the total deviation variance falling in that span is what parameter variation explains. The
span is six-dimensional and the records carry $160$ to $201$ samples, so the chance level is a
few percent and is printed beside the answer rather than assumed negligible.

Gelatin's per-event traces are not in `results.json`, so this covers the five PAAm records and
`gelatin_15C` is quoted from the earlier script for comparison.
"""

import json

import numpy as np

import records

PATHS = ("R0", "Req", "material.shear_modulus_pa", "material.viscosity_pa_s",
         "material.relaxation_time_s", "material.stiffening")


def lag_one(series):
  series = series - series.mean()
  return float(np.dot(series[:-1], series[1:]) / np.dot(series, series))


def one(dataset):
  import pyimr

  times, events = records.trials(dataset)
  fitted = json.load(open(records.HERE / "paam_lackoffit.json"))[dataset]["fitted"]
  maximum, stretch = records.PAAM_MAXIMUM, records.PAAM[dataset][2]
  material = pyimr.QuadraticZener(fitted["g"], fitted["mu"], fitted["lambda1"], 0.0,
                                  fitted["alpha"])
  config = pyimr.SimulationConfig(maximum, maximum / stretch, material,
                                  dynamics="keller-miksis", rtol=1e-9, atol=1e-11,
                                  max_steps=400_000)
  problem = pyimr.prepare(config)
  scale = np.array([maximum, maximum / stretch, fitted["g"], fitted["mu"],
                    fitted["lambda1"], fitted["alpha"]])
  try:
    jacobian = np.asarray(problem.solve_with_sensitivities(times, PATHS).radius_ratio,
                          dtype=float) * scale
  except Exception as failure:                                               # noqa: BLE001
    return dataset, {"failed": f"{type(failure).__name__}: {failure}"[:110]}

  deviations = events - events.mean(axis=0)
  basis, _ = np.linalg.qr(jacobian)
  explained = deviations @ basis @ basis.T
  leftover = deviations - explained
  share = 1.0 - float((leftover**2).sum() / (deviations**2).sum())
  return dataset, {
    "share": share, "chance": jacobian.shape[1] / len(times),
    "events": int(events.shape[0]), "samples": int(len(times)),
    "lag_one_raw": float(np.median([lag_one(d) for d in deviations])),
    "lag_one_leftover": float(np.median([lag_one(d) for d in leftover]))}


def main():
  order = list(records.PAAM)
  with records.pool(len(order)) as pool:
    got = dict(pool.map(one, order))

  print("  share of trial-to-trial variance lying in the sensitivity span\n")
  print(f"  {'record':>16s} {'events':>7s} {'samples':>8s} {'share':>8s} {'chance':>8s}"
        f" {'lag-1 raw':>10s} {'lag-1 left':>11s}")
  shares = []
  for dataset in order:
    value = got[dataset]
    if "failed" in value:
      print(f"  {dataset:>16s}  {value['failed']}")
      continue
    shares.append(value["share"])
    print(f"  {dataset:>16s} {value['events']:7d} {value['samples']:8d} {value['share']:8.1%}"
          f" {value['chance']:8.1%} {value['lag_one_raw']:10.3f}"
          f" {value['lag_one_leftover']:11.3f}")

  if shares:
    print(f"\n  PAAm: {np.median(shares):.1%} median, {min(shares):.1%} to {max(shares):.1%}"
          f" over {len(shares)} records")
    print(f"  gelatin_15C, from trial_variation.py: 39.3%  <- the number in use as"
          " PARAMETER_SHARE")
    print("\n  ---- what it says ----\n")
    if min(shares) > 0.20:
      print("  Parameter variation is a real and comparable share of the trial spread on the")
      print("  second material too, so PARAMETER_SHARE is not a gelatin peculiarity. Quote the")
      print("  range rather than the single gelatin value.")
    else:
      print("  The share is materially smaller on PAAm than the 39.3% gelatin number in use,")
      print("  so PARAMETER_SHARE does not transfer and sec:allocation has to say which")
      print("  records its number covers.")
  json.dump(got, open(records.HERE / "trial_variation_paam.json", "w"), indent=1)


if __name__ == "__main__":
  main()
