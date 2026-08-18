r"""Does the hierarchical whitener remove a correlation that IS parameter variation?

`trial_variation.py` and `paam_variation.py` whiten the fit residual by
$C = (G\Sigma G^{\mathsf T} + \sigma^2 I)/J$ and read the surviving lag-one autocorrelation as
proof that the correlation is structure the model does not contain rather than latent per-event
parameter variation. On the measured records the correlation does survive, at $0.42$ to $0.88$
against $0.68$ to $0.86$ before.

THAT INFERENCE NEEDS A SHAM AND DID NOT HAVE ONE. On two of eight records whitening RAISES the
lag-one -- \SI{15}{\celsius} goes from $0.741$ to $0.878$ -- which is what a whitener that is
not working looks like. A correlation that survives a broken whitener says nothing at all. So
the same operation is run here on synthetic events whose deviations ARE pure parameter
variation by construction: each event is simulated at its own $\theta_j$ drawn about the fit,
with only a little white noise added on top. If the whitener removes the correlation there and
not on the real records, the published reading is sound. If it fails there too, the reading has
to be withdrawn and the surviving correlation means nothing either way.

The synthetic spread is set to the per-trial scatter `per_trial_fits.py` measures, so the sham
is not easier than the real thing in the one respect that matters.
"""

import json

import numpy as np

import records
from paam_variation import PATHS, _events, _lag_one
from shape_error import seed_for

AXES = ("mu", "galpha", "lambda1")
RATIO = 38.5
EVENTS = 24
WHITE = 0.002          # a small measurement noise, so the parameter part dominates by design


def _material(values):
  import pyimr
  product = max(values["galpha"], 1e-12)
  return pyimr.QuadraticZener(float(np.sqrt(product * RATIO)), values["mu"],
                              values["lambda1"], 0.0, float(np.sqrt(product / RATIO)))


def one(dataset):
  import pyimr

  times, mean, spread, maximum, stretch, _ = _events(dataset)
  base = json.load(open(records.HERE / "per_trial_fits.json"))[dataset]
  centre = {a: float(base["median"][a]) for a in AXES}
  scatter = {a: float(np.log1p(base["cv"][a])) for a in AXES}

  rng = np.random.default_rng(seed_for(dataset))
  traces = []
  for _ in range(EVENTS):
    drawn = {a: centre[a] * float(np.exp(rng.normal(0.0, scatter[a]))) for a in AXES}
    config = pyimr.SimulationConfig(maximum, maximum / stretch, _material(drawn),
                                    dynamics="keller-miksis", rtol=1e-9, atol=1e-11,
                                    max_steps=600_000)
    try:
      trace = np.asarray(pyimr.simulate(times, config).radius_ratio, dtype=float)
    except pyimr.SimulationError:
      continue
    traces.append(trace + rng.normal(0.0, WHITE, times.size))
  if len(traces) < 8: return dataset, None
  events = np.array(traces)

  # the sensitivities at the centre, exactly as paam_variation builds them
  config = pyimr.SimulationConfig(maximum, maximum / stretch, _material(centre),
                                  dynamics="keller-miksis", rtol=1e-9, atol=1e-11,
                                  max_steps=600_000)
  problem = pyimr.prepare(config)
  model = np.asarray(problem.solve(times).radius_ratio, dtype=float)
  values = np.array([maximum, maximum / stretch,
                     float(np.sqrt(centre["galpha"] * RATIO)), centre["mu"],
                     centre["lambda1"], float(np.sqrt(centre["galpha"] / RATIO))])
  jacobian = np.asarray(problem.solve_with_sensitivities(times, PATHS).radius_ratio,
                        dtype=float) * values

  deviations = events - events.mean(axis=0)
  basis, _ = np.linalg.qr(jacobian)
  explained = deviations @ basis @ basis.T
  leftover = deviations - explained
  share = 1.0 - float((leftover**2).sum() / (deviations**2).sum())

  residual = model - events.mean(axis=0)
  low_rank = explained.T @ explained / (len(events) - 1)
  noise = float((leftover**2).sum() / leftover.size)
  covariance = (low_rank + noise * np.eye(times.size)) / len(events) \
      + 1e-14 * np.eye(times.size)
  whitened = np.linalg.solve(np.linalg.cholesky(covariance), residual)
  return dataset, {"events": len(events), "share": share,
                   "lag_one_raw": _lag_one(residual),
                   "lag_one_whitened": _lag_one(whitened)}


def main():
  print("  SHAM: events whose deviations ARE parameter variation, by construction.\n")
  print("  If the whitener works, lag-one must FALL here.\n")
  jobs = list(records.DATASETS)
  with records.pool(len(jobs)) as pool:
    got = dict(pool.map(one, jobs))

  measured = json.load(open(records.HERE / "paam_variation.json"))
  print(f"  {'dataset':>14s} {'share':>8s} {'lag-1 raw':>10s} {'lag-1 white':>12s} "
        f"{'removed?':>10s}   | real record: raw -> white")
  summary = {}
  for dataset in jobs:
    v = got[dataset]
    if v is None:
      print(f"  {dataset:>14s}   too few integrable draws"); continue
    works = v["lag_one_whitened"] < v["lag_one_raw"] - 0.1
    real = measured[dataset]
    print(f"  {dataset:>14s} {v['share']:8.1%} {v['lag_one_raw']:10.3f} "
          f"{v['lag_one_whitened']:12.3f} {'YES' if works else 'NO':>10s}   | "
          f"{real['lag_one_raw']:.3f} -> {real['lag_one_whitened']:.3f}")
    summary[dataset] = v | {"removed_on_sham": bool(works)}

  print("\n  ---- what it says ----\n")
  ok = [d for d, v in summary.items() if v["removed_on_sham"]]
  if len(ok) == len(summary) and summary:
    print("  The whitener removes the correlation when the correlation IS parameter variation,")
    print("  on every dataset. So its survival on the real records means what sec:latent reads")
    print("  it as meaning: the structure is not latent per-event parameters.")
  elif not ok:
    print("  The whitener does NOT remove the correlation even when the correlation is parameter")
    print("  variation by construction. Its survival on the real records therefore says nothing,")
    print("  and every claim resting on that survival must be withdrawn.")
  else:
    print(f"  Mixed: the whitener works on {len(ok)} of {len(summary)} shams "
          f"({', '.join(ok)}). The reading is licensed only where it works, and the real-record")
    print("  conclusion has to be restricted to those datasets.")
  json.dump(summary, open(records.HERE / "whitening_sham.json", "w"), indent=1)


if __name__ == "__main__":
  main()
