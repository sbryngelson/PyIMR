r"""Two chance levels in this document assume independence. Both are recomputed here.

The parameter share of \cref{sec:latent} is quoted against "a chance level of $6/201 = 3.0$
percent", and the enrichment screen of \cref{sec:enrich} reports a removable share with no
chance level at all. Both are projections of a measured curve onto a low-dimensional smooth
subspace, and both are read as large. A subspace of dimension $d$ in $\mathbb{R}^k$ captures
$d/k$ of WHITE noise, which is where $3.0$ percent comes from -- and these residuals are
correlated at $\rho = 0.68$ to $0.86$, which is the central claim of the document. Smooth noise
aligns with a smooth subspace far more often than white noise does.

SO THE NULL IS BUILT FROM THE DATA'S OWN AUTOCORRELATION. Surrogates are drawn as AR(1) series
at each record's measured $\rho$, projected through the SAME code that produces the measured
number, and the measured value is quoted as a percentile of that distribution rather than
against $d/k$. Nothing else changes.

WHAT IS AT STAKE. If the parameter share sits inside its null, then the $39.3$ percent that
\cref{sec:latent} reads as latent per-event parameter variation is what correlated noise
produces by itself, and the correction built on it in \cref{sec:fitquality} has no basis. If the
enrichment cosines sit inside theirs, then "no candidate enrichment accounts for the bulk" is
true but uninformative -- it would be true of any candidate whatever, including the right one.
"""

import json

import numpy as np

import records
from enrichment_screen import directions
from paam_variation import PATHS, _events, _lag_one
from shape_error import seed_for

DRAWS = 2000
ORDER = (*records.DATASETS, *records.PAAM)


def _surrogate(rng, k, rho):
  e = rng.standard_normal(k)
  for i in range(1, k):
    e[i] = rho * e[i - 1] + np.sqrt(max(1e-12, 1.0 - rho * rho)) * e[i]
  return e


def share_job(dataset):
  """The parameter share, and the share the same projection gives correlated noise."""
  import pyimr

  times, mean, spread, maximum, stretch, events = _events(dataset)
  fitted = json.load(open(records.HERE / "paam_lackoffit.json"))[dataset]["fitted"]
  config = pyimr.SimulationConfig(
    maximum, maximum / stretch,
    pyimr.QuadraticZener(fitted["g"], fitted["mu"], fitted["lambda1"], 0.0, fitted["alpha"]),
    dynamics="keller-miksis", rtol=1e-9, atol=1e-11, max_steps=600_000)
  problem = pyimr.prepare(config)
  values = np.array([maximum, maximum / stretch, fitted["g"], fitted["mu"],
                     fitted["lambda1"], fitted["alpha"]])
  jacobian = np.asarray(problem.solve_with_sensitivities(times, PATHS).radius_ratio,
                        dtype=float) * values
  basis, _ = np.linalg.qr(jacobian)

  deviations = events - events.mean(axis=0)
  explained = deviations @ basis @ basis.T
  measured = 1.0 - float(((deviations - explained) ** 2).sum() / (deviations**2).sum())
  # the autocorrelation the deviations actually carry, which is what the null must reproduce
  rho = float(np.median([_lag_one(d) for d in deviations]))

  rng = np.random.default_rng(seed_for(dataset))
  null = np.empty(DRAWS)
  for d in range(DRAWS):
    e = _surrogate(rng, times.size, rho)
    p = basis @ (basis.T @ e)
    null[d] = float((p @ p) / (e @ e))
  return dataset, {"measured": measured, "white_chance": jacobian.shape[1] / times.size,
                   "rho": rho, "null_mean": float(null.mean()),
                   "null_95": float(np.percentile(null, 95)),
                   "percentile": float((null < measured).mean())}


def screen_job(dataset):
  """Each enrichment candidate's removable share, against smooth directions of no mechanism."""
  from pyimr.noise import enrichment_overlap

  _, split, jacobian, found = directions(dataset)
  measured = enrichment_overlap(split.identifiable, jacobian, found)
  target = np.asarray(split.identifiable, dtype=float)
  rho = _lag_one(target)

  rng = np.random.default_rng(seed_for(dataset + "screen"))
  null = np.empty(DRAWS)
  # the JOINT statistic needs its own null: a cone over as many surrogate directions as there
  # are candidates reaches further than one direction does, so comparing the measured joint
  # against the single-direction null would repeat the error this module exists to correct
  joint_null = np.empty(DRAWS // 4)
  for d in range(DRAWS):
    fake = {"surrogate": _surrogate(rng, target.size, rho)}
    null[d] = enrichment_overlap(split.identifiable, jacobian, fake).removable["surrogate"]
  for d in range(DRAWS // 4):
    fake = {f"s{i}": _surrogate(rng, target.size, rho) for i in range(len(found))}
    joint_null[d] = enrichment_overlap(split.identifiable, jacobian, fake).joint
  return dataset, {
    "removable": measured.removable, "reachable": measured.reachable,
    "joint": measured.joint, "rho": rho, "candidates": len(found),
    "null_mean": float(null.mean()), "null_95": float(np.percentile(null, 95)),
    "joint_null_mean": float(joint_null.mean()),
    "joint_null_95": float(np.percentile(joint_null, 95)),
    "joint_beats_null": bool(measured.joint > np.percentile(joint_null, 95)),
    "beats_null": {k: bool(v > np.percentile(null, 95))
                   for k, v in measured.removable.items()}}


def main():
  print("  ---- the parameter share, against a null that is not white ----\n")
  with records.pool(len(ORDER)) as pool:
    shares = dict(pool.map(share_job, list(ORDER)))
  print(f"  {'dataset':>16s} {'measured':>9s} {'white d/k':>10s} {'rho':>6s} "
        f"{'null mean':>10s} {'null 95%':>9s} {'percentile':>11s}")
  for dataset in ORDER:
    v = shares[dataset]
    print(f"  {dataset:>16s} {v['measured']:9.1%} {v['white_chance']:10.1%} {v['rho']:6.3f} "
          f"{v['null_mean']:10.1%} {v['null_95']:9.1%} {v['percentile']:11.2f}")
  beat = [d for d in ORDER if shares[d]["measured"] > shares[d]["null_95"]]
  print(f"\n  {len(beat)} of {len(ORDER)} exceed their own 95th percentile"
        f"{': ' + ', '.join(beat) if beat else ' -- none'}.")

  print("\n  ---- the enrichment screen, against directions of no mechanism ----\n")
  with records.pool(len(ORDER)) as pool:
    screens = dict(pool.map(screen_job, list(ORDER)))
  names = sorted({k for v in screens.values() for k in v["removable"]})
  print(f"  {'candidate':28s} " + " ".join(f"{d.replace('paam_', '')[:9]:>10s}" for d in ORDER))
  for name in names:
    cells = []
    for dataset in ORDER:
      v = screens[dataset]
      got = v["removable"].get(name)
      if got is None: cells.append("         -")
      else: cells.append(f"{got:9.1%}" + ("*" if v["beats_null"].get(name) else " "))
    print(f"  {name:28s} " + " ".join(cells))
  print(f"  {'null 95% for this dataset':28s} "
        + " ".join(f"{screens[d]['null_95']:9.1%} " for d in ORDER))
  print(f"  {'ALL TOGETHER (cone)':28s} "
        + " ".join(f"{screens[d]['joint']:9.1%}" + ("*" if screens[d]["joint_beats_null"] else " ")
                   for d in ORDER))
  print(f"  {'joint null 95%':28s} "
        + " ".join(f"{screens[d]['joint_null_95']:9.1%} " for d in ORDER))
  print("\n  * exceeds the 95th percentile of a smooth direction carrying no mechanism at all")
  print("    (the joint row against a cone over the same NUMBER of surrogate directions)")

  print("\n  ---- what it says ----\n")
  survivors = {d: [n for n, ok in screens[d]["beats_null"].items() if ok] for d in ORDER}
  total = sum(len(v) for v in survivors.values())
  print(f"  Of {sum(len(screens[d]['removable']) for d in ORDER)} candidate-dataset cells, "
        f"{total} exceed a null direction of the same smoothness.")
  for d in ORDER:
    print(f"    {d:>16s}: {', '.join(survivors[d]) if survivors[d] else 'none'}")
  json.dump({"parameter_share": shares, "enrichment": screens},
            open(records.HERE / "correlated_nulls.json", "w"), indent=1)


if __name__ == "__main__":
  main()
