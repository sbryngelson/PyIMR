r"""What does the jittered fit target cost the paper's central negative result?

\#274: every fit in this document targets `record["mean"]`, the sample mean over trials at fixed
lab time. The trials' first-collapse times scatter by about $2\%$, so that mean is not a
trajectory --- averaging across the jitter fills and widens the trough, and the observed minimum
comes out $10$--$20\%$ shallow. No spherical solution of any candidate passes through the mean
curve near the collapse at any parameter value, because the mean curve is not a solution of the
equations. `warped_target.py` confirms the defect and confirms that a per-trial time warp removes
it, taking the depth bias from $+19.6/+9.9/+14.0\%$ to $+16.7/+0.0/+0.0$.

WHAT HAS NOT BEEN ASKED IS WHAT IT COSTS. This document's central negative result is not the
ranking; it is \cref{sec:fitquality} and \cref{sec:discrepancy}: the residual carries structure
no candidate contains, with lag-one autocorrelation $0.918$, $N_{\rm eff}$ near ten of $201$,
lack of fit rejecting every model on every record, and $65$--$69\%$ of $\hat\delta$ sitting in
the first collapse. \Cref{sec:enrich} then searches for the missing physics and fails to find it.

That is exactly the signature a biased target would produce, and the coincidence is close enough
to be worth measuring rather than admiring: a fit target that no solution can pass through, whose
error is concentrated where the jitter is, would look like model inadequacy concentrated in the
first collapse. So this refits every candidate against the raw target and against both warps and
reads the diagnostics that carry the claim.

WHAT IS COMPARABLE AND WHAT IS NOT. Warping rebuilds the trial spread, so $\chi^2/N$ and lack of
fit are on a different scale for each target and are never differenced across them. Three
quantities are dimensionless and do transfer: the lag-one autocorrelation of the residual, the
share of the residual's squared length falling in the first collapse, and the ORDERING of the
candidates within a target. Those are what the conclusions below rest on.

WHAT THIS IS NOT. It refits at each candidate's own optimum rather than re-quadraturing the
evidence, for the reason `noise_constants.py` gives: $68{,}720$ grid points across $23$ models
per record is prohibitive, and the best-fit ordering is what the evidence differences track when
the margins are hundreds of nats. It is a statement about the ordering and the residual, and is
reported as such.
"""

import json

import numpy as np

import records
from warped_target import CLOCK_DENSITY, TARGETS, build_targets, local_minima

BOX = {"g": (1e2, 1e5), "mu": (1e-4, 1.0), "lam": (1e-9, 1e-3), "alpha": (1e-3, 1e2)}
MODELS = ("qSLS", "SLS", "qKV", "NHKV")
AXES = {"qSLS": ("g", "mu", "lam", "alpha"), "SLS": ("g", "mu", "lam"),
        "qKV": ("g", "mu", "alpha"), "NHKV": ("g", "mu")}
# what each candidate degenerates into, so the redundancy prior and the nesting story hold
CONTAINS = {"qSLS": ("SLS", "qKV"), "SLS": ("NHKV",), "qKV": ("NHKV",), "NHKV": ()}
DENSITY = 1064.0                          # as published; this study varies the target, not rho
STARTS, EVALUATIONS = 24, 700


def candidate_for(name):
  """The paper's four tiers in their natural parameterisations."""
  import pyimr
  from pyimr.selection import CandidateModel

  def build(t):
    if name == "qSLS": return pyimr.QuadraticZener(t["g"], t["mu"], t["lam"], 0.0, t["alpha"])
    if name == "SLS": return pyimr.Zener(t["g"], t["mu"], t["lam"], 0.0)
    if name == "qKV": return pyimr.QuadraticKelvinVoigt(t["g"], t["mu"], t["alpha"])
    return pyimr.NeoHookeanKelvinVoigt(t["g"], t["mu"])

  return CandidateModel(name, build, AXES[name], CONTAINS[name])


def solver(times, maximum, stretch):
  import pyimr
  physics = pyimr.PhysicalParameters(medium_density_kg_m3=DENSITY)

  def solve(material, _config=None):
    config = pyimr.SimulationConfig(maximum, maximum / stretch, material,
                                    dynamics="keller-miksis", rtol=1e-8, atol=1e-10,
                                    max_steps=400_000, physics=physics)
    return np.asarray(pyimr.simulate(times, config).radius_ratio, dtype=float), None

  return solve


def first_collapse_share(residual, mean):
  """Share of the residual's squared length falling at or before the first minimum.

  sec:discrepancy reports 65-69% for the identified discrepancy, and that concentration is the
  reason the search of sec:enrich looked where it did. It is a ratio, so it survives the target
  changing the scale of the residual, which chi-squared does not.
  """
  cut = local_minima(mean)[0]
  total = float(residual @ residual)
  return float(residual[:cut + 1] @ residual[:cut + 1]) / total if total > 0 else float("nan")


def one(job):
  """One candidate fitted to one target on one record."""
  from pyimr.noise import lack_of_fit
  from pyimr.selection import evaluate_at, fit_candidate, physical_from_unit

  dataset, target, name = job
  times, _, _, maximum, stretch = records.load(dataset)
  cache = json.load(open(records.HERE / "warped_target_cache.json"))[dataset][target]
  mean = np.asarray(cache["mean"], dtype=float)
  spread = np.asarray(cache["spread"], dtype=float)
  candidate = candidate_for(name)
  solve = solver(times, maximum, stretch)
  box = {k: BOX[k] for k in AXES[name]}
  try:
    fit = fit_candidate(candidate, solve, mean, spread, bounds=box, starts=STARTS,
                        max_evaluations=EVALUATIONS)
  except ValueError as error:
    return job, {"failed": str(error)}
  axes = tuple(candidate.axes)
  fitted = dict(zip(axes, (float(v) for v in physical_from_unit(axes, fit.unit, box)), strict=True))
  model = np.asarray(evaluate_at(candidate, solve, fitted)[0], dtype=float)
  residual = (mean - model) / spread
  centred = residual - residual.mean()
  variance = float(centred @ centred)
  rho = float(centred[:-1] @ centred[1:]) / variance if variance > 0 else float("nan")
  ratio = lack_of_fit(mean, model, spread, records.trial_count(dataset), candidate.dimension).ratio
  return job, {"chi2_per_n": float(fit.chi_squared), "lack_of_fit": float(ratio),
               "lag_one": rho, "n_eff": float(len(residual) * (1 - rho) / (1 + rho)),
               "first_collapse": first_collapse_share(residual, mean),
               "fitted": fitted,
               "galpha": float(fitted["g"] * fitted.get("alpha", 1.0))}


def main():
  print("  rebuilding the three targets ...", flush=True)
  cache = {}
  for dataset in records.DATASETS:
    _, built, _ = build_targets(dataset)
    cache[dataset] = {k: {"mean": v[0].tolist(),
                          "spread": np.nan_to_num(v[1], nan=float(np.nanmedian(v[1]))).tolist()}
                      for k, v in built.items()}
  json.dump(cache, open(records.HERE / "warped_target_cache.json", "w"))

  jobs = [(d, t, m) for d in records.DATASETS for t in TARGETS for m in MODELS]
  print(f"  {len(jobs)} fits at rho = {DENSITY:.0f}, clock {CLOCK_DENSITY:.0f} ...", flush=True)
  with records.pool(len(jobs)) as pool:
    got = dict(pool.map(one, jobs))

  summary = {"density": DENSITY, "records": {}}

  print("\n  ==== does the ranking survive? best-fit chi2/N, ordering within a target ====\n")
  for dataset in records.DATASETS:
    print(f"  {dataset}")
    print(f"  {'target':>10s}  " + " ".join(f"{m:>9s}" for m in MODELS) + "   winner")
    for target in TARGETS:
      cells, values = [], {}
      for m in MODELS:
        v = got[(dataset, target, m)]
        cells.append(f"{'failed':>9s}" if "failed" in v else f"{v['chi2_per_n']:9.3f}")
        if "failed" not in v: values[m] = v["chi2_per_n"]
      winner = min(values, key=values.get) if values else "--"
      summary["records"].setdefault(dataset, {}).setdefault(target, {})["winner"] = winner
      print(f"  {target:>10s}  " + " ".join(cells) + f"   {winner}")
    print()

  print("  ==== what it costs the central claim: qSLS residual diagnostics ====\n")
  print(f"  {'record':>14s} {'target':>10s} {'lag-one':>8s} {'N_eff':>7s} "
        f"{'1st collapse':>13s} {'lack of fit':>12s} {'g*alpha':>9s}")
  for dataset in records.DATASETS:
    for target in TARGETS:
      v = got[(dataset, target, "qSLS")]
      if "failed" in v:
        print(f"  {dataset:>14s} {target:>10s}  {v['failed']}")
        continue
      summary["records"][dataset][target]["qSLS"] = v
      print(f"  {dataset:>14s} {target:>10s} {v['lag_one']:8.3f} {v['n_eff']:7.1f} "
            f"{100 * v['first_collapse']:12.1f}% {v['lack_of_fit']:12.2f} {v['galpha']:9.0f}")
    print()

  print("  ---- what it says ----\n")
  print("  chi2/N and lack of fit are NOT comparable across targets: warping rebuilds the trial")
  print("  spread, which is their scale. Lag-one, the first-collapse share and the ordering are.\n")
  held = all(summary["records"][d][t]["winner"] == "qSLS"
             for d in records.DATASETS for t in TARGETS
             if "winner" in summary["records"][d][t])
  summary["ranking_holds"] = bool(held)
  print(f"  qSLS is the best-fitting candidate on every record and every target: {held}")
  # sec:discrepancy's 65-69% is the share of the IDENTIFIED discrepancy -- the component
  # orthogonal to the parameter span in identified coordinates -- not of this whitened residual.
  # They are different quantities and are not printed side by side.
  for key, label, published in (("lag_one", "lag-one", "0.918 in sec:fitquality"),
                                ("first_collapse", "first-collapse share", "not the same quantity as sec:discrepancy's 65-69%")):
    raw = [summary["records"][d]["raw"]["qSLS"][key] for d in records.DATASETS]
    best = [min(summary["records"][d][t]["qSLS"][key] for t in TARGETS if t in summary["records"][d])
            for d in records.DATASETS]
    print(f"  {label:>22s}: raw {np.array2string(np.array(raw), precision=3)}, "
          f"best over targets {np.array2string(np.array(best), precision=3)}")
    print(f"  {'':>22s}  ({published})")
  print("\n  Read lag-one against the SHAM, never against raw. The warp resamples by linear")
  print("  interpolation, which low-pass filters the target and raises lag-one on its own; the")
  print("  sham applies the identical interpolation with a neighbour's knots, so it carries that")
  print("  artifact and none of the alignment. A real warp that beats the sham has removed")
  print("  something; one that matches it has only smoothed the trace.")
  json.dump(summary, open(records.HERE / "aligned_target.json", "w"), indent=1)


if __name__ == "__main__":
  main()
