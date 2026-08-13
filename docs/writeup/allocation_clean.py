r"""How much of the allocation gain is real, and how much is the clock?

\Cref{sec:allocation} is the largest lever in this chapter: re-splitting a fixed frame budget
between bubbles and samples is worth $+3.4$ to $+4.0$ nats, against $\le 1.34$ for the geometry.
The whole of it rests on $\Sigma_\theta$ being non-zero, because
%
    F(J, N) = J ( Sigma_theta + F_N^{-1} )^{-1}

collapses to $J F_N$ as $\Sigma_\theta \to 0$, and then only the product $JN$ matters and the
allocation question does not exist. So the gain is exactly as large as the between-bubble
parameter variation is.

AND \cref{sec:latent} HAS SINCE SHOWN THAT VARIATION IS NOT ALL PARAMETERS. A quarter of the
trial variance in these records lies along the time-dilation direction --- a clock error, not a
material difference. $\Sigma_\theta$ was estimated by fitting each trial separately, so whatever
the parameters could absorb of that clock error, they did. The number the allocation rests on is
contaminated by the amount, and nobody has said by how much.

THE TEST IS A PROCESSING CHANGE, PRICED. Each trial's own collapse time is measurable from its
own trace. Rescaling every trial onto the record's mean collapse time is what better processing
would do, costs nothing, and removes the part of the scatter that is timing. Refitting on the
aligned traces gives a $\Sigma_\theta$ with the clock taken out, and the allocation recomputed on
it says how much of $+3.93$ survives.

WHAT EITHER OUTCOME MEANS. If the gain survives, the allocation recommendation stands and the
clock contamination was a small share of a large effect. If it collapses, then the largest lever
in this chapter is substantially an artifact of processing, the remedy is to fix the processing
rather than to buy bubbles, and \cref{sec:allocation} needs rewriting rather than annotating.
"""

import json

import numpy as np

import records
from identified import BOX, candidate_at_ratio
from per_trial_fits import _trials

RATIO = 38.5
AXES = ("mu", "galpha", "lambda1")
STARTS, EVALUATIONS = 12, 400
# sec:allocation caps the sample count at the point where the collapse stops being resolved:
# N_eff is near ten of 201, and its own recommendation lands at N = 50. An unfloored search
# runs to the bottom of the range and reports an unphysical corner -- the first version of this
# returned J = 723, N = 5 on every record.
SAMPLE_FLOOR = 25


def collapse_time(times, trace):
  """The trial's own first minimum, in seconds, or None if it is not resolved."""
  from scipy.signal import argrelextrema

  minima = [i for i in argrelextrema(trace, np.less, order=3)[0] if times[i] > 0.2 * times[-1] / 5]
  if not minima: return None
  return float(times[minima[0]])


def one(job):
  """Fit one trial, optionally on a trace rescaled onto the record's mean collapse time."""
  from pyimr.selection import fit_candidate, physical_from_unit

  dataset, index, aligned = job
  times, mean, spread, maximum, stretch, window = _trials(dataset)
  trace = window[:, index]
  if aligned:
    own = collapse_time(times, trace)
    reference = np.median([collapse_time(times, window[:, j]) or np.nan
                           for j in range(window.shape[1])])
    if own is None or not np.isfinite(reference): return job, {"failed": "no collapse"}
    # stretch this trial's clock onto the record's, then resample on the shared grid
    trace = np.interp(times, times * (reference / own), trace)
  candidate = candidate_at_ratio(RATIO)
  solve = records.solver(times, maximum, stretch)
  try:
    fit = fit_candidate(candidate, solve, trace, spread, bounds=BOX, starts=STARTS,
                        max_evaluations=EVALUATIONS)
  except ValueError as error:
    return job, {"failed": str(error)}
  values = physical_from_unit(AXES, fit.unit, BOX)
  return job, {"fitted": dict(zip(AXES, (float(v) for v in values), strict=True))}


def covariance_of(rows):
  """`Sigma_theta` in log coordinates, as `per_trial_fits.py` reports it."""
  matrix = np.log(np.array([[r["fitted"][a] for a in AXES] for r in rows]))
  return np.cov(matrix, rowvar=False), matrix.std(axis=0, ddof=1)


def allocation_gain(covariance, fisher, samples, trials):
  """`(as run, best split, gain)` at a fixed frame budget, exactly as sec:allocation does."""
  def information(J, N):
    scaled = fisher * (N / samples)
    try:
      bracket = covariance + np.linalg.inv(scaled)
      return float(np.linalg.slogdet(J * np.linalg.inv(bracket))[1])
    except np.linalg.LinAlgError:
      return -np.inf

  budget = trials * samples
  here = information(trials, samples)
  best, best_value = (trials, samples), here
  for count in range(SAMPLE_FLOOR, samples + 1):
    J = max(1, budget // count)
    value = information(J, count)
    if value > best_value: best, best_value = (J, count), value
  return here, best, best_value


def main():
  from trials_versus_samples import per_sample_information

  cache = records.HERE / "allocation_clean_fits.json"
  jobs = [(d, i, a) for d in records.DATASETS
          for i in range(_trials(d)[5].shape[1]) for a in (False, True)]
  if cache.exists():
    stored = json.loads(cache.read_text())
    table = {(d, int(i), a == "True"): v
             for key, v in stored.items() for d, i, a in [key.split("|")]}
    print(f"  {len(table)} per-trial fits read from cache")
  else:
    print(f"  {len(jobs)} per-trial fits, raw and clock-aligned ...", flush=True)
    with records.pool(len(jobs)) as pool:
      table = dict(pool.map(one, jobs))
    cache.write_text(json.dumps({f"{d}|{i}|{a}": v for (d, i, a), v in table.items()}, indent=1))

  summary = {}
  print("\n  what aligning each trial's clock does to the between-bubble spread\n")
  print(f"  {'record':13s} {'J':>3s} " + " ".join(f"{a + ' raw':>13s} {a + ' aligned':>15s}"
                                                  for a in AXES))
  cleaned = {}
  for dataset in records.DATASETS:
    count = _trials(dataset)[5].shape[1]
    got = {}
    for aligned in (False, True):
      rows = [table[(dataset, i, aligned)] for i in range(count)
              if "failed" not in table[(dataset, i, aligned)]]
      if len(rows) < 4: continue
      got[aligned] = covariance_of(rows)
    if len(got) < 2: continue
    cleaned[dataset] = got
    cells = " ".join(f"{got[False][1][k]:13.1%} {got[True][1][k]:15.1%}"
                     for k in range(len(AXES)))
    print(f"  {dataset:13s} {count:3d} {cells}")
  print("  Standard deviations in log coordinates. A drop is scatter that was the clock and")
  print("  the parameters were absorbing; no drop is scatter that is genuinely the material.")

  print("\n  and what that does to the allocation the chapter recommends\n")
  print(f"  {'record':13s} {'Sigma':>9s} {'as run':>16s} {'best split':>16s} {'gain':>8s}")
  for dataset in records.DATASETS:
    if dataset not in cleaned: continue
    fisher, samples = per_sample_information(dataset)
    trials = _trials(dataset)[5].shape[1]
    entry = {}
    for label, aligned in (("raw", False), ("aligned", True)):
      covariance = cleaned[dataset][aligned][0]
      here, best, value = allocation_gain(covariance, fisher, samples, trials)
      entry[label] = {"as_run": here, "best": list(best), "best_value": value,
                      "gain": value - here}
      print(f"  {dataset if label == 'raw' else '':13s} {label:>9s} "
            f"J={trials:3d}, N={samples:4d} J={best[0]:4d}, N={best[1]:4d} "
            f"{value - here:8.3f}")
    summary[dataset] = entry
    lost = entry["raw"]["gain"] - entry["aligned"]["gain"]
    print(f"  {'':13s} {'->':>9s} aligning the clock removes {lost:+.3f} nats of the gain")

  print("\n  The gain exists only because Sigma_theta is non-zero: as Sigma -> 0 the information")
  print("  becomes J F_N, the product JN is all that matters, and there is nothing to allocate.")
  print("  So whatever share of Sigma was the clock was never a reason to buy bubbles.")

  json.dump(summary, open(records.HERE / "allocation_clean.json", "w"), indent=1)


if __name__ == "__main__":
  main()
