r"""Does the afterbounce signature survive the fact that a maximum of noisy data is biased high?

Everything built on the afterbounce ratios assumes the extracted maxima are the bubble's. They
are the largest of a handful of noisy samples, and the largest of $n$ noisy samples is biased
UPWARD by an amount that grows as the signal shrinks. The later bounces here oscillate over a
range comparable to the sample-to-sample scatter, so the bias is largest exactly where
`shape_anomaly.py` locates its anomaly, and the model carries no such bias because it is
noise-free. That is a mechanism which could manufacture the entire signature.

THE BETWEEN-TRIAL SPREAD DOES NOT PROTECT AGAINST IT. Dividing by the standard error of the
mean handles RANDOM error and this is a SYSTEMATIC one: every trial's maximum is biased the
same way, so averaging $39$ of them converges to the biased value rather than the true one,
and the error bar shrinks while the offset does not. A bias that survives averaging is not
visible in any statistic computed from the scatter.

SO THE CONTROL IS TO GIVE THE MODEL THE SAME DISADVANTAGE. Noise of the measured magnitude is
added to the model trace, the identical extraction is run, and the resulting ratios are
compared against the noise-free model ratios. Whatever moves is extraction bias rather than
physics. If the biased model reproduces the measured sequence, the signature was never in the
data; if it moves the model a little and the measurement remains far away, the signature
survives with a correction attached.

THE FIRST NOISE ESTIMATE WAS WRONG AND THE WAY IT WAS WRONG IS THE USEFUL PART. A
second-difference estimator, $\operatorname{sd}(\Delta^2 x) = \sqrt{6}\,\sigma$ for white
noise on a smooth signal, returns $0.0067$ on these datasets. Applied to the NOISE-FREE model
on the same grid it returns $0.0136$, which is larger. At $20$ samples per bounce the
estimator is measuring the signal's own curvature, so it is an upper bound and not a
measurement, and injecting at that level overstates the artifact rather than reproducing it.

SO THE NOISE IS CALIBRATED FROM THE DATA'S OWN ROBUSTNESS INSTEAD. Re-extracting the MEASURED
ratios at half-widths of $2$, $3$ and $4$ changes them by $0.000$ on the first three bounces
of every dataset, and by $0.004$ at worst on \SI{15}{\celsius} and \SI{23}{\celsius}
throughout. Injecting noise into the model and doing the same shows that invariance is only
achieved below $\sigma \approx 0.001$: at $0.001$ the extraction already moves by $0.02$ to
$0.03$, at $0.002$ by $0.12$, and at the second-difference value by $0.33$ to $0.38$. The
observed invariance therefore bounds the per-sample noise at roughly a seventh of the naive
estimate, and the bias is evaluated there.

WHICH TURNS A THREAT INTO A BOUND. The order-invariance of the measurement is not a
convenience: it is the measurement that rules the artifact out, because an extraction
corrupted by noise cannot be insensitive to how wide a window it searches.
"""

import json

import numpy as np

import records
from bounce_sweep import DATA, FILES, MAX_RATIO, RATIO, sequence
from frequency_space import KEEP, SPAN, _families, measured

BASE_RHO = 1064.0
DRAWS = 400
# calibrated from the order-invariance of the measured ratios, not from second differences
SIGMA_BOUND = 0.001
ORDERS = (2, 3, 4)          # the extraction half-width, swept: a robust signal survives it


def noise_level(dataset):
  """Per-sample high-frequency scatter, from second differences, per trial then median."""
  table = np.loadtxt(DATA / FILES[dataset], delimiter=",", ndmin=2)
  trials = table[:, 1:].T
  keep = ~(trials > MAX_RATIO).any(axis=0) & (trials.std(axis=0, ddof=1) > 0.0)
  trials = trials[:, keep] if trials.shape[1] == len(keep) else trials
  out = []
  for row in np.loadtxt(DATA / FILES[dataset], delimiter=",", ndmin=2)[:, 1:].T:
    d2 = np.diff(row, n=2)
    out.append(float(np.std(d2, ddof=1) / np.sqrt(6.0)))
  return float(np.median(out))


def _model_trace(dataset, on_grid):
  import pyimr

  times, _, _, maximum, stretch = records.load(dataset)
  m = json.load(open(records.HERE / "per_trial_fits.json"))[dataset]["median"]
  product = m["galpha"]
  material = pyimr.QuadraticZener(float(np.sqrt(product * RATIO)), m["mu"], m["lambda1"], 0.0,
                                  float(np.sqrt(product / RATIO)))
  physics = pyimr.PhysicalParameters(medium_density_kg_m3=BASE_RHO)
  config = pyimr.SimulationConfig(maximum, maximum / stretch, material,
                                  dynamics="keller-miksis", rtol=1e-9, atol=1e-11,
                                  max_steps=2_000_000, physics=physics)
  grid = times if on_grid else np.linspace(times[0], times[-1], 20001)
  trace = np.asarray(pyimr.simulate(grid, config).radius_ratio, dtype=float)
  phase = (grid - grid[0]) / (grid[-1] - grid[0]) * SPAN
  return trace, phase


def one(dataset):
  obs = measured(dataset)
  n = min(KEEP, len(obs["ratio"]))
  sigma = SIGMA_BOUND
  naive = noise_level(dataset)
  # the model is read on the DATASET's own grid here, because that is the grid the bias acts on
  clean, phase = _model_trace(dataset, on_grid=True)
  base = sequence(clean, phase)
  if base is None: return dataset, None

  out = {"sigma": sigma, "naive_estimate": naive, "events": obs["events"], "measured": obs["ratio"][:n],
         "clean": None, "by_order": {}}
  fam = _families(base["amplitudes"], base["times"])[0]
  out["clean"] = [float(x) for x in fam[:n]] if len(fam) >= n else None

  rng = np.random.default_rng(20260816)
  for order in ORDERS:
    draws = []
    for _ in range(DRAWS):
      noisy = clean + rng.normal(0.0, sigma, size=clean.shape)
      s = sequence(noisy, phase, order=order) if "order" in sequence.__code__.co_varnames \
          else _sequence_order(noisy, phase, order)
      if s is None: continue
      f = _families(s["amplitudes"], s["times"])[0]
      if len(f) >= n: draws.append(f[:n])
    if not draws: continue
    arr = np.array(draws, dtype=float)
    out["by_order"][order] = {"mean": arr.mean(axis=0).tolist(),
                              "sd": arr.std(axis=0, ddof=1).tolist(),
                              "draws": len(draws)}
  return dataset, out


def _sequence_order(trace, tau, order):
  """`sequence` with an explicit extraction half-width."""
  from bounce_sweep import maxima
  _, amps, times = maxima(trace, tau, order=order)
  if len(amps) < 3: return None
  return {"amplitudes": amps, "times": times,
          "ratios": [amps[k + 1] / amps[k] for k in range(len(amps) - 1)],
          "periods": [times[k + 1] - times[k] for k in range(len(times) - 1)]}


def main():
  print("  The largest of a few noisy samples is biased high, and the bias grows as the")
  print("  bounces shrink. The model carries no such bias. This gives it the same one.\n")

  with records.pool(len(records.DATASETS)) as pool:
    got = dict(pool.map(one, list(records.DATASETS)))

  summary = {}
  for dataset in records.DATASETS:
    v = got[dataset]
    if v is None or not v["by_order"]:
      print(f"  {dataset}: extraction failed"); continue
    summary[dataset] = v
    n = len(v["measured"])
    print(f"  ==== {dataset}, {v['events']} events, per-sample noise {v['sigma']:.4f} ====")
    print(f"  {'bounce':>16s} " + " ".join(f"{k+1:>8d}" for k in range(n)))
    print(f"  {'measured':>16s} " + " ".join(f"{x:8.3f}" for x in v["measured"]))
    if v["clean"]:
      print(f"  {'model, no noise':>16s} " + " ".join(f"{x:8.3f}" for x in v["clean"]))
    for order, d in v["by_order"].items():
      print(f"  {'model + noise':>16s} " + " ".join(f"{x:8.3f}" for x in d["mean"])
            + f"   (order {order}, {d['draws']} draws)")
    if v["clean"] and 2 in v["by_order"]:
      shift = np.array(v["by_order"][2]["mean"]) - np.array(v["clean"])
      gap = np.array(v["measured"]) - np.array(v["clean"])
      print(f"  {'bias from noise':>16s} " + " ".join(f"{x:+8.3f}" for x in shift))
      print(f"  {'measured - clean':>16s} " + " ".join(f"{x:+8.3f}" for x in gap))
      share = np.where(np.abs(gap) > 1e-9, shift / np.where(np.abs(gap) > 1e-9, gap, 1.0), np.nan)
      print(f"  {'bias/gap':>16s} " + " ".join(f"{x:8.2f}" for x in share))
      summary[dataset]["bias_share"] = [None if not np.isfinite(x) else float(x) for x in share]
    print()

  print("  ---- what it says ----\n")
  for dataset, v in summary.items():
    share = [x for x in (v.get("bias_share") or []) if x is not None]
    if not share: continue
    worst = max(share)
    print(f"  {dataset:>13s}: noise bias accounts for {100*np.median(share):5.1f} percent of the "
          f"measured-minus-model gap in the median bounce, {100*worst:5.1f} at worst")
  print("\n  A share near 1 means the signature is the extraction. Near 0 means the extraction")
  print("  is innocent and the gap is the model's. The sweep over the extraction half-width is")
  print("  there because a signal that changes with it was never a property of the bubble.")
  json.dump(summary, open(records.HERE / "bounce_noise.json", "w"), indent=1)


if __name__ == "__main__":
  main()
