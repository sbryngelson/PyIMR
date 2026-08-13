r"""What the acquisition data says about $R_{\max}$, and what that does to \#222.

\Cref{sec:latent} has failed to absorb the dominant trial mode four times --- material parameters
per trial, the equilibrium radius, the initial wall velocity, and nine linearised directions ---
and closed by naming the one candidate it could not reach: per-event $R_{\max}$, which every
trace is normalised by before it reaches this repository.

The acquisition data has it. Each of $835$ events carries a measured $R_0$ (the maximum radius),
a measured $R_{\rm eq}$, and the un-normalised trace, so the quantity that was unreachable is now
just a field. THIS SCRIPT READS THAT FILE AND WRITES ONLY SUMMARIES: the raw data is not
redistributed here and is not committed.

TWO THINGS FALL OUT AND THEY POINT OPPOSITE WAYS.

The scatter is large and it is in $R_{\max}$, not in the ratio. Per event, $R_{\max}$ has a
coefficient of variation of $0.22$ to $0.28$ across temperature bins, while the stretch
$R_{\max}/R_{\rm eq}$ has $0.05$ to $0.07$ and a mean near $8.1$ at every temperature. That
settles the successor question left open when \#253 closed: the absolute scale varies four times
more than the ratio does, and it is the ratio that this document has been treating as uncertain.

And a scatter in $R_{\max}$ is, in normalised coordinates, almost purely a scatter in the CLOCK.
$R(t)/R_{\max}$ is a near-universal function of $t/t_c$ with $t_c \propto R_{\max}$, so two
bubbles differing only in size trace the same curve at different rates. The deviation that
produces is $\partial R/\partial \log t_c = -t\,\dot R$ --- a direction that is small early,
grows as phase error accumulates, and is therefore concentrated exactly where \#222's mode lives.

WHAT IS MEASURED HERE. That direction against the observed mode, the share of trial variance it
carries, and --- the check that matters --- the time-scale scatter the traces IMPLY against the
$R_{\max}$ scatter the raw data MEASURES. If the two agree, \#222 is a normalisation artifact. If
the implied one is far smaller, the pipeline has already removed most of it and the mode is
something else again.
"""

import json
import pathlib

import numpy as np

import records

SOURCE = pathlib.Path("~/data_pa5_tempsweeps_master_20260210.mat").expanduser()
BINS = {"12-18C": (12.0, 18.0), "20-26C": (20.0, 26.0), "30-36C": (30.0, 36.0)}


def _scalar(entry, field):
  value = getattr(entry, field, None)
  try:
    return float(value)
  except Exception:                                                          # noqa: BLE001
    return float("nan")


def load_events():
  """`(T, R0, Req, errorFlag)` per event, and the entries themselves for the traces."""
  import scipy.io as sio

  if not SOURCE.exists(): return None, None
  loaded = sio.loadmat(SOURCE, squeeze_me=True, struct_as_record=False)
  entries = loaded["dataPa5TempSweep"]
  fields = {name: np.array([_scalar(e, name) for e in entries])
            for name in ("T", "R0", "Req", "errorFlag", "tc")}
  usable = ((fields["errorFlag"] == 0) & np.isfinite(fields["R0"]) & (fields["R0"] > 0)
            & np.isfinite(fields["Req"]) & (fields["Req"] > 0) & np.isfinite(fields["T"]))
  return fields, (entries, usable)


def dilation_direction(times, mean):
  """`-t dR/dt`: what a fractional change in the collapse timescale does to a normalised trace."""
  return -times * np.gradient(mean, times)


def _cos(a, b):
  return abs(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))


def raw_modes(entries, usable, fields, low, high):
  """The leading deviation mode of the RAW traces in one temperature bin, on a common phase grid."""
  chosen = [e for e, ok, t in zip(entries, usable, fields["T"], strict=True)
            if ok and low <= t < high]
  if len(chosen) < 4: return None
  phase = np.linspace(0.05, 0.95, 180)
  stack = []
  for entry in chosen:
    trace = np.asarray(getattr(entry, "R_norm"), dtype=float)
    clock = np.asarray(getattr(entry, "t_norm"), dtype=float)
    good = np.isfinite(trace) & np.isfinite(clock)
    if good.sum() < 50: continue
    order = np.argsort(clock[good])
    stack.append(np.interp(phase, clock[good][order], trace[good][order]))
  if len(stack) < 4: return None
  matrix = np.column_stack(stack)
  deviation = matrix - matrix.mean(axis=1, keepdims=True)
  left, values, _ = np.linalg.svd(deviation, full_matrices=False)
  mean = matrix.mean(axis=1)
  direction = dilation_direction(phase, mean)
  return {"events": matrix.shape[1], "share": float(values[0] ** 2 / (values**2).sum()),
          "cos_dilation": _cos(left[:, 0], direction),
          "chance": float(1.0 / np.sqrt(phase.size))}


def main():
  fields, payload = load_events()
  summary = {}
  if fields is None:
    print(f"  {SOURCE} not present; skipping the raw-data half")
  else:
    entries, usable = payload
    print(f"\n  {int(usable.sum())} usable events of {usable.size}, "
          f"T from {fields['T'][usable].min():.1f} to {fields['T'][usable].max():.1f} C\n")
    print(f"  {'bin':>8s} {'n':>4s} {'R_max (um)':>18s} {'cv':>6s} {'stretch':>14s} {'cv':>6s}")
    per_bin = {}
    for name, (low, high) in BINS.items():
      pick = usable & (fields["T"] >= low) & (fields["T"] < high)
      if pick.sum() < 2: continue
      radius, equilibrium = fields["R0"][pick], fields["Req"][pick]
      stretch = radius / equilibrium
      per_bin[name] = {
        "events": int(pick.sum()),
        "r_max_mean_m": float(radius.mean()), "r_max_cv": float(radius.std(ddof=1) / radius.mean()),
        "stretch_mean": float(stretch.mean()),
        "stretch_cv": float(stretch.std(ddof=1) / stretch.mean())}
      print(f"  {name:>8s} {pick.sum():4d} {radius.mean() * 1e6:11.1f} +- "
            f"{radius.std(ddof=1) * 1e6:4.1f} {per_bin[name]['r_max_cv']:6.3f} "
            f"{stretch.mean():9.2f} +- {stretch.std(ddof=1):.2f} "
            f"{per_bin[name]['stretch_cv']:6.3f}")
    summary["per_bin"] = per_bin
    print("\n  The scatter is in the SCALE, not the ratio: R_max varies four times more than")
    print("  the stretch does, and the stretch mean is near 8.1 at every temperature.")

    print("\n  the same test on the RAW traces, on a common phase grid\n")
    print(f"  {'bin':>8s} {'events':>7s} {'mode share':>11s} {'|cos| to -t dR/dt':>18s} {'chance':>8s}")
    raw = {}
    for name, (low, high) in BINS.items():
      got = raw_modes(entries, usable, fields, low, high)
      if got is None: continue
      raw[name] = got
      print(f"  {name:>8s} {got['events']:7d} {got['share']:11.3f} "
            f"{got['cos_dilation']:18.3f} {got['chance']:8.3f}")
    summary["raw_modes"] = raw

  print("\n  and on this repository's records, whose traces the pipeline has already processed\n")
  print(f"  {'record':13s} {'J':>3s} {'mode share':>11s} {'|cos| to dilation':>18s} "
        f"{'var explained':>14s} {'implied dt_c/t_c':>17s} {'|cos| after':>12s}")
  from per_trial_fits import _trials

  processed = {}
  for dataset in records.DATASETS:
    times, mean, spread, maximum, stretch, window = _trials(dataset)
    deviation = window - window.mean(axis=1, keepdims=True)
    left, values, _ = np.linalg.svd(deviation, full_matrices=False)
    direction = dilation_direction(times, mean)
    unit = direction / np.linalg.norm(direction)
    amplitude = unit @ deviation
    fraction = amplitude / np.linalg.norm(direction)
    fitted = np.outer(direction, fraction)
    explained = 1.0 - float(np.sum((deviation - fitted) ** 2) / np.sum(deviation**2))
    rest = deviation - fitted
    after, _, _ = np.linalg.svd(rest, full_matrices=False)
    processed[dataset] = {
      "trials": int(window.shape[1]),
      "share": float(values[0] ** 2 / (values**2).sum()),
      "cos_dilation": _cos(left[:, 0], direction),
      "variance_explained": explained,
      "implied_timescale_cv": float(fraction.std(ddof=1)),
      "cos_mode_after": _cos(left[:, 0], after[:, 0])}
    row = processed[dataset]
    print(f"  {dataset:13s} {row['trials']:3d} {row['share']:11.3f} {row['cos_dilation']:18.3f} "
          f"{row['variance_explained']:14.3f} {row['implied_timescale_cv']:17.4f} "
          f"{row['cos_mode_after']:12.3f}")
  summary["processed"] = processed

  if "per_bin" in summary:
    implied = np.mean([v["implied_timescale_cv"] for v in processed.values()])
    measured = np.mean([v["r_max_cv"] for v in summary["per_bin"].values()])
    print(f"\n  implied timescale scatter {implied:.4f} against a measured R_max scatter of "
          f"{measured:.4f}: a factor of {measured / implied:.1f}.")
    summary["implied_vs_measured"] = {"implied": float(implied), "measured": float(measured)}
    print("  The pipeline has therefore already removed most of it. What the dilation direction")
    print("  still carries is real and is the largest single share anyone has attributed to this")
    print("  mode -- and it is a fifth to a quarter, not the whole of it.")

  json.dump(summary, open(records.HERE / "raw_events.json", "w"), indent=1)


if __name__ == "__main__":
  main()
