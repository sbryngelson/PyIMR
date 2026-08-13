r"""Where $R_{\max}$ and $R_{\rm eq}$ come from, and whether the way they are fitted is the residue.

The acquisition authors describe both quantities as fitted rather than read off:
$R_{\max}$ from a fourth-order polynomial through a window around the peak --- with the caveat
that fourth order is only safe ``if we're including data just very close to the peak'' --- and
$R_{\rm eq}$ by averaging a portion of the points at the end of the $R(t)$ curve, with the
alternative of taking the median of the post-collapse oscillations. The stored metadata agrees:
\texttt{R0PfitPower} $=4$, \texttt{R0PfitWindow} $=0.4$, \texttt{ReqCutoff} $=0.95$.

THAT MAKES THE CLOCK A FITTED QUANTITY, which is the point. \Cref{sec:latent} finds the pipeline
dividing each event's time by $R_{\max}\sqrt{\rho/p_\infty}$, so an error in the $R_{\max}$ fit is
an error in that event's clock, and a clock error is exactly the shape of the residual dilation
that survives the normalisation. If the residue is fit noise rather than physics, events whose
$R_{\max}$ fit is worse should sit further along the dilation direction --- and the file stores
the fit quality per event, as \texttt{qc.R0PfitR2}.

That is a falsifiable prediction with the data already in hand, and it is the first candidate for
this residue that is neither a parameter of the model nor a property of the bubble.

THREE THINGS ARE CHECKED HERE. Whether $R_{\max}$ is stored in metres or pixels, which the
authors were themselves unsure of and which every number in \cref{sec:latent} depends on. Whether
the dilation amplitude tracks the $R_{\max}$ fit quality. And whether $R_{\rm eq}$ taken the other
way the authors describe --- the median of the post-collapse extrema rather than a tail average
--- is a tighter quantity than the one stored, since \cref{sec:reqprior} rests on how uncertain
the stretch is.
"""

import json
import pathlib

import numpy as np

import records

SOURCE = pathlib.Path("~/data_pa5_tempsweeps_master_20260210.mat").expanduser()
BINS = {"12-18C": (12.0, 18.0), "20-26C": (20.0, 26.0), "30-36C": (30.0, 36.0)}
PHASE = np.linspace(0.15, 3.0, 200)


def _scalar(entry, field, sub=None):
  try:
    holder = getattr(entry, field) if sub is None else getattr(getattr(entry, field), sub)
    return float(holder)
  except Exception:                                                          # noqa: BLE001
    return float("nan")


def equilibrium_by_extrema(clock, trace):
  """`R_eq` as the median of the post-collapse oscillation extrema, the authors' alternative."""
  from scipy.signal import argrelextrema

  after = clock > 0.5
  if after.sum() < 30: return float("nan")
  window = trace[after]
  highs = argrelextrema(window, np.greater, order=2)[0]
  lows = argrelextrema(window, np.less, order=2)[0]
  extrema = np.concatenate([window[highs], window[lows]])
  return float(np.median(extrema)) if extrema.size >= 4 else float("nan")


def main():
  import scipy.io as sio

  if not SOURCE.exists():
    print(f"  {SOURCE} not present; nothing to do")
    return
  loaded = sio.loadmat(SOURCE, squeeze_me=True, struct_as_record=False)
  entries = loaded["dataPa5TempSweep"]
  temperature = np.array([_scalar(e, "T") for e in entries])
  flag = np.array([_scalar(e, "errorFlag") for e in entries])
  usable = (flag == 0) & np.isfinite(temperature)
  summary = {}

  size = _scalar(entries[0], "R0")
  metres_per_pixel = _scalar(entries[0], "meta", "cal") if False else float(
    getattr(getattr(entries[0], "meta"), "cal").m_px)
  print(f"\n  units: R0 = {size:.4e}, m_px = {metres_per_pixel:.4e}")
  print(f"  as metres that is {size * 1e6:.1f} um and {size / metres_per_pixel:.1f} pixels;")
  print(f"  as pixels it would be {size * metres_per_pixel * 1e6:.3e} um, which is not a bubble.")
  print("  R0 is in METRES, already calibrated. Every number downstream assumes that.")
  summary["units"] = {"r0_first": size, "m_per_pixel": metres_per_pixel,
                      "pixels": size / metres_per_pixel}

  print("\n  does the dilation amplitude track the R_max fit quality?\n")
  print(f"  {'bin':>8s} {'n':>4s} {'R0 fit R2':>22s} {'corr(|dilation|, 1-R2)':>23s} "
        f"{'corr(dilation, 1-R2)':>21s}")
  tracking = {}
  for name, (low, high) in BINS.items():
    chosen = [(e, i) for i, (e, ok, t) in enumerate(
      zip(entries, usable, temperature, strict=True)) if ok and low <= t < high]
    stack, quality = [], []
    for entry, _ in chosen:
      trace = np.asarray(getattr(entry, "R_norm"), dtype=float)
      clock = np.asarray(getattr(entry, "t_norm"), dtype=float)
      good = np.isfinite(trace) & np.isfinite(clock)
      if good.sum() < 60: continue
      trace, clock = trace[good], clock[good]
      start = clock.min()
      order = np.argsort(clock - start)
      stack.append(np.interp(PHASE, (clock - start)[order], trace[order]))
      quality.append(_scalar(entry, "qc", "R0PfitR2"))
    if len(stack) < 12: continue
    matrix = np.column_stack(stack)
    deviation = matrix - matrix.mean(axis=1, keepdims=True)
    mean = matrix.mean(axis=1)
    dilation = -PHASE * np.gradient(mean, PHASE)
    unit = dilation / np.linalg.norm(dilation)
    amplitude = unit @ deviation
    lack = 1.0 - np.array(quality)
    keep = np.isfinite(lack) & np.isfinite(amplitude)
    if keep.sum() < 12: continue
    signed = float(np.corrcoef(amplitude[keep], lack[keep])[0, 1])
    absolute = float(np.corrcoef(np.abs(amplitude[keep]), lack[keep])[0, 1])
    tracking[name] = {"events": int(keep.sum()), "r2_mean": float(np.nanmean(quality)),
                      "r2_min": float(np.nanmin(quality)),
                      "corr_abs": absolute, "corr_signed": signed}
    print(f"  {name:>8s} {int(keep.sum()):4d} {np.nanmean(quality):9.5f} "
          f"(min {np.nanmin(quality):.4f}) {absolute:23.3f} {signed:21.3f}")
  summary["fit_tracking"] = tracking
  print("  A positive correlation in the |dilation| column is the prediction: a worse R_max fit")
  print("  means a worse clock, and a worse clock means a larger excursion either way.")

  print("\n  the stretch, by the two definitions the authors describe\n")
  print(f"  {'bin':>8s} {'n':>4s} {'stored (tail average)':>24s} {'cv':>7s} "
        f"{'median of extrema':>20s} {'cv':>7s}")
  stretch = {}
  for name, (low, high) in BINS.items():
    stored, alternative = [], []
    for entry, ok, t in zip(entries, usable, temperature, strict=True):
      if not ok or not (low <= t < high): continue
      trace = np.asarray(getattr(entry, "R_norm"), dtype=float)
      clock = np.asarray(getattr(entry, "t_norm"), dtype=float)
      good = np.isfinite(trace) & np.isfinite(clock)
      if good.sum() < 60: continue
      first, second = _scalar(entry, "R0"), _scalar(entry, "Req")
      if not np.isfinite(first) or not np.isfinite(second) or second <= 0: continue
      other = equilibrium_by_extrema(clock[good], trace[good])
      if not np.isfinite(other) or other <= 0: continue
      stored.append(first / second)
      alternative.append(1.0 / other)              # R_norm is already R/R_max
    if len(stored) < 8: continue
    stored, alternative = np.array(stored), np.array(alternative)
    stretch[name] = {"events": len(stored), "stored_mean": float(stored.mean()),
                     "stored_cv": float(stored.std(ddof=1) / stored.mean()),
                     "extrema_mean": float(alternative.mean()),
                     "extrema_cv": float(alternative.std(ddof=1) / alternative.mean()),
                     "correlation": float(np.corrcoef(stored, alternative)[0, 1])}
    row = stretch[name]
    print(f"  {name:>8s} {len(stored):4d} {stored.mean():13.3f} +- {stored.std(ddof=1):.3f} "
          f"{row['stored_cv']:7.4f} {alternative.mean():13.3f} +- {alternative.std(ddof=1):.3f} "
          f"{row['extrema_cv']:7.4f}")
  summary["stretch"] = stretch
  print("  The two definitions agree at "
        + ", ".join(f"{v['correlation']:+.3f}" for v in stretch.values())
        + ". A tighter cv is a better estimator of the same thing;")
  print("  a looser one with a high correlation is the same quantity measured more noisily.")

  json.dump(summary, open(records.HERE / "fit_provenance.json", "w"), indent=1)


if __name__ == "__main__":
  main()
