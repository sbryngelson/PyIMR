r"""Was the Zener relaxation arm standing in for thermal damping all along?

\Cref{sec:vapour} leaves the vapour test unfinished for a specific reason: refitted with the
thermal physics on, half the records drive $\lambda_1$ to the floor of a seven-decade prior, so
the corrected optimum sits on the qSLS family's own boundary and the $\hat\delta$ taken there
is not comparable with the interior one \cref{sec:universal} reports. The form the data are
asking for is the $\lambda_1 \to 0$ limit, and this document already carries it: \texttt{qKV},
which \cref{fig:nesting} notes is exactly \texttt{qSLS} at zero relaxation time.

THE PREDICTION IS STATED BEFORE THE RUN, because the interesting outcome is the one that would
embarrass it. Isothermally, dropping the relaxation arm is catastrophic: \cref{sec:universal}
reports \texttt{qKV} fitting at $\chi^2/N$ of $4$ to $13$ against $0.4$ to $1.4$ for
\texttt{qSLS}. If the arm is genuinely doing the work of the absent thermal damping, then
switching that damping ON should largely REMOVE the penalty for dropping the arm --- qKV with
vapour should fit far better than qKV without it, and land near where qSLS lands. If instead
qKV stays at $\chi^2/N \sim 4$ to $13$ with the thermal physics on, then $\lambda_1 \to 0$ was
the search leaving the family for some other reason and the reading in \cref{sec:vapour} is
wrong.

Both cells are fitted GLOBALLY over the wide prior box rather than warm-started, because a warm
box is what made the previous answer uninterpretable, and qKV has three axes rather than four so
a global search is affordable. Every fit reports where in its box it landed and whether the
residual is orthogonal to the span, so a boundary answer is visible instead of silent.
"""

import json

import numpy as np

import records
from lackoffit import WIDE
from shape_error import seed_for
from universal_delta import GRID, _surrogate

THERMAL = {"bubtherm": 1, "Nt": 25, "masstrans": 1, "vapor": 1}
ORDER = (*records.DATASETS, *records.PAAM)
CELLS = (("isothermal", {}, 6, 150), ("vapour", THERMAL, 4, 100))
WALL, DRAWS = 0.02, 2000


def one(job):
  from pyimr.noise import discrepancy
  from pyimr.selection import STANDARD_MODELS, evaluate_at, fit_candidate, physical_from_unit

  dataset, label, extra, starts, evaluations = job
  times, mean, spread, maximum, _ = records.load(dataset)
  stretch = json.load(open(records.HERE / "paam_stretch.json"))[dataset]["argmin"]
  candidate = STANDARD_MODELS["qKV"]
  box = {a: WIDE[a] for a in candidate.axes}
  solve = records.solver(times, maximum, stretch, max_steps=400_000, **extra)
  try:
    fit = fit_candidate(candidate, solve, mean, spread, bounds=box, starts=starts,
                        max_evaluations=evaluations)
    values = physical_from_unit(candidate.axes, fit.unit, box)
    fitted = dict(zip(candidate.axes, (float(v) for v in values), strict=True))
    base = np.asarray(evaluate_at(candidate, solve, fitted)[0], dtype=float)
  except Exception as failure:                                               # noqa: BLE001
    return (dataset, label), {"failed": f"{type(failure).__name__}: {failure}"[:110]}

  where = {a: float(np.log(fitted[a] / box[a][0]) / np.log(box[a][1] / box[a][0]))
           for a in candidate.axes}
  error_bar = spread / np.sqrt(records.trial_count(dataset))
  step, columns = 1e-4, []
  for axis in candidate.axes:
    up = dict(fitted, **{axis: fitted[axis] * (1 + step)})
    dn = dict(fitted, **{axis: fitted[axis] * (1 - step)})
    try:
      hi = np.asarray(evaluate_at(candidate, solve, up)[0], dtype=float)
      lo = np.asarray(evaluate_at(candidate, solve, dn)[0], dtype=float)
    except Exception:                                                        # noqa: BLE001
      return (dataset, label), {"failed": "jacobian"}
    columns.append((hi - lo) / (2 * step * error_bar))
  jacobian = np.column_stack(columns)
  split = discrepancy((mean - base) / error_bar, jacobian)
  tau = times / (maximum * np.sqrt(1064.0 / 101325.0))

  def onto(values):
    got = np.interp(GRID, tau[: values.size], values)
    got = got - got.mean()
    return got / (np.linalg.norm(got) + 1e-30)

  curve = np.asarray(split.identifiable, dtype=float)
  return (dataset, label), {
    "chi2_per_n": float(fit.chi_squared), "fitted": fitted, "where": where,
    "pinned": [a for a, u in where.items() if u < WALL or u > 1.0 - WALL],
    "at_optimum": bool(split.at_optimum),
    "leak": float(split.absorbed / (np.hypot(split.absorbed, split.size) + 1e-30)),
    "curve": onto(curve).tolist(), "span": [onto(c).tolist() for c in jacobian.T]}


def agreement(curves, spans, good, rng):
  """Across/within medians against the constrained null, as sec:universal builds it."""
  def constrained(d):
    out = _surrogate(rng, curves[d])
    basis, _ = np.linalg.qr(spans[d].T)
    out = out - basis @ (basis.T @ out)
    return out / (np.linalg.norm(out) + 1e-30)

  null = np.array([abs(constrained(good[a]) @ constrained(good[b]))
                   for a, b in rng.choice(len(good), size=(DRAWS, 2)) if a != b])
  cut = float(np.percentile(null, 95))
  across = [abs(curves[a] @ curves[b]) for i, a in enumerate(good) for b in good[i + 1:]
            if (a in records.DATASETS) != (b in records.DATASETS)]
  within = [abs(curves[a] @ curves[b]) for i, a in enumerate(good) for b in good[i + 1:]
            if (a in records.DATASETS) == (b in records.DATASETS)]
  return across, within, cut


def main():
  jobs = [(d, label, extra, s, e) for d in ORDER for label, extra, s, e in CELLS]
  print(f"  qKV fitted globally over the wide prior box, {len(jobs)} fits\n")
  with records.pool(len(jobs)) as pool:
    got = dict(pool.map(one, jobs))

  print(f"  {'record':>16s} " + "  ".join(
    f"{label + ' chi2/N':>17s} {'pinned':>14s}" for label, *_ in CELLS))
  for dataset in ORDER:
    cells = []
    for label, *_ in CELLS:
      v = got[(dataset, label)]
      cells.append(f"{'FAILED':>17s} {'':>14s}" if "failed" in v else
                   f"{v['chi2_per_n']:17.4f} "
                   f"{(','.join(v['pinned']) if v['pinned'] else '-'):>14s}")
    print(f"  {dataset:>16s} " + "  ".join(cells))

  print("\n  ---- the prediction ----\n")
  ok = {label: [d for d in ORDER if "failed" not in got[(d, label)]] for label, *_ in CELLS}
  cold = [got[(d, "isothermal")]["chi2_per_n"] for d in ok["isothermal"]]
  hot = [got[(d, "vapour")]["chi2_per_n"] for d in ok["vapour"]]
  print(f"  qKV isothermal: chi2/N {min(cold):.2f} to {max(cold):.2f}, "
        f"median {np.median(cold):.2f}   (sec:universal reports 4 to 13)")
  print(f"  qKV with vapour: chi2/N {min(hot):.2f} to {max(hot):.2f}, "
        f"median {np.median(hot):.2f}")
  paired = [got[(d, 'isothermal')]['chi2_per_n'] / got[(d, 'vapour')]['chi2_per_n']
            for d in ORDER if d in ok["isothermal"] and d in ok["vapour"]]
  print(f"  paired improvement from switching the thermal physics on: "
        f"median {np.median(paired):.2f}x, {min(paired):.2f}x to {max(paired):.2f}x")

  rng = np.random.default_rng(seed_for("kelvin_voigt"))
  summary = {}
  for label, *_ in CELLS:
    good = ok[label]
    if len(good) < 4: continue
    curves = {d: np.asarray(got[(d, label)]["curve"], dtype=float) for d in good}
    spans = {d: np.asarray(got[(d, label)]["span"], dtype=float) for d in good}
    across, within, cut = agreement(curves, spans, good, rng)
    boundary = sum(1 for d in good if got[(d, label)]["pinned"])
    settled = sum(1 for d in good if got[(d, label)]["at_optimum"])
    summary[label] = {"across_median": float(np.median(across)),
                      "within_median": float(np.median(within)), "null_95": cut,
                      "across_above": sum(1 for r in across if r > cut), "across_n": len(across),
                      "pinned": boundary, "at_optimum": settled, "n": len(good)}
    print(f"\n  {label:>10s}: ACROSS median {np.median(across):.3f} "
          f"({sum(1 for r in across if r > cut)} of {len(across)} above null {cut:.3f}), "
          f"within {np.median(within):.3f}")
    print(f"  {'':>10s}  {boundary} of {len(good)} on a box wall, "
          f"{settled} of {len(good)} at an interior optimum")
    # THE CONTROL THAT DECIDES IT. delta_hat is only meaningful where the residual is
    # orthogonal to the span; a fit stopped against a bound carries distance-to-optimum in
    # its curve instead. So the statistic is recomputed on the records that pass the
    # package's own at_optimum test, and if the agreement lives on the ones that FAIL it,
    # the headline is an artifact of unconverged fits rather than a property of the model.
    clean = [d for d in good if got[(d, label)]["at_optimum"]]
    cg = [d for d in clean if d in records.DATASETS]
    cp = [d for d in clean if d not in records.DATASETS]
    if cg and cp:
      pairs = [abs(curves[a] @ curves[b]) for a in cg for b in cp]
      print(f"  {'':>10s}  interior-optimum records only ({len(cg)} gelatin, {len(cp)} PAAm): "
            f"ACROSS median {np.median(pairs):.3f} over {len(pairs)} pairs, "
            f"{sum(1 for r in pairs if r > cut)} above the same null")
      summary[label]["clean_across_median"] = float(np.median(pairs))
      summary[label]["clean_n"] = len(pairs)
    else:
      print(f"  {'':>10s}  interior-optimum records do not span both materials "
            f"({len(cg)} gelatin, {len(cp)} PAAm) -- no clean cross-material statistic")
  print("\n  qSLS isothermal, for comparison: across 0.726, "
        "within-gelatin 0.551, within-PAAm 0.679")

  json.dump({f"{d}|{label}": {k: v for k, v in got[(d, label)].items()
                              if k not in ("curve", "span")}
             for d in ORDER for label, *_ in CELLS} | {"agreement": summary},
            open(records.HERE / "vapour_kelvin_voigt.json", "w"), indent=1)
  json.dump({f"{d}|{label}": {k: got[(d, label)].get(k) for k in ("curve", "span")}
             for d in ORDER for label, *_ in CELLS if "failed" not in got[(d, label)]},
            open(records.HERE / "vapour_kelvin_voigt_curves.json", "w"))


if __name__ == "__main__":
  main()
