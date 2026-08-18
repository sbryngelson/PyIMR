r"""Is ``correcting the fits destroys the cross-material agreement'' a result or a search failure?

`delta_corrected.py` with the thermal physics on refits all eight records and reports the
cross-material median $\lvert\cos\rvert$ falling from $0.726$ to $0.302$, which if true retracts
\cref{sec:universal}'s central claim. Before that is written down it has to survive the objection
the result itself raises: the corrected fits come back WORSE than the isothermal ones, at
$\chi^2/N$ up to $3.7$ against $1.2$ to $1.35$, and a fit that gets worse when the physics gets
more complete is the signature of a search that did not converge rather than of a model that
does not hold.

Two things would produce exactly that. The box is centred on the isothermal optimum and spans a
factor of three per axis, and the thermal objective at that centre is an order of magnitude
worse than the isothermal one, so the optimum may simply lie OUTSIDE the box -- in which case
the fit stops at a wall and $\hat\delta$ is contaminated by the misfit the search could not
remove. And $3\times100$ evaluations over four axes is a short search for a surface that moved
that far.

So the same fit is run at two box widths and the parameters are reported rather than discarded.
If the wide box finds materially better $\chi^2/N$, or if the narrow box's answers sit on their
walls, the retraction is premature and this script says so. If both widths agree, the
cross-material collapse is a property of the corrected model and not of the optimiser.
"""

import json

import numpy as np

import records
from shape_error import seed_for
from universal_delta import GRID, _surrogate

THERMAL = {"bubtherm": 1, "Nt": 25, "masstrans": 1, "vapor": 1}
ORDER = (*records.DATASETS, *records.PAAM)
WIDTHS = (3.0, 10.0)
STARTS, EVALUATIONS = 3, 100
WALL = 0.02   # within 2% of a bound, in log-space fraction of the box, counts as pinned
DRAWS = 2000


def one(job):
  from pyimr.noise import discrepancy
  from pyimr.selection import STANDARD_MODELS, evaluate_at, fit_candidate, physical_from_unit

  dataset, width = job
  times, mean, spread, maximum, _ = records.load(dataset)
  stretch = json.load(open(records.HERE / "paam_stretch.json"))[dataset]["argmin"]
  warm = json.load(open(records.HERE / "paam_lackoffit.json"))[dataset]["fitted"]
  candidate = STANDARD_MODELS["qSLS"]
  box = {a: (warm[a] / width, warm[a] * width) for a in candidate.axes}
  solve = records.solver(times, maximum, stretch, max_steps=400_000, **THERMAL)
  try:
    fit = fit_candidate(candidate, solve, mean, spread, bounds=box, starts=STARTS,
                        max_evaluations=EVALUATIONS)
    values = physical_from_unit(candidate.axes, fit.unit, box)
    fitted = dict(zip(candidate.axes, (float(v) for v in values), strict=True))
    base = np.asarray(evaluate_at(candidate, solve, fitted)[0], dtype=float)
  except Exception as failure:                                               # noqa: BLE001
    return (dataset, width), {"failed": f"{type(failure).__name__}: {failure}"[:110]}

  # where in its own box each axis landed, 0 at the low wall and 1 at the high one
  where = {a: float(np.log(fitted[a] / box[a][0]) / np.log(box[a][1] / box[a][0]))
           for a in candidate.axes}
  pinned = [a for a, u in where.items() if u < WALL or u > 1.0 - WALL]

  error_bar = spread / np.sqrt(records.trial_count(dataset))
  step, columns = 1e-4, []
  for axis in candidate.axes:
    up = dict(fitted, **{axis: fitted[axis] * (1 + step)})
    dn = dict(fitted, **{axis: fitted[axis] * (1 - step)})
    try:
      hi = np.asarray(evaluate_at(candidate, solve, up)[0], dtype=float)
      lo = np.asarray(evaluate_at(candidate, solve, dn)[0], dtype=float)
    except Exception:                                                        # noqa: BLE001
      return (dataset, width), {"failed": "jacobian"}
    columns.append((hi - lo) / (2 * step * error_bar))
  jacobian = np.column_stack(columns)
  curve = np.asarray(discrepancy((mean - base) / error_bar, jacobian).identifiable, float)
  tau = times / (maximum * np.sqrt(1064.0 / 101325.0))

  def onto(values):
    got = np.interp(GRID, tau[: values.size], values)
    got = got - got.mean()
    return got / (np.linalg.norm(got) + 1e-30)

  return (dataset, width), {
    "chi2_per_n": float(fit.chi_squared), "fitted": fitted, "where": where,
    "pinned": pinned, "curve": onto(curve).tolist(),
    "span": [onto(column).tolist() for column in jacobian.T]}


def main():
  jobs = [(d, w) for d in ORDER for w in WIDTHS]
  with records.pool(len(jobs)) as pool:
    got = dict(pool.map(one, jobs))

  print(f"  qSLS refitted with vapour on, box centred on the isothermal optimum\n")
  print(f"  {'record':>16s}  " + "  ".join(
    f"{'x' + str(int(w)) + ' chi2/N':>11s} {'pinned axes':>22s}" for w in WIDTHS))
  out = {}
  for dataset in ORDER:
    cells = []
    for width in WIDTHS:
      value = got[(dataset, width)]
      out[f"{dataset}|{width}"] = {k: v for k, v in value.items() if k not in ("curve", "span")}
      cells.append(f"{'FAILED':>11s} {'':>22s}" if "failed" in value else
                   f"{value['chi2_per_n']:11.4f} "
                   f"{(','.join(value['pinned']) if value['pinned'] else '-'):>22s}")
    print(f"  {dataset:>16s}  " + "  ".join(cells))

  print("\n  ---- did the wider box find a better optimum? ----\n")
  better = []
  for dataset in ORDER:
    narrow, wide = got[(dataset, WIDTHS[0])], got[(dataset, WIDTHS[1])]
    if "failed" in narrow or "failed" in wide: continue
    gain = narrow["chi2_per_n"] / wide["chi2_per_n"] - 1.0
    better.append(gain)
    print(f"  {dataset:>16s}  narrow {narrow['chi2_per_n']:8.4f}  wide {wide['chi2_per_n']:8.4f}"
          f"  {gain:+7.1%}")
  print(f"\n  median improvement from widening the box: {np.median(better):+.1%}")
  any_pinned = sum(1 for d in ORDER for w in WIDTHS
                   if "failed" not in got[(d, w)] and got[(d, w)]["pinned"])
  print(f"  fits with at least one axis on a wall: {any_pinned} of {len(jobs)}")

  print("\n  ---- and does the cross-material verdict depend on the box? ----\n")
  rng = np.random.default_rng(seed_for("vapour_convergence"))
  summary = {}
  for width in WIDTHS:
    good = [d for d in ORDER if "failed" not in got[(d, width)]]
    curves = {d: np.asarray(got[(d, width)]["curve"], dtype=float) for d in good}
    spans = {d: np.asarray(got[(d, width)]["span"], dtype=float) for d in good}

    def constrained(d):
      out_ = _surrogate(rng, curves[d])
      basis, _ = np.linalg.qr(spans[d].T)
      out_ = out_ - basis @ (basis.T @ out_)
      return out_ / (np.linalg.norm(out_) + 1e-30)

    null = np.array([abs(constrained(good[a]) @ constrained(good[b]))
                     for a, b in rng.choice(len(good), size=(DRAWS, 2)) if a != b])
    cut = float(np.percentile(null, 95))
    across = [abs(curves[a] @ curves[b]) for a in good for b in good
              if (a in records.DATASETS) != (b in records.DATASETS)]
    across = across[: len(across) // 2] if False else [
      abs(curves[a] @ curves[b]) for i, a in enumerate(good) for b in good[i + 1:]
      if (a in records.DATASETS) != (b in records.DATASETS)]
    within = [abs(curves[a] @ curves[b]) for i, a in enumerate(good) for b in good[i + 1:]
              if (a in records.DATASETS) == (b in records.DATASETS)]
    summary[width] = {"across_median": float(np.median(across)),
                      "across_above": sum(1 for r in across if r > cut), "across_n": len(across),
                      "within_median": float(np.median(within)), "null_95": cut}
    print(f"  box x{int(width):<3d}  ACROSS median {np.median(across):.3f} "
          f"({sum(1 for r in across if r > cut)} of {len(across)} above null {cut:.3f}), "
          f"within {np.median(within):.3f}")

  print("\n  ---- what it says ----\n")
  widened = np.median(better)
  if any_pinned or widened > 0.10:
    print("  The narrow-box fits were search-limited, so the cross-material collapse reported")
    print("  by delta_corrected.py is not yet a result about the model. Read the wide-box row.")
  else:
    print("  Widening the box does not find a better optimum and nothing sits on a wall, so")
    print("  the corrected fits are converged and the cross-material comparison below is a")
    print("  statement about the corrected model rather than about the optimiser.")
  json.dump({"fits": out, "universality": {str(k): v for k, v in summary.items()},
             "median_widening_gain": float(widened), "pinned_fits": any_pinned},
            open(records.HERE / "vapour_convergence.json", "w"), indent=1)


if __name__ == "__main__":
  main()
