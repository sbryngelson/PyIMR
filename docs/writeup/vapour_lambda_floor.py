r"""Where does the corrected fit's relaxation time actually want to be?

`vapour_convergence.py` found the vapour-corrected fits pinned on their boxes, and reading the
pinning in the RAW axes was misleading. In the identified coordinates of \cref{sec:identify} the
picture is clean: $g\alpha$, the combination the data determines, converges and agrees to about
\SI{3}{\percent} between a box of $\times3$ and one of $\times10$. What does not converge is
$\lambda_1$, and it does not merely sit on a wall -- it sits on the wall EXACTLY, at the
isothermal value divided by the box width, on seven of eight records at both widths, while
$\chi^2$ keeps falling as the wall moves down.

That is not the flat direction \cref{sec:identify} describes, where the objective is
indifferent and the answer is whatever the prior box says. It is an active descent toward
$\lambda_1 \to 0$: with the thermal and vapour physics switched on, the data stops wanting a
relaxing solid. The obvious reading is that the Zener arm in the isothermal fits was absorbing
thermal damping, and once that damping is modelled the arm is no longer needed.

So $\lambda_1$ is given the full prior range and the others are held near their converged
values. Three outcomes are separable. If $\chi^2$ plateaus at a finite $\lambda_1$, there is an
interior optimum and $\hat\delta$ can be measured at it. If it plateaus only as
$\lambda_1 \to 0$, the corrected model has left the qSLS family through a boundary and any
$\hat\delta$ from it belongs to the degenerate limit, which has to be said. And if $g\alpha$
moves once $\lambda_1$ is free, the earlier convergence was conditional on the wall.
"""

import json

import numpy as np

import records
from shape_error import seed_for
from universal_delta import GRID, _surrogate

THERMAL = {"bubtherm": 1, "Nt": 25, "masstrans": 1, "vapor": 1}
ORDER = (*records.DATASETS, *records.PAAM)
OTHERS = 10.0          # mu, g, alpha keep the width that already converged in g*alpha
LAMBDA = (1e-9, 1e-2)  # the full prior range from lackoffit.WIDE
STARTS, EVALUATIONS = 4, 150
WALL, DRAWS = 0.02, 2000


def one(dataset):
  from pyimr.noise import discrepancy
  from pyimr.selection import STANDARD_MODELS, evaluate_at, fit_candidate, physical_from_unit

  times, mean, spread, maximum, _ = records.load(dataset)
  stretch = json.load(open(records.HERE / "paam_stretch.json"))[dataset]["argmin"]
  warm = json.load(open(records.HERE / "paam_lackoffit.json"))[dataset]["fitted"]
  candidate = STANDARD_MODELS["qSLS"]
  box = {a: (warm[a] / OTHERS, warm[a] * OTHERS) for a in candidate.axes}
  box["lambda1"] = LAMBDA
  solve = records.solver(times, maximum, stretch, max_steps=400_000, **THERMAL)
  try:
    fit = fit_candidate(candidate, solve, mean, spread, bounds=box, starts=STARTS,
                        max_evaluations=EVALUATIONS)
    values = physical_from_unit(candidate.axes, fit.unit, box)
    fitted = dict(zip(candidate.axes, (float(v) for v in values), strict=True))
    base = np.asarray(evaluate_at(candidate, solve, fitted)[0], dtype=float)
  except Exception as failure:                                               # noqa: BLE001
    return dataset, {"failed": f"{type(failure).__name__}: {failure}"[:110]}

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
      return dataset, {"failed": "jacobian"}
    columns.append((hi - lo) / (2 * step * error_bar))
  jacobian = np.column_stack(columns)
  curve = np.asarray(discrepancy((mean - base) / error_bar, jacobian).identifiable, float)
  tau = times / (maximum * np.sqrt(1064.0 / 101325.0))

  def onto(values):
    got = np.interp(GRID, tau[: values.size], values)
    got = got - got.mean()
    return got / (np.linalg.norm(got) + 1e-30)

  return dataset, {"chi2_per_n": float(fit.chi_squared), "fitted": fitted, "where": where,
                   "g_alpha": float(fitted["g"] * fitted["alpha"]),
                   "curve": onto(curve).tolist(),
                   "span": [onto(column).tolist() for column in jacobian.T]}


def main():
  with records.pool(len(ORDER)) as pool:
    got = dict(pool.map(one, list(ORDER)))
  good = [d for d in ORDER if "failed" not in got[d]]
  for d in ORDER:
    if "failed" in got[d]: print(f"    {d}: {got[d]['failed']}")

  narrow = json.load(open(records.HERE / "vapour_convergence.json"))["fits"]
  iso = json.load(open(records.HERE / "paam_lackoffit.json"))
  print(f"\n  lambda1 given the full prior range {LAMBDA[0]:.0e} to {LAMBDA[1]:.0e}\n")
  print(f"  {'record':>16s} {'chi2/N':>9s} {'was (x10)':>10s} {'lambda1':>11s} "
        f"{'iso lambda1':>12s} {'at wall?':>9s} {'g*alpha':>10s} {'was':>10s}")
  floored = 0
  for d in good:
    v, prev = got[d], narrow.get(f"{d}|10.0", {})
    at_wall = v["where"]["lambda1"] < WALL
    floored += at_wall
    print(f"  {d:>16s} {v['chi2_per_n']:9.4f} {prev.get('chi2_per_n', float('nan')):10.4f}"
          f" {v['fitted']['lambda1']:11.3g} {iso[d]['fitted']['lambda1']:12.3g}"
          f" {('YES' if at_wall else 'no'):>9s} {v['g_alpha']:10.1f}"
          f" {prev.get('fitted', {}).get('g', float('nan')) * prev.get('fitted', {}).get('alpha', float('nan')):10.1f}")

  print(f"\n  fits still on the lambda1 floor with seven decades of room: {floored} of {len(good)}")
  gains = [narrow[f'{d}|10.0']['chi2_per_n'] / got[d]['chi2_per_n'] - 1.0
           for d in good if f"{d}|10.0" in narrow]
  print(f"  median further improvement from freeing lambda1: {np.median(gains):+.1%}")

  rng = np.random.default_rng(seed_for("vapour_lambda_floor"))
  curves = {d: np.asarray(got[d]["curve"], dtype=float) for d in good}
  spans = {d: np.asarray(got[d]["span"], dtype=float) for d in good}

  def constrained(d):
    out_ = _surrogate(rng, curves[d])
    basis, _ = np.linalg.qr(spans[d].T)
    out_ = out_ - basis @ (basis.T @ out_)
    return out_ / (np.linalg.norm(out_) + 1e-30)

  null = np.array([abs(constrained(good[a]) @ constrained(good[b]))
                   for a, b in rng.choice(len(good), size=(DRAWS, 2)) if a != b])
  cut = float(np.percentile(null, 95))
  across = [abs(curves[a] @ curves[b]) for i, a in enumerate(good) for b in good[i + 1:]
            if (a in records.DATASETS) != (b in records.DATASETS)]
  within = [abs(curves[a] @ curves[b]) for i, a in enumerate(good) for b in good[i + 1:]
            if (a in records.DATASETS) == (b in records.DATASETS)]
  print(f"\n  ACROSS materials: median {np.median(across):.3f} "
        f"({sum(1 for r in across if r > cut)} of {len(across)} above null {cut:.3f})")
  print(f"  within materials: median {np.median(within):.3f}")
  print(f"  for comparison, uncorrected: across 0.726, within-gelatin 0.551, within-PAAm 0.679")

  print("\n  ---- what it says ----\n")
  if floored > len(good) / 2:
    print("  The corrected model wants no relaxation at all: lambda1 runs to the floor with")
    print("  seven decades of room. The fit has left the qSLS family through a boundary, so")
    print("  any delta_hat from it is the degenerate limit's and the cross-material number")
    print("  below is not comparable to the interior fit sec:universal reports.")
  else:
    print("  lambda1 finds an interior optimum once it is given room, so these fits are")
    print("  converged and the cross-material comparison above is a statement about the")
    print("  corrected model.")
  json.dump({d: {k: v for k, v in got[d].items() if k not in ("curve", "span")} for d in good}
            | {"across": across, "within": within, "null_95": cut, "on_floor": floored},
            open(records.HERE / "vapour_lambda_floor.json", "w"), indent=1)


if __name__ == "__main__":
  main()
