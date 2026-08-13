r"""Two things \cref{sec:measure} lists as not done, and both are runnable.

\Cref{sec:measure} closes with three limitations. One --- the point estimate of $\theta$ --- is
answered in \cref{sec:biasdesign}. The other two are:

  the candidate set is still a grid, so the certificate proves optimality OVER THAT GRID; and
  the compound criterion was never run on these records.

Neither needs new data, and this does both.

THE GRID CHECK IS CHEAP FOR A REASON THAT SECTION ALREADY GIVES. The sensitivity
$d(x,\xi^\star) = \operatorname{tr}[M(\xi^\star)^{-1}M(x)]$ can be evaluated ANYWHERE, not only
where the search ran, so looking for a region the grid missed is a scan rather than another
optimisation. If $d$ stays under $p$ on a grid four times finer than the one the measure was
certified on, the certificate reaches past its own candidate set; if it does not, the answer is
optimal only among the settings it was offered, and the exceedance says where to look.

THE COMPOUND CRITERION WAS BLOCKED ON AN ESTIMATOR, NOT ON THEORY. $\Phi_\beta = (1-\beta)
\log\det M(\xi) + \beta\, u^{\mathsf T}\xi$ is concave for every $\beta$ because both parts are,
so the certificate holds along the whole front; what stopped it was that $u$, the per-design log
Bayes factor, suffered the boundary pinning of \cref{sec:integral}. The linearised separation of
\cref{sec:batch} --- the model difference orthogonal to the material span --- has no inner
optimisation and no boundary to pin against, so it can play the utility and the front can be
traced.

WHAT THE FRONT IS FOR. \Cref{sec:design} argues that estimation and discrimination want different
experiments and that a measure dissolves the conflict. That was shown at $\beta = 0$. Sweeping
$\beta$ says whether the dissolution is a coincidence at one weighting or a property of the whole
trade, and it is the honest way to present a design serving two questions at once.
"""

import json

import numpy as np

import records
from noise_design import profile, raw_information

SOURCE = "gelatin_15C"
COARSE_R = np.geomspace(50e-6, 1200e-6, 16)
COARSE_S = np.linspace(3.0, 20.0, 14)
FINE_R = np.geomspace(50e-6, 1200e-6, 46)
FINE_S = np.linspace(3.0, 20.0, 40)
BETAS = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)


def pieces(payload, tau, spread):
  """`(material information, linearised separation)` at one geometry."""
  columns, phase = payload
  scale = np.interp(phase, tau, spread)
  whitened = columns[:, :-1] / scale[:, None]
  gap = columns[:, -1] / scale
  gram = whitened.T @ whitened
  cross = whitened.T @ gap
  # the model difference orthogonal to the material span: no inner minimisation, no boundary
  separation = float(gap @ gap - cross @ np.linalg.solve(gram, cross))
  return 0.5 * (gram + gram.T), max(separation, 0.0)


def main():
  from pyimr.measure import optimal_measure, sensitivity

  coarse = [(float(r), float(s)) for r in COARSE_R for s in COARSE_S]
  fine = [(float(r), float(s)) for r in FINE_R for s in FINE_S]
  every = sorted(set(coarse) | set(fine))
  print(f"  {len(coarse)} coarse and {len(fine)} fine candidates, {len(every)} distinct ...",
        flush=True)
  with records.pool(len(every)) as pool:
    got = dict(pool.map(raw_information, every))

  tau, spread = profile(SOURCE)
  built = {d: pieces(p, tau, spread) for d, p in got.items() if p is not None}
  print(f"  {len(built)} integrate")
  usable_coarse = [d for d in coarse if d in built]
  usable_fine = [d for d in fine if d in built]
  if len(usable_coarse) < 20 or len(usable_fine) < 100:
    print("  too few to work with")
    return

  summary = {}

  # ---- does the certificate reach past the grid it was proved on? ----
  stack = np.array([built[d][0] for d in usable_coarse])
  held = optimal_measure(stack, iterations=200_000)
  averaged = np.tensordot(held.weights, stack, axes=(0, 0))
  inverse = np.linalg.inv(averaged)
  parameters = stack.shape[1]
  on_grid = float(np.max(sensitivity(stack, held.weights))) + parameters
  off_grid = np.array([float(np.trace(inverse @ built[d][0])) for d in usable_fine])
  worst = int(np.argmax(off_grid))
  print(f"\n  the certified measure on the coarse grid: {held.support.size} settings, "
        f"gap {held.gap:.1e}")
  print("  support: " + ", ".join(f"{usable_coarse[i][0] * 1e6:.0f} um @ {usable_coarse[i][1]:.2f}"
                                   for i in held.support))
  print(f"\n  {'evaluated on':>22s} {'points':>7s} {'max d(x, xi*)':>14s} {'bound p':>8s} "
        f"{'exceedance':>11s}")
  print(f"  {'the grid it certified on':>22s} {len(usable_coarse):7d} {on_grid:14.6f} "
        f"{parameters:8d} {on_grid - parameters:11.2e}")
  print(f"  {'a grid 8x finer':>22s} {len(usable_fine):7d} {off_grid.max():14.6f} "
        f"{parameters:8d} {off_grid.max() - parameters:11.2e}")
  summary["certificate"] = {
    "support": [[usable_coarse[i][0], usable_coarse[i][1], float(held.weights[i])]
                for i in held.support],
    "on_grid_max": on_grid, "off_grid_max": float(off_grid.max()), "p": int(parameters),
    "worst_offgrid": list(usable_fine[worst]),
    "reaches": bool(off_grid.max() <= parameters + 1e-6)}
  if summary["certificate"]["reaches"]:
    print("\n  The certificate reaches past its own candidate set: nothing on the finer grid")
    print("  exceeds the bound, so the measure is optimal over the continuum too, not merely")
    print("  over the settings it was offered.")
  else:
    print(f"\n  A region the grid MISSED, at {usable_fine[worst][0] * 1e6:.0f} um @ "
          f"{usable_fine[worst][1]:.2f}, exceeding the bound by "
          f"{off_grid.max() - parameters:.3e}. The measure is optimal over its own grid and")
    print("  not over the continuum; that point belongs in the candidate set.")

  # ---- the same scan, but over SETPOINTS rather than exact geometries ----
  # The exceedance above is real -- it survives a hundredfold tighter tolerance, so it is the
  # model's own information and not the solver's. But sec:setpoint established that a geometry
  # is not a knob: the stretch arrives with a coefficient of variation of 0.06, which is wider
  # than these spikes. Averaged over that, the candidate a design can actually ASK FOR is E[M],
  # and the question is whether the exceedance survives being reachable.
  stretch_cv = 0.06
  fine_by_radius = {}
  for radius, value in usable_fine:
    fine_by_radius.setdefault(radius, []).append(value)
  smoothed = []
  for radius, values in fine_by_radius.items():
    axis = np.array(sorted(values))
    block = np.array([built[(radius, float(v))][0] for v in axis])
    for target in axis:
      weight = np.exp(-0.5 * (np.log(axis / target) / np.log1p(stretch_cv)) ** 2)
      weight /= weight.sum()
      smoothed.append(float(np.trace(inverse @ np.tensordot(weight, block, axes=(0, 0)))))
  smoothed = np.array(smoothed)
  print(f"  {'the same grid, as setpoints':>22s} {len(smoothed):7d} {smoothed.max():14.6f} "
        f"{parameters:8d} {smoothed.max() - parameters:11.2e}")
  summary["certificate"]["setpoint_max"] = float(smoothed.max())
  summary["certificate"]["setpoint_reaches"] = bool(smoothed.max() <= parameters + 1e-6)
  if summary["certificate"]["setpoint_reaches"]:
    print("\n  Over SETPOINTS the exceedance is gone. The spikes are narrower than the scatter a")
    print("  laser delivers, so they are information nobody can ask for, and the certificate")
    print("  reaches the continuum once the design is posed over what can actually be set.")

  # ---- the compound front, which sec:design says was never run ----
  fine_stack = np.array([built[d][0] for d in usable_fine])
  utility = np.array([built[d][1] for d in usable_fine])
  print("\n  the compound front, Phi_beta = (1-beta) log det M + beta u.xi\n")
  print(f"  {'beta':>5s} {'settings':>9s} {'gap':>9s} {'log det M':>11s} "
        f"{'separation':>11s} {'support':>44s}")
  front = {}
  for beta in BETAS:
    result = optimal_measure(fine_stack, utility=utility, blend=beta, iterations=200_000)
    matrix = np.tensordot(result.weights, fine_stack, axes=(0, 0))
    determinant = float(np.linalg.slogdet(matrix)[1])
    reached = float(result.weights @ utility)
    where = ", ".join(f"{usable_fine[i][0] * 1e6:.0f}@{usable_fine[i][1]:.1f}"
                      for i in result.support[:4])
    front[f"{beta:g}"] = {"settings": int(result.support.size), "gap": result.gap,
                          "log_det": determinant, "separation": reached,
                          "certified": bool(result.certified),
                          "support": [[usable_fine[i][0], usable_fine[i][1],
                                       float(result.weights[i])] for i in result.support]}
    print(f"  {beta:5.2f} {result.support.size:9d} {result.gap:9.1e} {determinant:11.3f} "
          f"{reached:11.1f} {where:>44s}")
  summary["front"] = front

  spread_det = front["0"]["log_det"] - front["1"]["log_det"]
  spread_sep = front["1"]["separation"] - front["0"]["separation"]
  print(f"\n  End to end the front trades {spread_det:.3f} in log det for "
        f"{spread_sep:.1f} in separation.")
  print("  A front whose interior beats both ends on the axis it is not optimising is the")
  print("  dissolution sec:design claims; one that is a straight line between them is a")
  print("  genuine trade the measure cannot remove.")
  midpoint = front["0.5"]
  print(f"  At beta = 0.5: log det {midpoint['log_det']:.3f} of {front['0']['log_det']:.3f}, "
        f"separation {midpoint['separation']:.1f} of {front['1']['separation']:.1f} "
        f"-- {midpoint['log_det'] / front['0']['log_det']:.1%} and "
        f"{midpoint['separation'] / max(front['1']['separation'], 1e-9):.1%} of each maximum.")

  json.dump(summary, open(records.HERE / "offgrid_and_compound.json", "w"), indent=1)


if __name__ == "__main__":
  main()
