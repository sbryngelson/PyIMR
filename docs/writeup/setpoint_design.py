r"""Designing over what the experimenter SETS, not over what the bubble turns out to be.

Every design in this chapter names a target: \SI{416}{\micro\metre} at stretch $10.9$,
\SI{1015}{\micro\metre} at $6.7$, a batch of eight settings. Each is treated as a knob. The
acquisition data says it is not one --- per event, $R_{\max}$ has a coefficient of variation of
$0.25$, $0.28$ and $0.23$ across temperature bins, and the stretch a further $0.05$ to $0.07$.
A quarter is not a tolerance; it is the difference between \SI{400}{} and \SI{500}{\micro\metre}.

SO THE CANDIDATE IS A SETPOINT AND THE INFORMATION IS AN EXPECTATION. What the experimenter
chooses is a laser energy; what arrives is a draw. One run at setpoint $s$ therefore supplies
%
    M(s) = E_{x ~ p(. | s)} [ J(x)' J(x) ] ,

and the whole formulation survives it untouched: an expectation of information matrices is still
a matrix, $M(\xi) = \sum_i \xi_i M(s_i)$ is still linear in $\xi$, so $\log\det$ is still concave
and \cref{eq:equivalence} still certifies. Nothing is given up. What changes is WHERE the optimum
sits, and by how much a design that ignores the scatter is punished for it.

THE MECHANISM IS CURVATURE, and it has a sign. Where the information is a sharp ridge in
$R_{\max}$, averaging over a wide draw pulls the achievable value far below the nominal peak, and
the best setpoint moves toward a broad plateau even if its peak is lower. A certified optimum at
a spike is worth exactly what you can hit, which with a quarter of scatter is not much of it.

TWO QUANTITIES ARE WORTH HAVING. The REGRET of designing as though the setpoint were exact ---
what the nominal optimum actually delivers once its own scatter is applied, against what the
scatter-aware optimum delivers. And whether the recommendation MOVES, because a regret that is
real but leaves the answer where it was is a caveat, while one that moves it is a correction.

THE DISTRIBUTION IS MEASURED, NOT ASSUMED. Log-normal in $R_{\max}$ at the coefficient of
variation the acquisition file reports, and log-normal in the stretch likewise. Whether the
scatter is the same at a setpoint nobody has run is the standing caveat of \cref{sec:transfer},
and it is swept here rather than trusted: half the measured scatter, all of it, and twice.
"""

import json

import numpy as np

import records
from noise_design import profile, raw_information

SOURCE = "gelatin_15C"
# measured per event in raw_events.py; swept below rather than trusted at a geometry nobody ran
RADIUS_CV = 0.25
STRETCH_CV = 0.06
SWEEP = (0.5, 1.0, 2.0)
DRAWS = 64
SEED = 0


def gain_of(columns, phase, tau, spread):
  """`0.5 log det (I + s^2 J' Sigma^-1 J)` under the measured noise profile."""
  scale = np.interp(phase, tau, spread)
  whitened = columns / scale[:, None]
  fisher = whitened.T @ whitened
  prior = 1.0 / np.sqrt(12.0)
  # one factor of a half, not two. `whitened.T @ whitened` is symmetric by construction, so
  # `0.5 * (F + F.T)` is the identity on it and any second 0.5 is a straight halving of M.
  # The sibling scripts -- measured_scatter, control_value, bias_robust_design -- all fold the
  # symmetrisation and the prior scaling into one expression and are right; this one did not.
  return 0.5 * (fisher + fisher.T) * prior**2


def main():
  from pyimr.measure import optimal_measure
  from design_operator import PERFORMED, RADII, STRETCH

  designs = [(float(r), float(s)) for r in RADII for s in STRETCH] + list(PERFORMED.values())
  print(f"  building information at {len(designs)} geometries ...", flush=True)
  with records.pool(len(designs)) as pool:
    got = list(pool.map(raw_information, designs))
  usable = {d: payload for d, payload in got if payload is not None}
  print(f"  {len(usable)} of {len(designs)} integrate")

  tau, spread = profile(SOURCE)
  matrices = {}
  for design, (columns, phase) in usable.items():
    matrices[design] = gain_of(columns, phase, tau, spread)
  points = np.array(sorted(matrices))
  stack = np.array([matrices[tuple(p)] for p in points])

  # the achievable set: a setpoint's realised draws land at other grid points, so the
  # expectation is over the grid rather than over new solves
  radii = np.array(sorted({p[0] for p in points}))
  stretches = np.array(sorted({p[1] for p in points}))
  index = {tuple(p): i for i, p in enumerate(points)}
  rng = np.random.default_rng(SEED)

  def expected(setpoint, radius_cv, stretch_cv):
    """`E[M]` at a setpoint, over log-normal draws in radius and stretch, snapped to the grid."""
    target_r, target_s = setpoint
    drawn_r = target_r * np.exp(rng.normal(0.0, np.log1p(radius_cv), DRAWS))
    drawn_s = target_s * np.exp(rng.normal(0.0, np.log1p(stretch_cv), DRAWS))
    total, count = np.zeros_like(stack[0]), 0
    for r, s in zip(drawn_r, drawn_s, strict=True):
      near = (radii[np.argmin(np.abs(radii - r))], stretches[np.argmin(np.abs(stretches - s))])
      hit = index.get(near)
      if hit is None: continue
      total += stack[hit]
      count += 1
    return total / max(count, 1)

  summary = {}
  print(f"\n  {'scatter':>9s} {'formulation':>16s} {'best setpoint':>22s} {'nats':>8s} "
        f"{'gap':>9s} {'regret of ignoring it':>22s}")
  for factor in SWEEP:
    radius_cv, stretch_cv = RADIUS_CV * factor, STRETCH_CV * factor
    averaged = np.array([expected(tuple(p), radius_cv, stretch_cv) for p in points])
    posterior = np.eye(stack.shape[1])[None, :, :] + averaged
    realised = np.array([0.5 * float(np.linalg.slogdet(m)[1]) for m in posterior])
    nominal_posterior = np.eye(stack.shape[1])[None, :, :] + stack
    nominal = np.array([0.5 * float(np.linalg.slogdet(m)[1]) for m in nominal_posterior])

    best_nominal = int(np.argmax(nominal))
    best_aware = int(np.argmax(realised))
    regret = float(realised[best_aware] - realised[best_nominal])
    summary[f"x{factor:g}"] = {
      "radius_cv": radius_cv, "stretch_cv": stretch_cv,
      "nominal_setpoint": [float(points[best_nominal][0]), float(points[best_nominal][1])],
      "aware_setpoint": [float(points[best_aware][0]), float(points[best_aware][1])],
      "nominal_realised": float(realised[best_nominal]),
      "aware_realised": float(realised[best_aware]),
      "single_regret": regret, "moved": bool(best_nominal != best_aware)}
    print(f"  {'x' + format(factor, 'g'):>9s} {'nominal peak':>16s} "
          f"{points[best_nominal][0] * 1e6:8.0f} um at {points[best_nominal][1]:5.2f} "
          f"{realised[best_nominal]:8.3f} {'--':>9s} {'--':>22s}")
    print(f"  {'':>9s} {'scatter-aware':>16s} "
          f"{points[best_aware][0] * 1e6:8.0f} um at {points[best_aware][1]:5.2f} "
          f"{realised[best_aware]:8.3f} {'--':>9s} {regret:22.3f}")

  print("\n  and the same question for a whole certified batch, at the measured scatter\n")
  radius_cv, stretch_cv = RADIUS_CV, STRETCH_CV
  averaged = np.array([expected(tuple(p), radius_cv, stretch_cv) for p in points])
  print(f"  {'measure built on':>20s} {'settings':>9s} {'gap':>9s} {'nats delivered':>15s}")
  delivered = {}
  for label, built in (("nominal M(x)", stack), ("expected M(s)", averaged)):
    held = optimal_measure(built, iterations=200_000)
    # what it DELIVERS is always scored against the expectation, since that is what happens
    achieved = np.tensordot(held.weights, averaged, axes=(0, 0))
    value = 0.5 * float(np.linalg.slogdet(np.eye(stack.shape[1]) + achieved)[1])
    delivered[label] = {"settings": int(held.support.size), "gap": held.gap, "nats": value,
                        "support": [[float(points[i][0]), float(points[i][1]),
                                     float(held.weights[i])] for i in held.support]}
    print(f"  {label:>20s} {held.support.size:9d} {held.gap:9.1e} {value:15.3f}")
  summary["batch"] = delivered
  loss = delivered["expected M(s)"]["nats"] - delivered["nominal M(x)"]["nats"]
  print("\n  designing on the nominal information and running it into the real scatter costs")
  print(f"  {loss:.3f} nats against designing on the expectation -- and the certificate is")
  print("  identical either way, because an expectation of information matrices is still linear")
  print("  in the measure. Nothing is given up to get it.")
  summary["batch_regret"] = float(loss)

  print("\n  where the scatter-aware batch puts its runs\n")
  print(f"  {'R_max (um)':>11s} {'stretch':>8s} {'weight':>8s}")
  for radius, stretch, weight in sorted(delivered["expected M(s)"]["support"],
                                        key=lambda row: -row[2]):
    print(f"  {radius * 1e6:11.0f} {stretch:8.2f} {weight:8.3f}")

  json.dump(summary, open(records.HERE / "setpoint_design.json", "w"), indent=1)


if __name__ == "__main__":
  main()
