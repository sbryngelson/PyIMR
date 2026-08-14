r"""Does the solver reach where the setpoint quadrature needs it?

\Cref{sec:setpoint} evaluates $\mathbb{E}[M](s)$ over the measured scatter, so a design named at
\SI{50}{\micro\metre} draws realisations down to \SI{22}{\micro\metre} at four standard
deviations, and one named at \SI{1200}{\micro\metre} draws up to \SI{2685}{\micro\metre}. Every
solve this document holds stops at \SI{1200}{\micro\metre}. Nobody has checked whether the
integrator reaches the rest.

WHY A FAILED SOLVE IS WORSE HERE THAN A SLOW ONE. A geometry that does not integrate drops out of
the quadrature, the surviving weights no longer sum to one, and $\mathbb{E}[M]$ is biased LOW by
exactly the mass that went missing --- silently, and worst at the setpoints nearest the edges of
the box, which is where the certified support has repeatedly landed. Renormalising the survivors
does not repair it: that assumes the missing mass resembles the mass that stayed, which at a
collapse boundary is precisely what it does not do.

WHAT THIS MEASURES. Whether the solver reaches, across the whole realised range; and then, for
setpoints on the boundary of the design box, how much quadrature weight falls where it does not.
The second number is the one that decides whether the Monte-Carlo setpoint study is worth
building, because it is the bias that study would carry.

THE OUT-OF-BOX GEOMETRIES ARE NOT A DOMAIN ERROR, and this is the opposite of the bug that the
exact-geometry refinement had. There, points outside \SIrange{50}{1200}{\micro\metre} were being
offered as \emph{designs}, which is wrong because they cannot be requested. Here they are
\emph{realisations} of designs that can: you ask for \SI{50}{\micro\metre} and the laser
sometimes gives you \SI{22}{\micro\metre}. Clipping them would be the error.
"""

import json

import numpy as np

import records
from noise_design import profile, raw_information
from offgrid_and_compound import SOURCE, pieces

RADIUS_CV, STRETCH_CV = 0.223, 0.067      # measured on 437 events
BOX_R = (50e-6, 1200e-6)
BOX_S = (3.0, 20.0)
COVER = 4.0                                # standard deviations the quadrature must reach
PROBE_R, PROBE_S = 20, 12                  # a coarse map of the realised range


def main():
  sr, ss = np.log1p(RADIUS_CV), np.log1p(STRETCH_CV)
  lo_r, hi_r = BOX_R[0] * np.exp(-COVER * sr), BOX_R[1] * np.exp(COVER * sr)
  lo_s, hi_s = BOX_S[0] * np.exp(-COVER * ss), BOX_S[1] * np.exp(COVER * ss)
  print(f"  design box     {BOX_R[0] * 1e6:7.1f} - {BOX_R[1] * 1e6:7.1f} um, "
        f"stretch {BOX_S[0]:5.2f} - {BOX_S[1]:5.2f}")
  print(f"  realised range {lo_r * 1e6:7.1f} - {hi_r * 1e6:7.1f} um, "
        f"stretch {lo_s:5.2f} - {hi_s:5.2f}   (+-{COVER:.0f} sigma)")

  radii = np.geomspace(lo_r, hi_r, PROBE_R)
  stretches = np.geomspace(lo_s, hi_s, PROBE_S)
  jobs = [(float(r), float(s)) for r in radii for s in stretches]
  print(f"\n  probing {len(jobs)} geometries across the realised range ...", flush=True)
  with records.pool(len(jobs)) as pool:
    got = dict(pool.map(raw_information, jobs))

  tau, spread = profile(SOURCE)
  ok = {}
  for design, payload in got.items():
    if payload is None:
      ok[design] = False
      continue
    try:
      matrix = pieces(payload, tau, spread)[0]
      ok[design] = bool(np.all(np.isfinite(matrix)))
    except Exception:                                                        # noqa: BLE001
      ok[design] = False

  print(f"\n  reach map: '.' integrates, 'X' does not "
        f"({sum(ok.values())} of {len(ok)} succeed)\n")
  print(f"  {'stretch':>8s}  " + " ".join(f"{r * 1e6:5.0f}" for r in radii))
  for s in stretches:
    row = " ".join(f"{'    .' if ok[(float(r), float(s))] else '    X'}" for r in radii)
    print(f"  {s:8.2f}  {row}")
  print(f"\n  {'':>8s}  " + " ".join("  in " if BOX_R[0] <= r <= BOX_R[1] else " out "
                                     for r in radii) + "   <- inside the design box?")

  # the decision number: for setpoints at the edges, how much weight lands where it fails
  print("\n  quadrature mass landing on a geometry that does not integrate\n")
  print(f"  {'setpoint':>22s} {'lost mass':>10s} {'bias in E[M]':>13s}")
  interp_r, interp_s = np.log(radii), np.log(stretches)
  corners = [("box corner, small+low", BOX_R[0], BOX_S[0]),
             ("box corner, small+high", BOX_R[0], BOX_S[1]),
             ("box corner, large+low", BOX_R[1], BOX_S[0]),
             ("box corner, large+high", BOX_R[1], BOX_S[1]),
             ("box centre", float(np.sqrt(BOX_R[0] * BOX_R[1])),
              float(np.sqrt(BOX_S[0] * BOX_S[1])))]
  summary = {"succeeded": int(sum(ok.values())), "probed": len(ok), "corners": {}}
  for label, r0, s0 in corners:
    wr = np.exp(-0.5 * ((interp_r - np.log(r0)) / sr) ** 2)
    ws = np.exp(-0.5 * ((interp_s - np.log(s0)) / ss) ** 2)
    weight = np.outer(wr, ws)
    weight /= weight.sum()
    lost = sum(weight[i, j] for i, r in enumerate(radii) for j, s in enumerate(stretches)
               if not ok[(float(r), float(s))])
    summary["corners"][label] = float(lost)
    print(f"  {label:>22s} {lost:10.2%} {'low by that much' if lost > 1e-3 else 'negligible':>13s}")

  # the usable setpoint domain: where essentially all of a setpoint's realisations integrate
  print("\n  the usable setpoint domain: lost quadrature mass by setpoint\n")
  set_r = np.geomspace(*BOX_R, 12)
  set_s = np.geomspace(*BOX_S, 8)
  print(f"  {'stretch':>8s}  " + " ".join(f"{r * 1e6:5.0f}" for r in set_r))
  frontier = {}
  for s0 in set_s:
    cells, usable_to = [], None
    for r0 in set_r:
      wr = np.exp(-0.5 * ((interp_r - np.log(r0)) / sr) ** 2)
      ws = np.exp(-0.5 * ((interp_s - np.log(s0)) / ss) ** 2)
      w = np.outer(wr, ws); w /= w.sum()
      lost = sum(w[i, j] for i, r in enumerate(radii) for j, st in enumerate(stretches)
                 if not ok[(float(r), float(st))])
      cells.append(f"{lost:5.1%}" if lost > 5e-4 else "    .")
      if lost <= 0.01: usable_to = r0
    frontier[f"{s0:.2f}"] = None if usable_to is None else float(usable_to)
    print(f"  {s0:8.2f}  " + " ".join(cells))
  summary["frontier"] = frontier
  print("  '.' is below 0.05% lost; the usable region is where the mass loss is negligible.")

  print("\n  largest setpoint radius keeping the loss under 1%, by stretch\n")
  print(f"  {'stretch':>8s} {'max setpoint radius':>21s}")
  for k, v in frontier.items():
    print(f"  {k:>8s} {'none' if v is None else f'{v * 1e6:.0f} um':>21s}")

  worst = max(summary["corners"].values())
  print()
  if worst < 1e-3:
    print("  The solver reaches. E[M] can be built by quadrature or Monte Carlo over the whole")
    print("  realised range without a truncation bias worth reporting.")
  else:
    print(f"  Up to {worst:.1%} of the quadrature mass falls where the solver does not reach.")
    print("  E[M] would be biased low by that much at the worst setpoint, and renormalising")
    print("  the survivors would hide rather than fix it. The setpoint study needs either a")
    print("  restricted setpoint range or a solver that reaches, before it is worth running.")
  summary["worst_lost"] = float(worst)
  json.dump(summary, open(records.HERE / "setpoint_reach.json", "w"), indent=1)


if __name__ == "__main__":
  main()
