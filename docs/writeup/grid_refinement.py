r"""Refining where the scan flagged, and asking whether the certificate can be restored.

\Cref{sec:measure} certifies a measure globally optimal in $\xi$, and \cref{sec:offgrid} showed
the certificate does not reach past its own candidate set: on a grid eight times finer than the
one it was proved on, $\max_x d(x,\xi^\star) = 69.56$ against a bound of $p = 4$. That study
closed by saying where to refine and predicting that refining would move the answer. This does
it.

THE PROCEDURE IS THE VERTEX-DIRECTION STEP, AND IT IS NOT NEW. Adding $\arg\max_x d(x,\xi)$ to
the support is the Fedorov--Wynn move; iterating it converges to the continuous optimum for a
concave criterion. What is being asked here is not whether the algorithm works in principle but
whether the continuous optimum on THESE trajectories is something a finite candidate set can
represent, given that the exceedances are ridges of width $\sim 0.1$ in stretch.

THE CIRCULARITY THIS HAS TO AVOID. Refining a candidate set until $d \le p$ ON THAT SAME SET
proves nothing --- `optimal_measure` guarantees it by construction, so the test would be passing
its own answer back to itself. Every round here therefore certifies on the candidate set and then
scans an INDEPENDENT one: the midpoints of the current grid, which no candidate occupies, plus a
zoom around the current support and around whatever the last round flagged. Fresh solves
throughout; nothing is interpolated.

WHAT EACH OUTCOME MEANS AND BOTH ARE PUBLISHABLE. If $\max_x d$ falls to $p$ on independent scans
as points are added, the certificate is restored and \cref{sec:measure} may say `optimal' without
the qualifier --- at the cost of a candidate set with the refined points in it. If it stalls
above $p$ round after round while the flagged points keep moving, then the continuous optimum
concentrates on ridges no finite set represents, and `optimal over the candidate set' is not a
temporary weakness of this chapter but the strongest true statement available.

ONLY THE EXACT-GEOMETRY ARM IS RUN HERE, DELIBERATELY. \Cref{sec:offgrid} already reports that
averaging over the measured setpoint scatter cuts the exceedance from $65.6$ to $8.4$. Repeating
this loop in the setpoint formulation would require $\mathbb{E}[M]$ at every scan point from
fresh quadrature solves; done by interpolating one shared cube instead, the finer scan would
merely resample the same piecewise-linear surface and $d \le p$ would follow from convexity
rather than from the model. That is the same circularity as above wearing a different hat, so it
is left undone rather than done wrongly.
"""

import json

import numpy as np

import records
from noise_design import profile, raw_information
from offgrid_and_compound import COARSE_R, COARSE_S, FINE_R, FINE_S, SOURCE, pieces

ROUNDS = 4
TOLERANCE = 1e-6
ZOOM_SPAN = 0.04          # +/- 4% in each coordinate around a point worth looking at
ZOOM_STEPS = 5
# Only the worst exceedances join the candidate set each round. Adding all of them, with a zoom
# box each, grows the set faster than the solves can keep up and buys nothing: the Fedorov--Wynn
# step needs the argmax, and the rest are its neighbours.
MAX_ADD = 12
# the candidate space sec:design offers, which is what the certificate is about
RADIUS_RANGE = (float(FINE_R.min()), float(FINE_R.max()))
STRETCH_RANGE = (float(FINE_S.min()), float(FINE_S.max()))
# two candidates closer than this in both coordinates are the same setting to any experimenter,
# and keeping both splits the certified weight between duplicates -- the previous run reported
# 256.0 um @ 18.442 twice, with weights 0.325 and 0.152
SAME_RADIUS = 1e-3        # relative
SAME_STRETCH = 2e-3       # relative


def midpoints(values):
  """The gaps in a sorted axis: points no candidate on that axis occupies."""
  axis = np.sort(np.unique(np.asarray(values, dtype=float)))
  return np.sqrt(axis[:-1] * axis[1:]) if np.all(axis > 0) else 0.5 * (axis[:-1] + axis[1:])


def distinct(points, protect=()):
  """Drop candidates that duplicate one already kept, to within what a laser could tell apart.

  `protect` is seeded first and therefore always survives. That is not a nicety: the points the
  scan flagged ARE the vertex-direction step, and a greedy pass over sorted coordinates will
  otherwise discard a flagged point in favour of a neighbour a fraction of a percent away --- its
  own zoom box centres the radius at $\\sqrt{0.96\\times1.04}$ of it, which is inside the dedupe
  radius. The flagged point then never enters the candidate set, reappears in the next scan
  unchanged, and the loop reports the same exceedance at the same geometry for ever.
  """
  kept = list(protect)
  for radius, stretch in sorted(points):
    if any(abs(radius / r - 1.0) < SAME_RADIUS and abs(stretch / s - 1.0) < SAME_STRETCH
           for r, s in kept):
      continue
    kept.append((radius, stretch))
  return sorted(kept)


def zoom_around(points):
  """A small dense box around each point, clipped to the design domain.

  The clip is the whole point. Without it a flagged point near an edge walks the zoom outside
  the box on the next round, and the round after that walks it further: an unclipped version of
  this loop reported its worst exceedances at \\SI{37.6}{\\micro\\metre} and stretch $26.3$, both
  outside the \\SIrange{50}{1200}{\\micro\\metre} by $3$--$20$ candidate space the chapter
  offers. A design that cannot be requested is not a counterexample to a certificate over
  designs that can.
  """
  out = []
  for radius, stretch in points:
    for r in np.geomspace(radius * (1 - ZOOM_SPAN), radius * (1 + ZOOM_SPAN), ZOOM_STEPS):
      for s in np.linspace(stretch * (1 - ZOOM_SPAN), stretch * (1 + ZOOM_SPAN), ZOOM_STEPS):
        if RADIUS_RANGE[0] <= r <= RADIUS_RANGE[1] and STRETCH_RANGE[0] <= s <= STRETCH_RANGE[1]:
          out.append((float(r), float(s)))
  return out


def main():
  from pyimr.measure import optimal_measure

  tau, spread = profile(SOURCE)
  cache = records.HERE / "grid_refinement_cache.npz"
  built = {}
  if cache.exists():
    stored = np.load(cache, allow_pickle=True)
    # dtype=float is not decoration: `np.savez` with dtype=object on same-shaped matrices stores
    # a 3-D object array, and reading it back without a cast hands `np.linalg.inv` dtype('O')
    built = {tuple(k): (None if v is None else np.asarray(v, dtype=float))
             for k, v in zip(stored["keys"], stored["values"], strict=True)}
    print(f"  {len(built)} geometries read from cache")

  def ensure(wanted):
    """Solve and whiten every geometry not already held. Returns those that integrate."""
    missing = sorted({(round(r, 12), round(s, 9)) for r, s in wanted} - set(built))
    if missing:
      print(f"    solving {len(missing)} new geometries ...", flush=True)
      with records.pool(len(missing)) as pool:
        for design, payload in pool.map(raw_information, missing):
          built[design] = None if payload is None else pieces(payload, tau, spread)[0]
      # written after every batch: these solves cost tens of minutes and an interrupted run
      # that has to redo them is the difference between a rerun and an afternoon
      np.savez(cache, keys=np.array(list(built), dtype=object),
               values=np.array(list(built.values()), dtype=object))
    # sorted, not set order: the candidate list indexes the support, and a run whose
    # tie-breaking depends on hash order is not reproducible
    return sorted(d for d in {(round(r, 12), round(s, 9)) for r, s in wanted}
                  if built.get(d) is not None)

  # the independent scan set, fixed once: the gaps of the base grid in both coordinates
  SCAN_LATTICE = [(float(r), float(s))
                  for r in midpoints(FINE_R) for s in midpoints(FINE_S)]

  candidates = ensure([(float(r), float(s)) for r in FINE_R for s in FINE_S])
  print(f"\n  round 0 candidate set: {len(candidates)} geometries "
        f"({len(FINE_R)}x{len(FINE_S)}, the grid sec:offgrid scanned)\n")

  # what sec:offgrid reported, recomputed here so the two are on one footing
  coarse = [d for d in ensure([(float(r), float(s)) for r in COARSE_R for s in COARSE_S])]
  coarse_stack = np.array([built[d] for d in coarse], dtype=float)
  coarse_held = optimal_measure(coarse_stack, iterations=200_000)
  coarse_inverse = np.linalg.inv(np.tensordot(coarse_held.weights, coarse_stack, axes=(0, 0)))
  parameters = coarse_stack.shape[1]
  reference = max(float(np.trace(coarse_inverse @ built[d])) for d in candidates)
  print(f"  the coarse measure, rescanned: max d = {reference:.3f} against p = {parameters}\n")

  print(f"  {'round':>5s} {'candidates':>11s} {'settings':>9s} {'gap':>9s} {'scan':>6s} "
        f"{'max d off-set':>14s} {'exceedance':>11s} {'worst at':>22s}")
  history = []
  flagged = []
  for step in range(ROUNDS):
    stack = np.array([built[d] for d in candidates], dtype=float)
    held = optimal_measure(stack, iterations=200_000)
    inverse = np.linalg.inv(np.tensordot(held.weights, stack, axes=(0, 0)))
    support = [candidates[i] for i in held.support]

    # an INDEPENDENT scan: the FIXED midpoint lattice of the base grid, which no original
    # candidate occupies, plus zooms around the current support and the last round's flags.
    # The lattice is deliberately not rebuilt from `candidates`: once refinement adds scattered
    # points the candidate set stops being a grid, and the tensor product of its unique radii
    # and unique stretches explodes -- 46x40 becomes ~190x190 after one round, which is 36000
    # probe points for a scan that only ever needed the gaps in the original grid.
    probe = list(SCAN_LATTICE) + zoom_around(support) + zoom_around(flagged)
    scanned = ensure(probe)
    scanned = [d for d in scanned if d not in set(candidates)]
    if not scanned:
      print("  scan set empty; nothing independent left to test")
      break
    values = np.array([float(np.trace(inverse @ built[d])) for d in scanned])
    worst = int(np.argmax(values))
    exceed = float(values.max() - parameters)
    print(f"  {step:5d} {len(candidates):11d} {held.support.size:9d} {held.gap:9.1e} "
          f"{len(scanned):6d} {values.max():14.4f} {exceed:11.2e} "
          f"{scanned[worst][0] * 1e6:9.1f} um @ {scanned[worst][1]:6.3f}")
    history.append({"round": step, "candidates": len(candidates),
                    "settings": int(held.support.size), "gap": held.gap,
                    "scanned": len(scanned), "max_d": float(values.max()),
                    "exceedance": exceed, "worst": list(scanned[worst]),
                    "support": [[d[0], d[1], float(held.weights[i])]
                                for i, d in zip(held.support, support, strict=True)]})
    if exceed <= TOLERANCE:
      print("\n  The certificate now reaches past its candidate set: an independent scan finds")
      print("  nothing above the bound, so the measure is optimal over the continuum too.")
      break
    # the Fedorov-Wynn move: everything the independent scan flagged joins the candidate set
    over = [(v, d) for d, v in zip(scanned, values, strict=True) if v > parameters + TOLERANCE]
    flagged = [d for _, d in sorted(over, reverse=True)[:MAX_ADD]]
    print(f"  {'':5s} {len(over)} points exceeded; adding the worst {len(flagged)} "
          f"and their neighbourhoods")
    added = ensure(flagged + zoom_around(flagged))
    # the flagged points are protected: they are the argmax the step exists to add
    candidates = distinct(set(candidates) | set(added), protect=ensure(flagged))

  summary = {"p": int(parameters), "coarse_rescanned_max_d": float(reference),
             "rounds": history,
             "converged": bool(history and history[-1]["exceedance"] <= TOLERANCE)}

  print("\n  what the refinement did to the answer, which is the question sec:offgrid asked\n")
  if history:
    first, last = history[0], history[-1]
    print(f"  {'':>22s} {'round 0':>22s} {'final':>22s}")
    print(f"  {'candidates':>22s} {first['candidates']:22d} {last['candidates']:22d}")
    print(f"  {'settings in support':>22s} {first['settings']:22d} {last['settings']:22d}")
    print(f"  {'max d, independent':>22s} {first['max_d']:22.4f} {last['max_d']:22.4f}")
    moved = {(round(r * 1e6), round(s, 2)) for r, s, _ in first["support"]}
    ends = {(round(r * 1e6), round(s, 2)) for r, s, _ in last["support"]}
    summary["support_kept"] = len(moved & ends)
    print(f"  {'support kept':>22s} {len(moved & ends):22d} {'of ' + str(len(ends)):>22s}")
    print("\n  final support:")
    for radius, stretch, weight in sorted(last["support"], key=lambda row: -row[2]):
      print(f"    {radius * 1e6:8.1f} um @ {stretch:6.3f}   weight {weight:.4f}")

  json.dump(summary, open(records.HERE / "grid_refinement.json", "w"), indent=1)


if __name__ == "__main__":
  main()
