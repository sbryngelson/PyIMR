r"""Is $\alpha$ identified by the data, or by the floor under $g$?

\Cref{sec:identifiability} samples $\alpha$ against ceilings of $1$, $10$ and $100$, finds the
posterior stops tracking the ceiling once it is loose, and reads that as identification. The
reading does not follow, and the reason is arithmetic. The likelihood depends on the product
$g\alpha$; the box floors $g$ at \SI{100}{\pascal}; the identified product at \SI{15}{\celsius}
is \SI{1106}{\pascal}. So $\alpha$ cannot exceed $1106/100 \approx 11$ whatever its own ceiling
is set to, and the two loose rows report $97.5\%$ points of $9.04$ and $8.73$ just under that
cap. They agree with each other because the same wall stops both. Sweeping $\alpha$'s ceiling
cannot separate ``the data determine $\alpha$'' from ``the floor under $g$ determines
$\alpha$'', because it never moves the wall that binds.

THE TEST IS TO MOVE THE OTHER WALL, and it costs almost nothing once the structure is used. A
box bound is a PRIOR TRUNCATION, not a different likelihood, so every floor and every ceiling in
both sweeps is a different region of one and the same surface. Evaluate $\chi^2$ once on a grid
over $(g,\alpha)$, then read each box off it by truncating and renormalising. Sequential Monte
Carlo was tried first and abandoned: it runs order $10^5$ forward solves per box because it
tempers through many stages, and it would have recomputed the identical surface seven times.

WHAT IS HELD, AND WHY THAT IS FAIR. The viscosity and the relaxation time are fixed at their
fitted values. \Cref{sec:identifiability} measures the sloppy eigenvector as almost purely
modulus-against-stiffening, weights $-0.796$ and $+0.604$ against $-0.003$ and $-0.032$ for
those two, so they are close to orthogonal to the direction under test and holding them cannot
manufacture the effect. It does mean the marginals below are conditional on them, which is
stated rather than hidden.

WHAT WOULD REFUTE THE FLOOR READING. A median $\alpha$ that stays put as the floor moves four
decades, or an upper quantile that stops well short of $g\alpha/g_{\min}$ once that cap is
loose. Either would say the data are doing the work.
"""

import json

import numpy as np

import records

DATASET, R_MAX = "gelatin_15C", 277e-6
G_GRID = (1e0, 1e5, 96)                   # spans every floor either sweep asks for
ALPHA_GRID = (1e-3, 1e2, 96)
MU_BOX, LAM_BOX = (1e-4, 1.0), (1e-9, 1e-3)
G_FLOORS = (1e0, 1e1, 1e2, 1e3)           # 1e2 is the published floor
ALPHA_CEILINGS = (1.0, 10.0, 100.0)       # the published sweep, for contrast and provenance
BETA_GRID = np.geomspace(0.05, 10.0, 240)  # the half-Cauchy scale of sec:beta


def _load():
  record = json.load(open(records.HERE / "results.json"))[DATASET]
  t = np.asarray(record["times_s"], dtype=float)
  y = np.asarray(record["mean"], dtype=float)
  s = np.asarray(record["spread"], dtype=float)
  keep = s > 0
  return t[keep], y[keep], s[keep], record["stretch"]


def chi2_at(job):
  """`chi^2` at one $(g,\\alpha)$ node, with viscosity and relaxation time held."""
  import pyimr

  g, alpha, mu, lam, stretch = job
  times, mean, spread, _ = _load()
  config = pyimr.SimulationConfig(
    R_MAX, R_MAX / stretch, pyimr.QuadraticZener(g, mu, lam, 0.0, alpha),
    dynamics="keller-miksis", rtol=1e-8, atol=1e-10, max_steps=300_000)
  try:
    trace = np.asarray(pyimr.simulate(times, config).radius_ratio, dtype=float)
  except Exception:                                                          # noqa: BLE001
    return (g, alpha), None
  if not np.all(np.isfinite(trace)): return (g, alpha), None
  return (g, alpha), float(np.sum(((mean - trace) / spread) ** 2))


def marginal_log_likelihood(chi2, n):
  """`log int p(y|theta,beta) pi(beta) dbeta` on the half-Cauchy of \\cref{sec:beta}."""
  b = BETA_GRID
  terms = -0.5 * chi2 / b**2 - n * np.log(b) + np.log((2.0 / np.pi) / (1.0 + b**2))
  top = terms.max()
  return top + np.log(np.trapezoid(np.exp(terms - top), b))


def quantiles(logp, g, alpha, g_floor, alpha_ceiling):
  """Median and 97.5% of `alpha` under one box, by truncating the shared surface."""
  keep_g, keep_a = (g >= g_floor * 0.999), (alpha <= alpha_ceiling * 1.001)
  block = logp[np.ix_(keep_g, keep_a)]
  if not np.any(np.isfinite(block)): return None
  w = np.exp(block - np.nanmax(block))
  w[~np.isfinite(w)] = 0.0
  marg = w.sum(axis=0)                                    # marginalise g away
  if marg.sum() <= 0: return None
  marg = marg / marg.sum()
  a = alpha[keep_a]
  cdf = np.cumsum(marg)
  med = float(np.interp(0.5, cdf, a))
  up = float(np.interp(0.975, cdf, a))
  joint = w / w.sum()
  gg = g[keep_g][:, None] * np.ones_like(a)[None, :]
  aa = np.ones_like(g[keep_g])[:, None] * a[None, :]
  return {"alpha_median": med, "alpha_upper": up,
          "g_median": float(np.exp(np.sum(joint * np.log(gg)))),
          "galpha_median": float(np.exp(np.sum(joint * np.log(gg * aa)))),
          "cap_from_floor": float(np.exp(np.sum(joint * np.log(gg * aa))) / g_floor)}


def main():
  from pyimr.selection import fit_candidate, physical_from_unit

  from density_clock import _box, candidate_for
  times, mean, spread, stretch = _load()

  # the two axes that are held: taken from a fit rather than assumed
  cand = candidate_for(())
  from density_clock import solver_with_start
  solve = solver_with_start(times, R_MAX, stretch, 998.0, 1064.0)
  fit = fit_candidate(cand, solve, mean, spread, bounds=_box(()), starts=24, max_evaluations=700)
  vals = dict(zip(cand.axes, (float(v) for v in physical_from_unit(cand.axes, fit.unit, _box(()))),
                  strict=True))
  mu, lam, product = vals["mu"], vals["lambda1"], vals["galpha"]
  print(f"  held at the fit: mu = {mu:.5f}, lambda1 = {lam:.3e}; identified g*alpha = {product:.0f}\n")

  g = np.geomspace(*G_GRID[:2], G_GRID[2])
  alpha = np.geomspace(*ALPHA_GRID[:2], ALPHA_GRID[2])
  jobs = [(float(a), float(b), mu, lam, stretch) for a in g for b in alpha]
  print(f"  one surface, {len(jobs)} nodes over g x alpha ...", flush=True)
  with records.pool(len(jobs)) as pool:
    got = dict(pool.map(chi2_at, jobs))

  n = mean.size
  logp = np.full((g.size, alpha.size), -np.inf)
  for i, a in enumerate(g):
    for j, b in enumerate(alpha):
      c = got[(float(a), float(b))]
      if c is not None: logp[i, j] = marginal_log_likelihood(c, n)
  ok = np.isfinite(logp).sum()
  print(f"  {ok} of {logp.size} nodes integrate ({100*ok/logp.size:.0f}%)\n")

  summary = {"dataset": DATASET, "mu": mu, "lambda1": lam, "galpha_fit": product,
             "nodes": int(logp.size), "integrated": int(ok), "sweeps": {}}
  for kind, title, floors, ceilings in (
      ("g_floor", "A: sweeping the g FLOOR, alpha's box held at 100", G_FLOORS, [1e2] * 4),
      ("alpha_ceiling", "B: sweeping the alpha CEILING, g's floor held at 100 (the published table)",
       [1e2] * 3, ALPHA_CEILINGS)):
    print(f"  ==== {title} ====\n")
    print(f"  {'swept':>10s} {'median a':>9s} {'97.5% a':>9s} {'median g':>10s} "
          f"{'median ga':>10s} {'ga/g_min':>9s}")
    for floor, ceil in zip(floors, ceilings, strict=True):
      q = quantiles(logp, g, alpha, floor, ceil)
      swept = floor if kind == "g_floor" else ceil
      if q is None:
        print(f"  {swept:10.0f}   empty")
        continue
      summary["sweeps"].setdefault(kind, {})[f"{swept:.0f}"] = q
      print(f"  {swept:10.0f} {q['alpha_median']:9.2f} {q['alpha_upper']:9.2f} "
            f"{q['g_median']:10.0f} {q['galpha_median']:10.0f} {q['cap_from_floor']:9.1f}")
    print()

  a = summary["sweeps"].get("g_floor", {})
  if len(a) >= 2:
    med = np.array([v["alpha_median"] for v in a.values()])
    up = np.array([v["alpha_upper"] for v in a.values()])
    cap = np.array([v["cap_from_floor"] for v in a.values()])
    prod = np.array([v["galpha_median"] for v in a.values()])
    print("  ---- what the sweep says ----\n")
    print(f"  median alpha over the g floors:  {np.array2string(med, precision=2)}")
    print(f"  97.5% alpha:                     {np.array2string(up, precision=2)}")
    print(f"  cap the floor alone permits:     {np.array2string(cap, precision=1)}")
    print(f"  median g*alpha:                  {np.array2string(prod, precision=0)}")
    span, pspan = med.max() / med.min(), prod.max() / prod.min()
    summary["alpha_span"], summary["galpha_span"] = float(span), float(pspan)
    print(f"\n  alpha's median moves {span:.0f}x as the floor moves "
          f"{max(G_FLOORS)/min(G_FLOORS):.0f}x, while g*alpha moves {pspan:.2f}x")
    print("  A product that holds while its factors slide is the degeneracy; an alpha that")
    print("  slides with the floor is a bound reporting itself as a measurement.")
  json.dump(summary, open(records.HERE / "g_floor.json", "w"), indent=1)


if __name__ == "__main__":
  main()
