"""Design when the criteria disagree: trade-off fronts rather than a single optimum.

`optimize_design` maximises one number. That is the right tool only while the criteria
happen to agree. Measured on the qSLS study they do not: identifying `g` against `alpha`
is best at 277 um and stretch 5, separating qSLS from qKV is best at 100 um and stretch
20 -- the opposite corner, because detecting a relaxation mode needs a collapse fast
enough to excite it. There is no design that is best at both, so the honest answer is the
set of designs where improving one criterion costs another.

The search is ParEGO (Knowles 2006): one shared archive of evaluated designs, a fresh
random weight each iteration, and a Chebyshev scalarisation. Every evaluation feeds every
weight, which is what makes it affordable when one design costs a hundred solves.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from scipy.optimize import minimize

from .optimize import _fit, _expected_improvement, _physical, _posterior

__all__ = ["ParetoResult", "chebyshev_scalarize", "explore_tradeoff", "pareto_indices"]

def pareto_indices(values):
  """Indices of the non-dominated rows of `values`, every column treated as maximised.

  Row `a` dominates row `b` when it is at least as good everywhere and strictly better
  somewhere. Identical rows therefore do not dominate each other and both survive.
  """
  table = np.atleast_2d(np.asarray(values, dtype=float))
  if table.ndim != 2: raise ValueError(f"values must be (points, objectives); got shape {table.shape}")
  keep = []
  for index, row in enumerate(table):
    others = np.delete(table, index, axis=0)
    if others.size and np.any(np.all(others >= row, axis=1) & np.any(others > row, axis=1)): continue
    keep.append(index)
  return np.array(keep, dtype=int)

def chebyshev_scalarize(values, weights, *, augment=1e-3):
  """Augmented Chebyshev scalarisation of maximised objectives, one score per row.

  A weighted SUM cannot do this job. It returns only points on the convex hull of the
  front, so wherever the front bulges inward -- which is what a genuine trade-off between
  competing mechanisms looks like -- every weight returns the same two endpoints and the
  entire interior is unreachable. Scoring each design by its WORST weighted shortfall
  from the best observed value instead reaches every point of the front, convex or not.
  `augment` adds a small multiple of the total shortfall, which breaks ties among designs
  sharing a worst component without reintroducing the weighted sum's blind spot.

  Weights act the way the name suggests: raising `weights[k]` magnifies shortfalls on
  objective `k`, so the score is maximised by designs that do well on it. A unit vector
  recovers that objective alone.

  Columns are rescaled to their observed range first, so criteria with different units
  (an eigenvalue against a count of noise units) can be weighted meaningfully.
  """
  table = np.atleast_2d(np.asarray(values, dtype=float))
  vector = np.asarray(weights, dtype=float)
  if table.shape[1] != vector.size: raise ValueError(f"got {table.shape[1]} objectives and {vector.size} weights")
  if np.any(vector < 0.0) or not np.any(vector > 0.0): raise ValueError("weights must be non-negative and not all zero")
  if float(augment) < 0.0: raise ValueError("augment must be non-negative")

  spread = np.ptp(table, axis=0)
  spread[spread <= 0.0] = 1.0
  shortfall = (1.0 - (table - table.min(axis=0)) / spread) * vector
  return -(shortfall.max(axis=1) + float(augment) * shortfall.sum(axis=1))

@dataclass(frozen=True, slots=True)
class ParetoResult:
  """Every design evaluated, and which of them nothing else beat."""

  points: np.ndarray
  values: np.ndarray
  front: np.ndarray
  feasible: np.ndarray
  weights: np.ndarray

  @property
  def front_points(self): return self.points[self.front]

  @property
  def front_values(self): return self.values[self.front]

  def best_for(self, objective):
    """The evaluated design that scored highest on one objective alone."""
    column = np.where(self.feasible, self.values[:, int(objective)], -np.inf)
    return self.points[int(np.argmax(column))]

def _feasible_rows(values): return np.all(np.isfinite(values), axis=1)

def _latin_hypercube(count, dimension, rng):
  """One point in each of the `count` strata of every axis, jittered inside its stratum.

  Built here rather than taken from `scipy.stats.qmc`, whose seeding keyword was renamed
  across versions; the construction is two lines and this keeps the module independent of
  which scipy is installed.
  """
  strata = np.argsort(rng.random((count, dimension)), axis=0)
  return (strata + rng.random((count, dimension))) / count

def explore_tradeoff(objective, bounds, *, evaluations=32, initial=8, seed=0, restarts=8, augment=1e-3):
  """Trace the Pareto front of a vector-valued expensive `objective` over a box.

  `objective(point)` returns a sequence of `k` numbers, all of which are MAXIMISED. Return
  a non-finite entry for a design that cannot be evaluated -- an infeasible design is kept
  in the archive as a record but excluded from the surrogate and from the front, so a
  region the physics forbids does not poison the fit.

  The first `k` guided iterations use the pure single-objective weights, which pins the
  ends of the front before the random weights explore its interior.

  The initial design is a Latin hypercube, not uniform random. That is not a detail: with
  ten uniform draws over the qSLS design plane, seven landed above 650 um and the search
  never sampled the narrow E-optimality ridge near 277 um, reporting a best of 0.13
  against the ridge's 1.84. A Latin hypercube stratifies every axis by construction, so a
  feature that is narrow in one coordinate cannot be missed for want of a starting point.
  """
  box = np.asarray(bounds, dtype=float)
  if box.ndim != 2 or box.shape[1] != 2: raise ValueError(f"bounds must be (dimension, 2); got shape {box.shape}")
  if not np.all(box[:, 1] > box[:, 0]): raise ValueError("every upper bound must exceed its lower bound")
  if not isinstance(initial, Integral) or int(initial) < 2: raise ValueError("need at least two initial evaluations to fit a surrogate")
  if int(evaluations) < int(initial): raise ValueError("evaluations must cover the initial design")

  dimension = box.shape[0]
  rng = np.random.default_rng(seed)
  unit = _latin_hypercube(int(initial), dimension, rng)
  archive = [np.asarray(objective(_physical(row, box)), dtype=float).ravel() for row in unit]
  values = np.array(archive, dtype=float)
  if values.ndim != 2 or values.shape[1] < 2: raise ValueError("objective must return at least two values per design")
  count = values.shape[1]

  corners = list(np.eye(count))
  drawn = []
  for step in range(int(evaluations) - int(initial)):
    feasible = _feasible_rows(values)
    if feasible.sum() < 2: raise ValueError(f"only {int(feasible.sum())} of {len(values)} designs were feasible; widen the bounds")
    weights = corners[step] if step < len(corners) else rng.dirichlet(np.ones(count))
    drawn.append(weights)

    scored = chebyshev_scalarize(values[feasible], weights, augment=augment)
    seen = unit[feasible]
    centre, spread = scored.mean(), scored.std() or 1.0
    normalised = (scored - centre) / spread
    quiet = np.zeros_like(normalised)
    log_scale, log_lengths = _fit(seen, normalised, quiet)
    incumbent = float(_posterior(seen, seen, normalised, quiet, log_scale, log_lengths)[0].max())

    def negative_acquisition(candidate, _s=log_scale, _l=log_lengths, _i=incumbent, _p=seen, _v=normalised, _q=quiet):
      point = np.clip(np.atleast_2d(candidate), 0.0, 1.0)
      return -float(_expected_improvement(point, _p, _v, _q, _s, _l, _i)[0])

    best_point, best_acquisition = None, np.inf
    for start in rng.random((int(restarts), dimension)):
      outcome = minimize(negative_acquisition, start, method="L-BFGS-B", bounds=[(0.0, 1.0)] * dimension)
      if outcome.fun < best_acquisition: best_point, best_acquisition = np.clip(outcome.x, 0.0, 1.0), float(outcome.fun)
    if best_point is None: break

    unit = np.vstack([unit, best_point])
    values = np.vstack([values, np.asarray(objective(_physical(best_point, box)), dtype=float).ravel()])

  feasible = _feasible_rows(values)
  survivors = np.where(feasible)[0]
  front = survivors[pareto_indices(values[feasible])] if survivors.size else survivors
  return ParetoResult(
    points=_physical(unit, box), values=values, front=front, feasible=feasible,
    weights=np.array(drawn, dtype=float).reshape(-1, count),
  )
