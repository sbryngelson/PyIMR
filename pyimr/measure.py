"""Design measures: a convex formulation that can certify its own answer.

Every design search elsewhere in this package optimises over a single design point, and
that problem is not convex. The consequences were measured: a surrogate search recovered
16% of the best conditioning available, a dense grid over an unreliable inner criterion
produced a confidently wrong front, and nothing in any of those results indicated a
problem. There was no way to tell a good answer from a bad one.

The classical formulation avoids this. Optimise not over one design $x$ but over a
probability measure $\\xi$ on the candidate designs -- in practice, run $n_1$ bubbles at one
setting and $n_2$ at another, which is what an experimenter does anyway and is strictly
more expressive than choosing a single setting. The information matrix is then an average,

    M(xi) = sum_i xi_i M(x_i),

which is LINEAR in `xi`. Composing it with a concave criterion (`log det`, `-tr` of the
inverse) makes the whole problem concave over the simplex, so it has no local optima, and
its first-order condition is necessary and sufficient. That condition -- the General
Equivalence Theorem of Kiefer and Wolfowitz -- says a measure is optimal exactly when no
single design has a directional derivative pointing uphill:

    phi(x, xi*) <= 0   for every candidate x,   with equality on the support of xi*.

`optimal_measure` returns that number. A non-positive gap is a proof of global optimality,
not a report that a search stopped improving.

Discrimination fits the same frame: a utility that is an average over the design, such as
an expected log Bayes factor or a squared model discrepancy, is linear in `xi` and hence
concave, so it can be combined with the estimation criteria in one compound objective and
still be certified. That is the classical answer to the trade-off that `pyimr.pareto`
addresses by search, and it is a better one wherever it applies.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np

__all__ = ["BatchDesign", "ExactDesign", "MeasureResult", "apportion", "identification_front", "optimal_measure", "sensitivity"]

_FLOOR = 1e-12

def _checked(matrices):
  stack = np.asarray(matrices, dtype=float)
  if stack.ndim != 3 or stack.shape[1] != stack.shape[2]:
    raise ValueError(f"matrices must be (candidates, p, p); got shape {stack.shape}")
  if stack.shape[0] < 1: raise ValueError("need at least one candidate design")
  if not np.all(np.isfinite(stack)): raise ValueError("information matrices must be finite")
  if not np.allclose(stack, np.swapaxes(stack, 1, 2), atol=1e-8):
    raise ValueError("information matrices must be symmetric")
  return stack

def _averaged(stack, weights): return np.tensordot(weights, stack, axes=(0, 0))

def _value(stack, weights, utility, blend):
  """The compound criterion: `log det` of the averaged information, blended with a utility."""
  linear = 0.0 if utility is None else float(weights @ utility)
  if blend >= 1.0: return linear                       # the determinant is not merely unused:
  sign, magnitude = np.linalg.slogdet(_averaged(stack, weights))   # it may be -inf, and 0*-inf is nan
  determinant = magnitude if sign > 0 else -np.inf
  return determinant if utility is None else (1.0 - blend) * determinant + blend * linear

def sensitivity(matrices, weights, *, utility=None, blend=0.0):
  """Directional derivative of the criterion toward each candidate, one number per design.

  For `log det`, moving mass toward the design `x` changes the criterion at rate
  `tr[M(xi)^-1 M(x)] - p`; for a linear utility the rate is `u_x - u . xi`. The maximum
  over candidates is the equivalence-theorem gap, and it is zero at an optimum because a
  concave function on a simplex is maximised exactly where no vertex direction ascends.
  """
  stack = _checked(matrices)
  share = np.asarray(weights, dtype=float)
  if share.shape != (stack.shape[0],): raise ValueError(f"weights must have {stack.shape[0]} entries; got {share.shape}")
  if np.any(share < -1e-12) or not np.isclose(share.sum(), 1.0): raise ValueError("weights must be non-negative and sum to one")
  vector = None
  if utility is not None:
    vector = np.asarray(utility, dtype=float)
    if vector.shape != (stack.shape[0],): raise ValueError(f"utility must have {stack.shape[0]} entries; got {vector.shape}")

  try:
    return _sensitivity(stack, share, vector, float(blend))
  except np.linalg.LinAlgError as error:
    raise ValueError("the averaged information matrix is singular; spread the starting weights") from error

def _sensitivity(stack, share, vector, blend):
  """The derivative without revalidation: the iteration calls this once per step."""
  if blend >= 1.0: return vector - float(share @ vector)
  direction = np.einsum("jk,ikj->i", np.linalg.inv(_averaged(stack, share)), stack) - stack.shape[1]
  if vector is None: return direction
  return (1.0 - blend) * direction + blend * (vector - float(share @ vector))

@dataclass(frozen=True, slots=True)
class MeasureResult:
  """An optimal design measure, and the certificate that it is optimal."""

  weights: np.ndarray
  support: np.ndarray
  value: float
  gap: float
  iterations: int

  @property
  def certified(self):
    """Whether the equivalence theorem confirms global optimality, not merely convergence."""
    return bool(self.gap <= 1e-6 * max(1.0, abs(self.value)))

def optimal_measure(matrices, *, utility=None, blend=0.0, iterations=100_000, tolerance=1e-9, prune=1e-7):
  """Maximise a concave design criterion over measures on the candidates.

  `matrices[i]` is the information a single run at candidate `i` provides. `utility[i]`, if
  given, is a per-candidate quantity that averages over the design -- an expected log Bayes
  factor, say -- and `blend` in `[0, 1]` sets how much of the objective it carries, giving
  the compound criteria of Atkinson and of Cook and Wong. Both parts are concave, so their
  blend is too and the certificate remains valid for any `blend`.

  The search is the multiplicative algorithm of Silvey, Titterington and Torsney:
  `w_i <- w_i F_i / sum_j w_j F_j`, where `F_i` is the derivative of the criterion in the
  direction of candidate `i`. Every update stays on the simplex by construction and, for a
  concave criterion, increases it. Vertex-direction steps with a line search were tried
  first and converge too slowly to certify -- the step needed near the optimum falls below
  any fixed search grid, so the iteration stalls with a positive gap.

  Convexity is what makes any of this sufficient: there is nowhere else for a maximum to
  hide. The same derivative that drives the update is the quantity that certifies the
  answer once it stops being positive anywhere.

  Convergence slows markedly as candidates crowd together, because a design just off the
  support is nearly as good as one on it and its weight decays in proportion. Certifying a
  parabola on $[-1,1]$ took 6400 iterations over 51 candidates, 25000 over 101
  and 329000 over 401; the criterion value was correct to six figures long before any
  of those. So a large candidate set may exhaust `iterations` with a small positive gap, and
  `MeasureResult.certified` is how that is detected -- an uncertified result is a converged
  value without a proof, not a wrong answer.
  """
  stack = _checked(matrices)
  count = stack.shape[0]
  if not 0.0 <= float(blend) <= 1.0: raise ValueError("blend must lie in [0, 1]")
  if utility is None and float(blend) > 0.0: raise ValueError("blend is positive but no utility was given")
  if not isinstance(iterations, Integral) or int(iterations) < 1: raise ValueError("iterations must be a positive integer")

  share = np.full(count, 1.0 / count)
  if blend < 1.0 and not np.isfinite(_value(stack, share, None, 0.0)):
    raise ValueError("the candidates cannot jointly determine every parameter: their averaged information is singular")

  vector = None if utility is None else np.asarray(utility, dtype=float)
  # The multiplicative update needs a positive derivative. Shifting the utility to be
  # positive adds a constant to the criterion, so the maximiser is untouched.
  shifted = None if vector is None else vector - vector.min() + 1.0

  gap, step = np.inf, 0
  for step in range(1, int(iterations) + 1):
    gap = float(np.max(_sensitivity(stack, share, vector, blend)))
    if gap <= tolerance: break
    if blend >= 1.0:
      growth = shifted
    else:
      variance = np.einsum("jk,ikj->i", np.linalg.inv(_averaged(stack, share)), stack)
      growth = variance if shifted is None else (1.0 - blend) * variance + blend * shifted
    updated = share * growth
    total = updated.sum()
    if not np.isfinite(total) or total <= 0.0: break
    share = updated / total
    # A candidate just off the support decays like (d_i/p)^k with d_i/p barely below one,
    # so the tail costs more iterations than the answer does. Dropping it periodically
    # costs nothing -- the update keeps zeros at zero -- and the certificate is still
    # computed against every candidate afterwards, so a wrong drop cannot hide.
    if step % 200 == 0:
      surviving = share > prune
      if surviving.any() and not surviving.all():
        share = np.where(surviving, share, 0.0)
        share = share / share.sum()

  keep = share > prune
  if not keep.any(): keep = share >= share.max()
  cleaned = np.where(keep, share, 0.0)
  cleaned = cleaned / cleaned.sum()
  # pruning perturbs the measure, so the reported gap must be the one the answer actually has
  final = float(np.max(_sensitivity(stack, cleaned, vector, float(blend))))
  return MeasureResult(weights=cleaned, support=np.flatnonzero(keep), value=_value(stack, cleaned, utility, blend),
                       gap=final, iterations=step)


@dataclass(frozen=True, slots=True)
class ExactDesign:
  """An integer allocation of `runs` to candidates, and what it costs against the measure."""

  counts: np.ndarray
  support: np.ndarray
  efficiency: float
  runs: int

  @property
  def table(self):
    """`(candidate index, run count)` for the candidates that get runs, most runs first."""
    order = np.argsort(-self.counts[self.support])
    return [(int(self.support[i]), int(self.counts[self.support[i]])) for i in order]


def apportion(weights, runs, matrices=None, *, utility=None, blend=0.0):
  """Turn a design measure into a whole number of runs per candidate.

  `optimal_measure` returns weights, and an experiment runs integers. With three support
  points at 0.45/0.31/0.24 and twelve bubbles, somebody has to choose 5/4/3 or 6/4/2, and the
  equivalence-theorem certificate says nothing about which -- it certifies optimality over
  MEASURES, and an exact design is not one.

  The method is the efficient rounding of Pukelsheim and Rieder (1992): give every supported
  candidate one run, hand out the rest by largest remainder, and take the best over the
  admissible starting multipliers. It is the apportionment rule that maximises the smallest
  ratio `n_i / (runs * w_i)`, which is what keeps a support point from being rounded away.

  This is a heuristic and it is honest about that. The literature is explicit that rounding
  works well for large `runs` and carries no guarantee for small ones, which is exactly the
  regime IMR reaches. So pass `matrices` and the efficiency is MEASURED --
  `(det M(exact) / det M(xi))^(1/p)`, the D-efficiency of the integer design against the
  measure it came from -- rather than assumed. For a handful of runs, read that number before
  believing the design; an exact-design search is the alternative when it is poor.

  That efficiency is always the D-efficiency of the INFORMATION, even when `utility` and
  `blend` were used to build the measure. A ratio of compound criterion values is not an
  efficiency -- `log det` and a utility in nats have different units and either sign, so the
  ratio can exceed one for a design that is worse. `identification_front` reports the utility
  achieved separately, in nats, because the two halves do not share a scale.
  """
  share = np.asarray(weights, dtype=float)
  if share.ndim != 1: raise ValueError("weights must be one-dimensional")
  if not np.all(np.isfinite(share)) or np.any(share < 0.0): raise ValueError("weights must be finite and non-negative")
  total = share.sum()
  if total <= 0.0: raise ValueError("weights must not be all zero")
  share = share / total
  if not isinstance(runs, Integral) or int(runs) < 1: raise ValueError("runs must be a positive integer")
  runs = int(runs)
  support = np.flatnonzero(share > _FLOOR)
  if runs < support.size:
    raise ValueError(f"{runs} runs cannot cover {support.size} support points; "
                     "drop candidates or raise the budget")

  best = None
  # Pukelsheim-Rieder sweeps the multiplier: each choice gives a different integer design and
  # the rule is to keep the one whose smallest n_i/(runs w_i) is largest.
  for multiplier in range(support.size, 2 * runs + 1):
    counts = np.zeros(share.size, dtype=int)
    raw = multiplier * share[support]
    counts[support] = np.maximum(np.ceil(raw).astype(int), 1)
    excess = int(counts.sum()) - runs
    while excess != 0:
      if excess > 0:
        movable = support[counts[support] > 1]
        if movable.size == 0: break
        counts[movable[np.argmax(counts[movable] / (runs * share[movable]))]] -= 1
        excess -= 1
      else:
        counts[support[np.argmin(counts[support] / (runs * share[support]))]] += 1
        excess += 1
    if int(counts.sum()) != runs: continue
    score = float(np.min(counts[support] / (runs * share[support])))
    if best is None or score > best[0]: best = (score, counts)
  if best is None: raise ValueError(f"no admissible allocation of {runs} runs was found")
  counts = best[1]

  efficiency = float("nan")
  if matrices is not None:
    stack = _checked(matrices)
    if stack.shape[0] != share.size: raise ValueError("matrices and weights disagree in length")
    # ALWAYS the D-efficiency of the information, never the compound criterion. A ratio of
    # compound values is not an efficiency: `log det` and a utility in nats have different
    # units and either sign, so the ratio can exceed one for a design that is worse -- it did,
    # reporting 1.017 for a rounded compound design. `log det` differences exponentiate to a
    # determinant ratio, and the p-th root makes it per parameter, which is a real efficiency.
    # This measures the ESTIMATION half only; `identification_front` reports the other half in
    # nats beside it, because the two do not share a scale.
    exact = _value(stack, counts / runs, None, 0.0)
    ideal = _value(stack, share, None, 0.0)
    dimension = stack.shape[1]
    if np.isfinite(exact) and np.isfinite(ideal):
      efficiency = float(np.exp((exact - ideal) / dimension))
  return ExactDesign(counts=counts, support=support, efficiency=efficiency, runs=runs)


@dataclass(frozen=True, slots=True)
class BatchDesign:
  """One integer batch, scored on both of the things a batch is for."""

  blend: float
  counts: np.ndarray
  support: np.ndarray
  runs: int
  log_det: float
  discrimination: float
  efficiency: float
  gap: float
  certified: bool

  @property
  def settings(self) -> int:
    """Distinct settings the batch visits. One is a warning, not an achievement."""
    return int(np.count_nonzero(self.counts))

  @property
  def table(self):
    order = np.argsort(-self.counts)
    return [(int(i), int(self.counts[i])) for i in order if self.counts[i] > 0]


def identification_front(matrices, utility, runs, *, blends=None, iterations=100_000,
                         tolerance=1e-9):
  """Integer batches that trade parameter precision against telling the models apart.

  A batch is what an experimentalist actually runs, and it has to serve two purposes that
  pull in different directions: pinning the material, and deciding which model produced the
  data. `optimal_measure` already accepts both -- `log det M(xi)` for the first and a
  per-candidate utility for the second -- and their blend is concave for every weighting, so
  the equivalence-theorem certificate holds all along the front. This walks that front and
  hands back run counts rather than weights.

  `utility[i]` is the expected log Bayes factor from ONE run at candidate `i`, in nats. Log
  evidence is additive over independent experiments, so the batch total is `counts . utility`
  and it is reported directly: the question a collaborator asks is whether `runs` experiments
  will settle the model question, and that is a number of nats, not a rate.

  WHAT THE ENDS OF THE FRONT MEAN. At `blend = 0` this is D-optimality and ignores the model
  question entirely. At `blend = 1` the criterion is linear in the measure, so its optimum is
  a single candidate repeated `runs` times -- the best discriminating setting, run over and
  over. That is usually a bad experiment for a reason no criterion here can see: a batch on
  one setting has no replication across settings, so nothing in it can detect that BOTH models
  are wrong. `BatchDesign.settings` is reported so that collapse is visible rather than
  implied, and `pyimr.noise.lack_of_fit` is what those extra settings would have bought.

  Both scores are computed on the INTEGER batch, not on the measure it came from, because
  rounding is where small budgets lose their efficiency and the point of this is what actually
  gets run.
  """
  stack = _checked(matrices)
  vector = np.asarray(utility, dtype=float)
  if vector.ndim != 1 or vector.size != stack.shape[0]:
    raise ValueError(f"utility must be one value per candidate; got {vector.shape} for {stack.shape[0]}")
  if not np.all(np.isfinite(vector)): raise ValueError("utility must be finite")
  if not isinstance(runs, Integral) or int(runs) < 1: raise ValueError("runs must be a positive integer")
  weights = (0.0, 0.25, 0.5, 0.75, 1.0) if blends is None else tuple(float(b) for b in blends)
  if any(not 0.0 <= b <= 1.0 for b in weights): raise ValueError("blends must lie in [0, 1]")

  def summed_log_det(counts):
    sign, value = np.linalg.slogdet(np.tensordot(counts.astype(float), stack, axes=(0, 0)))
    return float(value) if sign > 0 else float("-inf")

  # The reference for `efficiency` is the best PARAMETER precision `runs` runs can buy, not
  # the compound measure each batch came from. Against the latter the number exceeds one
  # whenever rounding happens to move back toward D-optimality -- it read 1.114 -- which is
  # arithmetically fine and useless to read. Against a fixed D-optimal batch of the same size
  # it answers the question actually being asked: what does chasing the model question cost me
  # in parameter precision?
  best = optimal_measure(stack, iterations=iterations, tolerance=tolerance)
  reference = summed_log_det(apportion(best.weights, int(runs), stack).counts)

  front = []
  for blend in weights:
    measure = optimal_measure(stack, utility=vector, blend=blend, iterations=iterations,
                              tolerance=tolerance)
    counts = apportion(measure.weights, int(runs), stack, utility=vector, blend=blend).counts
    value = summed_log_det(counts)
    efficiency = (0.0 if not np.isfinite(value)
                  else float(np.exp((value - reference) / stack.shape[1])))
    front.append(BatchDesign(
      blend=float(blend), counts=counts, support=np.flatnonzero(counts), runs=int(runs),
      log_det=value, discrimination=float(counts @ vector), efficiency=efficiency,
      gap=measure.gap, certified=bool(measure.certified)))
  return front
