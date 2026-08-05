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

__all__ = ["MeasureResult", "optimal_measure", "sensitivity"]

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
