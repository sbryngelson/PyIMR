"""Design measures: a convex formulation that can certify its own answer.

Every design search elsewhere in this package optimises over a single design point, and that
problem is not convex -- a search can stop anywhere and nothing in the result says which.
Optimising instead over a probability measure `xi` on the candidates makes the information
matrix an average, `M(xi) = sum_i xi_i M(x_i)`, which is LINEAR in `xi`; composing it with a
concave criterion makes the problem concave over the simplex. Its first-order condition is
then necessary and sufficient -- the General Equivalence Theorem of Kiefer and Wolfowitz,

    phi(x, xi*) <= 0   for every candidate x,   with equality on the support of xi*,

so `optimal_measure` returns a proof of global optimality rather than a report that a search
stopped improving. A measure is also what an experimenter runs anyway: `n_1` bubbles at one
setting and `n_2` at another.

A discrimination utility that averages over the design is linear in `xi` too, hence concave,
so it blends into the same objective and stays certified. See `docs/writeup/selection.tex`
for the measurements behind these choices.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._validate import finite_array, positive_integer, unit_interval

__all__ = ["BatchDesign", "ConstrainedMeasure", "ExactDesign", "MeasureResult", "apportion", "augmented_information", "constrained_measure", "identification_front", "optimal_measure", "sensitivity", "separability"]

_FLOOR = 1e-12

def _checked_matrices(matrices):
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

  For `log det` the rate is `tr[M(xi)^-1 M(x)] - p`; for a linear utility it is
  `u_x - u . xi`. The maximum over candidates is the equivalence-theorem gap, zero at an
  optimum because a concave function on a simplex is maximised where no vertex ascends.
  """
  stack = _checked_matrices(matrices)
  share = finite_array("weights", weights, shape=(stack.shape[0],))
  if np.any(share < -1e-12) or not np.isclose(share.sum(), 1.0): raise ValueError("weights must be non-negative and sum to one")
  vector = None if utility is None else finite_array("utility", utility, shape=(stack.shape[0],))

  try:
    return _sensitivity(stack, share, vector, float(blend))
  except np.linalg.LinAlgError as error:
    raise ValueError("the averaged information matrix is singular; spread the starting weights") from error

_UNDERFLOW = 1e-30      # negligible in the information matrix, recoverable by the update

def _sensitivity(stack, share, vector, blend, criterion=None):
  """The derivative without revalidation: the iteration calls this once per step."""
  if criterion is not None: return _growth(stack, share, criterion)[1]
  if blend >= 1.0: return vector - float(share @ vector)
  direction = np.einsum("jk,ikj->i", np.linalg.inv(_averaged(stack, share)), stack) - stack.shape[1]
  if vector is None: return direction
  return (1.0 - blend) * direction + blend * (vector - float(share @ vector))

def _growth(stack, share, criterion):
  """`(growth, sensitivity)` for a general criterion, from its gradient `G = dPhi/dM`.

  The directional derivative toward candidate `i` is `tr[G M_i] - tr[G M(xi)]`, and the
  subtracted term is the same for every candidate -- so `tr[G M_i]` is the multiplicative
  update's growth factor, non-negative because `G` and `M_i` are both positive semidefinite.
  """
  _, gradient = criterion(_averaged(stack, share))
  growth = np.einsum("jk,ikj->i", gradient, stack)
  return growth, growth - float(share @ growth)

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

def optimal_measure(matrices, *, criterion=None, utility=None, blend=0.0, iterations=100_000, tolerance=1e-9, prune=1e-7):
  """Maximise a concave design criterion over measures on the candidates.

  `matrices[i]` is the information one run at candidate `i` provides. `utility[i]`, if given,
  is a per-candidate quantity that averages over the design -- an expected log Bayes factor,
  say -- and `blend` in `[0, 1]` sets its share of the objective, giving the compound criteria
  of Atkinson and of Cook and Wong. Both parts are concave, so the certificate holds at any
  `blend`.

  The search is the multiplicative algorithm of Silvey, Titterington and Torsney:
  `w_i <- w_i F_i / sum_j w_j F_j` with `F_i` the derivative toward candidate `i`. Every
  update stays on the simplex and increases a concave criterion.

  `criterion` replaces `log det` with any concave function of the averaged information,
  supplied as `M -> (value, dvalue/dM)`; `pyimr.gain.gain_criterion` builds the one that asks
  what an experiment would change your mind about. Concavity is the caller's to guarantee and
  is what the certificate rests on -- the equivalence theorem holds for any concave criterion,
  and the reported gap is a proof only if it does.

  A general criterion also gets a different search. The multiplicative update relies on
  `tr[G M(xi)]` being the constant `p`, which is true for `log det` and false otherwise, and
  measured on the operator design it stalls at a gap of `0.16` -- correctly reported as
  uncertified, but not an answer. Mirror descent on the simplex converges there instead, to a
  gap of `0` and a slightly higher value.

  Convergence slows as candidates crowd together -- a design just off the support is nearly
  as good as one on it, so its weight decays in proportion, and certifying a parabola on
  `[-1, 1]` took 6400 iterations over 51 candidates but 329000 over 401. The criterion value
  is correct to six figures long before then, so an uncertified result is a converged value
  without a proof, not a wrong answer. `MeasureResult.certified` is how to tell.
  """
  stack = _checked_matrices(matrices)
  count = stack.shape[0]
  blend = unit_interval("blend", blend)
  if utility is None and blend > 0.0: raise ValueError("blend is positive but no utility was given")
  if criterion is not None and (utility is not None or blend > 0.0):
    raise ValueError("criterion replaces the utility/blend compound; pass one or the other")
  iterations = positive_integer("iterations", iterations)

  share = np.full(count, 1.0 / count)
  # A general criterion is not required to be finite only where the information is invertible
  # -- that is the point of the gain criteria, which score a rank-deficient design rather than
  # refusing it -- so this premise belongs to `log det` alone.
  if criterion is None and blend < 1.0 and not np.isfinite(_value(stack, share, None, 0.0)):
    raise ValueError("the candidates cannot jointly determine every parameter: their averaged information is singular")

  vector = None if utility is None else finite_array("utility", utility, shape=(count,))
  # The multiplicative update needs a positive derivative. Shifting the utility to be
  # positive adds a constant to the criterion, so the maximiser is untouched.
  shifted = None if vector is None else vector - vector.min() + 1.0

  gap, step = np.inf, 0
  for step in range(1, iterations + 1):
    slope = _sensitivity(stack, share, vector, blend, criterion)
    gap = float(np.max(slope))
    if gap <= tolerance: break
    if criterion is not None:
      # mirror descent: multiplicative in form, but stepping along the derivative rather than
      # rescaling by it, which is what removes the `tr[G M] = p` assumption. The step decays
      # so late iterations refine rather than oscillate, and the gradient scale is divided out
      # so the same schedule works whatever units the criterion is in.
      derivative = _growth(stack, share, criterion)[1]
      pace = 2.0 / (1.0 + step / 2000.0) / max(float(np.abs(derivative).max()), 1e-300)
      growth = np.exp(np.clip(pace * derivative, -500.0, 500.0))
    elif blend >= 1.0:
      growth = shifted
    else:
      variance = np.einsum("jk,ikj->i", np.linalg.inv(_averaged(stack, share)), stack)
      growth = variance if shifted is None else (1.0 - blend) * variance + blend * shifted
    updated = share * growth
    total = updated.sum()
    if not np.isfinite(total) or total <= 0.0: break
    # Zero is ABSORBING for a multiplicative update, and a weight falling by up to e^-2 a step
    # reaches it in a few hundred. That turns a transient dip into a permanent exclusion: a
    # candidate the optimum needs is annihilated by arithmetic and the search ends short, with
    # a gap it cannot close. Holding live weights at a level that is negligible in `M` but
    # still reachable by the update keeps the floor a rounding decision, not a modelling one.
    share = np.maximum(updated / total, _UNDERFLOW)
    # A candidate just off the support decays like (d_i/p)^k with d_i/p barely below one, so
    # the tail costs more iterations than the answer does. Dropping it periodically costs
    # nothing and the certificate is still computed against every candidate afterwards, so a
    # wrong drop cannot hide -- but only a candidate the criterion is not currently asking for
    # may go. A positive sensitivity means this step wants MORE mass there, and demoting it
    # anyway is self-defeating twice over: the update cannot climb 23 decades back above the
    # threshold before the next prune re-demotes it, so an early transient dip becomes
    # permanent exile, and the search ends short with a gap it has forbidden itself to close.
    if step % 200 == 0:
      surviving = (share > prune) | (slope > 0.0)
      if surviving.any() and not surviving.all():
        share = np.where(surviving, share, _UNDERFLOW)
        share = share / share.sum()

  keep = share > max(prune, _UNDERFLOW)
  if not keep.any(): keep = share >= share.max()
  cleaned = np.where(keep, share, 0.0)
  cleaned = cleaned / cleaned.sum()
  # pruning perturbs the measure, so the reported gap must be the one the answer actually has
  final = float(np.max(_sensitivity(stack, cleaned, vector, blend, criterion)))
  value = criterion(_averaged(stack, cleaned))[0] if criterion is not None else _value(stack, cleaned, utility, blend)
  return MeasureResult(weights=cleaned, support=np.flatnonzero(keep), value=value,
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


def apportion(weights, runs, matrices=None, *, criterion=None, utility=None, blend=0.0, replicates=1):
  """Turn a design measure into a whole number of runs per candidate.

  The certificate says nothing about rounding -- it certifies optimality over MEASURES, and
  an exact design is not one. This is the efficient rounding of Pukelsheim and Rieder (1992),
  which maximises the smallest ratio `n_i / (runs w_i)` and so keeps a support point from
  being rounded away. Rounding is known to be safe for large `runs` and to carry no guarantee
  for small ones, which is the regime IMR reaches: pass `matrices` and the D-efficiency
  `(det M(exact) / det M(xi))^(1/p)` is MEASURED rather than assumed. Read it before
  believing a design built from a handful of runs.

  That efficiency is always the D-efficiency of the INFORMATION, even when the measure came
  from a `utility` and `blend`. A ratio of compound criterion values is not an efficiency:
  `log det` and a utility in nats have different units and either sign, so the ratio can
  exceed one for a design that is worse -- it read 1.017. `identification_front` reports the
  utility achieved separately, in nats, because the two halves do not share a scale.

  `criterion` must be the one the measure was optimised under, if it was not `log det`. The
  efficiency is then measured in that criterion's own currency -- `exp` of the nats lost to
  rounding -- because a D-efficiency computed against a measure that was maximising something
  else is not an efficiency at all: on the operator design it read `1.21` and `1.40`, which is
  arithmetic about two different objectives and not a loss.

  `replicates` forces every setting that gets used to get at least that many runs, which is
  what `pyimr.noise.lack_of_fit` needs to have any pure error to compare against. It is a
  constraint, not a criterion: no design criterion here can see model inadequacy, so the batch
  has to be told to stay testable, and the D-efficiency reports what that cost. When the budget
  cannot give `replicates` runs to every support point, the lowest-weight points are dropped
  first -- the measure's own ordering, not a new decision.
  """
  share = finite_array("weights", weights, non_negative=True, shape=("candidates",))
  if share.sum() <= 0.0: raise ValueError("weights must not be all zero")
  share = share / share.sum()
  runs = positive_integer("runs", runs)
  replicates = positive_integer("replicates", replicates)
  support = np.flatnonzero(share > _FLOOR)
  if runs < support.size:
    raise ValueError(f"{runs} runs cannot cover {support.size} support points; "
                     "drop candidates or raise the budget")
  if replicates > 1:
    keepable = runs // replicates
    if keepable < 1:
      raise ValueError(f"{runs} runs cannot give {replicates} replicates to even one setting")
    if keepable < support.size:
      support = support[np.argsort(-share[support])[:keepable]]
      share = np.where(np.isin(np.arange(share.size), support), share, 0.0)
      share = share / share.sum()

  # how far each candidate sits above (>1) or below (<1) the share it was promised: the
  # apportionment rule moves a run from the largest of these to the smallest
  def ratio(counts, where): return counts[where] / (runs * share[where])

  best = None
  # Pukelsheim-Rieder sweeps the multiplier: each choice gives a different integer design and
  # the rule keeps the one whose smallest ratio is largest.
  for multiplier in range(support.size, 2 * runs + 1):
    counts = np.zeros(share.size, dtype=int)
    counts[support] = np.maximum(np.ceil(multiplier * share[support]).astype(int), replicates)
    excess = int(counts.sum()) - runs
    while excess != 0:
      if excess > 0:
        movable = support[counts[support] > replicates]
        if movable.size == 0: break
        counts[movable[np.argmax(ratio(counts, movable))]] -= 1
        excess -= 1
      else:
        counts[support[np.argmin(ratio(counts, support))]] += 1
        excess += 1
    if int(counts.sum()) != runs: continue
    score = float(np.min(ratio(counts, support)))
    if best is None or score > best[0]: best = (score, counts)
  if best is None: raise ValueError(f"no admissible allocation of {runs} runs was found")
  counts = best[1]

  efficiency = float("nan")
  if matrices is not None:
    stack = _checked_matrices(matrices)
    if stack.shape[0] != share.size: raise ValueError("matrices and weights disagree in length")
    if criterion is not None:
      exact, ideal = criterion(_averaged(stack, counts / runs))[0], criterion(_averaged(stack, share))[0]
      if np.isfinite(exact) and np.isfinite(ideal): efficiency = float(np.exp(exact - ideal))
    else:
      # `log det` differences exponentiate to a determinant ratio and the p-th root makes it
      # per parameter. This is the ESTIMATION half only, whatever `utility` and `blend` were.
      exact = _value(stack, counts / runs, None, 0.0)
      ideal = _value(stack, share, None, 0.0)
      if np.isfinite(exact) and np.isfinite(ideal):
        efficiency = float(np.exp((exact - ideal) / stack.shape[1]))
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

  `utility[i]` is the expected log Bayes factor from ONE run at candidate `i`, in nats. Log
  evidence is additive over independent experiments, so the batch total is `counts . utility`
  and is reported as nats rather than as a rate: the question a collaborator asks is whether
  `runs` experiments settle the model question.

  At `blend = 0` this is D-optimality and ignores the model question. At `blend = 1` the
  criterion is linear, so its optimum is one candidate repeated `runs` times -- usually a bad
  experiment for a reason no criterion here can see, since a batch on one setting cannot
  detect that BOTH models are wrong. `BatchDesign.settings` makes that collapse visible, and
  `pyimr.noise.lack_of_fit` is what the extra settings would have bought.

  Both scores are computed on the INTEGER batch, since rounding is where small budgets lose
  their efficiency and what gets run is the point.
  """
  stack = _checked_matrices(matrices)
  vector = finite_array("utility (one value per candidate)", utility, shape=(stack.shape[0],))
  runs = positive_integer("runs", runs)
  weights = ((0.0, 0.25, 0.5, 0.75, 1.0) if blends is None
             else tuple(unit_interval("blends", b) for b in blends))

  def summed_log_det(counts):
    sign, value = np.linalg.slogdet(np.tensordot(counts.astype(float), stack, axes=(0, 0)))
    return float(value) if sign > 0 else float("-inf")

  # The reference is the best PARAMETER precision `runs` runs can buy, not the compound
  # measure each batch came from -- against the latter the number exceeds one whenever
  # rounding moves back toward D-optimality (it read 1.114). Against a fixed D-optimal batch
  # of the same size it answers what chasing the model question costs in precision.
  best = optimal_measure(stack, iterations=iterations, tolerance=tolerance)
  reference = summed_log_det(apportion(best.weights, runs, stack).counts)

  front = []
  for blend in weights:
    measure = optimal_measure(stack, utility=vector, blend=blend, iterations=iterations,
                              tolerance=tolerance)
    counts = apportion(measure.weights, runs, stack, utility=vector, blend=blend).counts
    value = summed_log_det(counts)
    efficiency = (0.0 if not np.isfinite(value)
                  else float(np.exp((value - reference) / stack.shape[1])))
    front.append(BatchDesign(
      blend=float(blend), counts=counts, support=np.flatnonzero(counts), runs=runs,
      log_det=value, discrimination=float(counts @ vector), efficiency=efficiency,
      gap=measure.gap, certified=bool(measure.certified)))
  return front


def _augmented_parts(jacobian, differences, weights):
  """Validated `(material, columns, amplitude)`, with `amplitude` one entry per rival.

  Amplitude, not variance: a half-weight rival contributes half the sensitivity, which is
  what makes the augmented determinant read as a posterior average rather than as a mixture
  of unrelated units. The columns are returned UNWEIGHTED so that quantities which are
  properties of the difference direction alone stay independent of the weighting.
  """
  material = finite_array("jacobian", jacobian, shape=("samples", "p"))
  columns = np.atleast_2d(finite_array("differences", differences))
  if columns.shape[1] != material.shape[0]:
    raise ValueError(f"differences must have {material.shape[0]} samples; got {columns.shape}")
  if weights is None: return material, columns, np.ones(columns.shape[0])
  share = finite_array("weights (one per rival)", np.ravel(weights), non_negative=True,
                       shape=(columns.shape[0],))
  return material, columns, np.sqrt(share)


def _augmented_gram(material, columns, amplitude):
  full = np.hstack([material, (columns * amplitude[:, None]).T])
  return full.T @ full


def augmented_information(jacobian, differences, *, weights=None):
  """Information about the material AND about which model produced the trace, in one matrix.

  Writing a model choice as a continuous coordinate, `R(theta, eps) = (1 - eps) R_A + eps R_B`,
  gives `dR/deps = R_B - R_A` at `eps = 0`. The model label is then one more column of the
  Jacobian: `J^T J` over the augmented set carries both questions, is linear in the design
  measure, and composes with `optimal_measure` and its certificate exactly as the
  material-only matrix does. Only the difference direction enters, and it is exact -- no
  prior per rival, and no evidence integral to collapse.

  It is local: `R_B - R_A` measures separability in a neighbourhood, which is the right
  question for close rivals and the wrong one for distant ones. Distant rivals are easy to
  separate and should have been dropped by `pyimr.discriminate.screen_models` first, leaving
  the close pairs this criterion is for.

  `jacobian` is the whitened material sensitivity, `(samples, p)`; `differences` is
  `(samples,)` or `(rivals, samples)`, whitened the same way; `weights` are the rivals'
  posterior probabilities, so a rival the data has settled contributes nothing.
  """
  return _augmented_gram(*_augmented_parts(jacobian, differences, weights))


def separability(jacobian, differences, *, weights=None):
  """Post-material variance of each model coordinate, and the share the material absorbs.

  What matters is not how far apart two models are but how much of that difference survives
  refitting the material: the part lying in the span of `dR/dtheta` is reproduced by adjusting
  the parameters and is invisible at every design. Smaller variance is a better-determined
  model choice; `inf` means the coordinate is not identified at all, and no amount of data at
  this design will separate the models.
  """
  material, columns, amplitude = _augmented_parts(jacobian, differences, weights)
  count, rivals = material.shape[1], columns.shape[0]
  try:
    variance = np.diag(np.linalg.inv(_augmented_gram(material, columns, amplitude)))[count:].copy()
  except np.linalg.LinAlgError:
    variance = np.full(rivals, np.inf)
  variance = np.where(variance > 0.0, variance, np.inf)

  basis, _ = np.linalg.qr(material)
  absorbed = np.empty(rivals)
  for index, row in enumerate(columns):
    size = np.linalg.norm(row)
    left = np.linalg.norm(row - basis @ (basis.T @ row))
    absorbed[index] = 1.0 - left / size if size > 0.0 else 1.0
  return variance, absorbed


@dataclass(frozen=True, slots=True)
class ConstrainedMeasure:
  """A measure guaranteed to spread over enough settings to be testable, and what that cost."""

  measure: MeasureResult
  settings: int
  nats_lost: float
  certified_under_floor: bool
  summary: str

  def __str__(self) -> str: return self.summary


def constrained_measure(matrices, settings, *, floor=None, criterion=None, iterations=100_000,
                        tolerance=1e-9, prune=1e-7):
  """`optimal_measure`, forced to spend real weight on at least `settings` distinct candidates.

  An information criterion is happiest concentrating, and a batch on too few settings cannot
  detect that every model is wrong: `pyimr.noise.lack_of_fit` needs pure error, which needs
  repeats, and a lack-of-fit numerator, which needs more distinct settings than parameters.
  `apportion(..., replicates=J)` cannot supply this -- the number of settings is a property of
  the MEASURE, and apportionment only allocates runs within the one it is handed (#232).

  Bounding the support size is a cardinality constraint, which is not convex, so it cannot join
  the criterion without forfeiting the equivalence-theorem certificate that makes
  `optimal_measure` worth using. Two steps buy back most of it. First choose WHICH settings to
  guarantee: those the free solution already funds, then, if that is too few, the candidates of
  largest sensitivity -- the ones the free problem came closest to wanting, nominated by the
  certificate itself rather than by taste. Then guarantee them with a floor.

  Naming the settings is not enough on its own: an optimiser merely restricted to them
  reproduces the free optimum by leaving the new ones at zero weight, and a setting that gets
  no runs is not a setting. The floor is imposed by the substitution
  `xi = floor * 1_chosen + (1 - k*floor) * eta`, which maps the constrained set onto a whole
  simplex in `eta` -- so the search is exactly the one `optimal_measure` already certifies, the
  floor holds by construction rather than by projection, and every candidate, guaranteed or
  not, still competes for the remaining mass. `floor` defaults to half an equal share, which is
  enough for `apportion` to give each guaranteed setting a run at any realistic budget. Note
  that this default makes the demands non-nested -- asking for more settings also asks less of
  each -- so `nats_lost` need not rise with `settings`. Pass a fixed `floor` to compare.

  The problem in `eta` is concave, so the result is certified optimal AMONG measures meeting
  the floor -- weaker than global optimality, and reported as `certified_under_floor` rather
  than as `certified`. `nats_lost` is the price, in the criterion's own units, of insisting on
  a batch that can be tested.
  """
  stack = _checked_matrices(matrices)
  settings = positive_integer("settings", settings)
  if settings > stack.shape[0]:
    raise ValueError(f"{settings} settings asked of {stack.shape[0]} candidates")
  share = 0.5 / settings if floor is None else unit_interval("floor", floor)
  if share * settings >= 1.0:
    raise ValueError(f"a floor of {share:g} on each of {settings} settings exceeds all the mass")

  free = optimal_measure(stack, criterion=criterion, iterations=iterations,
                         tolerance=tolerance, prune=prune)
  ranked = sorted(free.support, key=lambda i: -free.weights[i])[:settings]
  if len(ranked) < settings:
    slopes = _sensitivity(stack, free.weights, None, 0.0, criterion)
    ranked += [int(i) for i in np.argsort(-slopes) if int(i) not in ranked][:settings - len(ranked)]
  chosen = np.array(sorted(ranked), dtype=int)

  base = share * np.sum(stack[chosen], axis=0)     # the information the floor buys up front
  spare = 1.0 - share * settings

  def shifted(information):
    """The criterion seen through the substitution, so `eta` roams a full simplex."""
    total = base + spare * information
    if criterion is not None:
      value, gradient = criterion(total)
      return value, spare * gradient
    sign, magnitude = np.linalg.slogdet(total)
    return (magnitude if sign > 0 else -np.inf), spare * np.linalg.inv(total)

  held = optimal_measure(stack, criterion=shifted, iterations=iterations,
                         tolerance=tolerance, prune=prune)
  weights = spare * held.weights
  weights[chosen] += share
  measure = MeasureResult(weights=weights, support=np.flatnonzero(weights > 0.0),
                          value=held.value, gap=held.gap, iterations=held.iterations)
  lost = float(free.value - held.value)
  return ConstrainedMeasure(
    measure=measure, settings=int(measure.support.size), nats_lost=lost,
    certified_under_floor=bool(held.certified),
    summary=(f"{measure.support.size} settings, each at least {share:.3g} of the batch; "
             f"{lost:.3f} of the criterion given up to get there, "
             + (f"certified under the floor at gap {held.gap:.1e}" if held.certified else
                f"NOT certified -- the search stopped at gap {held.gap:.1e}")),
  )
