"""What an experiment would change your mind about, in nats.

Every design criterion in this package scores a different question --- `log det` for the
material, a Schur complement for the model label, a separate number for the initial radius ---
and none of them share a scale, so there is no way to ask which question a batch should serve.
They are all the same quantity. The value of an experiment is the mutual information between
the data and whatever you want to learn,

    U_A(xi) = E_y[ KL( p(A | y, xi) || p(A) ) ] = I(A ; Y | xi),

and for a Gaussian likelihood in prior-standardised coordinates, with `N = I + M(xi)` and `B`
the coordinates you do NOT care about,

    U_A(xi) = (1/2) log det [ N_AA - N_AB N_BB^-1 N_BA ] ,

the Schur complement of the nuisance block. `A` is any subset: the material parameters
recovers Bayesian D-optimality, the model-mixing coordinates of `pyimr.measure` recovers
D_s-optimality, `req_scale` alone asks whether the initial radius is driving the conclusion.
Because the answers are all in nats they can be compared and, with weights, added.

Three properties make this usable where `log det M` was not. It is concave in the design
measure for every `A`, so `optimal_measure` still certifies its answer. It is finite when `M`
is singular --- a design that cannot determine every parameter scores low rather than `-inf`,
which is what put 22 of 48 candidates out of reach of the sampled criterion. And the identity
`log det M = log det M_thth + log det S` shows that maximising `log det M` was never neutral:
it weights material volume and discrimination volume equally, which is a choice nobody made.

WHAT IT CANNOT REACH. Every question here is a COORDINATE. "Is every model in the catalogue
wrong?" is not one, and on these records the answer is yes (`pyimr.noise.lack_of_fit`). No
Jacobian column reaches model inadequacy, and the criterion above, maximised freely, walks
away from the batch that could detect it: a criterion linear in the measure is optimised by
one setting repeated. Testability is a CONSTRAINT on the batch, not a term in the objective ---
`replicates` in `pyimr.measure.apportion`, and `lack_of_fit_degrees` here.

`replicates` is only half of that constraint and the weaker half. It sets how many runs each
chosen setting gets; the lack-of-fit numerator needs more distinct SETTINGS than parameters,
and how many settings there are is a property of the measure, not of the rounding. Measured on
the operator design: the free batch takes two settings, and forcing three replicates each
still leaves `lack_df = -2`. Constraining the number of settings is a cardinality constraint
on the support and is not convex, so it cannot join the criterion without forfeiting the
certificate. `lack_of_fit_degrees` reports the shortfall rather than hiding it (#232).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scipy.stats import norm

from ._validate import finite_array, positive_integer, positive_scalar, unit_interval

__all__ = ["Question", "QuestionGain", "expected_gain", "gain_criterion", "lack_of_fit_degrees",
           "runs_to_precision", "runs_to_settle"]


@dataclass(frozen=True, slots=True)
class Question:
  """Coordinates you want to learn about, and how much that matters relative to the rest."""

  name: str
  coordinates: tuple[int, ...]
  weight: float = 1.0

  def __post_init__(self):
    if not self.coordinates: raise ValueError(f"question {self.name!r} names no coordinates")
    if len(set(self.coordinates)) != len(self.coordinates):
      raise ValueError(f"question {self.name!r} repeats a coordinate")
    positive_scalar(f"weight of {self.name!r}", self.weight, allow_zero=True)


@dataclass(frozen=True, slots=True)
class QuestionGain:
  """Expected information gain in nats. `per_question` is unweighted; `total` is not."""

  total: float
  per_question: dict[str, float]

  def __str__(self) -> str:
    parts = ", ".join(f"{k} {v:.2f}" for k, v in sorted(self.per_question.items()))
    return f"{self.total:.2f} nats total ({parts})"


def _checked_questions(questions, size):
  if not questions: raise ValueError("at least one question is required")
  for q in questions:
    if not isinstance(q, Question): raise TypeError("questions must be Question instances")
    if any(c < 0 or c >= size for c in q.coordinates):
      raise ValueError(f"question {q.name!r} names a coordinate outside the {size} available")
  return tuple(questions)


def _one_question(information, coordinates):
  """`(nats, dU/dM)` for one question. `dU/dM = P S^-1 P^T / 2` with `S = P^T (I+M) P`."""
  size = information.shape[0]
  want = list(coordinates)
  rest = [i for i in range(size) if i not in set(want)]
  matrix = np.eye(size) + information

  projector = np.zeros((size, len(want)))
  projector[want, :] = np.eye(len(want))
  if rest:
    # the envelope: u'Su = min_v [v;u]'N[v;u], attained at v = -N_BB^-1 N_BA u, which is what
    # makes S concave in M and lets the derivative ignore how the projector itself moves
    projector[rest, :] = -np.linalg.solve(matrix[np.ix_(rest, rest)], matrix[np.ix_(rest, want)])
  schur = projector.T @ matrix @ projector

  sign, magnitude = np.linalg.slogdet(schur)
  value = 0.5 * float(magnitude) if sign > 0 else -np.inf
  return value, 0.5 * projector @ np.linalg.solve(schur, projector.T)


def expected_gain(information, questions):
  """Nats of expected information gain: per question unweighted, and the weighted total.

  `information` is `M` in prior-standardised coordinates -- the whitened `J^T J` with the
  parameters scaled so the prior is unit on each axis, which is what makes `I + M` the
  posterior precision and the answer a number of nats rather than an arbitrary volume.
  """
  matrix = finite_array("information", information, shape=("d", "d"))
  bank = _checked_questions(questions, matrix.shape[0])
  per = {q.name: _one_question(matrix, q.coordinates)[0] for q in bank}
  total = float(sum(q.weight * per[q.name] for q in bank))
  return QuestionGain(total=total, per_question=per)


def gain_criterion(questions):
  """The portfolio criterion as `optimal_measure` wants it: `M -> (value, dvalue/dM)`.

  A weighted sum of concave functions is concave, so a portfolio of questions is certifiable
  on exactly the terms one question is. Weight zero drops a question without removing it,
  which is what `screen_models` produces for a rival the data has already decided.
  """
  bank = tuple(questions)
  if not bank: raise ValueError("at least one question is required")

  def criterion(information):
    _checked_questions(bank, information.shape[0])
    value, gradient = 0.0, np.zeros_like(information)
    for question in bank:
      nats, slope = _one_question(information, question.coordinates)
      value += question.weight * nats
      gradient += question.weight * slope
    return value, gradient

  return criterion


def lack_of_fit_degrees(counts, parameters):
  """`(settings, replicates, lack_df, pure_df)` a batch would leave for a lack-of-fit test.

  `pyimr.noise.lack_of_fit` needs pure error, which needs repeats, and a numerator, which
  needs more distinct settings than parameters. A batch can be information-optimal and leave
  neither: the criterion is linear in the measure at any single question, so its unconstrained
  optimum is one setting run `N` times. Non-positive `lack_df` means this batch cannot discover
  that every model in the catalogue is wrong, which on these records is the finding that
  already holds.
  """
  runs = finite_array("counts", counts, non_negative=True).astype(int)
  parameters = positive_integer("parameters", parameters, minimum=0)
  supported = runs[runs > 0]
  settings = int(supported.size)
  return (settings, int(supported.min()) if settings else 0,
          settings - parameters, int(supported.sum() - settings))


def _profiled(information, coordinate):
  """Information about one coordinate with every other one unknown and fitted out.

  The Schur complement, computed directly rather than as `1 / [M^-1]_kk`, because the design
  information alone -- no prior -- is often singular and the inverse does not exist. Zero means
  the coordinate is not identified at ANY budget: what it does to the trace, the other
  coordinates can reproduce.
  """
  rest = [i for i in range(information.shape[0]) if i != coordinate]
  own = float(information[coordinate, coordinate])
  if not rest: return own
  cross = information[rest, coordinate]
  block = information[np.ix_(rest, rest)]
  try:
    absorbed = float(cross @ np.linalg.solve(block, cross))
  except np.linalg.LinAlgError:
    absorbed = float(cross @ np.linalg.lstsq(block, cross, rcond=None)[0])
  return max(own - absorbed, 0.0)


def runs_to_settle(per_run, coordinate, *, threshold=5.0, confidence=0.95, budget=None):
  """How many runs before a model question DECIDES, and what `budget` runs would achieve.

  Nats measure how much a batch teaches. They do not say whether it settles anything, which
  is the question an experimentalist with a fixed budget is actually asking. It has a closed
  form. Write `s` for the per-run information about the coordinate that distinguishes the two
  models, after everything else is fitted out. Information adds over runs, so `N` runs give
  `S = N s`, and the log Bayes factor the experiment will return is not yet known but its
  distribution is:

      Lambda ~ Normal(S/2, S)   under whichever model is true,

  exact in the Gaussian-linear limit, not asymptotic. So the chance of clearing `threshold`
  nats in favour of the truth is `Phi((S/2 - threshold)/sqrt(S))`, and inverting for
  `confidence` gives

      N >= (z + sqrt(z^2 + 2 threshold))^2 / s ,   z = Phi^-1(confidence) .

  Returns `(runs, probability)`: the runs required, and the probability that `budget` runs
  decide it (`nan` when no budget is given). `inf` runs means no experiment of this shape ever
  settles it -- the difference lies in the span of the other coordinates and refitting
  reproduces it, which `pyimr.measure.separability` reports as infinite variance.

  TWO ASSUMPTIONS WORTH SAYING ALOUD. This presumes one of the two models is true; it is the
  probability of identifying WHICH, not of either being right. On these records
  `pyimr.noise.lack_of_fit` rejects both, so a small `runs` here answers "which of two rejected
  models fits better" and that may not be the question worth buying. And `s` must come from a
  likelihood that counts independent observations: with the correlated residuals these records
  carry, an independent likelihood overstates `s` by `N/N_eff` and understates the runs
  required by the same factor.
  """
  matrix = finite_array("per_run", per_run, shape=("d", "d"))
  coordinate = positive_integer("coordinate", coordinate, minimum=0)
  if coordinate >= matrix.shape[0]:
    raise ValueError(f"coordinate {coordinate} is outside the {matrix.shape[0]} available")
  threshold = positive_scalar("threshold", threshold)
  confidence = unit_interval("confidence", confidence)
  if not 0.0 < confidence < 1.0: raise ValueError("confidence must lie strictly inside (0, 1)")

  each = _profiled(matrix, coordinate)
  z = float(norm.ppf(confidence))
  needed = (z + np.sqrt(z * z + 2.0 * threshold)) ** 2
  runs = float("inf") if each <= 0.0 else needed / each

  probability = float("nan")
  if budget is not None:
    total = positive_integer("budget", budget) * each
    probability = 0.0 if total <= 0.0 else float(norm.cdf((0.5 * total - threshold) / np.sqrt(total)))
  return runs, probability


def runs_to_precision(per_run, coordinate, target, *, budget=None):
  """How many runs before one coordinate is pinned to `target` standard deviations.

  The same question as `runs_to_settle` in the same unit, for a parameter of a model already
  agreed on: `N` runs leave posterior variance `[(I + N M)^-1]_kk` in prior-standardised
  coordinates, so `target = 0.1` means "a tenth of the prior width". That shared unit -- runs
  -- is what makes "settle which model" and "pin this parameter" comparable, which nats alone
  are not: they measure different things about different objects.

  Returns `(runs, achieved)`, the second being the standard deviation `budget` runs would
  reach. `inf` runs means the prior is never improved on in that direction.
  """
  matrix = finite_array("per_run", per_run, shape=("d", "d"))
  coordinate = positive_integer("coordinate", coordinate, minimum=0)
  if coordinate >= matrix.shape[0]:
    raise ValueError(f"coordinate {coordinate} is outside the {matrix.shape[0]} available")
  target = positive_scalar("target", target)
  if target >= 1.0: raise ValueError("target must be tighter than the unit prior it is measured against")

  size = matrix.shape[0]
  def deviation(runs):
    return float(np.sqrt(np.linalg.inv(np.eye(size) + runs * matrix)[coordinate, coordinate]))

  runs = float("inf")
  if deviation(1e12) <= target:                 # bisect: the deviation falls monotonically
    low, high = 0.0, 1e12
    for _ in range(200):
      middle = 0.5 * (low + high)
      if deviation(middle) <= target: high = middle
      else: low = middle
    runs = high
  return runs, (deviation(positive_integer("budget", budget)) if budget is not None else float("nan"))
