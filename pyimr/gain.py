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

from ._validate import finite_array, positive_integer, positive_scalar

__all__ = ["Question", "QuestionGain", "expected_gain", "gain_criterion", "lack_of_fit_degrees"]


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
