"""Model discrimination as an integral rather than a minimum.

T-optimality scores a design by `min` over the rival's parameters -- how well the rival can
imitate at its best. That inner problem is multimodal here, and a local method that lands in
the wrong basin returns a wrong answer with nothing to signal it: on this study Nelder-Mead
and Gauss-Newton converged to different basins, and every improvement to the optimiser
lowered the scores.

Replacing that `min` with an integral makes the model label one more parameter, so the
design is scored by the expected log Bayes factor in favour of the true model,

    U(d) = E_{Y ~ M1} [ log p(Y | M1, d) - log p(Y | M2, d) ],

the Kullback-Leibler divergence between the two prior-predictive distributions. A missed
mode biases an integral slightly; a missed basin corrupts a minimum entirely.

The costs: the prior is now part of the criterion and has to be stated, and a prior-sample
estimate of the evidence degenerates once the likelihood is sharp relative to the prior --
the sum collapses onto its largest term. That failure is silent unless measured, so every
result here carries the effective sample size that detects it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from ._validate import deviation_for, finite_array, positive_integer, positive_scalar

__all__ = ["DiscriminationEvaluation", "ModelScreen", "expected_log_bayes_factor", "laplace_log_evidence", "log_evidence", "screen_models"]

def _checked_bank(bank, label):
  table = np.asarray(bank, dtype=float)
  if table.ndim != 2: raise ValueError(f"{label} must be (draws, samples); got shape {table.shape}")
  if table.shape[0] < 2: raise ValueError(f"{label} needs at least two draws to integrate over")
  if not np.all(np.isfinite(table)): raise ValueError(f"{label} contains non-finite trajectories")
  return table

def log_evidence(observations, bank, deviation, *, exclude=None):
  """`log p(Y | M)` for each row of `observations`, and the effective sample size.

  The evidence is a prior integral, estimated by averaging the likelihood over `bank` --
  trajectories already simulated from `pi(theta | M)`, so every observation reuses them and
  no further solves are needed.

  `exclude` drops one bank row per observation, which is required when the observation was
  generated from that row: leaving it in includes the exact trajectory that produced the
  data, and reports discrimination between a model and itself.

  Returns `(log_evidence, effective_draws)`, the second being `(sum w)^2 / sum w^2` over the
  per-draw likelihood weights -- how many bank members genuinely contributed. Near one the
  integral has collapsed onto a single draw and the value is unreliable however small its
  apparent scatter.
  """
  table = _checked_bank(bank, "bank")
  seen = np.atleast_2d(np.asarray(observations, dtype=float))
  if seen.shape[1] != table.shape[1]: raise ValueError(f"observations have {seen.shape[1]} samples, bank has {table.shape[1]}")
  scale = positive_scalar("deviation", deviation)

  # (observations, draws) matrix of log-likelihoods
  gap = seen[:, None, :] - table[None, :, :]
  terms = -0.5 * np.sum(gap**2, axis=-1) / scale**2
  if exclude is not None:
    rows = np.asarray(exclude, dtype=int)
    if rows.shape != (seen.shape[0],): raise ValueError("exclude must give one bank row per observation")
    terms[np.arange(seen.shape[0]), rows] = -np.inf

  count = np.sum(np.isfinite(terms), axis=1)
  if np.any(count < 1): raise ValueError("every observation needs at least one usable bank draw")
  normalisation = -0.5 * table.shape[1] * np.log(2.0 * np.pi * scale**2)
  evidence = logsumexp(terms, axis=1) - np.log(count) + normalisation

  shifted = terms - terms.max(axis=1, keepdims=True)
  weights = np.exp(np.where(np.isfinite(shifted), shifted, -np.inf))
  effective = np.sum(weights, axis=1) ** 2 / np.sum(weights**2, axis=1)
  return evidence, effective

def laplace_log_evidence(residual, jacobian, deviation, *, cap_at_prior=False):
  """`log p(Y | M)` by Laplace expansion at a fitted point, in unit prior coordinates.

  Use this wherever `log_evidence` degenerates, which is whenever the likelihood is much
  sharper than the prior: at 201 samples and the measured noise its effective sample size
  came out at 1.0 for every design here. This costs one least-squares solve and one Jacobian
  and does not care how sharp the likelihood is.

  `residual` is the whitened misfit `(m - y)/sigma` at the fit, `jacobian` its derivative
  with respect to parameters scaled so the prior is uniform on the unit cube. Then the prior
  density is one, the Hessian of the log-likelihood is `J^T J`, and

      log Z ~= -||r||^2/2 - (N/2) log(2 pi sigma^2) + (p/2) log(2 pi) - (1/2) log det(J^T J).

  The last two terms are the Occam factor -- the posterior's share of the prior volume --
  and their sign is what stops a more flexible model from winning on fit alone.

  `cap_at_prior` bounds each eigendirection's Occam factor at 1. Turn it on whenever the
  models compared differ in DIMENSION: an eigenvalue near zero sends `-log det / 2` to
  `+infinity`, so the plain form rewards a parameter the data cannot see. It is off by
  default because the expansion above is the textbook one; where every direction is sharper
  than the prior the two agree to round-off.

  Laplace's assumptions apply -- one dominant, well-separated, roughly Gaussian mode. Where
  several modes matter the evidences add, so summing this over the modes a multistart finds
  is the robust form.
  """
  misfit = finite_array("residual", residual).ravel()
  matrix = np.atleast_2d(finite_array("jacobian", jacobian))
  if matrix.shape[0] != misfit.size: raise ValueError(f"jacobian has {matrix.shape[0]} rows for {misfit.size} residuals")
  scale = deviation_for(deviation, misfit.size)

  samples, parameters = matrix.shape
  normalization = float(np.sum(np.log(2.0 * np.pi * np.broadcast_to(scale, (samples,)) ** 2)))
  information = matrix.T @ matrix
  if cap_at_prior:
    # A posterior cannot be wider than the prior it came from, so each eigendirection's
    # factor `sqrt(2 pi / lambda)` caps at 1: a direction the data cannot see then costs
    # nothing and buys nothing. Uncapped it goes to infinity as `lambda -> 0` and REWARDS
    # the useless parameter -- on a one-mode record the two-mode model won by 29.7 nats
    # exactly where its second arm did nothing, and lost by 2.6 where the arm did work.
    eigenvalues = np.linalg.eigvalsh(information)
    if np.any(eigenvalues < 0.0): raise ValueError("J^T J is not positive semidefinite")
    occam = float(np.sum(np.minimum(0.0, 0.5 * np.log(2.0 * np.pi / np.maximum(eigenvalues, 1e-300)))))
    return float(-0.5 * misfit @ misfit - 0.5 * normalization + occam)

  sign, magnitude = np.linalg.slogdet(information)
  if sign <= 0: raise ValueError("J^T J is singular: the fit does not determine every parameter")
  return float(-0.5 * misfit @ misfit
               - 0.5 * normalization
               + 0.5 * parameters * np.log(2.0 * np.pi)
               - 0.5 * magnitude)

@dataclass(frozen=True, slots=True)
class DiscriminationEvaluation:
  """One scored design. The criterion is in nats; larger separates the models better."""

  expected_log_bayes_factor: float
  standard_error: float
  draws: int
  effective_draws_true: float
  effective_draws_rival: float

  @property
  def reliable(self):
    """Whether both evidence integrals drew on more than a handful of their banks."""
    return min(self.effective_draws_true, self.effective_draws_rival) >= 2.0

def expected_log_bayes_factor(bank_true, bank_rival, deviation, *, seed=0, outer=None):
  """Score a design by how strongly data from `bank_true` favours its own model.

  Both banks are trajectories drawn from their model's prior at the design in question.
  Observations are rows of `bank_true` plus Gaussian noise of scale `deviation`, each scored
  by the log ratio of the two marginal likelihoods, with the generating row excluded from its
  own evidence. `outer` caps how many rows are used; the default uses all of them.
  """
  truth = _checked_bank(bank_true, "bank_true")
  rival = _checked_bank(bank_rival, "bank_rival")
  if truth.shape[1] != rival.shape[1]: raise ValueError(f"banks disagree on samples: {truth.shape[1]} and {rival.shape[1]}")
  scale = positive_scalar("deviation", deviation)

  total = truth.shape[0]
  chosen = np.arange(total) if outer is None else np.arange(total)[: positive_integer("outer", outer)]

  rng = np.random.default_rng(seed)
  observations = truth[chosen] + scale * rng.standard_normal((chosen.size, truth.shape[1]))
  own, effective_true = log_evidence(observations, truth, scale, exclude=chosen)
  other, effective_rival = log_evidence(observations, rival, scale)

  ratio = own - other
  return DiscriminationEvaluation(
    expected_log_bayes_factor=float(np.mean(ratio)),
    standard_error=float(np.std(ratio, ddof=1) / np.sqrt(ratio.size)) if ratio.size > 1 else float("nan"),
    draws=int(ratio.size),
    effective_draws_true=float(np.median(effective_true)),
    effective_draws_rival=float(np.median(effective_rival)),
  )


@dataclass(frozen=True, slots=True)
class ModelScreen:
  """Which rivals are still live, which the data has already settled, and their weights."""

  weights: np.ndarray
  decided: np.ndarray
  live: np.ndarray
  best: int
  margins: np.ndarray

  @property
  def undecided(self) -> int: return int(np.count_nonzero(~self.decided))

  def __str__(self) -> str:
    return (f"{self.undecided} of {self.weights.size} models still live; "
            f"best is index {self.best}, margins {np.round(self.margins, 1)}")


def screen_models(evidence, *, decisive=5.0, floor=1e-3):
  """Decide which rivals are worth designing an experiment for.

  A model the existing records have DECIDED against needs no experiment: once its evidence
  trails the best by more than `decisive` nats the question is closed, and ranking designs by
  how well they would distinguish it spends runs on a settled matter. `decided` marks those
  and the weights are renormalized over the rest.

  Weights rather than a single winner, because a design should serve the posterior it
  actually has: a rival at 0.001 contributes 0.001 of the discrimination utility, with nobody
  choosing a cutoff for it. This also handles the opposite failure -- rivals far apart enough
  that a local criterion misprices them -- since far apart means easy to tell apart, so they
  end up `decided` and drop out, leaving the close pairs a derivative criterion is right for.

  `evidence` is one log-evidence per model on the data in hand. `floor` clips the returned
  weights so a live model cannot round to zero and vanish silently.
  """
  values = finite_array("evidence", evidence).ravel()
  if values.size == 0: raise ValueError("at least one model is required")
  decisive = positive_scalar("decisive", decisive)
  floor = float(floor)                       # half-open: a floor of 1 leaves nothing to spare
  if not 0.0 <= floor < 1.0: raise ValueError(f"floor must lie in [0, 1); got {floor!r}")

  best = int(np.argmax(values))
  margins = values - values[best]
  decided = margins < -decisive
  live = np.flatnonzero(~decided)
  # a posterior over the survivors, computed in a way that does not overflow
  shifted = np.where(decided, -np.inf, margins)
  weights = np.exp(shifted - np.max(shifted[~decided]))
  weights = np.where(decided, 0.0, weights)
  weights = weights / weights.sum()
  if floor > 0.0 and live.size > 0:
    if floor * live.size >= 1.0:
      raise ValueError(f"floor {floor} cannot be met by {live.size} live models")
    # Raise the small ones to the floor and take the room from the LARGE ones. Renormalising
    # everything afterwards instead would push the floored weights straight back under it --
    # it did, landing at 0.009974 against a floor of 0.01.
    below = (weights > 0.0) & (weights < floor)
    if below.any():
      above = (weights > 0.0) & ~below
      weights = np.where(below, floor, weights)
      spare = 1.0 - floor * int(below.sum())
      weights = np.where(above, weights * spare / weights[above].sum(), weights)
  return ModelScreen(weights=weights, decided=decided, live=live, best=best, margins=margins)
