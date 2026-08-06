"""Model discrimination as an integral rather than a minimum.

T-optimality scores a design by `min` over the rival's parameters -- how well the rival can
imitate at its best. That inner problem is multimodal here, and a local method that lands in
the wrong basin returns a wrong answer with nothing to signal it: measured on this study,
Nelder-Mead and Gauss-Newton converged to different basins, and every improvement to the
optimiser lowered the scores.

The criterion here replaces that `min` with an integral. Treating the model label as a
discrete parameter makes discrimination an expected-information-gain problem, whose inner
quantity is the marginal likelihood

    p(Y | M, d) = \\int p(Y | theta, M, d) pi(theta | M) d theta,

and the design is scored by the expected log Bayes factor in favour of the true model,

    U(d) = E_{Y ~ M1} [ log p(Y | M1, d) - log p(Y | M2, d) ],

which is the Kullback-Leibler divergence between the two prior-predictive distributions. An
integral accumulates over every mode instead of requiring that the best one be found, so
multimodality becomes a property of the integrand rather than a failure mode. A missed mode
biases the evidence slightly; a missed basin corrupts a minimum entirely.

What this costs instead: the marginal likelihood depends on `pi(theta | M)` in a way the
minimum does not, so the prior is part of the criterion and has to be stated. And a prior
sample estimate of the evidence degenerates when the likelihood is sharp relative to the
prior -- almost every draw contributes nothing and the sum collapses onto its largest term.
That failure is silent unless measured, so every result here carries the effective sample
size that detects it.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from scipy.special import logsumexp

__all__ = ["DiscriminationEvaluation", "expected_log_bayes_factor", "laplace_log_evidence", "log_evidence"]

def _checked(bank, label):
  table = np.asarray(bank, dtype=float)
  if table.ndim != 2: raise ValueError(f"{label} must be (draws, samples); got shape {table.shape}")
  if table.shape[0] < 2: raise ValueError(f"{label} needs at least two draws to integrate over")
  if not np.all(np.isfinite(table)): raise ValueError(f"{label} contains non-finite trajectories")
  return table

def log_evidence(observations, bank, deviation, *, exclude=None):
  """`log p(Y | M)` for each row of `observations`, and the effective sample size.

  The evidence is a prior integral, estimated by averaging the likelihood over `bank` --
  trajectories already simulated from `pi(theta | M)`. Because the bank is fixed, every
  observation reuses it and no further solves are needed.

  `exclude` drops one bank row per observation. That matters when the observation was
  generated from that same row: leaving it in makes the estimate include the exact
  trajectory that produced the data, which inflates the evidence for whichever model
  generated it and would report discrimination between a model and itself.

  Returns `(log_evidence, effective_draws)`. The second is
  `(sum w)^2 / sum w^2` over the per-draw likelihood weights: it counts how many bank
  members genuinely contributed. Near one, the integral has collapsed onto a single draw
  and the value is unreliable however small its apparent scatter.
  """
  table = _checked(bank, "bank")
  seen = np.atleast_2d(np.asarray(observations, dtype=float))
  if seen.shape[1] != table.shape[1]: raise ValueError(f"observations have {seen.shape[1]} samples, bank has {table.shape[1]}")
  scale = float(deviation)
  if not np.isfinite(scale) or scale <= 0.0: raise ValueError("deviation must be finite and positive")

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

  `log_evidence` estimates the same integral by averaging the likelihood over prior draws,
  which fails once the likelihood is much sharper than the prior: with $201$ samples at the
  measured noise every draw but one contributes nothing, and the effective sample size on
  this study came out at $1.0$ for every design. Expanding about the fit instead costs one
  least-squares solve and one Jacobian, and does not care how sharp the likelihood is.

  `residual` is the whitened misfit `(m - y)/sigma` at the fit, `jacobian` its derivative
  with respect to parameters scaled so the prior is uniform on the unit cube. Then the
  prior density is one, the Hessian of the log-likelihood is `J^T J`, and

      log Z ~= -||r||^2/2 - (N/2) log(2 pi sigma^2) + (p/2) log(2 pi) - (1/2) log det(J^T J).

  The last two terms are the Occam factor: the posterior's share of the prior volume. Its
  sign is what stops a more flexible model from winning on fit alone.

  `cap_at_prior` bounds each eigendirection's Occam factor at 1, which is what a uniform
  prior on a bounded cube implies and what the plain form gets wrong once a parameter is
  weakly identified: an eigenvalue near zero sends `-log det / 2` to `+infinity` and the
  evidence rewards the useless parameter. Turn it on whenever the models being compared
  differ in dimension, which is the case it exists for.

  It is off by default because the expansion written above is the textbook one and should
  stay what this function means by default, not because anything depends on the plain form:
  where every direction is sharper than the prior the two agree to round-off, so the cap
  changes an answer only where the plain form was not entitled to one.

  This inherits Laplace's assumptions -- one dominant, well-separated, roughly Gaussian mode.
  Where several modes matter the evidences add, so summing this over the modes a multistart
  finds is the robust form, and missing one biases the total slightly rather than corrupting
  it, which is the whole reason for preferring an integral to a minimum.
  """
  misfit = np.asarray(residual, dtype=float).ravel()
  matrix = np.atleast_2d(np.asarray(jacobian, dtype=float))
  if matrix.shape[0] != misfit.size: raise ValueError(f"jacobian has {matrix.shape[0]} rows for {misfit.size} residuals")
  if not np.all(np.isfinite(misfit)) or not np.all(np.isfinite(matrix)): raise ValueError("residual and jacobian must be finite")
  # scalar or per-sample: the trial spread of a real record varies across the trace, and
  # collapsing it to one number changes which parts of the curve the fit is asked to match
  scale = np.asarray(deviation, dtype=float)
  if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0): raise ValueError("deviation must be finite and positive")
  if scale.ndim > 1 or (scale.ndim == 1 and scale.size not in (1, misfit.size)):
    raise ValueError(f"deviation must be scalar or one per sample; got {scale.size} for {misfit.size} residuals")

  samples, parameters = matrix.shape
  normalization = float(np.sum(np.log(2.0 * np.pi * np.broadcast_to(scale, (samples,)) ** 2)))
  information = matrix.T @ matrix
  if cap_at_prior:
    # A posterior cannot be wider than the prior it came from. The Occam factor is a
    # product over the eigendirections of `J^T J`, each contributing `sqrt(2 pi / lambda)`
    # -- the posterior width against a prior of width 1 -- and a direction the data does
    # not determine sends `lambda -> 0` and that factor to infinity. The uncapped form then
    # REWARDS a useless parameter: measured on a one-mode record, the two-mode model beat
    # the one-mode by 29.7 nats exactly where its second arm did nothing, and lost by 2.6
    # only where the arm did real work. That is the Gaussian spilling outside the unit cube
    # the prior lives on, not a fact about the models.
    #
    # Capping each direction at 1 says a direction the data cannot see costs nothing and
    # buys nothing, which is what a uniform prior on a bounded cube actually implies.
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
  Synthetic observations are formed by adding Gaussian noise of scale `deviation` to rows of
  `bank_true`, and each is scored by the log ratio of the two marginal likelihoods. The
  generating row is excluded from its own evidence, without which a model would appear to
  discriminate against itself.

  `outer` caps how many rows are used as observations; the default uses all of them.
  """
  truth = _checked(bank_true, "bank_true")
  rival = _checked(bank_rival, "bank_rival")
  if truth.shape[1] != rival.shape[1]: raise ValueError(f"banks disagree on samples: {truth.shape[1]} and {rival.shape[1]}")
  scale = float(deviation)
  if not np.isfinite(scale) or scale <= 0.0: raise ValueError("deviation must be finite and positive")

  total = truth.shape[0]
  if outer is None: chosen = np.arange(total)
  else:
    if not isinstance(outer, Integral) or int(outer) < 1: raise ValueError("outer must be a positive integer or None")
    chosen = np.arange(total)[: int(outer)]

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
