"""Redundancy-based parameter prior, Occam model prior, and normalized model posteriors.

A uniform prior over each model's parameter box is the maximum-entropy default, but it
misprices a model set whose members nest. A three-parameter model contains its
two-parameter special case, and across a whole region of its box it merely reproduces that
simpler model's stress history. Uniform weighting hands that region full prior mass, so
the flexible model is rewarded for behaviour it shares with a model that spends fewer
parameters to get it.

Sanchez et al. (Soft Matter 2026, https://doi.org/10.1039/D5SM01193K) price it directly:
compare each model's stress integral against every simpler model it contains, and
down-weight the parameters where the difference is below what the data can resolve
(eqns 22-26). `model_prior` then adds the BIC-style complexity term of eqn 27, and
`model_posterior` normalizes across the model set.

The mismatch of eqn 22 uses the same strain-rate weights as `pyimr.noise`, so a stress
difference concentrated where the measurement is unreliable counts for less.
"""

from __future__ import annotations

import numpy as np

__all__ = [
  "harmonic_bottleneck",
  "model_posterior",
  "model_prior",
  "normalize_log_coordinates",
  "parameter_prior",
  "redundancy_factor",
  "redundancy_weight",
  "relative_mismatch",
  "stress_scale",
]

_SCALE_FLOOR = 1e-6
_MEDIAN_FLOOR = 1e-12
_NORMAL_CONSISTENCY = 1.4826

def _weighted_norm(values, weights):
  values = np.asarray(values, dtype=float)
  if weights is None: return float(np.sqrt(values @ values))
  weights = np.asarray(weights, dtype=float)
  if weights.shape != values.shape: raise ValueError(f"weights must match values; got {weights.shape} and {values.shape}")
  if np.any(weights < 0.0): raise ValueError("weights must be non-negative")
  return float(np.sqrt(np.sum(weights * values**2)))

def stress_scale(stress, *, floor=_SCALE_FLOOR):
  """Smallest *relative* stress change a model can be said to resolve (eqn 23's `tau`).

  A robust spread over a robust level: `1.4826 * MAD(S) / median(|S|)`. The 1.4826 makes
  the MAD agree with the standard deviation for Gaussian data, and dividing by the median
  magnitude makes it dimensionless, matching `relative_mismatch`, which is also a ratio.

  Median and MAD rather than mean and standard deviation because the stress history is
  dominated by collapse spikes, which would inflate a non-robust spread and make every
  model look indistinguishable.
  """
  stress = np.asarray(stress, dtype=float).ravel()
  if stress.size == 0: raise ValueError("stress must be non-empty")
  if not np.all(np.isfinite(stress)): raise ValueError("stress must be finite")

  level = max(float(np.median(np.abs(stress))), _MEDIAN_FLOOR)
  spread = float(np.median(np.abs(stress - np.median(stress))))
  return max(_NORMAL_CONSISTENCY * spread / level, float(floor))

def relative_mismatch(parent_stress, child_stress, weights=None):
  """Weighted relative stress mismatch between a model and one it contains (eqn 22).

  `||S_P - S_C||_w / ||S_P||_w`. Zero means the parent is doing nothing its child does not
  already do at these parameters.
  """
  parent = np.asarray(parent_stress, dtype=float).ravel()
  child = np.asarray(child_stress, dtype=float).ravel()
  if parent.shape != child.shape: raise ValueError(f"stress histories must match; got {parent.shape} and {child.shape}")
  if weights is not None: weights = np.asarray(weights, dtype=float).ravel()

  denominator = _weighted_norm(parent, weights)
  if denominator <= 0.0: raise ValueError("parent stress history is identically zero")
  return _weighted_norm(parent - child, weights) / denominator

def redundancy_factor(mismatch, scale, *, exponent=2):
  """Map a mismatch onto `[0, 1)` against the resolvable scale (eqn 23).

  `F^n / (F^n + tau^n)`: near 0 when the parent only emulates the child, approaching 1
  when it does something genuinely different. `exponent` is 2 in the paper; the reference
  implementation raises it to the difference in parameter count, so that a model buying
  more parameters must earn them more convincingly.
  """
  mismatch, scale = float(mismatch), float(scale)
  if mismatch < 0.0: raise ValueError("mismatch must be non-negative")
  if not np.isfinite(scale) or scale <= 0.0: raise ValueError("scale must be finite and positive")
  if int(exponent) < 1: raise ValueError("exponent must be a positive integer")

  numerator = mismatch ** int(exponent)
  return float(numerator / (numerator + scale ** int(exponent)))

def redundancy_weight(parent_stress, child_stresses, *, weights=None, scale=None, exponent=2):
  """Redundancy against the *most* similar contained model (eqn 24).

  The minimum over children, not a product or a mean: a parent that can emulate any one
  simpler model is redundant there, however different it looks from the others.

  With no children -- the simplest model in a hierarchy -- there is nothing to be
  redundant with, so this is 1 and the prior reduces to the bottleneck alone.
  """
  children = [np.asarray(item, dtype=float).ravel() for item in child_stresses]
  if not children: return 1.0
  if scale is None: scale = stress_scale(parent_stress)
  return min(redundancy_factor(relative_mismatch(parent_stress, child, weights), scale, exponent=exponent) for child in children)

def normalize_log_coordinates(values, lower, upper):
  """Map parameters onto `[0, 1]` in log coordinates, making the prior range-invariant.

  IMR parameters span decades, so a linear normalization would make the prior depend on
  where the bounds were drawn rather than on the physics.
  """
  values = np.asarray(values, dtype=float)
  lower, upper = float(lower), float(upper)
  if not 0.0 < lower < upper: raise ValueError("require 0 < lower < upper")
  if np.any(values <= 0.0): raise ValueError("values must be positive to take logarithms")
  return (np.log(values) - np.log(lower)) / (np.log(upper) - np.log(lower))

def harmonic_bottleneck(coordinates, *, epsilon=1e-12):
  """Harmonic mean of normalized coordinates (eqn 25): `d / sum(1 / (theta + eps))`.

  A bottleneck rather than an average -- one coordinate pinned near the bottom of its
  range drags the whole prior down, because a parameter driven to its boundary is one the
  data did not need.
  """
  coordinates = np.asarray(coordinates, dtype=float)
  if coordinates.size == 0: raise ValueError("coordinates must be non-empty")
  if np.any(coordinates < 0.0): raise ValueError("coordinates must be non-negative; normalize first")
  return float(coordinates.shape[-1] / np.sum(1.0 / (coordinates + float(epsilon)), axis=-1))

def parameter_prior(bottlenecks, redundancies):
  """Combine the two factors and normalize over the grid (eqn 26).

  `P(theta|M) ∝ H(theta) * w_red(theta)`, summing to one across the supplied grid, so it
  is the discrete prior the eqn 21 quadrature weights need.
  """
  bottlenecks = np.asarray(bottlenecks, dtype=float).ravel()
  redundancies = np.asarray(redundancies, dtype=float).ravel()
  if bottlenecks.shape != redundancies.shape: raise ValueError("bottlenecks and redundancies must have the same shape")
  if np.any(bottlenecks < 0.0) or np.any(redundancies < 0.0): raise ValueError("prior factors must be non-negative")

  unnormalized = bottlenecks * redundancies
  total = float(unnormalized.sum())
  if total <= 0.0: raise ValueError("prior is identically zero; nothing to normalize")
  return unnormalized / total

def model_prior(free_parameters, effective_observations):
  """BIC-style Occam penalty (eqn 27): `exp(-k/2 * log N_eff) = N_eff ** (-k/2)`.

  Unnormalized -- `model_posterior` normalizes across the set.
  """
  count = int(free_parameters)
  observations = float(effective_observations)
  if count < 0: raise ValueError("free_parameters must be non-negative")
  if observations <= 1.0: raise ValueError("effective_observations must exceed 1 for the penalty to be a penalty")
  return float(observations ** (-0.5 * count))

def model_posterior(log_evidences, log_model_priors=None):
  """Normalized posterior over a model set: `P(M|D) ∝ P(M) P(D|M)`, summing to one.

  Done in log space through a log-sum-exp, because evidences across a model set span many
  orders of magnitude and direct summation underflows.
  """
  log_evidences = np.asarray(log_evidences, dtype=float).ravel()
  if log_evidences.size == 0: raise ValueError("need at least one model")
  if log_model_priors is None:
    log_priors = np.zeros_like(log_evidences)
  else:
    log_priors = np.asarray(log_model_priors, dtype=float).ravel()
    if log_priors.shape != log_evidences.shape: raise ValueError("log_model_priors must match log_evidences")
  if not np.all(np.isfinite(log_evidences)): raise ValueError("log_evidences must be finite")

  terms = log_evidences + log_priors
  peak = float(np.max(terms))
  weights = np.exp(terms - peak)
  return weights / float(weights.sum())
