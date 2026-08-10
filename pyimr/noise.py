"""Heteroscedastic observation weighting and a marginalized noise scale.

Optical bubble tracking is least reliable exactly where the dynamics are fastest: motion
blur and finite exposure degrade the edge during collapse, while a slow rebound is
measured cleanly. A single noise level for the whole trace therefore trusts the collapse
too much. Sanchez et al. (Soft Matter 2026, https://doi.org/10.1039/D5SM01193K) handle
this in two stages, both implemented here.

First, a logistic weight in the instantaneous strain rate inflates the variance where the
wall moves fastest (eqns 10-12). Second, a scalar `beta` multiplies every variance and is
marginalized out under a half-Cauchy prior (eqns 15-18), so a model that only fits by
inflating its own error bars is penalized rather than rewarded.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.special import expit

from ._config import SimulationError
from ._validate import finite_array, positive_integer, positive_scalar

__all__ = [
  "ATMOSPHERIC_PRESSURE",
  "LackOfFit",
  "ResidualCheck",
  "WATER_DENSITY",
  "beta_quadrature",
  "characteristic_time",
  "check_residuals",
  "elliptical_gate",
  "hencky_strain_rate",
  "lack_of_fit",
  "marginal_log_likelihood",
  "marginalize_evaluation",
  "predicted_spread",
  "strain_rate_weights",
  "Discrepancy",
  "Enrichment",
  "correlation_time_from",
  "discrepancy",
  "enrichment_overlap",
  "weighted_deviation",
  "whitening",
]

WATER_DENSITY, ATMOSPHERIC_PRESSURE = 998.0, 101325.0


@dataclass(frozen=True, slots=True)
class Whitening:
  """Turns a residual into standard normal coordinates, and carries what that costs.

  `apply` divides by the deviation when the samples are independent and solves the Cholesky
  triangle when they are not. `log_determinant` is `log det (2 pi Sigma)`, which the two cases
  compute differently and which every evidence in this package needs.
  """

  deviation: np.ndarray
  factor: np.ndarray | None
  log_determinant: float

  def apply(self, values):
    array = np.asarray(values, dtype=float)
    if self.factor is None:
      return array / (self.deviation[:, None] if array.ndim == 2 else self.deviation)
    from scipy.linalg import solve_triangular
    return solve_triangular(self.factor, array, lower=True)


def whitening(times, deviation, correlation_time=None):
  """The noise model as one object: independent, or exponentially correlated in time.

  A Gaussian likelihood with independent samples treats every point as fresh information.
  These records do not: `check_residuals` measures lag-one autocorrelation between $0.90$ and
  $0.97$, so $201$ samples carry between $4$ and $10$ independent observations and every
  evidence computed against the independent form is overconfident by that factor.

  `correlation_time` sets the exponential kernel `Sigma_ij = s_i s_j exp(-|t_i - t_j|/tau)`,
  the stationary covariance of an Ornstein-Uhlenbeck process, whose discrete sampling is
  AR(1) with `rho = exp(-dt/tau)`. That is the one-parameter form the measured residuals
  support; nothing here claims the process IS Ornstein-Uhlenbeck, only that one correlation
  length is a better description than none.
  """
  spread = finite_array("deviation", deviation)
  moment = finite_array("times", times).ravel()
  if np.any(spread <= 0.0): raise ValueError("deviation must be positive")
  spread = np.broadcast_to(spread, moment.shape).copy()
  if correlation_time is None:
    return Whitening(deviation=spread, factor=None,
                     log_determinant=float(np.sum(np.log(2.0 * np.pi * spread**2))))

  tau = positive_scalar("correlation_time", correlation_time)
  lag = np.abs(moment[:, None] - moment[None, :])
  covariance = np.outer(spread, spread) * np.exp(-lag / tau)
  factor = np.linalg.cholesky(covariance)
  return Whitening(deviation=spread, factor=factor,
                   log_determinant=float(moment.size * np.log(2.0 * np.pi)
                                         + 2.0 * np.sum(np.log(np.diag(factor)))))


def correlation_time_from(residual, times):
  """The `tau` whose AR(1) lag-one matches these residuals: `tau = -dt / log(rho)`.

  Estimated from the fit's own residual rather than assumed, and `inf` when the residuals
  show no positive correlation -- in which case the independent likelihood was right.
  """
  values = finite_array("residual", residual).ravel()
  moment = finite_array("times", times).ravel()
  if values.size < 8: raise ValueError("at least 8 residuals are needed")
  centred = values - values.mean()
  variance = float(centred @ centred)
  if variance <= 0.0: raise ValueError("residuals are constant")
  rho = float(centred[:-1] @ centred[1:]) / variance
  if rho <= 0.0: return float("inf"), rho
  spacing = float(np.median(np.diff(moment)))
  return -spacing / np.log(rho), rho

def characteristic_time(maximum_radius, *, density=WATER_DENSITY, ambient_pressure=ATMOSPHERIC_PRESSURE):
  """Inertial collapse time `R_max sqrt(rho / p_inf)` -- the usual nondimensionalization.

  Multiply `STRAIN_RATE_THRESHOLD_PER_S` by this to get the threshold `strain_rate_weights`
  expects, since the weighting works in nondimensional rate.
  """
  maximum_radius = positive_scalar("maximum_radius", maximum_radius)
  if float(density) <= 0.0 or float(ambient_pressure) <= 0.0: raise ValueError("density and ambient_pressure must be positive")
  return maximum_radius * np.sqrt(float(density) / float(ambient_pressure))

def hencky_strain_rate(radius_ratio, times, characteristic):
  """Nondimensional logarithmic strain rate, `d(ln R)/dt * t_c`.

  Logarithmic rather than engineering strain because the wall stretch spans a factor of
  several during a collapse, where the two disagree badly.
  """
  radius_ratio = np.asarray(radius_ratio, dtype=float)
  times = np.asarray(times, dtype=float)
  if radius_ratio.shape != times.shape: raise ValueError("radius_ratio and times must have the same shape")
  if np.any(radius_ratio <= 0.0): raise ValueError("radius_ratio must be positive")
  return np.gradient(radius_ratio, times) / radius_ratio * float(characteristic)

STRAIN_RATE_THRESHOLD_PER_S = 1e5
"""Onset of the high-strain-rate regime (eqn 5). Multiply by the characteristic time to
get the nondimensional threshold the weighting expects."""

_DEFAULT_FLOOR = 0.1
_DEFAULT_STEEPNESS = 1.0

def strain_rate_weights(strain_rate, threshold, *, steepness=_DEFAULT_STEEPNESS, floor=_DEFAULT_FLOOR):
  """Bounded logistic weight in the strain rate (eqns 10-11).

  `w = floor + (1 - floor) / (1 + exp(-steepness * (threshold - |rate|) / threshold))`,
  decreasing in `|rate|`: fast wall motion earns a smaller weight and, through
  `weighted_deviation`, a larger variance.

  The weight is bounded below by `floor` but does **not** reach 1 from above. Its supremum
  is `floor + (1 - floor) / (1 + exp(-steepness))` -- 0.758 at the defaults -- because
  `|rate| >= 0` caps the logistic argument at `steepness`. Only the ratio between weights
  matters here, since a constant factor on every variance is exactly what `beta` absorbs.
  """
  rate = finite_array("strain_rate", strain_rate)
  threshold = positive_scalar("threshold", threshold)
  steepness = positive_scalar("steepness", steepness)
  if not 0.0 < float(floor) <= 1.0: raise ValueError("floor must lie in (0, 1]")

  # expit, not 1/(1+exp(-x)): the argument runs to -inf as the rate grows, and the naive
  # form overflows there rather than saturating at the floor
  return float(floor) + (1.0 - float(floor)) * expit(steepness * (threshold - np.abs(rate)) / threshold)

def weighted_deviation(deviation, weights):
  """Turn weights into standard deviations (eqn 12 at `beta = 1`): `sigma^2 = sigma0^2 / w`.

  The result drops straight into `FieldObservation.standard_deviation`, which already
  accepts a per-sample array, so the weighting needs no change to the likelihood itself.
  """
  weights = np.asarray(weights, dtype=float)
  if np.any(weights <= 0.0): raise ValueError("weights must be positive")
  return np.asarray(deviation, dtype=float) / np.sqrt(weights)

def elliptical_gate(strain, strain_rate, strain_threshold, strain_rate_threshold):
  """Keep samples outside the ellipse of eqn 6 -- a boolean mask, `True` to retain.

  `(strain/strain_th)^2 + (rate/rate_th)^2 >= 1` drops near-equilibrium samples that carry
  little information about the constitutive model while still contributing noise.
  """
  strain = np.asarray(strain, dtype=float)
  rate = np.asarray(strain_rate, dtype=float)
  strain_threshold = positive_scalar("strain_threshold", strain_threshold)
  strain_rate_threshold = positive_scalar("strain_rate_threshold", strain_rate_threshold)
  return (strain / strain_threshold) ** 2 + (rate / strain_rate_threshold) ** 2 >= 1.0

def beta_quadrature(*, minimum=0.05, maximum=10.0, count=200):
  """Nodes and normalized prior weights for the half-Cauchy noise scale (eqns 16-18).

  `P(beta) = (2/pi) / (1 + beta^2)` on a uniform grid, integrated by the trapezoidal rule
  and renormalized so the weights sum to one.

  The default range is the paper's [0.05, 10], which it describes as over 99.9% of the prior
  mass; the exact mass for the scale-1 half-Cauchy of eqn 16 is `(2/pi)(atan(10) -
  atan(0.05)) = 0.9047`, so the figure does not follow. Renormalizing makes this the prior
  CONDITIONED on the range, proper either way -- but `beta > 10` is then excluded rather than
  negligible, so a fit driven to that boundary is a warning.
  """
  if not 0.0 < minimum < maximum: raise ValueError("require 0 < minimum < maximum")
  if int(count) < 2: raise ValueError("count must be at least 2")

  nodes = np.linspace(float(minimum), float(maximum), int(count))
  density = (2.0 / np.pi) / (1.0 + nodes**2)
  spacing = nodes[1] - nodes[0]
  rule = np.full(nodes.shape, spacing)
  rule[0] = rule[-1] = 0.5 * spacing
  weights = density * rule
  return nodes, weights / weights.sum()

def marginal_log_likelihood(chi_squared, count, *, nodes=None, weights=None, normalization=0.0):
  """Marginalize the noise scale out of a Gaussian likelihood (eqn 17).

  `beta` scales every variance, so its effect on the log-likelihood depends on the data
  only through `chi_squared = sum(r^2 / sigma^2)` at `beta = 1` and the number of samples:

      log L(beta) = -0.5 * (chi_squared / beta^2 + count * log(beta^2) + normalization)

  The forward model never re-enters, so marginalizing costs one pass over the grid rather
  than one solve per node. `normalization` is `sum(log(2 pi sigma^2))` -- constant in
  `beta`, so it may be left at zero when only differences matter.
  """
  chi_squared, count = float(chi_squared), int(count)
  if chi_squared < 0.0: raise ValueError("chi_squared must be non-negative")
  if count <= 0: raise ValueError("count must be positive")
  if (nodes is None) != (weights is None): raise ValueError("pass both nodes and weights, or neither")
  if nodes is None: nodes, weights = beta_quadrature()

  nodes = np.asarray(nodes, dtype=float)
  weights = np.asarray(weights, dtype=float)
  if np.any(nodes <= 0.0): raise ValueError("beta nodes must be positive")

  terms = np.log(weights) - 0.5 * (chi_squared / nodes**2 + count * np.log(nodes**2))
  peak = float(np.max(terms))
  return float(peak + np.log(np.sum(np.exp(terms - peak)))) - 0.5 * float(normalization)

def predicted_spread(config, times, *, relative=0.01, samples=5, field="radius_ratio"):
  """The trial-to-trial spread this configuration PREDICTS, given preparation scatter.

  A repeated experiment does not repeat its initial radius exactly, so a model implies a
  spread across trials. Comparing that with the measured spread is a test the usual
  chi-squared cannot make: it uses the scatter as a prediction target rather than only as
  a noise scale.

  It matters where the forward map is ill-conditioned: a quadratic Zener at strong stiffening
  turns a 1e-9 change in `R0` into 1e-3 of the trajectory and predicts a spread far beyond
  what the gelatin records show, so the data exclude those parameters whatever their
  chi-squared. That particular region is now refused earlier by `max_radius_ratio`; the
  argument stands on its own for models that are sensitive but physical (#203).

  Returns the median over time of the across-sample standard deviation, in the same units
  as `field`. `None` if any sample cannot be integrated, since a spread over a subset of
  an ensemble is not the ensemble's spread.
  """
  from ._solver import simulate

  if samples < 2: raise ValueError("samples must be at least 2 to have a spread")
  if not 0.0 < relative < 1.0: raise ValueError("relative scatter must lie in (0, 1)")
  traces = []
  for index in range(int(samples)):
    bump = relative * (2.0 * index / (samples - 1) - 1.0)  # a sweep, not a draw: reproducible
    try:
      result = simulate(times, replace(config, R0=config.R0 * (1.0 + bump)))
    except SimulationError:
      return None
    traces.append(np.asarray(getattr(result, field), dtype=float))
  return float(np.median(np.std(np.array(traces), axis=0, ddof=1)))

def marginalize_evaluation(evaluation, **options):
  """Marginal log-likelihood for a `LikelihoodEvaluation` from `pyimr.inference`.

  The stored residual is already whitened by the per-sample deviations, so its sum of
  squares is exactly the `chi_squared` the marginalization needs -- no re-solve, and no
  import of the inference module.

  Note this returns the marginal *without* the `sum(log(2 pi sigma^2))` offset that
  `evaluation.log_likelihood` carries, so the two are comparable only up to that constant.
  Pass it as `normalization` when an absolute value is wanted.
  """
  residual = np.asarray(evaluation.residual, dtype=float).ravel()
  return marginal_log_likelihood(float(residual @ residual), residual.size, **options)

@dataclass(frozen=True, slots=True)
class ResidualCheck:
  """Whether a fit's residuals look like the independent noise the likelihood assumed."""

  samples: int
  chi_squared_per_sample: float
  lag_one: float
  effective_samples: float
  inflation: float
  independent: bool
  summary: str

  def __str__(self) -> str: return self.summary

def check_residuals(residual, *, threshold=None):
  """Are these residuals white, and if not, by how much is the fit over-confident?

  `chi_squared/N` near 1 is evidence that a model fits, not that the LIKELIHOOD is right, and
  the two fail independently: correlated residuals mean most of the information a Gaussian
  independent likelihood counts is double counted, so the posterior is too narrow and the
  chi-squared does not say so. On a gelatin record `chi2/N = 0.974` read as an excellent fit
  alongside lag-one 0.90 and an effective sample size near 10 of 201 -- error bars four times
  too small (#216).

  `residual` is the whitened residual, as `LikelihoodEvaluation.residual` carries it.
  Returns a `ResidualCheck`; `inflation` is the factor by which parameter uncertainties are
  understated, `sqrt(N / N_eff)`.
  """
  values = finite_array("residuals", residual).ravel()
  count = values.size
  if count < 8: raise ValueError("at least 8 residuals are needed to estimate autocorrelation")

  centred = values - values.mean()
  variance = float(centred @ centred)
  if variance <= 0.0: raise ValueError("residuals are constant, so autocorrelation is undefined")

  # Sum autocorrelations until the first negative one -- Geyer's initial positive sequence.
  # Summing every lag instead adds noise from the long tail, where each estimate is built
  # from few pairs and is mostly sampling error.
  total, lag_one = 0.0, 0.0
  for lag in range(1, min(count // 2, 200)):
    rho = float(centred[:-lag] @ centred[lag:]) / variance
    if lag == 1: lag_one = rho
    if rho <= 0.0: break
    total += rho
  effective = count / (1.0 + 2.0 * total)

  band = 2.0 / np.sqrt(count) if threshold is None else float(threshold)
  independent = abs(lag_one) <= band
  inflation = float(np.sqrt(count / effective)) if effective > 0 else float("inf")
  chi2 = float(values @ values) / count
  if independent:
    summary = (f"residuals look independent: lag-one {lag_one:+.3f} within +/-{band:.3f}, "
               f"chi2/N {chi2:.3f}")
  else:
    summary = (f"residuals are correlated: lag-one {lag_one:+.3f} against a white-noise band of "
               f"+/-{band:.3f}, so {effective:.0f} of {count} samples are independent and "
               f"parameter uncertainties are about {inflation:.1f}x too small. chi2/N of "
               f"{chi2:.3f} does not show it")
  return ResidualCheck(
    samples=count, chi_squared_per_sample=chi2, lag_one=lag_one,
    effective_samples=float(effective), inflation=inflation,
    independent=bool(independent), summary=summary,
  )


@dataclass(frozen=True, slots=True)
class LackOfFit:
  """The replicate-based split of a residual into model error and measurement scatter."""

  pure_error: float
  lack_of_fit: float
  ratio: float
  pure_df: int
  lack_df: int
  trials: int
  summary: str

  def __str__(self) -> str: return self.summary


def lack_of_fit(observed, predicted, spread, trials, parameters):
  """Split a residual into what the apparatus cannot repeat and what the model cannot follow.

  With replicates, the question `chi_squared/N` is routinely misread as answering has a
  classical answer that costs no model at all: the spread BETWEEN repeated runs at one
  setting is pure error, and whatever the model misses beyond it is lack of fit.

      SS_pure = (J-1) sum_i s_i^2                  df = k(J-1)
      SS_lack = J sum_i (ybar_i - yhat_i)^2        df = k - p
      ratio   = (SS_lack/df_lack) / (SS_pure/df_pure)

  `observed` is the MEAN over `trials` repeats at each of `k` settings, `spread` their sample
  standard deviation, and `predicted` the model there. Note the `J` in `SS_lack`: the data is
  a mean, its standard error is smaller than the between-run spread by `sqrt(J)`, and a
  `chi^2` formed against `spread` is quietly too forgiving by that factor. Here it is part of
  the statistic rather than a correction anyone has to remember.

  On the gelatin records against their best-fitting qSLS the ratio is 14.2, 2.9 and 3.3
  against a five-percent critical value near 1.2 -- every record rejects, the coldest by an
  order of magnitude because its apparatus repeats three times more tightly, not because its
  fit is worse. Lag-one cannot see that difference; it reads 0.919, 0.968, 0.911.

  TWO THINGS THE RATIO IS NOT. Not a p-value: the F distribution wants errors independent
  across settings and these are correlated at 0.918, so read it as an effect size. And it is
  a LOWER bound on model inadequacy wherever the replicates differ by more than measurement,
  since real bubble-to-bubble variation inflates pure error and the denominator with it.
  """
  mean = np.asarray(observed, dtype=float).ravel()
  model = np.asarray(predicted, dtype=float).ravel()
  scale = finite_array("spread", spread, non_negative=True).ravel()
  if not (mean.size == model.size == scale.size):
    raise ValueError(f"observed, predicted and spread must agree in length; "
                     f"got {mean.size}, {model.size}, {scale.size}")
  if mean.size == 0: raise ValueError("at least one setting is required")
  trials = positive_integer("trials (pure error needs repeats)", trials, minimum=2)
  parameters = positive_integer("parameters", parameters, minimum=0)
  settings = mean.size
  if settings <= parameters:
    raise ValueError(f"{settings} settings cannot support {parameters} parameters plus a "
                     "lack-of-fit test; the numerator has no degrees of freedom")

  pure_df, lack_df = settings * (trials - 1), settings - parameters
  pure = float((trials - 1) * np.sum(scale**2)) / pure_df
  lack = float(trials * np.sum((mean - model) ** 2)) / lack_df
  ratio = lack / pure if pure > 0.0 else float("inf")
  verdict = ("consistent with pure error" if ratio < 1.2 else
             f"{ratio:.1f}x pure error -- the model misses structure the apparatus repeats")
  return LackOfFit(pure_error=pure, lack_of_fit=lack, ratio=ratio, pure_df=pure_df,
                   lack_df=lack_df, trials=trials,
                   summary=f"lack of fit {ratio:.2f} on {lack_df} and {pure_df} df: {verdict}")


@dataclass(frozen=True, slots=True)
class Discrepancy:
  """The part of the model error the data can see, and what the invisible part could cost."""

  identifiable: np.ndarray
  size: float
  absorbed: float
  at_optimum: bool
  bias_sigmas: float
  summary: str

  def __str__(self) -> str: return self.summary


def discrepancy(residual, jacobian, *, absorbed_ratio=1.0, tolerance=1e-2):
  """Split model error into the part a design can see and a bound on the part it cannot.

  `lack_of_fit` says a model is inadequate; it does not say by how much the parameters suffer.
  Write the truth as `y = m(theta) + delta + noise`. Projecting the residual onto the model's
  own span,

      delta_hat = (I - P) d,     P = J (J^T J)^-1 J^T,

  is the identifiable discrepancy: what no choice of parameters reproduces. The complement is
  the problem. At a converged interior fit the normal equations make `d` orthogonal to every
  column of `J`, so `P d = 0` EXACTLY -- the absorbed part of `delta` has already been taken up
  into `theta_hat` and left no trace at all. That is the Kennedy--O'Hagan unidentifiability, and
  it means the bias cannot be measured from the fit. It can only be bounded.

  The bound is unusually clean. Over all absorbed discrepancies with `||delta_par|| <= B`, the
  worst bias in coordinate `k` is `B * sqrt([(J^T J)^-1]_kk)` -- exactly `B` times that
  parameter's own standard deviation, the same multiple for every coordinate. So one number
  answers it, and `absorbed_ratio` is the only assumption: `B = absorbed_ratio * ||delta_hat||`,
  taking the unseen part to be at most `absorbed_ratio` times the seen part. One is the
  defensible default and is not conservative -- nothing forbids the invisible component being
  larger.

  `residual` must be whitened by the standard error of the MEAN, `spread / sqrt(trials)`, when
  `y` is an average over repeats. Dividing by `spread` instead understates the misfit by
  `sqrt(trials)` and is the same omission `lack_of_fit` exists to correct.

  `at_optimum` reports whether `||P d|| / ||d||` is below `tolerance`. When it is not, the fit
  has not converged or is against a bound, `delta_hat` is contaminated by the distance to the
  optimum, and the bias bound does not apply -- a stored grid argmax fails this outright (#235).
  """
  misfit = finite_array("residual", residual).ravel()
  matrix = np.atleast_2d(finite_array("jacobian", jacobian))
  if matrix.shape[0] != misfit.size:
    raise ValueError(f"jacobian has {matrix.shape[0]} rows for {misfit.size} residuals")
  ratio = positive_scalar("absorbed_ratio", absorbed_ratio, allow_zero=True)
  tolerance = positive_scalar("tolerance", tolerance)

  # lstsq, not a normal-equation solve: the sloppy directions make `J^T J` badly conditioned,
  # and the projection is what is wanted, not the coefficients
  coefficients, *_ = np.linalg.lstsq(matrix, misfit, rcond=None)
  absorbed = matrix @ coefficients
  left = misfit - absorbed
  total = float(np.linalg.norm(misfit))
  taken = float(np.linalg.norm(absorbed))
  size = float(np.linalg.norm(left))
  settled = bool(total <= 0.0 or taken / total <= tolerance)

  bias = ratio * size
  where = ("at the optimum" if settled else
           f"NOT at an optimum -- {taken / total:.1%} of the residual is still absorbable, so "
           "this is contaminated by the distance to it and the bound does not apply")
  return Discrepancy(
    identifiable=left, size=size, absorbed=taken, at_optimum=settled, bias_sigmas=bias,
    summary=(f"identifiable discrepancy {size:.1f} against {total:.1f} total, {where}; "
             f"every parameter could be biased by up to {bias:.0f} of its own standard "
             f"deviations if the unseen part is {ratio:g}x the seen one"),
  )


@dataclass(frozen=True, slots=True)
class Enrichment:
  """Which candidate physics could account for the discrepancy, and how much of it."""

  overlap: dict[str, float]
  removable: dict[str, float]
  joint: float
  summary: str
  reachable: dict[str, bool]

  def __str__(self) -> str: return self.summary

  @property
  def best(self):
    """The most promising candidate that can actually be reached, or `None` if none can.

    A one-sided candidate pointing the wrong way is excluded however large its overlap: it
    cannot reduce the discrepancy at any admissible amplitude.
    """
    live = {k: v for k, v in self.removable.items() if self.reachable[k]}
    return max(live.items(), key=lambda pair: pair[1]) if live else None


def enrichment_overlap(identifiable, jacobian, directions, *, one_sided=True):
  """Screen candidate physics against the discrepancy it would have to explain.

  `discrepancy` leaves a curve the current model cannot produce. Any enrichment proposes a new
  sensitivity direction, and only the part of it ORTHOGONAL to the existing model's span can
  reduce that curve -- the rest is absorbed by refitting the parameters already present, which
  improves `chi^2` while explaining nothing. So the question is a cosine,

      overlap = |<(I - P) v, delta_hat>| / (||(I - P) v|| ||delta_hat||),

  and the fraction of the discrepancy's variance a candidate removes is `overlap^2`.

  This inverts the usual order and is much cheaper for it. Fitting each candidate and comparing
  evidence costs a multistart per candidate and answers "which fits best", which among rejected
  models is not the question. This costs ONE solve per candidate and answers "could this
  possibly be the missing physics" -- a candidate with overlap near zero cannot help however
  many parameters it brings, and its Bayes factor will improve anyway by re-absorbing what was
  already absorbed. Screen first, fit what survives.

  `directions` maps a name to that candidate's whitened `dR/d(new parameter)`, on the same
  samples and whitening as `identifiable` and `jacobian`. `joint` is the fraction all of them
  remove together, which is smaller than the sum when candidates propose the same curve twice.

  SIGN, WHICH IS NOT A DETAIL. Most enrichment amplitudes are one-sided -- a Maxwell arm's
  share, a stiffening coefficient and a switched-on transport term are all non-negative -- so
  a candidate whose direction ANTI-aligns with the discrepancy cannot reduce it at any
  admissible amplitude. Scoring `|cos|` treats such a candidate as promising: measured on
  these records, a second relaxation arm scored 58% at 23 C from a cosine of
  -0.762, and it can only ever make the residual worse. `reachable` marks the sign, `best`
  skips what is unreachable, and `one_sided=False` restores the symmetric reading for an
  amplitude that really can take either sign.

  AND THE MAGNITUDE IS AN UPPER BOUND, not a forecast. It is what a candidate could remove if
  its amplitude were free and the base parameters moved only linearly. A fit obeys neither:
  fitting the two-mode model reduced the discrepancy by 1%, 11% and 10% where this
  screen projected 39%, 58% and 24%. Use it to RULE OUT -- near zero, or wrong
  signed, means a candidate cannot help -- and fit whatever survives rather than ranking by
  the number.

  For the design side of the same question -- what experiment would tell these candidates
  apart, or separate the discrepancy from a parameter shift -- pass `delta_hat` to
  `pyimr.measure.augmented_information` as a difference direction. Its amplitude then enters
  the Fisher matrix as one more coordinate, exactly as a rival model does, and
  `pyimr.gain` designs against it with no further machinery.
  """
  left = finite_array("identifiable", identifiable).ravel()
  matrix = np.atleast_2d(finite_array("jacobian", jacobian))
  if matrix.shape[0] != left.size:
    raise ValueError(f"jacobian has {matrix.shape[0]} rows for {left.size} residuals")
  if not directions: raise ValueError("at least one candidate direction is required")
  size = float(np.linalg.norm(left))
  if size <= 0.0: raise ValueError("the identifiable discrepancy is zero; nothing to explain")

  basis, _ = np.linalg.qr(matrix)                 # an orthonormal basis for what refitting absorbs
  overlap, reachable, surviving = {}, {}, []
  for name, direction in directions.items():
    column = finite_array(f"direction {name!r}", direction).ravel()
    if column.size != left.size:
      raise ValueError(f"direction {name!r} has {column.size} samples, expected {left.size}")
    free = column - basis @ (basis.T @ column)
    scale = float(np.linalg.norm(free))
    # A direction the existing parameters reproduce entirely explains nothing, however large.
    # The test is RELATIVE: what is left of it is round-off, and normalising by a round-off
    # norm turns that noise into a cosine -- it read 0.023 for a direction lying exactly in
    # the span.
    if scale <= 1e-8 * float(np.linalg.norm(column)):
      overlap[name] = 0.0
      reachable[name] = False
      continue
    aligned = float(free @ left) / (scale * size)
    overlap[name] = abs(aligned)
    reachable[name] = bool(aligned > 0.0 or not one_sided)
    if reachable[name]: surviving.append(free / scale)

  joint = 0.0
  if surviving:
    # SVD rather than QR: two candidates proposing the same curve make a rank-deficient
    # stack, and QR hands back an arbitrary second column spanning the null space, which
    # then collects projection that no candidate offered.
    stack = np.column_stack(surviving)
    left_vectors, values, _ = np.linalg.svd(stack, full_matrices=False)
    keep = values > max(stack.shape) * np.finfo(float).eps * values[0]
    joint = float(np.linalg.norm(left_vectors[:, keep].T @ left) ** 2) / size**2

  removable = {name: value**2 for name, value in overlap.items()}
  ranked = sorted(((k, v) for k, v in removable.items() if reachable[k]), key=lambda pair: -pair[1])
  blocked = [k for k in removable if not reachable[k]]
  return Enrichment(
    overlap=overlap, removable=removable, joint=min(joint, 1.0), reachable=reachable,
    summary=("at most, and only if a fit could reach it: "
             + (", ".join(f"{name} {value:.1%}" for name, value in ranked) or "nothing reachable")
             + f"; together {min(joint, 1.0):.1%}"
             + (f"; wrong-signed and unreachable: {', '.join(blocked)}" if blocked else "")),
  )
