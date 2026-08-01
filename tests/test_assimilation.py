"""Ensemble analysis against the exact linear-Gaussian update (#122)."""

import numpy as np
import pytest

from pyimr.assimilation import enkf_analysis, kalman_analysis

SECTION = "10. State estimation"


def _linear_gaussian(seed=0, width=4, observed=2):
  rng = np.random.default_rng(seed)
  factor = rng.normal(size=(width, width))
  covariance = factor @ factor.T + width * np.eye(width)
  mean = rng.normal(size=width)
  operator = rng.normal(size=(observed, width))
  observation = rng.normal(size=observed)
  noise = np.diag([0.3, 0.7][:observed]) ** 2
  return mean, covariance, observation, operator, noise


def test_the_ensemble_analysis_converges_on_the_exact_kalman_update(measured):
  """The only claim that pins EnKF: in the linear-Gaussian case it must approach the

  exact answer, at the Monte Carlo rate and no faster.
  """
  mean, covariance, observation, operator, noise = _linear_gaussian()
  exact_mean, exact_covariance = kalman_analysis(mean, covariance, observation, operator, noise)

  errors = []
  for size in (50, 500, 5000, 50000):
    rng = np.random.default_rng(1)
    members = rng.multivariate_normal(mean, covariance, size=size)
    analysis = enkf_analysis(members, observation, operator, noise, rng=rng)
    errors.append((
      float(np.linalg.norm(analysis.mean - exact_mean) / np.linalg.norm(exact_mean)),
      float(np.linalg.norm(analysis.covariance - exact_covariance) / np.linalg.norm(exact_covariance)),
    ))

  measured("enkf vs kalman", "  ".join(f"n={n}: {m:.3f}/{c:.3f}" for n, (m, c) in zip((50, 500, 5000, 50000), errors, strict=True)))
  assert [m for m, _ in errors] == sorted((m for m, _ in errors), reverse=True), "mean error must fall with ensemble size"
  assert errors[-1][0] < 0.05, f"50k members should land within 5% of the exact mean; got {errors[-1][0]:.3f}"
  assert errors[-1][1] < 0.02, f"50k members should land within 2% of the exact covariance; got {errors[-1][1]:.3f}"


def test_perturbing_the_observations_is_what_keeps_the_spread_honest():
  """Without perturbation the analysis ensemble is too tight -- the classic EnKF failure.

  Fully observed, so the effect is unmissable: the unperturbed update produces
  `(I-KH) P (I-KH)^T` and misses the `K R K^T` term, landing at a fifth of the true
  spread. Under partial observation the same bug shows up as a 3% deficit, which is
  exactly the size that survives review.
  """
  mean, covariance = np.zeros(2), np.eye(2)
  operator, noise = np.eye(2), 0.25 * np.eye(2)
  observation = np.array([1.0, -1.0])
  _, exact_covariance = kalman_analysis(mean, covariance, observation, operator, noise)
  rng = np.random.default_rng(7)
  members = rng.multivariate_normal(mean, covariance, size=40000)

  perturbed = enkf_analysis(members, observation, operator, noise, rng=np.random.default_rng(8))
  deterministic = enkf_analysis(members, observation, operator, noise, perturb=False)

  exact = float(np.trace(exact_covariance))
  assert abs(np.trace(perturbed.covariance) - exact) / exact < 0.05
  assert np.trace(deterministic.covariance) < 0.4 * exact, "the unperturbed update should collapse the spread"


def test_a_sharp_observation_pulls_the_mean_onto_it():
  mean, covariance, observation, operator, _ = _linear_gaussian()
  rng = np.random.default_rng(4)
  members = rng.multivariate_normal(mean, covariance, size=4000)
  analysis = enkf_analysis(members, observation, operator, 1e-4, rng=rng, perturb=False)
  np.testing.assert_allclose(operator @ analysis.mean, observation, rtol=1e-3)


def test_the_analysis_leaves_an_unobserved_direction_alone():
  """A direction the operator cannot see must keep its prior spread."""
  covariance = np.diag([1.0, 4.0])
  operator = np.array([[1.0, 0.0]])
  rng = np.random.default_rng(5)
  members = rng.multivariate_normal(np.zeros(2), covariance, size=40000)
  analysis = enkf_analysis(members, np.array([0.5]), operator, 0.2, rng=rng)
  assert abs(analysis.covariance[1, 1] - 4.0) / 4.0 < 0.05
  assert analysis.covariance[0, 0] < 0.2


@pytest.mark.parametrize(
  ("members", "operator", "noise", "message"),
  [
    (np.zeros((4, 3)), np.zeros((2, 5)), 1.0, "operator must be 2-D with 3 columns"),
    (np.zeros(4), np.zeros((2, 4)), 1.0, "members must be a 2-D array"),
    (np.zeros((1, 3)), np.zeros((2, 3)), 1.0, "at least two members"),
    (np.zeros((4, 3)), np.zeros((2, 3)), np.ones(5), "noise must supply 2 deviations"),
  ],
)
def test_the_analysis_refuses_malformed_input(members, operator, noise, message):
  with pytest.raises(ValueError, match=message):
    enkf_analysis(members, np.zeros(2), operator, noise)


def _flow():
  import pyimr
  from _validation_support import NHKV, R0, REQ

  problem = pyimr.prepare(pyimr.SimulationConfig(R0, REQ, NHKV, rtol=1e-11, atol=1e-13))
  times = np.linspace(0.0, 15e-6, 31)
  truth = np.asarray(problem.initial_state, dtype=float)
  operator = np.array([[1.0, 0.0]])  # radius only; velocity is never observed
  return problem, times, truth, operator


def test_the_variational_gradient_is_exact(measured):
  """The gradient is the whole claim. An einsum that contracts the wrong index of the

  tangent operator transposes it, which agrees with a difference quotient in the first
  component and is 52% wrong in the second -- so this compares every component.
  """
  from pyimr.assimilation import variational_cost

  problem, times, truth, operator = _flow()
  observations = problem.solve_states(times, truth) @ operator.T
  precision = np.diag([1e4, 1e2])
  background = truth + np.array([2e-3, 5e-3])

  _, gradient = variational_cost(problem, times, background, observations, operator, 1e-4, background, precision)

  step = 1e-6
  difference = np.zeros_like(background)
  for index in range(background.size):
    offset = np.zeros_like(background)
    offset[index] = step
    ahead, _ = variational_cost(problem, times, background + offset, observations, operator, 1e-4, background, precision)
    behind, _ = variational_cost(problem, times, background - offset, observations, operator, 1e-4, background, precision)
    difference[index] = (ahead - behind) / (2.0 * step)

  error = float(np.linalg.norm(gradient - difference) / np.linalg.norm(difference))
  measured("4D-Var gradient", f"rel={error:.2e}")
  assert error < 1e-7, f"gradient {gradient} vs difference {difference}"


def test_four_dvar_recovers_an_initial_state_it_never_observed(measured):
  """Radius alone is observed, yet the wall velocity comes back too -- the flow couples them."""
  from pyimr.assimilation import four_dvar

  problem, times, truth, operator = _flow()
  observations = problem.solve_states(times, truth) @ operator.T
  background = truth + np.array([5e-3, 2e-2])

  analysis = four_dvar(problem, times, observations, operator, 1e-5, background, np.diag([1e-6, 1e-6]), maximum_iterations=60)

  before = np.abs(background - truth)
  after = np.abs(analysis.state - truth)
  measured("4D-Var recovery", f"|err| {before.max():.1e} -> {after.max():.1e} in {analysis.iterations} iters")
  assert analysis.success
  assert after.max() < 1e-10, f"analysis still {after} from the truth"
  assert after.max() < before.max() / 1e6


def test_a_tight_background_holds_the_analysis_near_it():
  """With observations this weak the background term has to dominate, or the prior is decoration."""
  from pyimr.assimilation import four_dvar

  problem, times, truth, operator = _flow()
  observations = problem.solve_states(times, truth) @ operator.T
  background = truth + np.array([5e-3, 2e-2])

  analysis = four_dvar(problem, times, observations, operator, 1e6, background, np.diag([1e12, 1e12]), maximum_iterations=40)
  assert np.abs(analysis.state - background).max() < 1e-6 * np.abs(background - truth).max()


def test_the_cost_refuses_observations_that_do_not_line_up():
  from pyimr.assimilation import variational_cost

  problem, times, truth, operator = _flow()
  with pytest.raises(ValueError, match="observations must be"):
    variational_cost(problem, times, truth, np.zeros((3, 1)), operator, 1e-4, truth, np.eye(2))
  with pytest.raises(ValueError, match="observations must have 1 columns"):
    variational_cost(problem, times, truth, np.zeros((times.size, 2)), operator, 1e-4, truth, np.eye(2))


def test_a_row_of_nan_observations_is_skipped_rather_than_poisoning_the_cost():
  """Gaps in a record are normal. A NaN row must contribute nothing, not NaN."""
  from pyimr.assimilation import variational_cost

  problem, times, truth, operator = _flow()
  observations = problem.solve_states(times, truth) @ operator.T
  background = truth + np.array([1e-3, 1e-3])
  precision = np.diag([1e2, 1e2])

  full_cost, full_gradient = variational_cost(problem, times, background, observations, operator, 1e-4, background, precision)
  gapped = observations.copy()
  gapped[5] = np.nan
  gap_cost, gap_gradient = variational_cost(problem, times, background, gapped, operator, 1e-4, background, precision)

  assert np.isfinite(gap_cost) and np.all(np.isfinite(gap_gradient))
  assert gap_cost < full_cost, "dropping an observation cannot increase the misfit"
  # and dropping one of 31 should move it a little, not wipe it out
  assert gap_cost > 0.5 * full_cost


@pytest.mark.slow
def test_exact_gradients_beat_an_ensemble_smoother_on_known_truth(measured):
  """The claim from Spratt et al. that motivates building this here rather than porting:

  gradient-based estimation beats a plain ensemble one. Both methods get the same
  observations, background, background covariance and observation noise. The only
  difference is where the state-to-observation relationship comes from -- the exact
  tangent operator, or ensemble statistics.
  """
  from pyimr.assimilation import ensemble_smoother, four_dvar

  problem, times, truth, operator = _flow()
  sigma = 2e-4
  background_sd = np.array([5e-3, 2e-2])
  precision = np.diag(1.0 / background_sd**2)
  clean = problem.solve_states(times, truth) @ operator.T

  variational, ensemble, prior = [], [], []
  for trial in range(4):
    rng = np.random.default_rng(100 + trial)
    observations = clean + rng.normal(0.0, sigma, size=clean.shape)
    background = truth + rng.normal(0.0, background_sd)
    prior.append(np.abs(background - truth))

    analysis = four_dvar(problem, times, observations, operator, sigma**2, background, precision, maximum_iterations=80)
    variational.append(np.abs(analysis.state - truth))

    members = background + rng.normal(0.0, background_sd, size=(64, truth.size))
    smoothed = ensemble_smoother(problem, times, members, observations, operator, sigma**2, rng=rng)
    ensemble.append(np.abs(smoothed.mean - truth))

  prior, variational, ensemble = np.array(prior), np.array(variational), np.array(ensemble)
  rms = lambda values, column: float(np.sqrt((values[:, column] ** 2).mean()))  # noqa: E731
  measured(
    "4D-Var vs ensemble smoother",
    f"velocity rms: prior={rms(prior, 1):.1e} var={rms(variational, 1):.1e} ens={rms(ensemble, 1):.1e}",
  )

  assert rms(variational, 0) < rms(prior, 0) / 10.0, "4D-Var must improve substantially on the prior"
  assert rms(ensemble, 0) < rms(prior, 0) / 10.0, "the ensemble smoother must also be a real method here"
  # the unobserved component is where the exact operator pays: it is reached only through
  # the coupling, which one ensemble linearisation across the window represents poorly
  assert rms(variational, 1) < rms(ensemble, 1) / 5.0
