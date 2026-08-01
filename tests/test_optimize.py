"""Design search: validated on known optima before it is pointed at anything expensive.

A search over an expensive objective is exactly the kind of instrument that fails quietly
-- an earlier grid scan in this project reported a peak 8000 log units BELOW the truth and
looked entirely plausible. So every test here has an answer known independently of the
optimiser, and one of them checks it beats the cheap alternative it exists to replace.
"""

import numpy as np
import pytest

from pyimr.optimize import bayesian_maximize

SECTION = "12. Design search"

_BOX = [(-1.0, 1.0), (-1.0, 1.0)]


def _bowl(centre):
  return lambda point: -float(np.sum((np.asarray(point) - centre) ** 2))


def test_it_finds_a_known_optimum_in_one_dimension(measured):
  result = bayesian_maximize(_bowl(np.array([0.3])), [(-1.0, 1.0)], evaluations=20, initial=5, seed=0)
  error = abs(result.best_point[0] - 0.3)
  measured("1-D recovery", f"|x*-x|={error:.1e}")
  assert error < 0.02, f"landed at {result.best_point[0]:.4f}, truth 0.3"


def test_it_finds_a_known_optimum_in_two_dimensions(measured):
  truth = np.array([0.2, -0.4])
  result = bayesian_maximize(_bowl(truth), _BOX, evaluations=30, initial=6, seed=1)
  error = float(np.linalg.norm(result.best_point - truth))
  measured("2-D recovery", f"|x*-x|={error:.1e}")
  assert error < 0.03, f"landed at {result.best_point}, truth {truth}"


@pytest.mark.slow
def test_it_beats_random_search_on_the_same_budget(measured):
  """The justification for the whole surrogate. If it cannot beat drawing points at

  random, the right answer is to draw points at random.
  """
  def objective(point):
    x, y = point
    return -((x - 0.35) ** 2 + (y + 0.6) ** 2) + 0.35 * np.sin(6.0 * x) * np.sin(6.0 * y)

  budget = 30
  searched, sampled = [], []
  for seed in range(6):
    searched.append(bayesian_maximize(objective, _BOX, evaluations=budget, initial=6, seed=seed).best_value)
    draws = np.random.default_rng(seed).random((budget, 2)) * 2.0 - 1.0
    sampled.append(max(objective(point) for point in draws))

  searched, sampled = np.array(searched), np.array(sampled)
  measured("vs random search", f"BO {searched.mean():.3f} vs random {sampled.mean():.3f} over {len(searched)} trials")
  assert searched.mean() > sampled.mean() + 0.05, "a surrogate that does not beat random sampling is not worth its complexity"
  assert (searched > sampled).sum() >= 4


@pytest.mark.slow
def test_a_reported_error_bar_keeps_it_from_chasing_luck(measured):
  """`expected_information_gain` returns a standard error, so the surrogate is told how

  much to trust each score. Without that a point that scored well by chance is pursued.
  """
  rng = np.random.default_rng(0)
  truth, sigma = np.array([0.3, -0.5]), 0.05

  def noisy(point):
    return -float(np.sum((np.asarray(point) - truth) ** 2)) + rng.normal(0.0, sigma), sigma

  errors = [
    float(np.linalg.norm(bayesian_maximize(noisy, _BOX, evaluations=30, initial=8, seed=seed).best_point - truth))
    for seed in range(4)
  ]
  measured("noisy objective", f"mean |x*-x|={np.mean(errors):.3f} at sigma={sigma}")
  assert np.mean(errors) < 0.2, f"noise pulled the answer to {np.mean(errors):.3f} away"


def test_the_result_records_everything_it_evaluated():
  result = bayesian_maximize(_bowl(np.array([0.0, 0.0])), _BOX, evaluations=12, initial=4, seed=0)
  assert result.points.shape == (12, 2)
  assert result.values.shape == (12,) and result.deviations.shape == (12,)
  assert result.best_value == pytest.approx(result.values.max())
  np.testing.assert_allclose(result.best_point, result.points[int(result.values.argmax())])
  assert np.all(result.points >= -1.0) and np.all(result.points <= 1.0), "no evaluation may leave the box"


@pytest.mark.parametrize(
  ("bounds", "kwargs", "message"),
  [
    ([(0.0, 1.0, 2.0)], {}, "bounds must be"),
    ([(1.0, 0.0)], {}, "every upper bound must exceed"),
    ([(0.0, 1.0)], {"initial": 1}, "at least two initial evaluations"),
    ([(0.0, 1.0)], {"initial": 6, "evaluations": 3}, "evaluations must cover"),
  ],
)
def test_it_refuses_a_malformed_search(bounds, kwargs, message):
  with pytest.raises(ValueError, match=message):
    bayesian_maximize(lambda point: 0.0, bounds, **kwargs)


@pytest.mark.slow
def test_design_search_runs_against_real_expected_information_gain(measured):
  """The wrapper, end to end. Validated by consistency rather than a known optimum: the

  winning design is re-scored directly, and must reproduce the value the search recorded.
  A search that reports a number its own objective does not agree with is the failure
  worth catching here.
  """
  import pyimr
  from pyimr.design import design_inference, expected_information_gain
  from pyimr.inference import InferenceParameter
  from pyimr.optimize import optimize_design
  from _validation_support import NHKV, R0, REQ

  times = np.linspace(0.0, 20e-6, 30)
  parameters = (InferenceParameter("material.shear_modulus_pa", 2000.0, 3000.0),)

  def build(design):
    return design_inference(pyimr.SimulationConfig(R0, REQ, NHKV, pA=float(design[0])), times, 1e-8, parameters)

  result = optimize_design(build, [(0.0, 8e4)], draws=8, evaluations=6, initial=3, seed=0)

  assert result.values.shape == (6,)
  assert np.all(result.deviations >= 0.0), "every EIG must come back with an error bar"
  assert result.best_value >= result.values[:3].max(), "the search cannot end below its own initial design"

  rescored = expected_information_gain(build(result.best_point), draws=8, seed=0)
  measured("design search", f"best pA={result.best_point[0]:.3g} Pa  EIG={result.best_value:.3f} nats")
  assert rescored.expected_information_gain == pytest.approx(result.best_value, rel=1e-9)
