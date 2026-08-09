"""Expected information gain as the one criterion, and the certificate it keeps.

The gates are the ones that can fail quietly: a gain that is not the mutual information it
claims to be, a portfolio that is not the weighted sum of its parts, and a measure whose gap
is reported as a proof without being one. Each is checked against a closed form.
"""

import numpy as np
import pytest

from pyimr.gain import Question, expected_gain, gain_criterion, lack_of_fit_degrees
from pyimr.measure import apportion, optimal_measure, sensitivity


def information(rows, size, rng):
  jacobian = rng.standard_normal((rows, size))
  return jacobian.T @ jacobian


@pytest.fixture
def rng():
  return np.random.default_rng(0)


def test_the_gain_is_the_mutual_information_it_claims_to_be(rng):
  """`U_A = -(1/2) log det` of the posterior covariance restricted to `A`.

  The prior is unit in these coordinates, so its own log determinant is zero and the whole
  gain is the posterior's. Checked against the explicit inverse rather than the Schur form
  the implementation uses, so an algebra slip in the projector cannot pass.
  """
  matrix = information(30, 7, rng)
  wanted = (4, 5)
  posterior = np.linalg.inv(np.eye(7) + matrix)
  closed = -0.5 * np.linalg.slogdet(posterior[np.ix_(wanted, wanted)])[1]
  got = expected_gain(matrix, [Question("model", wanted)])
  assert got.per_question["model"] == pytest.approx(closed, rel=1e-12)


def test_asking_about_everything_is_bayesian_d_optimality(rng):
  matrix = information(30, 5, rng)
  got = expected_gain(matrix, [Question("all", tuple(range(5)))])
  assert got.total == pytest.approx(0.5 * np.linalg.slogdet(np.eye(5) + matrix)[1], rel=1e-12)


def test_a_design_that_determines_nothing_still_scores(rng):
  """`log det M` refuses a rank-deficient design; this is the failure that criterion has.

  22 of 48 candidate geometries were unscorable under the sampled criterion. A design that
  cannot pin every parameter is not an error, it is a bad design, and the difference matters
  because the optimiser has to be able to rank it.
  """
  thin = information(2, 6, rng)                       # rank 2 in six coordinates
  assert np.linalg.slogdet(thin)[0] <= 0, "the premise: log det is not available here"
  got = expected_gain(thin, [Question("model", (4, 5))])
  assert np.isfinite(got.total) and got.total > 0.0


def test_a_portfolio_is_the_weighted_sum_of_its_questions(rng):
  matrix = information(30, 6, rng)
  material, model = Question("material", (0, 1, 2)), Question("model", (4, 5), weight=3.0)
  alone = {q.name: expected_gain(matrix, [q]).per_question[q.name] for q in (material, model)}
  together = expected_gain(matrix, [material, model])
  # `per_question` is unweighted nats and does not depend on the company it keeps; `total`
  # is where the weights act, which is the split a caller has to be able to rely on
  assert together.per_question == pytest.approx(alone)
  assert together.total == pytest.approx(alone["material"] + 3.0 * alone["model"], rel=1e-12)
  # weight zero drops a question's influence without removing it from the report
  muted = expected_gain(matrix, [material, Question("model", (4, 5), weight=0.0)])
  assert muted.total == pytest.approx(alone["material"], rel=1e-12)
  assert set(muted.per_question) == {"material", "model"}
  # the criterion `optimal_measure` optimises must be the number reported here, weights and all
  assert gain_criterion([material, model])(matrix)[0] == pytest.approx(together.total, rel=1e-12)


def test_the_criterion_is_concave_so_the_certificate_means_something(rng):
  """Concavity is what the equivalence theorem rests on, and it is assumed, so it is checked.

  A chord below the function at any midpoint would make the reported gap a convergence report
  rather than a proof, which is the exact distinction this package exists to keep.
  """
  criterion = gain_criterion([Question("model", (4, 5))])
  for _ in range(50):
    a, b = information(20, 6, rng), information(20, 6, rng)
    t = rng.uniform(0.1, 0.9)
    assert criterion(t * a + (1 - t) * b)[0] >= t * criterion(a)[0] + (1 - t) * criterion(b)[0] - 1e-9


def test_the_gradient_is_the_derivative_it_is_used_as(rng):
  """`tr[G dM]` against a central difference: the multiplicative update is built on this."""
  criterion = gain_criterion([Question("model", (3, 4), weight=2.5), Question("material", (0, 1))])
  base, step = information(20, 5, rng), information(20, 5, rng)
  gradient = criterion(base)[1]
  h = 1e-6
  finite = (criterion(base + h * step)[0] - criterion(base - h * step)[0]) / (2 * h)
  assert float(np.sum(gradient * step)) == pytest.approx(finite, rel=1e-6)


def test_an_optimal_measure_under_the_gain_criterion_certifies(rng):
  """End to end: the generalised `optimal_measure` still returns a proof, not a stopping point.

  The gap is recomputed here from `sensitivity`'s own definition against the returned weights,
  so a criterion that silently reported its own convergence would not pass.
  """
  stack = np.array([information(12, 4, rng) for _ in range(9)])
  criterion = gain_criterion([Question("model", (2, 3))])
  got = optimal_measure(stack, criterion=criterion, tolerance=1e-11, iterations=200_000)
  assert got.certified, f"gap {got.gap:.2e}"
  gradient = criterion(np.tensordot(got.weights, stack, axes=(0, 0)))[1]
  direct = np.einsum("jk,ikj->i", gradient, stack)
  assert float(np.max(direct - got.weights @ direct)) <= 1e-6


def test_rounding_is_measured_in_the_criterion_it_was_optimised_under(rng):
  """A D-efficiency against a measure that maximised something else is not an efficiency.

  It read 1.21 and 1.40 on the operator design, which is arithmetic about two objectives. The
  integer batch is a feasible measure, so against the optimum its own criterion can only fall.
  """
  stack = np.array([information(12, 4, rng) for _ in range(9)])
  criterion = gain_criterion([Question("model", (2, 3))])
  measure = optimal_measure(stack, criterion=criterion, tolerance=1e-11, iterations=200_000)
  assert measure.certified
  matched = apportion(measure.weights, 12, stack, criterion=criterion)
  assert 0.0 < matched.efficiency <= 1.0 + 1e-9, matched.efficiency
  mismatched = apportion(measure.weights, 12, stack)
  assert mismatched.efficiency != pytest.approx(matched.efficiency), "the currencies differ"


def test_the_default_criterion_is_untouched(rng):
  """`optimal_measure` without a criterion must be the D-optimal answer it always was."""
  stack = np.array([information(12, 3, rng) for _ in range(7)])
  plain = optimal_measure(stack, tolerance=1e-10)
  assert plain.certified
  assert sensitivity(stack, plain.weights).max() <= 1e-6


def test_replicates_keep_a_batch_testable(rng):
  """A collapsed batch cannot detect that every model is wrong, and no criterion sees that.

  Ten runs on one setting has no pure error and no lack-of-fit numerator. The constraint is
  what buys both, and the D-efficiency is what it costs.
  """
  weights = [0.9, 0.07, 0.03]
  free = apportion(weights, 12)
  assert free.counts.min() == 1, "the premise: the free allocation nearly collapses"
  assert lack_of_fit_degrees(free.counts, parameters=2)[1] == 1        # one replicate: no pure error

  held = apportion(weights, 12, replicates=3)
  settings, replicates, lack_df, pure_df = lack_of_fit_degrees(held.counts, parameters=2)
  assert (settings, replicates) == (3, 3)
  assert lack_df == 1 and pure_df == 9, "a testable batch, which the free one was not"
  assert int(held.counts.sum()) == 12


def test_the_constraint_drops_settings_the_budget_cannot_replicate():
  got = apportion([0.4, 0.3, 0.2, 0.1], 6, replicates=3)
  assert int(np.count_nonzero(got.counts)) == 2 and int(got.counts.sum()) == 6
  assert list(np.flatnonzero(got.counts)) == [0, 1], "the measure's own ordering decides"
  with pytest.raises(ValueError, match="cannot give"):
    apportion([0.5, 0.5], 3, replicates=4)


@pytest.mark.parametrize(("bad", "message"), [
  ({"coordinates": ()}, "no coordinates"),
  ({"coordinates": (1, 1)}, "repeats a coordinate"),
  ({"weight": -1.0}, "non-negative"),
])
def test_impossible_questions_are_refused(bad, message):
  call = {"name": "q", "coordinates": (0, 1), "weight": 1.0} | bad
  with pytest.raises(ValueError, match=message):
    Question(**call)


def test_a_question_outside_the_coordinates_is_refused(rng):
  with pytest.raises(ValueError, match="outside the 3 available"):
    expected_gain(information(10, 3, rng), [Question("q", (0, 7))])
