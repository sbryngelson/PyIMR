"""Designing for the models still in question, and only those.

Two ways a discrimination criterion wastes a batch, at opposite ends. It can spend runs
separating a model the existing data has already settled; or it can be handed rivals so far
apart that a local criterion misprices them. `screen_models` handles the first directly and
the second as a consequence -- models that are far apart are easy to separate, so they end up
decided and drop out, and what survives is the close pairs a derivative criterion is right for.

`augmented_information` is then the design object: the model choice as a continuous coordinate,
`R(theta, eps) = (1-eps) R_A + eps R_B`, so `dR/deps = R_B - R_A` is one more Jacobian column.
No evidence integral, no prior on the rival, nothing to sample -- which matters because the
sampled version failed on this package's own records at one effective draw.
"""

import numpy as np
import pytest

from pyimr.discriminate import screen_models
from pyimr.measure import augmented_information, separability

SAMPLES = 200
GRID = np.linspace(0.0, 3.0, SAMPLES)
JACOBIAN = np.column_stack([np.sin(GRID), np.cos(GRID)])
INSIDE = JACOBIAN @ np.array([0.7, -0.3])          # reproducible by refitting the material
OUTSIDE = 0.2 * np.sin(np.linspace(0.0, 31.0, SAMPLES))


# --- screening ---------------------------------------------------------------------------

def test_a_decided_model_gets_no_weight():
  """The whole point: a rival the data has settled is not a design target."""
  screen = screen_models([2173.5, 2071.9, 2170.0])
  assert screen.decided.tolist() == [False, True, False]
  assert screen.weights[1] == 0.0
  assert screen.undecided == 2
  assert screen.best == 0


def test_the_survivors_carry_the_posterior():
  """Weights are the posterior over what is left, so a barely-live rival cannot dominate."""
  screen = screen_models([100.0, 96.5])
  assert screen.weights.sum() == pytest.approx(1.0)
  assert screen.weights[0] > screen.weights[1]
  assert screen.weights[1] / screen.weights[0] == pytest.approx(np.exp(-3.5), rel=1e-9)


def test_everything_within_the_margin_stays_live():
  screen = screen_models([10.0, 9.0, 8.0, 6.0], decisive=5.0)
  assert not screen.decided.any()
  assert screen.undecided == 4


def test_a_huge_gap_does_not_overflow():
  """Far-apart rivals are the case this must not blow up on -- and they drop out, which is
  the intended behaviour, not a limitation: an easy question needs no experiment.
  """
  screen = screen_models([0.0, -5e4, -1e5])
  assert screen.decided.tolist() == [False, True, True]
  assert np.all(np.isfinite(screen.weights))
  assert screen.weights[0] == pytest.approx(1.0)


def test_the_floor_keeps_a_live_model_from_vanishing():
  """The floor must survive normalisation, which is where the first attempt lost it."""
  screen = screen_models([0.0, -4.9], decisive=5.0, floor=1e-2)
  assert not screen.decided[1]
  assert screen.weights[1] >= 1e-2, "a live model must not round to zero"
  assert screen.weights.sum() == pytest.approx(1.0)
  # a floor that cannot be met is an error rather than a silently violated promise
  with pytest.raises(ValueError, match="floor"):
    screen_models([0.0, -1.0, -2.0, -3.0], floor=0.4)


@pytest.mark.parametrize(("bad", "message"), [
  ({"log_evidence": []}, "at least one"),
  ({"log_evidence": [1.0, np.nan]}, "finite"),
  ({"decisive": 0.0}, "decisive"),
  ({"floor": 1.0}, "floor"),
])
def test_impossible_screens_are_refused(bad, message):
  call = {"log_evidence": [1.0, 2.0], "decisive": 5.0, "floor": 1e-3}
  call.update(bad)
  with pytest.raises(ValueError, match=message):
    screen_models(call.pop("log_evidence"), **call)


# --- the augmented information ------------------------------------------------------------

def test_a_difference_the_material_absorbs_is_invisible():
  """The statement the whole absorption argument rests on, as a design quantity.

  A model difference lying in the span of `dR/dtheta` is reproduced by refitting, so no design
  can see it and the coordinate is unidentified. That must come back as infinite variance
  rather than a large finite number, because a large number invites a design that chases it.
  """
  variance, absorbed = separability(JACOBIAN, INSIDE[None, :])
  assert variance[0] == float("inf")
  assert absorbed[0] == pytest.approx(1.0, abs=1e-9)


def test_a_difference_outside_the_span_is_determined():
  variance, absorbed = separability(JACOBIAN, OUTSIDE[None, :])
  assert np.isfinite(variance[0]) and variance[0] > 0.0
  assert absorbed[0] < 0.1


def test_weighting_scales_the_information_by_the_posterior():
  """A rival at a tenth of the posterior contributes a tenth of the information, so its
  coordinate's variance is ten times larger. That is what makes a nearly-settled model stop
  driving the design without anyone choosing a cutoff.
  """
  full, _ = separability(JACOBIAN, OUTSIDE[None, :], weights=[1.0])
  tenth, _ = separability(JACOBIAN, OUTSIDE[None, :], weights=[0.1])
  assert tenth[0] == pytest.approx(10.0 * full[0], rel=1e-9)


def test_a_zero_weight_removes_the_rival_entirely():
  variance, _ = separability(JACOBIAN, OUTSIDE[None, :], weights=[0.0])
  assert variance[0] == float("inf")


def test_the_augmented_matrix_is_the_right_shape_and_symmetric():
  information = augmented_information(JACOBIAN, np.vstack([INSIDE, OUTSIDE]))
  assert information.shape == (4, 4)
  assert np.allclose(information, information.T)
  # the material block is untouched by augmenting
  assert np.allclose(information[:2, :2], JACOBIAN.T @ JACOBIAN)


def test_it_is_linear_in_the_design_so_the_certificate_applies():
  """`optimal_measure` requires the information to average over the measure. Two designs
  stacked must give the sum of their matrices, or none of the convexity argument holds.
  """
  other = np.column_stack([np.sin(2 * GRID), np.cos(2 * GRID)])
  difference = 0.1 * np.cos(5 * GRID)
  combined = augmented_information(np.vstack([JACOBIAN, other]),
                                   np.concatenate([OUTSIDE, difference])[None, :])
  parts = (augmented_information(JACOBIAN, OUTSIDE[None, :])
           + augmented_information(other, difference[None, :]))
  assert np.allclose(combined, parts)


@pytest.mark.parametrize(("bad", "message"), [
  ({"jacobian": np.ones(5)}, "samples, p"),
  ({"differences": np.ones((1, 5))}, "samples"),
  ({"weights": [1.0, 2.0]}, "one per rival"),
  ({"weights": [-1.0]}, "non-negative"),
])
def test_impossible_augmentations_are_refused(bad, message):
  call = {"jacobian": JACOBIAN, "differences": OUTSIDE[None, :], "weights": None}
  call.update(bad)
  with pytest.raises(ValueError, match=message):
    augmented_information(call.pop("jacobian"), call.pop("differences"), **call)
