"""Splitting a residual into what the model misses and what the apparatus cannot repeat.

`chi_squared/N` is routinely read as "fits to within the noise" and cannot answer that.
`check_residuals` attacks it from the correlation side; this attacks it from the replicate
side, and needs no assumption about the model at all -- the spread between repeated runs is
error by construction.

The gates that matter are the two closed forms. A perfect model must give exactly zero, and a
constant offset of one pure-error standard deviation must give exactly `J k / (k - p)` --
which pins the factor of `J`, the piece a chi-squared against the between-run spread silently
omits.
"""

import numpy as np
import pytest

from pyimr.noise import lack_of_fit

SETTINGS, TRIALS, PARAMETERS = 60, 12, 2
TRUTH = np.linspace(0.0, 1.0, SETTINGS)
SPREAD = np.full(SETTINGS, 0.1)


def test_a_perfect_model_has_no_lack_of_fit():
  got = lack_of_fit(TRUTH, TRUTH, SPREAD, TRIALS, PARAMETERS)
  assert got.ratio == 0.0
  assert got.pure_df == SETTINGS * (TRIALS - 1)
  assert got.lack_df == SETTINGS - PARAMETERS


def test_a_one_sigma_offset_recovers_the_replicate_factor():
  """The closed form that pins `J`.

  A constant offset of one pure-error standard deviation gives
  `SS_lack/df = J k s^2/(k-p)` against `MS_pure = s^2`, so the ratio is exactly
  `J k/(k-p)` -- here `12 x 60/58 = 12.414`. Getting `J` wrong is the specific error this
  statistic exists to prevent, so it is checked rather than assumed.
  """
  got = lack_of_fit(TRUTH + 0.1, TRUTH, SPREAD, TRIALS, PARAMETERS)
  expected = TRIALS * SETTINGS / (SETTINGS - PARAMETERS)
  assert got.ratio == pytest.approx(expected, rel=1e-12)
  assert got.ratio == pytest.approx(12.4137931, rel=1e-6)


@pytest.mark.parametrize("scale", [0.5, 2.0, 10.0])
def test_it_is_scale_free_in_the_residual(scale):
  """Doubling both the misfit and the spread must leave the ratio alone."""
  base = lack_of_fit(TRUTH + 0.1, TRUTH, SPREAD, TRIALS, PARAMETERS)
  scaled = lack_of_fit(TRUTH + 0.1 * scale, TRUTH, SPREAD * scale, TRIALS, PARAMETERS)
  assert scaled.ratio == pytest.approx(base.ratio, rel=1e-12)


def test_more_replicates_give_more_power():
  """`J` belongs in the numerator, so the same relative misfit is more significant with more
  repeats. This is why the three gelatin records -- 18, 14 and 7 trials -- are not directly
  comparable on `F` alone.
  """
  ratios = [lack_of_fit(TRUTH + 0.1, TRUTH, SPREAD, trials, PARAMETERS).ratio
            for trials in (4, 8, 16)]
  assert ratios[1] == pytest.approx(2.0 * ratios[0], rel=1e-12)
  assert ratios[2] == pytest.approx(2.0 * ratios[1], rel=1e-12)


def test_tighter_repeats_raise_the_ratio_at_fixed_misfit():
  loose = lack_of_fit(TRUTH + 0.1, TRUTH, SPREAD, TRIALS, PARAMETERS)
  tight = lack_of_fit(TRUTH + 0.1, TRUTH, SPREAD / 2.0, TRIALS, PARAMETERS)
  assert tight.ratio == pytest.approx(4.0 * loose.ratio, rel=1e-12)


def test_the_verdict_reads_the_right_way():
  assert "consistent with pure error" in str(lack_of_fit(TRUTH, TRUTH, SPREAD, TRIALS, PARAMETERS))
  assert "pure error" in str(lack_of_fit(TRUTH + 0.1, TRUTH, SPREAD, TRIALS, PARAMETERS))
  assert "misses structure" in str(lack_of_fit(TRUTH + 0.1, TRUTH, SPREAD, TRIALS, PARAMETERS))


def test_it_reproduces_the_gelatin_numbers():
  """The three records, from the sums `docs/writeup/lackoffit.py` reports.

  Reconstructed from the published mean squares rather than refitting, so this is a check of
  the statistic's arithmetic against an independent implementation of the same formula.
  """
  for settings, trials, pure, lack, expected in [
    (201, 18, 5.198e-4, 7.362e-3, 14.16),
    (192, 14, 1.861e-3, 5.469e-3, 2.94),
    (198, 7, 1.913e-3, 6.233e-3, 3.26),
  ]:
    # a flat spread and a flat offset reproducing those mean squares exactly
    spread = np.full(settings, np.sqrt(pure))
    offset = np.sqrt(lack * (settings - 4) / (trials * settings))
    got = lack_of_fit(np.full(settings, offset), np.zeros(settings), spread, trials, 4)
    assert got.ratio == pytest.approx(expected, rel=2e-3)


@pytest.mark.parametrize(("kwargs", "message"), [
  ({"trials": 1}, "at least 2"),
  ({"trials": 2.5}, "at least 2"),
  ({"parameters": -1}, "non-negative"),
])
def test_impossible_arguments_are_refused(kwargs, message):
  call = dict(observed=TRUTH, predicted=TRUTH, spread=SPREAD, trials=TRIALS, parameters=PARAMETERS)
  call.update(kwargs)
  with pytest.raises(ValueError, match=message):
    lack_of_fit(**call)


def test_a_model_with_no_room_left_is_refused():
  """`k <= p` leaves the numerator no degrees of freedom, and the ratio would be a division
  by a negative count rather than an error.
  """
  with pytest.raises(ValueError, match="degrees of freedom"):
    lack_of_fit(TRUTH[:4], TRUTH[:4], SPREAD[:4], TRIALS, 4)


def test_mismatched_lengths_are_refused():
  with pytest.raises(ValueError, match="agree in length"):
    lack_of_fit(TRUTH, TRUTH[:-1], SPREAD, TRIALS, PARAMETERS)


def test_a_negative_spread_is_refused():
  bad = SPREAD.copy()
  bad[3] = -1.0
  with pytest.raises(ValueError, match="non-negative"):
    lack_of_fit(TRUTH, TRUTH, bad, TRIALS, PARAMETERS)
