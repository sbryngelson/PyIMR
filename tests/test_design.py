"""Laplace/Fisher expected information gain (#25, piece 3).

What is checked here is the criterion, not the Jacobian: `J` comes from
`PreparedInference` and is verified against exact tangents in the sensitivity
suite. So these tests pin the contraction, its exact scalings, and the one
qualitative property the linearisation must not violate.
"""

import numpy as np
import pytest

import imr_design
import imr_fast
from _validation_support import R0, REQ
from imr_inference import InferenceParameter

SECTION = "6. Experiment design"

_TIMES = np.linspace(0.0, 20e-6, 40)
_NOISE = 5e-7
_PARAMETERS = (
  InferenceParameter("material.shear_modulus_pa", 1500.0, 4000.0),
  InferenceParameter("material.viscosity_pa_s", 0.05, 0.2),
)


def _design(times=_TIMES, noise=_NOISE):
  config = imr_fast.SimulationConfig(R0, REQ, imr_fast.NeoHookeanKelvinVoigt(2500.0, 0.1))
  return imr_design.design_inference(config, times, noise, _PARAMETERS)


@pytest.fixture(scope="module")
def design():
  return _design()


def test_gain_matches_the_closed_form(design, measured):
  """One draw, computed independently from the Jacobian the design layer used."""
  unit = np.random.default_rng(0).random((1, design.size))[0]
  jacobian = design.jacobian(unit)
  expected = 0.5 * np.linalg.slogdet(np.eye(design.size) + imr_design.UNIFORM_VARIANCE * (jacobian.T @ jacobian))[1]

  result = imr_design.expected_information_gain(design, draws=1)
  error = abs(result.expected_information_gain - expected) / abs(expected)
  measured("EIG vs closed form", f"{result.expected_information_gain:.6f} vs {expected:.6f}, rel={error:.2e}")
  assert error < 1e-12
  assert result.draws == 1 and result.failures == 0


def test_halving_the_noise_scales_the_information_by_four(design, measured):
  """`jacobian` carries a 1/sigma, so the Fisher information is exactly
  quadratic in the noise level. That makes the sigma dependence checkable
  without a second solve, which is the cheapest real test available here."""
  unit = np.random.default_rng(0).random((1, design.size))[0]
  jacobian = design.jacobian(unit)
  scaled = np.eye(design.size) + 4.0 * imr_design.UNIFORM_VARIANCE * (jacobian.T @ jacobian)
  expected = 0.5 * np.linalg.slogdet(scaled)[1]

  result = imr_design.expected_information_gain(_design(noise=_NOISE / 2.0), draws=1)
  error = abs(result.expected_information_gain - expected) / abs(expected)
  measured("EIG at sigma/2", f"{result.expected_information_gain:.6f} vs {expected:.6f}, rel={error:.2e}")
  assert error < 1e-12


def test_more_observations_cannot_reduce_the_gain(measured):
  """Adding observation times adds positive semidefinite terms to `J^T J`, so
  the gain is monotone. A criterion that violated this would be misreporting the
  information, not making a modelling choice."""
  sparse = imr_design.expected_information_gain(_design(times=_TIMES[::2]), draws=6)
  dense = imr_design.expected_information_gain(_design(), draws=6)
  measured("EIG 20 vs 40 frames", f"{sparse.expected_information_gain:.3f} -> {dense.expected_information_gain:.3f}")
  assert dense.expected_information_gain > sparse.expected_information_gain


def test_the_prior_average_is_not_the_nominal_value(design, measured):
  """The concern this criterion has to answer for: it is linearised, so a score
  computed at one nominal theta inherits that tangent's blind spots.

  Averaging over the prior is the mitigation, and this measures how much it
  matters -- the per-draw gains spread widely enough that a single nominal
  evaluation is not a usable substitute. That is a bound on the nominal-tangent
  shortcut, not a defect in the averaged criterion.
  """
  gains = [
    imr_design._gain(design, unit, np.full(design.size, imr_design.UNIFORM_VARIANCE))
    for unit in np.random.default_rng(0).random((8, design.size))
  ]
  centre = imr_design._gain(design, np.full(design.size, 0.5), np.full(design.size, imr_design.UNIFORM_VARIANCE))
  spread = float(np.std(gains)) / float(np.mean(gains))
  measured("per-draw EIG spread", f"mean={np.mean(gains):.3f} sd/mean={spread:.3f} centre={centre:.3f}")
  assert spread > 0.02


def test_a_tighter_prior_yields_less_to_learn(design, measured):
  """EIG is what the experiment adds. Shrinking the prior variance shrinks it."""
  wide = imr_design.expected_information_gain(design, draws=4)
  tight = imr_design.expected_information_gain(design, draws=4, prior_variance=imr_design.UNIFORM_VARIANCE / 100.0)
  measured("EIG vs prior width", f"{wide.expected_information_gain:.3f} -> {tight.expected_information_gain:.3f}")
  assert tight.expected_information_gain < wide.expected_information_gain


def test_failed_draws_are_dropped_and_counted(design, monkeypatch):
  """A failure must not enter the average as a zero, which would read as a
  design that runs fine and teaches nothing."""
  real, calls = imr_design._gain, []

  def flaky(inference, unit, variance):
    calls.append(unit)
    if len(calls) % 2:
      raise RuntimeError("stiff solve failed")
    return real(inference, unit, variance)

  monkeypatch.setattr(imr_design, "_gain", flaky)
  result = imr_design.expected_information_gain(design, draws=4)
  assert (result.draws, result.failures) == (2, 2)
  assert np.isfinite(result.expected_information_gain)


def test_every_draw_failing_raises_rather_than_returning_a_number(design, monkeypatch):
  monkeypatch.setattr(imr_design, "_gain", lambda *_: (_ for _ in ()).throw(RuntimeError("no")))
  with pytest.raises(RuntimeError, match="every design draw failed"):
    imr_design.expected_information_gain(design, draws=2)


@pytest.mark.parametrize(
  ("kwargs", "error"),
  (({"draws": 0}, ValueError), ({"workers": 0}, ValueError), ({"prior_variance": -1.0}, ValueError)),
)
def test_invalid_arguments_are_rejected(design, kwargs, error):
  with pytest.raises(error):
    imr_design.expected_information_gain(design, **kwargs)


def test_foreign_input_is_rejected():
  with pytest.raises(TypeError, match="PreparedInference"):
    imr_design.expected_information_gain(object())
