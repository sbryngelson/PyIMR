"""Laplace/Fisher expected information gain (#25, piece 3).

What is checked here is the criterion, not the Jacobian: `J` comes from
`PreparedInference` and is verified against exact tangents in the sensitivity
suite. So these tests pin the contraction, its exact scalings, and the one
qualitative property the linearisation must not violate.
"""

import numpy as np
import pytest

from imr_fast import design as imr_design
import imr_fast
import imr_fast.inference
from _validation_support import R0, REQ
from imr_fast.inference import InferenceParameter

SECTION = "6. Experiment design"

_TIMES = np.linspace(0.0, 20e-6, 40)
_NOISE = 5e-7
_PARAMETERS = (InferenceParameter("material.shear_modulus_pa", 1500.0, 4000.0), InferenceParameter("material.viscosity_pa_s", 0.05, 0.2))


def _flaky(real, *, every):
  """Fail every `every`-th call, succeed otherwise."""
  calls = []

  def wrapper(inference, unit):
    calls.append(unit)
    if len(calls) % every:
      raise RuntimeError("stiff solve failed")
    return real(inference, unit)

  return wrapper


_TRUTH_UNIT = np.array([(2500.0 - 1500.0) / 2500.0, (0.1 - 0.05) / 0.15])

def _collapse_design(count=40, half_width=1e-6):
  """Half the frames inside +-1 us of the first collapse, half spread over 60 us.

  The first LOCAL minimum, not the deepest: at low viscosity a later rebound
  collapses deeper, and `argmin` picks that one instead -- a 32 us error where
  the real spread between designs is 2.2 us.
  """
  fine = np.linspace(0.0, 60e-6, 4000)
  trace = np.asarray(imr_fast.simulate(fine, imr_fast.SimulationConfig(R0, REQ, imr_fast.NeoHookeanKelvinVoigt(2500.0, 0.1))).radius_m)
  interior = np.flatnonzero((trace[1:-1] < trace[:-2]) & (trace[1:-1] <= trace[2:])) + 1
  collapse = float(fine[interior[0]])
  near = np.linspace(max(collapse - half_width, 1e-7), collapse + half_width, count // 2)
  return np.unique(np.concatenate([near, np.linspace(1e-7, 60e-6, count - count // 2)]))

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
    imr_design._gain(design, unit, np.full(design.size, imr_design.UNIFORM_VARIANCE)) for unit in np.random.default_rng(0).random((8, design.size))
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


def test_a_failed_draw_raises_by_default(design, monkeypatch):
  """Censoring is opt-in. A dropped draw makes the average estimate
  `E[gain | solve succeeded]`, which is not what the caller asked for and is not
  what the error bar describes -- so silence is the wrong default."""
  monkeypatch.setattr(imr_design, "_fisher", _flaky(imr_design._fisher, every=2))
  with pytest.raises(RuntimeError, match="max_failure_fraction"):
    imr_design.expected_information_gain(design, draws=4)


def test_allowed_failures_are_counted_and_warned(design, monkeypatch):
  """With censoring opted into, the result must still say so: `draws` is what
  was requested, `successful` is what reached the average."""
  monkeypatch.setattr(imr_design, "_fisher", _flaky(imr_design._fisher, every=2))
  with pytest.warns(RuntimeWarning, match="conditional on"):
    result = imr_design.expected_information_gain(design, draws=4, max_failure_fraction=0.75)
  assert (result.draws, result.successful, result.failures) == (4, 2, 2)
  assert np.isfinite(result.expected_information_gain)


def test_a_failure_chains_its_cause(design, monkeypatch):
  """The bare handler this replaces turned a stale signature or a bad parameter
  path -- programming errors -- into an unattributable failure count."""

  def broken(*_args):
    raise KeyError("material.not_a_field")

  monkeypatch.setattr(imr_design, "_fisher", broken)
  with pytest.raises(RuntimeError) as caught:
    imr_design.expected_information_gain(design, draws=2)
  assert isinstance(caught.value.__cause__, KeyError)
  assert "not_a_field" in str(caught.value.__cause__)


def test_a_design_inference_refuses_to_be_fitted(design):
  """It holds placeholder radii, so a likelihood evaluated against them is
  meaningless -- and `RadiusObservation` cannot object, because a constant
  positive trace is valid."""
  assert isinstance(design, imr_design.DesignInference)
  unit = np.full(design.size, 0.5)
  for call in (lambda: design.evaluate(unit), lambda: design.residual(unit), lambda: design.fit_multistart(2)):
    with pytest.raises(TypeError, match="placeholders"):
      call()


def test_a_design_inference_still_scores(design):
  """The refusal must not reach the Jacobian, which needs no measured radii."""
  assert np.all(np.isfinite(design.jacobian(np.full(design.size, 0.5))))


@pytest.mark.parametrize(("kwargs", "error"), (({"draws": 0}, ValueError), ({"workers": 0}, ValueError), ({"prior_variance": -1.0}, ValueError)))
def test_invalid_arguments_are_rejected(design, kwargs, error):
  with pytest.raises(error):
    imr_design.expected_information_gain(design, **kwargs)


def test_foreign_input_is_rejected():
  with pytest.raises(TypeError, match="PreparedInference"):
    imr_design.expected_information_gain(object())


def test_a_prior_sweep_reuses_one_set_of_solves(design, monkeypatch):
  """`prior_variance` exists for sweeping several priors against one design, and
  the solves do not depend on the prior. Fused, a 5-prior sweep at draws=128
  paid 640 solves for 128 distinct Jacobians."""
  calls = []
  real = imr_design._fisher
  monkeypatch.setattr(imr_design, "_fisher", lambda inference, unit: (calls.append(unit), real(inference, unit))[1])

  information = imr_design.design_information(design, draws=4)
  gains = [
    imr_design.expected_information_gain(design, information=information, prior_variance=variance).expected_information_gain
    for variance in (imr_design.UNIFORM_VARIANCE, imr_design.UNIFORM_VARIANCE / 10.0, imr_design.UNIFORM_VARIANCE / 100.0)
  ]
  assert len(calls) == 4, f"expected 4 solves for 3 priors, got {len(calls)}"
  assert gains[0] > gains[1] > gains[2], "a tighter prior must leave less to learn"


def test_a_second_observable_raises_the_information(measured):
  """The BOED reason for observables: *what to measure* becomes a design
  variable, not a modelling choice.

  Adding an observable adds a positive semidefinite block to `J^T J`, so the
  gain cannot fall. It costs nothing in solve time -- one sensitivity solve
  already returns tangents for every field, and the likelihood used to discard
  all but one.
  """
  config = imr_fast.SimulationConfig(R0, REQ, imr_fast.NeoHookeanKelvinVoigt(2500.0, 0.1))
  radius = imr_fast.inference.RadiusObservation(_TIMES, np.full(_TIMES.size, R0), _NOISE)
  velocity = imr_fast.inference.FieldObservation("wall_velocity_m_s", _TIMES, np.zeros(_TIMES.size), 2.0)

  alone = imr_design.expected_information_gain(imr_design.DesignInference(config, radius, _PARAMETERS), draws=6).expected_information_gain
  together = imr_design.expected_information_gain(
    imr_design.DesignInference(config, (radius, velocity), _PARAMETERS), draws=6
  ).expected_information_gain

  measured("EIG radius vs radius+velocity", f"{alone:.3f} -> {together:.3f} nats")
  assert together > alone, "adding an observable cannot reduce the information"


@pytest.mark.parametrize("index", (5, 12, 20))
def test_the_time_gradient_matches_a_central_difference(index, measured):
  """Scoring a time grid needs `J`; moving one needs `dJ/dt`. For a radius
  observation that is the wall-velocity tangent, which the same solve already
  returns -- so this costs nothing beyond the arithmetic.

  The reference perturbs one observation time and re-scores, which exercises the
  whole path including the union-grid indexing.
  """
  times = np.linspace(2e-6, 4e-5, 25)
  config = imr_fast.SimulationConfig(R0, REQ, imr_fast.NeoHookeanKelvinVoigt(2500.0, 0.1))

  def score(grid):
    return imr_design.expected_information_gain(imr_design.design_inference(config, grid, _NOISE, _PARAMETERS), draws=4).expected_information_gain

  analytic = imr_design.information_time_gradient(imr_design.design_inference(config, times, _NOISE, _PARAMETERS), draws=4)
  step = 2e-9
  ahead, behind = times.copy(), times.copy()
  ahead[index] += step
  behind[index] -= step
  difference = (score(ahead) - score(behind)) / (2.0 * step)

  error = abs(analytic[index] - difference) / max(abs(difference), 1e-30)
  measured(f"dEIG/dt at t[{index}]", f"{analytic[index]:.4e} vs {difference:.4e}, rel={error:.2e}")
  assert error < 1e-5


def test_a_field_without_a_time_derivative_refuses(measured):
  """`dP/dt` would mean differentiating the right-hand side, which is a larger
  change than this. Refusing beats silently returning the wrong tangent."""
  config = imr_fast.SimulationConfig(R0, REQ, imr_fast.NeoHookeanKelvinVoigt(2500.0, 0.1))
  pressure = imr_fast.inference.FieldObservation("internal_pressure_pa", _TIMES, np.full(_TIMES.size, 1e5), 1e3)
  inference = imr_design.DesignInference(config, pressure, _PARAMETERS)
  with pytest.raises(NotImplementedError, match="no time derivative available"):
    inference.jacobian_time_derivative(np.full(inference.size, 0.5))

# Exact information gain for the design below, from tensor Gauss-Legendre
# quadrature of the posterior over the unit square -- the prior is Uniform(0,1)
# per parameter, so `0.5*(logdet Sigma_prior - logdet Sigma_posterior)` is a
# 2-D integral owing nothing to the Laplace approximation being checked. Stable
# to 1e-4 between order 40 and 64 (10.8069, 10.8068).
_EXACT_GAIN_AT_COLLAPSE = 10.807

def test_gain_matches_an_exact_posterior(measured):
  """The criterion against a posterior computed rather than sampled.

  Everything else here checks EIG's arithmetic -- the closed form, the sigma
  scaling, monotonicity. This checks the approximation itself: EIG assumes the
  posterior is the Gaussian its Fisher information implies, and quadrature says
  what the posterior actually is.

  #25 recorded EIG as over-predicting by 0.52-0.81 against NUTS. That was wrong,
  and this is the reference that shows it. The NUTS runs behind that number had
  R-hat 1.84 and 3.1 effective samples from 400 draws -- both chains stranded
  outside the posterior, one of them 7568 nats below the mode -- and pooling
  them inflated the covariance by ~4 nats of gain. `compute_convergence_checks`
  had been switched off, so nothing said so. Against the exact posterior the
  criterion is accurate to 0.003 nats here.

  The failure mode is worth keeping in view: this posterior has a unit-coordinate
  sd of 1.3e-03, about 68x tighter than the wide-prior case in `pymc_op`'s
  docstring, where NUTS converges fine. A better design gives a tighter
  posterior, so the sampler gets *harder* to trust exactly as the design
  improves.
  """
  times = _collapse_design()
  gain = imr_design._gain(_design(times=times), _TRUTH_UNIT, np.full(len(_PARAMETERS), imr_design.UNIFORM_VARIANCE))
  error = abs(gain - _EXACT_GAIN_AT_COLLAPSE)
  measured("EIG vs exact posterior", f"{gain:.4f} vs {_EXACT_GAIN_AT_COLLAPSE:.4f}, abs={error:.3f} nats")
  assert error < 0.05, f"EIG {gain:.4f} against an exact quadrature posterior of {_EXACT_GAIN_AT_COLLAPSE:.4f}"


# Batched design information. Scoring a design is `draws` sensitivity solves that
# differ only in the values of the inference parameters -- which the traced path
# takes as an argument -- so the graph is shared and `vmap` maps over the points
# rather than the loop re-tracing per draw.


def test_batched_design_information_matches_the_per_draw_loop(measured):
  """The whole point of batching is that it changes cost and not the answer.

  Compared against `_fisher` per draw, which is the numpy route, so the difference
  here is cross-backend rather than batching: the Fisher entries land at 1e-04 while
  the EIG lands at 5e-07, because a log-determinant is far less sensitive than the
  entries it reduces. The EIG is what a design is ranked by, so that is the number
  the bound is set on.
  """
  # `_fisher` and `jacobians` share one implementation now, so this compares batching
  # alone -- the per-draw loop against the vmapped program.
  inference = imr_design.design_inference(
    imr_fast.SimulationConfig(R0=R0, Req=REQ, material=imr_fast.NeoHookeanKelvinVoigt(2500.0, 0.1)),
    _TIMES, _NOISE, _PARAMETERS,
  )
  draws = 32
  points = np.random.default_rng(0).random((draws, len(_PARAMETERS)))
  looped = np.array([imr_design._fisher(inference, unit) for unit in points])
  batched, requested, failures = imr_design.design_information(inference, draws=draws, seed=0, batched=True)
  assert (requested, failures) == (draws, 0)
  assert batched.shape == looped.shape
  fisher = float(np.max(np.abs(looped - batched))) / float(np.max(np.abs(looped)))
  looped_gain = imr_design.expected_information_gain(inference, draws=draws, seed=0, information=(looped, draws, 0))
  batched_gain = imr_design.expected_information_gain(inference, draws=draws, seed=0, information=(batched, requested, failures))
  gain = abs(looped_gain.expected_information_gain - batched_gain.expected_information_gain) / abs(looped_gain.expected_information_gain)
  measured("batched vs looped design information", f"fisher={fisher:.1e} eig={gain:.1e}")
  # Same backend on both sides, so this is batching alone and the bound is tight.
  assert fisher < 1e-9, fisher
  assert gain < 1e-11, gain


def test_batched_jacobians_agree_with_the_single_draw_call():
  """`jacobians` skips the per-draw `config_from_unit` and `prepare` that `jacobian`
  performs, on the grounds that a draw varies only the traced parameters. This is
  what licenses that: the two agree where both run through the same backend."""
  config = imr_fast.SimulationConfig(R0=R0, Req=REQ, material=imr_fast.NeoHookeanKelvinVoigt(2500.0, 0.1))
  inference = imr_design.design_inference(config, _TIMES, _NOISE, _PARAMETERS)
  points = np.random.default_rng(1).random((5, len(_PARAMETERS)))
  looped = np.stack([inference.jacobian(unit) for unit in points])
  batched = inference.jacobians(points)
  assert batched.shape == looped.shape
  assert float(np.max(np.abs(looped - batched))) / float(np.max(np.abs(looped))) < 1e-8


def test_batching_refuses_to_pretend_it_can_count_failures():
  """`batched=True` is opt-in because a single traced program fails as a WHOLE. Rather
  than accept `max_failure_fraction` and silently ignore it -- which would report a
  gain as conditional on draws it never checked -- the combination is refused."""
  inference = imr_design.design_inference(
    imr_fast.SimulationConfig(R0=R0, Req=REQ, material=imr_fast.NeoHookeanKelvinVoigt(2500.0, 0.1)),
    _TIMES, _NOISE, _PARAMETERS,
  )
  with pytest.raises(ValueError, match="cannot honour max_failure_fraction"):
    imr_design.design_information(inference, draws=4, seed=0, batched=True, max_failure_fraction=0.5)
