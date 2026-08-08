"""Fitted axes that reach the configuration rather than the material.

`Req` is the case this exists for. It is inferred rather than measured, and
`docs/writeup/initial_state.py` finds a 1.68% error in it leaves the same residual as
changing the bubble-dynamics operator -- so a study that pins it asserts a precision the
experiment does not have. A `CandidateModel` could not express that: every axis it had went
into `build`, and `build` returns a material.

The gates here are recovery, not plumbing. A config axis that is threaded correctly but
never actually reaches the solver looks exactly like one that works: the fit still converges,
on the remaining axes, and reports a plausible number for the axis it ignored.
"""

import numpy as np
import pytest

from pyimr.selection import (CandidateModel, PARAMETER_BOUNDS, evaluate_at, fit_candidate,
                             physical_from_unit)
from pyimr._materials import NeoHookeanKelvinVoigt

SAMPLES = 40
GRID = np.linspace(1.0, 2.0, SAMPLES)
BOX = {"mu": (1e-3, 1e0), "g": (1e2, 1e4), "req_scale": (0.8, 1.25)}


def _forward(material, config):
  """A smooth stand-in whose response to `req_scale` is independent of the material.

  Deliberately not a sum: if the config axis entered the same way the material does, a fit
  could trade one against the other and recovery would prove nothing about the plumbing.
  """
  scale = float((config or {}).get("req_scale", 1.0))
  return (GRID * np.log(material.viscosity_pa_s) + GRID**2 * np.log(material.shear_modulus_pa)
          + 8.0 * np.sin(3.0 * GRID) * (scale - 1.0)), None


CANDIDATE = CandidateModel("kv|req", lambda t: NeoHookeanKelvinVoigt(t["g"], t["mu"]),
                           ("mu", "g", "req_scale"), config_axes=("req_scale",))


def test_a_config_axis_must_be_one_of_the_candidates_axes():
  with pytest.raises(ValueError, match="config_axes"):
    CandidateModel("bad", lambda t: NeoHookeanKelvinVoigt(t["g"], 0.1), ("g",),
                   config_axes=("req_scale",))


def test_the_fitted_value_reaches_the_solver():
  seen = {}

  def watch(material, config):
    seen.update(config)
    return _forward(material, config)

  evaluate_at(CANDIDATE, watch, {"mu": 0.05, "g": 500.0, "req_scale": 1.07})
  assert seen == {"req_scale": 1.07}, "only the config axes, and their fitted values"


def test_a_material_only_candidate_still_gets_an_empty_config():
  plain = CandidateModel("kv", lambda t: NeoHookeanKelvinVoigt(t["g"], t["mu"]), ("mu", "g"))
  seen = {}

  def watch(material, config):
    seen["config"] = config
    return _forward(material, config)

  evaluate_at(plain, watch, {"mu": 0.05, "g": 500.0})
  assert seen["config"] == {}, "an empty dict, not None: the signature must not vary"


@pytest.mark.parametrize("truth_scale", [0.93, 1.0, 1.09])
def test_it_recovers_a_displaced_configuration(truth_scale):
  """The gate that matters: a record generated at a shifted `req_scale` is fitted back.

  Run at three values including 1.0, because a mechanism that silently ignored the axis
  would still pass at 1.0 -- and passing only there is exactly the signature of an axis
  that is being fitted but never applied.
  """
  truth = {"mu": 0.05, "g": 500.0, "req_scale": truth_scale}
  observed = evaluate_at(CANDIDATE, _forward, truth)[0]
  fit = fit_candidate(CANDIDATE, _forward, observed, 1e-3, bounds=BOX, starts=8,
                      max_evaluations=400)
  recovered = dict(zip(CANDIDATE.axes,
                       physical_from_unit(CANDIDATE.axes, fit.unit, BOX), strict=True))
  assert recovered["req_scale"] == pytest.approx(truth_scale, rel=2e-2)
  assert recovered["g"] == pytest.approx(truth["g"], rel=5e-2)


def test_ignoring_the_config_axis_would_be_caught():
  """A mutation check kept as a test: pin that the axis changes the trace at all.

  Without this the recovery test above could pass for the wrong reason -- `least_squares`
  reports whatever it started from on an axis with no gradient, and a Latin hypercube start
  near the truth would look like a recovery.
  """
  fields = {"mu": 0.05, "g": 500.0, "req_scale": 1.0}
  base = evaluate_at(CANDIDATE, _forward, fields)[0]
  moved = evaluate_at(CANDIDATE, _forward, {**fields, "req_scale": 1.05})[0]
  assert np.max(np.abs(base - moved)) > 1e-2, "the config axis does nothing to the trace"


def test_the_default_bound_is_registered():
  # the study overrides it, but a candidate that names `req_scale` without passing bounds
  # must still fit rather than raise about a missing axis
  assert "req_scale" in PARAMETER_BOUNDS
  low, high = PARAMETER_BOUNDS["req_scale"]
  assert low < 1.0 < high, "the prior must contain the inferred value it scales"
  assert abs(np.log(low) + np.log(high)) < 1e-9, "log-symmetric about 1, or it biases Req"
