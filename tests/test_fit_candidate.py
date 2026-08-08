"""Fitting a candidate, which is what turns a registered model into a rankable one.

`candidate_log_evidence` expands about a fitted point and nothing produced one: a
candidate's axes need not be material fields, so `PreparedInference.fit_multistart`, which
works from attribute paths, cannot reach them.

Most of these run against a cheap analytic forward model with a known minimum, because what
needs testing is the search and its reporting -- that it finds the optimum, survives regions
where the material will not integrate, and says so when its own evaluations mostly failed.
One test runs the real solver, and is marked slow.
"""

import numpy as np
import pytest

from pyimr.selection import (
  PARAMETER_BOUNDS,
  STANDARD_MODELS,
  fit_candidate,
  physical_from_unit,
)

SAMPLES = 24
# a two-axis candidate is enough for the search, and 2 axes keeps the cheap tests instant
TWO = STANDARD_MODELS["NHKV"]          # axes ("mu", "g")


def analytic(target):
  """A forward model that is a smooth, well-conditioned function of the two axes.

  `_sensitive` in `test_selection.py` sums the fields, which makes every point on a line
  equally good -- fine for testing plumbing, useless for testing an optimiser. This has one
  minimum, at `target`.
  """
  grid = np.linspace(1.0, 2.0, SAMPLES)

  def solve(material, _config):
    coordinates = np.array([np.log(material.viscosity_pa_s), np.log(material.shear_modulus_pa)])
    offset = coordinates - np.log(np.array(target))
    trace = grid * (1.0 + offset[0]) + grid**2 * offset[1]
    return trace, trace

  return solve


def unit_of(candidate, values):
  from pyimr.prior import normalize_log_coordinates

  return np.array([normalize_log_coordinates(values[a], *PARAMETER_BOUNDS[a]) for a in candidate.axes])


def test_it_finds_the_optimum_of_a_model_with_a_known_one():
  target = {"mu": 0.05, "g": 2000.0}
  solve = analytic((target["mu"], target["g"]))
  observed = solve(TWO.build(target), {})[0]

  fit = fit_candidate(TWO, solve, observed, 1e-3, starts=6)
  recovered = dict(zip(TWO.axes, physical_from_unit(TWO.axes, fit.unit), strict=True))
  for axis in TWO.axes:
    assert recovered[axis] == pytest.approx(target[axis], rel=1e-3), axis
  assert fit.chi_squared < 1e-6
  assert fit.failure_fraction == 0.0


def test_it_fits_at_least_as_well_as_the_point_the_data_came_from():
  """The honest criterion where a model is sloppy.

  Asserting that a fit RECOVERS the generating parameters fails on a ridge, and the ridge is
  real -- on the actual records `g` and `alpha` trade against each other so freely that grid
  refinement moves the best point from `g = 658` to `g = 144`. What a fit owes you is a
  cost no worse than the truth's, and that holds ridge or no ridge.
  """
  target = {"mu": 0.05, "g": 2000.0}
  solve = analytic((target["mu"], target["g"]))
  rng = np.random.default_rng(0)
  observed = solve(TWO.build(target), {})[0] + rng.normal(0.0, 1e-3, SAMPLES)

  fit = fit_candidate(TWO, solve, observed, 1e-3, starts=6)
  truth_residual = (solve(TWO.build(target), {})[0] - observed) / 1e-3
  assert fit.chi_squared <= float(truth_residual @ truth_residual) / SAMPLES + 1e-9


def test_a_region_that_will_not_integrate_does_not_kill_the_fit():
  """An optimiser in a six-dimensional box WILL step somewhere the material refuses.

  Before this was handled, one such step raised out of the residual and took the whole fit
  with it -- the failure that this test exists for. The optimiser must instead see a large
  finite residual, leave, and still converge.
  """
  target = {"mu": 0.05, "g": 2000.0}
  clean = analytic((target["mu"], target["g"]))
  observed = clean(TWO.build(target), {})[0]

  def hostile(material, _config):
    # a whole slab of the box is unintegrable, on the far side from the optimum
    if material.shear_modulus_pa > 2e4: raise RuntimeError("the maximum number of solver steps was reached")
    return clean(material, _config)

  fit = fit_candidate(TWO, hostile, observed, 1e-3, starts=8)
  recovered = dict(zip(TWO.axes, physical_from_unit(TWO.axes, fit.unit), strict=True))
  assert recovered["g"] == pytest.approx(target["g"], rel=1e-3)
  assert fit.failures > 0, "the hostile region must actually have been entered"


def test_it_reports_the_share_of_evaluations_that_failed():
  """The diagnostic that says how much of the search was spent against a wall.

  Here two thirds of the box is unintegrable but the optimum is not, so the fit succeeds
  and `failure_fraction` is what tells you it was a fight. A fit reported without it looks
  identical to one that never met resistance.
  """
  target = {"mu": 0.05, "g": 500.0}
  clean = analytic((target["mu"], target["g"]))
  observed = clean(TWO.build(target), {})[0]

  def obstructed(material, _config):
    if material.shear_modulus_pa > 1e3: raise RuntimeError("nope")
    return clean(material, _config)

  fit = fit_candidate(TWO, obstructed, observed, 1e-3, starts=8)
  recovered = dict(zip(TWO.axes, physical_from_unit(TWO.axes, fit.unit), strict=True))
  assert recovered["g"] == pytest.approx(target["g"], rel=1e-2)
  assert 0.0 < fit.failure_fraction < 1.0, "the wall must be visible in the diagnostic"


def test_a_solve_that_never_works_is_an_error_not_a_fit():
  observed = np.zeros(SAMPLES)
  with pytest.raises(ValueError, match="did not fit"):
    fit_candidate(TWO, lambda _m: (_ for _ in ()).throw(RuntimeError("no")), observed, 1e-3, starts=2)


def test_the_modes_it_keeps_are_distinct_and_ordered():
  target = {"mu": 0.05, "g": 2000.0}
  solve = analytic((target["mu"], target["g"]))
  observed = solve(TWO.build(target), {})[0]

  fit = fit_candidate(TWO, solve, observed, 1e-3, starts=6, separation=1e-3)
  assert list(fit.costs) == sorted(fit.costs), "modes must be best first"
  assert np.array_equal(fit.unit, fit.modes[0])
  for earlier in range(len(fit.modes)):
    for later in range(earlier + 1, len(fit.modes)):
      assert np.linalg.norm(fit.modes[earlier] - fit.modes[later]) > 1e-3
  assert fit.converged <= fit.starts


def test_a_shape_mismatch_is_raised_rather_than_penalised():
  # penalising it would hide the caller's mistake behind a merely bad fit
  target = {"mu": 0.05, "g": 2000.0}
  solve = analytic((target["mu"], target["g"]))
  with pytest.raises(ValueError, match="samples against"):
    fit_candidate(TWO, solve, np.zeros(SAMPLES + 3), 1e-3, starts=1)


@pytest.mark.parametrize(("kwargs", "message"), [
  ({"starts": 0}, "starts"),
  ({"separation": 0.0}, "separation"),
  ({"separation": 1.0}, "separation"),
])
def test_it_refuses_malformed_requests(kwargs, message):
  solve = analytic((0.05, 2000.0))
  with pytest.raises(ValueError, match=message):
    fit_candidate(TWO, solve, np.zeros(SAMPLES), 1e-3, **kwargs)


def test_a_non_positive_deviation_is_refused():
  solve = analytic((0.05, 2000.0))
  with pytest.raises(ValueError, match="deviation"):
    fit_candidate(TWO, solve, np.zeros(SAMPLES), 0.0, starts=1)


@pytest.mark.slow
def test_it_fits_the_real_forward_model():
  """The cheap tests pin the search; this pins that it drives the actual solver.

  Asserted on the COST alone, not on recovered parameters. On this record the fit reaches a
  cost below the generating point's while placing `g` at its bound: `g` and `alpha` trade
  along a ridge so freely that grid refinement alone moves the best point from `g = 658` to
  `g = 144`. An earlier draft of this test asserted `mu` to 30%, which passed at one search
  budget and failed at a smaller one -- the parameter is no more pinned than the others,
  and a test that says otherwise is measuring the budget.

  The budget here is deliberate. At `starts=4, max_evaluations=80` this fit reaches
  chi2/N = 27.3 against the truth's 0.85 -- worse than the point the data came from, because
  the differenced Jacobian spends `p + 1` solves an iteration and 80 evaluations is about
  sixteen steps in four dimensions. At `starts=6, max_evaluations=150` it reaches 0.64. The
  search is budget-sensitive and this test pins a budget that suffices, not one that
  flatters it.
  """
  import pyimr

  times = np.linspace(0.0, 4e-5, 50)

  def solve(material, _config):
    config = pyimr.SimulationConfig(277e-6, 277e-6 / 7.09, material, dynamics="keller-miksis",
                                    rtol=1e-7, atol=1e-9, max_steps=200_000)
    trace = np.asarray(pyimr.simulate(times, config).radius_ratio, dtype=float)
    return trace, trace

  candidate = STANDARD_MODELS["qSLS"]
  truth = {"mu": 0.04651, "g": 204.3, "lambda1": 1.964e-7, "alpha": 5.301}
  observed = solve(candidate.build(truth), {})[0] + np.random.default_rng(0).normal(0.0, 2e-3, times.size)

  fit = fit_candidate(candidate, solve, observed, 2e-3, starts=6, max_evaluations=150)
  truth_residual = (solve(candidate.build(truth), {})[0] - observed) / 2e-3
  assert fit.chi_squared <= float(truth_residual @ truth_residual) / times.size + 1e-6
  assert fit.failure_fraction < 0.5
  assert 0.0 <= fit.unit.min() and fit.unit.max() <= 1.0
