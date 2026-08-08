"""Resolution calibration: choose discretization and tolerance by measuring.

The folklore this replaces was established at single configurations and did not
generalize -- Nt = 7 to 9 held for one collapse and failed by 4x over five. These tests
therefore check the guards as hard as the happy path: an unconverged reference and an
unreachable target must raise rather than return a plausible number.
"""

import numpy as np
import pytest

import pyimr
from pyimr.resolution import NT_LADDER, RTOL_LADDER, Resolution, _at
from _validation_support import NHKV, R0, REQ

SECTION = "17. Resolution calibration"

_TIMES = np.linspace(0.0, 20e-6, 60)


def _config(**kwargs):
  return pyimr.SimulationConfig(R0, REQ, NHKV, bubtherm=1, **kwargs)


@pytest.fixture(scope="module")
def reference_for():
  """Cache `_reference` by (ladder, fields) so tests that just need a converged reference
  for the same problem share the solves. `target` only gates whether `_reference` raises,
  not what it returns, so one loose target serves every entry."""
  from pyimr.resolution import _reference

  cache = {}

  def get(ladder, fields=("radius_ratio",)):
    key = (tuple(ladder), fields)
    if key not in cache:
      cache[key] = _reference(_config(), _TIMES, fields, 1.0, ladder)
    return cache[key]

  return get


def test_the_ladders_are_ascending():
  assert list(NT_LADDER) == sorted(NT_LADDER)
  assert list(RTOL_LADDER) == sorted(RTOL_LADDER)


def test_choose_resolution_rejects_a_non_ascending_nt_ladder():
  """A descending `nt_ladder` used to build the reference at the coarsest, not finest,
  entry -- measuring against a reference that shares the error under test."""
  from pyimr.resolution import choose_resolution

  with pytest.raises(ValueError, match="nt_ladder"):
    choose_resolution(_config(), _TIMES, 1e-2, nt_ladder=(13, 9, 5))


def test_choose_resolution_rejects_a_non_ascending_rtol_ladder():
  from pyimr.resolution import choose_resolution

  with pytest.raises(ValueError, match="rtol_ladder"):
    choose_resolution(_config(), _TIMES, 1e-2, rtol_ladder=(1e-4, 1e-6, 1e-8))


def test_choose_resolution_rejects_an_empty_nt_ladder():
  from pyimr.resolution import choose_resolution

  with pytest.raises(ValueError, match="nt_ladder"):
    choose_resolution(_config(), _TIMES, 1e-2, nt_ladder=())


def test_choose_resolution_rejects_an_empty_rtol_ladder():
  from pyimr.resolution import choose_resolution

  with pytest.raises(ValueError, match="rtol_ladder"):
    choose_resolution(_config(), _TIMES, 1e-2, rtol_ladder=())


def test_choose_resolution_rejects_a_nonpositive_target():
  from pyimr.resolution import choose_resolution

  with pytest.raises(ValueError, match="target must be positive"):
    choose_resolution(_config(), _TIMES, 0.0)


def test_choose_resolution_requires_bubtherm():
  from pyimr.resolution import choose_resolution

  config = pyimr.SimulationConfig(R0, REQ, NHKV, bubtherm=0)
  with pytest.raises(ValueError, match="bubtherm"):
    choose_resolution(config, _TIMES, 1e-2)


def test_apply_substitutes_only_the_resolution_fields():
  config = _config(pA=1234.0, dynamics="keller-miksis")
  setting = Resolution(thermal="fd", Nt=13, rtol=1e-6, atol=1e-8, achieved=1e-4, seconds=0.01)
  applied = setting.apply(config)

  assert (applied.thermal, applied.Nt, applied.rtol, applied.atol) == ("fd", 13, 1e-6, 1e-8)
  assert applied.pA == 1234.0 and applied.radial == 2 and applied.R0 == config.R0


def test_mt_follows_nt_only_when_the_medium_is_active():
  """`Mt` is tied to `Nt` by design. Leaving it alone when `medtherm=0` matters because
  the medium grid is then unused and rewriting it would be a silent, invisible change.
  """
  setting = Resolution(thermal="spectral", Nt=7, rtol=1e-6, atol=1e-8, achieved=0.0, seconds=0.0)
  assert setting.apply(_config(medtherm=0)).Mt == _config(medtherm=0).Mt
  assert setting.apply(_config(medtherm=1, Mt=100)).Mt == 7


def test_at_builds_the_same_config_apply_would():
  config = _config(medtherm=1)
  built = _at(config, "fd", 9, 1e-7, 1e-9)
  assert (built.thermal, built.Nt, built.Mt, built.rtol, built.atol) == ("fd", 9, 9, 1e-7, 1e-9)
  setting = Resolution(thermal="fd", Nt=9, rtol=1e-7, atol=1e-9, achieved=0.0, seconds=0.0)
  assert setting.apply(config) == built


def test_deviation_is_relative_to_each_field_own_peak(measured):
  """Observables do not converge together -- at identical settings, relative error was
  3.4e-07 for radius against 2.8e-05 for internal pressure, about 80x looser. Scaling by
  each field's own peak is what keeps one target meaningful across fields.
  """
  from pyimr.resolution import _deviation

  reference = {"a": np.array([0.0, 2.0]), "b": np.array([0.0, 200.0])}
  candidate = {"a": np.array([0.0, 2.2]), "b": np.array([0.0, 220.0])}
  # both are 10% of their own peak, so both must report 0.1 rather than 0.2 and 20.0
  assert _deviation(candidate, reference) == pytest.approx(0.1)
  measured("per-field scaling", "10% of peak reads 0.1 for peaks 2 and 200 alike")


def test_deviation_reports_the_worst_field():
  from pyimr.resolution import _deviation

  reference = {"a": np.array([1.0]), "b": np.array([1.0])}
  assert _deviation({"a": np.array([1.01]), "b": np.array([1.5])}, reference) == pytest.approx(0.5)


def test_solve_returns_only_the_requested_fields():
  from pyimr.resolution import _solve

  values = _solve(_at(_config(), "spectral", 5, 1e-6, 1e-8), _TIMES, ("radius_ratio",))
  assert set(values) == {"radius_ratio"}
  assert values["radius_ratio"].shape == _TIMES.shape


def test_timed_measures_actual_wall_clock(monkeypatch):
  """A `_timed` that stopped measuring (e.g. `return 0.5, values`) would pass every other
  test in this file, since none of them check `seconds` against a known bound."""
  import time as time_module
  import pyimr.resolution as resolution_module

  def slow_solve(config, times, fields):
    time_module.sleep(0.05)
    return {"radius_ratio": np.zeros(2)}

  monkeypatch.setattr(resolution_module, "_solve", slow_solve)
  seconds, _ = resolution_module._timed(_config(), _TIMES, ("radius_ratio",))
  assert 0.03 < seconds < 0.3


def test_the_reference_refuses_when_it_is_not_converged(measured):
  """The guard this module exists around. Everything downstream inherits the reference's
  error silently, and a reference sharing the error under test reports success -- that is
  how Nt = 5 was measured against an Nt = 5 reference and looked free at 2.07e-02.
  """
  from pyimr.resolution import _reference

  with pytest.raises(ValueError, match="reference is not converged"):
    # target/10 = 1e-30 is unreachable, so any real gap trips the guard
    _reference(_config(), _TIMES, ("radius_ratio",), 1e-29, (5, 7))
  measured("reference guard", "unreachable target/10 raises rather than returning")


def test_the_reference_returns_the_finer_solve_when_converged(reference_for):
  from pyimr.resolution import _at, _solve

  reference, _gap = reference_for((5, 9))
  finer = _solve(_at(_config(), "spectral", 18, 1e-12, 1e-14), _TIMES, ("radius_ratio",))
  np.testing.assert_allclose(reference["radius_ratio"], finer["radius_ratio"], rtol=0, atol=1e-12)


def test_the_reference_guard_uses_target_divided_by_10(measured):
  """Regression guard: ensures the convergence check uses target/10, not target.
  If the /10 margin is dropped, this test fails but the others pass -- the gap exceeds
  target/10 but not target, discriminating which boundary is checked.
  """
  from pyimr.resolution import _reference, _at, _solve, _deviation

  coarse = _solve(_at(_config(), "spectral", 7, 1e-10, 1e-12), _TIMES, ("radius_ratio",))
  finer = _solve(_at(_config(), "spectral", 14, 1e-12, 1e-14), _TIMES, ("radius_ratio",))
  gap = _deviation(coarse, finer)

  target = 5.0 * gap

  with pytest.raises(ValueError, match="reference is not converged"):
    _reference(_config(), _TIMES, ("radius_ratio",), target, (5, 7))
  measured("target/10 margin", "guard rejects when target/10 < gap < target")


def test_it_finds_the_smallest_adequate_node_count(reference_for):
  """Bisection over the ladder relies on discretization error falling with Nt."""
  from pyimr.resolution import _smallest_adequate, _at, _solve, _deviation

  ladder = (5, 9, 13)
  reference, _ = reference_for(ladder)
  index = _smallest_adequate(_config(), _TIMES, ("radius_ratio",), 1e-2, reference, "spectral", ladder)
  assert index is not None
  at_index = _solve(_at(_config(), "spectral", ladder[index], 1e-10, 1e-12), _TIMES, ("radius_ratio",))
  assert _deviation(at_index, reference) <= 1e-2, "entry at index must meet the target"
  if index > 0:
    below = _solve(_at(_config(), "spectral", ladder[index - 1], 1e-10, 1e-12), _TIMES, ("radius_ratio",))
    assert _deviation(below, reference) > 1e-2, "a smaller ladder entry also met the target"


def test_an_unreachable_target_is_reported_not_approximated(measured, reference_for):
  """Returning the best of an inadequate set is the same failure as an unconverged
  reference: a number that looks like success.
  """
  from pyimr.resolution import _choose_discretization

  ladder = (5, 7)
  reference, _ = reference_for(ladder)
  with pytest.raises(ValueError, match="no discretization in the ladder"):
    _choose_discretization(_config(), _TIMES, ("radius_ratio",), 1e-16, reference, ladder)
  measured("unreachable target", "raises rather than returning the best available")


def test_choose_discretization_returns_the_faster_candidate(reference_for):
  """The winner is picked by measured time, not by any analytic cost proxy."""
  from pyimr.resolution import _choose_discretization, _smallest_adequate, _at, _timed

  ladder = (5, 9)
  reference, _ = reference_for(ladder)
  timings = {}
  for thermal in ("spectral", "fd"):
    index = _smallest_adequate(_config(), _TIMES, ("radius_ratio",), 1e-2, reference, thermal, ladder)
    if index is not None:
      timings[thermal] = _timed(_at(_config(), thermal, ladder[index], 1e-10, 1e-12), _TIMES, ("radius_ratio",))[0]
  assert timings, "neither discretization met a loose target"
  chosen, _nodes = _choose_discretization(_config(), _TIMES, ("radius_ratio",), 1e-2, reference, ladder)
  assert chosen == min(timings, key=lambda name: timings[name])


def test_fd_is_genuinely_evaluated_not_just_offered(reference_for):
  """The spectral preference in this package was measured at one configuration and at
  tight accuracy, neither of which holds in the usual regime, so fd has to be a real
  candidate. Asserting the winner is one of two names would pass without fd ever running.
  """
  from pyimr.resolution import _smallest_adequate

  ladder = (5, 9, 13)
  reference, _ = reference_for(ladder)
  index = _smallest_adequate(_config(), _TIMES, ("radius_ratio",), 1e-2, reference, "fd", ladder)
  assert index is not None, "fd could not meet a loose target at any ladder entry"


def test_an_inadequate_grid_is_not_accepted(measured, reference_for):
  """The concrete failure this module was built around: Nt = 5 over a multi-collapse
  record carries 2.07e-02 discretization error, as large as the experimental noise it was
  supposed to sit beneath, while looking free against a reference computed at Nt = 5.
  """
  from pyimr.resolution import _smallest_adequate

  ladder = (5, 9, 13)
  reference, _ = reference_for(ladder)
  loose = _smallest_adequate(_config(), _TIMES, ("radius_ratio",), 1e-07, reference, "spectral", ladder)
  tight = _smallest_adequate(_config(), _TIMES, ("radius_ratio",), 1e-08, reference, "spectral", ladder)
  assert loose is not None and tight is not None
  assert tight > loose, "a tighter target must select a finer grid"
  measured("grid selection", f"target 1e-07 -> Nt={ladder[loose]}, 1e-08 -> Nt={ladder[tight]}")


def test_loosest_tolerance_meets_target_and_the_next_looser_entry_does_not(reference_for):
  """`_loosest_tolerance` must return the loosest ladder rtol that still meets `target`,
  not merely some rtol that meets it and not the tightest available either.
  """
  from pyimr.resolution import _loosest_tolerance, _at, _solve, _deviation

  ladder, rtol_ladder, target = (5, 9), (1e-8, 1e-6, 1e-4), 1e-5
  reference, _ = reference_for(ladder)
  chosen = _loosest_tolerance(_config(), _TIMES, ("radius_ratio",), target, reference, "spectral", 5, rtol_ladder)

  at_chosen = _solve(_at(_config(), "spectral", 5, chosen, chosen * 1e-2), _TIMES, ("radius_ratio",))
  assert _deviation(at_chosen, reference) <= target

  looser = rtol_ladder[rtol_ladder.index(chosen) + 1:]
  assert looser, "test fixture must leave a next looser entry to check against"
  next_looser = looser[0]
  at_next_looser = _solve(_at(_config(), "spectral", 5, next_looser, next_looser * 1e-2), _TIMES, ("radius_ratio",))
  assert _deviation(at_next_looser, reference) > target, "a looser entry also met the target"


def test_loosest_tolerance_raises_rather_than_returning_the_tightest_entry(reference_for):
  """The bug this guards: `rtol_ladder[0]` missing `target` used to be returned anyway.
  Measured: target=1e-7 against rtol_ladder=(1e-4,) used to return achieved=2.011e-04.
  """
  from pyimr.resolution import _loosest_tolerance

  reference, _ = reference_for((5, 9))
  with pytest.raises(ValueError, match="no tolerance in the ladder"):
    _loosest_tolerance(_config(), _TIMES, ("radius_ratio",), 1e-7, reference, "spectral", 5, (1e-4,))


@pytest.mark.slow
def test_it_returns_a_setting_that_meets_the_target(measured):
  from pyimr.resolution import choose_resolution

  setting = choose_resolution(_config(), _TIMES, 1e-2, nt_ladder=(5, 9), rtol_ladder=(1e-8, 1e-6, 1e-4))
  assert setting.achieved <= 1e-2
  assert setting.thermal in ("spectral", "fd") and setting.Nt in (5, 9)
  assert setting.atol == pytest.approx(setting.rtol * 1e-2)
  assert setting.seconds > 0.0
  measured("chosen setting", f"{setting.thermal} Nt={setting.Nt} rtol={setting.rtol:.0e} "
                             f"achieved {setting.achieved:.2e} in {setting.seconds * 1e3:.1f} ms")


@pytest.mark.slow
def test_choose_resolution_raises_when_the_ladder_cannot_reach_target():
  """The bug this guards, at the public entry point: target=1e-7 against
  rtol_ladder=(1e-4,) used to return achieved=2.011e-04 -- 2000x over target -- silently.
  """
  from pyimr.resolution import choose_resolution

  with pytest.raises(ValueError, match="no tolerance in the ladder"):
    choose_resolution(_config(), _TIMES, 1e-7, nt_ladder=(5, 9), rtol_ladder=(1e-4,))


@pytest.mark.slow
def test_the_returned_setting_reproduces_its_reported_error(reference_for):
  """A setting whose own config does not reproduce the achieved error is not a setting."""
  from pyimr.resolution import _solve, _deviation, choose_resolution

  ladder = (5, 9)
  setting = choose_resolution(_config(), _TIMES, 1e-2, nt_ladder=ladder, rtol_ladder=(1e-8, 1e-6))
  reference, _ = reference_for(ladder)

  again = _solve(setting.apply(_config()), _TIMES, ("radius_ratio",))
  assert _deviation(again, reference) == pytest.approx(setting.achieved, rel=1e-9)


@pytest.mark.slow
def test_a_tighter_target_never_returns_a_looser_tolerance():
  from pyimr.resolution import choose_resolution

  loose = choose_resolution(_config(), _TIMES, 1e-2, nt_ladder=(5, 9), rtol_ladder=(1e-8, 1e-6, 1e-4))
  tight = choose_resolution(_config(), _TIMES, 1e-5, nt_ladder=(5, 9), rtol_ladder=(1e-8, 1e-6, 1e-4))
  assert tight.rtol <= loose.rtol


@pytest.mark.slow
def test_multiple_fields_must_all_meet_the_target():
  """Internal pressure runs about 80x looser than radius at identical settings, so adding
  it can only make the requirement harder, never easier.
  """
  from pyimr.resolution import choose_resolution

  ladder, tolerances = (5, 9), (1e-8, 1e-6, 1e-4)
  radius = choose_resolution(_config(), _TIMES, 1e-3, field="radius_ratio",
                             nt_ladder=ladder, rtol_ladder=tolerances)
  both = choose_resolution(_config(), _TIMES, 1e-3, field=("radius_ratio", "internal_pressure_pa"),
                           nt_ladder=ladder, rtol_ladder=tolerances)
  assert both.rtol < radius.rtol
  assert both.Nt >= radius.Nt
