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


def test_the_ladders_are_ascending():
  assert list(NT_LADDER) == sorted(NT_LADDER)
  assert list(RTOL_LADDER) == sorted(RTOL_LADDER)


def test_apply_substitutes_only_the_resolution_fields():
  config = _config(pA=1234.0, radial=2)
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


def test_the_reference_returns_the_finer_solve_when_converged():
  from pyimr.resolution import _reference, _at, _solve

  reference, gap = _reference(_config(), _TIMES, ("radius_ratio",), 1e-2, (5, 9))
  assert gap <= 1e-3
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
