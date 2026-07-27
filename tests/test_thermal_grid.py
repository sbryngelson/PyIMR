"""The medium grid's far-field singularity, and the invariant that was
previously only asserted in a comment.

`xi = -1` exactly at the far-field node makes `2 / (xi + 1)` infinite. That is a
property of the grid map, not an accident, and the limits are exact: `yT -> inf`
with its inverse powers `-> 0`. What was never checked is *where* the
singularity is. A suppressed `np.errstate` produces `inf` wherever a node
happens to land on `-1` and says nothing. See issue #35.
"""

import numpy as np
import pytest

import imr_fast
from _imr_thermal import _far_field_singular_index
from _validation_support import NHKV, R0, REQ

_FIELDS = ("yT", "yT2", "yT3", "iyT3", "iyT4", "iyT6")


def _medium(backend, mt, nt=5):
  config = imr_fast.SimulationConfig(
    R0=R0, Req=REQ, material=NHKV, bubtherm=1, medtherm=1, Nt=nt, Mt=mt, thermal=backend
  )
  medium = imr_fast.prepare(config).medium
  assert medium is not None, "medtherm=1 must produce medium operators"
  return medium


@pytest.mark.parametrize("backend", ("fd", "spectral"))
@pytest.mark.parametrize("mt", (5, 9, 25))
def test_far_field_singularity_is_the_last_node_alone(backend, mt):
  xi = np.asarray(_medium(backend, mt).xi)
  singular = np.flatnonzero(xi + 1.0 == 0.0)
  assert singular.tolist() == [xi.size - 1], (
    f"{backend}/Mt={mt}: the grid map's singularity moved. The wall closure assumes it is the last node alone."
  )


@pytest.mark.parametrize("backend", ("fd", "spectral"))
@pytest.mark.parametrize("mt", (5, 9, 25))
def test_far_field_limits_are_exact(backend, mt):
  """`yT` diverges only at the wall, and the inverse powers are exactly zero
  there -- not merely small, since they are consumed by a dot product whose
  last weight is not always zero."""
  medium = _medium(backend, mt)
  values = {name: np.asarray(getattr(medium, name)) for name in _FIELDS}
  for name in ("yT", "yT2", "yT3"):
    assert np.all(np.isfinite(values[name][:-1])), f"{name} is non-finite at an interior node"
    assert np.isinf(values[name][-1]), f"{name} should diverge at the far-field node"
  for name in ("iyT3", "iyT4", "iyT6"):
    assert np.all(np.isfinite(values[name])), f"{name} should be finite everywhere"
    assert values[name][-1] == 0.0, f"{name} should be exactly zero at the far-field node"


@pytest.mark.parametrize("backend", ("fd", "spectral"))
@pytest.mark.parametrize("mt", (5, 9, 25))
def test_preparation_raises_no_floating_point_errors(backend, mt):
  """The point of #35. The grid is now built without suppressing anything, so
  under errstate(all="raise") preparation must complete: any divide or invalid
  here is a real event rather than expected noise.

  This is what a comment cannot do. The previous form suppressed the divide
  unconditionally, so a genuine new one would have been indistinguishable from
  the expected one.
  """
  with np.errstate(all="raise"):
    _medium(backend, mt)


def test_guard_rejects_a_moved_singularity():
  """The guard has to be exercised directly: no public configuration can
  produce a bad grid, so a test that only builds real grids would pass with the
  guard deleted."""
  good = np.array([1.0, 0.5, 0.0, -0.5, -1.0])
  assert _far_field_singular_index(good) == 4

  with pytest.raises(ValueError, match="far-field node alone"):
    _far_field_singular_index(np.array([1.0, -1.0, 0.0, -0.5, -0.9]))  # interior singularity
  with pytest.raises(ValueError, match="far-field node alone"):
    _far_field_singular_index(np.array([1.0, 0.5, 0.0, -0.5, -0.9]))  # none at all
  with pytest.raises(ValueError, match="far-field node alone"):
    _far_field_singular_index(np.array([1.0, -1.0, 0.0, -0.5, -1.0]))  # two
