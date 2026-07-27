"""The medium grid's far-field singularity, and the invariant that was
previously only asserted in a comment.

`xi = -1` exactly at the far-field node makes `2 / (xi + 1)` infinite. That is a
property of the grid map, not an accident, and the limits are exact: `yT -> inf`
with its inverse powers `-> 0`. What was never checked is *where* the
singularity is. A suppressed `np.errstate` produces `inf` wherever a node
happens to land on `-1` and says nothing. See issue #35.
"""

import inspect

import numpy as np
import pytest

from imr_fast import _thermal
import imr_fast
from imr_fast._thermal import _far_field_singular_index
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


_DISSIPATION_CASES = (
  (
    "instantaneous",
    imr_fast.InstantaneousMaterial(elastic=imr_fast.Gent(2500.0, 250.0), viscous=imr_fast.Newtonian(0.1)),
  ),
  ("distributed", imr_fast.Giesekus(0.1, 80e-6, 16e-6, 0.2, points=12)),
  ("closed form", NHKV),
  # QuadraticKelvinVoigt reaches a distinct branch of _dissipation that no other
  # test exercised: every pinned qKV case runs without medtherm. Slicing that
  # branch to the interior broke its shapes and nothing caught it.
  ("quadratic kelvin-voigt", imr_fast.QuadraticKelvinVoigt(2500.0, 0.1, 0.25)),
  ("no stress", imr_fast.NoStress()),
)


@pytest.mark.parametrize("backend", ("fd", "spectral"))
@pytest.mark.parametrize("label,material", _DISSIPATION_CASES, ids=[c[0] for c in _DISSIPATION_CASES])
def test_medium_dissipation_needs_no_suppression(label, material, backend):
  """The dissipation paths read yT, which is +inf at the wall, and used to
  suppress divide/invalid wholesale (#35).

  `_instantaneous_dissipation` formed inf/inf and overwrote the nan on the next
  line; that value is now set directly. `_distributed_dissipation`'s suppression
  turned out to be vestigial. Neither can regress silently while this runs the
  real solve with those errors raised.
  """
  config = imr_fast.SimulationConfig(
    R0=R0, Req=REQ, material=material, bubtherm=1, medtherm=1, masstrans=1, vapor=1, Nt=9, Mt=9, thermal=backend
  )
  # Underflow excluded deliberately: it fires inside SciPy's BDF `nextafter`,
  # is unrelated to this code, and NumPy ignores it by default.
  with np.errstate(divide="raise", invalid="raise", over="raise"):
    # 4 us, not 20: this only has to reach the dissipation code, and the
    # distributed+spectral combination is where the corrected Jacobian sparsity
    # is most expensive (#47). A longer window buys no coverage.
    imr_fast.simulate(np.linspace(0.0, 4e-6, 20), config)


def test_dissipation_paths_carry_no_suppression():
  """The behavioural test above cannot detect a re-added `np.errstate`: an inner
  suppression overrides the outer context, so it would pass either way. That is
  the same trap #35 is about, one level up. This checks the source directly."""
  source = inspect.getsource(_thermal)
  for name in ("_instantaneous_dissipation", "_distributed_dissipation"):
    body = source.split(f"def {name}(")[1].split("\ndef ")[0]
    assert "errstate" not in body, f"{name} suppresses floating-point errors again; see #35"


def _prepared_wall_inputs(Mt=9):
  """Real stencils and alpha_g, so the coefficient signs are the physical ones."""
  config = imr_fast.SimulationConfig(R0=R0, Req=REQ, material=NHKV, bubtherm=1, medtherm=1, Nt=9, Mt=Mt, thermal="fd")
  problem = imr_fast.prepare(config)
  medium = problem.medium
  assert medium is not None
  return problem.parameters["alpha_g"], np.asarray(medium.grad_Tm), np.asarray(medium.grad_Trans)


@pytest.mark.parametrize("scale", (0.0, 1e-3, 0.05, -0.02))
def test_wall_temperature_satisfies_its_own_boundary_condition(scale):
  """`_wall_theta_bw` solves the flux match in closed form (#57), so what needs
  guarding is the algebra and the branch choice, not convergence. This passes on
  the secant too, by construction -- it is not the #57 regression test below,
  but the thing that catches a mis-derived quadratic or the wrong root.
  """
  alpha_g, grad_Tm, grad_Trans = _prepared_wall_inputs()
  theta_tail = scale * np.arange(1.0, grad_Trans.size)
  Tm_tail = 1.0 + scale * np.arange(1.0, grad_Tm.size)

  theta_bw = _thermal._wall_theta_bw(0.0, theta_tail, Tm_tail, alpha_g, grad_Tm, grad_Trans)

  Tw = (alpha_g - 1.0 + np.sqrt(1.0 + 2.0 * theta_bw * alpha_g)) / alpha_g
  residual = grad_Tm[0] * Tw + np.sum(grad_Tm[1:] * Tm_tail)
  residual += grad_Trans[0] * theta_bw + np.sum(grad_Trans[1:] * theta_tail)
  assert abs(residual) < 1e-12 * max(1.0, abs(grad_Tm[0]))
  assert 1.0 + 2.0 * alpha_g * theta_bw >= 0.0, "root taken on the unphysical branch"


@pytest.mark.parametrize("ratio", (0.1, 0.3, 0.5))
def test_coupled_solve_survives_awkward_retardation_ratios(ratio):
  """These retardation ratios aborted the integration outright (#57): the secant
  failed on roughly one call in 4000, which is all it takes. Failure alternated
  with a smoothly varying parameter -- 0.2 and 0.4 worked -- so no single value
  is the regression test; the point is that the sensitivity to it is gone.
  """
  relaxation = 2.0 * R0 / np.sqrt(101325 / 1064)
  config = imr_fast.SimulationConfig(
    R0=R0,
    Req=REQ,
    material=imr_fast.OldroydB(0.1, relaxation, ratio * relaxation),
    bubtherm=1,
    medtherm=1,
    Nt=9,
    Mt=9,
    thermal="fd",
  )
  result = imr_fast.simulate(np.linspace(0.0, 6e-5, 60), config)
  assert result.stats.success
  assert result.medium_temperature_k is not None
  assert np.all(np.isfinite(result.medium_temperature_k))


# Mt values whose medium grid missed xi = -1 by one ulp under the accumulated
# `1 + arange(Mt) * deltaYm` construction, so `_far_field_singular_index`
# rejected them and an ordinary `medtherm=1, Mt=50` run raised outright. 20 of
# the 398 sizes in [3, 400] were affected; these are the first few.
_ULP_HOSTILE_MT = (50, 99, 104, 108, 162, 188, 197, 198, 207, 215)


@pytest.mark.parametrize("Mt", _ULP_HOSTILE_MT)
def test_medium_grid_lands_on_the_far_field_node_exactly(Mt):
  """The far-field check is only meaningful if the grid can satisfy it. A
  scheme that is correct for 95% of grid sizes and raises on the rest is a
  construction bug, not a validated invariant."""
  xi = np.linspace(1.0, -1.0, Mt)
  assert _far_field_singular_index(xi) == Mt - 1


@pytest.mark.parametrize("Mt", (25, 50, 99))
def test_medium_solves_run_at_ulp_hostile_sizes(Mt):
  """End to end: these sizes raised ValueError from `prepare` before the fix."""
  config = imr_fast.SimulationConfig(R0=R0, Req=REQ, material=NHKV, bubtherm=1, medtherm=1, Nt=9, Mt=Mt, thermal="fd")
  result = imr_fast.simulate(np.linspace(0.0, 2e-5, 40), config)
  assert np.all(np.isfinite(result.radius_ratio))
