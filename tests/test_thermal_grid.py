"""The medium grid's far-field singularity, and the invariant that was"""

import numpy as np
import pytest

from imr_fast import _thermal
import imr_fast
from imr_fast._thermal import _far_field_singular_index
from _validation_support import NHKV, R0, REQ

_FIELDS = ("yT", "yT2", "yT3", "iyT3", "iyT4", "iyT6")


def _medium(backend, mt, nt=5):
  config = imr_fast.SimulationConfig(R0=R0, Req=REQ, material=NHKV, bubtherm=1, medtherm=1, Nt=nt, Mt=mt, thermal=backend)
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
  """`yT` diverges only at the wall, and the inverse powers are exactly zero"""
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
  """The point of #35. The grid is now built without suppressing anything, so"""
  with np.errstate(all="raise"):
    _medium(backend, mt)


def test_guard_rejects_a_moved_singularity():
  """The guard has to be exercised directly: no public configuration can"""
  good = np.array([1.0, 0.5, 0.0, -0.5, -1.0])
  assert _far_field_singular_index(good) == 4

  with pytest.raises(ValueError, match="far-field node alone"):
    _far_field_singular_index(np.array([1.0, -1.0, 0.0, -0.5, -0.9]))  # interior singularity
  with pytest.raises(ValueError, match="far-field node alone"):
    _far_field_singular_index(np.array([1.0, 0.5, 0.0, -0.5, -0.9]))  # none at all
  with pytest.raises(ValueError, match="far-field node alone"):
    _far_field_singular_index(np.array([1.0, -1.0, 0.0, -0.5, -1.0]))  # two


_DISSIPATION_CASES = (
  ("instantaneous", imr_fast.InstantaneousMaterial(elastic=imr_fast.Gent(2500.0, 250.0), viscous=imr_fast.Newtonian(0.1))),
  ("distributed", imr_fast.Giesekus(0.1, 80e-6, 16e-6, 0.2, points=12)),
  ("closed form", NHKV),
  ("quadratic kelvin-voigt", imr_fast.QuadraticKelvinVoigt(2500.0, 0.1, 0.25)),
  ("no stress", imr_fast.NoStress()),
)


@pytest.mark.parametrize("backend", ("fd", "spectral"))
@pytest.mark.parametrize("label,material", _DISSIPATION_CASES, ids=[c[0] for c in _DISSIPATION_CASES])
def test_medium_dissipation_needs_no_suppression(label, material, backend):
  """The dissipation paths read yT, which is +inf at the wall, and used to"""
  config = imr_fast.SimulationConfig(R0=R0, Req=REQ, material=material, bubtherm=1, medtherm=1, masstrans=1, vapor=1, Nt=9, Mt=9, thermal=backend)
  with np.errstate(divide="raise", invalid="raise", over="raise"):
    imr_fast.simulate(np.linspace(0.0, 4e-6, 20), config)


def _prepared_wall_inputs(Mt=9):
  config = imr_fast.SimulationConfig(R0=R0, Req=REQ, material=NHKV, bubtherm=1, medtherm=1, Nt=9, Mt=Mt, thermal="fd")
  problem = imr_fast.prepare(config)
  medium = problem.medium
  assert medium is not None
  parameters = problem.parameters
  return (parameters["alpha_g"], parameters["beta_g"], np.asarray(medium.grad_Tm), np.asarray(medium.grad_Trans))


@pytest.mark.parametrize("scale", (0.0, 1e-3, 0.05, -0.02))
def test_wall_temperature_satisfies_its_own_boundary_condition(scale):
  """`_wall_theta_bw` solves the flux match in closed form (#57), so what needs"""
  alpha_g, beta_g, grad_Tm, grad_Trans = _prepared_wall_inputs()
  theta_tail = scale * np.arange(1.0, grad_Trans.size)
  Tm_tail = 1.0 + scale * np.arange(1.0, grad_Tm.size)

  theta_bw = _thermal._wall_theta_bw(theta_tail, Tm_tail, alpha_g, beta_g, grad_Tm, grad_Trans)

  Tw = _thermal.kirchhoff_temperature(theta_bw, alpha_g, beta_g)
  residual = grad_Tm[0] * Tw + np.sum(grad_Tm[1:] * Tm_tail)
  residual += grad_Trans[0] * theta_bw + np.sum(grad_Trans[1:] * theta_tail)
  assert abs(residual) < 1e-12 * max(1.0, abs(grad_Tm[0]))
  assert (alpha_g + beta_g) ** 2 + 2.0 * alpha_g * theta_bw >= 0.0, "root taken on the unphysical branch"


@pytest.mark.parametrize("ratio", (0.1, 0.3, 0.5))
def test_coupled_solve_survives_awkward_retardation_ratios(ratio):
  """These retardation ratios aborted the integration outright (#57): the secant"""
  relaxation = 2.0 * R0 / np.sqrt(101325 / 1064)
  config = imr_fast.SimulationConfig(
    R0=R0, Req=REQ, material=imr_fast.OldroydB(0.1, relaxation, ratio * relaxation), bubtherm=1, medtherm=1, Nt=9, Mt=9, thermal="fd"
  )
  result = imr_fast.simulate(np.linspace(0.0, 6e-5, 60), config)
  assert result.stats.success
  assert result.medium_temperature_k is not None
  assert np.all(np.isfinite(result.medium_temperature_k))


_ULP_HOSTILE_MT = (50, 99, 104, 108, 162, 188, 197, 198, 207, 215, 238, 240, 250, 254, 323, 348, 375, 390, 393, 395)


@pytest.mark.parametrize("Mt", _ULP_HOSTILE_MT)
def test_medium_grid_lands_on_the_far_field_node_exactly(Mt):
  """The far-field check is only meaningful if the grid can satisfy it. A"""
  config = imr_fast.SimulationConfig(R0=R0, Req=REQ, material=NHKV, bubtherm=1, medtherm=1, Nt=9, Mt=Mt, thermal="fd")
  medium = imr_fast.prepare(config).medium
  assert medium is not None
  assert _far_field_singular_index(np.asarray(medium.xi)) == Mt - 1


@pytest.mark.parametrize("Mt", (25, 50, 99))
def test_medium_solves_run_at_ulp_hostile_sizes(Mt):
  """End to end: these sizes raised ValueError from `prepare` before the fix."""
  config = imr_fast.SimulationConfig(R0=R0, Req=REQ, material=NHKV, bubtherm=1, medtherm=1, Nt=9, Mt=Mt, thermal="fd")
  result = imr_fast.simulate(np.linspace(0.0, 2e-5, 40), config)
  assert np.all(np.isfinite(result.radius_ratio))


@pytest.mark.parametrize("alpha", (0.05, 0.3, 0.5761, 1.1, 2.0))
@pytest.mark.parametrize("beta", (0.02, 0.4263, 1.0))
def test_kirchhoff_transform_round_trips(alpha, beta):
  """`theta` exists to satisfy `d(theta)/dT = alpha*T + beta`. Everything else in"""
  temperature = np.linspace(0.2, 20.0, 2000)
  recovered = _thermal.kirchhoff_temperature(_thermal.kirchhoff_theta(temperature, alpha, beta), alpha, beta)
  assert float(np.nanmax(np.abs(recovered - temperature))) < 1e-13


def test_the_transform_derivative_is_the_conductivity_the_rhs_uses(measured):
  """The defining property, checked against the expression `_rhs` evaluates."""
  alpha, beta, temperature, step = 0.5761, 0.4263, 3.0, 1e-6
  slope = (_thermal.kirchhoff_theta(temperature + step, alpha, beta) - _thermal.kirchhoff_theta(temperature - step, alpha, beta)) / (2 * step)
  conductivity = alpha * temperature + beta
  measured("d(theta)/dT vs K*(T)", f"{slope:.9f} vs {conductivity:.9f}")
  assert abs(slope - conductivity) < 1e-9


def test_wall_vapor_fraction_inverts_in_closed_form():
  """`_T_of_kv` is what turns admissibility into a constant bracket, so it has to"""
  P, T8, ratio, scale = 1.4, 298.15, 461.0 / 287.0, 101325.0
  fractions = np.concatenate([np.geomspace(1e-10, 1e-3, 40), np.linspace(1e-3, 1.0 - 1e-12, 400)])
  recovered = _thermal._kv_of_T(_thermal._T_of_kv(fractions, P, T8, ratio, scale), P, T8, ratio, scale)
  assert float(np.max(np.abs(recovered - fractions))) < 1e-12


def test_every_wall_solve_returns_a_physical_mass_fraction():
  """The property the bracket buys, asserted over a full collapse rather than at"""
  solves = []
  original = _thermal._bracketed_root

  def record(residual, **options):
    root = original(residual, **options)
    if options.get("xp", np) is np: solves.append(float(root))
    return root

  _thermal._bracketed_root = record
  try:
    config = imr_fast.SimulationConfig(R0=R0, Req=REQ, material=NHKV, bubtherm=1, vapor=1, masstrans=1, medtherm=1, Nt=11, Mt=11, thermal="fd")
    imr_fast.simulate(np.linspace(0.0, 25e-6, 200), config)
  finally:
    _thermal._bracketed_root = original
  fractions = np.array(solves)
  assert fractions.size >= 200, f"only {fractions.size} wall solves -- is masstrans still reaching the closure?"
  outside = int(np.sum((fractions <= 0.0) | (fractions >= 1.0)))
  assert outside == 0, f"{outside} of {fractions.size} wall solves left the physical range, worst {fractions.min():.3g} to {fractions.max():.3g}"


def test_the_wall_residual_has_one_root_on_the_bracket():
  """Uniqueness is what lets the bracket stand in for a branch choice. If the"""
  captured = []
  original = _thermal._bracketed_root

  def record(residual, **options):
    if options.get("xp", np) is np: captured.append(residual)
    return original(residual, **options)

  _thermal._bracketed_root = record
  try:
    config = imr_fast.SimulationConfig(R0=R0, Req=REQ, material=NHKV, bubtherm=1, vapor=1, masstrans=1, medtherm=1, Nt=9, Mt=9, thermal="fd")
    imr_fast.simulate(np.linspace(0.0, 25e-6, 60), config)
  finally:
    _thermal._bracketed_root = original
  grid = np.linspace(1e-13, 1.0 - 1e-13, 401)
  for residual in captured[::37]:
    values = np.array([float(residual(point)) for point in grid])
    assert np.all(np.isfinite(values)), "residual is not finite across the bracket"
    changes = int(np.sum(np.signbit(values[:-1]) != np.signbit(values[1:])))
    assert changes == 1, f"{changes} sign changes on the bracket, expected exactly 1"


def test_a_residual_with_no_admissible_root_says_so():
  """The raise is the alternative to returning a mass fraction outside `(0, 1)`,"""
  with pytest.raises(RuntimeError, match=r"no admissible wall vapour fraction.*residual"):
    _thermal._bracketed_root(lambda kv: 1.0 + kv)
