"""Unified forward sensitivities against centered differences.

Numerical content unchanged from `run_validation.py`; see issue #32.
"""

from dataclasses import replace

import numpy as np
import pytest

from imr_fast import _complex
import imr_fast
from imr_fast import sensitivity
from imr_fast._dual import _dual_config, _dual_forcing, _dual_medium, _dual_parameters, _initial_matrix, _normalize_parameters, _rhs_physical
from _validation_support import NHKV, R0, REQ

SECTION = "3. Unified forward sensitivities"

_TIMES = np.linspace(0.0, 20e-6, 80)


def _material_offset(config, field, amount):
  return replace(config, material=replace(config.material, **{field: getattr(config.material, field) + amount}))


def _centered_output(times, config, field, step, output):
  ahead = output(imr_fast.simulate(times, _material_offset(config, field, step)))
  behind = output(imr_fast.simulate(times, _material_offset(config, field, -step)))
  return (ahead - behind) / (2.0 * step)


@pytest.mark.parametrize("radial", range(1, 6))
def test_material_tangent_matches_centered_difference(radial, measured):
  config = imr_fast.SimulationConfig(R0, REQ, NHKV, radial=radial, rtol=1e-10, atol=1e-12)
  tangent = imr_fast.simulate_with_sensitivities(_TIMES, config, ["material.shear_modulus_pa"]).radius_ratio[:, 0]
  # A one-percent material step stays in the centered-difference regime while
  # remaining above radial=5's adaptive-integration noise floor.
  difference = _centered_output(_TIMES, config, "shear_modulus_pa", 25.0, lambda result: result.radius_ratio)
  error = float(np.linalg.norm(tangent - difference) / np.linalg.norm(difference))
  measured(f"radial={radial} material tangent", f"rel={error:.2e}")
  assert error < 2e-4


def test_coupled_heat_mass_transfer_output_tangent(measured):
  """Two orders looser than the mechanical tangents above. The cause is time
  integration of the augmented state/tangent system, not an error in the
  tangent equations -- issue #24, resolved.

  The two measurements that separate those. The error is flat in the
  finite-difference step across a 16x range (8.33e-05, 8.43e-05, 8.34e-05 at
  h = 0.2, 0.05, 0.0125), which rules out truncation in the check; and at
  h = 0.05 it moves 8.43e-05 -> 1.53e-06 when rtol/atol go from 1e-9/1e-11 to
  1e-12/1e-14, which a defect in the tangent equations would not do.

  A factor of 55 for three orders of tolerance, not the "two orders" first
  claimed here: that figure came from the most favourable point of a sweep
  whose error grew to 6.9e-06 as h shrank, so it described round-off in the
  reference rather than the tangent.

  Step-independence alone is *not* enough to conclude the tangent is fine --
  it is also the signature of a broken derivative, which is how #10's
  Powell-Eyring defect and #43's spectral wall stencils both presented. The
  tolerance response is what distinguishes them.

  The tolerance here stays at 2e-3 rather than tightening to the measured
  8e-05: it bounds the physical claim, and the achieved value is reported in
  the measured-values table where a drift is visible without a failure.
  """
  config = imr_fast.SimulationConfig(R0, REQ, NHKV, bubtherm=1, medtherm=1, masstrans=1, vapor=1, Nt=7, Mt=7, rtol=1e-9, atol=1e-11)
  times = np.linspace(0.0, 2e-6, 8)
  sensitivity = imr_fast.simulate_with_sensitivities(times, config, ["material.shear_modulus_pa"])
  difference = _centered_output(times, config, "shear_modulus_pa", 0.025, lambda result: result.medium_temperature_k)
  error = float(np.linalg.norm(sensitivity.medium_temperature_k[..., 0] - difference) / np.linalg.norm(difference))
  measured("medium temperature tangent", f"rel={error:.2e}")
  assert error < 2e-3


def test_collapse_shooting_tangent(measured):
  config = imr_fast.SimulationConfig(R0, REQ, imr_fast.Zener(2500.0, 0.1, 40e-6, 8e-6), radial=2, collapse=imr_fast.CollapseInitialization())
  tangent = imr_fast.simulate_with_sensitivities(np.array([0.0, 1e-8]), config, ["material.shear_modulus_pa"]).state[0, -1, 0]
  step = 0.025
  difference = (
    imr_fast.prepare(_material_offset(config, "shear_modulus_pa", step)).initial_state[-1]
    - imr_fast.prepare(_material_offset(config, "shear_modulus_pa", -step)).initial_state[-1]
  ) / (2.0 * step)
  error = abs(tangent - difference) / abs(difference)
  residual = abs(imr_fast.prepare(config).collapse_stats.maximum_radius_ratio - 1.0)
  measured("initial memory tangent", f"rel={error:.2e}  shooting residual={residual:.2e}")
  assert error < 1e-5
  assert residual < 2e-8


_COMPLEX_CASES = (
  ("bubtherm fd", dict(bubtherm=1, Nt=7, thermal="fd")),
  ("bubtherm spectral", dict(bubtherm=1, Nt=7, thermal="spectral")),
  ("coupled fd", dict(bubtherm=1, medtherm=1, Nt=7, Mt=7, thermal="fd")),
  ("coupled spectral", dict(bubtherm=1, medtherm=1, Nt=7, Mt=7, thermal="spectral")),
)
_MASSTRANS = dict(bubtherm=1, medtherm=1, masstrans=1, vapor=1, Nt=7, Mt=7, thermal="fd")


def _dual_and_complex_rhs(problem, names):
  """Both augmented RHS closures for the same problem, built as the solver does."""
  config = problem.config
  normalized, values, scales = _normalize_parameters(config, names)
  dual_config = _dual_config(config, normalized, values, scales)
  parameters = _dual_parameters(dual_config)
  medium = _dual_medium(problem, parameters)
  forcing = _dual_forcing(dual_config, parameters)
  width = len(normalized)
  packed = _initial_matrix(problem, dual_config, parameters, width).ravel()
  dual = _rhs_physical(
    0.0,
    packed,
    problem=problem,
    config=dual_config,
    parameters=parameters,
    medium=medium,
    wall_state=imr_fast._WallState(),
    forcing=forcing,
    width=width,
  )
  fast = _complex.rhs_complex(
    0.0,
    packed,
    problem=problem,
    prepared=_complex.directions(dual_config, parameters, medium, forcing, width),
    wall_states=[imr_fast._WallState() for _ in range(width)],
    width=width,
  )
  return np.asarray(dual, dtype=float), np.asarray(fast, dtype=float)


@pytest.mark.parametrize(
  "label,options", [*[(c[0], c[1]) for c in _COMPLEX_CASES], ("coupled+mass", _MASSTRANS)], ids=[*[c[0] for c in _COMPLEX_CASES], "coupled+mass"]
)
def test_complex_rhs_matches_dual_rhs(label, options, measured):
  """Compare the two augmented RHS closures directly, not through an integration.

  This is where a mistake would live -- the complex conversion of the parameter,
  medium and forcing structures -- and one evaluation costs milliseconds. The
  end-to-end check below cannot cover `masstrans` at all: its `Dual` reference
  needs minutes even over a 5e-7 s window, because the cost is the stiff initial
  transient rather than the output window, and this repo's CI runs every marker.
  Testing the RHS instead of the trajectory keeps the coverage and drops the bill.
  """
  problem = imr_fast.prepare(imr_fast.SimulationConfig(R0, REQ, imr_fast.NeoHookeanKelvinVoigt(2500.0, 0.1), **options))
  dual, fast = _dual_and_complex_rhs(problem, ["material.shear_modulus_pa", "R0"])

  error = float(np.max(np.abs(dual - fast))) / max(float(np.max(np.abs(dual))), 1e-30)
  measured(f"complex vs dual RHS, {label}", f"rel={error:.2e}")
  assert error < 1e-9


@pytest.mark.parametrize("label,options", _COMPLEX_CASES, ids=[c[0] for c in _COMPLEX_CASES])
def test_complex_step_matches_dual_tangents(label, options, measured, monkeypatch):
  """The thermal path now carries tangents in the imaginary part rather than in
  `Dual` (#44). The two routes share `_rhs`, so they must agree to solver
  tolerance -- this is the only thing standing between a 6-25x speedup and
  silently wrong derivatives.

  A finite difference would not do: it is orders of magnitude less accurate than
  either route, so it cannot distinguish them. `Dual` is the exact reference.
  """
  config = imr_fast.SimulationConfig(R0, REQ, imr_fast.NeoHookeanKelvinVoigt(2500.0, 0.1), **options)
  problem = imr_fast.prepare(config)
  times = np.linspace(0.0, 4e-6, 6)
  names = ["material.shear_modulus_pa", "R0"]

  monkeypatch.setattr(_complex, "complex_step_supported", lambda _problem: False)
  assert not _complex.complex_step_supported(problem), "the Dual route was not actually selected"
  reference = sensitivity.solve_with_sensitivities(problem, times, names)
  monkeypatch.undo()
  assert _complex.complex_step_supported(problem), "the complex route was not actually selected"
  fast = sensitivity.solve_with_sensitivities(problem, times, names)

  exact = np.asarray(reference.radius_m, dtype=float)
  error = float(np.max(np.abs(exact - np.asarray(fast.radius_m, dtype=float)))) / max(float(np.max(np.abs(exact))), 1e-30)
  measured(f"complex vs dual, {label}", f"rel={error:.2e}")
  assert error < 1e-6


def test_distributed_materials_stay_on_the_dual_route():
  """`_distributed_dissipation` reaches np.cbrt and np.interp, which reject
  complex input, and its np.maximum clamp is not analytic. The gate is what
  keeps that from being discovered at runtime."""
  distributed = imr_fast.prepare(imr_fast.SimulationConfig(R0, REQ, imr_fast.Giesekus(0.1, 80e-6, 16e-6, 0.2, points=12), bubtherm=1, Nt=7))
  assert not _complex.complex_step_supported(distributed)
  assert _complex.complex_step_supported(
    imr_fast.prepare(imr_fast.SimulationConfig(R0, REQ, imr_fast.NeoHookeanKelvinVoigt(2500.0, 0.1), bubtherm=1, Nt=7))
  )
