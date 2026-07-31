"""Unified forward sensitivities against centered differences.

Numerical content unchanged from `run_validation.py`; see issue #32.
"""

from dataclasses import replace

import numpy as np
import pytest

import imr_fast
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

# W11 stage 5 deleted both routes this section compared. Complex step and the `Dual`
# fallback were two augmented closures over one `_rhs`, and three tests kept them
# agreeing: an RHS-level comparison, an end-to-end tangent comparison, and a gate
# asserting distributed materials stayed off the complex path because `np.cbrt` and
# `np.interp` reject complex input.
#
# `_rhs` is traced now and there is no second closure. The replacement is weaker, and it
# is recorded where it is felt: tests/test_backend_jax.py checks every tangent against a
# central difference, which resolves 1e-09 to 1e-04 rather than 1e-13. The tests above --
# material, thermal-output and collapse tangents against differences of their own -- were
# always the independent ones and are unchanged.
