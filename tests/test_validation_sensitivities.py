"""Unified forward sensitivities against centered differences."""

from dataclasses import replace

import numpy as np
import pytest

import pyimr
from _validation_support import NHKV, R0, REQ

SECTION = "3. Unified forward sensitivities"

_TIMES = np.linspace(0.0, 20e-6, 80)


def _material_offset(config, field, amount):
  return replace(config, material=replace(config.material, **{field: getattr(config.material, field) + amount}))


def _centered_output(times, config, field, step, output):
  ahead = output(pyimr.simulate(times, _material_offset(config, field, step)))
  behind = output(pyimr.simulate(times, _material_offset(config, field, -step)))
  return (ahead - behind) / (2.0 * step)


@pytest.mark.parametrize(("dynamics", "liquid_eos"), pyimr.OPERATORS)
def test_material_tangent_matches_centered_difference(dynamics, liquid_eos, measured):
  config = pyimr.SimulationConfig(R0, REQ, NHKV, dynamics=dynamics, liquid_eos=liquid_eos, rtol=1e-10, atol=1e-12)
  tangent = pyimr.simulate_with_sensitivities(_TIMES, config, ["material.shear_modulus_pa"]).radius_ratio[:, 0]
  difference = _centered_output(_TIMES, config, "shear_modulus_pa", 25.0, lambda result: result.radius_ratio)
  error = float(np.linalg.norm(tangent - difference) / np.linalg.norm(difference))
  measured(f"{pyimr.operator_name(dynamics, liquid_eos)} material tangent", f"rel={error:.2e}")
  assert error < 2e-4


def test_coupled_heat_mass_transfer_output_tangent(measured):
  """Two orders looser than the mechanical tangents above. The cause is time"""
  config = pyimr.SimulationConfig(R0, REQ, NHKV, bubtherm=1, medtherm=1, masstrans=1, vapor=1, Nt=7, Mt=7, rtol=1e-9, atol=1e-11)
  times = np.linspace(0.0, 2e-6, 8)
  sensitivity = pyimr.simulate_with_sensitivities(times, config, ["material.shear_modulus_pa"])
  difference = _centered_output(times, config, "shear_modulus_pa", 0.025, lambda result: result.medium_temperature_k)
  assert sensitivity.medium_temperature_k is not None
  error = float(np.linalg.norm(sensitivity.medium_temperature_k[..., 0] - difference) / np.linalg.norm(difference))
  measured("medium temperature tangent", f"rel={error:.2e}")
  assert error < 2e-3


def test_collapse_shooting_tangent(measured):
  config = pyimr.SimulationConfig(R0, REQ, pyimr.Zener(2500.0, 0.1, 40e-6, 8e-6), dynamics="keller-miksis", collapse=pyimr.CollapseInitialization())
  tangent = pyimr.simulate_with_sensitivities(np.array([0.0, 1e-8]), config, ["material.shear_modulus_pa"]).state[0, -1, 0]
  step = 0.025
  difference = (
    pyimr.prepare(_material_offset(config, "shear_modulus_pa", step)).initial_state[-1]
    - pyimr.prepare(_material_offset(config, "shear_modulus_pa", -step)).initial_state[-1]
  ) / (2.0 * step)
  error = abs(tangent - difference) / abs(difference)
  stats = pyimr.prepare(config).collapse_stats
  assert stats is not None  # collapse=CollapseInitialization() above is what populates it
  residual = abs(stats.maximum_radius_ratio - 1.0)
  measured("initial memory tangent", f"rel={error:.2e}  shooting residual={residual:.2e}")
  assert error < 1e-5
  assert residual < 2e-8



def _initial_offset(config, field, amount):
  return replace(config, initial=replace(config.initial, **{field: getattr(config.initial, field) + amount}))


def test_an_off_equilibrium_start_still_yields_a_tangent(measured):
  """A deliberate start off thermal equilibrium drove the solver through unphysical states (#133).

  0.804203 is saturation at T8=298.15 K, so pairing it with 300 K is out of equilibrium on
  purpose -- the state the solver used to choke on, now stated explicitly rather than inherited.
  """
  config = pyimr.SimulationConfig(
    R0, REQ, NHKV, bubtherm=1, masstrans=1, vapor=1, Nt=15, rtol=1e-9, atol=1e-11,
    initial=pyimr.InitialState(bubble_temperature_k=300.0, vapor_mass_fraction=0.804203),
  )
  times = np.linspace(0.0, 2e-7, 6)
  tangent = pyimr.simulate_with_sensitivities(times, config, ["material.shear_modulus_pa"]).radius_ratio[:, 0]
  assert np.all(np.isfinite(tangent))
  difference = _centered_output(times, config, "shear_modulus_pa", 25.0, lambda result: result.radius_ratio)
  error = float(np.linalg.norm(tangent - difference) / np.linalg.norm(difference))
  measured("off-equilibrium start tangent", f"rel={error:.2e}")
  assert error < 1e-4
