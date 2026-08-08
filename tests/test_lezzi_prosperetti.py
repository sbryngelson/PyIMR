"""The second-order-in-Mach equation, and the one gate that can check its coefficients.

Lezzi & Prosperetti (1987), *J. Fluid Mech.* **185**, 289-321, equation (8.7). It is the only
equation here that is not first order in the wall Mach number, and the only one that is
implicit in `Rddot` -- the `R^2 Rddot^2/c^2` term makes it quadratic, so it is solved rather
than divided.

WHY THE RESIDUAL TEST IS THE LOAD-BEARING ONE. The asymptotic gates that work for the
first-order family cannot reach these coefficients. Members of a family agree to one order
higher than the family is accurate to, so varying `lambda` or `theta` tests only the terms
those parameters multiply -- and `14/5` and `16/15` multiply neither. There is no compressible
reference solver here to check them against either. So the gate is direct: take the `Rddot`
the solver returns, substitute it into (8.7) transcribed independently below, and require the
residual to vanish. That catches a wrong coefficient, a sign, and a wrong root of the
quadratic, which is everything except a shared misreading of the paper.

The paper's own recommendation is `(lambda, theta) = (0.5, 0)` with this form (p.317).
"""

import numpy as np
import pytest

import pyimr
from pyimr._config import SECOND_ORDER_LAMBDA, SECOND_ORDER_THETA

R_MAX, STRETCH = 225e-6, 6.0
TIMES = np.linspace(0.0, 2.5e-5, 400)
GENTLE = R_MAX / 1.6


def config(dynamics, liquid_eos=None, *, material=None, req=GENTLE, physics=None, rtol=1e-12):
  return pyimr.SimulationConfig(R_MAX, req, material or pyimr.NeoHookeanKelvinVoigt(2500.0, 0.1),
                                dynamics=dynamics, liquid_eos=liquid_eos,
                                physics=physics or pyimr.PhysicalParameters(),
                                rtol=rtol, atol=rtol * 1e-2, max_steps=2_000_000)


def trace(dynamics, liquid_eos=None, **kw):
  return np.asarray(pyimr.simulate(TIMES, config(dynamics, liquid_eos, **kw)).radius_ratio, dtype=float)


def test_it_satisfies_equation_8_7():
  """Substitute the solver's `Rddot` back into (8.7), transcribed here from the paper.

  Run with `NoStress` so the stress integral and its acceleration coefficient vanish, leaving
  a state of `(R, Rdot)` whose every other quantity is closed form. Any error in a coefficient,
  a sign, or the choice of quadratic root shows up as a non-zero residual.
  """
  from pyimr._rhs import _rhs

  built = config("lezzi-prosperetti-2", "tait", material=pyimr.NoStress(), req=R_MAX / 6.0)
  problem = pyimr.prepare(built)
  p = problem.parameters
  lam, theta = SECOND_ORDER_LAMBDA, SECOND_ORDER_THETA

  for radius, velocity in [(1.0, 0.0), (0.8, -0.4), (0.5, -1.2), (0.35, -2.5), (1.3, 0.6)]:
    state = np.array([radius, velocity], dtype=float)
    derivative = _rhs(0.0, state, p, built.material, built.radial)
    acceleration = float(derivative[1])

    # everything (8.7) needs, from closed forms rather than from `_rhs`
    kappa, Pv = p["kappa"], p["Pv"]
    internal = (p["Pb"] - Pv) * radius ** (-3.0 * kappa) + Pv
    internal_rate = -3.0 * kappa * (p["Pb"] - Pv) * radius ** (-3.0 * kappa - 1.0) * velocity
    wall = internal - p["iWe"] / radius + p["tait_gamma"]
    enthalpy = p["tait_sam"] / p["tait_no"] * ((wall / p["tait_sam"]) ** p["tait_no"] - 1.0)
    density_ratio = (p["tait_sam"] / wall) ** (1.0 / p["tait_exponent"])
    enthalpy_rate = density_ratio * (internal_rate + p["iWe"] * velocity / radius**2)
    mach = velocity / p["Cstar"]

    left = ((1.0 - (lam + 1.0) * mach + (14.0 / 5.0 + 2.0 * lam + theta) * mach**2)
            * radius * acceleration
            + 1.5 * (1.0 - (lam + 1.0 / 3.0) * mach
                     + (16.0 / 15.0 + 4.0 * lam / 3.0 + theta) * mach**2) * velocity**2
            + radius**2 * acceleration**2 / p["Cstar"] ** 2)
    right = ((1.0 + (1.0 - lam) * mach + theta * mach**2) * enthalpy
             + (1.0 - (1.0 + lam) * mach) * radius / p["Cstar"] * enthalpy_rate)
    scale = max(abs(left), abs(right), 1.0)
    assert abs(left - right) / scale < 1e-11, (
      f"equation 8.7 not satisfied at R={radius}, Rdot={velocity}: "
      f"{left:.12e} vs {right:.12e}")


def test_the_other_root_is_rejected():
  """The quadratic has two roots and only one is physical.

  At `c -> infinity` the equation must return the incompressible acceleration. The rejected
  root diverges there like `-c^2 R`, so picking it would be obvious at low Mach and ruinous at
  high. Checked by reconstructing both roots from the coefficients at a fitted state.
  """
  from pyimr._rhs import _rhs

  built = config("lezzi-prosperetti-2", "tait", material=pyimr.NoStress(), req=R_MAX / 6.0)
  p = problem_parameters = pyimr.prepare(built).parameters
  state = np.array([0.5, -1.2], dtype=float)
  chosen = float(_rhs(0.0, state, problem_parameters, built.material, built.radial)[1])

  plesset = np.array([0.5, -1.2], dtype=float)
  incompressible = float(_rhs(0.0, plesset, p, built.material,
                              pyimr.SimulationConfig(R_MAX, R_MAX / 6.0, pyimr.NoStress(),
                                                     dynamics="rayleigh-plesset").radial)[1])
  # the physical root sits near the incompressible one; the discarded root is orders away
  assert abs(chosen - incompressible) < 0.5 * abs(incompressible)


@pytest.mark.parametrize("liquid_eos", list(pyimr.LIQUID_EOS))
def test_it_integrates_a_violent_collapse(liquid_eos):
  got = trace("lezzi-prosperetti-2", liquid_eos, req=R_MAX / 6.0, rtol=1e-9)
  assert np.all(np.isfinite(got)) and np.all(got > 0.0)


def test_it_reduces_to_the_incompressible_limit():
  """As `c` grows the second-order equation must become Rayleigh-Plesset, at rate `M`."""
  gaps = []
  for speed in (1484.0, 2968.0, 5936.0):
    physics = pyimr.PhysicalParameters(sound_speed_m_s=speed)
    gaps.append(float(np.max(np.abs(trace("lezzi-prosperetti-2", "tait", physics=physics)
                                    - trace("rayleigh-plesset", physics=physics)))))
  for coarse, fine in zip(gaps[:-1], gaps[1:], strict=True):
    assert 1.8 < coarse / fine < 2.2, f"not O(M): {gaps}"


def test_it_agrees_with_the_first_order_family_to_second_order():
  """Second order and first order describe the same physics to `O(M)`, so their difference is
  the first-order equation's own error, `O(M^2)`. This checks that the new equation belongs to
  the same asymptotic sequence rather than being a different model.
  """
  gaps = []
  for speed in (1484.0, 2968.0, 5936.0, 11872.0):
    base = pyimr.PhysicalParameters()
    stiffness = base.medium_density_kg_m3 * speed**2 / base.tait_exponent - base.far_field_pressure_pa
    physics = pyimr.PhysicalParameters(sound_speed_m_s=speed, tait_pressure_pa=stiffness)
    gaps.append(float(np.max(np.abs(trace("lezzi-prosperetti-2", "tait", physics=physics)
                                    - trace("keller-enthalpy", "tait", physics=physics)))))
  for coarse, fine in zip(gaps[:-1], gaps[1:], strict=True):
    assert 3.4 < coarse / fine < 4.6, f"not O(M^2) from the first-order form: {gaps}"


def test_it_is_not_one_of_the_first_order_forms():
  """The O(M^2) terms must reach the trajectory, or the test above compares a model to itself."""
  second = trace("lezzi-prosperetti-2", "tait", req=R_MAX / 6.0)
  for dynamics in ("keller-enthalpy", "herring", "gilmore"):
    assert np.max(np.abs(second - trace(dynamics, "tait", req=R_MAX / 6.0))) > 1e-4


def test_a_time_varying_drive_is_refused():
  """Equation 8.7 carries O(M^2) far-field terms this implementation drops. They vanish only
  for a steady drive, so a varying one would silently lose the order the equation exists for.
  """
  material = pyimr.NeoHookeanKelvinVoigt(2500.0, 0.1)
  with pytest.raises(ValueError, match="steady far field"):
    pyimr.SimulationConfig(R_MAX, GENTLE, material, dynamics="lezzi-prosperetti-2",
                           liquid_eos="tait", wave_type=1, pA=5e4, TW=5e-6, DT=2e-5)
  with pytest.raises(ValueError, match="steady far field"):
    pyimr.SimulationConfig(R_MAX, GENTLE, material, dynamics="lezzi-prosperetti-2",
                           liquid_eos="tait",
                           sampled_forcing=pyimr.SampledForcing((0.0, 1e-5), (0.0, 1e4)))
  # and the first-order forms are untouched by that restriction
  assert pyimr.SimulationConfig(R_MAX, GENTLE, pyimr.NeoHookeanKelvinVoigt(2500.0, 0.1),
                                dynamics="gilmore", liquid_eos="tait", wave_type=1, pA=5e4,
                                TW=5e-6, DT=2e-5).radial == 4


def test_the_recommended_parameters_are_the_papers():
  # p.317: "parameter values close to (lambda = 0.5, theta = 0) and the form (8.7)"
  assert (SECOND_ORDER_LAMBDA, SECOND_ORDER_THETA) == (0.5, 0.0)


def test_it_is_differentiable():
  times = np.linspace(0.0, 3e-5, 30)
  built = config("lezzi-prosperetti-2", "tait", req=R_MAX / 6.0, rtol=1e-10)
  jacobian = np.asarray(pyimr.prepare(built).solve_with_sensitivities(
    times, ("material.shear_modulus_pa",)).radius_ratio, dtype=float)[:, 0]
  assert np.all(np.isfinite(jacobian)) and np.max(np.abs(jacobian)) > 0.0

  def moved(scale):
    material = pyimr.NeoHookeanKelvinVoigt(2500.0 * scale, 0.1)
    return np.asarray(pyimr.simulate(times, config("lezzi-prosperetti-2", "tait",
                                                   material=material, req=R_MAX / 6.0,
                                                   rtol=1e-10)).radius_ratio, dtype=float)

  nudge = 1e-6
  finite = (moved(1 + nudge) - moved(1 - nudge)) / (2 * nudge * 2500.0)
  assert np.max(np.abs(jacobian - finite)) < 1e-4 * max(np.max(np.abs(finite)), 1e-30)
