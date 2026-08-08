"""Herring, and the Noble-Abel stiffened gas.

Two additions on two different axes, and each has one gate that can actually catch a wrong
equation rather than wrong wiring.

`herring` is not a new theory. Prosperetti & Lezzi (1986) showed the first-order-in-Mach
equations form a one-parameter family, and that Keller and Herring are its `lambda = 0` and
`lambda = 1` members. Every member is correct to first order, so they must agree with each
other to `O(M^2)` while each differs from the incompressible limit at `O(M)`. That scaling is
the test: a mistyped coefficient still vanishes as `c -> infinity`, but at the wrong rate,
and a reduction test would pass anyway.

`nasg` is a new equation of state, and it has a genuine exact limit: at zero covolume the
Noble-Abel stiffened gas IS Tait, with `gamma` for `n` and `p_inf` for `B`. That reduction
must hold to round-off, and the covolume must then be shown to do something, or the reduction
is passing because the parameter is dead.
"""

import numpy as np
import pytest

import pyimr

MATERIAL = pyimr.NeoHookeanKelvinVoigt(2500.0, 0.1)
R_MAX = 225e-6
# a gentle collapse: the family is an expansion in the wall Mach number, and at the stretch
# used elsewhere in these tests the peak Mach is 0.16, where `O(M^2)` is not yet small
GENTLE = R_MAX / 1.6
TIMES = np.linspace(0.0, 2.5e-5, 400)
# Tait's own constants, so the two equations of state are describing one liquid
MATCHED = dict(nasg_exponent=pyimr.PhysicalParameters().tait_exponent,
               nasg_pressure_pa=pyimr.PhysicalParameters().tait_pressure_pa)


def trace(dynamics, liquid_eos=None, *, req=GENTLE, physics=None, rtol=1e-12, times=TIMES):
  config = pyimr.SimulationConfig(R_MAX, req, MATERIAL, dynamics=dynamics, liquid_eos=liquid_eos,
                                  physics=physics or pyimr.PhysicalParameters(),
                                  rtol=rtol, atol=rtol * 1e-2, max_steps=2_000_000)
  return np.asarray(pyimr.simulate(times, config).radius_ratio, dtype=float)


@pytest.mark.parametrize(("dynamics", "liquid_eos"), pyimr.OPERATORS)
def test_every_operator_integrates_a_violent_collapse(dynamics, liquid_eos):
  got = trace(dynamics, liquid_eos, req=R_MAX / 6.0, rtol=1e-9)
  assert np.all(np.isfinite(got)) and np.all(got > 0.0)


# --- the equation of state -------------------------------------------------------------

@pytest.mark.parametrize("dynamics", ["keller-enthalpy", "herring", "gilmore",
                                     "lezzi-prosperetti-2"])
def test_zero_covolume_makes_nasg_the_tait_it_contains(dynamics):
  """The one exact statement available about this equation of state.

  With `b = 0` the isentrope `(v - b)(p + p_inf)^(1/gamma) = const` becomes Tait's
  `p + B = A rho^n`, so the enthalpy, the density ratio and the sound speed must agree term
  for term -- not closely, but to round-off. Run at three members of the family because the
  closure is shared and a slip in any one of the three quantities shows up in all of them.
  """
  physics = pyimr.PhysicalParameters(**MATCHED, nasg_covolume_m3_kg=0.0)
  tait = trace(dynamics, "tait", physics=physics)
  nasg = trace(dynamics, "nasg", physics=physics)
  assert float(np.max(np.abs(tait - nasg))) < 1e-11, "NASG at zero covolume is not Tait"


def test_the_covolume_is_not_a_dead_parameter():
  """Or the reduction above passes for the reason a broken branch would pass."""
  off = trace("gilmore", "nasg", physics=pyimr.PhysicalParameters(**MATCHED, nasg_covolume_m3_kg=0.0))
  on = trace("gilmore", "nasg", physics=pyimr.PhysicalParameters(**MATCHED, nasg_covolume_m3_kg=3e-4))
  assert float(np.max(np.abs(on - off))) > 1e-3, "the covolume does nothing to the trace"


def test_the_covolume_cannot_reach_the_close_packed_limit():
  # `b rho = 1` sends the sound speed and the enthalpy to infinity; the constructor refuses it
  # rather than letting the integrator meet it
  with pytest.raises(ValueError, match="nasg_covolume"):
    pyimr.PhysicalParameters(nasg_covolume_m3_kg=1.0 / 1064.0)


def test_the_default_constants_imply_a_plausible_sound_speed():
  """A standing inconsistency, pinned so it stays known rather than becoming a surprise.

  The NASG defaults are Le Metayer & Saurel's constants for liquid WATER, used here with
  whatever medium density is configured -- exactly as the Tait defaults already are. With the
  gel density of 1064 the implied ambient sound speed is 9.7% above `sound_speed_m_s`,
  against -3.5% for Tait. NASG is the more sensitive of the two because the covolume
  enters as `1/(1 - b rho)`. The bound here is loose on purpose: it catches a change that
  breaks the constants, not the mismatch itself, which is documented rather than fixed.
  """
  physics = pyimr.PhysicalParameters()
  reference = np.sqrt(physics.far_field_pressure_pa / physics.medium_density_kg_m3)
  covolume = physics.nasg_covolume_m3_kg * physics.medium_density_kg_m3
  sam = 1.0 + physics.nasg_pressure_pa / physics.far_field_pressure_pa
  implied = np.sqrt(physics.nasg_exponent * sam / (1.0 - covolume)) * reference
  assert 0.85 < implied / physics.sound_speed_m_s < 1.15, f"NASG implies c = {implied:.0f} m/s"


# --- the Prosperetti-Lezzi family ---------------------------------------------------------

def test_the_family_members_agree_to_second_order_in_the_mach_number():
  """The gate that tests the equation rather than the plumbing.

  Every member of the family is correct to first order in `M`, so two members must agree to
  `O(M^2)` while each differs from the incompressible limit at `O(M)`. Halving `M` therefore
  halves one gap and quarters the other. Measured: the Keller-to-Rayleigh-Plesset ratios run
  1.98, 1.99, 2.00 and the Herring-to-Keller ratios 3.84, 3.92, 3.96.

  A mistyped lambda coefficient -- `(lambda+1)` for `(3 lambda+1)/3`, say -- leaves both gaps
  `O(M)` and both ratios near 2. That is what this catches and a reduction test does not: the
  wrong equation still converges to the right incompressible limit.
  """
  gaps = []
  for speed in (1500.0, 3000.0, 6000.0, 12000.0):
    physics = pyimr.PhysicalParameters(sound_speed_m_s=speed)
    plesset = trace("rayleigh-plesset", physics=physics)
    keller = trace("keller-enthalpy", "tait", physics=physics)
    herring = trace("herring", "tait", physics=physics)
    gaps.append((float(np.max(np.abs(keller - plesset))), float(np.max(np.abs(herring - keller)))))

  for (first_order, _), (halved, _) in zip(gaps[:-1], gaps[1:], strict=True):
    assert 1.8 < first_order / halved < 2.2, "the compressible correction is not O(M)"
  for (_, second_order), (_, halved) in zip(gaps[:-1], gaps[1:], strict=True):
    assert 3.4 < second_order / halved < 4.6, "the family members do not agree to O(M^2)"


def test_the_enthalpy_form_agrees_with_the_independent_pressure_form():
  """What the test above cannot do: pin the constant rather than the lambda dependence.

  Members of the family agree to `O(M^2)` whenever the `Rd^2` coefficient has the form
  `lambda + const`, for ANY constant. So the mutation `(3 lambda + 1)/3 -> (lambda + 1)`
  survives that test: it keeps the form and changes only the constant, and the members stay
  mutually consistent while all of them drift from the true first-order solution. Verified by
  running it.

  `keller-miksis` is the pressure form -- a separate branch of `_rhs`, written independently
  and matched against IMRv2 reference trajectories. It is correct to first order too, so the
  enthalpy form must agree with it to `O(M^2)`, and that comparison has no free constant in it.

  The sound speed and the equation of state must be raised together, `c^2 = n(P8 + B)/rho`.
  Holding `B` fixed while raising `c` asks the two forms about two different liquids, and the
  gap then plateaus at `1.1e-05` instead of falling.
  """
  base = pyimr.PhysicalParameters()
  gaps = []
  for speed in (1484.0, 2968.0, 5936.0, 11872.0):
    stiffness = base.medium_density_kg_m3 * speed**2 / base.tait_exponent - base.far_field_pressure_pa
    physics = pyimr.PhysicalParameters(sound_speed_m_s=speed, tait_pressure_pa=stiffness)
    plesset = trace("rayleigh-plesset", physics=physics)
    pressure = trace("keller-miksis", physics=physics)
    enthalpy = trace("keller-enthalpy", "tait", physics=physics)
    gaps.append((float(np.max(np.abs(pressure - plesset))), float(np.max(np.abs(enthalpy - pressure)))))

  for (first_order, _), (halved, _) in zip(gaps[:-1], gaps[1:], strict=True):
    assert 1.8 < first_order / halved < 2.2, "the compressible correction is not O(M)"
  for (_, second_order), (_, halved) in zip(gaps[:-1], gaps[1:], strict=True):
    assert 3.4 < second_order / halved < 4.6, "the two first-order forms do not agree to O(M^2)"


def test_herring_is_not_keller():
  """`lambda` must reach the trajectory, or the test above passes on two identical models."""
  keller = trace("keller-enthalpy", "tait", req=R_MAX / 6.0)
  herring = trace("herring", "tait", req=R_MAX / 6.0)
  assert float(np.max(np.abs(herring - keller))) > 1e-3


@pytest.mark.parametrize(("dynamics", "liquid_eos"),
                         [("herring", "tait"), ("herring", "nasg"), ("gilmore", "nasg"),
                          ("keller-enthalpy", "nasg")])
def test_the_new_operators_are_differentiable(dynamics, liquid_eos):
  """A new branch that breaks the tangent reports zero sensitivity rather than an error."""
  times = np.linspace(0.0, 3e-5, 30)
  config = pyimr.SimulationConfig(R_MAX, R_MAX / 6.0, MATERIAL, dynamics=dynamics,
                                  liquid_eos=liquid_eos, rtol=1e-9, atol=1e-11)
  jacobian = np.asarray(pyimr.prepare(config).solve_with_sensitivities(
    times, ("material.shear_modulus_pa",)).radius_ratio, dtype=float)[:, 0]
  assert np.all(np.isfinite(jacobian)) and np.max(np.abs(jacobian)) > 0.0

  def moved(scale):
    material = pyimr.NeoHookeanKelvinVoigt(2500.0 * scale, 0.1)
    return np.asarray(pyimr.simulate(times, pyimr.SimulationConfig(
      R_MAX, R_MAX / 6.0, material, dynamics=dynamics, liquid_eos=liquid_eos,
      rtol=1e-9, atol=1e-11)).radius_ratio, dtype=float)

  nudge = 1e-6
  finite = (moved(1 + nudge) - moved(1 - nudge)) / (2 * nudge * 2500.0)
  assert np.max(np.abs(jacobian - finite)) < 1e-4 * max(np.max(np.abs(finite)), 1e-30)


def test_the_new_operators_leave_the_old_codes_alone():
  """The IMRv2 reference trajectories were generated against codes 1-6 and are matched by
  number, so an operator inserted anywhere but the end would silently re-point them.
  """
  from pyimr._config import _CODES

  for pair, code in (((("rayleigh-plesset", None)), 1), (("keller-miksis", None), 2),
                     (("keller-enthalpy", "tait"), 3), (("gilmore", "tait"), 4),
                     (("keller-enthalpy", "mie-gruneisen"), 5), (("gilmore", "mie-gruneisen"), 6)):
    assert _CODES[pair] == code
