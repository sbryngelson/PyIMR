"""Naming the bubble-dynamics equations, on the two axes they actually have.

`_rhs.py` carries one enthalpy-form equation. Every enthalpy code reaches the same `num`/`den`
and differs only in the Prosperetti-Lezzi `lambda`, in which equation of state supplies `hB`,
`hH`, and in whether the sound speed is the constant `Cstar` or the local wall value. Worse, `keller-miksis`
named both the pressure form and the constant-sound-speed enthalpy form, which are different
equations. So the pair is the interface now, the integer is derived and unsettable, and these
tests pin what that buys: the pairing is checked, the 2x2 is complete, and each axis moves
the trajectory on its own.
"""

import dataclasses

import numpy as np
import pytest

import pyimr

MATERIAL = pyimr.NeoHookeanKelvinVoigt(2500.0, 0.1)
TIMES = np.linspace(0.0, 3e-5, 20)


def config(**kw):
  return pyimr.SimulationConfig(225e-6, 37.5e-6, MATERIAL, rtol=1e-7, atol=1e-9, **kw)


def trace(**kw):
  return np.asarray(pyimr.simulate(TIMES, config(**kw)).radius_ratio, dtype=float)


def test_the_table_is_the_product_of_its_axes():
  """Six dynamics, four of which take an equation of state, is exactly fourteen operators.

  Stated as a product rather than a count, so adding a dynamics or an EOS without extending
  the code table fails here rather than silently dropping a combination.
  """
  expected = {(d, None) for d in pyimr.DYNAMICS if d not in pyimr.NEEDS_EOS}
  expected |= {(d, e) for d in pyimr.NEEDS_EOS for e in pyimr.LIQUID_EOS}
  assert set(pyimr.OPERATORS) == expected
  assert len(pyimr.OPERATORS) == len(set(pyimr.OPERATORS)) == 14


@pytest.mark.parametrize(("dynamics", "liquid_eos"), pyimr.OPERATORS)
def test_every_operator_resolves_to_a_distinct_code_and_solves(dynamics, liquid_eos):
  built = config(dynamics=dynamics, liquid_eos=liquid_eos)
  assert built.dynamics == dynamics and built.liquid_eos == liquid_eos
  assert built.radial in range(1, 15)
  assert np.all(np.isfinite(trace(dynamics=dynamics, liquid_eos=liquid_eos)))


def test_the_codes_are_distinct_across_the_whole_table():
  codes = [config(dynamics=d, liquid_eos=e).radial for d, e in pyimr.OPERATORS]
  assert sorted(codes) == list(range(1, 15)), "the pairs must cover the codes exactly"


def test_the_derived_code_cannot_be_set():
  """`radial` is an implementation detail now. A caller that passes it must be told so,
  rather than have it silently ignored while the pair decides.
  """
  with pytest.raises(TypeError):
    config(dynamics="keller-miksis", radial=6)


def test_a_config_survives_a_replace():
  """`dataclasses.replace` re-runs `__post_init__` over every field including the derived
  one. An `init=False` field is the reason that works; a plain field would collide.
  """
  base = config(dynamics="gilmore", liquid_eos="tait")
  moved = dataclasses.replace(base, rtol=1e-6)
  assert (moved.dynamics, moved.liquid_eos, moved.radial) == ("gilmore", "tait", base.radial)


@pytest.mark.parametrize(("kw", "message"), [
  ({"dynamics": "gilmore"}, "needs a liquid_eos"),
  ({"dynamics": "keller-enthalpy"}, "needs a liquid_eos"),
  ({"dynamics": "keller-miksis", "liquid_eos": "tait"}, "takes no liquid_eos"),
  ({"dynamics": "rayleigh-plesset", "liquid_eos": "mie-gruneisen"}, "takes no liquid_eos"),
  ({"dynamics": "gilmore", "liquid_eos": "stiffened-gas"}, "unknown liquid_eos"),
  ({"dynamics": "gilmore-mie"}, "unknown dynamics"),
  ({"dynamics": "keller-miksis-tait"}, "unknown dynamics"),
])
def test_an_impossible_pairing_is_refused(kw, message):
  """Half the value of splitting the axes: these were unsayable before, so they could not be
  refused. The last two are the retired flat names, which must not quietly work as dynamics.
  """
  with pytest.raises(ValueError, match=message):
    config(**kw)


def test_the_message_lists_the_choices():
  with pytest.raises(ValueError, match="rayleigh-plesset"):
    config(dynamics="keller")
  with pytest.raises(ValueError, match="mie-gruneisen"):
    config(dynamics="gilmore", liquid_eos="stiffened-gas")


def test_each_axis_moves_the_trajectory_on_its_own():
  """The 2x2 must be four distinct answers, not two names for two equations.

  If the EOS were inert the rows would coincide; if the sound-speed switch were inert the
  columns would. Both comparisons are made at a fixed value of the other axis, which is the
  only way to attribute the difference to one axis.
  """
  grid = {(d, e): trace(dynamics=d, liquid_eos=e)
          for d in ("keller-enthalpy", "gilmore") for e in pyimr.LIQUID_EOS}
  for dynamics in ("keller-enthalpy", "gilmore"):        # the EOS axis, dynamics held
    gap = np.max(np.abs(grid[(dynamics, "tait")] - grid[(dynamics, "mie-gruneisen")]))
    assert gap > 1e-9, f"the equation of state does nothing under {dynamics}"
  for eos in pyimr.LIQUID_EOS:                           # the dynamics axis, EOS held
    gap = np.max(np.abs(grid[("keller-enthalpy", eos)] - grid[("gilmore", eos)]))
    assert gap > 1e-9, f"the sound-speed treatment does nothing under {eos}"


def test_the_pressure_form_is_not_the_enthalpy_form():
  """The naming defect this refactor exists to fix.

  `keller-miksis` used to name both of these. They are different equations -- one carries the
  wall pressure, the other a wall enthalpy from an equation of state -- and if they happened
  to agree to round-off the old name would have been merely redundant rather than wrong.
  """
  pressure = trace(dynamics="keller-miksis")
  enthalpy = trace(dynamics="keller-enthalpy", liquid_eos="tait")
  assert np.max(np.abs(pressure - enthalpy)) > 1e-9


def test_operator_name_round_trips_the_pair():
  assert pyimr.operator_name("gilmore", "tait") == "gilmore/tait"
  assert pyimr.operator_name("rayleigh-plesset", None) == "rayleigh-plesset"
  labels = [pyimr.operator_name(*operator) for operator in pyimr.OPERATORS]
  assert len(set(labels)) == len(labels), "labels must distinguish the operators they name"


def test_the_default_is_still_rayleigh_plesset():
  assert (config().dynamics, config().liquid_eos, config().radial) == ("rayleigh-plesset", None, 1)


def test_the_selection_table_is_the_same_object():
  # two copies of this would drift; `pyimr.selection` re-exports rather than repeats
  from pyimr.selection import DYNAMICS_MODELS

  assert DYNAMICS_MODELS is pyimr.OPERATORS
