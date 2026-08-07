"""Naming the bubble-dynamics equations.

`radial` has always been an integer and the integer says nothing on sight: `radial=4` does
not tell you it is Gilmore, nor which equation of state it carries, so code and prose both
kept a decoder ring nearby. A name normalises to the same code, so nothing downstream --
dispatch, the compile key, the validator -- sees any difference.
"""

import numpy as np
import pytest

import pyimr

MATERIAL = pyimr.NeoHookeanKelvinVoigt(2500.0, 0.1)
TIMES = np.linspace(0.0, 3e-5, 20)


def config(radial):
  return pyimr.SimulationConfig(225e-6, 37.5e-6, MATERIAL, radial=radial, rtol=1e-7, atol=1e-9)


def test_every_name_maps_to_a_distinct_valid_code():
  codes = sorted(pyimr.RADIAL_MODELS.values())
  assert codes == list(range(1, 7)), "the names must cover the codes exactly, with no gaps"
  assert pyimr.RADIAL_NAMES == {code: name for name, code in pyimr.RADIAL_MODELS.items()}


@pytest.mark.parametrize(("name", "code"), list(pyimr.RADIAL_MODELS.items()))
def test_a_name_is_the_number_it_denotes(name, code):
  assert config(name).radial == code == config(code).radial


@pytest.mark.parametrize("name", ["keller-miksis", "gilmore-tait"])
def test_a_named_configuration_solves_identically(name):
  """Normalisation happens before anything else runs, so the traces must be bit-identical.
  Anything less would mean the name took a different path through the solver.
  """
  by_name = np.asarray(pyimr.simulate(TIMES, config(name)).radius_ratio, dtype=float)
  by_code = np.asarray(pyimr.simulate(TIMES, config(pyimr.RADIAL_MODELS[name])).radius_ratio, dtype=float)
  assert np.array_equal(by_name, by_code)


def test_an_unknown_name_says_what_the_choices_are():
  with pytest.raises(ValueError, match="unknown radial model"):
    config("gilmore")                       # real model, not a key: the ambiguous case
  with pytest.raises(ValueError, match="rayleigh-plesset"):
    config("keller")                        # the message must list what is available


@pytest.mark.parametrize("bad", [0, 7, 2.5])
def test_a_bad_code_is_still_refused(bad):
  # naming must not have widened what the integer path accepts
  with pytest.raises(ValueError, match="radial"):
    config(bad)


def test_the_selection_table_is_the_same_object():
  # two copies of this mapping would drift; `pyimr.selection` re-exports rather than repeats
  from pyimr.selection import DYNAMICS_MODELS

  assert DYNAMICS_MODELS is pyimr.RADIAL_MODELS
