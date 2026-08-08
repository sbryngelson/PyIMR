"""Shear-thinning viscosities as comparison candidates.

Six of these laws were implemented and exported long before any of them could be compared:
only `powerlaw` had ever been made a `CandidateModel`, so the rest could be simulated and
never scored. Registering them is a handful of lines, and every one of those lines fails
quietly -- a transposed argument still builds, a wrong `contains` yields a plausible
redundancy rather than an error, and a model that cannot integrate over its own bounds
looks exactly like one that can until the sweep runs.

So the gates here are reductions with known answers and a coverage count, not regression
values: each law is driven to the limit where it BECOMES the model it declares it contains,
and the answer must match to round-off rather than merely closely.
"""

import numpy as np
import pytest

import pyimr
from pyimr.selection import PARAMETER_BOUNDS, STANDARD_MODELS, parameter_grid

TIMES = np.linspace(0.0, 4e-5, 40)
G, MU = 204.3, 0.04651
R_MAX, STRETCH = 277e-6, 7.09
THINNING = ("carreau", "cross", "eyring", "modeyring", "herschel", "bingham")

# below every crossover the collapse can reach: `lambda * gammadot` stays negligible even at
# the shear rates a collapsing bubble produces, which is where these laws go constant
UNTIMED = 1e-18

# what to change, at any point, to switch each law's thinning off without touching anything
# else -- the same limits the reduction tests below drive it to
FLATTENED = {
  "carreau": {"pl_n": 1.0}, "cross": {"lambda1": UNTIMED},
  "eyring": {"lambda1": UNTIMED}, "modeyring": {"lambda1": UNTIMED},
  "herschel": {"yield_pa": 0.0}, "bingham": {"yield_pa": 0.0},
}


def solve(material, rtol=1e-9):
  config = pyimr.SimulationConfig(R_MAX, R_MAX / STRETCH, material, dynamics="keller-miksis",
                                  rtol=rtol, atol=rtol * 1e-2, max_steps=400_000)
  return np.asarray(pyimr.simulate(TIMES, config).radius_ratio, dtype=float)


def build(name, **fields):
  return STANDARD_MODELS[name].build(fields)


def laws(name, **fields):
  """The `(elastic, viscous)` pair a thinning candidate builds, checked to be that pair.

  Every one of these is an `InstantaneousMaterial`, and asserting it here is what lets the
  fields below be read at all -- `build` is declared to return some `MaterialModel`, and
  most members of that union have neither attribute.
  """
  material = build(name, **fields)
  assert isinstance(material, pyimr.InstantaneousMaterial), f"{name} should build an instantaneous material"
  return material.elastic, material.viscous


@pytest.fixture(scope="module")
def kelvin_voigt():
  return solve(pyimr.NeoHookeanKelvinVoigt(G, MU))


# each law, and the parameters that send it to a constant viscosity `MU`
@pytest.mark.parametrize(("name", "fields"), [
  ("carreau", {"mu": MU, "g": G, "lambda1": 1e-6, "pl_n": 1.0}),        # unit power index
  ("cross", {"mu": MU, "g": G, "lambda1": UNTIMED, "cross_m": 1.0}),    # crossover out of reach
  ("eyring", {"mu": MU, "g": G, "lambda1": UNTIMED}),
  ("modeyring", {"mu": MU, "g": G, "lambda1": UNTIMED}),
  ("bingham", {"mu": MU, "g": G, "yield_pa": 0.0}),                     # nothing to yield
])
def test_each_thinning_law_becomes_the_kelvin_voigt_it_contains(name, fields, kelvin_voigt):
  """`contains` is what the redundancy prior builds children from, and naming the wrong one
  produces a number rather than an error. This is the claim behind the declaration.
  """
  assert "NHKV" in STANDARD_MODELS[name].contains
  assert float(np.max(np.abs(solve(build(name, **fields)) - kelvin_voigt))) < 1e-9


def test_herschel_bulkley_becomes_the_power_law_it_contains():
  # the one that nests into `powerlaw` rather than `NHKV`: with no yield stress it IS the
  # power law, and its axes are a superset of the power law's for exactly that reason
  assert "powerlaw" in STANDARD_MODELS["herschel"].contains
  reference = solve(build("powerlaw", g=G, pl_k=0.05, pl_n=0.8))
  yielded = solve(build("herschel", g=G, yield_pa=0.0, pl_k=0.05, pl_n=0.8))
  assert float(np.max(np.abs(yielded - reference))) < 1e-9


@pytest.mark.parametrize("name", THINNING)
def test_each_thinning_law_does_something(name):
  """A law that never departs from a constant viscosity is a reparameterisation.

  The comparison is against the SAME law at the SAME point with only its thinning
  parameter flattened, not against a fixed `NHKV`. Written the second way this test
  passes for the wrong reason: the grid centre sits at a different viscosity and modulus
  from any fixed reference, so it measures that changing `mu` changes the answer -- which
  it does whether or not the thinning law is wired up at all. Pinning Bingham's yield
  stress to zero was invisible to it.

  One point rather than the grid, so that the fast lane still holds both halves: the
  reductions above pin that each law CAN go constant, this pins that it need not.
  """
  candidate = STANDARD_MODELS[name]
  centre = {a: float(np.sqrt(PARAMETER_BOUNDS[a][0] * PARAMETER_BOUNDS[a][1])) for a in candidate.axes}
  thinning = solve(candidate.build(centre), rtol=1e-7)
  flat = solve(candidate.build({**centre, **FLATTENED[name]}), rtol=1e-7)
  assert float(np.max(np.abs(thinning - flat))) > 1e-3, f"{name}'s thinning parameter does nothing"


@pytest.mark.slow
@pytest.mark.parametrize("name", THINNING)
def test_each_thinning_law_integrates_across_its_own_bounds(name):
  """Gent is registered over a range where it mostly cannot be integrated at all (#199), so
  a candidate's bounds are a claim about solvability and this is the claim.
  """
  candidate = STANDARD_MODELS[name]
  points, _ = parameter_grid(candidate.axes, 3)
  solved = 0
  for row in points:
    try:
      trace = solve(candidate.build(dict(zip(candidate.axes, row, strict=True))), rtol=1e-7)
    except Exception:                                    # noqa: BLE001
      continue
    solved += int(np.all(np.isfinite(trace)))
  assert solved == len(points), f"{name} solved {solved} of {len(points)} points in its own bounds"


def test_the_thinning_candidates_are_wired_to_the_laws_they_name():
  """Every one takes its viscosity first and its time second; a transposition would still
  build and still solve. Read the laws back rather than trusting the calls.
  """
  carreau = laws("carreau", mu=MU, g=G, lambda1=2e-7, pl_n=0.6)[1]
  assert isinstance(carreau, pyimr.CarreauYasuda)
  assert (carreau.zero_shear_viscosity_pa_s, carreau.time_constant_s) == (MU, 2e-7)
  assert carreau.power_index == 0.6
  # pinned, not free: this is classical Carreau, not Carreau-Yasuda
  assert (carreau.infinite_shear_viscosity_pa_s, carreau.transition_exponent) == (0.0, 2.0)

  bingham = laws("bingham", mu=MU, g=G, yield_pa=12.0)[1]
  assert isinstance(bingham, pyimr.Bingham)
  assert (bingham.yield_stress_pa, bingham.plastic_viscosity_pa_s) == (12.0, MU)

  herschel = laws("herschel", g=G, yield_pa=12.0, pl_k=0.05, pl_n=0.8)[1]
  assert isinstance(herschel, pyimr.HerschelBulkley)
  assert (herschel.yield_stress_pa, herschel.consistency_pa_s_n, herschel.exponent) == (12.0, 0.05, 0.8)

  for name in THINNING:
    elastic = laws(name, **{a: 1.0 if a != "g" else G for a in STANDARD_MODELS[name].axes})[0]
    assert isinstance(elastic, pyimr.NeoHookean)
    assert elastic.shear_modulus_pa == G
