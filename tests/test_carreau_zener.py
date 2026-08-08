"""A relaxation time that moves with the shear rate.

`TwoModeQuadraticZener` answers the correlated one-mode residual with a second timescale.
This answers it with a timescale that changes: a Maxwell arm whose dashpot obeys Carreau
rather than Newton, so `lambda = eta/G` falls as the medium is sheared harder. Since a
collapse spans decades of shear rate, the two are genuinely different claims and the point
is to be able to compare them.

The gates are reductions with known answers. Three things in the wiring fail silently and
each has its own test here: `_stress_state_count` (a missing entry hands the RHS no memory
at all), the positional scales tuple (a missed slot reads a parameter as zero), and
`THROUGH_GROUPS` (a missed entry keeps every answer correct and recompiles per point).
"""

import numpy as np
import pytest

import pyimr
from pyimr._materials import _stress_state_count

TIMES = np.linspace(0.0, 6e-5, 60)
G, MU, LAM1, ALPHA = 204.3, 0.04651, 1.964e-7, 5.301
R_MAX, STRETCH = 277e-6, 7.09


def solve(material, rtol=1e-9):
  config = pyimr.SimulationConfig(R_MAX, R_MAX / STRETCH, material, dynamics="keller-miksis",
                                  rtol=rtol, atol=rtol * 1e-2, max_steps=400_000)
  return np.asarray(pyimr.simulate(TIMES, config).radius_ratio, dtype=float)


def thinning(time_s=1e-6, index=1.0):
  return pyimr.CarreauZener(G, MU, LAM1, 0.0, ALPHA, time_s, index)


def test_it_carries_one_memory_component():
  # one Maxwell arm, unlike the two-mode law: a wrong count here hands `_rhs` the wrong
  # slice and the failure is a TypeError far from the cause
  assert _stress_state_count(thinning(index=0.6)) == 1
  assert _stress_state_count(pyimr.TwoModeQuadraticZener(G, MU, LAM1, 0.0, ALPHA, 1e-6, 0.2)) == 2


@pytest.mark.parametrize("rtol", [1e-7, 1e-9, 1e-10])
def test_unit_power_index_is_the_one_mode_law_exactly(rtol):
  """Not `close to` -- identical, bit for bit, at every tolerance.

  The thinning factor is `(1 + x^2)^((n-1)/2)`, which at `n = 1` is `(1 + x^2)^0`: exactly
  one for every shear rate, so the arithmetic reaching the integrator is the same
  arithmetic. A reduction that merely tracked the tolerance would leave room for an error
  of its own size; this one leaves none, so it is asserted as equality.
  """
  assert np.array_equal(solve(pyimr.QuadraticZener(G, MU, LAM1, 0.0, ALPHA), rtol), solve(thinning(index=1.0), rtol))


def test_a_crossover_out_of_reach_is_also_the_one_mode_law():
  # the other way the thinning switches off: if the crossover sits far above any shear rate
  # the collapse produces, the factor is 1 whatever the index
  assert float(np.max(np.abs(
    solve(pyimr.QuadraticZener(G, MU, LAM1, 0.0, ALPHA)) - solve(thinning(time_s=1e-18, index=0.3))
  ))) < 1e-9


@pytest.mark.parametrize("index", [0.8, 0.5, 0.2])
def test_thinning_moves_the_trajectory(index):
  """Or the parameter is decoration. Departure grows as the index falls."""
  base = solve(pyimr.QuadraticZener(G, MU, LAM1, 0.0, ALPHA))
  assert float(np.max(np.abs(solve(thinning(index=index)) - base))) > 1e-2


def test_a_stronger_thinning_departs_further():
  base = solve(pyimr.QuadraticZener(G, MU, LAM1, 0.0, ALPHA))
  mild = float(np.max(np.abs(solve(thinning(index=0.8)) - base)))
  strong = float(np.max(np.abs(solve(thinning(index=0.3)) - base)))
  assert strong > mild, "a smaller power index must thin harder"


def test_the_new_parameters_are_differentiable():
  """They were given their own `SCALE_PATHS` slots. Without them the sensitivity is
  silently zero rather than an error, so the finite difference is the only witness.
  """
  config = pyimr.SimulationConfig(R_MAX, R_MAX / STRETCH, thinning(2e-6, 0.6), dynamics="keller-miksis",
                                  rtol=1e-9, atol=1e-11, max_steps=400_000)
  problem = pyimr.prepare(config)
  for path, value in (("material.thinning_time_s", 2e-6), ("material.power_index", 0.6)):
    jacobian = np.asarray(problem.solve_with_sensitivities(TIMES, (path,)).radius_ratio, dtype=float)[:, 0]
    assert np.all(np.isfinite(jacobian))
    assert np.max(np.abs(jacobian)) > 0.0, f"{path} has no effect on the solution"

    nudge = 1e-6
    def moved(scale, _p=path):
      fields = {"thinning_time_s": 2e-6, "power_index": 0.6}
      fields[_p.split(".")[1]] *= scale
      return solve(pyimr.CarreauZener(G, MU, LAM1, 0.0, ALPHA, fields["thinning_time_s"], fields["power_index"]))

    finite = (moved(1 + nudge) - moved(1 - nudge)) / (2 * nudge * value)
    assert np.max(np.abs(jacobian - finite)) < 1e-3 * max(np.max(np.abs(finite)), 1e-12)


def test_a_sweep_over_the_new_parameters_compiles_once():
  """Both numbers travel through `p`, so the sweep must share one program.

  Omitting the law from `THROUGH_GROUPS` leaves every answer correct and pays a fresh XLA
  compile per parameter set. Only a count catches that, which is how it survived in the
  two-mode law until it was measured.
  """
  from pyimr import _jax

  times = np.linspace(0.0, 4e-5, 30)

  def radius(time_s, index):
    config = pyimr.SimulationConfig(R_MAX, R_MAX / STRETCH, pyimr.CarreauZener(G, MU, LAM1, 0.0, ALPHA, time_s, index),
                                    dynamics="keller-miksis", rtol=1e-7, atol=1e-9)
    return np.asarray(pyimr.simulate(times, config).radius_ratio, dtype=float)

  _jax._COMPILED.clear()
  traces = [radius(t, n) for t, n in ((5e-7, 0.9), (1e-6, 0.7), (2e-6, 0.5), (4e-6, 0.3))]
  assert len(_jax._COMPILED) == 1, f"the sweep should compile once; got {len(_jax._COMPILED)}"
  for earlier, later in zip(traces[:-1], traces[1:], strict=True):
    assert np.abs(earlier - later).max() > 1e-9, "one program must not mean one answer"


def test_the_whole_memory_equation_thins_and_not_merely_part_of_it():
  """The thinning divides BOTH terms of the arm's ODE, and that is a physical claim.

  A Maxwell arm has one time constant. Writing `Z1d = [-(Z1 - Ze) + drive] / De` and then
  thinning only the first term would put two different relaxation times in one first-order
  equation, which is not a rheology anyone has. So the whole derivative must scale exactly
  as `1 / thinning` against the unthinned law at the same state.

  Every other test here passes either way: at `n = 1` the two forms are identical, and
  "the trajectory moves" is true of both. This one separates them, which is why it exists
  -- a mutation that thinned only the relaxation term survived the rest of the file.
  """
  from pyimr._prepare import params
  from pyimr._stress import _stress

  R, Rd, memory = 0.62, -1.9, np.array([0.37])
  reference, thinned = pyimr.QuadraticZener(G, MU, LAM1, 0.0, ALPHA), thinning(2e-6, 0.45)
  thin_p = params(R_MAX, R_MAX / STRETCH, thinned)

  _, _, plain_rate, _ = _stress(reference, params(R_MAX, R_MAX / STRETCH, reference), R, Rd, memory)
  _, _, thin_rate, _ = _stress(thinned, thin_p, R, Rd, memory)

  scaled = thin_p["Cu"] * 2.0 * np.sqrt(3.0) * abs(Rd / R)
  factor = (1.0 + scaled**2) ** (0.5 * (thin_p["nc"] - 1.0))
  assert factor < 0.9, "the chosen state must actually sit in the thinning regime"
  assert plain_rate is not None and thin_rate is not None
  assert float(thin_rate[0]) == pytest.approx(float(plain_rate[0]) / factor, rel=1e-12)


def test_the_candidate_wires_its_parameters_where_it_claims():
  from pyimr.selection import EXTENDED_MODELS

  theta = {"mu": 0.05, "g": 200.0, "lambda1": 2e-7, "alpha": 5.0, "thin_time": 3e-6, "pl_n": 0.6}
  material = EXTENDED_MODELS["qSLSthin"].build(theta)
  assert isinstance(material, pyimr.CarreauZener)
  assert (material.shear_modulus_pa, material.viscosity_pa_s) == (200.0, 0.05)
  assert (material.relaxation_time_s, material.stiffening) == (2e-7, 5.0)
  # the crossover is absolute here, unlike `qSLS2`'s second time which is a ratio
  assert (material.thinning_time_s, material.power_index) == (3e-6, 0.6)
  assert "qSLS" in EXTENDED_MODELS["qSLSthin"].contains


@pytest.mark.parametrize("index", [0.0, -0.5, np.inf, np.nan])
def test_the_power_index_must_be_a_real_index(index):
  with pytest.raises(ValueError, match="power_index"):
    thinning(index=index)


@pytest.mark.parametrize("time_s", [0.0, -1e-6, np.inf])
def test_the_crossover_must_be_a_real_timescale(time_s):
  with pytest.raises(ValueError, match="thinning_time_s"):
    thinning(time_s=time_s)
