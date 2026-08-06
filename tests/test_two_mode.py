"""A second relaxation time: the two-mode quadratic Zener.

Every other viscoelastic law in the package carries one relaxation time. The one-mode fit
leaves a residual correlated at lag one and concentrated in the first collapse, so a second
timescale is the first enrichment worth testing. A constitutive law that is subtly wrong
would corrupt every ranking downstream and look like physics while doing it, so the tests
here are reductions with known answers rather than regression values.
"""

import numpy as np
import pytest

import pyimr
from pyimr._materials import _stress_state_count

TIMES = np.linspace(0.0, 6e-5, 60)
G, MU, LAM1, ALPHA = 204.3, 0.04651, 1.964e-7, 5.301


def solve(material, rtol=1e-9):
  config = pyimr.SimulationConfig(277e-6, 277e-6 / 7.09, material, radial=2,
                                  rtol=rtol, atol=rtol * 1e-2, max_steps=400_000)
  return np.asarray(pyimr.simulate(TIMES, config).radius_ratio, dtype=float)


def test_it_carries_two_memory_components():
  assert _stress_state_count(pyimr.TwoModeQuadraticZener(G, MU, LAM1, 0.0, ALPHA, 1e-6, 0.2)) == 2
  assert _stress_state_count(pyimr.QuadraticZener(G, MU, LAM1, 0.0, ALPHA)) == 1


@pytest.mark.parametrize("rtol", [1e-7, 1e-9])
def test_zero_share_reduces_to_the_one_mode_law(rtol):
  """The reduction must be exact, not close.

  Both the elastic target and the viscous forcing are split by the share, so at zero the
  second memory is driven by nothing and decays from zero. The gap must therefore track the
  solver tolerance rather than resting on a floor -- a floor would be a real discrepancy
  wearing the disguise of round-off. Measured across four decades: 2.3e-7, 1.9e-9, 1.7e-11.
  """
  one = solve(pyimr.QuadraticZener(G, MU, LAM1, 0.0, ALPHA), rtol)
  two = solve(pyimr.TwoModeQuadraticZener(G, MU, LAM1, 0.0, ALPHA, 1e-6, 0.0), rtol)
  assert float(np.max(np.abs(one - two))) < 300.0 * rtol


def test_the_gap_shrinks_with_the_tolerance():
  loose = solve(pyimr.QuadraticZener(G, MU, LAM1, 0.0, ALPHA), 1e-7) - \
          solve(pyimr.TwoModeQuadraticZener(G, MU, LAM1, 0.0, ALPHA, 1e-6, 0.0), 1e-7)
  tight = solve(pyimr.QuadraticZener(G, MU, LAM1, 0.0, ALPHA), 1e-10) - \
          solve(pyimr.TwoModeQuadraticZener(G, MU, LAM1, 0.0, ALPHA, 1e-6, 0.0), 1e-10)
  assert np.max(np.abs(tight)) < 0.05 * np.max(np.abs(loose)), \
    "a discrepancy that survives tightening is not round-off"


def test_no_stiffening_and_no_share_reduces_to_the_plain_zener():
  # stiffening 0 sends alphax to 0, which collapses the quadratic elastic target onto the
  # neo-Hookean one. So the no-stiffening control model comes free and needs no class.
  plain = solve(pyimr.Zener(G, MU, LAM1, 0.0))
  two = solve(pyimr.TwoModeQuadraticZener(G, MU, LAM1, 0.0, 0.0, 1e-6, 0.0))
  assert float(np.max(np.abs(plain - two))) < 1e-6


@pytest.mark.parametrize("share", [0.1, 0.3, 0.5])
def test_a_second_mode_does_real_work(share):
  """The second arm must change the trajectory, or it is a reparameterisation."""
  one = solve(pyimr.QuadraticZener(G, MU, LAM1, 0.0, ALPHA))
  two = solve(pyimr.TwoModeQuadraticZener(G, MU, LAM1, 0.0, ALPHA, 2e-6, share))
  assert float(np.max(np.abs(one - two))) > 1e-3, "a share this large must be visible"


def test_a_longer_second_time_departs_further():
  one = solve(pyimr.QuadraticZener(G, MU, LAM1, 0.0, ALPHA))
  near = solve(pyimr.TwoModeQuadraticZener(G, MU, LAM1, 0.0, ALPHA, 2.2e-7, 0.3))
  far = solve(pyimr.TwoModeQuadraticZener(G, MU, LAM1, 0.0, ALPHA, 5e-6, 0.3))
  assert np.max(np.abs(one - far)) > np.max(np.abs(one - near))


def test_the_new_parameters_are_differentiable():
  # they were given their own SCALE_PATHS slots; without those the sensitivity is silently
  # zero rather than an error, which is the failure this catches
  material = pyimr.TwoModeQuadraticZener(G, MU, LAM1, 0.0, ALPHA, 2e-6, 0.25)
  config = pyimr.SimulationConfig(277e-6, 277e-6 / 7.09, material, radial=2,
                                  rtol=1e-9, atol=1e-11, max_steps=400_000)
  problem = pyimr.prepare(config)
  for path, value in (("material.second_relaxation_time_s", 2e-6), ("material.second_share", 0.25)):
    jacobian = np.asarray(problem.solve_with_sensitivities(TIMES, (path,)).radius_ratio, dtype=float)[:, 0]
    assert np.all(np.isfinite(jacobian))
    assert np.max(np.abs(jacobian)) > 0.0, f"{path} has no effect on the solution"
    nudge = 1e-6
    def moved(scale, _p=path):
      fields = {"second_relaxation_time_s": 2e-6, "second_share": 0.25}
      fields[_p.split(".")[1]] *= scale
      return solve(pyimr.TwoModeQuadraticZener(G, MU, LAM1, 0.0, ALPHA, fields["second_relaxation_time_s"],
                                               fields["second_share"]))
    finite = (moved(1 + nudge) - moved(1 - nudge)) / (2 * nudge * value)
    assert np.max(np.abs(jacobian - finite)) < 1e-3 * max(np.max(np.abs(finite)), 1e-12)


@pytest.mark.parametrize("share", [-0.1, 1.0, 1.5, np.nan])
def test_the_share_is_a_fraction(share):
  with pytest.raises(ValueError, match="second_share"):
    pyimr.TwoModeQuadraticZener(G, MU, LAM1, 0.0, ALPHA, 1e-6, share)


@pytest.mark.parametrize("time", [0.0, -1e-6, np.inf])
def test_the_second_time_must_be_a_real_timescale(time):
  with pytest.raises(ValueError, match="second_relaxation_time_s"):
    pyimr.TwoModeQuadraticZener(G, MU, LAM1, 0.0, ALPHA, time, 0.2)
