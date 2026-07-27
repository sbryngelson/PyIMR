"""Every physics law is implemented twice: once in Python for the forward solve
(`_rhs`, `_stress`, `_thermal`) and once in numba for the
sensitivities (`_mechanical`). Nothing else asserts that the two agree.

Three of four physics changes during the parity work introduced a bug in the
second copy -- Powell-Eyring's non-analytic `abs()`, the Mie-Gruneisen root, and
the Gauss quadrature weights. None was catchable by review, because each file is
individually correct. See issue #31.

These tests evaluate both implementations at the same state and assert they
agree, and separately assert the compiled path stays analytic so complex-step
differentiation remains valid.
"""

import numpy as np
import pytest

import imr_fast
from imr_fast._rhs import _rhs
from imr_fast import sensitivity
from imr_fast._mechanical import mechanical_rhs

_EMPTY = np.empty(0)

# Radial dynamics: RP, KM pressure, KM enthalpy/Tait, Gilmore/Tait,
# KM enthalpy/Mie-Gruneisen, Gilmore/Mie-Gruneisen.
_RADIAL = (1, 2, 3, 4, 5, 6)

_DISTRIBUTED_POINTS = 32  # far below the 240 default; consistency does not need convergence


def _instantaneous(elastic=None, viscous=None):
  return imr_fast.InstantaneousMaterial(elastic=elastic, viscous=viscous)


# Every material_code (0-8), every elastic_code (1-6), and every viscous_code
# (1-8) in `sensitivity._material_parameters`. A law absent here is a law
# whose two implementations nothing compares.
_MATERIALS = {
  "no_stress": imr_fast.NoStress(),
  "neo_hookean_kelvin_voigt": imr_fast.NeoHookeanKelvinVoigt(2500.0, 0.1),
  "quadratic_kelvin_voigt": imr_fast.QuadraticKelvinVoigt(2500.0, 0.1, 0.25),
  "zener": imr_fast.Zener(2500.0, 0.1, 40e-6, 8e-6),
  "quadratic_zener": imr_fast.QuadraticZener(2500.0, 0.1, 40e-6, 8e-6, 0.25),
  "oldroyd_b": imr_fast.OldroydB(0.1, 40e-6, 8e-6),
  "giesekus": imr_fast.Giesekus(
    viscosity_pa_s=0.1, relaxation_time_s=40e-6, retardation_time_s=8e-6, mobility=0.2, points=_DISTRIBUTED_POINTS
  ),
  "linear_ptt": imr_fast.LinearPTT(
    viscosity_pa_s=0.1, relaxation_time_s=40e-6, retardation_time_s=8e-6, extensibility=0.05, points=_DISTRIBUTED_POINTS
  ),
  "elastic_neo_hookean": _instantaneous(elastic=imr_fast.NeoHookean(2500.0)),
  "elastic_mooney_rivlin": _instantaneous(elastic=imr_fast.MooneyRivlin(2000.0, 500.0)),
  "elastic_yeoh": _instantaneous(elastic=imr_fast.Yeoh(2500.0, 120.0, 15.0)),
  "elastic_fung": _instantaneous(elastic=imr_fast.Fung(2500.0, 0.2)),
  "elastic_gent": _instantaneous(elastic=imr_fast.Gent(2500.0, 250.0)),
  "elastic_arruda_boyce": _instantaneous(elastic=imr_fast.ArrudaBoyce(2500.0, 30.0)),
  # Single-term alpha=2 is the neo-Hookean limit; the three-term case has a
  # negative and a fractional exponent, which is where Ogden differs in kind
  # from every other law here.
  "elastic_ogden_one_term": _instantaneous(elastic=imr_fast.Ogden((2500.0,), (2.0,))),
  "elastic_ogden_multi": _instantaneous(elastic=imr_fast.Ogden((1800.0, 600.0, -300.0), (1.3, 4.0, -2.0))),
  "viscous_newtonian": _instantaneous(viscous=imr_fast.Newtonian(0.1)),
  "viscous_power_law": _instantaneous(viscous=imr_fast.PowerLaw(0.1, 0.7)),
  "viscous_carreau_yasuda": _instantaneous(viscous=imr_fast.CarreauYasuda(0.5, 0.02, 20e-6, 2.0, 0.45)),
  "viscous_cross": _instantaneous(viscous=imr_fast.Cross(0.5, 0.02, 20e-6, 1.0)),
  "viscous_powell_eyring": _instantaneous(viscous=imr_fast.PowellEyring(0.5, 0.02, 20e-6)),
  "viscous_modified_powell_eyring": _instantaneous(viscous=imr_fast.ModifiedPowellEyring(0.5, 0.02, 20e-6)),
  "viscous_herschel_bulkley": _instantaneous(viscous=imr_fast.HerschelBulkley(50.0, 0.1, 0.7)),
  "viscous_bingham": _instantaneous(viscous=imr_fast.Bingham(50.0, 0.1)),
  "elastic_and_viscous": _instantaneous(
    elastic=imr_fast.Gent(2500.0, 250.0), viscous=imr_fast.CarreauYasuda(0.5, 0.02, 20e-6, 2.0, 0.45)
  ),
}


def _config(material, radial):
  return imr_fast.SimulationConfig(R0=225e-6, Req=37.5e-6, material=material, radial=radial)


def _compiled_arguments(problem, config):
  """The argument pack `imr_sensitivity` builds for the numba path."""
  values, _ = sensitivity._mechanical_parameters(problem.parameters, 0)
  material_code, elastic_code, elastic_values, _, viscous_code, viscous_values, _ = sensitivity._material_parameters(
    config.material, 0
  )
  instantaneous, distributed = problem.instantaneous_material, problem.distributed_stress
  return (
    values,
    material_code,
    elastic_code,
    elastic_values,
    viscous_code,
    viscous_values,
    instantaneous.interval_nodes if instantaneous is not None else _EMPTY,
    instantaneous.interval_weights if instantaneous is not None else _EMPTY,
    distributed.reference_radius if distributed is not None else _EMPTY,
    distributed.reference_radius_cubed if distributed is not None else _EMPTY,
    distributed.weights if distributed is not None and distributed.weights is not None else _EMPTY,
    config.radial,
  )


def _python_rhs(problem, config, time, state):
  return np.asarray(
    _rhs(
      time,
      np.asarray(state, dtype=float),
      problem.parameters,
      config.material,
      config.radial,
      forcing=problem.forcing,
      instantaneous_material=problem.instantaneous_material,
      distributed_stress=problem.distributed_stress,
    ),
    dtype=float,
  )


def _compiled_rhs(arguments, time, state):
  return mechanical_rhs(complex(time), np.asarray(state, dtype=np.complex128), *arguments)


# Compression, wall velocity: a bounded sweep around the prepared initial state.
# Two implementations of the same expression must agree wherever both are
# defined, so these need not lie on a trajectory -- and must not, because a
# fixed-step march on a stiff collapse drifts into exp overflow for the
# stiffer laws and makes the test brittle rather than sensitive. Radius stays
# above 0.55 to keep every law inside its material domain.
_RADIUS_FACTORS = (1.0, 0.85, 0.7, 0.55)
_WALL_VELOCITIES = (0.0, -0.8, 0.4)
_STRESS_LEVEL = 0.05  # nonzero, so memory-law rate terms are exercised


def _sample_states(problem):
  """A deterministic bounded family of states around the prepared initial state."""
  initial = np.asarray(problem.initial_state, dtype=float)
  # Alternating sign so radial and hoop components of the distributed models
  # differ; a uniform fill would leave `tau_rr - tau_tt` identically zero.
  memory = _STRESS_LEVEL * (-1.0) ** np.arange(initial.size - 2)
  states = []
  for factor in _RADIUS_FACTORS:
    for velocity in _WALL_VELOCITIES:
      state = initial.copy()
      state[0] = factor
      state[1] = velocity
      state[2:] = initial[2:] + memory
      states.append(state)
  return states


def _relative_deviation(left, right):
  scale = max(np.abs(left).max(), np.abs(right).max(), 1e-30)
  return float(np.abs(left - right).max() / scale)


@pytest.mark.parametrize("radial", _RADIAL)
@pytest.mark.parametrize("material_name", sorted(_MATERIALS))
def test_python_and_compiled_rhs_agree(material_name, radial):
  """The forward solve and the sensitivities must compute the same physics.

  Both paths evaluate the same closed-form expressions in the same order, so
  the bar is machine precision, not a physical tolerance. A law that drifts
  here has been edited in one file and not the other.
  """
  config = _config(_MATERIALS[material_name], radial)
  problem = imr_fast.prepare(config)
  arguments = _compiled_arguments(problem, config)
  for state in _sample_states(problem):
    expected = _python_rhs(problem, config, 0.0, state)
    actual = np.real(_compiled_rhs(arguments, 0.0, state))
    deviation = _relative_deviation(expected, actual)
    assert deviation < 1e-12, (
      f"{material_name} radial={radial}: _rhs._rhs and mechanical_rhs disagree by {deviation:.3e} "
      f"at state {np.array2string(state[:4], precision=6)}..."
    )


@pytest.mark.parametrize("material_name", sorted(_MATERIALS))
def test_compiled_rhs_stays_analytic(material_name):
  """`mechanical_tangent_rhs` differentiates by complex step (`+1j*1e-30`), so
  every operation on the compiled value path must be analytic.

  `abs()` discards the imaginary perturbation, and the symptom is a sensitivity
  error independent of the finite-difference step -- exactly how the
  Powell-Eyring defect presented. Compare a complex step against a real central
  difference: a non-analytic operation breaks the first while leaving the
  second intact.
  """
  config = _config(_MATERIALS[material_name], radial=2)
  problem = imr_fast.prepare(config)
  arguments = _compiled_arguments(problem, config)
  state = _sample_states(problem)[-1]

  complex_step, real_step = 1e-30, 1e-6
  for index in (0, 1):  # radius and wall velocity: both feed every stress law
    perturbation = np.zeros(state.size, dtype=np.complex128)
    perturbation[index] = 1.0
    derivative = np.imag(_compiled_rhs(arguments, 0.0, state + 1j * complex_step * perturbation)) / complex_step
    ahead = np.real(_compiled_rhs(arguments, 0.0, state + real_step * np.real(perturbation)))
    behind = np.real(_compiled_rhs(arguments, 0.0, state - real_step * np.real(perturbation)))
    reference = (ahead - behind) / (2.0 * real_step)
    deviation = _relative_deviation(derivative, reference)
    assert deviation < 1e-6, (
      f"{material_name}: complex-step derivative with respect to state[{index}] disagrees with a central "
      f"difference by {deviation:.3e}. A non-analytic operation (`abs`, `max`, a comparison on a complex "
      f"value) has entered the compiled value path."
    )


# The bubtherm=0 gap. `mechanical_rhs` uses the polytropic closure, so nothing
# above reaches the thermal path -- and that is where the sixth row of issue
# #31's duplication table lives: `_imr_prepare` and `imr_sensitivity` build the
# wall-flux stencils independently.
#
# They diverged. `_dual_medium` hardcoded the three-point uniform
# finite-difference stencil and overwrote the prepared operators with it, so on
# `thermal="spectral"` -- whose prepared weights are dense Chebyshev rows --
# every tangent differentiated a different operator than the forward solve
# integrated. The measured cost was a step-independent 2.5e-01 relative error
# on the coupled medium-temperature tangent.
_THERMAL_CONFIGURATIONS = [
  ("bubble_only", dict(bubtherm=1, medtherm=1, Nt=7, Mt=7)),
  ("mass_transfer", dict(bubtherm=1, medtherm=1, masstrans=1, vapor=1, Nt=7, Mt=7)),
  ("asymmetric_grids", dict(bubtherm=1, medtherm=1, masstrans=1, vapor=1, Nt=11, Mt=6)),
]


@pytest.mark.parametrize("backend", ("fd", "spectral"))
@pytest.mark.parametrize("label,options", _THERMAL_CONFIGURATIONS, ids=[c[0] for c in _THERMAL_CONFIGURATIONS])
def test_prepared_and_dual_wall_stencils_agree(label, options, backend):
  """The forward solve's wall-flux weights and the sensitivity path's rebuild of
  them must be the same numbers.

  The sensitivity path has to rebuild these because they carry `Dual` parameter
  dependence, but their stencils are pure grid geometry and must come from the
  prepared problem rather than be assumed. Costs no ODE solve, so it runs in
  the fast suite.
  """
  config = imr_fast.SimulationConfig(
    R0=225e-6, Req=37.5e-6, material=_MATERIALS["neo_hookean_kelvin_voigt"], thermal=backend, **options
  )
  problem = imr_fast.prepare(config)
  normalized, values, scales = sensitivity._normalize_parameters(config, ["material.shear_modulus_pa"])
  dual_config = sensitivity._dual_config(config, normalized, values, scales)
  dual = sensitivity._dual_medium(problem, sensitivity._dual_parameters(dual_config))

  for name in ("grad_Tm", "grad_Trans", "grad_C"):
    prepared = np.asarray(getattr(problem.medium, name), dtype=float)
    rebuilt = np.array([float(getattr(entry, "value", entry)) for entry in getattr(dual, name)])
    assert prepared.shape == rebuilt.shape, (
      f"{backend}/{label}: {name} has {prepared.shape} in the forward solve and {rebuilt.shape} in the sensitivity path"
    )
    deviation = _relative_deviation(prepared, rebuilt)
    assert deviation < 1e-13, (
      f"{backend}/{label}: {name} differs by {deviation:.3e} between the forward solve and the sensitivity "
      f"path. The sensitivity path is differentiating a different boundary closure than the solve uses.\n"
      f"  prepared: {np.array2string(prepared, precision=4, max_line_width=200)}\n"
      f"  rebuilt : {np.array2string(rebuilt, precision=4, max_line_width=200)}"
    )
