"""The jax backend against the scipy one (PLAN.md W11 stage 2b).

jax and diffrax are optional, so every test here skips without them, exactly as
`test_pymc_op.py` does for PyMC. The core install stays numpy/scipy/numba.

What makes this cheap to assert is that there is no second implementation to
compare against. Stage 2a made `_rhs` namespace-agnostic, so both backends
integrate the SAME right-hand side and any disagreement is the integrator's
alone -- diffrax's Tsit5 against scipy's LSODA.
"""

import importlib.util

import numpy as np
import pytest

import imr_fast
import imr_fast.sensitivity
from _validation_support import NHKV, R0, REQ, oldroyd_b, zener

# Scoped to the cross-backend test rather than the module: the refusals below
# need no jax at all -- `_jax.unsupported_reason` imports only numpy -- and CI
# has no jax, so a module-level skip would drop the coverage that runs there.
_HAS_JAX = all(importlib.util.find_spec(name) is not None for name in ("jax", "diffrax"))
requires_jax = pytest.mark.skipif(not _HAS_JAX, reason="jax and diffrax are optional; not installed")

SECTION = "7. jax backend"
_TIMES = np.linspace(0.0, 40e-6, 300)

# scipy itself sits at max 8.6e-06 / median 3.1e-08 against the pinned IMRv2
# trajectories, so requiring the two backends to agree an order of magnitude
# INSIDE that is requiring them to agree better than either agrees with the
# reference. Measured worst across these cases: 2.9e-06 max, 2.1e-07 median.
_MAX_BOUND, _MEDIAN_BOUND = 1e-05, 1e-06

_CASES = [(radial, "NHKV", NHKV) for radial in range(1, 7)] + [
  (2, "NoStress", imr_fast.NoStress()),
  (2, "qKV", imr_fast.QuadraticKelvinVoigt(2500.0, 0.1, 0.25)),
  (2, "Zener", zener()),
  (2, "OldroydB", oldroyd_b()),
  (4, "Zener", zener()),
]

@requires_jax
@pytest.mark.parametrize("radial,label,material", _CASES, ids=[f"radial{r}-{n}" for r, n, _ in _CASES])
def test_jax_backend_matches_scipy(radial, label, material, measured):
  reference = np.asarray(
    imr_fast.simulate(_TIMES, imr_fast.SimulationConfig(R0=R0, Req=REQ, material=material, radial=radial)).radius_ratio
  )
  result = imr_fast.simulate(_TIMES, imr_fast.SimulationConfig(R0=R0, Req=REQ, material=material, radial=radial, backend="jax"))
  computed = np.asarray(result.radius_ratio)
  worst, typical = float(np.nanmax(np.abs(reference - computed))), float(np.nanmedian(np.abs(reference - computed)))
  measured(f"jax vs scipy r={radial} {label}", f"max={worst:.2e} median={typical:.2e}")
  assert result.stats.backend == "jax-tsit5", "the jax backend was not actually selected"
  assert worst < _MAX_BOUND and typical < _MEDIAN_BOUND

def test_unsupported_configurations_are_refused_by_name():
  """A refusal at construction, not a tracer error several frames into diffrax.

  Stage 4 brought the thermal fields in. What is left out is the distributed
  stress, which packs its output into a preallocated buffer, and mass transfer,
  whose wall closure is an iterative solve -- covered separately below.
  """
  distributed = imr_fast.Giesekus(0.1, 80e-6, 16e-6, 0.2, points=12)
  with pytest.raises(ValueError, match="distributed-memory"):
    imr_fast.SimulationConfig(R0=R0, Req=REQ, material=distributed, backend="jax")

def test_backend_field_rejects_anything_else():
  with pytest.raises(ValueError, match="backend must be"):
    imr_fast.SimulationConfig(R0=R0, Req=REQ, material=NHKV, backend="torch")


_THERMAL_CASES = [
  ("bubtherm fd", dict(bubtherm=1, Nt=17, thermal="fd")),
  ("bubtherm spectral", dict(bubtherm=1, Nt=17, thermal="spectral")),
  ("coupled fd", dict(bubtherm=1, medtherm=1, Nt=13, Mt=13, thermal="fd")),
  ("coupled spectral", dict(bubtherm=1, medtherm=1, Nt=13, Mt=13, thermal="spectral")),
]

@requires_jax
@pytest.mark.parametrize("label,options", _THERMAL_CASES, ids=[c[0] for c in _THERMAL_CASES])
def test_jax_backend_matches_scipy_on_the_thermal_path(label, options, measured):
  """The thermal fields, which slice assignment kept off this backend until W11
  stage 4 routed them through `at_set`.

  `medtherm` needs no iterative wall solve -- #57 made that closure closed form
  -- which is why it lands here while `masstrans`, whose closure is still a
  secant with a fallback ladder, does not.
  """
  times = np.linspace(0.0, 25e-6, 200)
  reference = np.asarray(imr_fast.simulate(times, imr_fast.SimulationConfig(R0=R0, Req=REQ, material=NHKV, **options)).radius_ratio)
  result = imr_fast.simulate(times, imr_fast.SimulationConfig(R0=R0, Req=REQ, material=NHKV, backend="jax", **options))
  computed = np.asarray(result.radius_ratio)
  worst, typical = float(np.nanmax(np.abs(reference - computed))), float(np.nanmedian(np.abs(reference - computed)))
  measured(f"jax vs scipy {label}", f"max={worst:.2e} median={typical:.2e} steps={result.stats.nfev}")
  assert worst < _MAX_BOUND and typical < _MEDIAN_BOUND

def test_mass_transfer_is_still_refused():
  """The one thermal branch stage 4 does not reach: `_wall_theta_bw_full` is a
  secant with a fallback ladder, and a data-dependent loop needs
  `lax.custom_root` rather than a namespace swap."""
  with pytest.raises(ValueError, match="masstrans"):
    imr_fast.SimulationConfig(
      R0=R0, Req=REQ, material=NHKV, backend="jax", bubtherm=1, vapor=1, masstrans=1, medtherm=1, Nt=9, Mt=9
    )

_TANGENT_FIELDS = ("radius_ratio", "radius_m", "wall_velocity_m_s", "internal_pressure_pa", "stress_integral_pa")

@requires_jax
@pytest.mark.parametrize(
  "label,material,paths",
  [
    ("NHKV G", NHKV, ("material.shear_modulus_pa",)),
    ("NHKV G+mu", NHKV, ("material.shear_modulus_pa", "material.viscosity_pa_s")),
    ("Zener G", zener(), ("material.shear_modulus_pa",)),
  ],
  ids=["nhkv-G", "nhkv-G-mu", "zener-G"],
)
def test_jax_tangents_match_the_scipy_sensitivity_path(label, material, paths, measured):
  """Every output's tangent, from one `jacfwd`.

  What is being checked is not just the radius. `internal_pressure_pa` and
  `stress_integral_pa` are nonlinear in state and parameters, and on the scipy
  route their tangents come from `_output_duals` deriving each one. Here the
  traced function returns the primal outputs and `jacfwd` differentiates all of
  them together, so they cost no extra code -- and no extra chance to be wrong.
  """
  from imr_fast._jax import sensitivities_jax

  times = np.linspace(0.0, 20e-6, 80)
  config = imr_fast.SimulationConfig(R0=R0, Req=REQ, material=material, radial=2)
  problem = imr_fast.prepare(config)
  reference = imr_fast.sensitivity.solve_with_sensitivities(problem, times, paths)
  _, tangent = sensitivities_jax(problem, times, paths)

  worst = {}
  for index, field in enumerate(_TANGENT_FIELDS):
    expected = np.asarray(getattr(reference, field))
    scale = max(float(np.max(np.abs(expected))), 1e-30)
    worst[field] = float(np.max(np.abs(expected - tangent[:, index, :]))) / scale
  measured(f"jax tangents {label}", "  ".join(f"{f.split('_')[0]}={w:.1e}" for f, w in worst.items()))
  assert max(worst.values()) < 1e-05, worst

@requires_jax
def test_jax_tangents_converge_to_the_scipy_ones(measured):
  """The gate is convergence under refinement, not a fixed threshold.

  JAX cannot join the 5e-13 agreement complex-step and Dual reach, and PLAN.md
  W11 originally asked it to. Those two differentiate the SAME integration; JAX
  differentiates a different integrator and carries that integrator's error. So
  the question is whether the residual is error or a wrong derivative, and only
  refinement distinguishes them -- a fixed bound can be met by a derivative that
  is wrong but close.
  """
  from imr_fast._jax import sensitivities_jax

  times = np.linspace(0.0, 20e-6, 80)
  paths = ("material.shear_modulus_pa",)
  errors = []
  for tolerance in (1e-8, 1e-10, 1e-12):
    config = imr_fast.SimulationConfig(R0=R0, Req=REQ, material=NHKV, radial=2, rtol=tolerance, atol=tolerance * 1e-2)
    problem = imr_fast.prepare(config)
    expected = np.asarray(imr_fast.sensitivity.solve_with_sensitivities(problem, times, paths).radius_m)
    _, tangent = sensitivities_jax(problem, times, paths)
    scale = max(float(np.max(np.abs(expected))), 1e-30)
    errors.append(float(np.max(np.abs(expected - tangent[:, 1, :]))) / scale)
  measured("jax tangent convergence", " -> ".join(f"{e:.1e}" for e in errors))
  assert errors[-1] < errors[0] / 100.0, f"tangent error did not converge under refinement: {errors}"
