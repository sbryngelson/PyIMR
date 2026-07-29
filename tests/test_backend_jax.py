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

  Stage 2b is the mechanical path only. The thermal fields and the distributed
  stress both assemble their output by in-place assignment into a preallocated
  buffer, which needs `jnp.at[].set()` rather than a namespace swap -- stage 4.
  """
  with pytest.raises(ValueError, match="bubtherm"):
    imr_fast.SimulationConfig(R0=R0, Req=REQ, material=NHKV, backend="jax", bubtherm=1, Nt=9)
  distributed = imr_fast.Giesekus(0.1, 80e-6, 16e-6, 0.2, points=12)
  with pytest.raises(ValueError, match="distributed-memory"):
    imr_fast.SimulationConfig(R0=R0, Req=REQ, material=distributed, backend="jax")

def test_backend_field_rejects_anything_else():
  with pytest.raises(ValueError, match="backend must be"):
    imr_fast.SimulationConfig(R0=R0, Req=REQ, material=NHKV, backend="torch")
