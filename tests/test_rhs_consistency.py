"""The sensitivity path rebuilds some of what `prepare` builds, and nothing else
asserts the two agree.

This file used to be mostly about a larger version of that problem: every physics
law was implemented twice, once in Python for the forward solve and once in numba
for the sensitivities (`_mechanical.py`). Three of four physics changes during the
parity work introduced a bug in the second copy -- Powell-Eyring's non-analytic
`abs()`, the Mie-Gruneisen root, and the Gauss quadrature weights -- none of them
catchable by review, because each file was individually correct. See issue #31.

W11 stage 5 deleted that copy. `_rhs` is the only implementation now, traced by jax
for the sensitivities rather than mirrored, so the two tests that compared them are
gone with it. What survives is the one duplication that remains: `_dual_medium`
rebuilds the wall-flux weights that `prepare` also builds.
"""

import numpy as np
import pytest

import imr_fast
from imr_fast import sensitivity

_NHKV = imr_fast.NeoHookeanKelvinVoigt(2500.0, 0.1)

def _relative_deviation(left, right):
  scale = max(np.abs(left).max(), np.abs(right).max(), 1e-30)
  return float(np.abs(left - right).max() / scale)

# The sixth row of issue #31's duplication table: `_imr_prepare` and
# `imr_sensitivity` build the wall-flux stencils independently.
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
  config = imr_fast.SimulationConfig(R0=225e-6, Req=37.5e-6, material=_NHKV, thermal=backend, **options)
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
