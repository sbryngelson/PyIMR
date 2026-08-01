"""The sensitivity path rebuilds some of what `prepare` builds, and nothing else"""

import numpy as np
import pytest

import pyimr
from pyimr._prepare import medium_with_parameters

_NHKV = pyimr.NeoHookeanKelvinVoigt(2500.0, 0.1)

def _relative_deviation(left, right):
  scale = max(np.abs(left).max(), np.abs(right).max(), 1e-30)
  return float(np.abs(left - right).max() / scale)

_THERMAL_CONFIGURATIONS = [
  ("bubble_only", dict(bubtherm=1, medtherm=1, Nt=7, Mt=7)),
  ("mass_transfer", dict(bubtherm=1, medtherm=1, masstrans=1, vapor=1, Nt=7, Mt=7)),
  ("asymmetric_grids", dict(bubtherm=1, medtherm=1, masstrans=1, vapor=1, Nt=11, Mt=6)),
]


@pytest.mark.parametrize("backend", ("fd", "spectral"))
@pytest.mark.parametrize("label,options", _THERMAL_CONFIGURATIONS, ids=[c[0] for c in _THERMAL_CONFIGURATIONS])
def test_prepared_and_dual_wall_stencils_agree(label, options, backend):
  """The forward solve's wall-flux weights and the sensitivity path's rebuild of"""
  config = pyimr.SimulationConfig(R0=225e-6, Req=37.5e-6, material=_NHKV, thermal=backend, **options)
  problem = pyimr.prepare(config)
  rebuilt_medium = medium_with_parameters(problem.medium, problem.parameters)
  assert rebuilt_medium is not None

  for name in ("grad_Tm", "grad_Trans", "grad_C"):
    prepared = np.asarray(getattr(problem.medium, name), dtype=float)
    rebuilt = np.asarray(getattr(rebuilt_medium, name), dtype=float)
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
