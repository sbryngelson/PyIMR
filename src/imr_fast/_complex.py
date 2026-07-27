"""Complex-step forward sensitivities for the thermal configurations (#44).

`Dual` is a *scalar* carrying a tangent array, so every elementary operation is a
Python-level dispatch and an array of them is `dtype=object`. On the thermal path
that costs ~15,800 Python calls per RHS evaluation and makes a sensitivity solve
37-60x a forward solve, against 1.87x on the compiled mechanical path.

Complex128 carries the same first-order algebra -- `(a + ib)(c + id)` has
imaginary part `ad + bc`, which is the product rule -- but NumPy vectorises it
natively, so `D1 @ theta` is a BLAS matvec rather than a `frompyfunc` loop. The
value path is already required to be analytic (`_mechanical.py`), which is
exactly the condition complex-step needs, and `test_compiled_rhs_stays_analytic`
enforces it.

The trade is directions per solve: `Dual` carries all of them in one pass, so its
per-RHS cost is flat in width, while this evaluates the RHS once per direction.
Measured per-RHS at Nt=Mt=7, the crossover is ~45 directions (bubtherm) and ~28
(coupled); below that this wins, by 46x and 28x respectively at one direction.

Not used for distributed materials: `_distributed_dissipation` calls `np.cbrt`
and `np.interp`, neither of which accepts complex, and its `np.maximum` clamp is
not analytic anyway. Those keep the `Dual` path.
"""

from __future__ import annotations

import copy
import dataclasses

import numpy as np

import imr_fast as _solver
from ._autodiff import Dual

__all__ = ["STEP", "complex_step_supported", "rhs_complex"]

# Small enough that the O(h^2) term underflows, so the imaginary part is the
# exact derivative rather than a difference quotient -- there is no subtractive
# cancellation to trade against, which is the whole point of complex step.
STEP = 1e-30


def complex_step_supported(problem):
  """Distributed stress reaches np.cbrt/np.interp, which reject complex input."""
  return problem.distributed_stress is None


def _to_complex(value, index):
  """Map one direction of a `Dual` structure onto complex128.

  `Dual(v, t)` becomes `v + i*STEP*t[index]`, so the tangent rides in the
  imaginary part at the scale complex-step expects. The recursion covers
  everything the dual builders produce -- nested dataclasses (configs, materials,
  medium operators), dicts, object arrays and bare scalars -- so no per-container
  field list has to be maintained alongside them. That list is exactly what went
  stale in #43, where `_dual_medium` kept finite-difference wall stencils after
  the grid learned about Chebyshev.
  """
  if isinstance(value, Dual):
    return value.value + 1j * STEP * value.tangent[index]
  if isinstance(value, dict):
    return {key: _to_complex(item, index) for key, item in value.items()}
  if isinstance(value, np.ndarray):
    if value.dtype != object:
      return value.astype(complex)
    flat = [_to_complex(item, index) for item in value.ravel()]
    return np.array(flat, dtype=complex).reshape(value.shape)
  if isinstance(value, tuple):
    return tuple(_to_complex(item, index) for item in value)
  if dataclasses.is_dataclass(value) and not isinstance(value, type):
    duplicate = copy.copy(value)
    for field in dataclasses.fields(value):
      object.__setattr__(duplicate, field.name, _to_complex(getattr(value, field.name), index))
    return duplicate
  return value


def directions(config, parameters, medium, forcing, width):
  """One frozen complex copy of the whole parameter set per direction."""
  return [tuple(_to_complex(item, index) for item in (config, parameters, medium, forcing)) for index in range(width)]


def rhs_complex(time_s, packed, *, problem, prepared, wall_states, width):
  """Augmented RHS with the tangents carried in the imaginary part.

  Same packed layout as the `Dual` route -- (state, 1 + width) flattened -- so
  this is a drop-in for `_rhs_physical`. Each direction is one complex
  evaluation of the *real* RHS: the real part returns f(x), the imaginary part
  divided by STEP returns `J @ s_k + df/dtheta_k`, which is the augmented row.
  """
  state_width = problem.layout.size
  matrix = packed.reshape(state_width, width + 1)
  result = np.empty_like(matrix)
  for index in range(width):
    config, parameters, medium, forcing = prepared[index]
    state = matrix[:, 0] + 1j * STEP * matrix[:, index + 1]
    output = (
      np.asarray(
        _solver._rhs(
          time_s / parameters["t0"],
          state,
          parameters,
          config.material,
          config.radial,
          config.bubtherm,
          problem.bubble_D1,
          problem.bubble_D2,
          problem.bubble_grid,
          config.medtherm,
          medium,
          config.masstrans,
          wall_states[index],
          forcing,
          problem.instantaneous_material,
          problem.distributed_stress,
        ),
        dtype=complex,
      )
      / parameters["t0"]
    )
    if index == 0:
      result[:, 0] = output.real
    result[:, index + 1] = output.imag / STEP
  return result.ravel()
