"""Shared constants and helpers for the numerical validation suite.

Not collected by pytest (no `test_` prefix). The numerical content of these
checks is unchanged from the `run_validation.py` script this replaced; only the
harness around them moved. See issue #32.
"""

import functools
from dataclasses import replace
from pathlib import Path

import numpy as np

import imr_fast

DATA = Path(__file__).resolve().parent

R0 = 225e-6
REQ = R0 / 6
T0 = R0 / np.sqrt(101325 / 1064)

NHKV = imr_fast.NeoHookeanKelvinVoigt(2500.0, 0.1)


def zener():
  return imr_fast.Zener(2500.0, 0.1, 2 * T0, 0.4 * T0)


def oldroyd_b():
  return imr_fast.OldroydB(0.1, 2 * T0, 0.4 * T0)


@functools.lru_cache(maxsize=None)
def reference(name):
  """A pinned IMRv2 trajectory. Cached -- imr2_s06.csv alone is 234 kB and
  several modules read the same files."""
  return np.loadtxt(DATA / name, delimiter="," if name == "imr2_s06.csv" else None)


@functools.lru_cache(maxsize=None)
def reference_times():
  return reference("ref_t.csv")


def solve_radius(times, material=NHKV, R0_m=R0, Req_m=REQ, **options):
  config = imr_fast.SimulationConfig(R0=R0_m, Req=Req_m, material=material, **options)
  return imr_fast.simulate(times, config).radius_ratio


def deviation(left, right):
  return float(np.nanmax(np.abs(left - right)))


def median_deviation(left, right):
  """Median pointwise deviation -- the phase-insensitive companion to the max.

  The pointwise maximum on the 300-point reference grid is dominated by a
  sub-nanosecond timing difference at the collapse, amplified by
  |dR/dt| ~ 3.3e5 /s: a 25 ps shift of our own solution removes about 77% of it
  (issue #23). It therefore cannot be tightened without measuring integrator
  phase, and it hides real regressions inside its own slack.

  The median has no such sensitivity. Across the pinned suite it sits at
  3e-08 to 1.6e-06 where the maxima span 6e-06 to 2.9e-04, so bounding it
  is roughly a hundred times tighter in absolute terms at comparable relative
  margin.
  """
  return float(np.nanmedian(np.abs(left - right)))


def relative(value, expected):
  return float(abs(value - expected) / abs(expected))


def with_path(config, path, value):
  """`config` with one dotted scalar path replaced."""
  parts = path.split(".")
  if len(parts) == 1: return replace(config, **{parts[0]: value})
  return replace(config, **{parts[0]: replace(getattr(config, parts[0]), **{parts[-1]: value})})


def difference_tangent(config, path, times, field="radius_ratio", relative_step=1e-5):
  """d(field)/d(path) by central difference of the forward solve.

  The independent reference for a traced tangent, after W11 stage 5 left one
  implementation. It resolves 1e-09 to 1e-04 depending on the case -- two or more orders
  below every defect this suite has caught: 3.1e-02 for a sampled forcing missing `t0`,
  1.3e-02 for a medium missing `T8`, 1.11 for a temperature scaled by a concrete `T8`,
  0.80 for a polytropic pressure on a thermal path. It cannot distinguish two
  already-correct routes at 1e-13, which is what the deleted `Dual` reference did; it
  can distinguish a missing term, which is what the defects were.

  The step is SIGNED -- `base * relative_step`, not `abs(base) * relative_step`. On a
  negative base an unsigned divisor reads as a factor-two error in the tangent, which is
  indistinguishable from a real defect until you look.
  """
  parts = path.split(".")
  holder = config if len(parts) == 1 else getattr(config, parts[0])
  base = float(getattr(holder, parts[-1]))
  step = base * relative_step
  if step == 0.0: raise ValueError(f"{path!r} is zero here, so a multiplicative step cannot move it")
  ahead = np.asarray(getattr(imr_fast.simulate(times, with_path(config, path, base + step)), field), dtype=float)
  behind = np.asarray(getattr(imr_fast.simulate(times, with_path(config, path, base - step)), field), dtype=float)
  return (ahead - behind) / (2.0 * step)


def tangent_deviation(config, path, times, field="radius_ratio", relative_step=1e-5):
  """Relative deviation of a traced tangent from a central difference.

  Raises rather than returning zero when the traced tangent is identically zero, which
  is how a case that tests nothing announces itself -- several paths have no effect on a
  mechanical configuration and would otherwise compare zero against zero and pass.
  """
  import imr_fast.sensitivity as _sensitivity

  traced = np.asarray(getattr(_sensitivity.solve_with_sensitivities(imr_fast.prepare(config), times, (path,)), field), dtype=float)[..., 0]
  scale = float(np.max(np.abs(traced)))
  if scale == 0.0: raise ValueError(f"the {path!r} tangent of {field!r} is identically zero here, so this comparison tests nothing")
  return float(np.max(np.abs(traced - difference_tangent(config, path, times, field, relative_step)))) / scale
