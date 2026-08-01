"""Shared constants and helpers for the numerical validation suite."""

import functools
from dataclasses import replace
from pathlib import Path

import numpy as np

import pyimr

DATA = Path(__file__).resolve().parent

R0 = 225e-6
REQ = R0 / 6
T0 = R0 / np.sqrt(101325 / 1064)

NHKV = pyimr.NeoHookeanKelvinVoigt(2500.0, 0.1)


def zener():
  return pyimr.Zener(2500.0, 0.1, 2 * T0, 0.4 * T0)


def oldroyd_b():
  return pyimr.OldroydB(0.1, 2 * T0, 0.4 * T0)


@functools.lru_cache(maxsize=None)
def reference(name):
  """A pinned IMRv2 trajectory. Cached -- imr2_s06.csv alone is 234 kB and"""
  return np.loadtxt(DATA / name, delimiter="," if name == "imr2_s06.csv" else None)


@functools.lru_cache(maxsize=None)
def reference_times():
  return reference("ref_t.csv")


def solve_radius(times, material: pyimr.MaterialModel = NHKV, R0_m=R0, Req_m=REQ, **options):
  config = pyimr.SimulationConfig(R0=R0_m, Req=Req_m, material=material, **options)
  return pyimr.simulate(times, config).radius_ratio


def deviation(left, right):
  return float(np.nanmax(np.abs(left - right)))


def median_deviation(left, right):
  """Median pointwise deviation -- the phase-insensitive companion to the max."""
  return float(np.nanmedian(np.abs(left - right)))


def relative(value, expected):
  return float(abs(value - expected) / abs(expected))


def with_path(config, path, value):
  """`config` with one dotted scalar path replaced."""
  parts = path.split(".")
  if len(parts) == 1: return replace(config, **{parts[0]: value})
  return replace(config, **{parts[0]: replace(getattr(config, parts[0]), **{parts[-1]: value})})


def difference_tangent(config, path, times, field="radius_ratio", relative_step=1e-5):
  """d(field)/d(path) by central difference of the forward solve."""
  parts = path.split(".")
  holder = config if len(parts) == 1 else getattr(config, parts[0])
  base = float(getattr(holder, parts[-1]))
  step = base * relative_step
  if step == 0.0: raise ValueError(f"{path!r} is zero here, so a multiplicative step cannot move it")
  ahead = np.asarray(getattr(pyimr.simulate(times, with_path(config, path, base + step)), field), dtype=float)
  behind = np.asarray(getattr(pyimr.simulate(times, with_path(config, path, base - step)), field), dtype=float)
  return (ahead - behind) / (2.0 * step)


def tangent_deviation(config, path, times, field="radius_ratio", relative_step=1e-5):
  """Relative deviation of a traced tangent from a central difference."""
  import pyimr.sensitivity as _sensitivity

  traced = np.asarray(getattr(_sensitivity.solve_with_sensitivities(pyimr.prepare(config), times, (path,)), field), dtype=float)[..., 0]
  scale = float(np.max(np.abs(traced)))
  if scale == 0.0: raise ValueError(f"the {path!r} tangent of {field!r} is identically zero here, so this comparison tests nothing")
  return float(np.max(np.abs(traced - difference_tangent(config, path, times, field, relative_step)))) / scale
