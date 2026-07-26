"""Shared constants and helpers for the numerical validation suite.

Not collected by pytest (no `test_` prefix). The numerical content of these
checks is unchanged from the `run_validation.py` script this replaced; only the
harness around them moved. See issue #32.
"""

import functools
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


def relative(value, expected):
  return float(abs(value - expected) / abs(expected))
