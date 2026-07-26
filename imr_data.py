"""Trace-side estimators ported from IMR-vanilla.

These bridge a measured R(t) history to a prepared solver or inference
problem: equilibrium radius, natural frequency, and collapse/rebound feature
extraction. Video processing (IMR-vanilla ``calcRofT/``) is deliberately out of
scope; that is image analysis, not solver scope.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.optimize import brentq

import imr_fast
from imr_fast import C8, KAPPA, P8, RHO, SURF, pvsat

__all__ = [
  "collapse_features",
  "equilibrium_radius",
  "natural_frequency",
  "resolution_convergence",
  "saturated_vapor_pressure",
]


def _trace(time_s, radius_m):
  time = np.asarray(time_s, dtype=float)
  radius = np.asarray(radius_m, dtype=float)
  if time.ndim != 1 or radius.ndim != 1 or time.size != radius.size:
    raise ValueError("time_s and radius_m must be 1-D arrays of equal length")
  if time.size < 5:
    raise ValueError("a trace needs at least 5 samples")
  if not np.all(np.diff(time) > 0.0):
    raise ValueError("time_s must be strictly increasing")
  if not (np.all(np.isfinite(time)) and np.all(np.isfinite(radius))):
    raise ValueError("time_s and radius_m must be finite")
  return time, radius


def equilibrium_radius(
  R0_m,
  gas_pressure_pa,
  *,
  far_field_pressure_pa=P8,
  surface_tension_n_m=SURF,
  polytropic_exponent=KAPPA,
  vapor_pressure_pa=0.0,
):
  """Equilibrium radius from the initial gas partial pressure.

  Solves ``Pg0*(R0/Req)**(3*kappa) + Pv == P8 + 2*S/Req``, the stress-free
  equilibrium assumed when setting up the elastic term. This is IMR-vanilla
  ``IMRCalc_Req`` for its cold, no-mass-transfer branch, stated dimensionally.
  """
  if not (R0_m > 0.0 and np.isfinite(R0_m)):
    raise ValueError("R0_m must be finite and positive")
  if not (gas_pressure_pa > 0.0 and np.isfinite(gas_pressure_pa)):
    raise ValueError("gas_pressure_pa must be finite and positive")

  def residual(ratio):
    radius = ratio * R0_m
    return (
      gas_pressure_pa * ratio ** (-3.0 * polytropic_exponent)
      + vapor_pressure_pa
      - far_field_pressure_pa
      - 2.0 * surface_tension_n_m / radius
    )

  lower, upper = 1e-6, 1.0
  if residual(upper) > 0.0:
    raise ValueError("no equilibrium below R0: the gas pressure already exceeds the far-field pressure at R0")
  while residual(lower) < 0.0:
    lower *= 0.1
    if lower < 1e-14:
      raise ValueError("could not bracket an equilibrium radius")
  return brentq(residual, lower, upper, xtol=1e-15) * R0_m


def natural_frequency(
  maximum_radius_m,
  equilibrium_radius_m,
  shear_modulus_pa,
  viscosity_pa_s,
  *,
  far_field_pressure_pa=P8,
  density_kg_m3=RHO,
  sound_speed_m_s=C8,
  surface_tension_n_m=SURF,
  polytropic_exponent=KAPPA,
):
  """Linearised natural frequency and damping about equilibrium, in rad/s.

  Returns ``(omega_n, beta)`` for ``xddot + 2*beta*xdot + omega_n**2*x = 0``,
  from the Rayleigh-Plesset equation linearised about ``Req`` in a
  Kelvin-Voigt medium::

      omega_n**2 = (3k*(P8 + 2S/Req - Pv) - 2S/Req + 4G) / (rho*Req**2)
      beta       = 2*mu/(rho*Req**2) + omega_n**2*Req/(2*c)

  ``maximum_radius_m`` is accepted for interface symmetry with the rest of
  the module and is not used by the linearisation.

  IMR-vanilla's ``calc_omega_N`` is **not** ported. It is a scratch script
  with hardcoded values, and its formula treats the gas pressure at ``Rmax``
  as ``P8 + Laplace`` (the equilibrium condition, not the ``Rmax``
  condition), which inflates the stiffness by ``alpha**(-3*kappa)``. On the
  standard reference case it returns 2.30e+07 rad/s against a measured
  rebound frequency of 5.3e+05 -- 42x too high. The form above reproduces
  Minnaert exactly in the gas-only limit and matches the measurement to 3%.
  """
  if not (equilibrium_radius_m > 0.0 and np.isfinite(equilibrium_radius_m)):
    raise ValueError("equilibrium_radius_m must be finite and positive")
  if not 0.0 < equilibrium_radius_m < maximum_radius_m:
    raise ValueError("equilibrium radius must lie strictly inside (0, Rmax)")
  laplace = 2.0 * surface_tension_n_m / equilibrium_radius_m
  stiffness = 3.0 * polytropic_exponent * (far_field_pressure_pa + laplace) - laplace + 4.0 * shear_modulus_pa
  if stiffness <= 0.0:
    raise ValueError("linearised stiffness is non-positive; no oscillation")
  inertia = density_kg_m3 * equilibrium_radius_m**2
  omega = np.sqrt(stiffness / inertia)
  damping = 2.0 * viscosity_pa_s / inertia + omega**2 * equilibrium_radius_m / (2.0 * sound_speed_m_s)
  return omega, damping


def collapse_features(time_s, radius_m, *, refine=True):
  """Collapse times and rebound peak radii from a measured trace.

  IMR-vanilla ``calc_3tmins_3Rmaxs`` fits quartics through manually supplied
  index windows. This finds the interior extrema automatically and, when
  ``refine`` is set, locates each to sub-sample accuracy with a local
  three-point parabola. Returns ``(collapse_times_s, peak_radii_m,
  peak_times_s)``.
  """
  time, radius = _trace(time_s, radius_m)
  slope = np.diff(radius)
  turning = np.flatnonzero(slope[:-1] * slope[1:] < 0.0) + 1
  minima = turning[slope[turning] > 0.0]
  maxima = turning[slope[turning] < 0.0]

  def sharpen(index):
    left, middle, right = radius[index - 1], radius[index], radius[index + 1]
    denominator = left - 2.0 * middle + right
    if not refine or denominator == 0.0:
      return time[index], middle
    shift = 0.5 * (left - right) / denominator
    if abs(shift) > 1.0:
      return time[index], middle
    step = time[index + 1] - time[index - 1]
    return (time[index] + 0.5 * shift * step, middle - 0.25 * (left - right) * shift)

  collapse = np.array([sharpen(i)[0] for i in minima])
  peaks = np.array([sharpen(i) for i in maxima]).reshape(-1, 2)
  return collapse, peaks[:, 1], peaks[:, 0]


def resolution_convergence(config, times_s, resolutions, *, field="radius_ratio"):
  """Self-convergence of a configuration under thermal grid refinement.

  Replaces IMR-vanilla ``runIMR_convergence_check``. Solves ``config`` at each
  ``(Nt, Mt)`` pair and reports the max deviation of each against the finest.
  Returns a tuple of ``(resolution, max_deviation)``, finest last with 0.0.
  """
  grids = [tuple(r) if isinstance(r, tuple | list) else (r, r) for r in resolutions]
  if len(grids) < 2:
    raise ValueError("need at least two resolutions to compare")
  solved = []
  for nt, mt in grids:
    solution = imr_fast.simulate(times_s, replace(config, Nt=nt, Mt=mt))
    solved.append(np.asarray(getattr(solution, field), dtype=float))
  finest = solved[-1]
  return tuple((grid, float(np.nanmax(np.abs(values - finest)))) for grid, values in zip(grids, solved))


def saturated_vapor_pressure(temperature_k):
  """Saturated vapour pressure in Pa; thin wrapper over the solver's fit."""
  return pvsat(temperature_k)
