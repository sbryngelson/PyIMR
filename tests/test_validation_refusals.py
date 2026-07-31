"""What the package refuses, and the message it refuses with."""

import numpy as np
import pytest

import imr_fast
from imr_fast import thermal_fd
from _validation_support import NHKV, R0, REQ

SECTION = "8. Refusals"

_FORCING = imr_fast.SampledForcing(time_s=(0.0, 1e-5, 2e-5), pressure_pa=(0.0, 1e4, 0.0))


def _config(**overrides):
  overrides.setdefault("material", NHKV)
  return imr_fast.SimulationConfig(R0=R0, Req=REQ, **overrides)


_CONFIG_REFUSALS = [
  ("vapor fraction range", ValueError, "initial.vapor_mass_fraction must be between 0 and 1",
   lambda: imr_fast.InitialState(vapor_mass_fraction=1.5)),
  ("vapor fraction negative", ValueError, "initial.vapor_mass_fraction must be between 0 and 1",
   lambda: imr_fast.InitialState(vapor_mass_fraction=-0.1)),
  ("non-finite stress state", ValueError, "initial.stress_state must contain finite values",
   lambda: imr_fast.InitialState(stress_state=(0.0, np.inf))),
  ("bracket expansions", ValueError, "collapse.maximum_bracket_expansions must be a positive integer",
   lambda: imr_fast.CollapseInitialization(maximum_bracket_expansions=0)),
  ("material type", TypeError, "material must be a supported material model",
   lambda: _config(material=object())),
  ("physics type", TypeError, "physics must be PhysicalParameters",
   lambda: _config(physics=object())),
  ("initial type", TypeError, "initial must be InitialState",
   lambda: _config(initial=object())),
  ("sampled forcing type", TypeError, "sampled_forcing must be SampledForcing",
   lambda: _config(sampled_forcing=object())),
  ("collapse type", TypeError, "collapse must be CollapseInitialization",
   lambda: _config(collapse=object())),
  ("thermal name", ValueError, "thermal must be 'fd' or 'spectral'",
   lambda: _config(bubtherm=1, thermal="chebyshev")),
  ("max step", ValueError, "max_step_s must be finite and positive",
   lambda: _config(max_step_s=0.0)),
  ("sampled plus analytic", ValueError, "sampled_forcing cannot be combined with analytic forcing",
   lambda: _config(sampled_forcing=_FORCING, wave_type=1, pA=1e4)),
  ("collapse needs memory", ValueError, "collapse initialization requires a material with memory",
   lambda: _config(collapse=imr_fast.CollapseInitialization())),
  ("collapse excludes stress state", ValueError, "collapse initialization cannot be combined with initial.stress_state",
   lambda: _config(material=imr_fast.OldroydB(0.1, 2e-6, 4e-7), collapse=imr_fast.CollapseInitialization(),
                   initial=imr_fast.InitialState(stress_state=(0.0, 0.0)))),
  ("collapse excludes wall velocity", ValueError, "collapse initialization requires zero observed wall velocity",
   lambda: _config(material=imr_fast.OldroydB(0.1, 2e-6, 4e-7), collapse=imr_fast.CollapseInitialization(),
                   initial=imr_fast.InitialState(wall_velocity_m_s=-1.0))),
]


@pytest.mark.parametrize("label,error,message,build", _CONFIG_REFUSALS, ids=[c[0] for c in _CONFIG_REFUSALS])
def test_configuration_refusals(label, error, message, build):
  with pytest.raises(error, match=message):
    build()


_MATERIAL_REFUSALS = [
  ("Oldroyd-B retardation greater", ValueError, "retardation_time_s must be no greater than relaxation_time_s",
   lambda: imr_fast.OldroydB(0.1, 2e-6, 3e-6)),
  ("Giesekus retardation equal", ValueError, "retardation_time_s must be less than relaxation_time_s",
   lambda: imr_fast.Giesekus(0.1, 2e-6, 2e-6, 0.2, points=12)),
  ("Ogden non-positive modulus", ValueError, r"Ogden requires sum\(shear_moduli_pa \* exponents\) > 0",
   lambda: imr_fast.Ogden((1000.0,), (-2.0,))),
  ("Ogden mismatched lengths", ValueError, "Ogden requires equal, non-empty shear_moduli_pa and exponents",
   lambda: imr_fast.Ogden((1000.0, 500.0), (2.0,))),
  ("Ogden zero exponent", ValueError, "Ogden exponents must be finite and non-zero",
   lambda: imr_fast.Ogden((1000.0,), (0.0,))),
  ("elastic type", TypeError, "elastic must be a supported elastic model",
   lambda: imr_fast.InstantaneousMaterial(elastic=object())),  # pyright: ignore[reportArgumentType]
  ("viscous type", TypeError, "viscous must be a supported viscous model",
   lambda: imr_fast.InstantaneousMaterial(viscous=object())),  # pyright: ignore[reportArgumentType]
  ("empty instantaneous material", ValueError, "an instantaneous material requires an elastic or viscous law",
   lambda: imr_fast.InstantaneousMaterial()),
  ("quadrature points", ValueError, "quadrature_points must be an integer >= 8",
   lambda: imr_fast.InstantaneousMaterial(elastic=imr_fast.NeoHookean(2500.0), quadrature_points=4)),
]


@pytest.mark.parametrize("label,error,message,build", _MATERIAL_REFUSALS, ids=[c[0] for c in _MATERIAL_REFUSALS])
def test_material_refusals(label, error, message, build):
  with pytest.raises(error, match=message):
    build()


def test_finite_difference_matrix_refuses_an_unsupported_order():
  with pytest.raises(ValueError, match="order must be 1 or 2"):
    thermal_fd.finite_diff_mat(9, 3, 0)


def test_sensitivity_parameter_refuses_a_non_positive_scale():
  from imr_fast.sensitivity import SensitivityParameter

  with pytest.raises(ValueError, match="scale must be finite and positive"):
    SensitivityParameter("R0", scale=0.0)


_TIMES = np.linspace(1e-6, 2e-5, 8)
_VALUES = np.full(_TIMES.size, 1e-4)


def _observation(**overrides):
  from imr_fast.inference import FieldObservation

  fields = {"field": "radius_m", "time_s": _TIMES, "values": _VALUES, "standard_deviation": 5e-7}
  fields.update(overrides)
  return FieldObservation(fields["field"], fields["time_s"], fields["values"], fields["standard_deviation"])


def _inference_refusals():
  from imr_fast.inference import FieldObservation, InferenceParameter, prepare_inference

  good = InferenceParameter("material.shear_modulus_pa", 1500.0, 4000.0)
  return [
    ("bounds not increasing", ValueError, "inference bounds must be finite and increasing",
     lambda: InferenceParameter("R0", 2.0, 1.0)),
    ("bounds not finite", ValueError, "inference bounds must be finite and increasing",
     lambda: InferenceParameter("R0", 1.0, np.inf)),
    ("unknown transform", ValueError, "inference transform must be 'linear' or 'log'",
     lambda: InferenceParameter("R0", 1.0, 2.0, transform="sqrt")),
    ("log bounds must be positive", ValueError, "log-transformed inference bounds must be positive",
     lambda: InferenceParameter("R0", 0.0, 2.0, transform="log")),
    ("time_s not one-dimensional", ValueError, "must be one-dimensional with at least",
     lambda: _observation(time_s=_TIMES.reshape(2, 4), values=_VALUES.reshape(2, 4))),
    ("deviation shape", ValueError, "must be scalar or match time_s",
     lambda: _observation(standard_deviation=np.ones(3))),
    ("non-finite observation", ValueError, "observations and deviations must be finite",
     lambda: _observation(values=np.where(np.arange(_TIMES.size) == 2, np.nan, _VALUES))),
    ("non-positive deviation", ValueError, "must be positive",
     lambda: _observation(standard_deviation=0.0)),
    ("times not increasing", ValueError, "must be non-negative and increasing",
     lambda: _observation(time_s=_TIMES[::-1])),
    ("correlation time", ValueError, "correlation_time_s must be finite and positive",
     lambda: FieldObservation("radius_m", _TIMES, _VALUES, 5e-7, correlation_time_s=0.0)),
    ("unknown field", ValueError, "field must be one of",
     lambda: _observation(field="not_a_field")),
    ("observation type", TypeError, "observations must be RadiusObservation or FieldObservation",
     lambda: prepare_inference(_config(), object(), (good,))),
    ("no parameters", TypeError, "parameters must contain at least one InferenceParameter",
     lambda: prepare_inference(_config(), _observation(), ())),
    ("parameter type", TypeError, "parameters must contain at least one InferenceParameter",
     lambda: prepare_inference(_config(), _observation(), (object(),))),
    ("duplicate paths", ValueError, "inference parameter paths must be unique",
     lambda: prepare_inference(_config(), _observation(), (good, good))),
  ]


@pytest.mark.parametrize("label,error,message,build", _inference_refusals(), ids=[c[0] for c in _inference_refusals()])
def test_inference_refusals(label, error, message, build):
  with pytest.raises(error, match=message):
    build()


def test_batch_evaluation_refuses_malformed_unit_parameters():
  from imr_fast.inference import InferenceParameter, prepare_inference

  problem = prepare_inference(_config(), _observation(), (InferenceParameter("material.shear_modulus_pa", 1500.0, 4000.0),))
  with pytest.raises(ValueError, match="batch parameters must have shape"):
    problem.evaluate_batch(np.zeros(3))
  with pytest.raises(ValueError, match=r"batch parameters must be finite and within \[0, 1\]"):
    problem.evaluate_batch(np.array([[1.5]]))
  with pytest.raises(ValueError, match="workers must be a positive integer"):
    problem.evaluate_batch(np.array([[0.5]]), workers=0)


def test_the_log_transform_is_geometric_and_its_derivative_is_exact(measured):
  """Two claims: the map is geometric, so the unit midpoint lands on the geometric"""
  from imr_fast.inference import InferenceParameter

  lower, upper = 1e-3, 1e3
  parameter = InferenceParameter("material.viscosity_pa_s", lower, upper, transform="log")
  assert parameter.physical_value(0.0) == pytest.approx(lower)
  assert parameter.physical_value(1.0) == pytest.approx(upper)
  assert parameter.physical_value(0.5) == pytest.approx(np.sqrt(lower * upper))

  step = 1e-6
  worst = 0.0
  for unit in (0.1, 0.35, 0.5, 0.9):
    difference = (parameter.physical_value(unit + step) - parameter.physical_value(unit - step)) / (2.0 * step)
    worst = max(worst, abs(parameter.derivative(unit) - difference) / abs(difference))
  measured("log transform derivative", f"rel={worst:.2e}")
  assert worst < 1e-9, worst
