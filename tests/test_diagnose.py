"""Failure classification: does it name the mechanism, or just restate the symptom?

Every case here is one that actually cost time. `maximum number of solver steps` is the
message for all four, and they want opposite responses -- raise the budget, lower the
stiffening, accept the answer, or stop trusting it.
"""

import numpy as np
import pytest

import pyimr
from pyimr.diagnose import diagnose

SECTION = "19. Failure diagnosis"

_TIMES = np.linspace(0.0, 1.4e-4, 201)
_R0, _REQ = 277e-6, 277e-6 / 7.09


def _config(material, **overrides):
  options = {"dynamics": "keller-miksis", "rtol": 1e-6, "atol": 1e-8, "max_steps": 50_000} | overrides
  return pyimr.SimulationConfig(_R0, _REQ, material, **options)


def test_a_healthy_configuration_reports_ok_and_its_conditioning(measured):
  found = diagnose(_config(pyimr.NeoHookeanKelvinVoigt(2500.0, 0.1)), _TIMES)
  measured("healthy", f"{found.outcome}, amplification {found.amplification:.1e}")
  assert found.outcome == "ok"
  assert found.steps is not None and found.steps > 0
  assert found.amplification is not None and found.amplification < 1e3


def test_a_budget_failure_is_named_as_expensive_rather_than_ill_posed():
  """The same message as every other failure, and the only one a bigger budget fixes."""
  found = diagnose(_config(pyimr.NeoHookeanKelvinVoigt(2500.0, 0.1), max_steps=120), _TIMES)
  assert found.outcome == "budget", found.summary
  assert found.steps is not None and found.steps > 120


def test_a_material_refusing_its_state_is_named_as_such(measured):
  """Gent locks up as `I1 - 3 -> Jm`. Under tracing the guard cannot fire -- a tracer has
  no value to test -- so it reaches the caller as a step-budget failure. The diagnosis has
  to leave the tracer to recover the real reason.
  """
  found = diagnose(_config(pyimr.InstantaneousMaterial(elastic=pyimr.Gent(2500.0, 200.0))), _TIMES)
  measured("Gent", found.summary[:60])
  assert found.outcome == "domain", found.summary
  assert "lock-up" in found.summary


def test_a_runaway_is_named_as_the_model_failing_not_as_a_hard_integration(measured):
  """This case was read as ill-conditioning for a long time, and it is not.

  After its collapse the trajectory expands to R/R0 = 2132 -- identically at rtol 1e-3,
  1e-4 and 1e-5, so a converged property of the equations rather than a tolerance artifact.
  The integrator was faithfully chasing a runaway. Calling that `ill-conditioned` sends the
  reader to look for a better solver, which cannot exist; the honest answer is that the
  model is not physical at these parameters.
  """
  found = diagnose(_config(pyimr.QuadraticZener(4640.0, 1e-4, 2.78e-7, 0.0, 3.59)), _TIMES)
  measured("runaway qSLS", f"{found.outcome}: {found.summary[:44]}")
  assert found.outcome == "runaway", found.summary
  assert "not physical" in found.summary


def test_a_sensitive_trajectory_is_not_reported_as_needing_more_steps():
  """The distinct case the runaway one used to be confused with: a trajectory that
  integrates but has shed the precision demanded of it, so a bigger budget cannot help.
  """
  found = diagnose(_config(pyimr.QuadraticZener(4640.0, 1e-4, 2.78e-7, 0.0, 3.59), max_radius_ratio=None), _TIMES)
  assert found.outcome in {"ill-conditioned", "unresolved"}, found.summary
  assert "more steps will not help" in found.summary.lower() or "not converged" in found.summary.lower()


def test_the_outcome_reads_as_a_sentence():
  found = diagnose(_config(pyimr.NeoHookeanKelvinVoigt(2500.0, 0.1)), _TIMES)
  assert str(found).startswith("ok: ")


@pytest.mark.parametrize("field", ["outcome", "summary"])
def test_every_diagnosis_carries_the_fields_a_caller_branches_on(field):
  found = diagnose(_config(pyimr.NeoHookeanKelvinVoigt(2500.0, 0.1)), _TIMES)
  assert getattr(found, field)
