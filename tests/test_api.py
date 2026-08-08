import ast
import importlib
import types
import typing
from pathlib import Path

import numpy as np
import pytest

from pyimr import (
  Giesekus,
  InitialState,
  NeoHookeanKelvinVoigt,
  PhysicalParameters,
  SampledForcing,
  SimulationConfig,
  SimulationError,
  SimulationResult,
  Zener,
  prepare,
  simulate,
)


def base_config(**overrides):
  values = {"R0": 225e-6, "Req": 225e-6 / 6, "material": NeoHookeanKelvinVoigt(2500.0, 0.1)}
  values.update(overrides)
  return SimulationConfig(**values)


def test_simulate_matches_prepared_solver():
  times = np.linspace(0.0, 1.2e-4, 100)
  config = base_config()

  result = simulate(times, config)
  prepared = prepare(config).solve(times)

  assert isinstance(result, SimulationResult)
  np.testing.assert_array_equal(result.time_s, times)
  np.testing.assert_array_equal(result.radius_ratio, prepared.radius_ratio)
  np.testing.assert_allclose(result.radius_m, config.R0 * result.radius_ratio)


def test_structured_result_arrays_are_read_only():
  result = simulate(np.linspace(0.0, 1e-5, 10), base_config())

  for array in (result.time_s, result.radius_ratio, result.wall_velocity_m_s, result.internal_pressure_pa, result.stress_integral_pa):
    with pytest.raises(ValueError):
      array.flat[0] = 2.0


def test_the_accepted_materials_are_exactly_the_declared_union():
  """One declaration, checked. The union in `_materials` is what the type checker reads
  and what `_config` admits at runtime; when those were separate hand-written lists,
  adding a material could leave them disagreeing with no test able to see it.
  """
  from typing import get_args

  from pyimr import _config
  from pyimr._materials import MaterialModel

  declared = get_args(MaterialModel)
  assert set(_config._MATERIALS) == set(declared) and len(declared) > 5
  for material in declared:
    assert isinstance(material, type), f"{material} is not a class the isinstance check can use"


def test_no_entry_point_freezes_the_arrays_it_was_handed():
  """Results are read-only; the caller's own inputs must not become read-only with them.

  `np.asarray` on a float64 array returns that same array, so freezing without copying
  reached back through the call: `solve_sweep` used to hand `times` back unwritable.
  """
  problem = prepare(base_config())
  entries = (
    lambda t: simulate(t, base_config()),
    lambda t: problem.solve(t),
    lambda t: problem.solve_states(t),
    lambda t: problem.state_tangents(t),
    lambda t: problem.solve_sweep(t, ("material.shear_modulus_pa",), [[2000.0], [3000.0]]),
    lambda t: problem.solve_with_sensitivities(t, ("material.shear_modulus_pa",)),
  )
  for index, entry in enumerate(entries):
    times = np.linspace(0.0, 2e-5, 12)
    entry(times)
    assert times.flags.writeable, f"entry point {index} froze the caller's time grid"
    times[0] = 0.0  # the failure this actually causes downstream


def test_prepared_problem_is_reusable_and_returns_active_fields():
  times = np.linspace(0.0, 2e-5, 25)
  config = base_config(material=Zener(2500.0, 0.1, relaxation_time_s=1e-6), bubtherm=1, medtherm=1, masstrans=1, vapor=1, Nt=9, Mt=7)
  problem = prepare(config)

  first = problem.solve(times)
  second = problem.solve(times)

  np.testing.assert_array_equal(first.radius_ratio, second.radius_ratio)
  # these four are `None` unless the matching option is on, which this config turns on
  assert first.bubble_temperature_k is not None and first.medium_temperature_k is not None
  assert first.vapor_mass_fraction is not None and first.stress_state is not None
  assert first.bubble_temperature_k.shape == (times.size, config.Nt)
  assert first.medium_temperature_k.shape == (times.size, config.Mt)
  assert first.vapor_mass_fraction.shape == (times.size, config.Nt)
  assert first.stress_state.shape == (times.size, 1)
  assert first.stats.success and first.stats.nfev > 0


def test_sampled_constant_forcing_matches_analytic_forcing():
  times = np.linspace(0.0, 2e-5, 30)
  analytic = simulate(times, base_config(pA=4e4))
  sampled = simulate(times, base_config(sampled_forcing=SampledForcing(time_s=(times[0], times[-1]), pressure_pa=(4e4, 4e4))))

  deviation = float(np.nanmax(np.abs(np.asarray(sampled.radius_ratio) - np.asarray(analytic.radius_ratio))))
  assert deviation < 1e-14, f"a constant sampled forcing should match the analytic one; differs by {deviation:.2e}"


def test_configurable_physics_and_initial_conditions_are_dimensional():
  times = np.linspace(0.0, 1e-6, 5)
  config = base_config(
    bubtherm=1,
    physics=PhysicalParameters(far_field_pressure_pa=9e4, medium_density_kg_m3=1000.0, polytropic_exponent=1.3),
    initial=InitialState(wall_velocity_m_s=2.0, internal_pressure_pa=1.5e5, bubble_temperature_k=310.0),
  )

  result = simulate(times, config)

  assert result.internal_pressure_pa is not None and result.bubble_temperature_k is not None
  assert result.wall_velocity_m_s[0] == pytest.approx(2.0)
  assert result.internal_pressure_pa[0] == pytest.approx(1.5e5)
  assert result.bubble_temperature_k[0, 0] == pytest.approx(310.0)


def test_distributed_constitutive_state_uses_prepared_grid():
  model = Giesekus(viscosity_pa_s=0.1, relaxation_time_s=2e-6, retardation_time_s=4e-7, mobility=0.1, points=24)
  result = simulate(np.linspace(0.0, 2e-6, 5), base_config(material=model))

  assert result.stress_state is not None and result.stress_reference_radius_ratio is not None
  assert result.stress_state.shape == (5, 2 * model.points)
  assert result.stress_reference_radius_ratio.shape == (model.points,)
  with pytest.raises(ValueError):
    result.stress_reference_radius_ratio[0] = 2.0


@pytest.mark.parametrize(
  ("override", "message"),
  [
    ({"R0": 0.0}, "R0 must be finite and positive"),
    ({"dynamics": "gilmore"}, "needs a liquid_eos"),
    ({"dynamics": "spinodal"}, "unknown dynamics"),
    ({"liquid_eos": "tait"}, "takes no liquid_eos"),
    ({"medtherm": 1}, "medtherm=1 requires bubtherm=1"),
    ({"masstrans": 1}, "masstrans=1 requires bubtherm=1"),
  ],
)
def test_config_rejects_invalid_inputs(override, message):
  with pytest.raises(ValueError, match=message):
    base_config(**override)


@pytest.mark.parametrize("times", [[0.0], [0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, np.nan]])
def test_simulation_rejects_invalid_time_grids(times):
  with pytest.raises(ValueError):
    simulate(times, base_config())


def test_structured_api_requires_config():
  with pytest.raises(TypeError, match="SimulationConfig"):
    simulate([0.0, 1.0], object())  # pyright: ignore[reportArgumentType] - the wrong type IS the test


PUBLIC_MODULES = ("pyimr", "pyimr.sensitivity", "pyimr.inference", "pyimr.data", "pyimr.design", "pyimr.pymc_op", "pyimr.assimilation", "pyimr.optimize", "pyimr.pareto", "pyimr.discriminate", "pyimr.measure")
PACKAGE = "pyimr"


def _foreign(value):
  owner = getattr(value, "__module__", None)
  return owner is not None and owner.split(".")[0] != PACKAGE


@pytest.mark.parametrize("name", PUBLIC_MODULES)
def test_all_is_declared_sorted_and_defined(name):
  module = importlib.import_module(name)
  assert hasattr(module, "__all__"), f"{name} declares no __all__"
  missing = [n for n in module.__all__ if not hasattr(module, n)]
  assert not missing, f"{name}.__all__ lists undefined names: {missing}"
  assert list(module.__all__) == sorted(module.__all__)


@pytest.mark.parametrize("name", PUBLIC_MODULES)
def test_star_import_leaks_no_foreign_names(name):
  namespace = {}
  exec(f"from {name} import *", namespace)
  leaked = []
  for key, value in namespace.items():
    if key.startswith("__"):
      continue
    if isinstance(value, types.ModuleType):
      leaked.append((key, "module"))
    elif typing.get_origin(value) in (typing.Union, types.UnionType):
      bad = [a for a in typing.get_args(value) if _foreign(a)]
      if bad:
        leaked.append((key, bad))
    elif _foreign(value):
      leaked.append((key, value.__module__))
  assert not leaked, f"{name} re-exports foreign names: {leaked}"


def test_equilibrium_radius_rejects_unbracketed_input():
  from pyimr import data

  with pytest.raises(ValueError, match="no equilibrium below R0"):
    data.equilibrium_radius(225e-6, 5e5)


def test_natural_frequency_rejects_equilibrium_above_maximum():
  from pyimr import data

  with pytest.raises(ValueError, match="strictly inside"):
    data.natural_frequency(225e-6, 300e-6, 2500.0, 0.1)


@pytest.mark.parametrize("time,radius", [([0.0, 1.0], [1.0, 1.0]), ([0.0, 1.0, 0.5, 2.0, 3.0], [1.0, 1.0, 1.0, 1.0, 1.0])])
def test_collapse_features_rejects_bad_traces(time, radius):
  from pyimr import data

  with pytest.raises(ValueError):
    data.collapse_features(time, radius)


def test_max_step_forces_finer_integration():
  times = np.linspace(0.0, 60e-6, 200)
  loose = simulate(times, base_config())
  capped = simulate(times, base_config(max_step_s=1e-7))
  assert capped.stats.nfev > loose.stats.nfev
  assert np.nanmax(np.abs(loose.radius_ratio - capped.radius_ratio)) < 1e-4


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
def test_max_step_rejects_invalid(bad):
  with pytest.raises(ValueError, match="max_step_s"):
    base_config(max_step_s=bad)


def test_step_budget_is_honoured_after_an_unbounded_solve():
  """The budget is static to the compiled program, so it has to key the compile cache:
  running the default first must not let a later tight budget reuse that program.
  """
  times = np.linspace(0.0, 60e-6, 200)
  spent = simulate(times, base_config()).stats.nfev
  assert simulate(times, base_config(max_steps=10 * spent)).stats.nfev == spent
  with pytest.raises(SimulationError, match="maximum number of solver steps"):
    simulate(times, base_config(max_steps=10))


@pytest.mark.parametrize("script", ["validate_thermal_fd.py", "validate_bubtherm_adiabatic.py"])
def test_standalone_validation_scripts_still_run(script):
  import pathlib
  import subprocess
  import sys

  root = pathlib.Path(__file__).resolve().parent.parent
  done = subprocess.run([sys.executable, str(root / "tools" / script)], capture_output=True, text=True, cwd=root, timeout=300)
  assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]


ERRSTATE_ALLOWED: set[str] = set()


def _own_nodes(node):
  for child in ast.iter_child_nodes(node):
    if not isinstance(child, ast.FunctionDef):
      yield child
      yield from _own_nodes(child)


def _errstate_sites(node, prefix):
  for child in ast.iter_child_nodes(node):
    nested = isinstance(child, ast.FunctionDef)
    qualified = f"{prefix}.{child.name}" if nested else prefix
    if nested and any(isinstance(n, ast.Attribute) and n.attr == "errstate" for n in _own_nodes(child)):
      yield qualified
    yield from _errstate_sites(child, qualified)


def test_floating_point_suppression_stays_where_it_was_argued_for():
  """#35's own miscount is the argument for checking this structurally."""
  root = Path(str(importlib.import_module(PACKAGE).__file__)).resolve().parent
  found = {site for path in sorted(root.glob("*.py")) for site in _errstate_sites(ast.parse(path.read_text()), path.stem)}
  assert found == ERRSTATE_ALLOWED, f"floating-point suppression moved: {found ^ ERRSTATE_ALLOWED}; see #35"


def test_a_prepared_problem_survives_pickling():
  """Every `workers > 1` path sends this to another process. `parameters` is a

  mappingproxy, which pickle refuses, so all of them died before running any work.
  Checked here rather than only through a subprocess, because this is the actual cause
  and the check costs nothing.
  """
  import pickle

  times = np.linspace(0.0, 1e-5, 20)
  problem = prepare(base_config(bubtherm=1, Nt=7))
  restored = pickle.loads(pickle.dumps(problem))

  assert type(restored.parameters).__name__ == "mappingproxy", "the round trip must not quietly widen the type"
  np.testing.assert_array_equal(problem.solve(times).radius_ratio, restored.solve(times).radius_ratio)
