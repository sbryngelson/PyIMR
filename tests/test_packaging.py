"""The wheel must contain every module the package imports."""

import importlib
import re
import tempfile
import types
from importlib.metadata import version
from pathlib import Path

import numpy as np

import pytest

import pyimr

_ROOT = Path(__file__).resolve().parents[1]
_PACKAGE = Path(pyimr.__file__).resolve().parent

_UNSHIPPED = frozenset({"validate_bubtherm_adiabatic", "validate_thermal_fd"})
_TOOLS = _ROOT / "tools"


def _package_modules():
  return sorted(f"pyimr.{path.stem}" for path in _PACKAGE.glob("*.py") if path.stem != "__init__")


def test_every_declared_dependency_is_actually_imported():
  """A dependency nobody imports still gets installed. `numba` outlived the migration
  that used it by long enough to ship in every install.
  """
  import tomllib

  root = Path(__file__).resolve().parent.parent
  declared = tomllib.loads((root / "pyproject.toml").read_text())["project"]["dependencies"]
  sources = "\n".join(p.read_text() for p in (root / "pyimr").glob("*.py"))
  imported = set(re.findall(r"^\s*(?:import|from)\s+([A-Za-z_][A-Za-z0-9_]*)", sources, re.M))
  unused = [d for d in declared if re.split(r"[><=!\[]", d)[0].strip() not in imported]
  assert not unused, f"declared but never imported: {unused}"


@pytest.mark.slow
def test_the_wheel_ships_every_package_module():
  """The package sits at the repo root, so the suite imports the source tree and a

  module missing from the wheel would still pass every other test here. Under the
  old `src/` layout that could not happen. Building is the only way to check it.
  """
  import subprocess
  import sys
  import zipfile

  with tempfile.TemporaryDirectory() as out:
    build = subprocess.run([sys.executable, "-m", "build", "--wheel", "-o", out], cwd=_ROOT, capture_output=True, text=True, timeout=600)
    assert build.returncode == 0, build.stdout[-2000:] + build.stderr[-2000:]
    wheel = next(Path(out).glob("*.whl"))
    shipped = {Path(n).stem for n in zipfile.ZipFile(wheel).namelist() if n.startswith("pyimr/") and n.endswith(".py")}
    tops = {n.split("/")[0] for n in zipfile.ZipFile(wheel).namelist()}

  on_disk = {path.stem for path in _PACKAGE.glob("*.py")}
  assert on_disk - shipped == set(), f"modules missing from the wheel: {sorted(on_disk - shipped)}"
  assert tops == {"pyimr", f"pyimr-{version('PyIMR')}.dist-info"}, f"wheel ships extra top-level entries: {sorted(tops)}"


def test_the_exposed_version_matches_the_one_that_would_be_released():
  """Reads `pyproject.toml`, not the installed metadata -- comparing `__version__` to
  `importlib.metadata` is the same source twice and cannot fail. The release workflow
  refuses a tag that disagrees with the built wheel; this catches the same drift here,
  where it is still fixable, because a PyPI version cannot be replaced once uploaded.
  """
  import tomllib

  declared = tomllib.loads((_ROOT / "pyproject.toml").read_text())["project"]["version"]
  assert pyimr.__version__ == declared, f"__version__ is {pyimr.__version__}, pyproject says {declared}"


@pytest.mark.parametrize("module", _package_modules())
def test_every_shipped_module_imports(module):
  """A module that ships but cannot be imported -- a stale intra-package import,"""
  assert importlib.import_module(module) is not None


def test_modules_shadowed_by_a_re_export_are_still_importable():
  """`__init__` re-exports functions named `_rhs` and `_stress` from modules of"""
  for name in ("pyimr._rhs", "pyimr._stress"):
    assert isinstance(importlib.import_module(name), types.ModuleType)


def test_no_importable_modules_remain_at_the_repo_root():
  """A loose `.py` beside the package is a top-level name that shadows or leaks."""
  stray = sorted(path.stem for path in _ROOT.glob("*.py"))
  assert stray == [], f"unexpected top-level modules: {stray}"


def test_unshipped_scripts_exist():
  """Keeps `_UNSHIPPED` honest: a renamed or deleted dev script would otherwise"""
  missing = sorted(name for name in _UNSHIPPED if not (_TOOLS / f"{name}.py").exists())
  assert not missing, f"_UNSHIPPED names modules that no longer exist: {missing}"


def test_documentation_links_resolve():
  """Splitting the README into `docs/` pages (#30) turned prose cross-references"""
  pages = [_ROOT / "README.md", *sorted((_ROOT / "docs").glob("*.md"))]
  broken = [
    (page.name, target)
    for page in pages
    for target in re.findall(r"\]\((?!https?:|#)([^)#]+)", page.read_text())
    if not (page.parent / target).exists()
  ]
  assert not broken, f"unresolved documentation links: {broken}"


def test_a_parameter_sweep_reuses_one_compiled_program():
  """`p` is passed to the compiled program, not baked into it, so a sweep compiles once.

  Baking it in made the cache key depend on the parameter VALUES: a 20-point sweep cost
  27.3 s against 0.04 s of solving, one fresh XLA compile per point (#163). The closed-form
  materials reach the solve only through `p`, so their fields are keyed by type alone --
  which is safe exactly as long as distinct parameters still give distinct answers, and
  that is asserted here rather than assumed.
  """
  import pyimr
  from pyimr import _jax

  times = np.linspace(0.0, 20e-6, 40)

  def radius(modulus, viscosity=0.1):
    material = pyimr.NeoHookeanKelvinVoigt(modulus, viscosity)
    return pyimr.simulate(times, pyimr.SimulationConfig(225e-6, 225e-6 / 6, material)).radius_ratio

  _jax._COMPILED.clear()
  traces = [radius(modulus) for modulus in (2000.0, 2250.0, 2500.0, 2750.0, 3000.0)]
  assert len(_jax._COMPILED) == 1, f"a sweep should compile once; cached {len(_jax._COMPILED)} programs"

  for earlier, later in zip(traces[:-1], traces[1:], strict=True):
    assert np.abs(np.asarray(earlier) - np.asarray(later)).max() > 1e-6, "sharing a program must not share an answer"

  # a different material type, and a different flag, must not reuse it
  before = len(_jax._COMPILED)
  radius(2500.0, viscosity=0.2)
  assert len(_jax._COMPILED) == before, "viscosity also travels through p"
  pyimr.simulate(times, pyimr.SimulationConfig(225e-6, 225e-6 / 6, pyimr.Zener(2500.0, 0.1, 40e-6, 8e-6)))
  assert len(_jax._COMPILED) > before, "a different material type needs its own program"
