"""The wheel must contain every module the package imports."""

import importlib
import re
import tempfile
import types
from importlib.metadata import version
from pathlib import Path

import pytest

import pyimr

_ROOT = Path(__file__).resolve().parents[1]
_PACKAGE = Path(pyimr.__file__).resolve().parent

_UNSHIPPED = frozenset({"validate_bubtherm_adiabatic", "validate_thermal_fd"})
_TOOLS = _ROOT / "tools"


def _package_modules():
  return sorted(f"pyimr.{path.stem}" for path in _PACKAGE.glob("*.py") if path.stem != "__init__")


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
    build = subprocess.run(
      [sys.executable, "-m", "build", "--wheel", "-o", out], cwd=_ROOT, capture_output=True, text=True, timeout=600
    )
    assert build.returncode == 0, build.stdout[-2000:] + build.stderr[-2000:]
    wheel = next(Path(out).glob("*.whl"))
    shipped = {Path(n).stem for n in zipfile.ZipFile(wheel).namelist() if n.startswith("pyimr/") and n.endswith(".py")}
    tops = {n.split("/")[0] for n in zipfile.ZipFile(wheel).namelist()}

  on_disk = {path.stem for path in _PACKAGE.glob("*.py")}
  assert on_disk - shipped == set(), f"modules missing from the wheel: {sorted(on_disk - shipped)}"
  assert tops == {"pyimr", f"pyimr-{version('PyIMR')}.dist-info"}, f"wheel ships extra top-level entries: {sorted(tops)}"


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
