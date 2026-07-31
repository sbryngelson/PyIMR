"""The wheel must contain every module the package imports."""

import importlib
import re
import types
from pathlib import Path

import pytest

import imr_fast

_ROOT = Path(__file__).resolve().parents[1]
_PACKAGE = Path(imr_fast.__file__).resolve().parent

_UNSHIPPED = frozenset({"validate_bubtherm_adiabatic", "validate_thermal_fd"})


def _package_modules():
  return sorted(f"imr_fast.{path.stem}" for path in _PACKAGE.glob("*.py") if path.stem != "__init__")


def test_the_suite_imports_the_installed_package():
  """The point of `src/`. If `imr_fast` resolved from the repo root, the suite"""
  assert _PACKAGE.parent != _ROOT, f"imr_fast imported from the repo root: {_PACKAGE}"
  assert imr_fast.__file__ is not None and Path(imr_fast.__file__).name == "__init__.py"


@pytest.mark.parametrize("module", _package_modules())
def test_every_shipped_module_imports(module):
  """A module that ships but cannot be imported -- a stale intra-package import,"""
  assert importlib.import_module(module) is not None


def test_modules_shadowed_by_a_re_export_are_still_importable():
  """`__init__` re-exports functions named `_rhs` and `_stress` from modules of"""
  for name in ("imr_fast._rhs", "imr_fast._stress"):
    assert isinstance(importlib.import_module(name), types.ModuleType)


def test_no_importable_modules_remain_at_the_repo_root():
  """Anything left beside `src/` is a top-level name that shadows or leaks. The"""
  stray = sorted(path.stem for path in _ROOT.glob("*.py"))
  assert set(stray) == _UNSHIPPED, f"unexpected top-level modules: {sorted(set(stray) - _UNSHIPPED)}"


def test_unshipped_scripts_exist():
  """Keeps `_UNSHIPPED` honest: a renamed or deleted dev script would otherwise"""
  missing = sorted(name for name in _UNSHIPPED if not (_ROOT / f"{name}.py").exists())
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
