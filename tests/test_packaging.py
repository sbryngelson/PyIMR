"""The wheel must contain every module the package imports.

This used to guard a hand-maintained `py-modules` array, which was forgotten
twice during file splits (#34): the code worked locally and under pytest because
the repo root was on `sys.path` either way, so the failure appeared only in the
built wheel, and only once execution reached the missing import.

The `src/` layout (#61) removes both halves of that. There is no list to forget,
because `packages.find` discovers the tree; and the suite can no longer import
from the working tree, because the working tree no longer holds importable
modules. What is left to check is that those two properties actually hold --
they are load-bearing, and both fail silently.
"""

import importlib
import re
import types
from pathlib import Path

import pytest

import imr_fast

_ROOT = Path(__file__).resolve().parents[1]
_PACKAGE = Path(imr_fast.__file__).resolve().parent

# Standalone development checks: they import the package but nothing imports
# them, they are not referenced by CI or the README, and they are deliberately
# not shipped. Living outside `src/` is what excludes them now, rather than an
# opt-out list -- but they are still named here so that deleting or renaming one
# is a conscious edit.
_UNSHIPPED = frozenset({"validate_bubtherm_adiabatic", "validate_thermal_fd"})


def _package_modules():
  """Every module in the installed package, dotted."""
  return sorted(f"imr_fast.{path.stem}" for path in _PACKAGE.glob("*.py") if path.stem != "__init__")


def test_the_suite_imports_the_installed_package():
  """The point of `src/`. If `imr_fast` resolved from the repo root, the suite
  would be validating the working tree and the wheel would go untested -- which
  is exactly how #34 stayed invisible."""
  assert _PACKAGE.parent != _ROOT, f"imr_fast imported from the repo root: {_PACKAGE}"
  assert imr_fast.__file__ is not None and Path(imr_fast.__file__).name == "__init__.py"


@pytest.mark.parametrize("module", _package_modules())
def test_every_shipped_module_imports(module):
  """A module that ships but cannot be imported -- a stale intra-package import,
  a module left out of the tree -- gives an installed user ModuleNotFoundError
  at the point of use rather than at install time."""
  assert importlib.import_module(module) is not None


def test_modules_shadowed_by_a_re_export_are_still_importable():
  """`__init__` re-exports functions named `_rhs` and `_stress` from modules of
  the same name, so `imr_fast._rhs` is the function -- the `datetime.datetime`
  pattern, and unavoidable while the package root calls both.

  What must stay true is that the modules remain reachable, because the parity
  tests import them directly. `from imr_fast import _rhs` gets the function;
  `from imr_fast._rhs import ...` gets the module. Only the second is reliable.
  """
  for name in ("imr_fast._rhs", "imr_fast._stress"):
    assert isinstance(importlib.import_module(name), types.ModuleType)


def test_no_importable_modules_remain_at_the_repo_root():
  """Anything left beside `src/` is a top-level name that shadows or leaks. The
  dev scripts are the only permitted exception."""
  stray = sorted(path.stem for path in _ROOT.glob("*.py"))
  assert set(stray) == _UNSHIPPED, f"unexpected top-level modules: {sorted(set(stray) - _UNSHIPPED)}"


def test_unshipped_scripts_exist():
  """Keeps `_UNSHIPPED` honest: a renamed or deleted dev script would otherwise
  silently weaken the check above."""
  missing = sorted(name for name in _UNSHIPPED if not (_ROOT / f"{name}.py").exists())
  assert not missing, f"_UNSHIPPED names modules that no longer exist: {missing}"


def test_documentation_links_resolve():
  """Splitting the README into `docs/` pages (#30) turned prose cross-references
  into paths, and a path that stops resolving breaks silently: nothing imports
  these files and nothing renders them in CI."""
  pages = [_ROOT / "README.md", *sorted((_ROOT / "docs").glob("*.md"))]
  broken = [
    (page.name, target)
    for page in pages
    for target in re.findall(r"\]\((?!https?:|#)([^)#]+)", page.read_text())
    if not (page.parent / target).exists()
  ]
  assert not broken, f"unresolved documentation links: {broken}"
