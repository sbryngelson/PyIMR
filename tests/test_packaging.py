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


def test_no_submodule_name_is_shadowed_by_an_exported_symbol():
  """`__init__` re-exports internals by name, so a submodule called `_rhs`
  alongside a re-exported function called `_rhs` makes `imr_fast._rhs` mean two
  different things depending on when you look. The re-export wins after package
  init, which silently makes the submodule unreachable by attribute -- that is
  why `_rhs` and `_stress` are `_equations` and `_constitutive` (#61)."""
  shadowed = []
  for module in _package_modules():
    name = module.rpartition(".")[2]
    importlib.import_module(module)
    if not isinstance(getattr(imr_fast, name, None), types.ModuleType):
      shadowed.append(name)
  assert not shadowed, f"submodules shadowed by re-exported names: {shadowed}"


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
