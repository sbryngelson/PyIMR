"""`[tool.setuptools] py-modules` is maintained by hand, and was forgotten twice
during one session of file splits. Both times the code worked locally and under
pytest, because the repo root is on `sys.path` either way -- the failure only
appears in the built wheel, i.e. only for an installed user, and only when
execution reaches the missing import.

These tests check the property directly rather than relying on vigilance. See
issue #34.
"""

import ast
import re
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]

# Standalone development checks: they import the package but nothing imports
# them, they are not referenced by CI or the README, and they are deliberately
# not shipped. Listed explicitly so that adding a module fails loudly while
# adding a dev script stays a conscious edit.
_UNSHIPPED = frozenset({"validate_bubtherm_adiabatic", "validate_thermal_fd"})


def _declared_modules():
  """The `py-modules` list from pyproject.toml."""
  text = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
  if sys.version_info >= (3, 11):
    import tomllib

    return frozenset(tomllib.loads(text)["tool"]["setuptools"]["py-modules"])
  # 3.10 has no tomllib and the project takes no test-time TOML dependency.
  block = re.search(r"py-modules\s*=\s*\[(.*?)\]", text, re.DOTALL)
  assert block is not None, "py-modules array not found in pyproject.toml"
  return frozenset(re.findall(r"[\"']([^\"']+)[\"']", block.group(1)))


def _source_modules():
  """Every top-level module in the repo root."""
  return frozenset(path.stem for path in _ROOT.glob("*.py"))


def _imported_top_level_names(path):
  """Top-level module names imported anywhere in one source file, including
  inside functions -- `imr_fast` defers several imports that way."""
  names = set()
  for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
    if isinstance(node, ast.Import):
      names.update(alias.name.split(".")[0] for alias in node.names)
    elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
      names.add(node.module.split(".")[0])
  return names


def test_pyproject_ships_every_source_module():
  """A new module that nobody remembered to add to `py-modules` fails here."""
  expected = _source_modules() - _UNSHIPPED
  declared = _declared_modules()
  assert declared == expected, (
    f"py-modules is out of sync with the source tree.\n"
    f"  in the tree but not shipped: {sorted(expected - declared)}\n"
    f"  shipped but not in the tree: {sorted(declared - expected)}\n"
    f"If a new module is intentionally not shipped, add it to _UNSHIPPED here."
  )


@pytest.mark.parametrize("module", sorted(_declared_modules()))
def test_shipped_modules_import_only_shipped_modules(module):
  """The failure mode that makes #34 nasty: an installed package that imports
  fine until execution reaches the one module the wheel does not contain.

  Any local import made by a shipped module must itself be shipped.
  """
  declared = _declared_modules()
  local = _source_modules()
  missing = sorted((_imported_top_level_names(_ROOT / f"{module}.py") & local) - declared)
  assert not missing, (
    f"{module}.py imports {missing}, which the wheel does not contain. "
    f"An installed user gets ModuleNotFoundError at the point of use."
  )


def test_unshipped_scripts_exist():
  """Keeps `_UNSHIPPED` honest: a renamed or deleted dev script would otherwise
  silently weaken the completeness check above."""
  missing = sorted(name for name in _UNSHIPPED if not (_ROOT / f"{name}.py").exists())
  assert not missing, f"_UNSHIPPED names modules that no longer exist: {missing}"
