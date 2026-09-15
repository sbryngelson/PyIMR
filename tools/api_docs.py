"""Build the API reference over every public module, not just the package.

`pyimr/__init__.py` defines `__all__`, and pdoc takes that as the complete list of
what to document -- so `python -m pdoc pyimr` produces one page and silently omits
`pyimr.inference`, `pyimr.selection` and the rest. The CI step carried a hand-written
module list instead, which was forgotten as modules were added: seven names out of
nineteen by the time anyone looked.

The list is derived here, by the rule the package already uses for "public": a
module that does not start with an underscore and declares `__all__`.

  python tools/api_docs.py            serve at http://localhost:8080
  python tools/api_docs.py -o site    write HTML to site/
"""

import importlib
import pathlib
import pkgutil
import subprocess
import sys

# So the source tree works uninstalled too; pip's editable install makes this a no-op.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pyimr


def public_modules():
  names = (f"pyimr.{m.name}" for m in pkgutil.iter_modules(pyimr.__path__) if not m.name.startswith("_"))
  return ["pyimr"] + [n for n in names if hasattr(importlib.import_module(n), "__all__")]


if __name__ == "__main__":
  sys.exit(subprocess.call([sys.executable, "-m", "pdoc", *sys.argv[1:], *public_modules()]))
