"""Run pyright and fail only on diagnostics that are not already recorded.

pyright has no built-in baseline. This started at 114 pre-existing diagnostics
(issue #33), because requiring them cleared before any type checking runs at all
is how type checking never gets started.

**The baseline is now empty**, so the bar is simply zero and this is the runner
that enforces it (#142). It stays rather than being replaced by a bare `pyright`
call for two reasons: it supplies `--pythonpath`, without which pyright resolves
against whatever interpreter it finds and reports every third-party import as
missing; and if a dependency ships a broken stub, recording that one diagnostic
is better than turning the gate off.

Fingerprint is (path, rule, message), deliberately without a line number:
including one makes the baseline churn on every unrelated edit, which trains
people to regenerate it without reading it. Counts are compared, so a second
occurrence of an already-recorded message in the same file is still caught.

  python tools/pyright_baseline.py            check, non-zero exit on new
  python tools/pyright_baseline.py --update   rewrite the baseline
"""

import argparse
import collections
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASELINE = pathlib.Path(__file__).resolve().parent / "pyright_baseline.json"


def _fingerprints():
  # --pythonpath explicitly: without it pyright resolves against whatever
  # interpreter it discovers, which was not the one holding numpy/scipy/numba,
  # and every third-party import landed in the baseline as
  # reportMissingImports. Baselining those would mask a genuinely broken
  # import, which is the opposite of the point.
  process = subprocess.run(
    [sys.executable, "-m", "pyright", "--pythonpath", sys.executable, "--outputjson", str(ROOT)], capture_output=True, text=True, cwd=ROOT
  )
  if not process.stdout.strip():
    raise SystemExit(f"pyright produced no output\n{process.stderr}")
  report = json.loads(process.stdout)
  counts = collections.Counter()
  for diagnostic in report["generalDiagnostics"]:
    path = pathlib.Path(diagnostic["file"])
    relative = path.relative_to(ROOT).as_posix() if path.is_absolute() else path.as_posix()
    message = " ".join(diagnostic["message"].split())
    counts[(relative, diagnostic.get("rule", "(none)"), message)] += 1
  return counts


def _load():
  if not BASELINE.exists():
    return collections.Counter()
  return collections.Counter({(k["file"], k["rule"], k["message"]): k["count"] for k in json.loads(BASELINE.read_text())})


def _store(counts):
  entries = [{"file": file, "rule": rule, "message": message, "count": count} for (file, rule, message), count in sorted(counts.items())]
  BASELINE.write_text(json.dumps(entries, indent=2) + "\n")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--update", action="store_true", help="rewrite the baseline from the current state")
  arguments = parser.parse_args()

  current = _fingerprints()
  if arguments.update:
    _store(current)
    print(f"baseline updated: {sum(current.values())} diagnostics across {len({k[0] for k in current})} files")
    return 0

  baseline = _load()
  added = current - baseline
  removed = baseline - current

  if removed:
    print(f"{sum(removed.values())} baselined diagnostic(s) no longer occur. Run --update to shrink the baseline:")
    for (file, rule, message), count in sorted(removed.items()):
      print(f"  -{count}  {file}  [{rule}]  {message[:100]}")

  if not added:
    print(f"no new type errors ({sum(baseline.values())} baselined)")
    return 0

  print(f"\n{sum(added.values())} NEW type error(s):")
  for (file, rule, message), count in sorted(added.items()):
    print(f"  +{count}  {file}  [{rule}]  {message}")
  return 1


if __name__ == "__main__":
  raise SystemExit(main())
