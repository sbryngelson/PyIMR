"""Harness for the numerical validation suite.

The script this replaced printed a table of measured values rather than
pass/fail alone, which is the useful part of its output: a deviation that is
still passing but has moved an order of magnitude is worth seeing. Green
checkmarks lose that. The `measured` fixture keeps it -- tests record their
numbers, and the table is printed in the terminal summary, grouped by the
section each module declares.

See issue #32.
"""

import pytest

_MEASUREMENTS = []


def pytest_configure(config):
  config.addinivalue_line(
    "markers", "slow: convergence studies that solve at high resolution; deselect with -m 'not slow'"
  )


@pytest.fixture
def measured(request):
  """Record a measured value for the end-of-run table.

  `measured(label, "max|dR|=1.2e-05")` -- the caller formats the value, because
  the useful precision and units differ per check.
  """
  section = getattr(request.module, "SECTION", request.module.__name__)

  def record(label, text):
    _MEASUREMENTS.append((section, label, text))

  return record


def pytest_terminal_summary(terminalreporter):
  if not _MEASUREMENTS:
    return
  width = max(len(label) for _, label, _ in _MEASUREMENTS)
  terminalreporter.write_sep("=", "measured values")
  seen = None
  for section, label, text in _MEASUREMENTS:
    if section != seen:
      terminalreporter.write_line("")
      terminalreporter.write_line(section)
      seen = section
    terminalreporter.write_line(f"    {label:{width}s}  {text}")
