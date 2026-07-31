"""Harness for the numerical validation suite."""

import pytest

_MEASUREMENTS = []


def pytest_configure(config):
  config.addinivalue_line("markers", "slow: convergence studies that solve at high resolution; deselect with -m 'not slow'")


@pytest.fixture
def measured(request):
  """Record a measured value for the end-of-run table."""
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
