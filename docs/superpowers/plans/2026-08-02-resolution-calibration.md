# Resolution Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Choose `thermal`, `Nt` and solver tolerance from a requested accuracy by measuring on the caller's own problem, instead of from folklore that does not generalize.

**Architecture:** A new leaf module `pyimr/resolution.py` with no dependants inside the package. It calls `pyimr.simulate` and returns a `Resolution` value object; nothing in the package consumes it. A converged reference is built and self-checked first, then a two-stage search runs: `(thermal, Nt)` at tight tolerance, then `rtol` at the winning discretization.

**Tech Stack:** Python 3.12+, numpy, `dataclasses.replace` over the frozen `SimulationConfig`, pytest.

## Global Constraints

- Comments at bare minimum; rationale belongs in commit messages, not inline.
- `ruff check pyimr tests` must pass; line length and style follow the existing config.
- `python tools/pyright_baseline.py` must report no new type errors. Annotate any dict that is splatted into `SimulationConfig`, or every field becomes a union and pyright rejects it.
- Tests must be fast. The full suite budget is about 5 minutes on a two-core runner, so every test here passes small explicit ladders rather than the module defaults.
- `NT_LADDER = (5, 7, 9, 13, 17, 25, 33)` and `RTOL_LADDER = (1e-10, 1e-8, 1e-6, 1e-4, 1e-3)`, verbatim from the spec.
- Reference tolerances are `rtol = 1e-10, atol = 1e-12`; the reference check runs at doubled `Nt` with `rtol = 1e-12, atol = 1e-14`.
- `atol = rtol * 1e-2` throughout stage 2.
- Never rank candidates by an analytic cost proxy. `fd` needs more nodes than spectral and can cross `_CHORD_ABOVE = 80`, where the root finder changes, so cost is not monotone in `Nt`. Rank by measured wall time only.

---

## File Structure

- **Create `pyimr/resolution.py`** — the whole feature. Ladders, `Resolution`, the error metric, the reference, both search stages, `choose_resolution`.
- **Create `tests/test_resolution.py`** — `SECTION = "17. Resolution calibration"`.
- **Modify `README.md`** — a subsection under "Model selection".
- **Modify `CHANGELOG.md`** — an entry under `## 0.3.0 — unreleased`.

No change to `pyimr/__init__.py`. Sibling modules `noise`, `prior` and `selection` are reached as `pyimr.noise` etc. and are not re-exported; follow that.

---

### Task 1: Resolution value object and config substitution

**Files:**
- Create: `pyimr/resolution.py`
- Test: `tests/test_resolution.py`

**Interfaces:**
- Consumes: `pyimr.SimulationConfig` (frozen dataclass; fields `thermal`, `Nt`, `Mt`, `medtherm`, `rtol`, `atol`).
- Produces: `__all__ = ["NT_LADDER", "RTOL_LADDER", "Resolution"]` — declare only what exists;
  `choose_resolution` joins it in Task 5, where it is defined. A forward declaration fails
  pyright's `reportUnsupportedDunderAll`, which `# noqa: F822` does not suppress.
  `NT_LADDER`, `RTOL_LADDER`, `Resolution(thermal: str, Nt: int, rtol: float, atol: float, achieved: float, seconds: float)` with `.apply(config) -> SimulationConfig`; and `_at(config, thermal, nt, rtol, atol) -> SimulationConfig`.

- [ ] **Step 1: Write the failing test**

```python
"""Resolution calibration: choose discretization and tolerance by measuring.

The folklore this replaces was established at single configurations and did not
generalize -- Nt = 7 to 9 held for one collapse and failed by 4x over five. These tests
therefore check the guards as hard as the happy path: an unconverged reference and an
unreachable target must raise rather than return a plausible number.
"""

import numpy as np
import pytest

import pyimr
from pyimr.resolution import NT_LADDER, RTOL_LADDER, Resolution, _at
from _validation_support import NHKV, R0, REQ

SECTION = "17. Resolution calibration"

_TIMES = np.linspace(0.0, 20e-6, 60)


def _config(**kwargs):
  return pyimr.SimulationConfig(R0, REQ, NHKV, bubtherm=1, **kwargs)


def test_the_ladders_are_ascending():
  assert list(NT_LADDER) == sorted(NT_LADDER)
  assert list(RTOL_LADDER) == sorted(RTOL_LADDER)


def test_apply_substitutes_only_the_resolution_fields():
  config = _config(pA=1234.0, radial=2)
  setting = Resolution(thermal="fd", Nt=13, rtol=1e-6, atol=1e-8, achieved=1e-4, seconds=0.01)
  applied = setting.apply(config)

  assert (applied.thermal, applied.Nt, applied.rtol, applied.atol) == ("fd", 13, 1e-6, 1e-8)
  assert applied.pA == 1234.0 and applied.radial == 2 and applied.R0 == config.R0


def test_mt_follows_nt_only_when_the_medium_is_active():
  """`Mt` is tied to `Nt` by design. Leaving it alone when `medtherm=0` matters because
  the medium grid is then unused and rewriting it would be a silent, invisible change.
  """
  setting = Resolution(thermal="spectral", Nt=7, rtol=1e-6, atol=1e-8, achieved=0.0, seconds=0.0)
  assert setting.apply(_config(medtherm=0)).Mt == _config(medtherm=0).Mt
  assert setting.apply(_config(medtherm=1, Mt=100)).Mt == 7


def test_at_builds_the_same_config_apply_would():
  config = _config(medtherm=1)
  built = _at(config, "fd", 9, 1e-7, 1e-9)
  assert (built.thermal, built.Nt, built.Mt, built.rtol, built.atol) == ("fd", 9, 9, 1e-7, 1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_resolution.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'pyimr.resolution'`

- [ ] **Step 3: Write minimal implementation**

```python
"""Choose thermal discretization and solver tolerance by measuring, not by folklore.

Requirements depend on record length, material stiffness and which observable is fitted.
A lookup table sees none of those, so this measures on the caller's own problem and
reports what it achieved.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

__all__ = ["NT_LADDER", "RTOL_LADDER", "Resolution"]

NT_LADDER = (5, 7, 9, 13, 17, 25, 33)
RTOL_LADDER = (1e-10, 1e-8, 1e-6, 1e-4, 1e-3)

_REFERENCE_RTOL, _REFERENCE_ATOL = 1e-10, 1e-12
_CHECK_RTOL, _CHECK_ATOL = 1e-12, 1e-14


@dataclass(frozen=True, slots=True)
class Resolution:
  """A measured setting, with the error and cost it was chosen on."""

  thermal: str
  Nt: int
  rtol: float
  atol: float
  achieved: float
  seconds: float

  def apply(self, config):
    return _at(config, self.thermal, self.Nt, self.rtol, self.atol)


def _at(config, thermal, nt, rtol, atol):
  """`config` with one resolution substituted. `Mt` follows `Nt` only if the medium runs."""
  fields = {"thermal": thermal, "Nt": int(nt), "rtol": float(rtol), "atol": float(atol)}
  if config.medtherm: fields["Mt"] = int(nt)
  return replace(config, **fields)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_resolution.py -q`
Expected: PASS, 4 passed

- [ ] **Step 5: Commit**

```bash
git add pyimr/resolution.py tests/test_resolution.py
git commit -m "Resolution value object and config substitution"
```

---

### Task 2: Per-field error metric

**Files:**
- Modify: `pyimr/resolution.py`
- Test: `tests/test_resolution.py`

**Interfaces:**
- Consumes: `_at` from Task 1.
- Produces: `_solve(config, times, fields) -> dict[str, np.ndarray]`, `_deviation(candidate, reference) -> float`, `_timed(config, times, fields) -> tuple[float, dict]`.

- [ ] **Step 1: Write the failing test**

```python
def test_deviation_is_relative_to_each_field_own_peak(measured):
  """Observables do not converge together -- at identical settings, relative error was
  3.4e-07 for radius against 2.8e-05 for internal pressure, about 80x looser. Scaling by
  each field's own peak is what keeps one target meaningful across fields.
  """
  from pyimr.resolution import _deviation

  reference = {"a": np.array([0.0, 2.0]), "b": np.array([0.0, 200.0])}
  candidate = {"a": np.array([0.0, 2.2]), "b": np.array([0.0, 220.0])}
  # both are 10% of their own peak, so both must report 0.1 rather than 0.2 and 20.0
  assert _deviation(candidate, reference) == pytest.approx(0.1)
  measured("per-field scaling", "10% of peak reads 0.1 for peaks 2 and 200 alike")


def test_deviation_reports_the_worst_field():
  from pyimr.resolution import _deviation

  reference = {"a": np.array([1.0]), "b": np.array([1.0])}
  assert _deviation({"a": np.array([1.01]), "b": np.array([1.5])}, reference) == pytest.approx(0.5)


def test_solve_returns_only_the_requested_fields():
  from pyimr.resolution import _solve

  values = _solve(_at(_config(), "spectral", 5, 1e-6, 1e-8), _TIMES, ("radius_ratio",))
  assert set(values) == {"radius_ratio"}
  assert values["radius_ratio"].shape == _TIMES.shape
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_resolution.py -q -k "deviation or solve_returns"`
Expected: FAIL, `ImportError: cannot import name '_deviation'`

- [ ] **Step 3: Write minimal implementation**

Add to `pyimr/resolution.py` (and add `import time` and `import numpy as np` at the top):

```python
def _solve(config, times, fields):
  from . import simulate

  result = simulate(times, config)
  return {name: np.asarray(getattr(result, name), dtype=float) for name in fields}


def _timed(config, times, fields):
  start = time.perf_counter()
  values = _solve(config, times, fields)
  return time.perf_counter() - start, values


def _deviation(candidate, reference):
  """Worst per-field deviation, each scaled by that field's own peak magnitude."""
  worst = 0.0
  for name, truth in reference.items():
    scale = float(np.max(np.abs(truth))) or 1.0
    worst = max(worst, float(np.max(np.abs(candidate[name] - truth))) / scale)
  return worst
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_resolution.py -q`
Expected: PASS, 7 passed

- [ ] **Step 5: Commit**

```bash
git add pyimr/resolution.py tests/test_resolution.py
git commit -m "Per-field error metric scaled by each field's peak"
```

---

### Task 3: Self-validating reference

**Files:**
- Modify: `pyimr/resolution.py`
- Test: `tests/test_resolution.py`

**Interfaces:**
- Consumes: `_at`, `_solve`, `_deviation`.
- Produces: `_reference(config, times, fields, target, nt_ladder) -> tuple[dict[str, np.ndarray], float]`, raising `ValueError` when unconverged.

- [ ] **Step 1: Write the failing test**

```python
def test_the_reference_refuses_when_it_is_not_converged(measured):
  """The guard this module exists around. Everything downstream inherits the reference's
  error silently, and a reference sharing the error under test reports success -- that is
  how Nt = 5 was measured against an Nt = 5 reference and looked free at 2.07e-02.
  """
  from pyimr.resolution import _reference

  with pytest.raises(ValueError, match="reference is not converged"):
    # target/10 = 1e-30 is unreachable, so any real gap trips the guard
    _reference(_config(), _TIMES, ("radius_ratio",), 1e-29, (5, 7))
  measured("reference guard", "unreachable target/10 raises rather than returning")


def test_the_reference_returns_the_finer_solve_when_converged():
  from pyimr.resolution import _reference, _at, _solve

  reference, gap = _reference(_config(), _TIMES, ("radius_ratio",), 1e-2, (5, 9))
  assert gap <= 1e-3
  finer = _solve(_at(_config(), "spectral", 18, 1e-12, 1e-14), _TIMES, ("radius_ratio",))
  np.testing.assert_allclose(reference["radius_ratio"], finer["radius_ratio"], rtol=0, atol=1e-12)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_resolution.py -q -k reference`
Expected: FAIL, `ImportError: cannot import name '_reference'`

- [ ] **Step 3: Write minimal implementation**

```python
def _reference(config, times, fields, target, nt_ladder):
  """A converged solve, plus proof that it is converged.

  Solves at the ladder ceiling and again at doubled `Nt`. The finer one is the reference;
  their gap must be under `target / 10` or there is nothing to measure against.
  """
  ceiling = int(nt_ladder[-1])
  coarse = _solve(_at(config, "spectral", ceiling, _REFERENCE_RTOL, _REFERENCE_ATOL), times, fields)
  finer = _solve(_at(config, "spectral", 2 * ceiling, _CHECK_RTOL, _CHECK_ATOL), times, fields)

  gap = _deviation(coarse, finer)
  if gap > target / 10.0:
    raise ValueError(
      f"reference is not converged: Nt={ceiling} and Nt={2 * ceiling} differ by {gap:.2e}, "
      f"above target/10 = {target / 10.0:.2e}. Raise the Nt ladder or loosen target."
    )
  return finer, gap
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_resolution.py -q`
Expected: PASS, 9 passed

- [ ] **Step 5: Commit**

```bash
git add pyimr/resolution.py tests/test_resolution.py
git commit -m "Reference that proves its own convergence or refuses"
```

---

### Task 4: Stage one, search both discretizations

**Files:**
- Modify: `pyimr/resolution.py`
- Test: `tests/test_resolution.py`

**Interfaces:**
- Consumes: `_at`, `_solve`, `_timed`, `_deviation`.
- Produces: `_smallest_adequate(config, times, fields, target, reference, thermal, nt_ladder) -> int | None` (ladder index), `_choose_discretization(config, times, fields, target, reference, nt_ladder) -> tuple[str, int]` raising `ValueError` when unreachable.

- [ ] **Step 1: Write the failing test**

```python
def test_it_finds_the_smallest_adequate_node_count():
  """Bisection over the ladder relies on discretization error falling with Nt."""
  from pyimr.resolution import _reference, _smallest_adequate

  ladder = (5, 7, 9)
  reference, _ = _reference(_config(), _TIMES, ("radius_ratio",), 1e-2, ladder)
  index = _smallest_adequate(_config(), _TIMES, ("radius_ratio",), 1e-2, reference, "spectral", ladder)
  assert index is not None and 0 <= index < len(ladder)


def test_an_unreachable_target_is_reported_not_approximated(measured):
  """Returning the best of an inadequate set is the same failure as an unconverged
  reference: a number that looks like success.
  """
  from pyimr.resolution import _choose_discretization, _reference

  ladder = (5, 7)
  reference, _ = _reference(_config(), _TIMES, ("radius_ratio",), 1e-2, ladder)
  with pytest.raises(ValueError, match="no discretization in the ladder"):
    _choose_discretization(_config(), _TIMES, ("radius_ratio",), 1e-16, reference, ladder)
  measured("unreachable target", "raises rather than returning the best available")


def test_fd_is_genuinely_evaluated_not_just_offered():
  """The spectral preference in this package was measured at one configuration and at
  tight accuracy, neither of which holds in the usual regime, so fd has to be a real
  candidate. Asserting the winner is one of two names would pass without fd ever running.
  """
  from pyimr.resolution import _reference, _smallest_adequate

  ladder = (5, 9, 13)
  reference, _ = _reference(_config(), _TIMES, ("radius_ratio",), 1e-2, ladder)
  index = _smallest_adequate(_config(), _TIMES, ("radius_ratio",), 1e-2, reference, "fd", ladder)
  assert index is not None, "fd could not meet a loose target at any ladder entry"
  assert ladder[index] in ladder


def test_an_inadequate_grid_is_not_accepted(measured):
  """The concrete failure this module was built around: Nt = 5 over a multi-collapse
  record carries 2.07e-02 discretization error, as large as the experimental noise it was
  supposed to sit beneath, while looking free against a reference computed at Nt = 5.
  """
  from pyimr.resolution import _reference, _smallest_adequate

  ladder = (5, 9, 13)
  reference, _ = _reference(_config(), _TIMES, ("radius_ratio",), 1e-4, ladder)
  loose = _smallest_adequate(_config(), _TIMES, ("radius_ratio",), 1e-2, reference, "spectral", ladder)
  tight = _smallest_adequate(_config(), _TIMES, ("radius_ratio",), 1e-6, reference, "spectral", ladder)
  assert loose is not None and tight is not None
  assert tight >= loose, "a tighter target must never select a coarser grid"
  measured("grid selection", f"target 1e-2 -> Nt={ladder[loose]}, 1e-6 -> Nt={ladder[tight]}")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_resolution.py -q -k "smallest or unreachable or fd_is or inadequate"`
Expected: FAIL, `ImportError: cannot import name '_smallest_adequate'`

- [ ] **Step 3: Write minimal implementation**

```python
def _smallest_adequate(config, times, fields, target, reference, thermal, nt_ladder):
  """Index of the smallest ladder entry meeting `target`, or None. Bisection, since
  discretization error falls with `Nt` even though cost does not rise smoothly."""
  low, high = 0, len(nt_ladder) - 1

  def misses(index):
    candidate = _at(config, thermal, nt_ladder[index], _REFERENCE_RTOL, _REFERENCE_ATOL)
    return _deviation(_solve(candidate, times, fields), reference) > target

  if misses(high): return None
  while low < high:
    middle = (low + high) // 2
    if misses(middle): low = middle + 1
    else: high = middle
  return low


def _choose_discretization(config, times, fields, target, reference, nt_ladder):
  """The cheapest `(thermal, Nt)` meeting `target`, ranked by measured time."""
  best = None
  for thermal in ("spectral", "fd"):
    index = _smallest_adequate(config, times, fields, target, reference, thermal, nt_ladder)
    if index is None: continue
    nodes = nt_ladder[index]
    seconds, _ = _timed(_at(config, thermal, nodes, _REFERENCE_RTOL, _REFERENCE_ATOL), times, fields)
    if best is None or seconds < best[0]: best = (seconds, thermal, nodes)

  if best is None:
    raise ValueError(
      f"no discretization in the ladder reaches target={target:.2e}; "
      f"raise the Nt ladder or loosen target"
    )
  return best[1], best[2]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_resolution.py -q`
Expected: PASS, 13 passed

- [ ] **Step 5: Commit**

```bash
git add pyimr/resolution.py tests/test_resolution.py
git commit -m "Stage one: search spectral and fd for the cheapest adequate grid"
```

---

### Task 5: Stage two and the public entry point

**Files:**
- Modify: `pyimr/resolution.py`
- Test: `tests/test_resolution.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `_loosest_tolerance(config, times, fields, target, reference, thermal, nt, rtol_ladder) -> float` and the public `choose_resolution(config, times, target, *, field="radius_ratio", nt_ladder=NT_LADDER, rtol_ladder=RTOL_LADDER) -> Resolution`.

- [ ] **Step 1: Write the failing test**

```python
def test_it_returns_a_setting_that_meets_the_target(measured):
  from pyimr.resolution import choose_resolution

  setting = choose_resolution(_config(), _TIMES, 1e-2, nt_ladder=(5, 9), rtol_ladder=(1e-8, 1e-6, 1e-4))
  assert setting.achieved <= 1e-2
  assert setting.thermal in ("spectral", "fd") and setting.Nt in (5, 9)
  assert setting.atol == pytest.approx(setting.rtol * 1e-2)
  assert setting.seconds > 0.0
  measured("chosen setting", f"{setting.thermal} Nt={setting.Nt} rtol={setting.rtol:.0e} "
                             f"achieved {setting.achieved:.2e} in {setting.seconds * 1e3:.1f} ms")


def test_the_returned_setting_reproduces_its_reported_error():
  """A setting whose own config does not reproduce the achieved error is not a setting."""
  from pyimr.resolution import _reference, _solve, choose_resolution

  ladder = (5, 9)
  setting = choose_resolution(_config(), _TIMES, 1e-2, nt_ladder=ladder, rtol_ladder=(1e-8, 1e-6))
  reference, _ = _reference(_config(), _TIMES, ("radius_ratio",), 1e-2, ladder)
  from pyimr.resolution import _deviation

  again = _solve(setting.apply(_config()), _TIMES, ("radius_ratio",))
  assert _deviation(again, reference) == pytest.approx(setting.achieved, rel=1e-9)


def test_a_tighter_target_never_returns_a_looser_tolerance():
  from pyimr.resolution import choose_resolution

  loose = choose_resolution(_config(), _TIMES, 1e-2, nt_ladder=(5, 9), rtol_ladder=(1e-8, 1e-6, 1e-4))
  tight = choose_resolution(_config(), _TIMES, 1e-5, nt_ladder=(5, 9), rtol_ladder=(1e-8, 1e-6, 1e-4))
  assert tight.rtol <= loose.rtol


def test_multiple_fields_must_all_meet_the_target():
  """Internal pressure runs about 80x looser than radius at identical settings, so adding
  it can only make the requirement harder, never easier.
  """
  from pyimr.resolution import choose_resolution

  ladder, tolerances = (5, 9), (1e-8, 1e-6, 1e-4)
  radius = choose_resolution(_config(), _TIMES, 1e-3, field="radius_ratio",
                             nt_ladder=ladder, rtol_ladder=tolerances)
  both = choose_resolution(_config(), _TIMES, 1e-3, field=("radius_ratio", "internal_pressure_pa"),
                           nt_ladder=ladder, rtol_ladder=tolerances)
  assert both.rtol <= radius.rtol and both.Nt >= radius.Nt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_resolution.py -q -k "returns_a_setting or reproduces"`
Expected: FAIL, `ImportError: cannot import name 'choose_resolution'`

- [ ] **Step 3: Write minimal implementation**

First extend the module's public surface, now that the symbol exists:

```python
__all__ = ["NT_LADDER", "RTOL_LADDER", "Resolution", "choose_resolution"]
```

Then add:

```python
def _loosest_tolerance(config, times, fields, target, reference, thermal, nt, rtol_ladder):
  """The loosest ladder tolerance still meeting `target`. Error rises with `rtol`, so this
  keeps the last passing entry and stops at the first failure."""
  chosen = rtol_ladder[0]
  for rtol in rtol_ladder:
    candidate = _at(config, thermal, nt, rtol, rtol * 1e-2)
    if _deviation(_solve(candidate, times, fields), reference) > target: break
    chosen = rtol
  return chosen


def choose_resolution(config, times, target, *, field="radius_ratio",
                      nt_ladder=NT_LADDER, rtol_ladder=RTOL_LADDER):
  """Cheapest `(thermal, Nt, rtol)` whose error meets `target`, measured on this problem.

  `target` is a deviation relative to each field's own peak magnitude. `field` may be one
  name or several, and every one named must meet it. Costs roughly ten to twelve solves,
  which pays for itself across a sampling or sensitivity campaign and not otherwise.
  """
  target = float(target)
  if target <= 0.0: raise ValueError("target must be positive")
  if not config.bubtherm: raise ValueError("resolution calibration needs bubtherm=1; nothing to choose otherwise")

  fields = (field,) if isinstance(field, str) else tuple(field)
  reference, _ = _reference(config, times, fields, target, nt_ladder)
  thermal, nodes = _choose_discretization(config, times, fields, target, reference, nt_ladder)
  rtol = _loosest_tolerance(config, times, fields, target, reference, thermal, nodes, rtol_ladder)

  final = _at(config, thermal, nodes, rtol, rtol * 1e-2)
  seconds, values = _timed(final, times, fields)
  return Resolution(thermal, int(nodes), float(rtol), float(rtol * 1e-2),
                    _deviation(values, reference), seconds)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_resolution.py -q`
Expected: PASS, 17 passed

- [ ] **Step 5: Verify lint and types**

Run: `.venv/bin/ruff check pyimr tests && .venv/bin/python tools/pyright_baseline.py`
Expected: `All checks passed!` and `no new type errors`

- [ ] **Step 6: Commit**

```bash
git add pyimr/resolution.py tests/test_resolution.py
git commit -m "Stage two and choose_resolution entry point"
```

---

### Task 6: Documentation

**Files:**
- Modify: `README.md` (insert immediately before the line `## Model selection`)
- Modify: `CHANGELOG.md` (insert immediately after the line `## 0.3.0 — unreleased`)

**Interfaces:**
- Consumes: the public `choose_resolution` and `Resolution` from Task 5.
- Produces: nothing consumed by code.

- [ ] **Step 1: Add the README section**

Insert before `## Model selection`:

````markdown
### Choosing a resolution

`Nt` and tolerance requirements depend on record length, material stiffness and which
observable is fitted, so a setting that is adequate for one collapse can be badly wrong
over five. `pyimr.resolution` measures on your own problem rather than guessing.

```python
from pyimr.resolution import choose_resolution

setting = choose_resolution(config, times, target=1e-3, field="radius_ratio")
# Resolution(thermal='spectral', Nt=9, rtol=1e-06, atol=1e-08,
#            achieved=5.2e-04, seconds=0.091)

config = setting.apply(config)
```

It builds a reference and checks it is converged, searches both `spectral` and `fd` for
the cheapest grid meeting `target`, then loosens tolerance as far as that grid allows.
Roughly ten to twelve solves, which is worth paying before a sampling or sensitivity
campaign and not worth paying for a single run.

`target` is relative to each field's own peak magnitude, and `field` accepts several
names. That matters because observables do not converge together: at identical settings
relative error was 3.4e-07 for radius and 2.8e-05 for internal pressure.

It raises rather than guessing if the reference is not converged or the target is out of
reach, since a number built on either is indistinguishable from a real answer.
````

- [ ] **Step 2: Add the CHANGELOG entry**

Insert after `## 0.3.0 — unreleased`:

```markdown
### Resolution calibration

`pyimr.resolution.choose_resolution` picks `thermal`, `Nt` and tolerance from a requested
accuracy by measuring on the caller's problem. Requirements depend on record length,
material stiffness and the observable being fitted, so the settings that were adequate for
one collapse were 4x wrong over five, and the preference for spectral over fd had only
been measured at one configuration and at tight accuracy.
```

- [ ] **Step 3: Verify the examples in the docs are real**

Run: `.venv/bin/python -c "
import numpy as np, pyimr, sys
sys.path.insert(0, 'tests')
from _validation_support import NHKV, R0, REQ
from pyimr.resolution import choose_resolution
cfg = pyimr.SimulationConfig(R0, REQ, NHKV, bubtherm=1)
print(choose_resolution(cfg, np.linspace(0, 20e-6, 60), 1e-3, nt_ladder=(5, 9, 13), rtol_ladder=(1e-8, 1e-6)))
"`
Expected: a `Resolution(...)` line. Correct the README comment if the printed values differ in shape.

- [ ] **Step 4: Commit**

```bash
git add README.md CHANGELOG.md
git commit -m "Document resolution calibration"
```

---

## Verification

- [ ] `.venv/bin/pytest tests/ -q -m "not slow" -p no:randomly` passes
- [ ] `.venv/bin/ruff check pyimr tests examples` passes
- [ ] `.venv/bin/python tools/pyright_baseline.py` reports no new type errors
- [ ] `tests/test_resolution.py` runs in under 30 seconds on its own
