# Resolution calibration

Pick the thermal discretization and solver tolerance from a requested accuracy, by
measuring on the caller's own problem.

## Problem

Choosing `thermal`, `Nt` and `rtol` currently requires folklore, and the folklore is
wrong often enough to matter. Two examples from the work that motivated this, both mine:

- "Accuracy saturates at Nt = 7 to 9." True for a single 20 us collapse, where Nt = 9
  agreed with Nt = 60 to 1e-09. False over a five-collapse record, where Nt = 9 carries
  5e-03 error and Nt = 5 carries 2.07e-02 -- as large as the experimental noise it was
  meant to sit under.
- "Spectral is 4.6x faster than fd at matched accuracy." Measured at one configuration and
  at tight accuracy. Exponential convergence buys least when the target is loose, and
  spectral's N^4 stiffness costs most on long records, so neither condition holds in the
  regime this package is normally used in.

The requirement depends on record length, material stiffness and which observable is being
fitted. A lookup table cannot see any of those, so a table would encode today's traps and
ship them as authority.

The cases that make this worth solving are sampling campaigns of order 1000 solves and
sensitivity runs, where a per-solve factor is multiplied by the campaign. A one-off run
does not need this and should not use it.

## Approach

`choose_resolution(config, times, target, field=...)` builds a converged reference, walks a
ladder from cheap to expensive, and returns the cheapest setting whose error against that
reference meets `target`.

```python
from pyimr.resolution import choose_resolution

setting = choose_resolution(config, times, target=1e-3, field="radius_ratio")
# Resolution(thermal="spectral", Nt=9, rtol=1e-6, atol=1e-8,
#            achieved=5.2e-04, seconds=0.091)

cfg = setting.apply(config)
```

Cost is roughly 18 solves with the default ladders: 2 for the reference and its check, 5
for spectral and 5 for fd in stage 1, 5 for rtol in stage 2, plus 1 final solve. Amortized
against thousands.

## Search

Two stages, not a grid. The two error sources are close to independent: at Nt = 9 the error
was 5.09e-03 at rtol = 1e-8 and 5.24e-03 at rtol = 1e-6, so tolerance contributed almost
nothing while discretization dominated.

1. Tolerance held at the reference's (`rtol = 1e-10`, `atol = 1e-12`), so only
   discretization error is in play. For each of `thermal in ("spectral", "fd")`, bisect
   over `Nt` in `NT_LADDER = (5, 7, 9, 13, 17, 25, 33)` for the smallest meeting `target`.
   Keep whichever `(thermal, Nt)` measured faster.
2. That `(thermal, Nt)` held fixed. Step `rtol` up through
   `RTOL_LADDER = (1e-10, 1e-8, 1e-6, 1e-4, 1e-3)` with `atol = rtol * 1e-2`, and keep the
   loosest whose total error still meets `target`.

Both discretizations are searched because which one wins is exactly the question the
helper exists to answer empirically. If spectral does dominate, it will keep being chosen,
and the preference becomes a measurement rather than an inheritance.

`Mt` is tied to `Nt` when `medtherm = 1`. Searching it independently would make the ladder
three-dimensional and undo the saving the two-stage split buys. Revisit only if medium
resolution is shown to bind separately.

Cheapest is decided by **measured wall time**, never an analytic cost proxy. fd needs more
nodes than spectral and can cross `_CHORD_ABOVE = 80`, where the root finder changes, so
the cost curve has a discontinuity and is not reliably monotone in `Nt`.

## Error

`field` defaults to `radius_ratio` and accepts a tuple; `target` must hold for every field
named. Error is the maximum deviation from the reference relative to that field's own peak
magnitude.

Per-field matters because observables do not converge together. At identical settings,
relative error was 3.4e-07 for radius and 2.8e-05 for internal pressure -- about 80x
looser. Calibrating for a pressure fit must give a different answer than for a radius fit.

## Reference

Solve at `NT_LADDER[-1]` with `rtol = 1e-10, atol = 1e-12`, then again at `Nt` doubled and
`rtol = 1e-12`, and compare on every requested field. If they disagree by more than
`target / 10`, raise rather than return.

This guard is the point, not a nicety. Everything downstream inherits the reference's error
silently, and a reference that shares the error being measured reports success. That
failure occurred twice during the work leading to this spec: once comparing Nt = 5 against
a reference computed at Nt = 5, which cancelled the discretization error under test and
made an unusable setting look free.

If stage 1 cannot meet `target` at any `Nt` within range, raise and say so. Returning the
best of an inadequate set is the same failure in a different place.

## Out of scope

No disk caching. No automatic application to the config. No re-calibration when parameters
drift -- a campaign whose material class changes enough to matter should re-calibrate, and
that is a caller's judgement rather than a heuristic worth guessing at.

## Testing

- A problem with a known-adequate setting is found, and a known-inadequate one is
  rejected. Nt = 5 on a five-collapse record is a ready-made negative case at 2.07e-02.
- An unconverged reference raises rather than returning.
- An unreachable target raises rather than returning the best available.
- Per-field targets select different settings for radius against internal pressure.
- `Resolution.apply` produces a config that reproduces the achieved error.
- fd is chosen when it is genuinely cheaper at the target, confirming the search is not
  hard-wired to spectral.

## Risks

The reference cost grows with the `Nt` ceiling, and on long multi-collapse records that is
the most expensive solve in the procedure. If calibration cost becomes comparable to the
campaign it serves, the helper is not worth using and should say so rather than run.

Calibration is only valid for the config it measured. Nothing enforces that, and a caller
who reuses a `Resolution` across a materially different problem gets a silently wrong
answer. Documented, not prevented.
