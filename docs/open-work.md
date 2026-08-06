# Open work

What is known to be unfinished, and why it matters. Ordered by what blocks what, not by
size. Entries are removed when done rather than ticked, so this file is always what is
left.

## The question none of this has answered yet

**1. Nothing has been fitted to data.** Three models now exist to answer the model-form
error in #222 -- `qSLS` (one relaxation time), `qSLS2` (two), `qSLSthin` (one that moves
with shear rate) -- and not one of them has been scored against the 15 C record. The
machinery was the easy half. Until this is run, the honest statement about the second
relaxation time is that it is *available*, not that it helps.

Blocked on item 2.

**2. There is no way to fit a `CandidateModel`.** `candidate_log_evidence` needs a fitted
point in unit coordinates and nothing produces one: `PreparedInference.fit_multistart`
works from `InferenceParameter` paths, and a candidate's axes need not be material fields
at all (`tau_ratio` is a ratio of two, `thin_time` is absolute, `lam` is a ratio). So every
`EXTENDED_MODELS` member can be scored only at a point supplied by hand.

A `fit_candidate(candidate, solve, observed, deviation, bounds, starts)` closes it and
unblocks item 1. It is the single highest-value thing on this list.

**3. `qSLS` loses 2.7 nats to the Occam cap.** Measured while building the cap: the
four-parameter model already has directions this data barely determines. Which ones is not
known -- an SVD of the Jacobian at the fit, or a profile likelihood, would say. It bears
directly on whether adding a fifth and sixth parameter can mean anything.

## Judgment calls someone should make deliberately

**4. `cap_at_prior` defaults to `False`.** The uncapped Laplace evidence pays a model for
parameters its data cannot see -- it favoured the two-mode model by 29.7 nats exactly where
its second arm did nothing. Nothing else in the package calls `laplace_log_evidence`, so
flipping the default breaks nothing and removes a live footgun. It was left off because the
plain expansion is the textbook one. Worth a deliberate decision rather than inheriting mine.

**5. `yield_pa` bounds are analogy, not measurement.** "The shear modulus's decades extended
two below" is reasoning by resemblance. The `gent_jm`, `fung_b` and `alpha` bounds were each
set from data, and the difference is visible in the comments. This one should be measured or
narrowed.

**6. `thin_time`'s low end weakly identifies `pl_n`, unverified.** With the crossover below
any shear rate the collapse reaches, the thinning factor is 1 whatever the index. The
comment claims the redundancy prior reports this; that was asserted, not checked. The
equivalent claim for `tau_ratio` *was* checked (8e-12 of Rmax at ratio 1). Do the same here.

## Testing

**7. Exact-reduction gates are blind to wrong-but-consistent physics.** Both new
constitutive models had a physics mutation survive the whole file: for `TwoModeQuadraticZener`
and for `CarreauZener`, thinning or splitting only part of the memory equation passed every
test. The reason is structural -- the reduction limit is precisely where competing
formulations agree, so a gate built on it cannot separate them.

Standing rule worth adopting: every new constitutive branch needs at least one test that
pins its equation *away* from the reduction limit. `test_the_whole_memory_equation_thins_and_not_merely_part_of_it`
is the pattern -- assert the derivative against a hand-computed form at a fixed state.

**8. `test_every_model_builds_from_exactly_its_own_axes` does not test "exactly".** It
checks that `build` succeeds given those axes, so it passes for a builder that silently
ignores one. Either make it check, or rename it to what it does.

## Documentation, now behind the code

**9. `docs/materials.md` has neither new model.** `TwoModeQuadraticZener` and `CarreauZener`
are undocumented; every other constitutive law is there. Nothing enforces this -- the
packaging test only checks that links resolve -- so a test that every exported material
appears would stop it recurring.

**10. `docs/writeup/selection.tex` predates all of it.** It says nothing about the six
shear-thinning candidates, the two Zener variants, the Occam cap, or the grid-free evidence
path. The fit-quality and trial-spread sections are still correct; the model set is not.

## Infrastructure

**11. Nothing tells a developer to install pyright.** CI type-checks on 3.12 only, and it is
not in any local instruction, so a whole branch was written type-blind and the errors only
surfaced in CI. It is pinned in `pyproject.toml` (`pyright==1.1.411`); running
`tools/pyright_baseline.py` locally reproduces the gate exactly. Worth a line in the README
or a dev extra that installs it.

**12. `PronyZener` -- the general n-mode law.** The remaining model from the original ask,
and the invasive one: `De` and `LAM` are scalars throughout `_stress` and would have to
become per-mode arrays, with `SCALE_PATHS` reindexed. `TwoModeQuadraticZener` is the n=2
special case and should fall out of it.

**13. Watch the fast-lane budget.** The new files add roughly 25 s to the not-slow lane
against a ~5 min target. The coverage sweeps are already behind `@pytest.mark.slow`; the
next thing added at this size should probably start there.
