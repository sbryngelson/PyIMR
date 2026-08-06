# Open work

What is known to be unfinished, and why it matters. Ordered by what blocks what, not by
size. Entries are removed when done rather than ticked, so this file is always what is left.

## The question none of this has answered yet

**1. Nothing has been fitted to data.** Three models now exist to answer the model-form
error in #222 -- `qSLS` (one relaxation time), `qSLS2` (two), `qSLSthin` (one that moves with
shear rate) -- and not one has been scored against the 15 C record. The machinery was the
easy half. Until this is run, the honest statement about a second relaxation time is that it
is *available*, not that it helps.

`fit_candidate` and `candidate_log_evidence` are what it needs, and both exist. What remains
is to run them on the three models, sum the evidence over the modes the fit returns, and
report each answer with its `chi_squared` beside it.

Two cautions for whoever does it. The search is budget-sensitive: a differenced Jacobian
costs `p + 1` solves an iteration, and an under-converged fit reports a `chi_squared` *worse*
than the generating point's rather than announcing itself --- measured, 27.3 against 0.85 at
`starts=4, max_evaluations=80`, and 0.64 at `starts=6, max_evaluations=150`. And `g` and
`alpha` trade along a ridge, so recovered parameters move between runs while the cost barely
does. Rank on evidence, not on parameter agreement.

**2. `qSLS` loses 2.7 nats to the Occam cap.** Measured while building the cap: the
four-parameter model already has directions this data barely determines. Which ones is not
known --- an SVD of the Jacobian at the fit, or a profile likelihood, would say. It bears
directly on whether adding a fifth and sixth parameter can mean anything.

## Judgment calls someone should make deliberately

**3. `cap_at_prior` defaults to `False`.** The uncapped Laplace evidence pays a model for
parameters its data cannot see --- it favoured the two-mode model by 29.7 nats exactly where
its second arm did nothing. Nothing else in the package calls `laplace_log_evidence`, so
flipping the default breaks nothing and removes a live footgun. It was left off because the
plain expansion is the textbook one. Worth a deliberate decision rather than inheriting one.

**4. `yield_pa` bounds are analogy, not measurement.** "The shear modulus's decades extended
two below" is reasoning by resemblance. The `gent_jm`, `fung_b` and `alpha` bounds were each
set from data, and the difference is visible in the comments. Measure or narrow this one.

**5. `thin_time`'s low end weakly identifies `pl_n`, unverified.** With the crossover below
any shear rate the collapse reaches, the thinning factor is 1 whatever the index. The comment
claims the redundancy prior reports this; that was asserted, not checked. The equivalent
claim for `tau_ratio` *was* checked (8e-12 of Rmax at ratio 1). Do the same here.

## Testing

**6. Exact-reduction gates are blind to wrong-but-consistent physics.** Both new constitutive
models had a physics mutation survive their whole test file: for `TwoModeQuadraticZener` and
for `CarreauZener`, splitting or thinning only part of the memory equation passed every test.
The reason is structural --- the reduction limit is precisely where competing formulations
agree, so a gate built on it cannot separate them.

Standing rule worth adopting: every new constitutive branch needs at least one test pinning
its equation *away* from the reduction limit.
`test_the_whole_memory_equation_thins_and_not_merely_part_of_it` is the pattern --- assert the
state derivative against a hand-computed form at a fixed `(R, Rd, Z)`, calling `_stress`
directly.

**7. `test_every_model_builds_from_exactly_its_own_axes` does not test "exactly".** It checks
that `build` succeeds given those axes, so it passes for a builder that silently ignores one.
Either make it check, or rename it to what it does.

**8. No test requires a material to be documented.** `docs/materials.md` carries every
constitutive law, but only by habit: the packaging test checks that links resolve, nothing
more. Two models shipped undocumented before this was noticed. A test that every exported
material appears there would stop it recurring.

## Infrastructure

**9. Nothing tells a developer to install pyright.** CI type-checks on 3.12 only, and it is
in no local instruction, so a whole branch was written type-blind and the errors surfaced
only in CI. It is pinned in `pyproject.toml` (`pyright==1.1.411`); running
`tools/pyright_baseline.py` locally reproduces that gate exactly. Worth a line in the README
or a dev extra that installs it.

**10. Watch the fast-lane budget.** Against a ~5 min target, the recent additions cost about
25 s. `tests/test_fit_candidate.py` shows the way to keep that down: its search tests run
against a cheap analytic forward model with a known minimum and finish in 0.5 s, with a
single real-solver test behind `@pytest.mark.slow`.

## Widening the candidate set

Every constitutive law in the package is now a comparison candidate: 23 in `STANDARD_MODELS`,
4 in `EXTENDED_MODELS`. Nothing is implemented-but-unreachable any more, which was true of six
thinning laws and of `Ogden` until recently. Adding models from here means writing new
physics, not registration.

**11. `PronyZener` --- the general n-mode law.** The principled version of the enrichment in
`TwoModeQuadraticZener`: a relaxation *spectrum* rather than one or two times, which is what a
crosslinked biopolymer actually has, and it subsumes the two-mode law as its n=2 case.
Invasive: `De` and `LAM` are scalars throughout `_stress` and would become per-mode arrays,
with `SCALE_PATHS` reindexed. The highest-value new model, because the others are variations
on a theme it generalises.

**12. Stiffening elastics with relaxation.** Gent, Fung, Arruda-Boyce, Yeoh and Ogden exist
only as *instantaneous* elastics, so the set offers stiffening without memory, or memory with
quadratic stiffening, but never Gent-with-relaxation. Since the records are fitted by a Zener
and the stiffening exponent is identified, that gap is real. Each is a `Zener`-shaped branch
with a different `Z_e`: repetitive rather than hard.

**13. Diminishing returns, honestly.** More candidates only sharpen a comparison that is
already running. Until item 1 is done, a 28th model adds breadth to a ranking nobody has
computed.
