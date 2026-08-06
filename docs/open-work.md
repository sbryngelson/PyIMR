# Open work

What is known to be unfinished, and why it matters. Ordered by what blocks what, not by
size. Entries are removed when done rather than ticked, so this file is always what is left.

## Answered: a richer relaxation is not the missing physics

Run on the 15 C record (`docs/writeup/enrichment.py`), both enrichments are preferred and
neither margin survives inspection:

| model | p | chi2/N | log Z | vs qSLS | lag-1 | N_eff |
|---|---|---|---|---|---|---|
| qSLS | 4 | 0.962 | 529.26 | --- | 0.918 | 10.3 |
| qSLS2 | 6 | 0.950 | 531.03 | +1.78 | 0.918 | 7.2 |
| qSLSthin | 6 | 0.922 | 532.05 | +2.79 | 0.875 | 8.2 |

The autocorrelation is the defect they were reached for and it does not move. Both `log Z`
and `chi2/N` presume independent residuals, so with `N_eff` near 8 of 201 the likelihood
overstates its information by roughly 25x; deflating a 2.79-nat margin by anything of that
order leaves nothing. The second arm is not idle --- 16% of the memory at 15.7x the first
timescale --- it simply does not address what is wrong. `qSLSthin` pins `thin_time` at the
bottom of its range, so that margin is bound-limited too.

The one-mode `chi2/N` of 0.962 reproduces `fit_quality.py` exactly through a different
optimiser in different coordinates, which is what makes the comparison trustworthy.

**1. Ruled out: averaging is not the cause.** The hypothesis was that fitting the MEAN of 18
trials whose collapses are misaligned --- their minima spread over a third of the record, and
the mean is 1.6x the noise shallower at the collapse than any trial --- produces a residual
concentrated at the collapse, smooth, correlated, and immune to constitutive change. It fits
every symptom, and it is wrong.

Fitting the same qSLS to each trial separately (`docs/writeup/per_trial.py`) leaves the
correlation where it was:

| target | chi2/N | lag-1 | N_eff |
|---|---|---|---|
| mean trace | 0.962 | 0.918 | 10.3 |
| single trials, median | 1.414 | 0.910 | 18.7 |
| single trials, range | 0.764--11.879 | 0.806--0.940 | |

Lag-one falls by 0.008. The correlation is a property of the fit to an individual bubble, not
an artefact of averaging over several. (The chi2/N spread is expected and not informative
here: every trial is fitted at the mean's `R_MAX` and stretch, since per-trial values are not
in the record, so the worst fits are paying for geometry rather than physics. Lag-one against
lag-one is the comparison this test supports.)

So: two constitutive enrichments do not fix it, and neither does the averaging. What has
never been varied is below.

**2. The bubble-dynamics model, which nobody has varied --- now the leading suspect.** Every candidate in the package
fixes `radial=2` (Keller--Miksis with Tait). Holding the material at the qSLS fit and
changing only that: `radial=3` moves the trace 0.104, `radial=4` 0.210, `radial=5` 0.078,
`radial=6` 0.262 --- against a median noise of 0.018, so **4 to 14 times the noise**. This is a
larger unexamined freedom than every constitutive difference tested this session put
together, and it is a model-selection axis in exactly the sense `pyimr.selection` means.

**3.** The residual is concentrated in the first collapse, where
sphericity and the far-field boundary are least secure. That is the next hypothesis, and it
is not a constitutive one --- which is worth saying plainly, because every model added this
session was constitutive.

A cheaper check first: fit with a likelihood that does not assume independence. The
hierarchical covariance in `figures_trial.py` already exists, and every number in the table
above is computed under an assumption the residual is known to violate. Until that is
redone, all three evidences are on the same wrong footing --- comparable to each other, but
not to be quoted absolutely.

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

**13. Diminishing returns, honestly.** The comparison has now been run, and it says a richer
relaxation spectrum is not what this record is missing. That is an argument for spending the
next effort on the likelihood (item 1) rather than on a 28th constitutive law --- and it is
the reason `PronyZener` above, which generalises exactly the direction that just came back
negative, is less attractive than its position on this list suggests.
