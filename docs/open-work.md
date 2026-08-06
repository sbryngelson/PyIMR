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

**2. The operator cannot be selected from this record, and the reason matters more than the
answer.** With `qSLS` and the package's own `PARAMETER_BOUNDS`, the ranking put
Keller-Miksis/Mie-Grueneisen first (+4.02 nats) and Gilmore/Tait at -27.66. Three of six fits
were sitting exactly on `g = 1e2` or `lambda1 = 1e-7` --- their own floors --- so those were
bounds, not fits. Widening the box to `g (1e0, 1e6)`, `lambda1 (1e-9, 1e-2)`,
`alpha (1e-4, 1e3)` reverses it: **Gilmore/Tait becomes best at +21.93** and KM/Mie-G falls to
+7.32. A 50-nat swing from moving the prior.

Widening did not remove the pinning, it moved it. Five of six now sit on `alpha = 1e3`, the
new ceiling, with `g` collapsed to 1.1--3.5 Pa. That is the g--alpha ridge running to whatever
corner the box provides: the quadratic law's stiff response goes roughly as `g * alpha`, so
`g -> 0` with `alpha -> inf` costs nothing, and the reported shear modulus is set by where
the prior cuts the ridge rather than by the data. Across operators it spans 1.1 to 5019 Pa.

The control confirms it. Repeating the comparison with `NHKV`, which has no ridge, gives
`chi2/N = 13.2` for every operator and a total spread of 7.8 nats with Rayleigh--Plesset
nominally best --- an inadequate material makes every operator equally wrong, and the ranking
is noise. So the comparison is degenerate with `qSLS` and uninformative without it.

**What this costs elsewhere.** The enrichment comparison in item 1 was run in the narrow box
against a `qSLS` baseline whose `g` was pinned at 100 Pa. Its +1.78 and +2.79 nats are
subject to the same objection. The fits themselves are sound --- `chi2/N` between 0.75 and
1.0, so the models do reproduce the record --- but on this data **evidence differences between
these models are prior-dominated, and parameter values are not identified.**

**Which combination is identified was already known.** The writeup derives it analytically:
the likelihood depends on the product `g*alpha` alone, and `fig_valley` shows why the study's
estimates of `g` disagree by an order of magnitude. An SVD of the Jacobian at the fit
confirms it numerically -- the sloppiest right singular vector is `g^+1.00 alpha^-1.00` to two
decimals, with condition number 1.07e3 and a smallest singular value whose posterior width is
2.07 times the prior, so the cap of item 3 bites exactly there. That is a check on the
existing result, not a new one.

**What IS new is the consequence for operator selection.** Nobody had varied the operator, so
nobody had found that a known parameter degeneracy makes the operator comparison
prior-dominated. The two questions were previously independent; they are not.

**Done, on all three records** (`docs/writeup/identified.py`). Fitting `g*alpha` with
`g/alpha` fixed, at three ratios spanning two decades, on 15 C, 23 C and 33 C. No fit is
pinned (0 of 54). What survives is much less than the first pass suggested.

Discriminating power varies enormously between records, and that has to be handled before
anything else is read:

| record | trials | best chi2/N | spread across the six operators |
|---|---|---|---|
| 15 C | 18 | 0.90 | ~170 nats |
| 23 C | 14 | 1.01 | ~77 nats |
| 33 C | 7 | 1.15 | **~0.5 nats -- decides nothing** |

At 33 C every operator returns `log Z ~ 476.9` and `chi2/N = 1.15`. A test that only asks
whether an ordering's sign held would let that record veto results from records that can
actually see something; orderings there are noise, not contradictions. So a claim counts as
supported when some record decides it by more than a nat and none contradicts it, and the
records that cannot tell abstain.

**The claim about this package's default, corrected.** `radial=2` is what every candidate and
example assumes.

  - `radial=5` (Keller-Miksis / Mie-Grueneisen) beats it on **two independent records**
    (15 C +4.5, 23 C +1.5). This is the one that replicates.
  - `radial=3` (Keller-Miksis enthalpy/Tait) beats it by more, +5.8, but **on one record
    only**; 23 C and 33 C cannot decide it.

An earlier pass here highlighted `radial=3` because its margin was largest -- on the only
record then examined. That is selecting on the strongest record, and the correction is the
reason the other two were run.

Also replicated: every compressible operator beats Rayleigh-Plesset where any record can tell
(15 C +162.5, 23 C +74.4), and the Keller-Miksis family beats Gilmore/Mie-Grueneisen on both.

**Next: change the default, or justify it.** `radial=2` is assumed by every example and every
candidate; on this record it is beaten robustly by two other operators. Either is a one-line
change, and the fits and evidences of the whole study would move with it.

**3. The diagnostic was never specific.** Lag-one at N = 201 measures how much of a residual
is SMOOTH, not what made it smooth. Sampled at this density: white noise 0.037, one sine
period 1.000, a smooth random curve 0.999, a mixture 80% smooth 0.944. And the trial
deviations from the mean --- measurement variation, no model fitted --- carry lag-one 0.859
(median, range 0.725-0.890).

So 0.918 says the fit residual is about three-quarters smooth structure, and roughly that
much smoothness is already in the data. Whitening by the hierarchical covariance removes the
part lying along the parameter sensitivities, which is why it removed a third; the rest is
smooth for reasons those sensitivities cannot represent. Model-form error is one candidate
among several, not the established remainder --- which is the reading #222 was given and the
reason three enrichments were built against it.

**4. What would actually settle it.** A diagnostic that distinguishes SHAPES of smooth
residual rather than measuring smoothness: where in the trace the discrepancy sits, and
whether it lies in the span of the sensitivities (already computed --- 39.3%), of the
operator differences (computable from the table above), or of neither. The last is the only
one that is evidence for missing physics.

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

## Designing an experiment that could settle the operator

Run with `pyimr.measure`, treating the operator as a parameter: with
`R(theta, eps) = (1-eps) R_A + eps R_B`, the sensitivity to `eps` is `R_B - R_A`, so
discrimination is one more Jacobian column and D-optimality on the augmented set applies
with the certificate unchanged. `docs/writeup/design_operator.py`.

The measure certifies (gap 1.0e-9) on seven support points spanning `R_max` 50 to 1200 um,
and the performed geometries sit at 4 to 38 percent efficiency against it. **But that ranking
is about geometry only, and it inverts the empirical one**, which is the useful part.

At each record's own fitted material the two operators differ by 4.17 noise units at 15 C,
3.06 at 23 C, and 0.00005 at 33 C. The 33 C fit barely collapses --- `R/Rmax = 0.146` against
`0.068` at 15 C --- and a wall that slow never makes compressibility matter. So the quantity
that governs operator information is collapse DEPTH, which the material sets, and a design
measure computed at a single material cannot see it.

**Next here:** recompute the measure over a grid of materials as well as geometries, or
maximise the `eps` information directly rather than the augmented determinant. Either makes
the design answer conditional on the material, which the physics says it must be.

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
