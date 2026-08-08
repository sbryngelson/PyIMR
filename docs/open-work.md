# Open work

What is known to be unfinished. Entries are removed when done rather than ticked, so this
file is always what is left. Settled results appear first only where they constrain what is
worth trying next.

## What is settled, and what it rules out

The `qSLS` residual is correlated at lag one (0.918) and concentrated in the first collapse.
Four explanations were tested and none accounts for it:

| tried | result |
|---|---|
| a second relaxation time (`qSLS2`) | lag-one 0.918, unchanged |
| a rate-dependent one (`qSLSthin`) | lag-one 0.875 |
| averaging misaligned trials | single trials give 0.910 |
| the bubble-dynamics operator | 0.877 to 0.948 across all six |

**The diagnostic was never specific.** Lag-one at `N = 201` measures how much of a residual
is *smooth*, not what made it smooth: white noise gives 0.037, one sine period 1.000, an
80%-smooth mixture 0.944. The trial deviations from the mean --- measurement variation, no
model fitted --- give 0.859. So 0.918 says the residual is about three-quarters smooth, and
most of that is already in the data. Model-form error is one candidate among several, which
is not the reading #222 was given.

**Evidence differences here are prior-dominated.** The likelihood depends on `g*alpha` alone
(derived in the writeup; an SVD at the fit gives the sloppiest direction as
`g^+1.00 alpha^-1.00`). Fitting `g` and `alpha` separately slides along that ridge into
whatever corner the prior box provides, and moving the box swung the operator ranking by 50
nats. The fits are sound --- `chi2/N` 0.75 to 1.0 --- but **parameter values are not
identified, and a model ranking must be checked against the box before it is believed.**

**What survives, in identified coordinates and across records** (`starts=10`): compressibility
is required, decisively on all three (+162 to +182, +74 to +79, +74 to +75). **Gilmore/Tait
beats `keller-miksis` on every record** (+17.1, +4.2, +1.2), and both enthalpy Keller--Miksis
forms beat it on two. `keller-miksis` is the pressure form every candidate in this package
assumes, and three separate operators beat it.

**A search budget, not a record, produced the earlier answer.** At `starts=6` the same
comparison put Gilmore/Tait fourth on two records and first on the third and called the
operator undetermined, and found no ordering at all at 33 C, which was read as a record
without discriminating power. At `starts=10` the 33 C fits reach `chi2/N = 0.42` where six
starts found 1.151: every operator had been stuck in the same poor basin, which in the output
is indistinguishable from every operator fitting equally well. A record that cannot decide and
a search that did not converge look alike, and only a better search tells them apart.

**Thermal, scored at last.** Bubble thermal is preferred at 15 C (+8.65) and 23 C (+2.64);
bubble+medium is preferred at 23 C (+8.02) and 33 C (+1.65) and strongly rejected at 15 C
(-131). The pattern does not replicate and is more consistent with residual convergence
trouble than with physics, so nothing is claimed from it yet. Lag-one stays 0.897 to 0.968
throughout, so thermal joins the constitutive, averaging and operator axes in leaving the
correlation untouched.

**Designing against three model axes is well posed, but not evenly.** Perturbing each axis
from one base and projecting off the material sensitivity span (`confounding.py`):

| axis | size | after refit | absorbed |
|---|---|---|---|
| dynamics (KM -> KM/Mie-G) | 8.89 | **1.98** | **77.7%** |
| constitutive (one -> two modes) | 14.42 | 8.14 | 43.5% |
| thermal (cold -> bubble+medium) | **15.99** | 8.46 | 47.1% |

in units of the record's noise. No pair is confounded --- the residual directions sit at 46,
71 and 38 degrees --- so all three are separable in principle and the design problem is well
posed.

Two things follow. **Thermal is the largest lever, not the smallest**: it moves the trace most
and survives refitting most, which contradicts the expectation that it was negligible here.
And **the operator is the most absorbed axis by far** --- 77.7%, leaving under two noise units
--- which is why its ranking slid along the g--alpha ridge and reversed with the prior box.
Most of an operator change can be mimicked by refitting the material.

Caveat: absorption is measured by a LINEAR projection at the base point, so real nonlinear
refitting could absorb more. These are upper bounds on what is detectable.

## Open

**0. No criterion here takes model error as an argument, and the ranking of designs depends on
it.** `sloppy_design.py` fits cold one-mode qSLS to a thermal truth and to a two-mode truth at
four geometries. Robust across both: the discrimination-optimal geometry is worse on BOTH
counts BOTH times (fit 1.8x/2.1x, `g*alpha` recovery 9.8x/12.2x). Not robust: the E-optimal
geometry loses on both counts under thermal error and is neutral under constitutive error, and
the 60 um point goes from best to 8.7x worse recovery. So the actionable statement is not
"optimal designs are worse" but "which design is safe depends on which physics is missing" --
and `lackoffit.py` says the model IS inadequate on every record without saying in what
direction. Designing robustly needs a criterion that takes the model-error hypothesis as an
input; none of ours does.

**0b. The lack-of-fit test is now the cheapest diagnostic here, and it is not wired in.**
`lackoffit.py` runs it standalone. It belongs beside `chi2/N` in `records.score`, because it
answers the question `chi2/N` is usually mistaken for.

**1. Change the default operator, or justify it.** `dynamics="keller-miksis"` is assumed by every candidate
and example, and is beaten on two independent records. A one-line change that would move
every fit and evidence in the study, so it is a decision rather than a task.

  Narrowed by `req_prior.py`: with `Req` fitted rather than pinned, Gilmore still beats the
  pressure form at 15 C by 5 to 11 nats at every prior width, but the 23 C and 33 C margins
  collapse to under a nat. So the case for changing the default now rests on one record.

**1b. Explain the Req offset, or measure it independently.** The fit wants `Req` 3-11% larger
than inferred, consistently across six operators, three records and two prior widths, and the
offset shrinks as the gel warms (11%, 8%, 3.4%). chi2/N improves markedly with it free. This
is either a systematic in how `Req` is inferred or physics the model is absorbing into a
geometry parameter, and nothing here distinguishes them.

**1c. Warm-start the thermal fits.** `thermal.py` does not converge on 15 C or 33 C at any of
three budgets tried, and 24 restarts fit WORSE than 10 on two cells -- the multistart is
landing in different basins rather than refining one. 23 C is bit-stable across a 2.4x budget
change and says bubble +2.6, bubble+medium +8.0. The fix is to start each thermal fit from the
converged cold optimum instead of sampling the box afresh; more restarts is demonstrably not
the remedy.

**2. A likelihood that does not assume independence.** Every `chi2/N` and `log Z` here
presumes white residuals; with `N_eff` near 8 of 201 the likelihood overstates its
information roughly 25-fold. The hierarchical covariance in `figures_trial.py` already
exists. Until this is redone the evidences are comparable to each other and should not be
quoted absolutely.

**3. A diagnostic that discriminates shapes.** Lag-one cannot say what made a residual
smooth. What would: where in the trace the discrepancy sits, and whether it lies in the span
of the parameter sensitivities (39.3%, already computed), of the operator differences
(computable from `dynamics.json`), or of neither. Only the last is evidence for missing
physics.

**4. Act on the three-axis design.** `design_three.py` certifies a measure over four
material parameters and all three model axes (gap 9.7e-10, four support points), with the
information averaged over a decade of `g*alpha` so the answer is not conditional on one
material --- the flaw that made `design_operator.py` rank the records backwards.

Per axis, which the determinant alone hides:

| axis | variance at the optimum | at the 15 C geometry | efficiency |
|---|---|---|---|
| dynamics | 0.006776 (binding) | 0.009912 | 68.4% |
| constitutive | 0.000545 | 0.02239 | 2.4% |
| thermal | 0.000440 | 0.01049 | 4.2% |

The experiments performed are near-best for the axis that is hardest to see and 24 to 41
times off for the two that are easiest. So a geometry change buys a great deal of
constitutive and thermal discrimination at little cost to the operator --- the support wants
`R_max` at 60 um with high stretch and at 1200 um with low stretch, against the 277 um and
stretch 7.1 performed.

Caveat: 21 of 64 designs failed to integrate at some stiffness and were dropped, which biases
the support toward regions that always integrate. Worth checking before anyone runs it.

**5. `cap_at_prior` defaults to `False`.** The uncapped Laplace evidence pays a model for
parameters its data cannot see --- 29.7 nats, in the case that mattered. Nothing else calls
`laplace_log_evidence`, so flipping the default breaks nothing and removes a live footgun.
Left off because the plain expansion is the textbook one; worth a deliberate decision.

**6. `yield_pa` bounds are analogy, not measurement.** "The modulus's decades extended two
below" is reasoning by resemblance; `gent_jm`, `fung_b` and `alpha` were each set from data.

**7. `thin_time`'s low end weakly identifies `pl_n`, unverified.** The comment claims the
redundancy prior reports it. The equivalent claim for `tau_ratio` was checked; this was not.

## Testing

**8. Exact-reduction gates are blind to wrong-but-consistent physics.** For both
`TwoModeQuadraticZener` and `CarreauZener`, a mutation changing only part of the memory
equation passed the whole file: the reduction limit is precisely where rival formulations
agree. Every new constitutive branch needs one test pinning its equation *away* from that
limit --- `test_the_whole_memory_equation_thins_and_not_merely_part_of_it` is the pattern.

**9. `test_every_model_builds_from_exactly_its_own_axes` does not test "exactly".** It passes
for a builder that silently ignores an axis. Make it check, or rename it.

**10. No test requires a material to be documented.** Two models shipped undocumented before
this was noticed; the packaging test only checks that links resolve.

## Infrastructure

**11. Nothing tells a developer to install pyright.** CI type-checks on 3.12 only and it is
in no local instruction, so a whole branch was written type-blind. It is pinned in
`pyproject.toml`; `tools/pyright_baseline.py` reproduces the gate exactly.

**12. Watch the fast-lane budget.** Against a ~5 min target the recent additions cost about
25 s. `tests/test_fit_candidate.py` shows how to keep it down: search tests against a cheap
analytic model with a known minimum, one real-solver test behind `@pytest.mark.slow`.

## New models

Every constitutive law in the package is now a comparison candidate --- 23 standard, 4
extended --- so adding one means new physics rather than registration.

**13. Stiffening elastics with relaxation.** Gent, Fung, Arruda--Boyce, Yeoh and Ogden exist
only as *instantaneous* elastics, so the set offers stiffening without memory, or memory with
quadratic stiffening, but never Gent-with-relaxation. Each is a `Zener`-shaped branch with a
different `Z_e`.

**14. `PronyZener`, with a caveat.** The general n-mode law, subsuming
`TwoModeQuadraticZener` at n=2. Invasive: `De` and `LAM` are scalars throughout `_stress` and
would become per-mode arrays. But it generalises exactly the direction that came back
negative, so it is a weaker prospect than its scope suggests.
