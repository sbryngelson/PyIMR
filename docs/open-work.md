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

**What survives, in identified coordinates and across records:** compressibility is required
(Keller--Miksis over Rayleigh--Plesset, +162.5 and +74.4 on the two records with power);
`radial=5` beats the `radial=2` every candidate assumes, on two records (+4.5, +1.5);
`radial=3` beats it by more but on one record only.

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

**1. Change the default operator, or justify it.** `radial=2` is assumed by every candidate
and example, and is beaten on two independent records. A one-line change that would move
every fit and evidence in the study, so it is a decision rather than a task.

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

**4. Design conditional on the material.** `design_operator.py` certifies a measure
(gap 1.0e-9, seven support points) but holds the material at the 15 C fit, so it ranks
geometries only --- and its ranking inverts the empirical one. Operator information is
governed by collapse depth, which the material sets: at their own fits the operators separate
by 4.17, 3.06 and 0.00005 noise units on the three records, the last because that fit barely
collapses. Recompute over materials as well as geometries, or maximise the `eps` information
directly.

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
