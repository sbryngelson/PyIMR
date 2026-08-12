# What is wrong with `selection.tex`, and what was tried

A working record of claims in the document that measurement has contradicted, and of the fixes
that were attempted and failed. Each entry states the evidence, what would correct it, and —
for the failures — why the idea did not work and the conditions under which it might.

Kept because a refuted approach is expensive to re-derive and cheap to record. Four of the
attempts below looked obviously right beforehand.

---

## Part 1 — Claims in the document that are wrong

### 1.1 The design chapter assumes a constant noise scale the records contradict

**Severity: high.** Every Fisher matrix, expected gain and recommended batch in `sec:design`
rests on it.

`RELATIVE_NOISE = 0.018` appears in `design_operator.py`, `design_three.py`, `identify.py`,
`figures_measure.py` and `figures_geometry.py`. The measured per-sample spread varies within a
single trace by:

| record | min σ | max σ | ratio | median |
|---|---|---|---|---|
| 15 °C | 0.00056 | 0.0696 | 124× | 0.0182 |
| 23 °C | 0.00030 | 0.2682 | 882× | 0.0231 |
| 33 °C | 0.00025 | 0.3228 | 1316× | 0.0218 |

Fitting has always used the per-sample spread, so this is a gap *between* the two halves of the
document rather than an error inside either. At the same Jacobian the constant overstates
`log det F` by 1.3–2.2 nats and understates per-parameter σ by up to 2×.

**Correction.** Use the measured σ(t). For a geometry nobody has run, transfer a measured
profile by phase (`t/t_c`) — see §3.2 for why that step is the weakest link here.

Scripts: `noise_profile.py`.

### 1.2 "The portfolio spreads on its own, so the constraint never binds"

**Severity: high.** This sentence is false under a noise model derived from the very records the
document fits.

`sec:design` states that `eq:constrained` never binds because the portfolio optimum spreads
over six geometries unaided, and calls it "the only aggressive option that stays testable."
Recomputed under each measured profile:

| noise model | settings | lack-of-fit df |
|---|---|---|
| constant | 6 | +2 |
| 15 °C | 4 | **0** |
| 23 °C | 4 | **0** |
| 33 °C | 6 | +2 |

On two of three profiles the batch has *no* degrees of freedom for lack of fit — the test has
no power at all.

**Correction, and it is cheap.** Impose the floor with `constrained_measure` (#232). Holding
the measure to five settings restores df = +1 for **0.010 and 0.084 nats of 36.5** — 0.0% to
0.2% of the objective. Six settings and df = +2 costs at most 0.237. Every constrained measure
is certified under its floor.

So the recommendation does not need rethinking; one sentence needs replacing and one constraint
needs imposing.

Scripts: `noise_portfolio.py`, `noise_testable.py`.

### 1.3 The recommended geometry itself moves

**Severity: medium.** Depended on §3.2's transfer assumption, which has since been tested and
holds — under the smoothed transfer all three records agree on 416 µm @ 10.85 and none agrees
with the constant, so the "two of three" caveat below is an artifact of interpolating a raw
estimate.

| noise model | best U | R_max | stretch | regret of the constant's pick |
|---|---|---|---|---|
| constant | 9.028 | 514.2 µm | 6.92 | — |
| 15 °C | 9.198 | 514.2 µm | 6.92 | 0.000 |
| 23 °C | 8.891 | **416.0 µm** | **10.85** | **1.344 nats** |
| 33 °C | 8.627 | **416.0 µm** | **10.85** | 0.502 nats |

Up to 15% of the available gain. The two profiled models agree with *each other* and not with
the constant, so the flat scale is the outlier rather than one eccentric record — but this is
"two of three", not unanimous.

Scripts: `noise_design.py`.

### 1.4 The likelihood leans on samples that are 1 by construction

**Severity: medium, and unresolved.**

Each trace is divided by its own maximum, so at the sample where it attains that maximum every
trial equals 1 exactly and the spread there measures nothing. That sample is index 0–1 on all
three records, 30–90× below the median. Weighted by 1/σ², **the first sample alone carries 28%,
48% and 63%** of the likelihood; the first five carry 81–91%.

Measured consequences:

- **Parameters are robust.** Flooring σ at its 10th or 25th percentile moves the identified
  coordinates by ≤22%, usually under 5%. The published fits do not rest on this.
- **Statistics are not.** The same floor moves χ²/N from 1.029 to 0.401 at 23 °C — a factor of
  2.6. Every goodness-of-fit statement, lack-of-fit ratio and evidence inherits a weighting
  choice nobody has been making explicitly.
- **Dropping rather than flooring is the wrong fix.** It moves parameters by 12–46×, because
  those samples carry the initial condition and phase. They are load-bearing, not spurious.

**Correction.** A stated floor policy, and a re-run of every statistic that depends on it. Not
yet done.

Scripts: `pinned_samples.py`.

### 1.5 Fixed already

- **`results.json` stored a grid argmax, not a fit** (#235). At 15 °C the qSLS grid argmax gives
  χ²/N = 1.283 against the refit's 0.587 — a factor of 2.2. Regenerated with `best_grid_point`
  and `fit`. This also moved `trial_modes.py`'s dominant-mode R² at 15 °C from 0.43 to 0.536,
  making a docstring claim that switching fits "moves it by 0.02" wrong.
- **`enrichment_screen.json`'s `joint` was stale** in exactly the field #242's cone fix rewrote:
  71.5% against the correct 15.1% at 23 °C, with every neighbouring value bit-identical.
- **`enrichment_overlap`'s docstring** carried projections (39%, 58%, 24%) from a version of the
  screen fixed twice since.

---

## Part 2 — Ideas that were tried and do not work

### 2.1 Integrate `log R` instead of `R`

**Idea.** Near a deep collapse the solver chases an absolute tolerance toward zero: `atol` on
`R` is fixed while `R` falls by decades, so the controller is asked for ever more *relative*
precision. Integrating `u = log R` makes a relative tolerance stay relative.

**Result: it buys exactly nothing.** Implemented end to end and verified to reproduce the
existing formulation to 2.6×10⁻¹¹ in radius. On the band it was built for, 5/7 dead either way,
and the wall is in *precisely* the same place — last surviving end time 2.395777×10⁻⁵ s, minimum
radius 0.007896, identical to the last digit.

**Why it fails.** The reasoning is true and not what binds. Changing the *radial coordinate*
does not change how fast things happen *in time*, and the step size here is set by temporal
resolution. Confirmed independently: at loose tolerance (rtol 1e-4) the solver accepts 399,963
steps and rejects 37 — it marches happily and still exhausts the budget. That is a genuine
resolution requirement, not an error-control artifact.

**How it could work.** Paired with a **time reparametrisation** — a Sundman transformation,
`dt/ds = R^p` — so that steps are uniform in `s` through the collapse. `log R` rescales the
state; the clock is what needs rescaling. This is the standard pairing for collapse problems,
and `log R` alone is half of it.

Closed as #247.

### 2.2 Integrate `W = Z1 − Ze`, the deviation from quasi-equilibrium

**Idea.** The memory state is driven toward an elastic target that diverges as `R⁻⁹`.
Integrating the deviation removes the large cancellation. There is an existing design for this
(`2026-08-04-qsls-conditioning-design.md`, Phase B).

**Result: contraindicated here.** `De = λ₁/t₀ = 1.114`. At the stall `Z1 = −0.10` while
`Ze = 8650`, so `W = Z1 − Ze ≈ −8650` — the substituted state would be **86,000× larger** than
the one it replaces, with a correspondingly coarser tolerance scale.

**Why it fails.** The substitution assumes `De ≪ 1`, where `Z1` tracks `Ze` and the difference
is small and smooth. At `De ≈ 1` the arm does not track its target at all. `Z1` is the
well-behaved quantity in this problem; `Ze` is the one that explodes.

**How it could work.** Exactly as designed, for the qSLS conditioning case it was written for,
where `De` is small. The idea is sound; it was applied to the wrong regime.

### 2.3 A noise model derived from the normalisation

**Idea.** `y = R/R_max` is a ratio of two measured lengths. With independent error `s` on each,
`δy = (δR − y·δR_max)/R_max`, so `var(y) = (s/R_max)²(1 + y²)` — affine in `1 + y²`, slope set
by measurement error alone, and structurally **zero** at the peak where numerator and
denominator are the same measurement. A closed form is exactly what the design work needs,
since a per-sample estimate cannot be evaluated at an unrun geometry.

**Result: the sign is wrong.** Away from the peak, `corr(σ², 1 + y²)` is **−0.269, −0.088,
−0.052**. The phase medians say it plainly: 0.008–0.013 before the collapse against 0.022–0.036
at it. The spread is *largest* where `y` is *smallest*.

**Why it fails.** Independent length errors are not what dominates this spread. The profile is
dynamical — bubbles differ most where the dynamics amplify the difference, which is the collapse
and the rebound after it.

**What survives.** The cancellation at the peak is real and is the correct explanation of §1.4.
It is local, not a model of the whole profile.

**How it could work.** In a regime where instrument error dominates dynamical variation — a
quieter apparatus, or a gentler collapse. Not these records.

### 2.4 `R_max` normalisation error as the dominant trial mode

**Idea.** #222's own next step. If a trial's `R_max` divisor is off by `1 + ε`, every sample of
that trace scales by `1 − ε`, so the deviation runs along the trace itself. The existing `Rmax`
sensitivity column could not have tested this — `trace(rmax=…)` returns `radius_ratio`, already
normalised by the rescaled maximum, so it divides the effect back out.

**Result: not it.** Added as a ninth column, the dominant mode correlates with it at **0.152,
0.166, 0.100** — among the weakest of the nine.

**Why it fails.** The mode is not a scaling of the trace. Measured model-free, it lives in the
**rebound** (98.0% and 97.2% of its squared length after the collapse window at 23 and 33 °C)
and it is **not one curve** across records (|cos| 0.075, 0.252, 0.082 against 0.050 for
unrelated curves), so it is not an apparatus signature either.

**How it could work.** Not as the dominant explanation. It may be a minor component.

### 2.5 `B(d)` as a design screen

**Idea.** From T. Chen's surrogate-BOED note, where `B(d) = sup_θ KL[L_d(·|θ) ‖ m_d]` is the
constant in an EIG error bound. Since `U(d)` is the prior average of the same quantity, `B`
bounds it — and a candidate whose ceiling falls below the best design's average can be discarded
with no assumption about θ. For a Gaussian likelihood with a linearised model it is `p×p`
algebra on the Jacobian the design work already computes.

**Result: valid, and weak.** Holds on all 227 candidates. Eliminates **7 of 227**.

**Why it fails as a screen.** `U` spans 3.83 → 9.03 across the whole design space (2.4×) while
the ceiling carries 1.7 of slack. Slack comparable to the spread leaves nothing to eliminate.
And `corr(B/U, U) = −0.874`, so the bound is tightest on the designs that need it least.

**How it could work.** On a design space whose candidates differ more, or where the criterion is
expensive and the ceiling cheap. Here they cost the same Jacobian.

**What it is still good for.** `B/U ∈ [1.51, 2.21]` says no design's value is concentrated at
the edge of the prior — the average is a fair summary everywhere, which is reassuring for
batches chosen on `U`. And a ceiling below the quantity it bounds would expose an error in
either, so it earns its place as a check.

### 2.6 Solver-side fixes for the divergence band

All measured against the same case and all negative:

| attempted | range | effect |
|---|---|---|
| `max_steps` | 100k → 128M | none; wall time linear throughout, every budget exhausted |
| `rtol` | 1e-6 → 1e-10 | none |
| implicit solver | Kvaerno3/5, Chord *and* Newton | same rejection cascade |
| controller tuning | pure I, PI, `factormax`, `safety` | 144 → 193 accepted of 400k |
| per-component tolerance | rtol 1e-2 on `Z1` alone | none |
| quadrature | 8 → 512 points | none |

**Why they fail.** Not classical stiffness — an implicit method would fix that. The elastic
integrand goes as `R⁻¹²` (Yeoh's `c₃(I₁−3)³` with `I₁−3 → λ⁻⁴`), so `Ze ~ R⁻⁹` and the
derivatives gain four decades between `R/R_max` 0.02 and 0.009. At that point the law demands
2×10¹⁵ Pa from a 3 kPa gel.

**What worked instead.** Refusing the region: `min_radius_ratio` (#248), implemented as a solver
event because the runs it catches never *reach* a post-hoc check. Turns a 6–9 s grind ending in
"maximum number of solver steps" into a 0.9 s refusal that names the cause.

**How the region could be made reachable.** A time reparametrisation (§2.1). Nothing else tried
touches it.

**Superseded — the cause was elsewhere.** Everything above is measured and stands, but the
attribution was wrong, and so was the remedy it implied. Instrumenting the terminal state
settles it: every failure ends with `Rd = Cstar` to six digits, the radius frozen, and the step
size at machine epsilon. That is the Keller–Miksis denominator `(1 − Rd/c)R + 4μ/(ρc)` going to
zero — a coordinate singularity of the first-order-in-Mach **dynamics**, not of the memory
integral and not a blow-up of the constitutive law. It is approached from below and never
crossed, which is exactly why the budget scaled linearly and why none of the seven knobs above
moved it: they were all aimed at the material.

The elastic law is still absurd there — 2×10¹⁵ Pa from a 3 kPa gel is unchanged — but that is a
second problem, and `min_radius_ratio` already covers it. `max_wall_mach` covers this one, and
the two are not interchangeable: on this band `λ₁ = 10⁻³` survives `R/R₀ = 0.010186` while
`6.81×10⁻⁶` stalls at `0.010447`, so no radius floor separates them, while peak Mach separates
them 0.312 against 0.99999.

**And one hypothesis of mine that its own test refuted.** I proposed the band was a *resonance*:
an arm relaxing on the collapse timescale stores elastic energy through the collapse and
releases it into the rebound, so peak rebound speed should occur near `De = λ₁/t₀ ~ 1` and grow
with stiffness. Swept below the sonic line where the answer is not clipped, it does neither —
peak Mach across four stiffnesses lands at `De` = 1.11, 11.1, 0.198, 11.1, with heights 0.136,
0.063, 0.241, 0.068. No peak near `De ~ 1`, no monotone rise with stiffness. The transition is
bimodal instead: runs either stay under Mach 0.25 or sit on exactly 1.000. Why the band has
edges where it does remains unexplained; what stops the solve inside it does not.

---

## Part 3 — Open, and the weakest links

### 3.1 The dominant trial mode is still unexplained — but it is now definitively not parameters

**Updated.** #222's title hypothesis is refuted by the nonlinear test. Nine *linearised*
projections had failed; the objection was that a linearisation about one point can miss
variation large enough to turn the Jacobian. So every trial was fitted separately — 39 fits,
in the identified coordinates.

Giving each trial its own parameters does **not** absorb the mode. At 23 °C and 33 °C it keeps
its shape (|cos| between the mode before and after of **0.987** and **0.943**) and takes a
*larger* share of the reduced residual (0.681 → 0.772, 0.720 → 0.835). An absorbed mode
collapses; these do not. Only 15 °C partially absorbs (0.539 → 0.412 at |cos| 0.699).

**And a claim I made from it was wrong — see the note at the end of this section.**

**The first version of that test had a gap, and the claim was too strong.** It fitted
`(mu, galpha, lambda1)` — material axes — with `Req` pinned at `maximum/stretch` for every
trial, so it could not have absorbed variation in the initial condition. That matters here
specifically: the mode is 97–98% post-collapse and the afterbounce period is set by `Req`, which
is the quantity nucleation is least repeatable in.

Refitting with `Req` free over its ±11% prior:

| record | raw | material only | + Req | \|cos\| to raw | fits on a bound |
|---|---|---|---|---|---|
| 15 °C | 0.539 | 0.412 | 0.338 | 0.683 | **7 of 18** |
| 23 °C | 0.681 | 0.772 | **0.791** | **0.988** | **4 of 14** |
| 33 °C | 0.720 | 0.835 | **0.717** | **0.955** | 0 of 7 |

At 33 °C the answer is clean — nothing rests on a bound, `Req` moves 3.0% between trials, and
the mode keeps its shape. At the other two the test is **inconclusive rather than negative**:
those fits wanted a larger `Req` than the prior allows and were cut off, one-sidedly, which
reads as a systematic offset in the record's stretch rather than as trial-to-trial scatter.

So the licensed claim is narrower than what was first written: not the *material* parameters
anywhere, and not the initial condition either at the one record where it could move freely.

**What was gained.** `Σθ` is now measured rather than inferred: 15–34% on `mu`, ~20% on `g·α`,
59–117% on `lambda1` (log coordinates). It is an overestimate — a single trial's noise is not
separable from the trial spread with this data — but it is the first direct estimate, and it is
what §3.4 below runs on.

**Still open.** What the mode *is*. Known: 54–72% of the scatter, concentrated in the rebound,
record-specific, autocorrelation decaying faster than any exponential and turning negative by
lag 8, and not absorbed by a full nonlinear refit.

The structural consequence holds regardless: **the trial scatter and the model discrepancy
occupy different parts of the record** — scatter in the rebound, δ̂ 65–69% in the first
collapse. Anything that whitens one using the other is mixing things that do not overlap in
time.

Scripts: `per_trial_fits.py`.

### 3.2 The profile transfer — tested, and it strengthens Part 1

**Closed.** Three coordinates were compared against the same measured spreads.

| record | raw phase | absolute time | smoothed phase |
|---|---|---|---|
| 15 °C | 514.2 @ 6.92 | 50.0 @ 20.00 | **416.0 @ 10.85** |
| 23 °C | 416.0 @ 10.85 | 50.0 @ 20.00 | **416.0 @ 10.85** |
| 33 °C | 416.0 @ 10.85 | 50.0 @ 20.00 | **416.0 @ 10.85** |

Median-filtering the spread before transferring makes **all three records agree**, and none
agrees with the constant. So the 15 °C outlier under the raw transfer was estimator noise — an
18-trial standard deviation carries 17% relative error and should not be interpolated raw — and
the disagreement that remains is with the flat scale rather than among the records, which is
the opposite of what a transfer artifact looks like.

Absolute time picks a different geometry entirely and is wrong for a statable reason: `t_c`
spans a factor of 24 across the candidates, so the same clock time lands at a different point
of the collapse. It is reported because it is what a reader would do by default.

The estimator warning stands and is now load-bearing rather than incidental: a raw per-sample
participation ratio of 2.3–4.7 looked like a property of the apparatus and is a property of the
estimator — 7–21 under a median filter, 45–111 under a percentile floor. **Smooth before
transferring.**

Scripts: `noise_transfer.py`.

### 3.4 The allocation nobody optimised is worth more than the geometry

**New.** With `Σθ` measured (§3.1), the split between bubbles and frames can be priced.
Averaging `J` trials of `N` samples gives `F = J(Σθ + F_N⁻¹)⁻¹`: trials enter linearly and never
saturate, samples only through `F_N`, which stops mattering once `F_N⁻¹` is small against `Σθ`.

| record | current | best at the same frame budget | gain |
|---|---|---|---|
| 15 °C | J=18, N=201 | **J=72, N=50** | +3.93 nats |
| 23 °C | J=14, N=192 | **J=56, N=48** | +3.96 nats |
| 33 °C | J=7, N=198 | **J=28, N=49** | +3.40 nats |

Doubling the samples buys 0.035–0.188 nats. For comparison: choosing the geometry under the
wrong noise model costs at most 1.34, and restoring testability costs 0.01–0.24. **The
allocation is worth three to ten times either.**

Two readings the table does not survive without. The "+2.079 nats" for doubling trials is
`p·log 2` *by construction* — `F` is linear in `J` — so that column is arithmetic; the
measurement is the sample column. And `F_N` is scaled linearly in `N` as though samples within
a trace were independent, which they are not (lag-one above 0.9, `N_eff` ≈ 10 of 201), so the
true `F_N` is flatter and cutting `N` costs *less* than stated — conservative in direction, but
not to be pushed past `N ≈ N_eff`. `N` is also not a free dial: frame rate and record length
set it.

**Not folded into the geometry optimisation, deliberately.** Doing so needs `Σθ` at an unrun
geometry, and we have three measurements of it against a 24× span in `t_c`. That is a weaker
transfer than the σ(t) one in §3.2 and has not been attempted.

Scripts: `trials_versus_samples.py`, `per_trial_fits.py`.

### 3.3 The regret bound does not cover our case as stated

T. Chen's bound `|U − Û| ≤ B√(2ε) + ε` perturbs the **prior**; the likelihood appears exactly in
both terms. Our misspecification is a **likelihood** (a noise covariance).

Routing through an intermediate `Ũ = I(π̂, L)` splits the problem, and one broken step can be
repaired: his data-processing bound `KL[m̂‖m] ≤ KL[π̂‖π]` requires `m̂` and `m` to share a Markov
kernel, which a perturbed likelihood breaks. The KL chain rule on the joints fixes it —
`KL[L̂π̂ ‖ Lπ] = KL[π̂‖π] + E_π̂[KL(L̂‖L)]`, then data processing on the deterministic
marginalisation — giving `KL[m̂‖m] ≤ ε + κ̄`, which telescopes against a new term to leave
`max(ε, κ̄)`.

One term resists: `∫π̂∫(L − L̂)·log(L/m)` needs `sup_Y |log(L/m)|`, which diverges quadratically
for Gaussians.

**But for our case the bound is unnecessary.** `L` and `L̂` share a mean function and differ only
in covariance, so `KL(L̂‖L)` is closed-form and θ-independent, and the EIG difference is computed
outright. That is what Part 1 does. The bound is the right tool for prior error; direct
computation is the right tool for ours.

---

## Part 4 — On the borrowed document

T. Chen's surrogate-BOED note contributed §2.5 and §3.3. Two observations for whoever uses it
further:

- **Its central modelling choice is not supported by its own numbers.** §5.3 argues that
  propagating LMC uncertainty matters at small `Nd`, but Table 1's EIG RMSE differs by under 1%
  between the uncertainty-aware and plug-in surrogates, and at `Nd = 32` the plug-in is *better*
  (0.0089 against 0.0097).
- **The hard part is assumed away for us.** §6.2 is explicit that `Σθ(d)` is supplied
  analytically rather than learned. That is precisely the object §1.1 and §3.1 are stuck on.
- Minor: "the reference grid maximizer is recovered exactly" is quantised by a 31×31 grid whose
  `d₂` spacing is 0.033, and the document itself notes the continuous optimum is (0.212, 0.333)
  against the grid's (0.200, 0.333) — so recovery is of the grid's answer, not the optimum.

---

## Ledger

| | count |
|---|---|
| Document claims contradicted by measurement | 4 (2 high severity) |
| Already fixed and merged | 3 |
| Approaches tried and refuted | 6 |
| Refutations with a stated path to working | 3 (§2.1, §2.2, §2.5) |
| Load-bearing assumptions tested and held | 1 (§3.2) |
| Hypotheses refuted by a definitive test | 1 (§3.1 — #222's title) |
| New findings larger than the issue that prompted them | 1 (§3.4) |
| Central questions still open | 1 (what the dominant mode IS) |

The ledger has shifted since the first version. §3.2 was the weakest link and now supports the
conclusions it threatened; §3.1's hypothesis is refuted rather than merely unconfirmed; and
§3.4 — an allocation question raised in passing — turns out to be worth more than every
geometry effect in Part 1 combined.


---

## Postscript: a claim of mine that the source data refuted

Worth recording separately because it went into the document before it was checked.

From the one-sided pinning in the `Req` refits I concluded the stored stretch was "too large on
every record" and wrote that into `sec:reqprior`. Chasing the provenance to
`paper_imr_windowing` refuted it on two counts.

**The definition is right.** That source defines `R_∞` as the *equilibrium bubble-wall radius* —
exactly the quantity the solver wants — so the loader handing `R_max/stretch` to `Req` is
correct. There is no convention error.

**The offsets are mostly inside the tabulated uncertainty.** The stretch is a mean over the
events in each bin, with a spread:

| record | fitted | tabulated | deviation | N |
|---|---|---|---|---|
| 15 °C | 6.40 | 7.09 ± 0.36 | 1.9σ | 18 |
| 23 °C | 6.80 | 7.37 ± 1.28 | **0.45σ** | 14 |
| 33 °C | 6.59 | 6.83 ± 0.49 | **0.49σ** | 7 |

Only 15 °C sits outside at all, marginally. Two of three are well within.

**What is actually true**, and what `sec:reqprior` now says: a *population mean is applied to
every trial*. The same table gives `R_max` as 277±48, 298±59, 312±32 µm — individual bubbles
differ by 10–20% in the absolute scale that sets the Reynolds, Weber and Deborah groups, and one
value is used for all of them. The per-trial `Req` spreads (1.9%, 6.7%, 3.1%) are *smaller* than
the source's own scatter, so they corroborate it rather than showing anything new.

**Still open, and now better posed:** per-event `R_max` varies more than the ratio does, is not
fitted anywhere, and is not recoverable from the traces, since each is normalised by its own
maximum before it reaches this repository. That is a live candidate for §3.1's dominant mode and
it needs the un-normalised radii.

**The lesson.** Two tests agreed the fits wanted a larger `Req`, and I read agreement between
two model-based tests as convergence on a fact about the apparatus. Neither could see the
tabulated uncertainty, because neither had the source. Checking provenance before writing a
calibration claim would have cost one `grep`.
