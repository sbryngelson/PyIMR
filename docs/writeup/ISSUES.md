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

## Part 5 — six changes to how the experiments are optimised

Prompted by the observation that the design chapter's own numbers rank the geometry
optimisation *last*: the bubbles-against-frames split is worth +3.4 to +4.0 nats, geometry ≤1.34,
testability 0.01–0.24, robustness 0.21. Four of the six below worked; two refuted the prediction
that motivated them, and those are the more useful entries.

### 5.1 Price the candidates, and the allocation joins the certified problem ✓

`budgeted_measure`. Making a candidate a pair `(geometry, N)` and letting ξ allocate **trials**
keeps `F = J Σ ξ_i M_i` linear, so Kiefer–Wolfowitz survives; the only breakage is that an
N-sample trace costs N frames. Normalising by cost instead of count is the same simplex seen
through `η_i = ξ_i c_i`, so the certificate is inherited and reads `tr[M⁻¹M(x)]/c_x ≤ p`.

**+9.99, +5.41, +2.67, +0.90 nats** over the geometry-only formulation as one bubble is priced at
0, 25, 100, 400 frames. Testability floor costs 0.008–0.034. Σ swept ×½–×2 moves the recommended
N once. **The Σ objection was overstated**: Σ is a covariance of *material* parameters and the
three records measuring it differ in temperature, not geometry (277/298/312 µm).

### 5.2 Make the discrepancy a coordinate ✓

`m(θ) + η·δ̂`, with δ̂ transferred by phase. Transfer premise holds: the three δ̂ overlap at
0.44/0.42/0.81 against 0.05 for unrelated curves. Visible share predicts the parameter error in
the right direction under **both** synthetic mismatches (rank −0.40, −0.60) — weak, four points,
and reported as not-refuted rather than calibrated. Material `log det` wants 89 µm; discrepancy
visibility wants 1200 µm.

### 5.3 Design under the correlated likelihood — prediction refuted ✗

**Predicted:** the correlated likelihood would reorder the geometries.
**Measured:** it does not move the recommendation under any of the three noise profiles.

Two further corrections in the same direction. The operator discrimination is **amplified**
(median ×3.23/×2.85/×2.51, range ×0.52 to ×10.4), not deflated — the document's ×0.05 figures are
for a constant offset and a smooth half-cycle, and these differences are not smooth enough to sit
there. But the *mechanism* holds: amplification ranks at +0.66/+0.64/+0.65 against the share of
each difference's energy above eight cycles. A correlated likelihood pays for fast differences
and charges for smooth ones, even where the ranking is unchanged.

### 5.4 Maximin over the deflation, instead of guessing where the cliff is ✓

Worst-case share of live questions funded, over both readings of the truth:

| axis | designed at face value | designed at full deflation |
|---|---|---|
| constitutive | 0.978 | **1.000** |
| bubble dynamics | 0.595 | **1.000** |
| thermal | 0.410 | **1.000** |

The asymmetry is the argument: wasted effort on a decided rival is recoverable, a question never
funded is not. §screening's "precondition for the design work" dissolves.

### 5.5 Two-stage budgets — prediction refuted ✗

**Predicted:** staging would beat one-shot, because the rival list is uncertain.
**Measured:** +0.000 rivals at every split of every budget from 2 to 12.

The one-shot arm already decides all four live operator rivals at a budget of **two**, at 971 µm
and stretch 5.62. The useful part is what that says about the records rather than about staging:
the operator margins are small because the records sit at 277 µm and stretch 7.09, not because
the question is hard.

Two things had to be fixed before the number meant anything, and both looked like results.
Without Step 0 the rival list carries Rayleigh–Plesset, whose separation spans eleven decades and
saturates everything — compressibility is not in doubt. And a substring test matched `SLS` inside
`qSLS`, scoring the incumbent against itself on an axis that has no live rival at 15 °C at all.

### 5.6 Temperature as a design axis ✓ (and near-tautological)

Under an Arrhenius sharing model, **the six-parameter information is singular with temperature
held** — the design is refused outright. A slope cannot be measured at one point, so no geometry
optimisation at any budget identifies the temperature dependence. Freeing it certifies at
8e-10 on six settings spanning 12–36 °C. The most conditional result here: three temperatures
fitting six coefficients leaves one residual df per axis, so the Arrhenius form is assumed.

---

## Part 6 — what a referee's report changed

Five items from a critical read of the document. Three were bookkeeping; two were real.

### 6.1 The compressibility conclusion contradicted the paper's own deflation ✗→fixed

The conclusions called compressibility "the most robust statement in this work" at 74–179 nats.
§screening deflates every margin by `N/N_eff ≈ 20`, which puts Rayleigh–Plesset at −4.04 and
−3.79 on two records — **live** under the document's own 5-nat screen. Both statements could not
stand.

Recomputed properly, with Σ in the likelihood rather than a scalar divisor:

| record | independent | ρ=0.8 | ρ=0.9 | ρ=0.95 | fitted |
|---|---|---|---|---|---|
| 15 °C | −185.7 | **−4.3** | −5.6 | −8.5 | −5.1 |
| 23 °C | −81.3 | −16.8 | **−3.3** | −6.0 | −9.2 |

It does not rescue the ranking; it **dissolves** it. The compressible operators span 0.31 nats at
15 °C and 0.33 at 23 °C against the 17.1 and 4.2 the independent likelihood reports — they are
not ordered by these data at all. Compressibility itself straddles the threshold.

**One bug caught on the way, and it produced a plausible wrong answer first.** Letting each
operator estimate τ from its own residual gave RP *winning* by 62 nats at 23 °C and losing by
10887 at 33 °C. Evidences under different covariances differ by `log|Σ|` and are not a Bayes
factor: a model allowed to pick its own noise model picks a flattering one, uncharged. One τ per
record, shared, fixes it. **33 °C is excluded** — it lands three orders of magnitude out of
family, the signature of an ill-conditioned Σ on the record with seven trials.

### 6.2 "A ranking among wrong models is still a ranking" — tested, and it holds ✓

Generated from two truths outside the catalogue, ranked by evidence, and separately measured
which candidate is genuinely closest (residual of its own best fit against the noiseless truth):

| truth | evidence winner | closest | rank correlation |
|---|---|---|---|
| thermal | qSLS | qSLS | **+1.000** |
| two-mode | qSLS | qSLS | **+1.000** |

Identical orderings, including the bottom pair whose distances differ by 1.3%. Two limits: both
truths are physics already screened, and the catalogue's tiers are thousands of nats apart, so
the test is not stressed in the few-nat regime §screening says the operator axis is in.

### 6.3 Three sections reported log evidence on three scales ✗→documented

−2289.86 (grid), +529.26 (capped Laplace), +2173.55 (SMC), same model, same record. Decomposed:
the σ convention alone moves `log Z` by **1373 nats** on one residual. The capped Laplace
reproduces to 2.15 nats; the grid is a different object (marginalised β, redundancy, BIC); the
SMC sits +1646 away and is **not** reconciled. What saves the conclusions is that every
comparison is drawn *within* one pipeline. New appendix §What a nat means here.

### 6.4 The geometry extrapolation was never stated as one assumption ✗→fixed

Three transfers (σ, τ, δ̂) carry the design chapter across a 24× span in `R_max`, and each is
validated by cross-record agreement — across records that differ in **temperature**, not size.
Now a standing subsection at the head of §design, referenced from all three users.

### 6.5 The discrepancy had never been screened against first-collapse physics ✓ (new finding)

65–69% of δ̂ is in the first collapse; every candidate on the old screen was a bulk effect.
§enrich even named sphericity as least secure there and never screened it.

| candidate | 15 °C | 23 °C | 33 °C |
|---|---|---|---|
| asphericity n=2 (silhouette) | 0.4% | 0.6%† | 0.0% |
| asphericity n=2 (volume equiv.) | 0.2%† | 2.5%† | 5.0%† |
| **initial wall velocity** | **22.6%** | **37.6%** | **11.7%** |

† anti-aligned. Shape modes from Plesset (1954), integrated along the fitted trajectory, screened
under both imaging conventions.

**Asphericity is not it** — at most 5%, below thermal transport, and it was the obvious suspect.
**The initial condition is the largest candidate ever screened here** at 37.6%, against 23.7% for
the initial radius and 6.5% for the best genuine physics on the old list. Read with §reqprior the
two point the same way: the discrepancy is better aimed at *how the records begin* than at the
constitutive law.

---

## Part 7 — the initial condition, fitted rather than screened

Following §6.5, which made a non-zero initial wall velocity the largest candidate ever put
through the discrepancy screen at 37.6%. Fitted with the **stretch also free**, since pinning a
quantity known to ±5–17% would let one configuration error masquerade as the other.

### 7.1 The 37.6% does not survive the fit ✗

| free | 15 °C lack-of-fit | 23 °C | 33 °C |
|---|---|---|---|
| nothing | 16.79 | 2.95 | 1.80 |
| `R_eq` | **8.43** | **2.15** | 1.60 |
| `u0` | 15.38 | 2.86 | 1.57 |
| both | *withheld* | 2.30 | **1.38** |

χ²/N falls everywhere (0.97→0.50, 1.03→0.32, 0.44→0.23) — that is what free parameters do and is
not the question. On **lack of fit**, adding `u0` on top of a free stretch improves 33 °C and
worsens 23 °C. **15 °C is withheld**: all three prior widths fail the nesting gate — a box
containing another scoring worse than it — at 16 restarts and again at 32. That surface is
multimodal, so the cell is not read.

The stretch remains the larger configuration effect on every record.

### 7.2 Two things that did come out clean ✓

**`u0` is determined, not bounded.** −0.022 to −0.033 of the Rayleigh scale, always inward,
identical at ±10% and ±30% prior width.

**`R_eq` and `u0` are separately identified.** corr = +0.06 to +0.13, cond(JᵀJ) ~ 1e2 — against
−1.00000 and 1e8 for the g–α pair. Two configuration quantities that could easily have been one
direction are not.

**The obvious interpretation fails a quick check.** An inward velocity at the imaged maximum
suggests the maximum is a sampled frame, not the true turning point. Half a frame of the model's
own deceleration is −0.007 to −0.008 — three to four times too small. Sign and order right,
magnitude wrong, so sampling is part of it at most.

### 7.3 #222: the mode is not the initial condition either — four for four ✗

| record | raw | material | +`R_eq` | +`R_eq`+`u0` | \|cos\| to raw |
|---|---|---|---|---|---|
| 15 °C | 0.539 | 0.412 | 0.338 | 0.340 | 0.678 |
| 23 °C | 0.681 | 0.772 | 0.791 | **0.800** | **0.987** |
| 33 °C | 0.720 | 0.835 | 0.717 | **0.840** | **0.945** |

Nothing moves. Nothing pinned, so the boxes were not the constraint. There *is* real trial
scatter in both — `u0` by 0.008–0.016 of the Rayleigh scale, `R_eq` by 2.0–6.4% — it just is not
what carries the mode.

**#222 can now be stated at full strength**: whatever carries half to three-quarters of the trial
scatter is not any quantity this model has. Not the material parameters at each trial's own
optimum, not the equilibrium radius over a box that does not cut off, and not the initial wall
velocity. Four absorptions attempted, four failed — and the last two were the configuration
quantities the mode's post-collapse position pointed at most directly.

---

## Part 8 — the acquisition data arrives, and #222 gets a mechanism

`data_pa5_tempsweeps_master_20260210.mat`: 835 events, each with a measured `R0` (= R_max), a
measured `Req`, and the un-normalised trace. The quantity #222 was blocked on for the whole
investigation turns out to be a field. (Summaries only are committed; the raw file is not.)

### 8.1 The scatter is in the scale, not the ratio ✓

| bin | n | R_max | cv | stretch | cv |
|---|---|---|---|---|---|
| 12–18 °C | 253 | 311.1 ± 78.8 µm | **0.253** | 8.21 ± 0.43 | 0.052 |
| 20–26 °C | 100 | 301.7 ± 82.9 | **0.275** | 8.11 ± 0.53 | 0.066 |
| 30–36 °C | 84 | 341.3 ± 76.9 | **0.225** | 8.12 ± 0.50 | 0.062 |

R_max varies **four times more** than the stretch. That settles the successor question left open
when #253 closed. Two consequences for the paper beyond #222: the stretch is available **per
event**, so §reqprior's "population mean applied to every trial" need not be an assumption at
all; and the raw mean stretch is ~8.1 at every temperature against the table's 7.09/7.37/6.83, a
10–15% discrepancy worth chasing.

### 8.2 A scatter in R_max is a scatter in the CLOCK, and that is #222's mode ✓

`R(t)/R_max` is near-universal in `t/t_c` with `t_c ∝ R_max`, so size scatter is pure time
dilation, whose signature `∂R/∂log t_c = −t·Ṙ` is small early and grows as phase error
accumulates — concentrated post-collapse, which is where #222's mode has always been.

| | events | leading share | \|cos\| to −t·Ṙ | chance |
|---|---|---|---|---|
| **raw** 12–18 °C | 253 | 0.882 | **0.951** | 0.075 |
| **raw** 20–26 °C | 100 | 0.869 | **0.976** | 0.075 |
| **raw** 30–36 °C | 84 | 0.861 | **0.973** | 0.075 |
| processed 15 °C | 18 | 0.539 | 0.572 | 0.071 |
| processed 23 °C | 14 | 0.681 | 0.581 | 0.072 |
| processed 33 °C | 7 | 0.720 | 0.518 | 0.071 |

**In the acquisition data the dominant mode IS the time dilation** — |cos| 0.95–0.98, seven
eighths of the variance. Nothing to explain there.

What reaches this repo is the residue. The pipeline already removes most of it: implied timescale
scatter 1.0–2.1% against 22–28% measured per event, a **factor of 15**. What survives still
aligns at 0.52–0.58 and carries **24–27% of the trial variance** — the largest single share
anyone has attributed to this mode, against 0.467 for the best of nine linearised parameter
directions.

**Not the whole of it.** Projecting the dilation out leaves a leading mode overlapping the
original at |cos| 0.81–0.85. A quarter is accounted for; the rest is not. But #222 has gone from
"a mode with no candidate" to "a mode with one identified mechanism carrying a quarter of it".

---

### 8.3 Which clock the pipeline uses, and what each one buys ✓

The file says it by its own fields: `R_norm = R/R_max` to the last bit, and
`d(t_norm)/dt = 10/R_max` with `sqrt(p_inf/rho) = 10.07 m/s`. **Each event's time is divided by
its own inertial scale.** That is where the factor of fifteen went. The file also stores
`tc_norm` — the *measured* collapse time in units of that estimate — which scatters by 3.1–3.3%.

Rebuilding the deviation matrix under three clocks on the same events:

| bin | clock | share | \|cos\| to −t·Ṙ | implied cv | total variance |
|---|---|---|---|---|---|
| 12–18 °C | population | 0.674 | 0.944 | 0.226 | 5.36 |
| | per-event inertial | 0.756 | 0.852 | 0.024 | 0.383 |
| | measured collapse | 0.637 | 0.926 | 0.017 | **0.245** |
| 20–26 °C | population | 0.729 | 0.914 | 0.244 | 6.27 |
| | per-event inertial | 0.770 | 0.872 | 0.026 | 0.408 |
| | measured collapse | 0.632 | 0.953 | 0.017 | **0.228** |
| 30–36 °C | population | 0.694 | 0.879 | 0.200 | 5.15 |
| | per-event inertial | 0.615 | 0.919 | 0.023 | 0.352 |
| | measured collapse | 0.586 | 0.872 | 0.017 | **0.266** |

**The per-event clock is worth a factor of 14.** A population `R_max` inflates trial variance
14–15×, and the scatter it implies (0.20–0.24) matches the measured `R_max` CV (0.22–0.28) — the
identification is exact, not suggestive.

**Realigning on the measured collapse time buys a further ×0.56–0.76.** So `tc_norm` is a real
residual the inertial estimate leaves behind.

**And the dilation shape survives every clock** — |cos| 0.87–0.95, carrying 0.47–0.61 of the
variance at all three. One scalar per event cannot remove it, and that is the physical content:
**the collapse and the rebound do not scale together.** The rebound period is set by the gas the
bubble is left with, so aligning on `t_c` aligns the collapse and leaves the afterbounce
misaligned — and the afterbounce is exactly where this mode lives (97–98%).

The question is now specific: not *what* the mode is, but whether a **two-timescale** alignment —
collapse and rebound separately, both measurable per event — removes what one cannot.

---

### 8.4 #222 explained: the clock has two hands ✓

Both timescales are measurable per event — peak at `t_norm = 0`, first minimum ≈0.92, second
≈1.73 — so a piecewise warp taking (peak, min₁, min₂) → (0,1,2) aligns collapse and afterbounce
at once. They are **not** the same quantity: collapse cv 0.031–0.033, afterbounce cv 0.056–0.069,
correlating at only +0.35 to +0.52.

| bin | alignment | share | \|cos\| to −t·Ṙ | dilation share | total var |
|---|---|---|---|---|---|
| 12–18 °C | inertial only | 0.817 | 0.850 | 0.599 | 0.323 |
| | collapse only | 0.609 | 0.756 | 0.408 | 0.107 |
| | **two timescales** | 0.424 | **0.094** | **0.262** | **0.054** |
| | sham (population knots) | 0.602 | 0.969 | 0.571 | 0.463 |
| 20–26 °C | inertial only | 0.784 | 0.879 | 0.614 | 0.287 |
| | **two timescales** | 0.510 | **0.133** | **0.233** | **0.059** |
| 30–36 °C | inertial only | 0.654 | 0.898 | 0.552 | 0.316 |
| | **two timescales** | 0.682 | **0.297** | **0.193** | **0.065** |

Aligning both removes **four fifths** of the trial variance (×0.168, ×0.205, ×0.206) and
**destroys the dilation signature** — |cos| falls from 0.85–0.90 to 0.09–0.30. What is left is no
longer a clock error of any kind.

**The sham is what makes this physics rather than arithmetic.** Two knots per event remove
variance whatever the knots are, so the same warp was run with knots drawn from the population.
It makes things *worse* (0.40–0.48 against the inertial clock's 0.29–0.32), and the measured warp
beats it by a further factor of 7–9. The reduction comes from the timescales being the event's
own.

On the sweep data the mode is a clock error with two hands: the collapse set by `R_max` and the
ambient pressure, the afterbounce set by the gas the bubble is left with, and the processing
normalising by the first.

### 8.5 But it does NOT carry to the records this document fits ✗

**These are different experiments**, which I nearly failed to check. The file is a temperature
sweep of `pa5` events at R_max ≈ 310 µm, stretch ≈ 8.1. The records come from
`abeid2024experimental` at 277/298/312 µm and stretches 7.09/7.37/6.83. A mechanism shown on one
is a hypothesis about the other until run on the other:

| record | J | alignment | share | \|cos\| to −t·Ṙ | total var |
|---|---|---|---|---|---|
| 15 °C | 18 | inertial only | 0.541 | 0.706 | 0.0637 |
| | | two timescales | 0.635 | **0.877** | 0.0328 |
| 23 °C | 14 | inertial only | 0.533 | 0.612 | 0.0580 |
| | | two timescales | 0.591 | **0.852** | 0.0339 |
| 33 °C | 7 | inertial only | 0.787 | 0.815 | 0.1155 |
| | | two timescales | 0.642 | **0.024** | 0.0345 |

The warp halves the variance on all three, so it does something. But the dilation signature it
*destroys* on the sweep it **strengthens** here at 15 and 23 °C — 0.71→0.88 and 0.61→0.85 — and
removes only at 33 °C. One record of three behaves like the sweep.

Candidate differences, stated rather than resolved: different material at a different stretch;
7–18 trials against 84–253 events; and minima located on 201 samples over five collapse times,
where a one-sample knot error is comparable to the collapse-time scatter being removed.

**Honest statement:** on acquisition data of this kind the dominant trial mode is a two-timescale
clock error and aligning both removes it. On the three records fitted here the same alignment
removes half the variance and leaves the mode *more* dilation-like on two of three. Why is the
open question — and it is far narrower than what #222 began with.

**Process note.** PR #261 was written claiming #222 explained, on the strength of the sweep data
alone. Checking whether the two datasets were the same experiment — they are not — is what caught
it before merge. Comparing `R_max` and stretch between a new data file and the records already in
hand costs one command and should be the first thing done, not the last.

---

### 8.6 Where R_max and R_eq come from, per the authors — and what that eliminates

Confirmed against the stored metadata: `R_max` is a 4th-order polynomial fit through a window at
the peak (`R0PfitPower=4`, `R0PfitWindow=0.4`), `R_eq` is a tail average of R(t)
(`ReqCutoff=0.95`). Both are fitted, not read off.

**R_max is in metres, already calibrated.** 266.3 µm = 91.3 px at `m_px = 2.9157e-6`. (The
authors were unsure; worth telling them.)

**The R_max fit is NOT the residual dilation ✗ — my hypothesis, refuted.** Since the clock is
`R_max·√(ρ/p∞)`, a fit error is a clock error, which is the right shape. But the fits are
uniformly excellent — R² mean 0.9996, worst 0.9944 — and the dilation amplitude does not track
what variation exists: corr(|dilation|, 1−R²) = +0.064, +0.016, **−0.220**. Eliminated.

**R_eq is estimator-noise-limited ✓ — a new limit.** The authors describe two estimators; both
are computable:

| bin | n | stretch (tail average) | cv | stretch (median of extrema) | cv | corr |
|---|---|---|---|---|---|---|
| 12–18 °C | 253 | 8.211 ± 0.431 | 0.053 | 8.291 ± 0.540 | 0.065 | +0.50 |
| 20–26 °C | 100 | 8.106 ± 0.534 | 0.066 | 8.075 ± 0.642 | 0.080 | +0.66 |
| 30–36 °C | 84 | 8.120 ± 0.499 | 0.062 | 8.059 ± 0.541 | 0.067 | +0.80 |

They agree on the mean to under 1% and correlate at only **+0.50 to +0.80**. Two estimators of
one quantity, comparable spreads, correlation of a half → a large share of the apparent scatter
is the *estimator*, not the bubble. Both read the same trace's tail, so their errors are if
anything positively correlated, making that conservative.

**Consequence for §reqprior:** having the stretch per event removes the assumption that every
bubble shares one value. It does **not** make any one bubble's value precise. A substantial part
of the 5–8% scatter is how `R_eq` is measured.

---

### 8.7 Why it doesn't carry: not resolution, not replicates — the experiment ✓

Three candidates for why the two-timescale warp works on the sweep and fails on these records.
Two are settled with the data in hand.

**Resolution is eliminated — arithmetic.** Records: 201 samples over 5 t_c = **0.0250 t_c**
spacing. Sweep: median **0.0315**. The records are *finer* than the data the warp works on.

**Replicate count is eliminated — measured.** Drawing n events at random from a bin where the
warp works, 200 draws per cell:

| bin | n | \|cos\| before | \|cos\| after | P(after > before) |
|---|---|---|---|---|
| 12–18 °C | 7 | 0.818 | 0.502 | 0.10 |
| | 18 | 0.831 | 0.446 | 0.06 |
| | 80 | 0.847 | 0.353 | 0.01 |
| 20–26 °C | 18 | 0.856 | 0.263 | 0.01 |
| 30–36 °C | 18 | 0.875 | 0.302 | 0.00 |

At the records' own counts the warp still works — 0.83 → 0.26–0.50 — and makes the signature
*worse* only 0–10% of the time. Seeing that on **two of three** records is a **1–3% event** under
this explanation.

**So what remains is the experiment.** Different material at a different stretch, same pipeline.
Both datasets show a dilation-aligned leading mode under the inertial clock, and only one of them
is a two-timescale clock error. **They may not be the same phenomenon** — a sharper statement
than before, and a worse one for anyone hoping a processing change fixes these records.

---

## Part 9 — design: the setpoint is not the geometry

The acquisition data exposes a gap in the design chapter that has nothing to do with the paper it
came from. Every certified design names a target `R_max` — 416, 514, 1015 µm — and treats it as a
knob. **It is not one.** Per event, `R_max` scatters at cv 0.25 / 0.28 / 0.23 and the stretch at
0.05–0.07. A certified optimum sitting on a ridge is worth only what can be hit.

### 9.1 The repair costs nothing and the certificate is identical ✓

What the experimenter chooses is a setpoint; what arrives is a draw. So one run supplies

    M(s) = E_{x ~ p(·|s)} [ J(x)ᵀ J(x) ]

and every property the chapter relies on survives: an expectation of information matrices is a
matrix, `M(ξ) = Σ ξᵢ M(sᵢ)` is still **linear** in ξ, so the criterion is still concave and
Kiefer–Wolfowitz still certifies. Nothing is given up. Only the answer moves.

| scatter | nominal peak, delivered | scatter-aware optimum, delivered | regret |
|---|---|---|---|
| ×½ | 514 µm @ 6.92, 7.823 | 636 µm @ 6.92, 8.312 | 0.489 |
| **measured** | 514 µm @ 6.92, 7.740 | **786 µm @ 6.92**, 8.859 | **1.118** |
| ×2 | 514 µm @ 6.92, 6.193 | 786 µm @ 5.62, 7.667 | 1.474 |

Both columns are scored against what actually happens, so the difference is the real cost of
ignoring the scatter. At the measured value the recommended bubble is **half again larger** and
the regret is 1.118 nats. For a whole certified batch: nominal-built measure delivers 8.467,
expectation-built delivers **9.449** — a loss of **0.982**.

### 9.2 Read against the chapter's other effects

| lever | worth |
|---|---|
| bubbles vs. frames (§allocation) | +3.4 to +4.0 nats |
| **ignoring setpoint scatter** | **0.98 to 1.12** |
| geometry under the wrong noise model | ≤ 1.34 |
| restoring testability | 0.01–0.24 |
| robust vs. per-record batch | 0.21 |

It costs about as much as the entire noise-model correction the chapter already treats as
important — and unlike that correction it needs no new measurement and no new theory, only
averaging over a distribution already measured.

### 9.3 The direction is physical

Scatter pushes the recommendation **up** in `R_max`. The information ridge is asymmetric:
undershooting a small bubble loses more than overshooting a large one, so a setpoint with room
beneath it beats a sharper peak without. The batch spreads accordingly — 0.467 at 636 µm, 0.150
at 786 µm, with the small hard-driven corner kept at reduced weight.

---

### 9.4 Pricing the control: account for the scatter, don't pay to remove it ✓

The investment question that follows §9.1. Sweeping the coefficient of variation, other
coordinate held at measured:

| R_max cv | best single setpoint | delivered | under an optimal measure |
|---|---|---|---|
| 0 (perfect) | 531 µm | 9.100 | **10.434** |
| 0.10 | 576 µm | 9.124 | 10.365 |
| 0.25 (measured) | 736 µm | 9.623 | 10.242 |
| 0.50 | 678 µm | 9.760 | 10.049 |

**The two columns disagree in sign, and that is the finding.** As a single setting, scatter
*helps* — perfect control is worth **−0.523** nats, because the laser's imprecision accidentally
spreads a design that would otherwise be a point, and §measure is an argument that spread designs
beat points. As a measure, scatter hurts: perfect control is worth **+0.191**.

The sign of the second column is a **theorem, not a measurement**: reachable E[M] under any
scatter is a convex combination of {M(x)}, hence inside the hull the equivalence theorem already
optimises over, so a measure can emulate any scatter and cannot do worse. Only the magnitude is
informative.

**And the magnitude settles it.**

| lever | worth |
|---|---|
| bubbles vs. frames | +3.93 |
| geometry under the right noise model | +1.34 |
| **accounting** for setpoint scatter | +1.118 |
| restoring testability | +0.24 |
| **perfect R_max control** | **+0.191** |
| perfect stretch control | +0.100 |

**Account for the scatter; do not pay to remove it.** Designing as though the setpoint were exact
costs about a nat and fixing it is free. Making the bubbles actually alike costs an apparatus and
returns a fifth of a nat — because a certified measure wants a spread anyway and is largely
indifferent to who chooses it.

---

### 9.5 Is the allocation gain the clock? — hypothesis not supported ✗

The allocation is the chapter's biggest lever (+3.4 to +4.0 nats), and it rests **entirely** on
Σ_θ ≠ 0: as Σ_θ → 0 the information becomes `J·F_N`, only the product `JN` matters, and there is
nothing to allocate. §latent then showed a quarter of the trial variance is a *clock* error, and
Σ_θ came from per-trial fits — so the parameters were absorbing whatever of it they could.

Rescaling every trial onto the record's median collapse time and refitting:

| record | J | μ raw | aligned | gα raw | aligned | λ₁ raw | aligned |
|---|---|---|---|---|---|---|---|
| 15 °C | 18 | 14.1% | 14.9% | 31.5% | **13.2%** | 51.0% | 93.8% |
| 23 °C | 14 | 29.2% | 30.3% | 18.4% | 35.9% | 46.3% | 65.5% |
| 33 °C | 7 | 20.8% | 20.4% | 53.4% | 52.9% | 86.0% | 79.3% |

Allocation gain moves by **+0.589, +0.043, −0.416** against gains of 5.78 / 5.67 / 5.35. **The
recommendation survives** — at most a tenth of the effect on one record, nothing on another,
wrong sign on the third.

**The test is weaker than it looks, and the Σ table says why.** Alignment mostly *raises* the
spread (λ₁ at 15 °C: 51% → 94%), because rescaling a trace's clock and resampling distorts it in
ways the material parameters then absorb. Only gα at 15 °C behaves as predicted. So this does not
support the contamination reading and is not clean enough to refute it — what it establishes is
that the allocation is not *obviously* a clock artifact, which is what was asked.

**Two implementation notes.** The first version searched `N ≥ 5` and returned J = 723, N = 5 on
every record — an unphysical corner well below the N_eff ≈ 10 floor §allocation itself sets.
Floored at 25. The optimum still sits *on* the floor, which is §allocation's own finding restated:
the optimum wants fewer samples than any defensible floor allows. And the per-trial fits are now
cached, since the cheap half of this study was being blocked behind 40 minutes of the expensive
half on every re-run.

---

### 9.6 Robust to the bias bound — the last of §measure's three limitations ✓

§measure certifies globally in ξ and then admits the information is evaluated at a point estimate
of θ. Its own repair — average over a prior — was never taken because there was no prior.
`eq:biasbound` supplies something better: a *computed* set containing the truth unless the unseen
half of the discrepancy exceeds the seen half. At 15 °C that is ×1.58 in μ, ×1.30 in gα, ×3.11 in λ₁.

**The bound reaches the design.** Across the box's nine corners the best single geometry is one
of **five** distinct answers, from 50 to 307 µm. The recommendation is not stable under the
parameter uncertainty the document itself computes.

| design built on | settings | at the fit | average over box | worst corner |
|---|---|---|---|---|
| nominal, at the fit | 4 | **10.951** | 10.708 | 8.795 |
| Bayesian, max E[log det] | 5 | 10.536 | **11.101** | 9.044 |
| maximin, max min log det | 6 | 9.779 | 10.312 | **9.464** |

Each design is best in its own column — the check that all three are being *optimised* rather
than merely evaluated. Designing at the point estimate costs **0.393** nats on average and
**0.669** in the worst case.

Both robustifications keep the certificate (averaging concave criteria is concave; a pointwise
minimum of concave criteria is concave), implemented by stacking each candidate's corner matrices
block-diagonally so `optimal_measure`'s criterion interface reduces over blocks.

**One honest qualification.** The Bayesian design certifies at 1e-9; the maximin does **not**
converge, stopping at a gap of **4.4**. A pointwise minimum isn't smooth and the subgradient
iteration isn't the right tool. Its value is a lower bound reached, not an optimum proved.

**A Jensen error caught mid-way.** The first version optimised `log det E[M]` and scored
`E[log det M]` — not the same criterion — which showed up as the "averaged" design performing
*worse on the average* than the nominal one, which is impossible if it were optimising that
column. Both reductions now go through the block stack so each optimises the column it is
reported in.

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

## Part 10 — the certificate is smaller than it looked

`docs/writeup/offgrid_and_compound.py`. §measure closed with three limitations. The point
estimate was answered by the bias-box design; these were the other two, and both were runnable
without new data.

**The certificate does not reach past its own grid.** On the 224-point grid the measure certifies
exactly — `max d(x,ξ*) = 4.000000` against `p = 4`, gap 8.1e-10. On a grid eight times finer:

```
evaluated on              points   max d(x, xi*)   bound p   exceedance
the grid it certified on     224        4.000000         4     8.10e-10
a grid 8x finer             1840       69.560989         4     6.56e+01
```

at 552 µm @ stretch 17.82. So every certified batch in this chapter is optimal **over its
candidate set** and none is proved optimal over the continuum.

**Three things it is not.** Not ill-conditioning: `cond(M(ξ*)) = 1.97e4`, eigenvalues
`[14.2, 468, 3288, 278419]`. Not a solver artifact: at 552 µm the spike survives a hundredfold
tolerance tightening — `‖M‖ = 1.0086e6` at rtol 1e-7 *and* 1e-9, identical to five digits — while
`min R/R_max` moves smoothly (0.0379 → 0.0326 → 0.0287) straight through it. Not broad: scanned in
stretch, `d` sits at 0.4–0.9 almost everywhere, hits 9.69 at 17.80 and 64.06 at 17.82, and is back
to 0.52 by 18.00. These are ridges where a small change flips the post-collapse history.

**Where I got it wrong, and how the full run caught it.** Since §setpoint establishes that a
geometry is not a knob, I expected averaging over the measured stretch scatter (cv 0.06, wider
than the ridges) to erase the exceedance — and a spot check at 552 µm said exactly that: `d`
9.690 → 1.018. It does not generalise. Over the whole grid the setpoint-averaged maximum is
**12.38**, exceedance 8.4 rather than 65.6. Averaging helps by a factor of eight and does not
close it; the ridges are not merely information nobody can ask for. One radius is not the grid,
and the tidy version of this finding was wrong.

**The compound front, which §design said was never run.** It was blocked on an estimator, not on
theory: Φ_β is concave for every β, but the per-design log Bayes factor pins at the boundary. The
linearised separation of §batch has no inner optimisation and no boundary, so it can play the
utility.

| β | settings | gap | log det M | separation | of each max |
|---|---|---|---|---|---|
| 0 | 4 | 7e-10 | **31.92** | 179.4 | 100%, 19.9% |
| 0.10 | 3 | 6e-10 | 24.45 | 887.9 | 76.6%, 98.3% |
| 0.50 | 3 | 1e-12 | 20.74 | 902.1 | 65.0%, 99.9% |
| 1 | 1 | 0 | 18.82 | **903.0** | 58.9%, 100% |

Certified along the whole front. Not the dissolution §design claims — no β attains both maxima —
but steeply asymmetric: **a tenth of a weight on discrimination buys 98.3% of it for a quarter of
the estimation.** Take a small β, not a balanced one.

**The lesson, and it is the session's recurring one.** The nice story — "the two loose ends solve
each other" — was constructed from one radius and did not survive the full scan. Every prior
instance this session had the same shape: a result that resolves neatly is the one to run
completely before writing down.

## Part 11 — the data was already here

Two things I said were blocked on data nobody had handed over. Both were runnable from
`~/fastscratch/imr-data-tempsweeps/data`, which holds 192 files. This document was using three.

### The third explanation, the material, is now tested directly

`docs/writeup/material_clock.py`. §twotime found the warp removes half the gelatin trial
variance while *strengthening* the dilation signature on two records of three, and named three
possible reasons: time resolution, replicate count, and the experiment itself. It eliminated the
first two and left the third open.

The third is testable with files already on disk. The processed per-trial traces for **both**
materials sit in the same directory in the same format — a normalised clock in column zero, one
`R/R_max` trace per column — and, decisively, **at the same sampling step** (`dt = 0.025`). So
the PAAm arm can be run through the identical warp, sham and diagnostic: 249 events against
gelatin's 39, with resolution and processing held fixed by construction rather than by argument.

The gelatin columns below reproduce §twotime's table to three digits from an independently
written loader, which is worth having as a check but is not the new result.

| record | material | n | variance removed | \|cos\| before → after |
|---|---|---|---|---|
| gelatin 15 °C | gelatin | 18 | 48.5% | 0.706 → 0.877 |
| gelatin 23 °C | gelatin | 14 | 40.6% | 0.610 → 0.855 |
| gelatin 33 °C | gelatin | 7 | 70.1% | 0.812 → **0.020** |
| PAAm 0.5% | PAAm | 52 | 77.1% | 0.956 → 0.845 |
| PAAm 0.5% sweep | PAAm | 117 | 72.9% | 0.874 → 0.285 |
| PAAm 0.5/0.03% sweep | PAAm | 80 | 67.1% | 0.975 → 0.804 |
| **gelatin** | | **39** | **49.5%** | 0.691 → 0.716 |
| **PAAm** | | **249** | **71.9%** | 0.923 → 0.568 |

**The reduction is entirely earned.** The sham — each event warped with another event's knot
*pair*, so the knot geometry stays realistic and only the match to the event is broken —
*increases* the variance in all six records. Two fitted knots buy nothing for free here.

**The material explanation survives, and it is now the only one left standing.** §twotime
eliminated resolution by arithmetic and replicate count by resampling; this eliminates processing
and sampling by construction, since both arms run the same code over files with the same step. A
difference that persists through all four is the material.

**What differs by material is the residual, not the removal.** In PAAm the dominant mode
starts strongly clock-aligned (0.923) and de-aligns once warped (0.568): the clock *was* the mode.
In gelatin it starts moderate (0.691) and stays (0.716): the warp removes half the variance and
what is left is still as clock-like as what it started with. The split is not clean at record
level — 33 °C gelatin collapses to 0.020, the most complete removal anywhere.

**A note on the first sham I wrote.** It permuted the two knots *independently*, which lets the
first minimum land past the second, breaks the warp's monotonicity, and inflated the sham's
variance for a reason with nothing to do with alignment. That produced "net" reductions above
100%. A number over 100% of a variance is not a subtle error; it is the arithmetic telling you the
control is broken.

### The setpoint scatter, measured on 437 events

`docs/writeup/measured_scatter.py`. §setpoint and §control rest on an `R_max` cv of 0.25 and a
stretch cv of 0.06, taken from a three-row table, plus two unexamined assumptions: log-normal
shape, and independence of the two coordinates. `PA0503_radius.csv` carries **437 per-event
`(R_max, R_eq)` pairs in µm** — the absolute radii Part 9 called "not recoverable."

| quantity | assumed | measured (437 events) |
|---|---|---|
| `R_max` cv | 0.250 | **0.223** |
| stretch cv | 0.060 | **0.067** |
| corr(log `R_max`, log stretch) | 0 | **−0.140** |

Both widths hold and independence is nearly right. Four laws, separating shape from structure:

| law | best setpoint | regret | support shared with measured |
|---|---|---|---|
| assumed log-normal | 678 µm @ 6.70 | 0.434 | 3 of 5 |
| log-normal, measured cv | 678 µm @ 6.70 | 0.436 | 3 of 5 |
| measured, shuffled | 866 µm @ 6.70 | 0.334 | 4 of 5 |
| measured, joint | 866 µm @ 6.70 | 0.267 | 5 of 5 |

**The conclusion survives; the number and the support do not.** The regret stays real and positive
and the setpoint moves under all four laws, so §setpoint's *claim* never depended on the assumed
distribution. But the assumed law overstates the regret by 63% (0.434 against 0.267) and places
**two of five certified settings wrong**. Delivered information is nearly identical (10.240 vs
10.208 nats), so the cost of the wrong law is misplacement, not loss.

**Control is worth nothing measurable at a single setpoint.** Perfect control scores −0.116 nats
against the measured scatter, and the ladder wanders by ±0.16 across the range — the size of the
resampling noise. This is the negative sign §control already established for a setpoint, now
against a measured law rather than an assumed one.

**The caveat that has to be stated.** These 437 events are PAAm (`PA0503`, mean `R_max`
\SI{359}{\micro\metre}), and they are being applied to a gelatin design whose records sit at 277
to \SI{312}{\micro\metre}. That the scatter *law* transfers between materials is §transfer's
standing assumption, not something these events measure. What they do settle is that the assumed
widths and the independence assumption were sound.

### The lesson

I wrote "this needs raw per-event traces nobody has handed over" without running `ls` on the data
directory, which holds 192 files of which this document used three. The PAAm arm that closes
§twotime's last open explanation, and the 437 measured radii that check §setpoint's assumptions,
were both sitting there the whole time. The correct claim was "I have not looked."

A second miss, caught only by grepping the chapter before committing this: I first wrote the
gelatin result up as *overturning* §twotime, having forgotten the chapter already contained it.
It reproduces §twotime to three digits. Re-deriving a result you already have is cheap; announcing
it as a correction is not.

## Part 12 — the refinement closes, after three bugs of my own

`docs/writeup/grid_refinement.py`. Part 10 left the certificate exceeding its bound by 65.6 and
predicted refining would move the answer. It does — to an excess of **0.017**, four tenths of a
percent of the bound:

| round | candidates | max d, independent scan | excess | points above p |
|---|---|---|---|---|
| 0 | 1840 | 6.897 | 2.90 | 12 |
| 1 | 2114 | 4.045 | 0.045 | 5 |
| 2 | 2181 | 4.030 | 0.030 | 1 |
| 3 | 2181 | **4.017** | **0.017** | 1 |

Monotone, violating set 12 → 1, and the surviving point sits at 1023 µm @ 6.70 drifting under a
micrometre per round toward a support point already at 1023.6 µm — the residual of a discretised
vertex-direction method, not a new ridge. The 937 µm ridge Part 10 flagged is now *in* the
support at weight 0.033.

**Three bugs, each of which produced a plausible wrong answer.** Worth recording because none of
them crashed, and two of them printed converged-looking numbers.

1. **The scan set exploded.** I built the independent scan as the tensor product of every unique
   radius × every unique stretch *in the candidate set*. Correct while the candidates are a clean
   46×40 grid; once refinement adds scattered points the set has ~190 unique values per axis, so
   the probe becomes ~36,000 points. It died partway through solving 13,798. Fixed by computing
   the midpoint lattice **once** from the base grid.

2. **The zoom escaped the design box.** `zoom_around` had no clip, so a flagged point near an edge
   walked its zoom outside 50–1200 µm × 3–20, and the next round walked it further. That run
   reported its worst exceedances at 37.6 µm and stretch 26.3 and appeared to *stall at 0.4*. A
   design that cannot be requested is not a counterexample to a certificate over designs that can.

3. **The dedupe ate the vertex-direction step.** After clipping, the run looked clean — every
   point in-domain, no duplicate support — and reported a stall at 0.33. The tell was that the
   worst point was byte-identical across rounds 1, 2 and 3: 937.2 µm @ 17.601, three times. If a
   flagged point actually enters the candidate set the certificate forces `d ≤ p` there, so the
   same exceedance at the same geometry after twice adding it meant it was never getting in. Its
   own zoom box centres the radius at `sqrt(0.96·1.04) = 0.9992` of it — 0.08%, inside my 0.1%
   dedupe radius — so `distinct()` kept the neighbour and discarded the argmax. Flagged points are
   now seeded first and always survive.

**The lesson, and it is not the one about being careful.** Bug 2 and bug 3 each produced a
*stall*: a monotone-looking sequence flattening at a small positive number, which is exactly what
the interesting negative result would look like. Twice I was one write-up away from reporting
"the continuous optimum lives on ridges no finite set represents." What caught both was not
suspicion of the code but arithmetic that could not be true — a worst point outside the domain,
and an identical worst point after adding it. **A converged number is not evidence; a number that
cannot be what it claims is.**

**The chapter's claim, corrected in both directions.** Part 10 said the certificate does not reach
past its grid, which was right. It also implied that was a standing limitation, which was wrong:
the distance from the candidate set to the continuum is a refinement anyone can run.
