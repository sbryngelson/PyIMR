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

**Severity: medium.** Depends on §3.2's transfer assumption.

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

---

## Part 3 — Open, and the weakest links

### 3.1 The dominant trial mode is still unexplained

#222's central question. Nine candidate directions have failed: timing, normalisation, `Req`,
`R_max` through the nondimensional groups, and the four material axes. What is known is
negative or descriptive — it carries 54–72% of the trial scatter, lives in the rebound, is
record-specific, and its autocorrelation decays faster than any exponential and turns negative
by lag 8.

Worth noting the structural consequence, which holds regardless: **the trial scatter and the
model discrepancy occupy different parts of the record** — scatter in the rebound, δ̂ 65–69% in
the first collapse. Anything that whitens one using the other is mixing things that do not
overlap in time.

### 3.2 The profile transfer is the load-bearing assumption in Part 1

Everything in §1.2 and §1.3 depends on transferring a measured σ(t) to an unrun geometry by
phase (`t/t_c`). That is one modelling choice, defended by §2.3's finding that the profile is
dynamical rather than instrumental — but it is not tested.

Two ways to attack it, neither done:

- transfer by absolute time instead of phase, and see whether the conclusion survives;
- fit a smoothed two-component form (a flat floor plus a dynamical term) and transfer that,
  rather than interpolating a raw estimate that carries 17–29% relative error per sample.

A related warning: a raw per-sample participation ratio of 2.3–4.7 samples looked like a
property of the apparatus and is a property of the estimator — it moves to 7–21 under a median
filter and 45–111 under a percentile floor. Nothing weighted by raw `1/σ²` should be reported
without that sensitivity beside it.

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
| Central questions still open | 2 (§3.1, §3.2) |
