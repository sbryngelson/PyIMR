# PyIMR / `selection.tex` — what the effort found, and what it did not

*Covering 2026-07-24 through 2026-08-18, 485 commits. Written as a record, not as
advocacy: the retractions are listed with the results, and in the same detail.*

---

## 1. Three eras

**Era 1 — build the solver (Jul 24–31, roughly 200 commits).**
PyIMR from nothing. Forward and differentiable IMR solvers; the constitutive suite
(Kelvin–Voigt, SLS/Zener, quadratic Zener, two-mode and Carreau Zener, UCM/Oldroyd-B,
Giesekus, PTT, Powell–Eyring, Ogden, Yeoh); eleven bubble-dynamics operators
(Rayleigh–Plesset through Keller–Miksis enthalpy, Gilmore/Tait, Mie–Grüneisen, Herring,
Noble–Abel stiffened gas, second-order Lezzi–Prosperetti); the gas thermal PDE, the liquid
thermal boundary layer, vapour mass transfer; spectral quadrature for the distributed
stress integral and Chebyshev collocation for the thermal PDEs.

Then the whole package was migrated to JAX in five staged passes (`W11`), and the numba
mirror and the numpy backend were deleted. Ancillary outcomes: test suite from unbounded
to 2m42s; comments from 31 % of source to 3 %; pyright baseline from 114 diagnostics to
zero; coverage 93 % → 96 %.

**Era 2 — build the inference and design machinery (Aug 1–12).**
NUTS driven by exact tangents; SMC marginal likelihood validated against exact quadrature;
expected information gain, differentiable with respect to the observation times;
Bayesian-optimization and Pareto design search; 4D-Var, ensemble Kalman analysis and an
iterative ensemble smoother; resolution calibration by measurement rather than folklore;
the model-selection stack with a redundancy-based parameter prior and an Occam model prior;
lack of fit as a first-class diagnostic; the Kennedy–O'Hagan θ–δ split into identifiable
and unidentifiable discrepancy; certified batches via Kiefer–Wolfowitz.

**Era 3 — point it at real data, and watch most of it not survive (Aug 13–18).**
This is where the science is, and where most of the retractions are.

---

## 2. What we found

### 2.1 The result the paper is now about

**The identifiable model discrepancy δ̂ is the same curve on gelatin and polyacrylamide.**
A physical gel of triple helices and a covalent network, sharing an instrument and nothing
else. Cross-material median |r| = 0.726 over 15 pairs, against a null of 0.508 that grants
each curve its own power spectrum *and* its own orthogonality to the sensitivity span;
13 of 15 pairs clear it. PCA over the eight records puts 73.5 % in one mode, and all eight
loadings carry the same sign (0.647–0.932).

Consequence: whatever every model here is missing is **not the chemistry of either
material**, its shape is *measured* rather than postulated, and it is a target any proposed
physics can be tested against.

### 2.2 The controls that make it a claim rather than a coincidence

| Control | Result |
|---|---|
| Is it the fitted form's shadow? | No — 0.699 across qSLS/SLS on 8/8; universality reproduces under SLS at 0.608 (23/28) |
| Is it a shared-orthogonality artifact? | No — the *constrained* null is **lower** (0.466) than the unconstrained one (0.508) |
| Is it the pinned equilibrium radius? | No — 0.844 same-record, 0.629 across materials, after refitting R_eq |
| Is it a function of Ṙ or R̈? | No — dictionary shares 29–70 % on functions of R alone, < 9 % on derivative terms |
| Is it an early-time normalisation artifact? | No — survives window restriction against per-window nulls (0.654/0.596/0.542 vs 0.458/0.452/0.445) |
| Is it rank-one? | Yes to 73.5 %, falling only to 61.4 % with the early window dropped |

**Where it lives.** By squared norm of the common mode: 16.9 % below τ = 0.2, 19.5 % in
0.2–0.5, 11.0 % through the first collapse, 25.1 % and 27.5 % after it. Over half is
post-collapse, in the ringdown, and the *least* of it is at the collapse minimum — which
makes the universal curve and the afterbounce failure **the same object seen twice**.

### 2.3 Identifiability, in closed form

λ₁ is unidentifiable, and not by accident. The Zener arm's effective viscosity
μ_eff = μ/(1 + (ωλ₁)²) implies sd(log λ₁) ∝ ω⁻² ∝ R_max², and every reachable IMR geometry
sits about two decades below the material's own Debye loss peak, where a Zener is
indistinguishable from a Newtonian fluid of viscosity μ. A profiled (Schur-complement)
sweep over R_max, stretch and observation window finds no corner of the reachable set that
resolves it.

**Therefore: every relaxation time reported from an IMR fit is a statement about the prior
box rather than about the material.** Falsifiable by anyone who finds a geometry the grid
missed.

### 2.4 The design machinery is calibrated against data

The Fisher width of a *single* event, corrected by that record's own N/N_eff (≈ 20),
predicts the observed per-event scatter of the fitted parameters. This validates the design
chapter — and simultaneously removes the premise of its own allocation result, because if
estimation noise explains the spread, Σ_θ is small and there is little material variation
to allocate.

### 2.5 The model is rejected, properly

Against replicate pure error, the lack-of-fit F is significant everywhere. Rebuilt so the
numerator is what the fit actually minimises (MS_pure ≡ 1.000 exactly on all 8 records),
F_w runs 3.11–50.56 against AR(1) surrogate nulls at each record's own ρ of 1.52–1.82.
χ²/N of 1.2–1.35 (gelatin) and 0.38–0.62 (PAAm) looks fine and is not.

### 2.6 Instrument- and material-level findings

- R_max and R_eq provenance traced to source; the "trial variation" mode was largely the
  acquisition clock, and the clock has two hands — aligning both removes the mode.
- Medium density is a real sink for a 5–9 % collapse-time deficit; the two halves of a
  record want different densities, which is what settles the value.
- The depth bias and the collapse-time deficit are separate defects, not one.
- **Gelatin refuses time–temperature superposition** (6 standard errors), so the Arrhenius
  axis was assuming something false.
- Half the temperature trend is the vapour pressure the package originally left off.
- R_eq, not R_max, is the sensitive radius — so **no F anywhere in the document is pinned
  better than a factor of two**.

---

## 3. What we did not find

This list is longer than the previous section, and that is the honest shape of the work.

| Claim | Fate |
|---|---|
| Afterbounce spectroscopy: a 72 kHz anomalous bounce | **Retracted** — did not survive a circular-shift null |
| Nonlinear dissipation | **Refuted** — a linear medium with a radiating bubble reproduces the A² scaling |
| A μ_eff scaling law for the identifiability | **Refuted** — predicted R_max², measured R_max^−0.03 |
| Surface tension as the missing term | **Refused** — exponent unresolved over (0.05 … 2.0) |
| The hierarchical whitener | **Retracted** — a sham with the same transform and the intent broken passed identically |
| A monotonicity theorem | **Withdrawn** |
| Four null-acceptance claims against 1/√N | **Corrected**, each with the direction it moved |
| Enrichment-screen shares of ~2 % | **Void** — below the screen's own 20 % noise floor |
| Shape modes / apparatus resonance behind the anomaly | **Refuted** |
| Ratios being ρ-blind; ratios separating μ from ρ; gas loss shrinking R₀ | **All died** within a session, to cheap controls |

And the exclusion list the universal curve leaves behind — the missing term is **not**:
the rate dependence of dissipation, the strain dependence of dissipation, the elastic
nonlinearity, the chemistry of either network, any function of Ṙ or R̈, or surface tension
specifically. α, the stiffening parameter, tracks whichever prior box wall binds and never
the data.

---

## 4. Methodological lessons

These are the transferable part, and several were learned expensively.

**An automated verdict that fails toward the hoped-for answer is worse than no verdict.**
Three did, all in the same direction. A budget check predicted one-sidedness and gelatin
went the other way (F rose 14.16 → 16.69 *with* a better fit). A model-free test pooled
three form-pairs of unequal standing and reported "ROTATES". An exponent sweep declared
resolution from eight argmins sitting on a bracket wall. Each was caught — and in one case
**the failed prediction became the finding**.

**An argmin at a bracket edge is not an optimum.** Twice. Now handled structurally: flag or
discard, and find where the physics caps the sweep.

**A transform needs a sham, not a baseline.** Reading a result against the untransformed
case proves nothing. Reading it against *the same transform with its intent broken* is what
killed the whitening inference.

**Controls are the output.** Five apparent findings dissolved in one session under controls
that cost minutes. Match the control to the step you doubt — validating the *fitter* does
nothing for a claim about the *extraction* — and prefer nulls built from the data by
permutation over nulls built from a noise model.

**A null must carry the structure the measurement carries.** Phase randomisation for
autocorrelation, then *also* projecting out the sensitivity span. Notably the constrained
null came out **lower** than the unconstrained one, so the shared orthogonality was helping
the null rather than the claim — which is only knowable by building it.

**Watch for normalisation artifacts in ratio statistics.** δ̂ = (measured − model)/error, and
every trace starts at R/R_max = 1 by construction, so the early between-event spread is
0.001–0.002 against a median of 0.018–0.025. That made the common mode *appear* to peak at
τ = 0.10. It is a small denominator, not a large disagreement, and I had written the
quasi-static reading of it into the paper before catching it.

**The blind spot that produced it.** Every control I had built varied the *model* or the
*fit*. None varied the *window*. A shared early-time normalisation artifact would therefore
have passed the entire control set. It didn't — but the control set, not the result, was
the thing at fault.

**Check magnitude before spiralling.** 4 nats against a 76-nat margin.

**Don't let a subthread become the paper.** Recency won twice. When asked what the work adds
up to, inventory the whole body first, then ask of each candidate thesis: who is the
audience, and what do they currently believe that this changes?

---

## 5. The incidental things worth remembering

- **The Sanchez `.mat` data is not in the public repo.** I twice declared the IMR data
  directory missing without looking. It holds 192 files, including 437 per-event absolute
  radii — which turned out to be PAAm and a superset, and which unlocked the entire
  second-material comparison the universality result rests on.
- **Ogden never ran.** Its first exponent was being fed an Arruda–Boyce chain count.
- **A factor of two in `setpoint_design.py`**, and every number it fed had to be updated.
- **A spectral convergence "plateau" was measuring integrator noise** — twice, once for the
  thermal scheme and once for the convergence rate itself.
- **`pkill -f delta_corrected` killed my own shell** (exit 144). My own notes had warned
  about exactly this.
- **`abs(hash(...))` is salted per process**, so a multistart seed was not reproducible
  across runs. Measured impact: negligible (spread 0.000–0.042). That was luck.
- **An error handler that discarded the message** turned three distinct failures into the
  bare word `ValueError`.
- **An alignment bug** mapped each curve's index fraction onto [0,1], stretching a 3.9 t_c
  record onto a 4.8 t_c one. The correlations then clustered by *observation window* instead
  of by material — caught only because the grouping split PAAm against itself.
- **A run reported "exit code 0" whose log's last line said `EXIT 124`.**
- **A diagnostic probe passed axes in the wrong order** (qSLS is `mu, g, lambda1, alpha`),
  producing μ = 1000 Pa·s and a bubble that never collapsed.
- **`cd` persists between shell calls**, so two patches printed "patched" against a file
  nobody was executing.
- The commit log is deliberately a lab notebook. Titles like *"The bounce sweep is a
  diagnostic, not an identification, and my reason for hoping otherwise was wrong"* are the
  record of what was believed, and when.

---

## 6. Where it stands

*Rewritten 2026-08-18 after an outside-eyes audit. The audit's own headline is that most of
what it flagged was conservative — and that the one thing it did not anticipate, a solver
default, had silently produced a false conclusion in the text and a false reassurance beside
it.*

`selection.tex` is 83 pages, restructured around the universality result, compiling clean with
zero undefined references (there were eight; `eq:lackoffit` was cited eight times and defined
never).

### 6.1 The vapour question, which turned out to be a solver setting

The paper said the vapour-corrected forward model "cannot be integrated on most of these
records", and I had written that the session before, pleased that a cost hedge had become a
measured limitation. The measurement was real — 6 of 8 records raising, 400 000 steps exhausted
after four minutes, the failure localised to the forward solve — and the conclusion was wrong.

Those runs carried `Nt = 7` spectral nodes against a package default of 25. At `Nt = 15` and
`Nt = 25`, **8 of 8 records integrate**. Worse, the stall is *slower* than the success it stood
in for: ~250 s of collapsing stepsize against ~35 s of completed solve, so the setting chosen
for speed cost seven times the time. An implicit Kvaerno5 was in use throughout, which is why
the symptom read as genuine stiffness and why every remedy I reached for was aimed at a solver
that was already correct.

The reassurance was the same artifact. The one record that completed at `Nt = 7` moved χ² from
25207 to 24911 — about a percent — which I reported as evidence the correction was harmless. At
`Nt = 25` the same evaluation gives **80507**. The comforting number and the blocking failure
had one cause, and I had quoted both.

### 6.2 What the correction then did, and why it is not yet a retraction

With `Nt = 25` all eight records refit, and the cross-material median |cos| falls from **0.726
to 0.302**. That is a threat to the paper's central claim, so it got the control the claim
deserves rather than the write-up it invited.

The corrected fits come back *worse* than the isothermal ones (χ²/N up to 3.66 against ~1.3),
and a richer, more correct physics should not fit worse at a converged optimum. Checking the
box walls: **14 of 16 fits sit on one**. Read in the raw axes that says "search failed". Read
in the identified coordinates it says something much sharper:

- **gα, the combination the data determines, converged** — agreeing to ~3 % between a ×3 and a
  ×10 box on 7 of 8 records. `g` and `α` individually sat on opposite walls, which is just the
  documented g–α trade and not a failure at all.
- **λ₁ did not.** It landed on the lower wall *exactly* — the isothermal value divided by the
  box width, to three digits, at both widths — while χ² kept falling as the wall moved down.

That is not the flat direction the identifiability chapter describes, where the objective is
indifferent and the answer is whatever the prior box says. It is an active descent toward
λ₁ → 0: with the thermal physics modelled, the data stops wanting a relaxing solid, which reads
as the Zener arm having been absorbing thermal damping all along. So the 0.302 is measured at a
boundary of the model class, not at an optimum. `vapour_lambda_floor.py` gives λ₁ the full
seven-decade prior range to find out whether there is an interior optimum at all. **That result
is the open question the paper now turns on.**

### 6.3 The independent test of the same hypothesis

Separately and far more cheaply, `thermal_signature.py` asks whether the correction's *own
shape* is the universal curve — two forward solves per record, no fitting, differenced in noise
units and projected into δ̂'s coordinates. It is not: median |cos| 0.300 against a per-record
null of 0.367, nowhere near the 0.726 the two materials agree at. And it is not a null result
for want of a big enough correction, which is 2–4× the size of the discrepancy it would have to
explain on seven of eight records.

Honest residue: 3 of 8 records *do* clear their own null where 0.4 would be expected, so a
minority component of the correction resembles the curve. The candidate is reduced, not
eliminated. The first version of that script's verdict read the median alone and printed the
answer I wanted; it now reports the count as well.

Reconciling 6.2 against 6.3 is a live task. Both can hold — a large correction pointing
elsewhere still moves the fitted parameters a long way, and δ̂ is defined *at* the fitted point,
so the curve can be destroyed by a correction whose shape it does not share.

### 6.4 What the audit corrected, all of it in the paper's favour or neutral

- **The universality null was quoted at the wrong level.** `tab:universal` compared a median
  over 15 *dependent* pairs against a single-pair null. The ensemble 95th point is **0.315**,
  not 0.508; observed 0.726 sits beyond all 4000 draws. The error was conservative, and
  correcting it most helps the cell that looked thinnest — within-gelatin's 0.551, hedged in
  the text as "sitting ON" the null, is p ≈ 0.003 at the level the claim is quoted.
- **"65–69 % sits in the first collapse"** described one gelatin record. Measured: 46 %, 69 %,
  47 %; 41–69 % across all eight; common mode 47.5 %. A correction, not a reconciliation.
- **The 0.998 "visible share"** in `sec:deltadesign` is first-order optimality restated — a
  least-squares residual is orthogonal to the sensitivity span *at* the fitted point — so its
  four-point rank correlation carries three points. Both now stated. Its stale 1/√N null of
  0.05, already corrected in `sec:transfer`, is brought into line.
- **The form-shadow control** spans constitutive form only; every fit shares a Keller–Miksis
  operator and an isothermal gas. Scope stated, and handed to the vapour section.
- **The dictionary is worse conditioned than suspected**: the polytropic gas term and the
  internal pressure are the *same direction* to numerical precision (|cos| = 1.000 on 8/8 — one
  is the other times a constant), so ten entries span nine directions and the Gram is singular.
  The two terms the headline reading is taken from overlap at 0.88. The subspace comparison
  survives; naming a single term does not.
- **PARAMETER_SHARE = 0.393 transfers.** PAAm gives 28.2–45.8 %, median 44.5 %, against chance
  levels of 3.0–3.8 %; gelatin sits inside that range. The leftover lag-one stays at 0.59–0.73,
  so on the second material too the correlation is mostly structure the model lacks.
- **Bibliography closed.** The three `VERIFY BEFORE SUBMISSION` entries verified against
  publishers' records — every field correct as written, DOIs added, attributed claims re-read
  and accurate. The eleven uncited entries placed rather than deleted; they are the canonical
  citations for machinery the document actually uses. 35 entries, 0 uncited.

### 6.5 Still open

1. **The λ₁ floor run**, which decides §6.2 — and with it whether `sec:universal` needs
   restating. Everything else on this list is smaller than this one item.
2. Two OED-referee findings unverified: certificate scope, prior standardisation (1/√12 vs
   `sec:prior`).
3. The editor's ~20-page cut (`app:background`, `sec:genealogy`) — the author's call.
4. The two-papers question, whose answer now depends on item 1.
5. **A convergence sweep over every other numerical default.** `Nt = 7` was found only by
   asking what the default was. Nothing says it is the only one.
