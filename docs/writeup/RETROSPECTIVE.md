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

`selection.tex` is 78 pages, restructured around the universality result, compiling clean.
Title: *The same curve on two materials: a measured, material-independent discrepancy in
inertial microcavitation rheometry.*

**Ran, and failed:** the vapour-corrected refit. `delta_corrected.py` with `bubtherm`,
`masstrans` and `vapor` on refitted 2 of 8 records — both gelatin — and raised `qSLS did
not fit from any of 3 starts` on the other six, every PAAm record among them. With no PAAm
survivor there is no cross-material comparison to make, and the script's own `len(good) < 4`
guard returned without writing a result. The wrapper printed `All checks passed!` and exited
0 on top of that.

The failure is not the optimiser running out of budget. `selection.py:614` raises only when
every start ends above `_UNREACHABLE`, the penalty region, and the warm box is centred on
the isothermal optimum — so the first start *is* the point the isothermal fit already found.
A probe evaluating exactly that point confirmed it: the thermal solve completes in 9.8 s
against 3.0 s isothermal on `gelatin_15C`, and on `gelatin_23C`, `paam_PA05` and
`paam_PA05003` exhausts 400 000 integrator steps after ~four minutes without reaching the end
of the record — `SimulationError: The maximum number of solver steps was reached`. One
completion, three stalls. A stalled solve burns 20× the steps of a successful one and still
does not finish, so this is stepsize collapse, not a ceiling set too low; raising `max_steps`
buys nothing a fit could use. Where it does complete it barely moves anything: χ² on
`gelatin_15C` goes 25207 → 24911, a little over a percent.

`sec:universal` had said switching vapour on "raises inside the multistart on some" records,
written from a one-record diagnostic. That paragraph now carries the measured version: 6 of 8,
localised to the forward solve, with the mechanism and the cost named.

So the vapour correction is not merely expensive, it is unavailable at this configuration,
and the cross-material curve stands tested against the radius correction and untested against
vapour. Beyond it, only the single-instrument, single-laboratory limit remains, and that one
belongs in the text permanently rather than being defended.

**Open, not blocked:**

1. An unreconciled inconsistency: `sec:enrich` says 65–69 % of each δ̂ sits in the first
   collapse; the mode measures 47 %. Different definitions, possibly — but unreconciled.
2. Collinearity in the dictionary regression is unquantified (1/R and R/R_max both score
   high and are strongly related).
3. `trial_variation.py` on PAAm proper — PARAMETER_SHARE = 0.393 is a gelatin number.
4. Three bibliography entries flagged `VERIFY BEFORE SUBMISSION`; ten uncited entries to
   cite or remove.
5. The editor's recommendation to cut ~20 pages (`app:background`, `sec:genealogy`).
6. OED-referee findings not independently verified: certificate scope, δ̂-circularity in
   `sec:deltadesign`, prior standardisation (1/√12 vs `sec:prior`).
