# Review: *Single-LMC Surrogate Bayesian Optimal Experimental Design*

T. Chen, prepared 6 August 2026 — report (13 pp.) plus the `sec:multi-output_GP` and
surrogate-BOED LaTeX.

Issues found, ordered by how much they affect the conclusions, each with why the current form
does not work and what would fix it. The theory section is the strongest part of the document
and most of what follows is about making its claims survive contact with the benchmark.

---

## What is sound, and worth keeping

- **The three-layer uncertainty taxonomy** — physical `Σθ(d)`, surrogate epistemic `Cf(d)`,
  measurement `p(y|θ,d)` — and the caution never to substitute one for another. This is the
  right decomposition and it is stated more clearly here than in most of the BOED literature.
- **The error analysis** (`sec:eig_error`). The decomposition in `eq:exact_eig_error` into a
  prior-displacement term and a predictive-divergence term is clean, and the chain to
  `eq:design_regret_KL` via Hölder → data processing → Pinsker is the right shape: it converts
  surrogate error into *decision* regret, which is what a design practitioner needs.
- **The regret factor of 2** (`eq:design_regret`) and the Pinsker step (`eq:pinsker`, correctly
  accounting for `‖·‖₁ = 2‖·‖_TV`) are both right.
- **Known heteroscedastic noise with `Σ_mean = Σ_single/R`** (`eq:MOGP_known_noise_matrix`) is
  the correct treatment of replicated training means, and the output-major assembly is careful.

---

## 1. `B(d)` is infinite for the benchmark, so the bound is vacuous as applied

**Severity: high — it affects the central theorem.**

`eq:B_definition` assumes `B(d) = sup_θ KL[L_d(·|θ) ‖ m_d] < ∞`, and every subsequent bound
carries it. For the demonstrated benchmark this supremum diverges.

The prior `p(θ|d)` is Gaussian on all of `ℝ²` (unbounded support), and the forward response
`eq:17` is **cubic in θ₁**. As `|θ₁| → ∞` the likelihood mean `Ȳ(θ,d) → ±∞` while the noise
width stays fixed, so `L_d(·|θ)` remains a bump of fixed height that walks off to infinity. The
evidence `m_d` is the pushforward of a Gaussian through a cubic, so `m_d(Ȳ) → 0`. Then

```
A_d(θ) = KL[L_d(·|θ) ‖ m_d] ≥ ∫ L_d log(L_d/m_d) → +∞
```

Hence `B(d) = +∞`, `eq:eig_KL_bound` reads `|U − Û| ≤ ∞`, and `eq:uniform_eig_bound` and
`eq:design_regret_KL` inherit it.

**Why the current form does not work.** The Hölder step `eq:first_error_bound` uses the
`L^∞`–`L^1` pairing, which asks for `A_d` to be *uniformly* bounded over `Θ`. That is far more
than the argument needs, and it is exactly the assumption an unbounded prior with a polynomial
forward model violates.

**How it could work — three options, in increasing order of effort.**

1. **Compact `Θ`.** State the bound on a truncated parameter set and carry the truncation mass
   as an explicit additive term. This is the honest minimum and costs one paragraph. It is also
   what makes the quantity computable in practice — we evaluated `B(d)` over a bounded box on a
   227-candidate design problem and it was finite and cheap.
2. **Weaken the pairing.** Use `L^q`–`L^p` with `1/p + 1/q = 1`, `q < ∞`:
   `|∫(π − π̂)A| ≤ ‖A‖_{L^q(π̂)} ‖(π − π̂)/π̂‖_{L^p(π̂)}`. Then you need a *moment* of `A_d`
   rather than its supremum — finite whenever the prior has enough moments against a polynomial
   response, which the Gaussian does.
3. **Bound `A_d` by a χ² or moment condition** on the forward model directly, giving an explicit
   `B` in terms of the response's growth rate and the noise scale.

Option 2 is the most useful: it keeps the theorem general and makes the constant estimable.

---

## 2. The report's central modelling choice is not supported by its own numbers

**Severity: high — it is the claim in the title of §5.3.**

§5.3 argues the posterior covariance `Cf(d)` changes the EIG landscape at small `Nd`, and the
uncertainty-aware prior `Σθ(d) + Cf(d)` is presented throughout as the method's contribution.
Table 1 says the difference is not measurable:

| `Nd` | EIG RMSE, predictive | EIG RMSE, plug-in |
|---|---|---|
| 8 | 0.1756 | 0.1772 |
| 16 | 0.0498 | 0.0500 |
| 32 | 0.0097 | **0.0089** |
| 64 | 0.0089 | 0.0090 |

Under 1% everywhere, including at `Nd = 8` where `Cf` is largest, and the plug-in is *better* at
`Nd = 32`. The falling mean trace of `Cf` (8.0×10⁻³ → 4.5×10⁻⁴) shows the covariance contracts;
it does not show that carrying it changed anything.

**Why it does not work here.** `Cf` enters only as an addition to `Σθ` in `eq:10`, so its effect
is set by the ratio `tr Cf / tr Σθ`. On this benchmark that ratio is small even at `Nd = 8`, so
the predictive and plug-in priors are nearly the same distribution and the EIG cannot tell them
apart. The metric compounds it: EIG RMSE is dominated by the *mean* field, which both variants
share.

**How it could work.**

- **Report a metric that is sensitive to the covariance.** Predictive log-likelihood of held-out
  parameter draws, or coverage of the surrogate predictive intervals. These respond to `Cf`
  directly; EIG RMSE mostly does not.
- **Exhibit a regime where the ratio is not small** — `Nd = 4`, or a rougher latent field
  (shorter length scales relative to design spacing), or a higher-dimensional design domain
  where coverage is genuinely poor. The claim is plausible; the benchmark is simply too easy to
  demonstrate it.
- **Or state it as a null result.** "Propagating `Cf` is principled and made no measurable
  difference at this noise level" is a perfectly good sentence, and more useful than an
  unsupported one. The *caution* in §6.3 — never substitute `Cf` for `Σθ` — stands on its own
  and does not need the empirical claim.

---

## 3. "Recovered exactly" is an artifact of grid quantisation

**Severity: medium.**

§5.2 reports the reference grid maximizer recovered exactly for `Nd ≥ 16`, with regret 0.0000.
The study grid is 31×31, so `d₂` spacing is `1/30 = 0.0333`. The `Nd = 8` "error" — (0.200,
0.300) against (0.200, 0.333) — is *exactly one grid interval*, the smallest nonzero value the
metric can report.

The document supplies the counter-evidence itself: the continuous optimum is approximately
(0.212, 0.333) with `U = 2.415`, against the grid's (0.200, 0.333) and `U ≈ 2.407`. So the
"exactly recovered" answer is off by 0.012 in `d₁` **by construction**, and the design-error
metric cannot resolve it.

**Why it does not work.** Design error and regret are both computed against a discretisation
coarser than the effect being measured. "Regret 0.0000" means "matched the grid," not "found the
optimum," and the two differ by more than the reported `Nd = 8` regret of 0.002.

**How it could work.** Optimise the surrogate EIG continuously (the surrogate is cheap and
differentiable) and report design error against the continuous reference. Failing that, refine
the grid until the reference maximizer stops moving, and say what resolution was needed.

---

## 4. Every number is one realization

**Severity: medium.**

§7 states the figures and metrics correspond to *the* `R = 48` realization. Common random
numbers are reused across sample budgets, which correctly removes one variance source between
rows — but leaves the whole table a single draw.

The document half-acknowledges this in §5.4: "at `Nd = 8` ... increasing replication alone
cannot guarantee a smaller global error for one stochastic realization." That caution applies to
the convergence claim too.

**Why it does not work.** The headline result is a *threshold* — the optimum is recovered from
`Nd = 16` onward. With one realization there is no way to distinguish a threshold from a lucky
draw, particularly when the `Nd = 8` miss is one grid step and the regret is 0.002 nats.

**How it could work.** Repeat over 20–50 seeds and report median and interquartile range for
prior-mean RMSE, EIG RMSE and regret. If the threshold is real it will show as a sharp drop in
median regret with a tight spread; if it is luck, the spread at `Nd = 8` and `16` will overlap.
This is cheap — the forward model is analytic.

---

## 5. The prior-mean RMSE saturates, which contradicts the stated interpretation

**Severity: medium.**

| `Nd` | 8 | 16 | 32 | 64 |
|---|---|---|---|---|
| prior-mean RMSE | 0.1227 | 0.0410 | 0.0236 | 0.0214 |

Doubling from 32 to 64 improves it by 9%. §5.1 reads this as "additional unique designs are most
valuable," but the trend says the opposite at the top end: something other than design coverage
is setting a floor near 0.021.

**Candidate causes, none tested.**

- **Replicate noise.** Each training mean carries `Σθ(dᵢ)/48`, so the fitted mean field cannot be
  more accurate than that, however dense the designs.
- **Model misspecification.** `Q = 2` ARD-SE kernels are both infinitely smooth. If `μθ(d)` from
  `eq:15` has a localized feature — the response in `eq:17` has one "centered near `d₁ = 0.2`" —
  a stationary smooth kernel will not represent it, and the residual is a bias no `Nd` removes.

**How to tell them apart.** Vary `R` at fixed `Nd = 64`. If RMSE falls with `R`, the floor is
replicate noise; if it does not, it is model bias, and the fix is a kernel with a shorter
length-scale component or a non-stationary term. Figure 8 compares `R = 4` and `R = 48` but at
small `Nd`, which is the regime where coverage confounds the answer.

---

## 6. The hard object is supplied, not learned — and the report's framing understates it

**Severity: structural.**

§6.2 is explicit and correct: `fitLMCGP.m` does not infer a design-dependent intrinsic
covariance `Σθ(d)`; on this benchmark it is analytically available and supplied at both training
and prediction designs. The LaTeX confirms the fallback is a single spatially uniform,
output-independent noise variance per output (`eq:MOGP_learned_noise_matrix`).

That is the whole difficulty in any real application. The report is presented as an end-to-end
workflow evaluation, but the layer it validates is the surrogate over `μθ(d)` *conditional on*
knowing `Σθ(d)`.

**Why the proposed extension is harder than it looks.** §6.2 suggests replicates can estimate
`Σθ` at observed designs and a second surrogate can interpolate it. Two obstacles:

- A `2×2` covariance from `R` replicates carries relative error `≈ 1/√(2(R−1))` — about 10% at
  `R = 48`, and the estimate must then be *interpolated*, compounding it. The current
  `KnownNoiseCovariance` path supplies `Σθ/R` exactly and accounts for no such error.
- `Σθ(d)` and the roughness of the mean field are only weakly separable from averaged
  observations: a rough `f` and an inflated `Σθ` explain the same training means. With `Σθ`
  supplied this never bites; when estimated, it is a genuine identifiability problem.

**How it could work.** A heteroscedastic multi-output GP with a log-covariance surrogate
(Cholesky-factor outputs to preserve positive-definiteness), fitted jointly with the mean field
rather than in sequence, and validated on a benchmark where `Σθ(d)` is known but *withheld* —
so the cost of not knowing it is measured rather than assumed.

For context on why this matters: on our bubble-collapse records the analogous intrinsic
covariance varies by 124–1316× within a single trace, is dominated by dynamical rather than
instrumental variation, and is not described by the closed form the measurement geometry
predicts. There is no shortcut to it.

---

## 7. Reproducibility: the MATLAB was not run

**Severity: low, but worth resolving before circulation.**

§7 states: "The MATLAB files were statically checked in the current environment; direct MATLAB
execution was not available during report generation."

If the code was not executed, Table 1's numbers were produced by something else. Most likely
they come from an earlier run on another machine, in which case the sentence should say so and
name the environment. As written, a reader cannot tell whether the reported results correspond
to the described implementation.

---

## 8. Smaller points

- **`eq:MOGP_learned_noise_matrix` fallback.** When noise is not supplied the model assumes
  spatially uniform, output-independent noise. Worth stating explicitly that this fallback is
  inappropriate for replicated means with design-dependent `Σθ`, since that is precisely the
  intended use case and a user who omits `KnownNoiseCovariance` gets a silently wrong model.
- **`Q = 2` is asserted, not selected.** No comparison against `Q = 1` (the intrinsic
  coregionalization model, which the LaTeX derives) or `Q = 3`. One marginal-likelihood
  comparison would justify the choice.
- **Absolute continuity in `eq:data_processing`.** The condition `π̂ ≪ π` is stated. Worth noting
  it can fail in practice: a Gaussian surrogate posterior with a smaller variance than the true
  prior in some direction is fine, but a compactly-supported surrogate against a Gaussian prior
  would break it, and `KL[π̂‖π]` is the direction that matters.
- **Terminology.** The report calls `Cf(d)` "surrogate epistemic uncertainty" and `Σθ(d)`
  "intrinsic variability." Standard aleatoric/epistemic language would connect this to the wider
  literature at no cost.

---

## Suggested priority

1. **§1** — fix `B(d) < ∞`, either by compactifying `Θ` or by weakening the Hölder pairing. This
   is the one issue that affects a theorem rather than a demonstration.
2. **§4** — repeat over seeds. Cheap, and it is what turns the convergence claim into evidence.
3. **§2** — either find a regime where `Cf` matters, change the metric, or report it as a null
   result.
4. **§3, §5** — continuous optimisation of the surrogate EIG; an `R`-sweep at fixed `Nd = 64`.
5. **§6** — the heteroscedastic extension, which is the actual research problem.

The theory is the contribution here. §1 is worth fixing carefully because the regret bound is
genuinely useful — we have adopted `B(d)` as a design diagnostic on a separate problem, and the
error decomposition is what let us repair a related bound for likelihood misspecification, where
the data-processing step needs the KL chain rule instead.
