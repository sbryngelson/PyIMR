# Experimental design for IMR: what we have, what the field has, and what a collaborator can actually run

The design work in `docs/writeup/selection.tex` is mathematically complete and operationally
unusable. This note says precisely why, surveys what the literature offers, and lists the
gaps that stand between the two. The organizing constraint is the real one: a collaborator
running laser-induced cavitation gets tens of usable bubbles, not thousands, and cannot dial
in a design point exactly.

## 1. Where this package currently stands

We compute a **continuous design measure**. `pyimr.measure` maximizes `log det M(xi)` over
probability measures on a grid of candidate settings, and the Kiefer--Wolfowitz equivalence
theorem certifies the answer: `max_x d(x, xi*) - p = 9.7e-10` on three support points. That
certificate is real and is stronger than anything else in this document -- it proves global
optimality rather than reporting convergence.

It is also the wrong object to hand an experimentalist, for one structural reason.

**A measure assigns weights; an experiment runs integers.** The certified optimum splits runs
three ways -- roughly 0.45 / 0.31 / 0.24. With `N = 12` bubbles that is 5.4 / 3.7 / 2.9, and
somebody has to choose 5/4/3 or 6/4/2. The certificate says nothing about which, and says
nothing about how much is lost by rounding. This is the *approximate-to-exact* gap, and at the
sample sizes IMR actually reaches it is not a rounding detail: it is the difference between a
theorem and a protocol.

**CLOSED.** `pyimr.measure.apportion` implements Pukelsheim--Rieder efficient rounding and
reports the D-efficiency achieved rather than asserting one, because the rule carries no
guarantee at small budgets. On cubic regression, where everything is closed form: budgets
divisible by four lose nothing, six runs give 2/2/1/1 at efficiency 0.9428 exactly, and at
weights 0.7/0.2/0.1 with four runs it keeps the support point that proportional rounding drops
-- which would otherwise leave the information matrix singular.

Three further gaps, each specific:

- **The design space is not a box we control.** Our answer wants `R_max` near 60 um at high
  stretch and near 1200 um at low stretch. Laser cavitation does not have `R_max` and stretch
  as independent setpoints; laser energy and focus produce a *joint distribution* over what
  is achieved, with scatter. A design specified as a point in `(R_max, lambda)` is a request
  the apparatus cannot fill exactly.
- **The information is evaluated at a point estimate of `theta`.** The design is locally
  optimal in the parameters while globally optimal in the measure. Given that
  `docs/writeup` finds the shear modulus unidentified along a ridge, "locally optimal at the
  fitted point" is a weaker statement here than it usually is.
- **`R_eq` is a nuisance parameter and is not treated as one.** `req_prior.py` finds the fit
  wants `R_eq` 3--11% larger than inferred. A design that maximizes information about
  `(g, mu, lambda_1, alpha)` while `R_eq` is quietly also unknown is optimizing the wrong
  matrix.

## 2. The literature, in three families

### 2a. Classical optimal design theory

Kiefer--Wolfowitz equivalence, the multiplicative algorithm (Silvey--Titterington--Torsney),
Fedorov--Wynn vertex-direction methods. This is what we already use, and it is mature. Two
parts of it we have not used:

- **Efficient rounding / apportionment.** Pukelsheim & Rieder (1992, *Biometrika*) give an
  apportionment method for turning an approximate design into an exact `N`-run design with an
  efficiency bound. The literature is explicit that these methods "usually work well for large
  enough `n` but with no guarantee for small sample size" -- which is exactly our regime, and
  is the reason this cannot be adopted uncritically.
- **Exact design algorithms.** For small `N` the honest approach is to optimize over exact
  designs directly: Fedorov exchange, the Meyer--Nachtsheim coordinate-exchange algorithm, or
  -- newer, and giving certified optima -- mixed-integer second-order cone programming
  (Sagnol & Harman, *Annals of Statistics* 2015). The MISOCP route is attractive here because
  it preserves the one property our current answer has that the alternatives do not: a
  certificate.

### 2b. Bayesian optimal experimental design (BOED)

Maximize the expected information gain, `EIG(d) = E_Y[ KL(posterior || prior) ]`. This is where
the field's activity is, and where our own group already works.

**The closest prior art is this group's own**: *An optimal sequential experimental design
approach ... for bubble cavitation* (`~/fastscratch/papers/paper_boed_cavitation`). It runs
**sequential** BOED with Bayesian optimization over the design space and En4D-Var for
inference, and reports 1% relative error and >99% correct model probability within 10
sequential designs.

**That result is on synthetic data, and the distinction is the whole point of this section.**
The synthetic measurements are generated from the forward model itself with additive noise
`sigma = |R*-1|/50 + t*/160`. Two things follow by construction: the generating model is in
the candidate set, and the residuals are independent. Neither holds on the records analyzed in
`docs/writeup`, and we have measured how badly:

- The fitted residual has lag-one autocorrelation **0.918**, so 201 samples carry the
  information of about **10** independent ones. Any EIG computed from an independent
  likelihood therefore overstates what an experiment delivers by roughly **25x**.
- Whitening by the hierarchical trial covariance removes only a third of that correlation.
  The rest is **model-form error** -- structure the model does not contain -- which synthetic
  data drawn from the model cannot exhibit at all.
- `req_prior.py` finds a systematic 3--11% offset in `R_eq`, and the operator margins collapse
  once it floats. A synthetic study with a known truth and a prior near it cannot see this.

So "1% in 10 experiments" is a validation of the machinery, which is a necessary first step
and was worth doing. It is **not** a forecast for a real collaborator, and should not be
quoted to one as a budget. It is a floor, optimistic in two independent ways: the noise model
and the forward model. EIG is an expectation under the model's own predictive distribution --
when the model is wrong, it prices information about a process that did not generate the data.

The paper's own stated limitations are the right starting point for anything new:
- En4D-Var assumes the material properties are multivariate normal, and degrades when they are
  not. MCMC fixes that at a sample cost (~1e4) far above the `N_En = 48` ensemble used.
- The approach needs the model set as a prior. If every candidate is inadequate, model
  probabilities rank them against each other and say nothing about the gap to reality. Our
  lag-one measurement says that gap is the dominant term on real records.

Current SOTA beyond that, from the 2024--2026 literature:
- **Tractable EIG bounds.** Nested Monte Carlo is expensive and needs an explicit likelihood.
  Variational and contrastive lower bounds (PCE/NCE-style) replace it with an expectation of a
  tractable integrand that tightens with simulation budget.
- **Amortized / policy-based sequential design.** Deep adaptive design and policy-gradient
  reinforcement learning train a *policy* mapping observed history to the next design, so the
  per-experiment cost at run time is a forward pass rather than an optimization. This is the
  most relevant recent direction for an experimentalist standing at the bench between shots.
- **Goal-oriented OED.** Design for a specific quantity of interest rather than the full
  parameter vector -- the Bayesian analogue of `D_s`-optimality.
- **Design robust to misspecification.** Directly aimed at the problem above. Robust EIG over
  a KL ambiguity set around the prior reduces to a log-sum-exp stabilization of the EIG samples
  and is cheap to implement; generalised (Gibbs) BOED replaces the likelihood with a loss and
  gets robustness in design and inference together; integral-probability-metric utilities are
  more stable than EIG under surrogate and prior error. This is the part of the 2024--2026
  literature that speaks directly to our situation, and none of it is in the package.

### 2c. Design under model uncertainty

Our situation exactly: three uncertain model axes (constitutive, operator, thermal).
- **Compound criteria** (Atkinson; Cook & Wong) combine discrimination and estimation in one
  concave objective, which is what `selection.tex` already derives but never ran on the
  records.
- **Robust / worst-case design** over a parameter set, rather than at a point estimate,
  addresses the local-optimality gap above.
- **Model-averaged design** weights the information matrix by model posterior probabilities.

### 2d. Four literatures we had not looked at, three of which change a recommendation

**Sampling-time design.** A whole branch of OED asks *when to measure*, not only what to set
up: optimal sampling instants for ODE models, chosen so a fixed number of measurements carries
the most information. IMR's camera frame rate and observation window are design variables and
we have never treated them as such. Our own diagnostics say there is enormous slack here:
`N_eff ~ 10` of 201 samples means the time series is roughly **20x redundant**. The question
is not "can we sample faster" but whether frames should be traded for bubbles, for a longer
window, or for signal-to-noise.

**Design under correlated errors.** The classical equivalence theory -- our certificate --
assumes independent observations. There is an extension of it to dependent data, and an
Ornstein--Uhlenbeck process is the continuous-time analogue of AR(1), which is exactly the
structure our lag-one 0.918 residual has. So the certificate is not merely optimistic under
correlation; it is answering a different problem. This is the theoretical counterpart to gap 0
below.

**Replication versus distinct settings, and the lack-of-fit test.** Each record here is a set
of repeated bubbles -- 18, 14 and 7, not the uniform 18 we had assumed -- and we already use
their spread as the noise scale. That is textbook *pure
error*, and the classical apparatus built on it is the **lack-of-fit F-test**: it separates
model inadequacy from measurement error using exactly the ingredients we have. `selection.tex`
spends several pages establishing "the residual is model-form error, not noise" via lag-one
autocorrelation and hierarchical whitening, and never runs the standard test for that claim.
The caveat is real -- the classical test assumes independent errors, which we do not have --
but the framing is directly ours, and replication versus spread is itself a design decision we
have never posed: 18 bubbles at 3 settings, or 6 at 9?

**Sloppy models and the limits of optimal design.** This is the one that should change what we
recommend. The literature on sloppy systems reports that models may fit *worse* on data from
their own optimal experiments, with less predictive power after optimal selection than before,
because the optimal design pushes into the regime where the model's inadequacy is exposed. Our
situation is precisely the setup for that failure: the `g`--`alpha` ridge is textbook
sloppiness, our design wants `R_max` near 60 um at stretch 20 -- far more violent than anything
performed -- and we have already measured that the model is inadequate on the gentler records
we have. **Designing hard against the ridge may buy an identifiable fit to a model that then
describes the data less well.** That is a testable prediction and it argues for a staged
design rather than a jump to the optimum.

Worth noting alongside it: sloppiness and unidentifiability are distinct. A model can be sloppy
-- FIM eigenvalues spread over decades -- and still identifiable. `selection.tex` moves between
the two more freely than that distinction allows.

**Batch design.** Collaborators run batches, not one bubble at a time with a re-fit between.
Fully joint batch optimization does not scale, but if the criterion is submodular and
non-decreasing, greedy batch selection is within a factor `1 - 1/e ~ 0.63` of the joint
optimum. That is a usable guarantee and a realistic protocol.

**Amortized sequential design.** iDAD trains a design policy up front and then makes design
decisions in **milliseconds** at the bench, and -- the part that matters for us -- it trains on
any model that produces *differentiable simulated samples*. PyIMR is JAX-differentiable
end to end, so that prerequisite is already met. Step-DAD (2025) refines the policy as data
arrives, which is the semi-amortized middle ground between our static measure and full
sequential BOED.

## 3. What the collaborator actually needs

Stated as deliverables rather than criteria:

1. **A table of `N` settings**, integers, for the `N` they can afford -- not weights.
2. **The efficiency they give up** by running that table instead of the ideal measure, and
   instead of `N -> infinity`.
3. **Tolerance**: how much does the answer degrade if the achieved `R_max` misses the target
   by the scatter their apparatus actually has? A design at a sharp optimum that collapses
   under 20% scatter is worse than a flat one nearby.
4. **A stopping rule**: after `k` experiments, is the question answered? Sequential BOED gives
   this naturally; a static design does not.
5. **What the design is for.** Our own measurements say the three axes are not equally worth
   designing against: the performed geometry is already 68.4% efficient for the operator axis
   but 2.4% and 4.2% for constitutive and thermal. And `req_prior.py` says two of three
   records never discriminated operators at all once `R_eq` floats. Designing hard against the
   operator axis is optimizing the thing we are least able to improve and least able to
   resolve.

## 4. Gaps, ranked by what they would change

0. **The likelihood assumes independence, and it is wrong by about 25x.** With
   `N_eff ~ 10` of 201, every EIG, every evidence and every "how many experiments do we need"
   in this package and in the sequential pipeline is inflated by the same factor. This is
   already item 2 of `open-work.md`; it is listed first here because it is the one defect that
   corrupts the static and sequential answers identically, and because a collaborator planning
   a campaign is precisely the person who gets hurt by an optimistic count.

1. **Exact designs for small `N`.** Convert the certified measure to an `N`-run table, and
   report the efficiency lost. Efficient rounding first because it is cheap; MISOCP if a
   certificate is wanted at the exact-design level. **This is the gap between our result and a
   protocol.**
2. **`R_eq` as a nuisance parameter in the design.** We now know it is not exact and by how
   much. `D_s`-optimality over `(g, mu, lambda_1, alpha)` with `R_eq` marginalized is a small
   change to the information matrix and changes which experiment is best.
3. **Design under implementation error.** Optimize the *expected* information over the
   achieved-setting distribution rather than at the requested point. Needs one number from the
   collaborators: the scatter in `R_max` and stretch at fixed laser settings.
4. **Run the compound criterion.** Already derived in `selection.tex`, never run, because the
   per-design log Bayes factor still suffers boundary pinning. That blocker is now better
   understood.
5. **Connect to the group's sequential pipeline.** The static measure and the sequential BOED
   answer different questions -- "what should we run" before any data, versus "what next" after
   each shot. The honest comparison is whether the certified static design beats 10 sequential
   ones, and that has not been measured -- on real records, where both are exposed to the model
   error the synthetic study could not show.

6. **The lack-of-fit test: RUN, and it confirms the claim by an independent route.** `F` is
   14.2 at 15 C, 2.9 at 23 C and 3.3 at 33 C against a critical value near 1.2, and 23.3 / 4.8 /
   5.4 once the 39.3% of trial variance that is bubble-to-bubble parameter spread is divided
   out. It also separates the records, which lag-one does not: the three lag-one values are
   effectively identical (0.919 / 0.968 / 0.911), but 15 C repeats itself three times more
   tightly, so its inadequacy stands 14 times above its own scatter where the others stand 3.
   The strongest evidence the model is incomplete comes from the record with the best
   apparatus.

7. **Treat sampling times as a design variable.** `N_eff ~ 10` of 201 says the time series is
   about 20x redundant. Frames are cheap but not free, and the trade against bubble count,
   window length and SNR has never been posed.

8. **The sloppy-model warning: TESTED under two mismatches, and only partly confirmed.** Cold
   one-mode qSLS fitted to a thermal truth and to a two-mode truth, at four geometries. Running
   two was the point: with one, a property of the missing physics is indistinguishable from a
   property of the design.

   ROBUST: the discrimination-optimal geometry is worse on BOTH counts under BOTH mismatches --
   fits 1.8x and 2.1x worse than the performed geometry, recovers `g*alpha` 9.8x and 12.2x
   worse. A design chosen to separate constitutive models degrades the parameter it was meant
   to measure, whichever physics is missing.

   NOT ROBUST, and this corrects an earlier claim of mine: under thermal error the E-optimal
   geometry loses on both counts, but under constitutive error it is indistinguishable from the
   performed one (0.97, 0.98). The 60 um support point reverses harder -- least model error
   under thermal truth, 8.7x worse `g*alpha` recovery under two-mode truth.

   So the conclusion is narrower than "optimal designs are worse", and for a collaborator it is
   worse than that: **which design is safe depends on which physics is missing, and we do not
   know which is missing.** The lack-of-fit test says the model is inadequate on every record;
   it does not say in what direction. A design cannot be certified robust without naming the
   model error it is robust to, and no criterion in this package takes model error as an
   argument at all.

   The mechanism generalizes even though the ranking does not: a criterion computed under the
   correct-model assumption sends the experiment where the model is most SENSITIVE, which is
   often where it is most WRONG, and the fitted parameters absorb the difference.

9. **Frames: measured, and there is an enormous amount of slack.** The D-optimal measure over
   the 201 candidate times is certified on SIX support points, and 50 observations placed on
   them carry more information than all 201 uniform frames. That is slack, not a
   recommendation -- six distinct times leaves no degrees of freedom to notice the model is
   wrong, which is precisely what the lack-of-fit test consumes. The usable conclusion is that
   frames are cheap relative to their information and should be traded for something scarce:
   bubbles, window length, or the replication `F` needs.

## 5. The one thing worth saying to a collaborator now

Not "run 60 um at stretch 20". Rather: the experiments already performed are near-optimal for
telling bubble-dynamics models apart and 24--41 times off for telling constitutive and thermal
models apart, so the geometry should move only if the constitutive or thermal question is the
one they care about. And whatever they run, `R_eq` should be reported with its uncertainty
rather than as a measured constant, because the fit says it is wrong by 3--11% and that error
is worth as much as the entire operator question.
