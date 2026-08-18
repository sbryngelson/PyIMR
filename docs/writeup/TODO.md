# TODO — the road from audit to submission, and past it

*From the audit of 2026-08-18. Ordered by what would change the paper, not by effort.
Companion documents: `ISSUES.md` (claims refuted and fixes that failed),
`RETROSPECTIVE.md` (the full record). Every item states what "done" looks like, because
an open item without an acceptance criterion becomes a mood.*

---

## Tier 1 — science that could change the paper

### 1. Get one PAAm record through the vapour-corrected solve

**The paper's largest exposure.** The strongest mundane candidate for the universal curve's
identity is the omitted thermal/vapour physics: it is material-independent and common to the
modelling rather than the chemistry — exactly the properties the universality test selects
for — and over half the common mode's squared norm sits post-collapse in the ringdown, where
thermal damping is the textbook dominant dissipation channel for afterbounces. A
bubble-dynamics referee goes here first.

Current evidence is one record on one side: `gelatin_15C` completes under
`bubtherm/masstrans/vapor` and moves χ² by ~1.2 % (25207 → 24911), evaluated at the
isothermal optimum without refit. The other three probed records exhaust 400 000 integrator
steps — stepsize collapse, not a low ceiling (a stalled solve burns ~20× the steps of a
successful one and still does not finish). No PAAm record integrates, so the cross-material
comparison under vapour cannot currently be made at all.

**Try, in order:** an implicit/stiff diffrax solver (Kvaerno family) in place of the current
one; coarser thermal grid (`Nt` below 7); relaxed `rtol/atol` on the thermal state only;
integrating only through the first collapse plus one afterbounce (the mode's support);
`vapor` on with `masstrans` off, to isolate which term stiffens.

**Done when:** one PAAm record completes, δ̂ is extracted under the corrected model on one
record per material, and the two are correlated. The curve surviving and the curve
dissolving are both results — the second is arguably the better paper.

### 2. Name the alternative hypothesis in the text

`sec:universal` currently frames vapour as a limitation of the test. Restructure it as a
named candidate identity for the curve, with the evidence on both sides stated: for it, the
post-collapse concentration and material independence; against it, the one completed record
where the correction moves χ² by a percent. One paragraph. Independent of item 1, and worth
doing even if item 1 fails — *especially* if item 1 fails, because then the candidacy is
open and must be said to be open.

**Done when:** a referee who believes the curve is thermal physics finds their hypothesis
already stated, with its current evidence, in the section that reports the curve.

### 3. Predict the isothermal error's shape without fitting anything

If the curve *is* the isothermal approximation's error, that error has a computable shape:
simulate a thermally-corrected truth at the fitted parameters on the one geometry that
integrates (`gelatin_15C`), fit the isothermal model to that synthetic trace, and extract
the synthetic δ̂. Correlate it against the measured common mode. No real-data refit needed,
so the stiffness wall of item 1 is partially bypassed. `thermal.py` / `thermal_vapour.py`
already hold most of the machinery.

**Done when:** the synthetic isothermal-error curve is on the common grid with a |cos|
against the measured mode, under the same ensemble null as item 4.

---

## Tier 2 — corrections to the current text (cheap; do before anything ships)

### 4. Put the ensemble-level null into `tab:universal`

The table's headline is a **median over 15 pairs sharing 8 curves**, compared against the
95th point of a **single-pair** null — dependent pairs, uncalibrated comparison, and a
referee can say so. Measured (2026-08-18, `ensemble_null.py`, artifact
`ensemble_null.json`): drawing full ensembles of eight surrogates, the null 95th point for
the across-material median is **0.315**, not 0.508; observed 0.726 has p < 2.5×10⁻⁴. The
error is conservative, and fixing it *strengthens* the weakest cell: within-gelatin 0.551,
hedged in the text as sitting nearly on the pair-level null, has **p ≈ 0.003** at the level
the table quotes. Keep the pair-level statement about the weakest single pair (0.49 against
0.508 — that comparison is at the right level); add the ensemble row or a sentence.

**Done when:** the table or its caption carries the ensemble null, and the "sits ON it"
sentence no longer conflates a pair-level bar with a median-level statistic.

### 5. Reconcile 65–69 % against 47.5 %

`selection.tex:1857` says 65–69 % of each gelatin δ̂ sits in the first collapse.
`universal_curve.json` puts the common mode's first-collapse share at 47.5 %, and
`sec:universal` says over half is post-collapse. Both sentences are in the same document
and a referee gets to quote them side by side. Likely different objects (per-record curves
on their own clocks vs. the common mode on the shared grid) and possibly different window
definitions — but *likely* is not *reconciled*. Compute both numbers under both
definitions, then either bridge them explicitly or correct one.

**Done when:** the two passages cite the same definition or explain the difference in place.

### 6. Mark the tautological row in `sec:deltadesign`

The "performed" geometry's visible share of 0.998 is first-order optimality restated: a
least-squares residual is orthogonal to the model's sensitivity span at the fitted point,
so scoring visibility at (approximately) the fit's own geometry and point must return ≈ 1.
It is 0.998 rather than 1.000 only because `discrepancy_design.py` uses the per-trial
median point and a different grid. The rank correlation (−0.40 / −0.60) counts this forced
point among its four; effective evidence is three points. The section already calls itself
"weak evidence" — it should also say *why the fourth point does not count*, or drop the row
from the correlation.

**Done when:** the row is labelled as by-construction, and the rank correlation is quoted
on the points that carry information.

### 7. Bound the form-shadow control to what it tested

Every counted form in the "not the shadow of the fitted form" control is Zener-family
(SLS is nested in qSLS; Kelvin–Voigt is reported-but-not-counted), and all share the
Keller–Miksis operator and isothermal treatment. The conclusion as worded ("not the
chemistry of either network") is compatible — but "every model in this document" invites
the reading "any model", which item 1 shows is not supportable. One sentence: the control
spans constitutive form; operator-level physics remains inside "common to the modelling of
the experiment", which is where item 2 picks it up.

**Done when:** the claim's scope is explicit in `sec:universal` and the conclusion.

---

## Tier 3 — pre-submission bookkeeping

8. **Bibliography.** Three entries flagged `VERIFY BEFORE SUBMISSION` in `selection.bib`;
   ten uncited entries to cite or remove. Done when the flags are gone and every entry is
   cited.
9. **`trial_variation.py` on PAAm proper.** `PARAMETER_SHARE = 0.393` is a gelatin number
   currently standing in for both materials. Done when PAAm has its own number or the text
   says which records the number covers.
10. **Dictionary collinearity.** `1/R` and `R/R_max` both score high and are strongly
    related; the screen's shares are quoted without a conditioning statement. Compute the
    dictionary Gram's condition number or per-column VIF. Done when the screen's resolution
    limit is stated next to its shares.
11. **Remaining OED-referee findings, unverified:** certificate scope, and the prior
    standardisation (1/√12 vs `sec:prior`). Done when each is either confirmed-and-fixed or
    refuted-with-a-measurement, in `ISSUES.md` either way.

---

## Tier 4 — structural (the author's call, not an agent's)

12. **The editor's ~20-page cut** (`app:background`, `sec:genealogy`). The document is 79
    pages.
13. **The two-papers question.** The title promises the universal curve; roughly half the
    document is OED methodology that neither supports nor draws on it. The audit's honest
    read: the design chapters dilute the discovery paper, and `sec:deltadesign` (item 6) is
    the weakest chapter to carry into review alongside the strongest result. Splitting is a
    real option: *Paper A*, the measured material-independent discrepancy; *Paper B*, the
    design/identifiability machinery, citing A's curve as its test target.

---

## Past the paper — what makes this a programme rather than a manuscript

- **Publish the curve as a target.** `universal_curve.json` (grid, mode, loadings) as a
  first-class data deliverable with a stated protocol: any proposed physics must, simulated
  at these geometries, reproduce this shape. The paper's central object becomes a standing
  challenge rather than a figure.
- **The λ₁ falsification hunt.** The identifiability chapter's claim — every reachable IMR
  geometry sits ~two decades below the Debye peak, so every reported relaxation time is a
  statement about the prior box — is deliberately falsifiable. Invite (or run) the search
  for a geometry the profiled sweep missed: acoustic driving, multi-bubble, or
  non-spherical geometries are outside the sweep and would be the honest counterattack.
- **A second instrument.** The single-instrument, single-laboratory limit is the one the
  text keeps permanently. The decisive control is the same extraction on someone else's
  apparatus: the curve reproducing localises the missing term to the modelling; the curve
  vanishing localises it to the instrument. Either outcome is a second paper. Requires a
  collaborator with raw per-event radii — the Sanchez `.mat` provenance path shows what to
  ask for.
- **A third material.** Two materials rule out chemistry-in-common by having none; a third
  (ideally a fluid, or an elastomer far from both networks) turns "two unrelated materials"
  into "materials are irrelevant", and is cheap once the pipeline exists — the PAAm
  onboarding took one session.
