# TODO — the road from audit to submission, and past it

*Opened from the audit of 2026-08-18 and updated the same day as items closed. Ordered by
what would change the paper, not by effort. Companion documents: `ISSUES.md` (claims refuted
and fixes that failed), `RETROSPECTIVE.md` (the full record). Every item states what "done"
looks like, because an open item without an acceptance criterion becomes a mood.*

---

## The state of play, in one paragraph

The audit found the paper's statistics conservative rather than fragile — the universality
null was quoted at the wrong level in the paper's own disfavour. What it did **not** anticipate
is that the whole vapour question rested on a solver setting. `Nt = 7`, chosen once for speed
against a package default of 25, was the sole reason the corrected model looked un-integrable,
and it was also the reason the correction looked harmless. Both of those statements are now
withdrawn from the text. Correcting them opened the real experiment, and the real experiment
is the one live threat to the paper's central claim.

---

## Tier 0 — LIVE: the claim is under test

### 1. Does the cross-material curve survive the vapour correction? *(control running)*

**This is now the paper's decisive open question, and the first answer went against it.**
With `Nt = 25`, all eight records refit under `bubtherm`/`masstrans`/`vapor`
(`delta_corrected.py`, artifact `delta_corrected_vapour.json`). The cross-material median
|cos| falls from **0.726 to 0.302**, with 5 of 15 pairs above the constrained null of 0.458.
Within-material holds up better at 0.564. Same-record corrected-against-uncorrected is 0.414,
so the corrected δ̂ is a substantially different curve.

**Do not write this into the paper yet.** The result carries its own alarm: the corrected fits
come back *worse* than the isothermal ones, at χ²/N up to 3.66 against ~1.3, and a fit that
degrades when the physics gets more complete is the signature of a search that did not
converge. Two mechanisms would produce exactly that — the box spans ×3 per axis around a
centre whose thermal objective is an order of magnitude worse, so the optimum may lie outside
it, and 3 starts × 100 evaluations is short for a surface that moved that far.

`vapour_convergence.py` is the control: the same fit at box widths ×3 and ×10, reporting the
fitted parameters and flagging any axis within 2 % of a wall, then recomputing the
universality statistics at each width. **Done when** either (a) both widths agree, nothing sits
on a wall, and the collapse is a property of the corrected model — in which case
`sec:universal` needs restating and that is the paper's biggest revision; or (b) the wide box
finds materially better χ²/N, in which case the first answer was search-limited and the
experiment must be rerun at a budget that converges.

If (b), the follow-on is a genuinely global corrected fit rather than a local refinement — the
warm-box shortcut exists only because the cold multistart was intractable at `Nt = 7`, and
that premise is now known to be false.

### 2. Reconcile the two thermal results against each other

They currently point opposite ways and both are measured, so the resolution is physics, not
arithmetic. `thermal_signature.py` finds the correction's *own shape* is **not** the universal
curve — median |cos| 0.300 against a per-record null of 0.367, nowhere near 0.726 — while
being 2–4× the size of the discrepancy it would explain. Yet refitting under that same
correction collapses the cross-material agreement. Both can hold: a large correction pointing
somewhere else still moves the fitted parameters a long way, and δ̂ is defined *at* the fitted
point, so the curve can be destroyed by a correction whose own shape it does not share. That
reading needs stating and testing rather than asserting.

Note the honest residue in the signature result: 3 of 8 records align above their own null
where 0.4 would be expected, so a minority component of the correction *does* resemble the
curve. The candidate is reduced, not eliminated.

**Done when** one paragraph in `sec:vapour` states which of the two the paper is claiming and
why the other is consistent with it.

---

## Tier 1 — closed today

- ~~**The vapour stall.**~~ Not stiffness, not a step ceiling, not cost. `Nt = 7` gave 2/8
  records integrating; `Nt = 15` and `Nt = 25` both give 8/8, and the stall is *slower* than
  the success it stood in for (250 s of collapsing stepsize against 35 s of solve). An
  implicit Kvaerno5 was in use throughout, which is why raising `max_steps` was never going to
  help. `thermal_gate.py`.
- ~~**"Vapour barely moves the objective."**~~ Same artifact. `gelatin_15C` goes 25207 → 24911
  at `Nt = 7` and 25207 → **80507** at `Nt = 25`. The correction is large; it looked small only
  where it was unconverged. Withdrawn from the text.
- ~~**The ensemble-level null.**~~ `tab:universal` compared a median over 15 *dependent* pairs
  against a single-pair null. The correct 95th point is 0.315, not 0.508; observed 0.726 sits
  beyond all 4000 draws. The error was conservative, and within-gelatin — the cell the text
  hedged — is p ≈ 0.003. `ensemble_null.py`.
- ~~**65–69 % in the first collapse.**~~ Described one gelatin record. Measured: 46 %, 69 %,
  47 % for the three; 41–69 % across all eight; common mode 47.5 %. Corrected, not reconciled.
  `first_collapse_share.py`.
- ~~**The 0.998 visible share.**~~ First-order optimality restated, not evidence; its
  four-point rank correlation carries three points. Both now said in `sec:deltadesign`, whose
  stale 0.05 null is also brought into line with `sec:transfer`'s correction.
- ~~**Form-shadow scope.**~~ The control spans constitutive form only; every fit shares a
  Keller–Miksis operator and an isothermal gas. Stated, and handed to `sec:vapour`.
- ~~**Dictionary collinearity.**~~ Worse than suspected: the polytropic gas term and the
  internal pressure are the *same direction* to numerical precision (|cos| = 1.000 on 8/8 —
  one is the other times a constant), so the dictionary has ten entries and nine directions
  and a singular Gram. The two terms the headline reading is taken from overlap at 0.88. The
  subspace comparison survives; naming a single term does not. `dictionary_collinearity.py`.
- ~~**Bibliography.**~~ The three `VERIFY BEFORE SUBMISSION` entries verified against
  publishers' records (all fields correct as written, DOIs added, attributed claims re-read).
  The eleven uncited entries placed rather than deleted — they are the canonical citations for
  machinery the document uses. 35 entries, 0 uncited, 0 undefined references.
- ~~**`eq:lackoffit`.**~~ Referenced eight times, defined zero times, rendering as "??".
  Defined where first used.

---

## Tier 2 — remaining bookkeeping

3. **`trial_variation.py` on PAAm proper.** `PARAMETER_SHARE = 0.393` is a gelatin number
   currently standing in for both materials. Done when PAAm has its own number, or the text
   says which records the number covers.
4. **Two OED-referee findings still unverified:** certificate scope, and the prior
   standardisation (1/√12 vs `sec:prior`). Done when each is confirmed-and-fixed or
   refuted-with-a-measurement, recorded in `ISSUES.md` either way.

---

## Tier 3 — structural (the author's call, not an agent's)

5. **The editor's ~20-page cut** (`app:background`, `sec:genealogy`). The document is 81 pages.
6. **The two-papers question.** The title promises the universal curve; roughly half the
   document is OED methodology that neither supports nor draws on it. The audit's read: the
   design chapters dilute the discovery paper, and `sec:deltadesign` is the weakest chapter to
   carry into review beside the strongest result. *Paper A*, the measured material-independent
   discrepancy; *Paper B*, the design and identifiability machinery, citing A's curve as its
   test target. Item 1's outcome bears directly on this — if the curve does not survive
   correction, Paper A's thesis changes and the split question changes with it.

---

## Past the paper — what makes this a programme rather than a manuscript

- **Publish the curve as a target.** `universal_curve.json` as a first-class deliverable with a
  stated protocol: any proposed physics must, simulated at these geometries, reproduce this
  shape. Conditional on item 1 — publish whichever curve survives.
- **The λ₁ falsification hunt.** The claim that every reachable IMR geometry sits two decades
  below the Debye peak is deliberately falsifiable. Acoustic driving, multi-bubble, and
  non-spherical geometries lie outside the profiled sweep and are the honest counterattack.
- **A second instrument.** The single-laboratory limit is the one the text keeps permanently.
  The curve reproducing on another rig localises the missing term to the modelling; the curve
  vanishing localises it to the instrument. Either is a second paper.
- **A third material.** Two materials rule out chemistry-in-common by having none; a third —
  ideally a fluid, or an elastomer far from both networks — turns "two unrelated materials"
  into "materials are irrelevant", and is cheap now the pipeline exists.
- **Audit the other solver defaults.** `Nt = 7` was wrong for a year and cost a false
  conclusion in the text. It was found by asking what the default was. Nothing guarantees it is
  the only such setting; the thermal grid, `rtol`, `max_steps` and the spectral quadrature
  orders all deserve one convergence sweep each, run once and recorded.
