# Capability parity plan

Goal: match or exceed IMRv2 and IMR-vanilla across the full solver capability
surface, with every claim backed by a pinned reference trajectory rather than
internal self-consistency.

Reference trees:

| tree | path | revision |
|---|---|---|
| IMRv2 | `~/research/docs/_ideas-boed/upstream/IMRv2` | `dea31cd` |
| IMR-vanilla / IMR-old | `~/research/solvers/IMR` | `9bdd08d` |

Ordering rationale: W1 measures the size of W2, so it comes first. W0 is
independent and cheap. W3--W6 are independent of each other.

## Progress

| workstream | status |
|---|---|
| W0 API hygiene | **done** |
| W1 validation coverage | **done** -- outcome reshaped W2, W3, W4 |
| W2 spectral quadrature | **done** -- reframed; default now Gauss-Legendre |
| W3 viscosity parity | **done** -- rescoped to original work |
| W4 radial 6 | **done** -- resolved with a measurement |
| W5 IMR-vanilla estimators | **done** -- new `imr_data.py` |
| W6 solver control surface | **done** -- 86-option audit, `max_step_s` added |
| W7 tinygrad style | **done** -- config, indent, docstrings, six splits, formatter settled |
| W8 measured collapse gaps | **done** -- stubs, plus the Zener offset quantified |
| W9 fix upstream defects here | **done** -- Mie-Gruneisen root; `radial = 6` now works |
| W10 thermal collocation | **done** -- spectral(25) matches fd(400); no plateau |

Revised priority after W1: **W8 -> W3 -> W5 -> W6 -> W2 -> W7.**

---

## W0. API hygiene

Small, independent, no physics risk.

- [x] Add `__all__` to `pyimr.py` (42 names). It previously re-exported
      `np`, `solve_ivp`, `brentq`, `csr_matrix`, `lil_matrix`, `dataclass`,
      `field`, `PchipInterpolator`, `perf_counter`, `Mapping`,
      `MappingProxyType`, `Integral`, and `annotations`.
- [x] Add `__all__` to `imr_sensitivity.py` (4) and `imr_inference.py` (7).
- [x] Add regression tests in `tests/test_api.py`:
      `test_all_is_declared_sorted_and_defined` and
      `test_star_import_leaks_no_foreign_names`. The latter walks the
      star-import namespace and rejects modules, foreign `__module__` owners,
      and union aliases whose members are not repo-owned.

**Done when:** `dir(pyimr)` minus dunders equals the documented API surface.

**Status: done.** 22 passed, ruff clean, `tests/run_validation.py` fully green.
Note: `imr_inference` imports `_normalize_parameters` from `imr_sensitivity` --
a private cross-module import. Left as-is; fold into W7.

---

## W1. Close validation coverage

The physics added in PRs #5 and #6 is the least validated in the repo. Existing
pinned CSVs in `tests/` cover qKV, Oldroyd-B, KM/NHKV, KM/Zener, all forcing
types, vapor, bubtherm, medtherm, masstrans, qZener, and radial 3/4/5. There is
**no pinned upstream trajectory** for the models below; they are currently
checked only against their own reduction limits.

Requires MATLAB to run IMRv2. If unavailable locally, the deliverable for each
item is the case-generation script plus harness wiring, with CSV drop-in later.

- [x] Write `tools/gen_imrv2_cases.m` driving `f_imr_fd` over the case list and
      dumping CSVs matching the `tests/ref_*.csv` convention. Also
      `tools/probe_viscosity.m` for reduction-limit diagnostics.
- [x] **Giesekus / linear PTT: cannot be generated.** `f_call_params.m:376-385`
      dispatches `stress` 6 (PTT) and 7 (Giesekus) and forces `spectral = 1`
      for both, but the input gate at `f_call_params.m:206-208` rejects
      `stress > 5`. Both raise `INPUT ERROR: stress must be between 0 to 5`.
      They are dead code in IMRv2 at `dea31cd`. **No upstream reference exists
      or can exist.**
- [x] **Non-Newtonian viscosity: only `nu_model = 1` runs.** Generated
      `tests/ref_numodel1.csv`. Models 2--7 all raise; see defect list below.
- [x] Pin **collapse-initialization** trajectories. IMRv2 requires the full
      coupled model (`f_call_params.m:234-236`: bubtherm, medtherm, masstrans
      and vapor all 1). Generated `ref_collapse_zener.csv`,
      `ref_collapse_oldb.csv`, `ref_collapse_nhkv.csv`.
- [x] Hyperelastic laws: **moot.** IMRv2 expresses only neo-Hookean (`stress=1`)
      and quadratic KV (`stress=2`), both already pinned. The other ten laws in
      PyIMR have no upstream counterpart to pin against.
- [x] Extend `tests/run_validation.py` -- new section `1c. COLLAPSE
      INITIALIZATION vs IMRv2`, plus a `KNOWN GAPS` summary block so
      discrepancies are reported rather than hidden behind a passing suite.

### Measured results

| case | max abs dR/R0 | verdict |
|---|---:|---|
| collapse Zener | 1.55e-03 | pass, but 100x looser than the 1e-5 band |
| coupled Oldroyd-B (no precursor) | 1.56e-05 | pass -- see W8, upstream stubs collapse here |
| coupled NHKV (no precursor) | 6.22e-06 | pass -- same |

`ref_numodel1.csv` is deliberately **not** wired in as a passing target: IMRv2's
own Carreau model fails its own Newtonian reduction (see defect 3), so pinning
to it would pin to a bug. The CSV is retained as evidence.

### Consequences for the rest of the plan

- **W2 collapses in scope.** Its premise was matching IMRv2's spectral
  reference for Giesekus/PTT. That reference is unreachable, so PyIMR is not
  behind on those models -- it implements capability upstream advertises but
  cannot run. The 1e-3 reduction errors are PyIMR FD versus its own
  Oldroyd-B limit, not a deficit against a reference. Spectral becomes an
  optional accuracy improvement, not a parity requirement.
- **W3 changes character.** Only Carreau works upstream, and incorrectly.
  PyIMR already exceeds IMRv2 here. Any Powell-Eyring work must be built on
  correct formulas, not ported ones.
- **W4 is resolved.** See below.
- **New work surfaced:** the collapse Oldroyd-B discrepancy and the memoryless
  collapse restriction, tracked in W8 -- where both turned out to be upstream
  stubs rather than PyIMR defects.

---

## W2. Spectral collocation path

The largest structural gap. IMRv2 ships `src/f_imr_spectral.m` (542 lines,
Chebyshev collocation for the thermal and viscoelastic PDEs) alongside
`src/f_imr_fd.m` (585 lines). Critically, `src/f_call_params.m:376--385`
forces `spectral = 1` for `stress == 6` (PTT) and `stress == 7` (Giesekus)
unconditionally — there is no upstream FD path for those models.

PyIMR solves them on a 480-point FD Lagrangian grid
(`Giesekus.points=480, extent=60.0`). This is a different discretization from
the reference, not a tolerance difference.

- [ ] Size the work from W1's measured deviations before starting.
- [ ] Implement Chebyshev collocation for the viscoelastic stress PDE.
- [ ] Implement Chebyshev collocation for the bubble and medium thermal PDEs
      (`bubtherm` / `medtherm`).
- [ ] Expose discretization choice on the config (default preserving current
      behaviour; no silent change to pinned trajectories).
- [ ] Verify spectral Giesekus/PTT against the W1 pinned trajectories in the
      1e-5 band.
- [ ] Confirm FD and spectral agree with each other under refinement.
- [ ] Extend forward sensitivities to the spectral path, or document it as
      value-only and raise a structured error on request.
- [ ] Benchmark spectral vs FD; record in the README.

**Done when:** Giesekus and PTT match pinned IMRv2 trajectories at the same
tolerance as the existing closed-form models.

**Risk:** sensitivity support for the spectral path may be a second increment.
Decide explicitly rather than by omission.

### REVISED after W1 -- demoted from blocking to optional

The stated "done when" above is **unachievable and no longer the point**:
IMRv2 cannot run Giesekus or PTT at all (upstream defect 1), so there is no
pinned trajectory to match and no parity deficit to close. PyIMR is ahead
here, not behind.

What survives as genuine motivation:

- Spectral collocation converges faster than the 480-point FD Lagrangian grid,
  so it is an accuracy and cost improvement on PyIMR's own terms.
- A second independent discretization is the strongest available check on the
  distributed models, precisely *because* no upstream reference exists.

Reframed acceptance: **spectral and FD agree with each other under refinement,
and spectral reaches the same accuracy in fewer points.** Do this only after
W8, which addresses measured discrepancies rather than hypothetical ones.

---

## W3. Viscosity model parity

IMRv2 `src/f_viscosity.m` dispatches seven models: Carreau, Carreau-Yasuda,
Cross, modified Cross, Powell-Eyring, modified Powell-Eyring, simplified Cross.

PyIMR has Newtonian, PowerLaw, CarreauYasuda, Cross, HerschelBulkley,
Bingham. PowerLaw / HerschelBulkley / Bingham exceed upstream.

- [ ] Add `PowelEyring` — genuinely absent, no parameter case reaches it.
- [ ] Add `ModifiedPowellEyring` — likewise.
- [ ] Add `ModifiedCross` and `SimplifiedCross` as named laws, or document the
      exact `Cross` parameter cases that reproduce them.
- [ ] Add `Carreau` as a named law, or document that it is
      `CarreauYasuda(transition_exponent=2)`.
- [ ] Analytic viscosity tangents for each new law, checked against centered
      differences at the same 2.4e-11 standard as the existing suite.
- [ ] Validate each against the W1 `nu_model` trajectories.
- [ ] Update the README law list.

**Done when:** every IMRv2 `nu_model` is reachable by name and pinned.

### REVISED after W1 -- this is now original work, not porting

Six of the seven upstream models do not run (defects 2 and 3), and the seventh
is quantitatively wrong (defect 4). "Pinning" against them is impossible and
matching them would be a mistake. Revised tasks:

- [x] Implement `PowellEyring` from the **correct** constitutive law,
      `eta = eta_inf + (eta_0 - eta_inf)*asinh(lambda*gdot)/(lambda*gdot)`.
      IMRv2's `f_viscosity.m:143` uses `sinh`, not `asinh` -- not copied.
- [x] Implement `ModifiedPowellEyring` as `log1p(x)/x`. IMRv2 generalizes the
      denominator to `x^nc`, which has no finite zero-shear limit unless
      `nc == 1` -- not copied.
- [x] Document `Carreau` as `CarreauYasuda(transition_exponent=2)` and
      simplified Cross as `Cross(transition_exponent=1)` in the README rather
      than adding redundant classes.
- [x] Analytic viscosity tangents for both, wired through all three paths:
      `_viscosity_and_tangent` (NumPy + `Dual`), `_imr_mechanical._viscosity`
      (numba codes 7 and 8), and `imr_sensitivity._material_parameters`.
      Suite-wide analytic-vs-centered-difference error **2.74e-11**, holding
      the existing 2.4e-11 standard.
- [x] Added `arcsinh` and `log1p` to `Dual` in `_imr_autodiff.py`; the generic
      forward-mode path needs both and had neither.
- [x] Validate by reduction limit: both laws hit **0.00e+00** against
      `Newtonian(eta_0)` at `lambda = 0`. Forward sensitivities agree with
      centered differences at 7.4e-06 to 8.0e-05, matching the CarreauYasuda
      control (2.1e-05).

**Status: done.**

**Trap worth recording:** `mechanical_tangent_rhs` differentiates the compiled
RHS by **complex step** (`+1j*1e-30*tangent`, read back as `.imag/step`). Every
operation on the value path must therefore stay analytic. A first cut using
`abs(time_constant*shear_rate)` produced a sensitivity error of 2.06e-01 that
was *independent of the finite-difference step* -- the signature of a broken
derivative rather than FD noise, since `abs()` discards the imaginary
perturbation. The fix is `sqrt(x*x)` for the analytic `|x|`, with `abs()`
confined to the branch predicate. `log1p`/`arcsinh` are also spelled via
`log` so numba accepts complex input. Any future viscous law must follow this.

---

## W4. Radial equation 6

IMRv2 `src/f_radial_eq.m` branches 1--6. PyIMR supports 1--5. The README
asserts the upstream branch produces non-real states in the tested pressure
range; that assertion is currently untested.

- [x] Run IMRv2 `radial = 6` and record whether it produces non-real states.
      **It does.** On the standard 300-point reference case (R0=225e-6,
      Req=R0/6, NHKV G=2500 mu=0.1, no forcing), `f_imr_fd` returns a complex
      trajectory: **max|imag(R/R0)| = 4.069, non-real at 299 of 300 points.**
      MATLAB does not error -- it silently returns complex values.
- [x] Upstream is defective. Keep `radial = 6` unsupported; the README claim is
      now a measurement.
- [x] Bonus: `radial = 7` is rejected by a *second* gate whose message reads
      `radial must be 1-6`, while the gate at `f_call_params.m:187` reads
      `radial must be 1, 2, 3, or 4` and actually permits 1--7. Three
      inconsistent statements of the same constraint.
- [ ] Update the README boundaries entry to quote the measured max|imag|
      instead of asserting non-real states.

**Done when:** the claim is a measurement, not an assertion, either way.

**Status: resolved.** PyIMR's decision to stop at `radial = 5` is correct
and now evidenced.

---

## W5. IMR-vanilla estimators

`~/research/solvers/IMR/IMR-vanilla` carries an experimental pipeline with no
counterpart here. `imr_inference` covers fitting; nothing covers getting from
data to the quantities a fit needs.

- [x] New module `imr_data.py`, registered in `pyproject.toml`, with `__all__`
      and covered by the W0 leak tests.
- [x] `equilibrium_radius` from `IMRCalc_Req.m` (cold, no-mass-transfer branch),
      restated dimensionally. Round-trips the solver's own pressure/radius
      relation at **0.00e+00**.
- [x] `collapse_features` replacing `calc_3tmins_3Rmaxs.m`. Upstream fits
      quartics through manually supplied index windows; this finds interior
      extrema automatically and refines each with a local three-point parabola.
- [x] `resolution_convergence` replacing `runIMR_convergence_check.m`.
      Monotone on the thermal grid: 3.7e-02 -> 1.3e-02 -> 0.
- [x] `natural_frequency` -- **not** ported from `calc_omega_N.m`; see below.
      Reproduces Minnaert exactly (0.00e+00) in the gas-only limit and matches
      the measured rebound frequency to **2.69e-02**.
- [x] Validation section `3b. TRACE ESTIMATORS` added, five checks.
- [x] Three input-validation tests in `tests/test_api.py`.

**Status: done.**

### Upstream defect 9: `calc_omega_N.m` overpredicts by 42x

It is a scratch script (`clear all;`, hardcoded values, commented-out
alternative formulas), not a library function. Two problems:

1. Its stiffness term `nT1 = (1+1/We)*(3k)/alpha^(3k)` treats the gas pressure
   at `Rmax` as `P8 + Laplace` -- the *equilibrium* condition, not the `Rmax`
   condition. Since IMR initialises `P0 = (P8 + 2S/Req - Pv)*(Req/R0)^3`, the
   gas at `Rmax` is strongly rarefied, and this inflates the stiffness by
   `alpha^(-3*kappa)`.
2. It closes with `omega_n = omega_n_star*c/Rmax`, using the medium sound speed
   where the nondimensionalisation is by `c0 = sqrt(P8/rho)` -- a further
   factor of `C = c/c0` (about 152x for water).

On the standard reference case the ported formula gives **2.30e+07 rad/s**
against a measured rebound frequency of **5.33e+05**, and with the `c` scaling
**3.50e+09**. The standard linearisation used instead gives 5.469e+05.

**Explicitly out of scope:** `calcRofT/` video processing (Hough circles,
Taubin fit, `LoadVideo`). That is image analysis, not solver scope, and
scikit-image covers it. Note the decision in the README boundaries section.

**Done when:** an R(t) trace can go from raw data to a prepared inference
problem without leaving Python.

---

## W6. Solver control surface

Minor parity items on the numerics options IMRv2 exposes.

- [x] `Lt` was already exposed as `PhysicalParameters.medium_grid_length`
      (default 2.0, matching `default_case.m`) and is functional -- varying it
      moves the coupled trajectory by 5.8e-03 (Lt=1.5) and 1.0e-02 (Lt=3.0).
- [x] `Lv` is **not** exposed, deliberately. `f_call_params.m:377-384` forces
      `Nv = 1` unless `spectral == 1`, and `Lv` only enters
      `f_pre_stress_int(Lv, Nv)`. Since spectral is reachable only for the
      gated-out stress 6/7 (plus stress 5 with an explicit `spectral=1`), `Lv`
      is effectively dead upstream. It belongs to W2, not here.
- [x] Integrator selection: added `SimulationConfig.max_step_s`, the dimensional
      equivalent of upstream `divisions` (which is `tspan(end)/divisions`).
      Verified: `max_step_s=1e-7` raises `nfev` 2457 -> 2748 and agrees with the
      unconstrained solve to 2.4e-06. `method` is **not** exposed -- the
      automatic LSODA / sparse-BDF choice plus `rtol`/`atol` supersedes an
      integer selector. Documented as a deliberate divergence.
- [x] Full audit of all 86 `f_call_params.m` options, below.

**Done when:** every upstream option is either supported, deliberately
superseded and documented, or listed as a known gap.

**Status: done.**

### Audit: all 86 upstream options

| group | upstream options | disposition |
|---|---|---|
| equation selectors | `radial` `bubtherm` `medtherm` `masstrans` `vapor` | supported directly |
| constitutive | `stress` `eps3` `g` `mu` `mu0` `du` `lambda1` `lambda2` `alphax` `nu_model` `v_a` `v_nc` `v_lambda` | **superseded** by the typed `material` object |
| geometry / IC | `r0` `req` `u0` `p0` `stress0` `collapse` | supported (`initial`, `collapse`) |
| time | `tfin` `tvector` | supported as the `times` argument |
| grids | `nt` `mt` `lt` | supported (`Nt`, `Mt`, `medium_grid_length`) |
| forcing | `pa` `omega` `tw` `dt` `mn` `wave_type` | supported directly |
| acoustics | `rho8` `p8` `c8` `gam` `nstate` `hugoniot_s` `surft` | supported via `PhysicalParameters` |
| thermal / mass | `t8` `kappa` `atg` `btg` `atv` `btv` `km` `dmass` `lheat` `rv` `ra` | supported via `PhysicalParameters` |
| solver | `method` `divisions` | superseded by automatic LSODA/BDF + `rtol`/`atol` + new `max_step_s` |
| spectral | `spectral` `nv` `lv` | W2; effectively dead upstream |
| output | `dimout` `progdisplay` | superseded -- output is always dimensional; no progress printing |
| nondimensional overrides | `re` `we` `ca` `de` `lam` `br` `chi` `foh` `fom` `iota` `dre` `ee` `om` `gama` `cstar` `p0star` `rzero` `rdotzero` `dtx` `twx` `alpha_g` `alpha_v` `beta_g` `beta_v` `dm` | **not exposed by design** -- the public API is dimensional; these are derived internally |
| minor gaps | `pv` | PyIMR derives vapour pressure from `T8` via `pvsat`; no direct override |

Only one true gap (`pv`), and it is reachable by adjusting `T8`. PyIMR also
exposes `medium_specific_heat_j_kg_k`, which IMRv2 fixes in `default_case.m`
and does not accept as an option.

---

## W8. Close the measured collapse gaps

Surfaced by W1. These are real, numeric disagreements with a runnable upstream
reference -- the highest-value remaining work, and it displaces W2 in priority
because it is measured rather than hypothetical.

- [x] **The Oldroyd-B "gap" was not a gap.** `f_call_params.m:458-460` reads
      `elseif stress == 5 / % TODO initial max stress for UCM and Oldroyd-B /
      Szero = zeros(...)`, and `stress < 3` leaves `Szero = []`. IMRv2 applies
      a collapse precursor **only** for stress 3 and 4. The recomputed
      `Req_zero` also comes back identical (0.16666667 = the input `Req/R0`),
      so `collapse=1` is a total no-op for NHKV and Oldroyd-B.
      Confirmed numerically: PyIMR reproduces both references **without**
      collapse at **1.56e-05** (Oldroyd-B) and **6.22e-06** (NHKV) -- inside
      the normal pinned band. The 1.51e-01 figure was PyIMR doing real
      physics against an upstream stub.
- [x] **Memoryless collapse: PyIMR is correct to refuse.** IMRv2 accepts
      `collapse=1` with `stress=1` and silently ignores it. Refusing is the
      stricter behaviour; the harness now asserts the refusal so it cannot
      regress.
- [x] Reclassified in `tests/run_validation.py`. `ref_collapse_oldb.csv` and
      `ref_collapse_nhkv.csv` renamed to `ref_coupled_oldb.csv` and
      `ref_coupled_nhkv.csv` -- they pin the fully coupled
      bubtherm+medtherm+masstrans+vapor model, which is **new coverage**: the
      Oldroyd-B case is the first pinned coupled-model trajectory with a
      memory material.
- [x] **Collapse Zener 1.55e-03 explained; PyIMR is the more accurate side.**
      The hypothesis above was right. Resolved quantitatively:

      | quantity | value |
      |---|---|
      | PyIMR S0 (event root-find, velocity = 0) | `-0.1599451098` |
      | IMRv2 `Szero` (`tools/probe_collapse_stress.m`) | `-0.1600469117` |
      | S0 that best fits the reference | `-0.1600464610` |
      | resulting best-fit deviation | **1.48e-05** |

      All precursor inputs agree to machine precision (`Pv`, `Pb` exactly;
      `req` differs by 1.2e-10 because upstream recomputes `Req_zero` from
      vapour equilibrium). The whole 1.55e-03 is the single 1.02e-04 offset in
      S0, and the coupled solve is exact: injecting IMRv2's own `Szero`
      reproduces the reference at **2.08e-05**, now asserted in the harness.

      Why the offset is inherent rather than a tolerance choice: at the true
      maximum `t = 0.84989883`, `R = 1.000000000000`, `v = -4.3e-19`,
      `dS/dt = +0.053248`. IMRv2's `Szero` corresponds to sampling
      `dt = -1.91e-03` before the peak -- where `R` is only **2.06e-06** lower.
      `R` is locally quadratic at the maximum and `S` locally linear, so a
      discrete argmax over solver output points cannot resolve the peak
      position better than O(sqrt(tol)), and carries that error straight into
      the stress. PyIMR's event root-find on `v = 0` is O(tol).

      Confirmed converged and integrator-independent: the event method returns
      `-0.15994511` under LSODA, Radau and BDF at both 1e-6 and 1e-9, while the
      argmax method scatters from `-0.1585` to `-0.1605` depending purely on
      where steps happen to land.

      **No code change.** Matching upstream here would mean adopting its error.
      Documented in the README and asserted in `tests/run_validation.py`.

**Done when:** the `KNOWN GAPS` block in `tests/run_validation.py` is empty, or
every remaining entry has a written justification.

**Status: done.** The `KNOWN GAPS` block is empty and all five checks in
section 1c pass. The remaining 1.55e-03 is upstream's error, quantified above.

---

## W2 (revised). Spectral-accuracy quadrature for the distributed stress

The original plan was to port IMRv2's Chebyshev collocation solver. On
inspection that would have solved the wrong problem.

### Why the original framing was wrong twice

First, W1 showed IMRv2 cannot run Giesekus or PTT at all, so there was no
spectral reference to match.

Second, and more usefully: **the distributed constitutive equations contain no
spatial derivatives.** Each material point evolves an ODE in its own stress;
the only spatial operation in the whole model is the quadrature for

    I = int_R^inf 2 (t_rr - t_hh) / r dr .

IMRv2 needs Chebyshev collocation because it writes the stress equation in
Eulerian form, where advection introduces spatial derivatives. PyIMR is
Lagrangian, so advection is exact and a PDE discretization has nothing to
discretize. The right independent check is a different **quadrature rule**.

### Diagnosis

Error is quadrature-limited, not truncation-limited. At fixed point count the
error *grows* with domain extent (2.4e-03 at extent 30 to 7.1e-03 at extent
240) because points thin out near the wall where the integrand is largest, and
falls about 4x per doubling of points -- classic O(h^2) trapezoid.

### Implementation

Mapping to the Lagrangian coordinate `u`, with `r_ref = 1 + (E-1) u^4` and
`r^3 = r_ref^3 + R^3 - 1`:

    I = int_0^1 [2 dt / r^3] * r_ref^2 * 4 (E-1) u^3 du

Only the bracketed factor moves. The rest folds into fixed weights, so the
integral is a dot product and **its time derivative is exact** rather than the
discrete difference the trapezoid path had to use. Gauss-Legendre nodes in `u`
then converge spectrally.

### Measured

Maximum absolute `R/R0` deviation from a converged reference, nonlinear
Giesekus at mobility 0.2:

| points | trapezoid | Gauss |
|---:|---:|---:|
| 60 | 3.4e-01 | 3.9e-04 |
| 120 | 2.2e-01 | 1.5e-07 |
| 240 | 7.4e-02 | 6.0e-08 |
| 480 | 2.6e-02 | 5.1e-08 |
| 960 | 6.8e-03 | -- |

Converged at 120 points across mobility 0..0.5 and De 2..8. The default moved
from `trapezoid`/480 to `gauss`/240: **1.4x faster and about five orders of
magnitude more accurate.**

### The finding that matters

**The former default carried 1.9-4.4% error on nonlinear Giesekus.** The
Oldroyd-B reduction limit -- the only check these models had -- reported
3.4e-03 and hid it, because the linear limit is far easier to integrate than
the nonlinear problem. A reduction limit is a necessary check, not a
sufficient one.

Reduction limits improved by two to three orders of magnitude as a
side effect: Giesekus -> UCM from 3.43e-03 to **3.46e-05**, KM Giesekus -> UCM
from 9.84e-04 to **9.10e-07**.

### Delivered

- [x] `quadrature` selector on `Giesekus` and `LinearPTT`; `"trapezoid"`
      retained so the old behaviour stays reachable.
- [x] Fixed-weight mapped quadrature in `_imr_stress`, with an exact integral
      rate.
- [x] Weights threaded through the compiled path so sensitivities are correct
      under both rules (2.1e-05 against centred differences).
- [x] New harness section `2c`: spectral convergence, a cross-check of
      `gauss(240)` against `trapezoid(3840)` at 4.4e-04, and the former
      default's error recorded for reference.
- [x] Default changed to `gauss` with 240 points.

**Still open:** a Chebyshev path for the *thermal* PDEs, where there genuinely
are spatial derivatives, remains unexplored. That is the only part of the
original W2 that still has a target.

---

## W10. Chebyshev collocation for the thermal PDEs

The one part of the original W2 with a real target left. Unlike the
distributed stress -- whose Lagrangian form has no spatial derivatives at all
-- the thermal fields genuinely carry them, so collocation buys something.

### Diagnosis

`thermal_fd` is correct and cleanly second order. Tested directly on
`cos(2y)`, the spherical Laplacian errs by 6.6e-02, 1.7e-02, 4.2e-03,
1.0e-03 at N = 11, 21, 41, 81 -- ratio 4.00 per doubling, exactly as designed.

The trajectory is the problem. Against a converged `Nt = 1600` reference:

| Nt | max abs dR/R0 | ratio |
|---:|---:|---:|
| 25 | 2.59e-02 | -- |
| 50 | 1.82e-02 | 1.42 |
| 100 | 9.80e-03 | 1.86 |
| 200 | 5.06e-03 | 1.94 |
| 400 | 2.37e-03 | 2.13 |
| 800 | 8.37e-04 | 2.83 |

Second order asymptotically, but it only gets there slowly, and **the default
`Nt = 25` carries 2.6% error** -- the same shape of problem the quadrature
work found, in a different place.

### Implementation

`thermal_spectral.py` mirrors `thermal_fd`'s interface. Two points of care:

- **Regularity at the bubble centre.** The gas operator is built on the *even
  extension* of the grid to `[-1,1]`: a function smooth at the origin of a
  sphere is even in `y`, so the removable `2/y` singularity never has to be
  special-cased the way `thermal_fd` does at row 0.
- **Boundary flux weights.** The wall closure used hardcoded three-point
  stencils. These now come from the relevant row of the first-derivative
  matrix, full length, so the closure works on a non-uniform grid. Verified
  the derivation reproduces the finite-difference stencils exactly:
  `grad_Trans = chi * D1_bubble[-1, ::-1]`, with a zero tail.

### Measured

| | Chebyshev | finite difference |
|---|---:|---:|
| spherical Laplacian, N = 17 | **1.9e-11** | 2.6e-02 |
| trajectory, Nt = 25 | **6.3e-04** | 2.6e-02 |
| Nt needed for ~1e-3 | ~25 | ~800 |

Both backends converge to the same limit -- finite difference at Nt = 200,
400, 800 approaches the spectral answer monotonically (5.5e-03, 2.8e-03,
1.3e-03).

### The reported plateau was a measurement artifact

The first pass reported that the spectral trajectory plateaus near 2.7e-04 and
guessed the implicit wall-temperature solve was the limit. **Both were wrong.**

The plateau was the output grid. The check sampled `R(t)` at 200 points over
60 us -- 0.3 us spacing -- while the collapse minimum is far sharper than that.
As the solution shifts slightly with resolution, those samples land at
different phases of the collapse, producing pointwise differences around 1e-04
that have nothing to do with the discretization. Two things gave it away: the
maximum difference sat at exactly the same sample index for every resolution,
and the median difference over the trace was 8.2e-06 -- two orders below it.

The wall-solve guess was also wrong for a simpler reason: the test case was
`bubtherm` only, so `medium is None` and the wall closure is never called.
`thetadot[-1] = 0` there -- a frozen Dirichlet value, exact by construction.

Ruled out separately: solver tolerance. Varying `rtol` from 1e-8 to 1e-12
changes the result in no digit.

Measured properly, on collapse depth and timing with a resolved output grid:

| | minimum radius | collapse time |
|---|---:|---:|
| spectral, Nt = 25 | 4.4e-06 | 0.0005 ns |
| spectral, Nt = 100 | 9.9e-08 | 0.0001 ns |
| finite difference, Nt = 25 | 8.6e-04 | 0.5053 ns |
| finite difference, Nt = 400 | 4.7e-06 | 0.0119 ns |

**Spectral at 25 points matches or beats finite difference at 400**, and keeps
converging. The real result is considerably stronger than the 41x first
claimed.

### The lesson

Both mistakes in this workstream, and the flaky assertion in W2, are the same
error: choosing a metric or tolerance that is sensitive to something other
than the quantity of interest. Pointwise `max|dR|` on a coarse grid measures
sampling phase at a sharp feature, not accuracy. A strict monotonicity
assertion near a floor measures noise, not convergence. Prefer physically
meaningful, sampling-independent quantities -- here, collapse depth and time.

### Delivered

- [x] `thermal_spectral.py` with `chebyshev_diff_mat` and `nodes`, validated
      standalone against analytic derivatives.
- [x] Boundary flux weights generalized to full-length dot products, derived
      from `D1` rows.
- [x] `SimulationConfig.thermal` selector; `"fd"` remains the default because
      it is what every pinned IMRv2 trajectory was generated against.
- [x] All four thermal configurations run under both backends.
- [x] Harness section `2d`: operator accuracy, 41x trajectory improvement at
      equal resolution, and cross-backend convergence.

**Still open:** whether the default should move. The accuracy case is now
unambiguous; the argument against is that every pinned IMRv2 trajectory was
generated with `thermal="fd"`, so switching invalidates the pinned suite.

---

## W9. Fix the upstream defects in our own code

Not just report them -- work out the correct physics and validate against
independent results. Done for the Mie-Gruneisen equation of state, which was
the only defect that cost PyIMR a capability.

### The defect

`f_radial_eq.m`'s `f_mie_gruneisen_eos_scalar` solves `a*mu^2 + b*mu + A = 0`
for the density strain `mu`, with `a = A*s^2 - nog`, `b = -2*A*s - 1`. As
`A -> 0` this degenerates to `-nog*mu^2 - mu = 0`, whose roots are `mu = 0`
(physical: undisturbed liquid at ambient pressure) and `mu = -1/nog = -0.325`
(spurious). Upstream takes `(-b + sqrt(d))/(2a)`, which is the spurious branch.

Consequences, all measured:

| quantity | upstream root | corrected root |
|---|---|---|
| `rho/rho0 - 1` at ambient | -0.3253 | 4.3e-05 |
| `c/c0 - 1` at ambient | complex | 2.8e-04 |
| enthalpy at `P = 2` | 1.482 (should be ~1) | 1.000 |
| enthalpy at `P = 1000` | 1599 (should be ~999) | 980.9 |
| sound-speed numerator | -2.36 (negative) | +1.00 |

The negative `c^2` is precisely why `radial = 6` returns complex radii: it is
the only branch that takes the sound speed from the EoS rather than using `c0`.
The earlier note in the code -- blaming a missing `iWe/R` -- was wrong; that
term shifts the numerator by 1e-4 and changes nothing.

Second defect in the same branch: `Pb` omits the stress term that `radial = 3`
and `4` both include.

### Validation, against independent results rather than upstream

- **EoS calibration.** `c(ambient) = c0` to 2.8e-04 -- the defining property of
  the EoS, and impossible to satisfy with the spurious root.
- **Weakly-compressible limit.** The analytic enthalpy matches `h ~ P - 1` to
  2.1e-03 over `P = 2..100`, and matches a direct numerical integral of
  `1/rho(P)` to machine precision, confirming `_mie_F` itself was always right.
- **Independent equation of state.** Tait and Mie-Gruneisen enthalpies agree to
  0.03% at `P = 100` and 0.26% at `P = 1000`; sound speeds agree to 3.7% at
  ambient, diverging under compression as two different EoS should.
- **Cross-model consistency.** All Keller-Miksis forms must converge for a
  weakly-compressible liquid. Corrected `radial = 5` agrees with `radial = 3`
  to **5.2e-04**, inside the 1.8e-03 spread the Tait forms already have among
  themselves. As shipped it differs by **4.8e-01**.
- **Collapse depth.** Upstream `radial = 5` reaches `R/R0 = 0.0536`; its own
  Tait branch reaches `0.0821`; corrected, it reaches `0.0818`.
- **Sensitivities improved too.** The `radial = 5` material tangent went from
  1.46e-05 to **6.76e-07**, matching every other radial model.

### Delivered

- [x] Corrected root in `_imr_thermal._mu_of_A` and `_imr_mechanical._mie_mu`.
- [x] Stress term restored to `Pb` for `radial = 5`.
- [x] **`radial = 6` implemented** -- the one configuration IMRv2 cannot run.
      Validates at 1.44e-02 against Gilmore/Tait and 2.74e-02 against
      KM/Mie-Gruneisen, both inside the 1.27e-02 Gilmore-vs-KM reference
      spread. Sensitivities agree with centered differences.
- [x] New harness section `1d`, which asserts the physics rather than upstream
      agreement. `ref_radial5.csv` retained as a record of upstream behaviour,
      removed from the pinned-target list.

### Method note

An edit changing `+` to `-` preserves file size, and CPython's `.pyc` staleness
check is `(mtime, size)` with one-second mtime granularity -- so the change was
silently ignored and three consecutive measurements were taken against stale
bytecode. One of them reversed a conclusion. **Clear `__pycache__` (and the
numba cache) after any single-character edit before trusting a measurement.**

---

## Ledger: where PyIMR already exceeds upstream

Keep current as work lands; this is the other half of "parity or better".

- Six hyperelastic laws vs IMRv2's neo-Hookean and quadratic KV only.
- Yield-stress laws (Herschel-Bulkley, Bingham) absent upstream entirely.
- Power-law viscosity absent upstream.
- Sampled PCHIP forcing with analytic derivative.
- Configurable `PhysicalParameters` and `InitialState`.
- Forward sensitivities of the production RHS — no upstream counterpart.
- Collapse shooting with implicit event and root sensitivities.
- Always-dimensional public output; typed API; structured failures.
- Sparse BDF on coupled nonlinear cases: 5.64 s -> 0.73 s.

Added after W1 measurement:

- **Giesekus and linear PTT actually run.** IMRv2 dispatches both but gates
  them out; PyIMR is the only working implementation of the two.
- **A correct generalized-Newtonian suite.** Six laws that pass their own
  Newtonian reduction exactly, against an upstream suite where six of seven
  models raise and the seventh is quantitatively wrong.
- **No silent complex arithmetic.** `radial = 6` is refused with a structured
  error rather than returning complex radii, which is what upstream does.
- **Consistent input validation.** Upstream states the `radial` constraint
  three mutually inconsistent ways and dispatches `stress` codes its own gate
  rejects.

---

## W7. Style: write it like tinygrad

Target: tight, dense, minimal-surface Python. Measured against
`tinygrad/tinygrad` as the reference aesthetic.

Baseline as of this plan:

| metric | tinygrad | PyIMR | verdict |
|---|---|---|---|
| comment-only lines | 1.2% | 1.0% | already there |
| blank lines | 3.0% | 9.2% | 3x too airy |
| docstrings | ~0 | 82 | strip most |
| indent | 2 space | 4 space | convert |
| largest file | -- | `pyimr.py`, 2683 lines | split |

The house rules, taken from tinygrad's own config and source:

- `line-length = 120`, and actually use it. One-line bodies:
  `def prod(x:Iterable[T]) -> T|int: return functools.reduce(operator.mul, x, 1)`
- 2-space indent.
- Ruff ignores that license the dense idioms: `E401` (multi-import lines),
  `E402`, `E701`/`E702`/`E703` (colon and semicolon multi-statements),
  `E731` (assigned lambdas), `E741` (short math names -- `l`, `I`, `af`).
- No docstrings in internal code. Comments are rare, prefixed `# NOTE:` when
  they carry a real warning, and a bare URL when the reference is the point.
- Tuple assignment over stacked statements:
  `OSX, WIN = platform.system() == "Darwin", sys.platform == "win32"`
- Type hints tight: `x:Iterable[T]) -> T|int`, no spaces around `:`.
- Blank lines only where they separate genuinely distinct sections.

### The one deliberate divergence

tinygrad is an internals-heavy library; PyIMR is a scientific API that
people call from notebooks, where `help()` and IDE hovers matter. Resolution:
**keep docstrings on the public API surface only** -- the exported material
classes, `SimulationConfig`, `simulate`, `prepare`, `solve_with_sensitivities`,
`prepare_inference` -- and strip every internal one. That is roughly 15 of the
current 82.

### Tasks

- [x] Set `line-length = 120`, `indent-width = 2`, and the tinygrad ignore list
      (`E401`, `E402`, `E701`--`E703`, `E731`, `E741`) in `pyproject.toml`.
      Also widened `select` from `["E9","F63","F7","F82","I"]` to
      `["E4","E7","E9","F"]`, which immediately caught a dead assignment
      (`iota` in the medium-thermal RHS) that the narrower set had missed.
- [x] Convert to 2-space indent via `ruff format` (AST-safe) rather than regex.
- [x] Strip internal docstrings with an AST tool that keeps module docstrings
      and anything named in `__all__`: 82 -> 53, all remaining ones public.
- [x] Split `pyimr.py`: extracted the material value objects into
      `_imr_materials.py` (436 lines), registered in `pyproject.toml` and the
      W0 leak test. **2683 -> 2214 lines.**
- [x] Re-measure and record.
- [ ] Collapse blank lines from 10.7% toward 3-4%. **Blocked by the formatter**
      -- see trade-off below.
- [x] Two further splits of `pyimr.py`, both AST-driven and bitwise-verified:
      `_imr_stress.py` (359) takes the constitutive evaluation -- material plus
      kinematic state in, stress and rate out, with no dependency on the RHS --
      and `_imr_thermal.py` (287) takes the Tait/Mie-Gruneisen parameters,
      `pvsat`, the dissipation terms and the implicit wall-temperature solve.
      **`pyimr.py` 2683 -> 1686 across the three splits (-37%).**
- [x] `_imr_rhs.py` (315): the right-hand side plus forcing evaluation --
      `_rhs`, `_pinf`, `_sampled_pressure`, `_nZ`, `_radius_floor_event`. A
      clean leaf: it reads a prepared problem but never builds one. The two
      material predicates `_is_distributed_stress` and `_stress_state_count`
      moved to `_imr_materials.py` where they belong, which is what made the
      cut acyclic. **`pyimr.py` 2683 -> 1396 over four splits (-48%).**
- [x] `_imr_config.py` (568): the frozen config, prepared-problem and result
      dataclasses, their validation, and the physical defaults.
      `PreparedProblem.solve` defers its `_solve_prepared` import, the same
      trick it already used for `solve_with_sensitivities`.
- [x] `_imr_prepare.py` (530): `params`, the `_prepare_*` helpers, the collapse
      precursor and `prepare` itself.
- [x] **`pyimr.py` 2683 -> 482 over six splits (-82%).** Every module is now
      under the ~800 target except `imr_sensitivity.py` (1001), which is
      untouched and dominated by `solve_with_sensitivities` (146),
      `_material_parameters` (107) and `_output_duals` (101).

      | module | lines |
      |---|---|
      | `imr_sensitivity.py` | 1001 |
      | `_imr_mechanical.py` | 647 |
      | `_imr_config.py` | 568 |
      | `_imr_prepare.py` | 530 |
      | `pyimr.py` | 482 |
      | `_imr_materials.py` | 452 |
      | `_imr_stress.py` | 359 |
      | `imr_inference.py` | 331 |
      | `_imr_rhs.py` | 315 |
      | `_imr_thermal.py` | 287 |
      | `imr_data.py` | 181 |

      **Method notes, all learned the hard way:**

      1. Extract with AST spans and delete in descending line order. Naive line
         slicing silently corrupted a first attempt.
      2. AST `lineno` for a decorated class points at `class`, **not** the
         decorator. Use `min(node.lineno, *[d.lineno for d in
         node.decorator_list])` or every `@dataclass` is orphaned in the source
         file and missing from the destination -- which turns frozen
         dataclasses into plain classes without any import error.
      3. Tuple-target assignments (`_ATG, _BTG = ...`) are invisible to a scan
         that only matches `ast.Name` targets.
      4. Do not run `ruff check --fix` on a new module before its imports are
         complete -- it strips not-yet-used imports as F401 and the failure
         then presents as a missing name. It also silently dropped
         `finite_diff_mat` from `pyimr`, breaking
         `validate_bubtherm_adiabatic.py`, which nothing in CI ran. Those two
         standalone scripts are now covered by `tests/test_api.py`.
- [ ] Fold one-line function bodies onto the `def` line -- see trade-off below.
- [ ] Add `.git-blame-ignore-revs` for the reformat commit when this is
      committed.

### Measured

| metric | tinygrad | before | after |
|---|---|---|---|
| total lines | -- | 6290 | **6245** (and 469 fewer in `pyimr.py`) |
| comment-only | 1.2% | 1.0% | 1.3% |
| blank | 3.0% | 9.2% | 10.7% |
| docstrings | ~0 | 82 | **53** (public only) |
| indent | 2 space | 4 space | **2 space** |
| largest file | -- | 2683 | **2214** |

### The formatter question, resolved by measurement

tinygrad uses no formatter; its density is hand-maintained. The earlier note
framed this as keep-the-formatter *or* reach tinygrad density. That was a false
choice -- `ruff format` has one knob that recovers most of the difference.

`skip-magic-trailing-comma = true` collapses any call, signature or literal
whose trailing comma was forcing it to stay exploded, wherever it fits in 120
columns. Measured across the repo:

| | lines | blank |
|---|---:|---:|
| default `ruff format` | 6632 | 729 (11.0%) |
| `skip-magic-trailing-comma` | **6078** | 729 (12.0%) |

**554 fewer lines of identical content, -8.4%.** The blank *count* is
unchanged; its share only rises because the denominator shrank, so the earlier
"blank lines are too high" reading was measuring the wrong thing -- the problem
was exploded call sites, not vertical whitespace.

**Decision: keep `ruff format`, with `skip-magic-trailing-comma`.** Reasoning:

- This repo's whole claim is machine-verified correctness. Machine-enforced
  style is the same argument; hand-maintained density is a review burden that
  drifts, and tinygrad gets away with it only because a very small group with
  strong shared taste maintains it.
- Work here arrives as parallel agent branches. A formatter keeps style churn
  out of those diffs entirely.
- The structural wins -- 120 columns, 2-space indent, stripped docstrings, six
  file splits -- are already banked and are worth far more than one-line `def`
  bodies.

What is given up: one-line `def` bodies and semicolon multi-statements, which
`ruff format` will always expand. The lint ignores for them stay in place, so
the decision is reversible by deleting three lines of config.

**Verification:** every step was checked bitwise. A 10-case trajectory hash
(NHKV, Zener, Oldroyd-B, qKV, Giesekus, PTT, radial 3, radial 5, fully coupled,
Powell-Eyring) is **identical** before and after the whole W7 sequence:
`3aa31b7cb1d4...`. 32 tests pass, ruff clean, full validation green, and the
wheel builds with all eight modules.

---

## Upstream defects found

Record as encountered; useful for the reference-implementation README section.

All verified empirically against `dea31cd` with MATLAB R2025a, not inferred
from reading. Reproduce with `tools/gen_imrv2_cases.m` and
`tools/probe_viscosity.m`.

- [x] **1. Giesekus and PTT are unreachable.** `f_call_params.m:376-385`
      dispatches `stress` 6 and 7 and forces `spectral = 1` for both, but the
      input gate at `f_call_params.m:206-208` rejects `stress > 5`. The
      spectral solver `f_imr_spectral.m` (542 lines) is reachable only for
      `stress = 5`, where `spectral` stays caller-controlled. Severity: high --
      two advertised constitutive models cannot be run at all.
- [x] **2. `nu_model` 3--7 never assign their outputs.** `f_viscosity.m:66-81`
      sets only `f` for Powell-Eyring, modified Powell-Eyring, Cross,
      simplified Cross and modified Cross; `intf`, `dintf` and `ddintf` are
      left unassigned. `f_imr_fd.m:419-421` requests all four whenever
      `nu_model ~= 0`, so every one of these raises
      `Output argument "intf" ... not assigned`.
- [x] **3. `nu_model = 2` calls a 4-argument function with 3 arguments.**
      `f_viscosity.m:45` is `f = sf_carreau_yasuda(nc,lambda,gammadot_R)` but
      the definition is `sf_carreau_yasuda(a,nc,lambda,gammadot_R)`. Raises
      `Not enough input arguments`. Carreau-Yasuda is unusable.
- [x] **4. `nu_model = 1` (Carreau) fails its own Newtonian reduction.** With
      `v_nc = 1` the shear-thinning factor is identically 1, so the model must
      reduce to Newtonian at `mu8 + Dmu`. Measured `max|dR| = 6.7e-01` against
      `mu = 0.5`; the `v_lambda -> 0` limit fails identically. The
      `v_lambda -> inf` limit does reduce correctly to `mu = mu8` (9.4e-04).
      Structurally, `f_viscosity.m:36` forms `intf = gammadot_num*I1`, which is
      quadratic in the strain rate, while the Newtonian term it must reduce to
      (`f_stress.m:24`, `-4/Re8*Rdot/R`) is linear. The extra `gammadot_num`
      factor is the likely cause. Severity: high -- the one working
      non-Newtonian model is quantitatively wrong.
- [x] **5. `radial = 6` returns complex trajectories.** max|imag(R/R0)| = 4.069
      over 299 of 300 points, without raising. See W4.
- [x] **6. Three inconsistent statements of the `radial` constraint.** The gate
      at `f_call_params.m:187` permits 1--7 while its message says
      `1, 2, 3, or 4`; a second gate rejects 7 with `radial must be 1-6`.
- [x] **7. `src/f_init_stress.m` uses an undefined `z1`** in the
      `De == 0 || De == Inf` branch. Unreachable for the memory models that
      call it -- latent, not active. (Already in README.)
- [x] **8. `src/default_case.m` stress-code comment is stale:** it documents
      `4: PTT, 5: Giesekus`, but `f_call_params.m` dispatches
      `6: PTT, 7: Giesekus`.

Worth reporting upstream. Defects 1--4 mean IMRv2's entire non-Newtonian
viscosity suite and both nonlinear-memory models are non-functional at this
revision.

## W11. JAX as a second backend

**Status: Stage 0 run and passed. Stages 1-5 not started.**

### The bet

JAX replaces `_mechanical.py` (611 lines), `_dual.py` (469) and `_complex.py`
(125) -- **1205 lines, 21% of the package** -- and deletes the hand-written
forward-mode AD entirely. It would also make the exact Hessian free, which is
the open question behind #88's Gauss-Newton diagnostic.

What numba currently buys, measured: **2.2x** on the mechanical sensitivity
solve (46.6 ms compiled against 103.7 ms on the Dual path), for 611 lines of
duplicated physics that every change must land in twice. Only a *numeric*
consistency test catches divergence between them; nothing compares structure,
which is how the compiled path came to have its radial branches collapsed while
the Python path still had four copies of the same formula (#94).

### The risk that actually matters

Not the port. Every pinned IMRv2 trajectory is a statement about *scipy's*
integrator at specific tolerances -- medians bounded from 2e-07 (Keller-Miksis
NHKV) to 8e-06 across eighteen cases. A different integrator moves all of them,
and there is no principled way to re-pin without redoing the IMRv2 comparison.

### Safety principle

JAX arrives as a **selectable backend, defaulting off**, the same shape
`thermal="fd"|"spectral"` took in #20/#26. Two rules make it reversible:

- **The reference for the port is scipy-backend output, not IMRv2.**
  Cross-backend agreement on identical configs is tighter and far cheaper to
  check than re-deriving IMRv2 agreement. The pinned tests stay on scipy.
- **No pinned bound is ever relaxed to make JAX pass.** Disagreement beyond the
  cross-backend bound is a finding to explain, not a tolerance to widen.

### Stages

- [x] **Stage 0 -- throwaway spike, outside the repo.** `radial=2`, NHKV, no
      thermal, on JAX + diffrax. Results below. Abort condition was: cannot hold
      2e-07, or slower than the 103.7 ms Dual path. Neither fired.
- [ ] **Stage 1 -- the seam, with no JAX in it.** `backend: str = "scipy"` on
      `SimulationConfig`, routing `simulate` / `solve_with_sensitivities`, with
      only `"scipy"` implemented. Gate: existing tests green AND trajectories
      bit-identical.
- [ ] **Stage 2 -- forward solve, mechanical only.** `backend="jax"` for radial
      1-6, no bubtherm, no forcing. Gate: a cross-backend test bounding
      `|R_jax - R_scipy|` over a config sweep. **Design revised -- see "JAX
      traces the existing RHS" below. No transcribed JAX module.**
- [ ] **Stage 3 -- sensitivities.** `jax.jacfwd` through the diffrax solve
      replaces the Dual path for the JAX backend only. The strongest gate in the
      plan: two independently verified tangent paths already agree at
      5e-13--5e-11, so JAX has to join that agreement rather than establish it.
      **Blocker resolved and the approach validated. The GATE as written here is
      wrong** -- corrected under "What the gate should be" below.
- [x] **Stage 4 -- thermal.** bubtherm and medtherm land; mass transfer does
      not, because its wall closure is still a secant with a fallback ladder.
      **Two claims here were wrong -- see "Stage 4 corrections" below.**
- [ ] **Stage 5 -- remove numba, and only then.** Delete `_mechanical.py` once
      JAX covers the mechanical path at >= parity; delete `_dual.py` and
      `_complex.py` once JAX covers everything they do. Keeping numba through
      stages 0-4 is deliberate -- it is the performance reference.

### Stage 0 results

Ported `radial=2` + NHKV standalone (no `pyimr` import, so nothing could
accidentally depend on the code it is meant to replace). Against the tightest
pinned median bound in the suite, 2e-07, where scipy sits at 3.128e-08:

```
solver              vs IMRv2 max   vs IMRv2 median   vs scipy backend
Kvaerno5 (stiff)      9.368e-06       3.018e-08          8.15e-07
Tsit5 (non-stiff)     9.375e-06       3.012e-08          8.22e-07
```

Sensitivity throughput, 200 points x 2 parameters:

```
Tsit5 (non-stiff)        8.8 ms     5.3x FASTER than numba
numba compiled          46.6 ms
Dual path              103.7 ms
Kvaerno5 (stiff)       832.3 ms     18x slower than numba
```

Both gates pass, and JAX is marginally *more* accurate than scipy here.

Two things not to over-read. The 8.8 ms is a bare jitted `jacfwd`; numba's
46.6 ms includes `prepare`, dual-config construction and packing, so some of
that gap is framework overhead rather than the integrator. And Kvaerno5's
832 ms is an implicit solver doing Newton iterations on a problem that is not
stiff -- it says nothing about genuinely stiff configurations.

### Stiffness is static, so there is nothing to auto-switch

The 832 ms was **my misconfiguration, not a diffrax limitation**, and the first
draft of this section reported it as a property of the tool. Kvaerno5 takes
*fewer* steps than explicit Bosh3 (4424 against 7436) and still costs 17x more
per step -- 119.6 us against Tsit5's 7.2 us. That is the Newton solve, which is
pure waste on a problem that is not stiff. The mechanical problem is not stiff:
eigenvalue ratio 1.18.

diffrax has no auto-detecting switcher, which is true and was the stated risk.
It does have IMEX (`KenCarp3/4/5`, `Sil3`), where the stiff term is designated
rather than detected -- a better fit here, because IMR's stiffness is
structurally the thermal diffusion operator.

But the switching is not needed at all. Measured `|lambda_max / lambda_min|` of
`df/dy` along the trajectory:

```
mechanical NHKV                 1.18e+00     not stiff
bubtherm=1, Nt=25, fd           8.06e+03     mildly stiff
bubtherm + medtherm, fd         3.28e+11     very stiff
bubtherm + medtherm, spectral   6.90e+16     extremely stiff
```

**Median equals worst in every case** -- the ratio is constant along the
trajectory. Stiffness belongs to the configuration, not to a phase of the solve,
so the solver can be selected at `prepare` time exactly as scipy already selects
BDF when `jacobian_sparsity` is present. LSODA's auto-switching is not
load-bearing for this problem.

(The spectral case being five orders stiffer than finite difference is the
Chebyshev `D**2` eigenvalue scaling of `N**4` against `N**2`, already recorded
in W10.)

Consequence for the plan: Stage 2 picks explicit for mechanical, Stage 4 picks
implicit or IMEX for coupled thermal, and neither has to detect anything.

### JAX traces the existing RHS, so there is nothing to transcribe

The obvious shape for stage 2 -- a `_jax.py` holding the physics written in
`jnp` -- would have been a second implementation of the right-hand side. That is
precisely the `_mechanical.py` problem this migration exists to remove, with the
same divergence risk, and it would have been added rather than avoided.

It is not necessary. `_rhs` already runs `Dual` objects through
`__array_ufunc__`, so it is written against an overloadable numeric protocol
rather than against numpy concretely. Measured: with the array namespace swapped
for `jax.numpy` inside `_rhs`, `_stress` and `_thermal`, and Python's `max`
routed to `jnp.maximum`, `jax.jit` traces the real function and agrees with the
numpy path to **4.44e-16** -- one ULP, from operation ordering, not structure.

An AST survey says why so little is needed. On the mechanical path `_rhs` makes
exactly three namespace-specific calls (`np.sqrt` once, plus `np.empty_like` and
`np.zeros_like` off-path) and one Python `max`. Every in-place mutation --
`thetadot[-1] = 0.0`, `Tmdot[0] = 0.0`, the buffer packing at lines 221-235 --
sits inside `if bubtherm:`, `if medtherm:`, or the distributed-stress branch.
The mechanical path returns `out = [Rd, Rdd]`, a plain Python list, which JAX
accepts as-is.

So stage 2 becomes:

- **2a** -- make the mechanical path namespace-agnostic, with numpy injected by
  default. No JAX involved. Gate: bit-identical trajectories, the same gate
  stage 1 met, using the same harness.
- **2b** -- the diffrax driver, the `backend` field, jax/diffrax as OPTIONAL
  dependencies (the `pymc_op` precedent: core stays numpy/scipy/numba), and the
  cross-backend test.

Splitting it this way keeps the risky part -- touching the most-validated
function in the package -- in a change whose gate is exact equality, separate
from the part that introduces new numerics.

The in-place mutations remain the blocker for stages 4 and beyond, where the
thermal fields are assembled by slice assignment. `jnp` has `.at[].set()` for
this, which is a real edit rather than a namespace swap. That work belongs with
stage 4, not before it.

### How a parameter reaches the traced region (stage 3)

Stage 2b differentiates nothing: it integrates a right-hand side whose
parameters are already concrete. Stage 3 has to get a *physical* parameter --
`material.shear_modulus_pa` -- into the trace, and that turns out to be the
whole difficulty.

The obvious route is to rebuild the material from a traced value and re-run
`params`. Measured, it fails immediately:

```
jax.jacfwd(lambda g: NeoHookeanKelvinVoigt(g, 0.1).shear_modulus_pa)(2500.0)
  -> TracerArrayConversionError
```

The material dataclasses validate in `__post_init__`, and `_finite_positive`
calls `np.isfinite`, which converts a tracer. So a material cannot be
CONSTRUCTED from a traced value at all -- the blocker is validation, not the
arithmetic. I had expected the two `if G > 0` / `if mu > 0` guards in `params`
to be the obstacle; they are not reached.

That rules out one design and points at the right one. **Structure stays
concrete; only scales are traced.** `_material_scales` is pure dispatch --
material in, `(G, mu, lam1, lam2, alphax)` out, no numerics -- so `params`
should accept an override for that tuple:

```
G, mu, lam1, lam2, alphax = _material_scales(material) if scales is None else scales
```

The degenerate guards then read the CONCRETE material, not the override, which
is correct on its own terms: whether a material has elasticity is structural and
cannot change under differentiation. `params`'s only other namespace-specific
call is one `np.sqrt` for `Uc`, which depends on ambient pressure and density
rather than on anything differentiated.

So stage 3 is: `scales=` and `xp=` on `params`, a traced solve built from a
concrete config plus a traced scale vector, `jax.jacfwd` over it, and the
existing complex-step/Dual agreement as the gate. That is a real signature
change on a function with fourteen arguments, which is why it is bigger than 2a
or 2b rather than the same size.

### What the gate should be (stage 3)

The plan said JAX must join the 5e-13 agreement between complex-step and Dual.
That is not achievable, and chasing it would have been a bug hunt with no bug.
Those two agree to 5e-13 because they are two AD methods differentiating **the
same integration** -- same solver, same steps. JAX differentiates a *different*
integrator, so its tangent carries that integrator's error exactly as its
trajectory does.

Measured, with `params` taking traced scales and `jax.jacfwd` over the diffrax
solve, against `solve_with_sensitivities`:

```
NHKV,  dR/dG           max rel 5.81e-07
NHKV,  dR/d(G, mu)     max rel 3.13e-07
Zener, dR/dG           max rel 9.76e-07
```

Tightening diffrax alone plateaus at 5.67e-07, which says the residual is
scipy's error rather than diffrax's. Tightening BOTH is the real test:

```
both rtol=1e-08     5.81e-07
both rtol=1e-10     2.99e-09
both rtol=1e-12     3.51e-11
```

It converges. The JAX tangent is the same derivative and the whole residual is
mutual integration error. **The gate is convergence under refinement, not a
fixed threshold** -- the stronger claim, since a threshold can be met by a wrong
derivative that happens to land close.

`params` now takes `xp=` and `scales=`, bit-identical on the numpy path across
all twenty traces. What remains is assembling a full `SensitivityResult` from a
JAX solve: the state tangent is immediate, but `internal_pressure_pa` and
`stress_integral_pa` are nonlinear functions of state and parameters that the
Dual path gets for free and a jacfwd path has to derive.

### Stage 4 corrections

**"Stage 4 picks implicit or IMEX for coupled thermal" was wrong.** Explicit
wins even on the stiffest configuration measured:

```
coupled fd        Tsit5      56.5 ms    822 steps      Kvaerno5  2716.8 ms  3697 steps
coupled spectral  Tsit5     549.2 ms   8912 steps      Kvaerno5  4177.1 ms  6077 steps
```

The implicit solver does take fewer steps, and the stiffness ratios (3.3e+11
and 6.9e+16) are real -- 8912 steps against 326 for the mechanical case is the
stiffness biting. But at 26 to 53 states the Newton solve costs ~7x per step and
never repays that. It would at larger N; it does not here.

**The backend was re-tracing on every call.** Compilation dominates the solve,
so the JAX path measured 796 ms against scipy's 30 ms while a jitted spike of
the same problem ran in 4 ms. Now cached on the prepared problem's identity plus
the shapes that change the traced program, bounded at 32 entries so a design
sweep over time grids cannot grow it without limit.

With that fixed, prepared once and solved repeatedly -- which is how the
inference layer calls it:

```
mechanical           scipy  27.7 ms    jax   3.1 ms    8.8x
bubtherm fd                122.3            22.0       5.6x
coupled fd                 276.1            62.1       4.4x
coupled spectral           527.1           634.1       0.8x
```

The stiffest configuration is the one JAX loses, and neither diffrax solver
beats LSODA there. IMEX is the untried option -- `KenCarp` wants a `MultiTerm`
split designating the stiff part, which the diffusion operator supplies
naturally, and an earlier attempt passed a single `ODETerm` and failed for that
reason rather than on the merits.

### IMEX: tried, and the blocker is compile time

W11 recorded IMEX as the untried option for the stiff thermal configurations, on
the grounds that `KenCarp` wants a `MultiTerm` split naming the stiff part and
the diffusion operator supplies one naturally. Tried it.

The split itself is sound. Writing only the IMPLICIT half -- the `ddtheta` and
`ddTm` second-derivative terms -- and taking the explicit half as `_rhs` minus
it makes the two sum to the shipped right-hand side by construction, verified at
**0.00e+00**. So the spike cannot integrate different physics than the package.

What stopped it is compilation. `KenCarp4` on the coupled-fd problem did not
finish compiling in **15 minutes**, against 52 ms of solve time for `Tsit5` on
the same configuration. An implicit Runge-Kutta stage generates a Newton solve
per stage over the whole coupled state, and the resulting graph is enormous.
Kvaerno5 compiles in seconds by comparison, which is why it produced numbers
earlier and `KenCarp` did not.

So the ordering for the stiff thermal case stands as measured, with IMEX now
struck off rather than pending:

```
Tsit5      549 ms    8912 steps
Kvaerno5  4177 ms    6077 steps
KenCarp4  compilation did not complete in 15 minutes
```

Explicit remains the right choice. The 0.8x on coupled spectral is a real loss
against LSODA and there is currently no diffrax solver that recovers it.

### Making the jax backend start fast

A fresh process pays before its first solve, and the cost has two halves with
different remedies. Isolated, on the coupled thermal problem:

```
                        cold        remedy                      after
trace + lower          579 ms       jax.export (deserialize)     0.8 ms
XLA compilation        803 ms       persistent on-disk cache      95 ms
```

**Shipped: the persistent cache.** One config line, no new dependency, no
behaviour change. `IMR_FAST_JAX_CACHE` picks the directory; setting it empty
turns it off. The thresholds are zeroed because jax's defaults skip short
compilations and here every one is worth keeping.

Through the public API it is worth about 16% of first-solve rather than the 8x
the isolated number suggests -- min 1804 ms against 2150 ms over three runs
each -- because tracing and imports dominate once XLA is cached. Steady-state
solves are ~64 ms either way.

**Not shipped: `jax.export`.** It works, and it is the larger half: serialising
the lowered StableHLO makes a fresh process skip tracing entirely, taking the
pre-run cost from ~1382 ms to ~161 ms with the same answer to 7.81e-07. Three
things stand in the way, and the third is the real one:

- `EQX_ON_ERROR=nan` is required. diffrax reports solver failures through
  equinox host callbacks and those cannot be serialised; the flag turns them
  into NaN instead, which `_jax` would still catch through its finiteness check,
  but it is a change in error behaviour.
- `flatbuffers` becomes a dependency of serialization.
- **A stale blob silently integrates old physics.** The exported artefact is a
  frozen copy of the right-hand side. Invalidation has to key on the jax
  version, the platform, the problem shape AND the source of every function that
  went into the trace -- and getting that wrong produces correct-looking
  trajectories from code that no longer exists. That is a worse failure than
  waiting a second, so it needs its own change with its own tests rather than
  riding along with the cache.

### What the jax backend now covers

Forcing and the distributed-memory materials came in for the price of the same
two mechanisms stage 2a and stage 4 already introduced.

**Forcing.** `wave_type` and the amplitude are configuration and stay Python
branches. The WINDOWS are not -- they test `tn`, the integration time, which a
tracer supplies -- so those became `where`. One consequence worth stating: both
arms of a `where` are evaluated, so the histotripsy cosine is clamped rather
than guarded. Outside its window `c` would go negative and `c ** (mn - 1)`
produce a nan, and a nan selected AGAINST still poisons a gradient.

**Distributed memory.** Giesekus and linear PTT pack `2*points` derivatives --
480 at the default -- into a preallocated buffer, which is why that branch never
joined the list the mechanical path builds. `at_set` fills it for both backends.
`_distributed_stress`, `_distributed_stress_integral` and
`_distributed_dissipation` take `xp`; the Python interpolation loop inside the
last is the object-dtype branch that only the Dual route reaches, so jax takes
`interp` and never sees it.

Measured against scipy, 150 points to 25 us:

```
gaussian forcing      1.63e-06     giesekus              3.49e-07
heaviside step        8.45e-08     linear PTT            3.32e-07
histotripsy pulse     2.33e-06     giesekus + medtherm   4.68e-07
```

Still refused, and both for the same underlying reason -- a search or a solve
whose trip count depends on the data:

- `masstrans=1`, whose wall closure is a secant with a fallback ladder
- `sampled_forcing`, whose interpolation searches its own knots

### Sensitivities reach the public API

`solve_with_sensitivities` dispatches on `backend`. Until it did,
`sensitivities_jax` was written, tested and unreachable: the field was ignored
there, so a config asking for jax got a jax forward solve and Dual-route
tangents. `backend` has to mean the same thing for derivatives as for
trajectories.

Every field of the result matches the scipy path, not just the radius:

```
NHKV,  dG           worst 1.3e-06 across all fields   trajectory 5.7e-08
Zener, d(G, mu)     worst 5.2e-07                     trajectory 1.7e-08
```

`state`, the tangent of the whole state vector, and the `SimulationResult`
embedded in the sensitivity result are included in that comparison.

Two refusals, both by name rather than a silent fallback:

- `bubtherm=1`. The thermal outputs are produced by replaying the boundary
  closure per time point with a carried wall state, which under trace unrolls to
  one boundary solve per output time -- three hundred of them in a single graph.
  That is its own piece of work.
- Any parameter outside the five material scales. `R0` and the physical
  parameters would need `params` traced through more than its `scales`
  override.

Falling back quietly would be worse than either: it would make one config field
mean two different things depending on which method you called.

### Mass transfer: attempted, and the blocker is root precision

The wall closure is a secant with a fallback ladder, and the plan called for
`lax.custom_root`. A simpler thing works for the iteration itself: a fixed-trip
secant, since at the root the update is already zero and extra iterations are
no-ops. It matched `_secant_root` to 2.2e-16 on three residual shapes and
differentiated exactly, recovering d(sqrt(c))/dc to 1e-16.

It still does not integrate. Three hypotheses, all wrong, each cheap to test and
worth recording so they are not retried:

1. **Warm start.** `_WallState` carries the previous root, and a tracer stored
   there leaks out of the trace, so the traced path restarts from `theta[-1]`
   each step. Measured: warm and cold starts land on the same root to 1e-16.
   Not the cause.
2. **Post-convergence drift.** The real wall root is ~1e-17, so both residuals
   reach the noise floor and the fixed-trip iterate random-walks where
   `_secant_root` would have exited. Freezing the step under tolerance fixed
   that, and the integration still failed.
3. **A discontinuous bracket.** `_secant_root` flips its initial offset sign at
   zero, which is exactly where these roots sit. Removing the flip changed
   nothing.

What it actually is: **the traced root differs enough to make the right-hand
side inconsistent at the integrator's tolerance.** Sampled along the scipy
trajectory rather than at one state, `|dRHS|` between the two backends reaches
**2.19e-09** against `rtol = 1e-08`. An adaptive controller cannot satisfy an
error estimate on a function that disagrees with itself at the tolerance, so it
shrinks `dt` until the step budget is gone -- 1e6 steps, on a problem whose
stiffness ratio is only 1.34e+04 with `|lambda_max| = 196`, where a hundred
steps should do.

Two sampled states agreed to 9e-16, which is why this looked fine before the
trajectory sweep. Sampling where the integrator actually goes is the check that
matters.

So mass transfer needs a root solver whose answer matches `_secant_root` to
much better than the integrator tolerance -- `lax.custom_root` with a proper
tangent solve, or a polish step -- not a namespace fix. Reverted rather than
shipped behind a guard, so there is no dead code waiting for it.

### Mass transfer, second attempt: the formulation was right, the warm start is the wall

Reformulating the closure fixed the conditioning. `theta` vanishes at the
reference state by construction -- `theta(T) = 0` at `T = 1` -- and the wall
starts there, so the root is ~1e-17 against an ABSOLUTE tolerance of 1e-13, four
orders the wrong way. Solved in `Tw` the root is O(1) and the same tolerance is
effectively relative. `kirchhoff_theta` converts back exactly; #57 used the same
substitution for the medtherm closed form.

With that, mass transfer integrated on jax for the first time:

```
span     max |dR/R0|    steps    min R/R0
 2 us      1.37e-12        13      0.996
 5 us      8.92e-12        22      0.976
10 us      7.87e-10        37      0.900
18 us      1.82e-08        64      0.610
20 us      FAILED                  0.461
```

Thirteen to sixty-four steps, against a million exhausted before. The roots
agree with `_secant_root` to 2e-16 on captured inputs.

What stops it is the **warm start**, not the root. numpy carries the previous
step's root in `_WallState`, which tracks the wall as it heats through the
collapse; a tracer stored there escapes the trace, so the traced path restarts
cold every step. `theta[-1]` is frozen at the reference value -- its slot is
algebraic -- so the guess stays put while the true root climbs away. Starting
from the adjacent interior node `theta[-2]` reaches R/R0 = 0.46 but degrades
agreement to 4.5e-03, well outside the 1e-05 bound, and the deep collapse
(0.09) still fails.

Moving that boundary needs one of: the wall temperature promoted to a real state
variable so the warm start rides in the state legitimately, or a bracketing
solver (Brent) that converges from any start rather than a local iteration that
needs a good one. Both are real changes; neither is a namespace fix.

Reverted again rather than shipped, because a backend that silently loses
accuracy through the collapse is worse than one that refuses the configuration.

### Mass transfer, resolved: the closure was solved outside its physical domain

The literature settled it. Barajas & Johnsen -- the source of IMRv2's
mass-transfer model -- close the wall with equilibrium phase change,
`rho_w C_w = pvsat(Tw)/(Rv Tw)`, and solve the resulting algebraic condition
"using an algorithm based on a combination of bisection, secant, and inverse
quadratic interpolation". That is Brent's method: a **bracketing** solve. The
warm-started plain secant IMRv2 uses, and this package inherited, is a deviation
from the reference method, and the deviation is what all the trouble above was.

Two facts make the bracket obvious once looked for. `kv` is a **mass fraction**,
so `(0, 1)` is physics rather than convenience. And `_kv_of_T` reaches 1 exactly
where `pvsat(Tw) = P` -- so the residual's `1/(1 - kv)` **pole sat precisely on
the edge of its own admissible range**, and iterating from a guess walked through
it. Measured over a 25 us NHKV collapse: 26 of 1480 wall solves returned mass
fractions outside `(0, 1)`, as far out as +179 and -603, every one of them in the
last 2% of the solve.

The fix is two changes, both forced by the above:

- **Solve for `kv`, not `theta_bw` or `Tw`.** `_T_of_kv` inverts the equilibrium
  condition in closed form, so admissibility becomes a bracket -- and a
  *constant* one, the same for every state. In `Tw` the interval is `(0, Tsat(P))`
  and has to be recomputed per state.
- **Multiply through by the denominator.** `D = (kv*Rva_diff + Rg_star)*Tw*(1 - kv)`
  is strictly positive inside the bracket, so clearing it removes the pole while
  preserving every root, and the product extends continuously to both endpoints.

Measured on the same 1480 solves: exactly one sign change each, finite
throughout, roots in `[0.472, 0.804]`. One root on the bracket means **no branch
has to be chosen**, which is what finally makes the closure a function of state
alone.

What that bought, beyond correctness:

| | before | after |
|---|---|---|
| wall solves outside `(0, 1)` | 26 of 1480 | **0** |
| complex-step vs `Dual`, coupled+mass | 1e-13 | **5.2e-15** |
| `errstate` suppressions in the package | 1 | **0** |
| masstrans+medtherm solve | 314 ms | 511 ms |

`_WallState`, `_secant_root`, the four-rung fallback ladder and the package's
last floating-point suppression are all deleted -- the warm start was the only
thing that needed any of them. Removing it also cleared 12 baselined pyright
errors in `_rhs.py`.

**The accepted trajectory does not move**: 1.4e-12 relative over 2000 samples,
and the pinned references are unchanged at printed precision. The excursions were
on trial steps the integrator later *rejected*, so they never reached the
solution -- which corrects the prediction made when this was authorised, that
the deep-collapse trajectory would move and `ref_masstrans_medtherm.csv` would
have to be rebaselined. It does not. IMRv2's trajectory is fine here; what was
broken is the closure's robustness and its purity. A rejected step's garbage root
is one accuracy test away from being kept, and warm-starting from it is what made
the right-hand side depend on the integrator's step history rather than on `(t, y)`.

The cost is real and is the bracket's, not the algorithm's: brentq needs ~10
residual evaluations from a width-1 bracket against the warm-started secant's
~4. Measured and rejected as levers -- `brenth`/`toms748`/`ridder` (all slower or
equal), loosening `xtol` from 1e-15 to 1e-4 (10%), and narrowing the bracket
around the state's own `kv[-1]` (a poor hint: median gap 0.144, inside +/-0.1
only 16.7% of the time).

Mass transfer is still refused on jax, but for a mechanical reason now --
`brentq` wants concrete floats -- rather than because the closure tracked a
branch through history. A traced bisection on the same constant bracket is the
remaining work, and it is a namespace-level change.

### Mass transfer reaches the jax backend

With the closure a function of state alone, what remained was mechanical.
`_thermal._traced_root` bisects the same constant bracket a FIXED number of times,
so the traced program has a fixed shape -- and because `xp.where` and a sign
comparison are the only namespace operations it needs, `_thermal` requires no
`lax` primitive and no jax import. The `lax.custom_root` this was expected to
need never came up.

The two paths keep different solvers, and that is a measured choice rather than
an oversight. Brent needs ~12 residual evaluations; the traced bisection needs 20
halvings plus 3 quadratic polish steps, because bisection cannot use a
data-dependent iteration count. Forcing both onto bisection was tried:

| solver on the numpy path | ms | max dR/Rmax vs brentq |
|---|---|---|
| brentq (shipped) | 476 | - |
| bisect 20 + 3 polish | 843 | 7.7e-13 |
| bisect 14 + 3 polish | 707 | 5.7e-13 |
| bisect 10 + 4 polish | 668 | 2.2e-12 |
| bisect 8 + 4 polish | 618 | 8.9e-13 |
| bisect 6 + 5 polish | 627 | 3.3e-13 |

The reason to unify would have been bit-identity across backends. It does not buy
that: unifying left cross-backend agreement **unchanged to every printed digit**
(1.44e-12 / 1.07e-11 / 2.46e-10 / 2.97e-09 / 3.23e-08 across the five spans
below, before and after). The residual difference is the *integrators* -- scipy's
LSODA against diffrax's Tsit5 -- not the root-finder, so unifying would have cost
30% on the numpy path for nothing. The polish counts differ for the same reason:
`_bracketed_root` shares one slope across two steps because Brent has already
converged and the steps exist only to carry a tangent, while `_traced_root`
recomputes the slope because it is still converging.

Mass transfer on jax, against the scipy trajectory:

```
span     max |dR|/Rmax    nfev    min R/R0
 2 us       1.44e-12        11      0.996
 5 us       1.07e-11        19      0.976
10 us       2.46e-10        30      0.900
18 us       2.97e-09        51      0.610
25 us       3.23e-08       191      0.133
```

Compare the previous attempt, which failed outright at 20 us and R/R0 = 0.46.
The full collapse to 0.133 now runs, and `coupled+mass fd` joins the
parametrised cross-backend suite at max 1.47e-07 / median 1.89e-09.

Compile time does not suffer from unrolling 29 residual evaluations into the
graph: 1172 ms on the first call against 1532 ms for coupled-without-mass.

`unsupported_reason` is down to one entry, `sampled_forcing`. jax *sensitivities*
still cover only the mechanical path -- `sensitivities_jax` builds its `args` with
the thermal slots zeroed -- and refuse thermal configurations by name, which is
unchanged and separate from this.

### Thermal forward sensitivities on the jax backend

The `bubtherm=1` refusal in `sensitivity._jax_sensitivities` was standing in front
of a duplicated argument list, not a missing capability. `sensitivities_jax` built
its own `_rhs` positional tuple with every thermal slot zeroed -- `bubtherm=0`,
`medtherm=0`, `masstrans=0`, `medium=None` -- so the traced program was mechanical
whatever the configuration said. `_rhs_args` is now one definition, shared with
the forward path in `_integrate_prepared`.

**The prepared thermal operators pass through as constants, and that is correct
rather than convenient.** The medium's flux weights are built from `chi`, `iota`,
`Fom` and `L_heat_star`, and its grid powers from `Lt` -- all thermal or
geometric, none derived from the five material scales this traces. `Br` does carry
viscosity, but it lives in `p`, which is rebuilt from tracers. So no jax
counterpart to `_dual_medium` is needed. Checked against the scipy tangents rather
than argued from the parameter list alone.

Two real defects surfaced once the thermal cases could run at all:

**The pressure output used the wrong branch.** `outputs` computed the internal
pressure from the polytropic closed form, which holds only while it is algebraic
in the radius. With `bubtherm=1` it is a state variable. Every thermal pressure
tangent was 8.0e-01 wrong while radius, velocity and stress agreed to 1e-07 --
one output wrong and the rest right is what pointed at the closed form rather
than at the tangent machinery.

**The three thermal output tangents were hardcoded to None.** Worse than a
refusal: `backend="jax"` returned a `SensitivityResult` whose thermal tangents
were silently absent while the scipy one filled them in, which is the exact
asymmetry `_jax_sensitivities`' own docstring argues against. They are built with
`jax.vmap` over the time axis. The numpy path's per-timestep Python loop would
unroll one wall closure per output time into the graph, and with mass transfer
each copy carries the bracketed solve's 29 residual evaluations -- ~1700 traced
for a 60-point request against one for the mapped form.

Agreement with the scipy sensitivity path, worst field per case, default
tolerances:

| case | worst field | rel |
|---|---|---|
| mechanical | internal_pressure_pa | 3.3e-07 |
| bubtherm fd | bubble_temperature_k | 2.1e-06 |
| bubtherm spectral | bubble_temperature_k | 8.6e-06 |
| coupled fd | bubble_temperature_k | 1.8e-06 |
| coupled spectral | medium_temperature_k | 6.1e-05 |
| coupled+mass fd | internal_pressure_pa | 8.6e-07 |

The spectral medium temperature is the one case outside the 1e-05 bound the other
tangent tests use, and it is INTEGRATOR error rather than a discrepancy. Tightening
both integrators together: 1.2e-01 at 1e-07, 1.6e-03 at 1e-09, 5.3e-07 at 1e-11,
7.7e-10 at 1e-13 -- eight orders over six of tolerance, so the two backends compute
the same tangent. Bounds are therefore per case at ~5x measured, on the precedent
argued in `test_validation_trajectories`, with the convergence asserted separately
as the property-level claim.

`sensitivities_jax` now returns a mapping rather than a flat tuple. It returned
ten positional values by the end, and callers reached in with `*_, tangent`, which
silently picked up the vapour tangent -- None for a mechanical config -- the moment
the thermal fields were appended.

### Collapse shooting was never a jax blocker

Listed as one for most of W11, and it is not. Collapse shooting is a
`prepare`-time computation -- `solve_ivp` with a terminal maximum-radius event, a
bracket-expansion loop, and a `brentq` over the two -- and a traced program never
sees any of it. It produces `problem.initial_state`, a concrete array, and the
traced solve starts from that like any other. The three parts that would be hard
to trace all happen before tracing begins.

Measured rather than reasoned: Zener with `collapse=CollapseInitialization()`
agrees between backends at 1.58e-06, and now has a test saying so. Differentiating
THROUGH the shooting is still out, but that needs `R0` and `Req` as traced
parameters -- a `SCALE_PATHS` limitation, filed below, not a collapse one.

### Sampled forcing: `unsupported_reason` is now empty

Three data-dependent uses of `tn`, none of which needed a new algorithm.
`searchsorted` exists in both namespaces and takes a traced needle; the index it
returns is clamped rather than compared; and the coefficient rows are read through
`xp`, because a tracer cannot index a numpy array at all.

The out-of-range early return became a multiplicative 0/1 MASK rather than an
`xp.where` over the results, and that choice is load-bearing: `where` is not a
ufunc, so `np.where` on a `Dual` returns an object array and would have broken the
tangent path silently. The mask is built from plain floats in every arithmetic and
the polymorphic multiply applies it. Verified bit-identical across the change on
the numpy trajectory (both inside and past the last knot) AND on the Dual tangent.

numpy's `complex128` supports ordering comparisons on the real part, so the
complex-step path takes the same code unchanged -- checked, because a `>=` that
raises for `complex` would have been a silent regression on the one route the
bit-identity harness does not cover by default.

Measured against scipy: 8.2e-06 mechanical, 9.8e-08 coupled fd, 6.1e-06 past the
last knot.

`unsupported_reason` now returns None for every configuration. The function is
kept rather than deleted: its shape is what makes a future restriction a named
refusal at construction instead of a tracer error three frames inside diffrax.

### Configuration scalars as traced parameters

`CONFIG_PATHS` -- `R0`, `Req`, `T8`, `pA`, `omega`, `TW`, `DT` -- needed no
`SCALE_PATHS`-style indirection: `params` already takes all seven positionally, and
its only `np.` call is `sqrt(P8/rho)` on concrete physics values. `physics.*` and
`initial.*` stay out, because tracing one means constructing a dataclass from a
tracer and `__post_init__` validates with `np.isfinite` -- the same wall the
material scales hit.

What they did need was FOUR places that were reading a concrete field where a
traced value belonged. Each showed up as one wrong output with the rest right,
which is what located them:

| site | symptom |
|---|---|
| `derived` scaled by `config.R0` | wrong tangent for the output named after it |
| `_thermal_fields` scaled by `config.T8` | 1.11 relative error, temperatures only |
| `initial_state_vector` not rebuilt | `Pb`, `kv0`, `Uc` contribute zero to the start |
| medium wall weights not rebuilt | see below |

The medium is the subtle one, and the earlier claim in this plan that it needs no
jax counterpart to `_dual_medium` was **too strong**. It holds for the five
material scales. It does not hold for `R0` or `T8`, because the weights are built
from `chi = T8*K8/(P8*R0*Uc)`.

Why only `T8` showed it: **`_wall_theta_bw` is invariant to a common scaling of the
weights.** Multiply `chi` by any factor and `b`, `c` and `k` all scale with it,
which cancels out of `s = (-b + sqrt(b*b - 2*alpha*c*k))/c`. `R0` enters only
through `chi`, so a constant medium gave it a correct tangent by accident. `iota`
multiplies `grad_Tm` alone, so `T8` is not a common factor and it plateaued at
1.30e-02 however tightly either backend integrated.

Then rebuilding the medium for `_rhs` but not for `_thermal_fields` -- which runs
the same wall closure -- left the two temperature tangents at 5.75e-04, again
tolerance-independent, while all five state tangents converged. `_rhs_args` now
takes `medium` as a REQUIRED keyword so a second consumer cannot silently use the
prepared one.

**Convergence is what separated missing terms from integrator error throughout**,
and a finite difference of the forward solve is what settled which backend was
wrong when both were plausible: at the 5.75e-04 stage scipy sat 1.9e-09 from the
difference and jax 5.75e-04, so the defect was mine.

Agreement after all four fixes, worst field per case: 2.3e-06 mechanical R0,
5.2e-07 Req, 1.2e-06 pA, 2.1e-07 T8-with-vapour, 3.2e-05 coupled R0, 1.1e-04
coupled Req, 1.2e-05 coupled T8, 7.9e-07 mass-transfer R0 -- all converging.

`initial_state_vector` and `medium_with_parameters` are shared with `prepare`
rather than reimplemented, on the `_rhs_args` precedent from the thermal work: the
duplicated argument list there is exactly how the thermal slots came to be zeroed
without anyone noticing. Verified bit-identical across the refactor on 14 cases --
every forcing, collapse initialisation, mass transfer, and both tangent routes.

Removing `_pinf`'s `if ee == 0.0` early-out was part of this: `ee` scales with
`pA`, so that was a Python branch on a traced value. Its own comment had asserted
`ee` was configuration, which stopped being true. No compensating arithmetic was
needed because every arm is already proportional to `ee`.

### The traced sensitivities were never compiled, and it inverted the migration

`integrate_jax` caches a `jax.jit` per (caller, shape). `sensitivities_jax` did
not, so every call retraced the whole `jacfwd`. Measured against the compiled
numba route on the mechanical sensitivity solve:

| route | ms |
|---|---:|
| `_mechanical.py` (numba) | 45 |
| complex step (the fallback if it went) | 87-105 |
| jax, retracing every call | **1121** |
| jax, jit cached | **5-11** |

So the jax path was **24x slower** than the file the migration exists to delete,
and is now **3-10x faster** -- a 100-200x swing from adding a cache that the
forward path already had. Any stage 5 argument made on the earlier number would
have been backwards, which is why this is recorded rather than folded in silently.
First call is 2.6-3.5 s of tracing and compilation, amortised over repeated solves,
which is what inference and design do.

Jitting required two guards inside `params` to stop branching on values the traced
path differentiates. Both had worked under `jacfwd`, which evaluates a JVP with
CONCRETE primals, and both raised `TracerBoolConversionError` under `jit`, which is
fully abstract. That distinction is the lesson: `jacfwd` alone does not prove a
function is traceable.

- `if Pv_star > 0` is really asking whether vapour is on, since `pvsat` is an
  exponential. Reads `vapor` now, which is discrete configuration.
- `if lam1 > 0` is really asking whether the material HAS a relaxation time, which
  its type fixes and `_material_scales` reports as 0.0. Reads the concrete scales.

Verified bit-identical on the same 14 cases, and the `kv0`/`De`/`LAM` edge values
for a no-vapour, non-relaxing material are unchanged at exactly 0.0.

### Stage 5 scope, with jax made mandatory

Measured, the deletion is gated on a dependency decision rather than on
capability: `_mechanical.py` is dominated by the jax path but only for someone who
has jax, and `_dual.py`/`_complex.py` are the route for someone who does not.

Resolved by making jax mandatory. That drops Python 3.10 and 3.11 -- jax requires
3.12 -- and puts ~100 MB of jaxlib in a core install. Both are real costs and were
confirmed rather than assumed.

#### Stage 5a: jax is mandatory, and `_mechanical.py` is gone

jax and diffrax are core dependencies. `requires-python` goes to 3.12 because jax
requires it, so **Python 3.10 and 3.11 support is dropped** -- a breaking change,
confirmed rather than assumed, and the reason the optional-jax design existed at
all.

What that removed, beyond the file:

| deleted | lines |
|---|---:|
| `_mechanical.py` | 611 |
| `_rhs_mechanical_compiled` + `_compiled_mechanical_outputs` + `_prepared_arrays` | ~85 |
| the numba route's two consistency tests | ~180 |
| `available()`, `unsupported_reason()`, `_MISSING`, the import guard | ~50 |
| `requires_jax` scaffolding and the two "nothing is refused" tests | ~55 |

`_mechanical.py` had no configuration to itself. `use_compiled_mechanical` fired
for `not bubtherm and forcing is None`, and every such configuration is also
complex-step-supported -- so it was an accelerator for a route that already
existed, at 45 ms against complex step's 87-105 ms. `backend="jax"` does the same
work in 5-11 ms. It was the duplicate AND the slow path.

Deleting it also deleted the test that existed because it could drift.
`test_rhs_consistency.py` opened by saying "every physics law is implemented twice
... nothing else asserts that the two agree", and three of four physics changes
during the parity work had introduced a bug in the second copy. That file is now 74
lines covering the one duplication that remains: `_dual_medium`'s rebuild of the
wall stencils.

`unsupported_reason` went with it. It had been reduced to `return None` for every
configuration, and a function that refuses nothing plus a test asserting it refuses
nothing is not a safety net.

#### Stage 5b: the parameter gap closes, and one blocker remains

`physics.*` and `initial.*` are traced now. Not by passing them beside the
structure as `scales` does -- `params` reads twenty-five `physics` fields at
scattered sites and `initial_state_vector` five off `initial`, so an override
argument would have put the same lookup in thirty places. `_jax._Overridden`
substitutes at ATTRIBUTE ACCESS instead: a read-only wrapper whose `__getattr__`
returns the traced value for overridden names and delegates the rest. No read site
changed.

Two `xp` leaks fell out of it. `Uc = np.sqrt(P8/density)` breaks once `P8` is
traced, and `params` called `_mu_of_A`/`_mie_F` at their `xp=np` defaults while
`Cstar` carried a tracer.

`mn` was also missing from `CONFIG_PATHS`, and was found by **enumerating** the
scalar fields of the config, `physics`, `initial` and the material, keeping the ones
the scipy route accepts, and diffing against what the traced path covers -- not by
reading the code. That check is now a test, because the enumeration is the only
thing that makes "the traced path covers everything" an assertion rather than a
belief. It reports **0 of 28 paths refused**.

Its tangent converges: `dR/dmn` agrees at 8.7e-05 at rtol 1e-08 and 1.1e-07 at
1e-11. The first attempt measured identically zero, because the histotripsy window
is `DT +/- pi/omega` = [2e-05, 4e-05] and the sampled span stopped at 2e-05 -- a
test that would have passed while testing nothing.

#### Differentiating through the collapse shooting

The last blocker, and it needed neither a differentiable event nor a differentiable
root-find. Both were the wrong framing. Two implicit conditions define the answer,
and both can be applied AFTER a fixed-endpoint solve:

    v(T) = 0        the precursor is at a maximum
    R(T) - 1 = 0    that maximum is the observed one

`prepare` already locates `(v0, T)` concretely by shooting in numpy, so the traced
flow integrates to that FIXED `T` -- no event to differentiate. Then, writing `y'`
for the right-hand side at the endpoint:

    dT/dq  = -(dv/dq) / y'[1]
    dm/dq  =  dm/dq|_T + y'[2:] * dT/dq
    dv0/dq = -(dR/dq) / (dR/dv0)

and `dR/dq` needs no `dT` term at all, because the velocity is zero at the maximum.
That is what decouples a 2x2 implicit system into two scalar divisions, and it is the
same arrangement the deleted `Dual` route used -- the algorithm was already the right
one, only its implementation was tied to `Dual`.

`CollapseStats` gained `maximum_time_nondimensional` to carry `T`. The result is
returned as a value plus a CONSTANT Jacobian and applied inside the trace as a linear
surrogate: exact in value at the point, exact in first derivative, which is all a
tangent is.

| | |
|---|---:|
| starting memory tangent vs scipy | 1.7e-10 to 2.8e-08 |
| radius tangent convergence | 3.7e-02 -> 2.3e-04 -> 7.8e-07 |
| memory tangent convergence | 6.4e-09 -> 3.5e-10 -> 9.4e-12 |

The loose default-tolerance figure is integrator error amplified by the Zener
collapse, so the test asserts convergence rather than a single bound.

#### Stage 5b: the deletion is blocked on the CACHE KEY, not on capability

Every capability gap is closed -- all 28 parameter paths, and the collapse tangent
above -- and the deletion was written, run, and reverted anyway.

`sensitivities_jax` caches its compiled `jacfwd` on `id(problem)`. `inference.py`
calls `simulate_with_sensitivities`, which is `prepare(config).solve_with_sensitivities(...)`,
so it builds a FRESH `PreparedProblem` on every likelihood, gradient and EIG
evaluation -- lines 331, 355 and 376. The key therefore never hits, and each
evaluation pays a full retrace:

| | per sensitivity solve |
|---|---:|
| numpy route (complex step) | 45-100 ms |
| traced, same problem reused | 5-11 ms |
| traced, fresh problem each call | **~2500 ms** |

That is a 25-50x regression on the workload this package exists for, and it is
what made the suite go from 4:30 to over 7 minutes at 38%. Measured, not inferred:
four problems differing only in `R0` took 2019, 2445, 2536 and 2635 ms.

The cause is that `id(problem)` keys on OBJECT IDENTITY where the traced program
depends only on structure. Two configurations differing in `R0` produce the same
graph with different constants -- so the constants belong in the arguments, not the
closure. Fixing it means keying on shape (layout, material type, flags, grid sizes,
paths, grid length) and passing every non-differentiated scalar as a traced
argument, including the ones reached through `physics`, `initial` and the collapse
surrogate. That is real work with real room to bake in the wrong thing, and it is
the actual remaining gate on stage 5b.

Kept from the attempt: the collapse tangent, the parameter coverage, and
`CollapseStats.maximum_time_nondimensional`. Reverted: the deletion and the
always-traced routing.

### The cache key, and how it kept lying about jax

`_COMPILED` keyed on `id(problem)`. Both entry points that matter prepare per call --
`simulate(times, config)`, and `inference` on every likelihood evaluation -- so the
cache never hit where it counted, and a fresh but IDENTICAL configuration paid a
full retrace: 2500 ms against 1.3 ms.

It produced three wrong measurements in this work before being found, twice making
jax look slower than it is. Once as "jax sensitivities are 24x slower than the numba
mirror", which nearly argued stage 5 backwards. Once as "the forward backend is 0.08x
to 0.64x", i.e. jax slower on everything, when prepare-once gives 1.3x to 10.3x.
And once inside the first `jacobians`, which called `prepare` itself and so
reproduced the very fault it existed to fix.

Keyed on CONTENT now -- a recursive hash of everything the traced closure can read.
The faster repair, a structural key plus promoting the varying constants to
arguments, was rejected on risk: the closure reads the prepared arrays, the untraced
scalars, `physics`, `initial` and the collapse surrogate, and any one left behind
would silently reuse the first problem's value for the second. Content hashing cannot
do that -- differ anywhere and you get a different program. It costs microseconds
against a millisecond solve, and the eager collapse integration is cached on the same
key.

### Bayesian optimal design, and the MCMC path

"The MCMC path falls out for free" was **wrong**. MCMC varies the parameters every
step, so each configuration is genuinely different content and misses legitimately.
What it needed was the parameters as ARGUMENTS, which is the same `at=` mechanism
BOED needed:

| | scipy | jax |
|---|---:|---:|
| MCMC step, `evaluate_with_jacobian` | 70.8 ms | **11.0 ms** |
| BOED, 64 draws | 6.31 s | **77 ms** |

BOED's draws differ only in the values of the inference parameters, so one `prepare`
serves all of them and `vmap` maps over the points. Batched matches sequential to
6e-13, and the EIG to 2e-16 on the same backend.

`batched=True` is opt-in, and refuses `max_failure_fraction` rather than accepting it
and ignoring it: one traced program fails as a whole, so per-draw failure accounting
has nothing to attribute to.

**Switching the likelihood's integrator changes the model.** Routing `evaluate`
unconditionally through the traced path broke four estimator tests, one asserting the
residual is exactly zero at the truth -- a legitimate property, since observations
generated on one backend do not match the other's prediction to machine precision
(4.8e-03 whitened). The mixed state was worse than either: with `evaluate` on scipy
and the gradient traced, the two are derivatives of different functions and their
consistency check fell from 5e-08 to 1.7e-04. So the pair dispatches on
`config.backend`, together.

That dispatch then made the new batching test vacuous -- with no backend declared it
compared the loop against itself and reported a flattering 2.0e-16. Pointed at
`backend="jax"` it reads 8.9e-15 and tests batching.

### The pinned bands hold on a jax forward backend

Checked before proposing the removal, by forcing every configuration in
`test_validation_trajectories.py` onto `backend="jax"`: **38 passed**, with the
deviations essentially unchanged -- collapse Zener 1.46e-03 against scipy's 1.47e-03,
coupled NHKV 7.15e-05 against 7.04e-05, masstrans 2.89e-04 against 2.90e-04.

### Why the non-jax backend stays, after three attempts to remove it

Every capability gate is cleared -- all 28 differentiable paths, the collapse
shooting tangent, sampled forcing -- and the deletion was written three times and
reverted three times. Each attempt was stopped by something only measurement found,
and the third is a hard blocker rather than a cost.

**Attempt 1 was stopped by a silently wrong tangent.** The traced path received the
collapse stress state as a constant, so its tangent was zero -- relative 1.00, wrong
outright rather than imprecise. Caught by running the existing validation suite rather
than only the new tests.

**Attempt 2 was stopped by the cache key.** `sensitivities_jax` keyed on
`id(problem)`, and `inference` prepares per evaluation, so MCMC would have paid 2.5 s
a step against 45 ms. Fixed since -- content keying, plus `at=` for the varying
parameters -- and that fix is what took MCMC to 11.0 ms and BOED to 77 ms.

**Attempt 3 was stopped by stiffness, and this one does not have a workaround.**
Removing the scipy forward solver breaks grid refinement outright. The bubble thermal
PDE's stiffness scales like `Nt**2`, so an explicit solver's step count does too:

| bubtherm grid | Tsit5 steps | |
|---|---:|---|
| Nt = Mt = 10 | 6,196 | fine |
| Nt = Mt = 20 | 110,572 | fine, 1.6 s |
| Nt = Mt = 40 | > 1,000,000 | **fails** |

`test_thermal_grid_convergence` refines to Nt = 40 and passes on LSODA, which switches
to implicit. Tsit5 cannot.

**That blocker is now removed, and the earlier IMEX note does not generalise.** "IMEX:
tried, and the blocker is compile time" was about KenCarp. `Kvaerno5` with an
`optimistix.Newton` root finder behaves nothing like it -- at Nt = 40 it COMPILES
FASTER than Tsit5, 3.5 s against 44.8 s, because Tsit5's cost there is the unrolled
step budget it cannot meet:

| Nt = Mt | Tsit5 | Kvaerno5 |
|---|---|---|
| 10 | 6,196 steps, 51 ms | 882 steps, 141 ms |
| 20 | 110,572 steps, 774 ms | 927 steps, 382 ms |
| 40 | > 1,000,000, FAILS | 968 steps, 558 ms |

Forward-mode composes with it: `jacfwd` through `ForwardMode` works because optimistix
differentiates the Newton solve by the implicit function theorem rather than unrolling
it. At Nt = 40 Tsit5 needs 44.8 s to compile, 42.5 s to run, and returns a tangent that
is 1.00 wrong because the solve never finished; Kvaerno5 needs 3.5 s and 796 ms and is
right.

**And the two removals are coupled, which was not obvious.** Deleting the numpy
sensitivity route forces the likelihood and its gradient onto the traced path. Leave
the forward solve on scipy and the pair is mixed: their consistency check degrades from
5e-08 to 1.7e-04, so the gradient becomes the derivative of a slightly different
function than the log-likelihood it ships with -- the exact failure
`evaluate_with_jacobian` exists to prevent. So it is one decision, and stiffness gates
it.

#### The dividing line is the GRID, not the physics

Nearly mis-drawn. Two measurements disagreed about whether Nt = 40 fails, and the
difference was neither the grid size nor the time span but `thermal`, which defaults to
`"spectral"`. Chebyshev nodes cluster at the boundaries, so the diffusion operator's
spectral radius grows like `Nt**4` where the finite-difference grid's grows like
`Nt**2`. Steps over 25 us:

| case | Nt | explicit | implicit | |
|---|---|---|---|---|
| bubtherm spectral | 11 | 9,083 | 884 | Tsit5 faster on the clock |
| bubtherm spectral | 25 | 279,694 | 930 | Kvaerno5 5x faster |
| bubtherm spectral | 41 | > 1e6 | 965 | Tsit5 exhausts max_steps |
| coupled spectral | 11 | 7,922 | 778 | Kvaerno5 1.7x faster |
| coupled spectral | 25 | 244,552 | 861 | Kvaerno5 12x faster |
| bubtherm fd | 41 | 11,953 | 826 | Tsit5 4x faster |

So the rule is `bubtherm and thermal == "spectral"` -> implicit, and the
finite-difference grids stay explicit at every size measured. An implicit step costs
roughly ten explicit ones on systems this size, which is the whole reason the rule is
not simply "thermal -> implicit".

The Newton tolerance is stated rather than defaulted, and it is NOT what limits the
tangent: tightening it from 1e-8 to 1e-13 left the sensitivity agreement at 2.26e-04,
2.52e-04, 2.30e-04 while runtime grew 2.6x. That number is integration error at the
configured `rtol`, the same as everywhere else here.

#### What a finite-difference reference can and cannot do

Worth recording, because it was tried and it nearly worked. A central difference of the
forward solve resolves 5.6e-09 to 3.6e-06 across fifteen tangent cases, which is two to
four orders below every defect this suite has caught. It is adequate for missing terms,
which is what the defects were.

Two cases were not straightforward and both misled at first:

- **Collapse** read 2.8e-02 and was STEP-INDEPENDENT -- so not differencing error at
  all, but the integrators' own error at the default tolerance, which the Zener collapse
  amplifies. At rtol 1e-12 it is 1.15e-06.
- **Sampled forcing** was INVERTED: 4.6e-04 at a 1e-03 step, 1.6e-02 at 1e-05, 2.6e+00
  at 1e-07. The forcing is a piecewise cubic, so a smaller step shrinks the signal but
  not the error from knots crossing evaluation points. Usable at 1.8e-04 -- a 170x
  margin against the 3.1e-02 defect it has to catch, against roughly 10^4 elsewhere.
  That is the one place a difference is close to unable to do the job.

### A measurement error worth recording

Four hypotheses were tested against a sweep that rebuilt only `R`, `Rd` and `P`
from the scipy trajectory and left the thermal fields at their INITIAL values --
states the integrator never visits. The 2.19e-09 figure that first diagnosis
rested on was measured there. Comparing roots on inputs captured from a real
solve took two minutes and gave the opposite answer: the solvers agree to 2e-16.

A harness that samples where the integrator does not go is worse than no
harness, because it produces numbers that look like evidence.

### Risks

| risk | why it bites | where it is handled |
|---|---|---|
| ~~**LSODA has no diffrax equivalent**~~ | **Downgraded -- see "Stiffness is static" below.** Stiffness is a property of the configuration, not of the trajectory, so there is nothing to auto-switch | solver chosen at `prepare` time, as scipy already does for BDF |
| **Terminal events** | `_radius_floor_event` is `terminal=True, direction=-1`; diffrax's event model differs | Stage 2; the floor is a guard, so a divergent-solve check may substitute |
| **float32 default** | would pass loose tests and fail tight ones, confusingly | `jax_enable_x64` before any dtype is touched; already required in Stage 0 |
| **Dispatch under `jit`** | the #96 tables are Python-object keyed | closed over at trace time; static per solve |
| **Compile latency** | 1.4 s per distinct configuration | amortised for BOED, but a design sweep that varies the time grid may retrigger |
| **Two backends drift** | exactly the failure mode `_mechanical.py` already has | the cross-backend test must cover every config the JAX backend claims |

---

## Open issues, banked

GitHub is the source of truth; this is the copy that survives losing the issue
tracker. Updated 2026-08-01. Each entry records what was **measured**, because
in four cases (#119, #120, #122, #129) the issue's stated cause or scope was
wrong and the proposed remedy would have been aimed at the wrong thing.

| # | title | state |
|---|---|---|
| 122 | Data assimilation: EnKF / IEnKS / 4D-Var | **done** -- see below |
| 120 | Spectral thermal implicit solve is a dense `O(Nt^3)` Newton | **closed**, every claim stale |
| 142 | Remaining pyright diagnostics | **closed**, baseline is empty |
| 124 | Remaining coverage gaps | open, P3, the valuable item done |
| 123 | CI test step vs the 5-minute target | closed **prematurely**, still unjudged |

### #120 -- dense thermal Newton, closed

Every claim in it was stale by the time it was re-measured on 2026-08-01, and
none of the proposed work was needed. Kept because the shape of the mistake is
the useful part.

- "does not complete at Nt=200" -- completes in **54.3 s**. The numbers predated
  the `VeryChord` root finder; above `_CHORD_ABOVE = 80` the solve reuses the
  Jacobian instead of refactorizing every Newton iteration
- "cost grows like `Nt^3`" -- measured `Nt^1.3`: 8x the nodes for 15x the time
- "exploit sparsity in the thermal Jacobian" -- Chebyshev differentiation
  matrices are dense; there was never sparsity to exploit
- "the spectral convergence rate is no longer asserted" -- already restored,
  `_REFERENCE_NT = 200`
- my own narrowing, "high-resolution *sensitivities* capped near Nt=80" -- also
  wrong: sensitivities run to **Nt=300 in 33 s**, finite throughout

### #122 -- data assimilation, done

Delivered across #146, #147, #148, #149, #150, #151, #152:

| piece | entry point |
|---|---|
| window restart from a raw state | `PreparedProblem.solve_from`, `solve_states` |
| tangent linear operator | `PreparedProblem.state_tangents` |
| ensemble propagation | `PreparedProblem.solve_ensemble` |
| EnKF analysis | `assimilation.enkf_analysis`, `ensemble_update` |
| true 4D-Var | `assimilation.four_dvar`, `variational_cost` |
| iterative smoother | `assimilation.ienks` |

**The issue's scope was wrong in the direction of overstating the work.** It said
initial-state tangents were missing and `start` had to be lifted out of the
traced function. `integrate_jax`'s inner `solve(grid, start)` already took the
start state as an argument, so `y0` was in a differentiable position all along;
what was missing was a `jacfwd` over it with the `ForwardMode` adjoint.

The validation it asked for reproduces, but the honest statement is narrower
than "gradient methods win":

  velocity rms (never observed)   prior 2.5e-02
                                  4D-Var 1.7e-04
                                  one-shot ensemble smoother 3.5e-03
                                  IEnKS 1.9e-04

4D-Var and IEnKS **agree** -- 3.2e-09 apart after 12 iterations. Only the
one-shot ensemble update is beaten, and no ensemble size fixes it: 8x the
members moved the answer in the fourth significant figure, so that gap is
structural rather than sampling error. What separates 4D-Var from IEnKS is what
they require -- the tangent operator, or an ensemble spanning the subspace --
not what they deliver.

Sliding windows remain out of scope by maintainer decision.

### #124 -- coverage gaps, narrowed to one group

Two of the three groups are done, and the second was mis-described here and in
the issue.

**Distributed stress rate (#145).** The trapezoid branch of the explicit `dS/dt`
was unreachable by any test: the default quadrature is `gauss` and the default
radial is `1`, so nothing selected both. Asserted as an identity against a
centered difference -- gauss 1.5e-10, trapezoid 3.1e-11.

**Multiprocessing (#153).** Described as "never run their `workers > 1` branch
under test". They were not untested, they were **broken**, and had been since the
parallel code was written. Two faults, the second hidden behind the first:
`PreparedProblem.parameters` is a mappingproxy so nothing could be pickled to a
worker; and once that was fixed, `ProcessPoolExecutor`'s default fork start
method deadlocked against an initialised jax runtime. Fixed with
`__getstate__`/`__setstate__` and an explicit spawn context. Spawn costs a full
jax import per worker, so `workers > 1` is now slower than serial on small
batches and only pays on large ones.

**Left: failure paths needing a constructed physical state.** Collapse precursor
that never reaches a maximum radius, shooting that cannot bracket, invalid
generalized viscosity mid-solve, unbracketed equilibrium radius, the
`except Exception` around the compiled solve. Each needs a configuration
engineered to fail in one specific way, and the tests are only as good as that
engineering: a state that fails for the wrong reason exercises the line without
checking the guard, which reads as coverage while testing nothing.

### #123 -- CI timing, closed prematurely

Auto-closed by a `Closes #123` line in PR #132 before the evidence it was
waiting for arrived. The diagnosis history is worth keeping because two of the
three explanations were wrong:

1. blamed test selection -- wrong
2. blamed XLA cache keys not being portable across runner CPUs, and added
   `XLA_FLAGS=--xla_cpu_max_isa=AVX2` -- **wrong**; measured a job restoring 450
   entries and reusing them, so keys were already portable
3. the actual cause: the prune cap (700) sat **below** the suite's working set
   (912 programs), so every run pruned the oldest half, which is the half it had
   just restored. It alternated forever. Cap raised to 4000/3000 in #132, and
   the workflow added to the cache key in #134 because `actions/cache` keys are
   write-once and the fix could not otherwise take effect

Still unsettled: whether the AVX2 pin earns its 3.6% throughput cost, and
whether the per-push step now sits under 5 minutes. The flatten in #137 rotated
the key from `src/**/*.py` to `pyimr/**/*.py`, so the cache is cold again and
the next few runs say nothing. Reopen and judge after several warm runs.

### Not an issue, but do not lose it

`_bracketed_root` (numpy) strips the tracer from its finite-difference slope via
`_value`, while `_traced_root` (the jax path) differentiates the identical
expression. Same routine, two behaviours. It is **not** the cause of #133 --
`stop_gradient` on the slope changed nothing -- but the asymmetry is real and
unexplained.

---

## How to benchmark this package

Established by measurement on 2026-08-02, after a performance claim of 1.53x turned out
to be 1.02x. Read this before quoting a speedup.

### The workload is serial. Measure it on one core.

25 repeats of an IDENTICAL coupled spectral solve, by thread regime:

| regime | min | CV | max/min |
|---|---|---|---|
| default (128 cores) | 1.718 s | 19.9% | 1.54 |
| 8 cores | 1.706 s | 19.6% | 1.67 |
| 4 cores | 1.696 s | 19.0% | 1.55 |
| **1 core (`taskset -c 0`)** | **1.692 s** | **3.3%** | **1.11** |

One core is both the **least noisy and the fastest**. The matrices are 25x25 and 78x78 --
far too small for XLA's thread pool, which contributes synchronisation overhead and
variance and no speedup. On the default regime the noise floor (1.54x) EXCEEDS most
effects worth measuring, so any comparison there is meaningless.

Consequences:

- benchmark under `taskset -c 0`, which resolves effects down to a few percent
- for a campaign, run independent solves across cores as separate processes
  (`workers > 1`); do NOT `vmap` the implicit path, which measured 1.8-2.5x WORSE per
  point than solving one at a time
- report `min` over repeats, not mean or median -- `min` was stable to 1.5% where the
  mean moved by 20%

### Use a paired, interleaved design

Sequential blocks -- measure all of A, then all of B -- are not robust to drift. Compile
both programs up front, then alternate, and report the paired ratio distribution. That is
what took the halvings comparison from "1.53x" (single sample, noisy regime) to "1.020x,
range 1.008-1.022, unanimous over 12 pairs".

### A microbenchmark of a component does not predict the system

The wall root-find is 75% of the coupled RHS op count, and one jitted RHS call drops from
105.6 us to 62.5 us when it is cut. The end-to-end solve moved by **2%**. Whatever
dominates a step, it is not what the isolated RHS timing suggests, and the chain
"RHS -> jacfwd is 5.4x RHS -> per-iteration cost" does NOT predict end-to-end behaviour.
Do not infer a system speedup from a component measurement; measure the system.
