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

- [x] Add `__all__` to `imr_fast.py` (42 names). It previously re-exported
      `np`, `solve_ivp`, `brentq`, `csr_matrix`, `lil_matrix`, `dataclass`,
      `field`, `PchipInterpolator`, `perf_counter`, `Mapping`,
      `MappingProxyType`, `Integral`, and `annotations`.
- [x] Add `__all__` to `imr_sensitivity.py` (4) and `imr_inference.py` (7).
- [x] Add regression tests in `tests/test_api.py`:
      `test_all_is_declared_sorted_and_defined` and
      `test_star_import_leaks_no_foreign_names`. The latter walks the
      star-import namespace and rejects modules, foreign `__module__` owners,
      and union aliases whose members are not repo-owned.

**Done when:** `dir(imr_fast)` minus dunders equals the documented API surface.

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
      imr-fast have no upstream counterpart to pin against.
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
  reference for Giesekus/PTT. That reference is unreachable, so imr-fast is not
  behind on those models -- it implements capability upstream advertises but
  cannot run. The 1e-3 reduction errors are imr-fast FD versus its own
  Oldroyd-B limit, not a deficit against a reference. Spectral becomes an
  optional accuracy improvement, not a parity requirement.
- **W3 changes character.** Only Carreau works upstream, and incorrectly.
  imr-fast already exceeds IMRv2 here. Any Powell-Eyring work must be built on
  correct formulas, not ported ones.
- **W4 is resolved.** See below.
- **New work surfaced:** the collapse Oldroyd-B discrepancy and the memoryless
  collapse restriction, tracked in W8 -- where both turned out to be upstream
  stubs rather than imr-fast defects.

---

## W2. Spectral collocation path

The largest structural gap. IMRv2 ships `src/f_imr_spectral.m` (542 lines,
Chebyshev collocation for the thermal and viscoelastic PDEs) alongside
`src/f_imr_fd.m` (585 lines). Critically, `src/f_call_params.m:376--385`
forces `spectral = 1` for `stress == 6` (PTT) and `stress == 7` (Giesekus)
unconditionally — there is no upstream FD path for those models.

imr-fast solves them on a 480-point FD Lagrangian grid
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
pinned trajectory to match and no parity deficit to close. imr-fast is ahead
here, not behind.

What survives as genuine motivation:

- Spectral collocation converges faster than the 480-point FD Lagrangian grid,
  so it is an accuracy and cost improvement on imr-fast's own terms.
- A second independent discretization is the strongest available check on the
  distributed models, precisely *because* no upstream reference exists.

Reframed acceptance: **spectral and FD agree with each other under refinement,
and spectral reaches the same accuracy in fewer points.** Do this only after
W8, which addresses measured discrepancies rather than hypothetical ones.

---

## W3. Viscosity model parity

IMRv2 `src/f_viscosity.m` dispatches seven models: Carreau, Carreau-Yasuda,
Cross, modified Cross, Powell-Eyring, modified Powell-Eyring, simplified Cross.

imr-fast has Newtonian, PowerLaw, CarreauYasuda, Cross, HerschelBulkley,
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

IMRv2 `src/f_radial_eq.m` branches 1--6. imr-fast supports 1--5. The README
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

**Status: resolved.** imr-fast's decision to stop at `radial = 5` is correct
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
| minor gaps | `pv` | imr-fast derives vapour pressure from `T8` via `pvsat`; no direct override |

Only one true gap (`pv`), and it is reachable by adjusting `T8`. imr-fast also
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
      Confirmed numerically: imr-fast reproduces both references **without**
      collapse at **1.56e-05** (Oldroyd-B) and **6.22e-06** (NHKV) -- inside
      the normal pinned band. The 1.51e-01 figure was imr-fast doing real
      physics against an upstream stub.
- [x] **Memoryless collapse: imr-fast is correct to refuse.** IMRv2 accepts
      `collapse=1` with `stress=1` and silently ignores it. Refusing is the
      stricter behaviour; the harness now asserts the refusal so it cannot
      regress.
- [x] Reclassified in `tests/run_validation.py`. `ref_collapse_oldb.csv` and
      `ref_collapse_nhkv.csv` renamed to `ref_coupled_oldb.csv` and
      `ref_coupled_nhkv.csv` -- they pin the fully coupled
      bubtherm+medtherm+masstrans+vapor model, which is **new coverage**: the
      Oldroyd-B case is the first pinned coupled-model trajectory with a
      memory material.
- [x] **Collapse Zener 1.55e-03 explained; imr-fast is the more accurate side.**
      The hypothesis above was right. Resolved quantitatively:

      | quantity | value |
      |---|---|
      | imr-fast S0 (event root-find, velocity = 0) | `-0.1599451098` |
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
      the stress. imr-fast's event root-find on `v = 0` is O(tol).

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
Eulerian form, where advection introduces spatial derivatives. imr-fast is
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
the only defect that cost imr-fast a capability.

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

## Ledger: where imr-fast already exceeds upstream

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
  them out; imr-fast is the only working implementation of the two.
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

| metric | tinygrad | imr-fast | verdict |
|---|---|---|---|
| comment-only lines | 1.2% | 1.0% | already there |
| blank lines | 3.0% | 9.2% | 3x too airy |
| docstrings | ~0 | 82 | strip most |
| indent | 2 space | 4 space | convert |
| largest file | -- | `imr_fast.py`, 2683 lines | split |

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

tinygrad is an internals-heavy library; imr-fast is a scientific API that
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
- [x] Split `imr_fast.py`: extracted the material value objects into
      `_imr_materials.py` (436 lines), registered in `pyproject.toml` and the
      W0 leak test. **2683 -> 2214 lines.**
- [x] Re-measure and record.
- [ ] Collapse blank lines from 10.7% toward 3-4%. **Blocked by the formatter**
      -- see trade-off below.
- [x] Two further splits of `imr_fast.py`, both AST-driven and bitwise-verified:
      `_imr_stress.py` (359) takes the constitutive evaluation -- material plus
      kinematic state in, stress and rate out, with no dependency on the RHS --
      and `_imr_thermal.py` (287) takes the Tait/Mie-Gruneisen parameters,
      `pvsat`, the dissipation terms and the implicit wall-temperature solve.
      **`imr_fast.py` 2683 -> 1686 across the three splits (-37%).**
- [x] `_imr_rhs.py` (315): the right-hand side plus forcing evaluation --
      `_rhs`, `_pinf`, `_sampled_pressure`, `_nZ`, `_radius_floor_event`. A
      clean leaf: it reads a prepared problem but never builds one. The two
      material predicates `_is_distributed_stress` and `_stress_state_count`
      moved to `_imr_materials.py` where they belong, which is what made the
      cut acyclic. **`imr_fast.py` 2683 -> 1396 over four splits (-48%).**
- [x] `_imr_config.py` (568): the frozen config, prepared-problem and result
      dataclasses, their validation, and the physical defaults.
      `PreparedProblem.solve` defers its `_solve_prepared` import, the same
      trick it already used for `solve_with_sensitivities`.
- [x] `_imr_prepare.py` (530): `params`, the `_prepare_*` helpers, the collapse
      precursor and `prepare` itself.
- [x] **`imr_fast.py` 2683 -> 482 over six splits (-82%).** Every module is now
      under the ~800 target except `imr_sensitivity.py` (1001), which is
      untouched and dominated by `solve_with_sensitivities` (146),
      `_material_parameters` (107) and `_output_duals` (101).

      | module | lines |
      |---|---|
      | `imr_sensitivity.py` | 1001 |
      | `_imr_mechanical.py` | 647 |
      | `_imr_config.py` | 568 |
      | `_imr_prepare.py` | 530 |
      | `imr_fast.py` | 482 |
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
         `finite_diff_mat` from `imr_fast`, breaking
         `validate_bubtherm_adiabatic.py`, which nothing in CI ran. Those two
         standalone scripts are now covered by `tests/test_api.py`.
- [ ] Fold one-line function bodies onto the `def` line -- see trade-off below.
- [ ] Add `.git-blame-ignore-revs` for the reformat commit when this is
      committed.

### Measured

| metric | tinygrad | before | after |
|---|---|---|---|
| total lines | -- | 6290 | **6245** (and 469 fewer in `imr_fast.py`) |
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
- [ ] **Stage 4 -- thermal.** bubtherm/medtherm, then the implicit wall solve.
      Narrower than it looks: #57 made the medtherm-only case closed-form, so
      only mass transfer needs `lax.custom_root`.
- [ ] **Stage 5 -- remove numba, and only then.** Delete `_mechanical.py` once
      JAX covers the mechanical path at >= parity; delete `_dual.py` and
      `_complex.py` once JAX covers everything they do. Keeping numba through
      stages 0-4 is deliberate -- it is the performance reference.

### Stage 0 results

Ported `radial=2` + NHKV standalone (no `imr_fast` import, so nothing could
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

### Risks

| risk | why it bites | where it is handled |
|---|---|---|
| ~~**LSODA has no diffrax equivalent**~~ | **Downgraded -- see "Stiffness is static" below.** Stiffness is a property of the configuration, not of the trajectory, so there is nothing to auto-switch | solver chosen at `prepare` time, as scipy already does for BDF |
| **Terminal events** | `_radius_floor_event` is `terminal=True, direction=-1`; diffrax's event model differs | Stage 2; the floor is a guard, so a divergent-solve check may substitute |
| **float32 default** | would pass loose tests and fail tight ones, confusingly | `jax_enable_x64` before any dtype is touched; already required in Stage 0 |
| **Dispatch under `jit`** | the #96 tables are Python-object keyed | closed over at trace time; static per solve |
| **Compile latency** | 1.4 s per distinct configuration | amortised for BOED, but a design sweep that varies the time grid may retrigger |
| **Two backends drift** | exactly the failure mode `_mechanical.py` already has | the cross-backend test must cover every config the JAX backend claims |
