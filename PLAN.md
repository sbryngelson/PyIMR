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
| W2 spectral collocation | **demoted to optional** (see revision note) |
| W3 viscosity parity | **done** -- rescoped to original work |
| W4 radial 6 | **done** -- resolved with a measurement |
| W5 IMR-vanilla estimators | **done** -- new `imr_data.py` |
| W6 solver control surface | **done** -- 86-option audit, `max_step_s` added |
| W7 tinygrad style | **partly done** -- config, indent, docstrings, one split |
| W8 measured collapse gaps | **done** -- both "gaps" were upstream stubs |

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
- [ ] **Tighten collapse Zener from 1.55e-03.** The one genuine collapse
      comparison. Every other pinned trajectory sits at 1e-5; this is 100x
      looser. Establish whether it is the precursor bracket tolerance, the
      event-time root, or a formulation difference. Upstream integrates the
      precursor with `RelTol/AbsTol = 1e-6` and locates the maximum by
      `max(abs(X(:,1)))` over ode23tb output points -- a discrete argmax, not a
      root-solve, which alone could explain 1e-3.

**Done when:** the `KNOWN GAPS` block in `tests/run_validation.py` is empty, or
every remaining entry has a written justification.

**Status: substantially done.** The `KNOWN GAPS` block is now empty; all four
checks in section 1c pass. Only the Zener tolerance item remains, and it is a
precision question, not a correctness one.

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
- [ ] Further splits of `imr_fast.py` (2214) and `imr_sensitivity.py` (1001) to
      get every file under ~800. Remaining seams in `imr_fast.py`: radial
      equations, thermal/mass transfer, prepared-problem machinery, public API.
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

### The formatter trade-off, stated plainly

tinygrad does **not** use a formatter. Its density -- one-line `def` bodies,
semicolon multi-statements, hand-packed lines -- is hand-maintained, and
`ruff format` actively undoes all of it. It also enforces blank lines around
definitions, which is why the blank-line share went *up* (9.2% -> 10.7%) even
as total lines fell.

So the last three unchecked items are not merely unfinished, they are
**mutually exclusive with keeping `ruff format`**. The choice is:

- **keep the formatter** (current state): 2-space indent and 120 columns are
  enforced automatically and cannot drift, but blank-line share and one-line
  bodies stay formatter-controlled; or
- **drop `ruff format`**, keep only `ruff check`, and hand-tune density the way
  tinygrad does -- reaching ~3% blank lines and one-line bodies, at the cost of
  style becoming a review concern rather than a machine-checked one.

I took the first because it is verifiable and reversible. Flipping to the
second is a one-line config change plus a manual pass.

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
