# Changelog

Notable changes per release. Versions follow [semantic versioning](https://semver.org);
while the major version is `0`, breaking changes move the minor version.

## 0.3.0 — unreleased

Everything merged after `0.2.0` was set (PR #7). Several changes are breaking in
the way that matters most for a solver: **the same call returns different
numbers**, with no error and no deprecation path.

### Breaking: results change

- **`radial = 5` (KM enthalpy / Mie-Gruneisen) produces a different
  trajectory** (#18). IMRv2 takes the wrong root of its own Mie-Gruneisen
  density quadratic; the root is corrected here. On the standard
  `R0 = 225 um, Req = 37.5 um` neo-Hookean Kelvin-Voigt case the collapse
  minimum moves from `R/R0 = 0.0536` to `0.0818`, a pointwise deviation of
  `4.8e-01` from the pinned `tests/ref_radial5.csv`.

  This is a fix, not a regression: the corrected value now agrees with the
  Tait branch (`radial = 3`, `R/R0 = 0.0821`) to `3e-4`, which is the physical
  expectation, whereas the old value did not. `tests/ref_radial5.csv` is
  retained as a record of upstream behaviour and is no longer asserted against.

  **If you have published numbers from `radial = 5` under 0.2.0, they were
  computed with the upstream defect.**

- **Distributed-memory defaults changed** (#19). `Giesekus` and `LinearPTT`
  moved from `points = 480` with trapezoid quadrature to `points = 240` with
  a new `quadrature = "gauss"` field. For a `mobility = 0.2` Giesekus solve the
  two defaults differ by `1.6e-02` pointwise — because the old default was that
  far from converged, not because the new one is. Against a 1920-point
  reference the new default is accurate to `2.2e-07`; the old default was the
  `1.6e-02`.

  Pass `points=480, quadrature="trapezoid"` explicitly to reproduce 0.2.0.

### Breaking: API

- **Every module now lives inside the `imr_fast` package** (#61). The repo held
  16 installed top-level modules; it now installs one name. Import paths change:

  | 0.2.0 | 0.3.0 |
  |---|---|
  | `import imr_sensitivity` | `from imr_fast import sensitivity` |
  | `import imr_inference` | `from imr_fast import inference` |
  | `import imr_data` | `from imr_fast import data` |
  | `import imr_design` | `from imr_fast import design` |
  | `import imr_pymc` | `from imr_fast import pymc_bridge` |
  | `import thermal_fd`, `thermal_spectral` | `from imr_fast import thermal_fd`, `thermal_spectral` |
  | `_imr_config`, `_imr_materials`, ... | `imr_fast._config`, `imr_fast._materials`, ... |
  | `_imr_rhs`, `_imr_stress` | `imr_fast._equations`, `imr_fast._constitutive` |

  `from imr_fast import ...` for the solver, materials, and result API is
  unchanged — that is the documented surface, and it was already the only
  top-level name most callers used.

  Three things forced this. `thermal_fd` and `thermal_spectral` were generic
  names occupying the global module namespace of anyone who installed the
  package. The test suite put the repo root on `sys.path`, so it validated the
  working tree and never the installed artefact — the failure mode a `src/`
  layout exists to prevent, and the reason #34 stayed invisible. And the shipped
  module list was maintained by hand in two places; `packages.find` now
  discovers it.

  `imr_pymc` became `pymc_bridge` rather than `imr_fast.pymc`, which would have
  shadowed the real `pymc` inside the package. `_imr_rhs` and `_imr_stress`
  became `_equations` and `_constitutive` because `__init__` re-exports
  functions named `_rhs` and `_stress`: a submodule of the same name is
  reachable before package init and shadowed after it, so `imr_fast._rhs`
  would have meant two different objects depending on when it was read. A
  test now asserts no submodule name is shadowed.

- **`from imr_fast import *` no longer exports** `MediumOperators`,
  `PreparedDistributedStress`, `PreparedForcing`,
  `PreparedInstantaneousMaterial`, `StateLayout`, `params`, or `pvsat` (#8).
  All seven remain importable by name from `imr_fast`; only the star-import
  surface shrank.

- **Internals moved modules** (#10, #14–#16). `imr_fast.py` went from 2683 to
  454 lines across six splits into `_imr_config`, `_imr_prepare`, `_imr_rhs`,
  `_imr_stress`, `_imr_thermal`, and `_imr_materials`. The documented public
  API is unchanged and covered by the star-import leak test, but code reaching
  into `imr_fast` internals by name may break.

### Added

- `radial = 6` (Gilmore / Mie-Gruneisen), previously rejected during
  configuration (#18).
- `thermal` config field selecting the gas/medium thermal discretization:
  `"fd"` (default, second-order finite difference) or `"spectral"` (Chebyshev
  collocation) (#20). Spectral at `Nt = 25` matches or beats finite difference
  at `Nt = 400` on collapse depth and timing.
- `max_step_s` config field bounding the integrator step (#8).
- `PowellEyring` and `ModifiedPowellEyring` generalized-Newtonian laws (#10).
- `imr_data` module: equilibrium radius, natural frequency, and collapse
  feature estimators operating on measured traces (#10).
- `thermal_spectral` module: Chebyshev differentiation and spherical Laplacian
  operators for the thermal PDEs (#20).
- `ElasticModel`, `MaterialModel`, and `ViscousModel` protocols, and the
  ambient constants `C8`, `KAPPA`, `P8`, `RHO`, `SURF`, are now public.

### Fixed

- The reported spectral thermal accuracy "plateau" at `2.7e-04` was a
  measurement artifact, not solver behaviour (#21). `R(t)` was sampled at 200
  points over 60 us while the collapse minimum is far sharper, so the pointwise
  maximum measured sampling phase rather than error. The median deviation over
  the trace was `8.2e-06`, two orders lower. Spectral convergence has no floor.
- Missing stress term in `radial = 5` (#18).
- **Sensitivities with `thermal = "spectral"` were wrong** (#43). The tangent
  path rebuilt the wall-flux weights with a hardcoded three-point uniform
  finite-difference stencil, overwriting the prepared dense Chebyshev boundary
  rows, so every spectral sensitivity differentiated a different boundary
  closure than the forward solve integrated. The coupled medium-temperature
  tangent was off by `2.5e-01` relative, step-independently; it is now
  `5.2e-05`. Forward trajectories were never affected, on either backend.

  This shipped in 0.3.0-development only — `thermal` itself is new in this
  release (#20) — so no released version returned the bad gradients.

## 0.2.0

Prepared solver core, production forward sensitivities, and the inference
layer (`imr_sensitivity`, `imr_inference`).

## 0.1.0

Initial validated solver API.
