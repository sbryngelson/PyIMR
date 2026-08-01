# Changelog

Notable changes per release. Versions follow [semantic versioning](https://semver.org);
while the major version is `0`, breaking changes move the minor version.

## 0.3.0 — unreleased

Everything merged after `0.2.0` was set (PR #7). Several changes are breaking in
the way that matters most for a solver: **the same call returns different
numbers**, with no error and no deprecation path.

### Breaking: the project is now PyIMR

Every import changes. Nothing about the numbers does.

| was | now |
|---|---|
| `import imr_fast` | `import pyimr` |
| `from imr_fast import simulate` | `from pyimr import simulate` |
| `pip install imr-fast` | `pip install PyIMR` |
| `pip install 'imr-fast[inference]'` | `pip install 'PyIMR[inference]'` |
| `github.com/sbryngelson/imr-fast` | `github.com/sbryngelson/PyIMR` |

The old GitHub URL redirects; the old distribution name does not. There is no
compatibility shim — `import imr_fast` fails outright rather than warning, which
is the honest failure for a package still at `0.x`.

The layout is flat: the package is `pyimr/` at the repo root, not `src/pyimr/`.
That trades away the guarantee `src/` gave — that the suite imports the
*installed* package rather than the working copy — so a module missing from the
wheel can no longer be caught by ordinary tests. `test_the_wheel_ships_every_package_module`
builds a wheel and checks its contents instead, and runs in the nightly set.

`validate_bubtherm_adiabatic.py` and `validate_thermal_fd.py` moved from the
repo root to `tools/`. `pyproject.toml` gained `authors`, `keywords`,
`classifiers` and `[project.urls]`.

### The project is MIT licensed

Previously unlicensed, which meant no permission to use it. `LICENSE` is now in
the repo and ships inside the wheel at `dist-info/licenses/`, declared as a
PEP 639 SPDX expression (`license = "MIT"`) rather than the deprecated
`License ::` classifier. Building therefore needs `setuptools>=77`.

### Breaking: results change

- **`Mt` now defaults to 100 rather than 25** (#69). `Nt` is unchanged at 25.
  Any run with `medtherm=1` that does not set `Mt` explicitly returns different
  numbers, and costs a little under 2x more.

  On the fully coupled model the error is set by the medium grid almost alone.
  Against a converged reference, holding one grid fixed and refining the other
  from the old default:

  ```
  refine Mt alone, 25 -> 100:   4.98e-02 -> 4.46e-05     1100x better
  refine Nt alone, 25 -> 100:   4.98e-02 -> 5.03e-02     no improvement
  ```

  The full sweep, spectral, reference = spectral `Nt = Mt = 150`:

  | Nt | Mt | cost | error in `R/R0` |
  |---:|---:|---:|---:|
  | 25 | 25 | 9 s | 4.98e-02 |
  | 25 | 50 | 13 s | 1.03e-02 |
  | **25** | **100** | **17 s** | **4.46e-05** |
  | 100 | 25 | 69 s | 5.03e-02 |
  | 100 | 100 | 129 s | 1.55e-04 |

  Every column is set by `Mt`: at `Mt = 25` the error is 5.0e-02 whether `Nt` is
  25 or 100. Medium nodes are cheap because they carry no mass-transfer
  coupling, which is why a thousandfold accuracy gain costs under 2x.

  `Mt = 25` matched `Nt` and matched IMRv2; neither is a convergence argument.
  Pass `Mt=25` to reproduce 0.2.0 numbers.

  Caveat, also on #69: the `4.46e-05` sits below the reference's own
  self-consistency of `1.55e-04`, so it means "converged as far as this
  reference can tell". And this is one material, one `R0/Req` and one window --
  `Mt` dominating is a statement about the medium conduction length scale
  relative to the collapse, expected to hold generally but checked in one place.

- **Thermal trajectories move: the Kirchhoff transform and its inverse are now
  consistent with the conductivity the diffusion term uses** (#75).

  The bubble thermal PDE integrates `theta`, defined so that
  `d(theta)/dT = K*(T) = alpha*T + beta`. The shipped inverse was
  `T = (alpha - 1 + sqrt(1 + 2*alpha*theta))/alpha`, which inverts a transform
  whose derivative is `alpha*T + (1 - alpha)` — equal only when
  `alpha + beta = 1`. It is `1.0024` at the defaults, because air and water
  vapour happen to have nearly the same conductivity at 298 K, and it is
  arbitrarily far from 1 for any other gas.

  Six sites each re-derived that algebra inline. They now share one definition.

  The clearest symptom: `alpha` and `beta` are normalised by `K8`, which
  cancels from `chi * K*(T)` and is therefore pure convention — so a **dry-gas**
  run should not care about the *vapour* conductivity at all. It did, by up to
  `1.5e-03` in `R/R0`. It no longer does, to `2e-09`.

  Measured against the pinned IMRv2 references:

  | case | max &#124;dR&#124; | pinned band |
  |---|---:|---|
  | `bubtherm` | 2.35e-04 | still inside |
  | `medtherm` | 2.07e-04 | still inside |
  | `masstrans + medtherm` | 7.01e-05 | still inside |
  | `masstrans` | 2.90e-04 | **leaves it** |

  Only `masstrans` diverges enough to leave its band, because the mixture
  coefficients vary with the vapour fraction and no fixed normalisation absorbs
  them. `tests/ref_masstrans.csv` is retained as a record of upstream behaviour
  and is no longer asserted in the pinned band, as `ref_radial5.csv` was in #18.

  IMRv2 contains the same defect, so this is a divergence from upstream. It is
  not a modelling disagreement: the old code made results depend on an input
  that cannot physically matter.

  **If you have published thermal numbers from 0.2.0, they carry the upstream
  defect and are not reproducible from 0.3.0.**

- **`thermal` now defaults to `"spectral"` (Chebyshev collocation) rather than
  `"fd"`** (#26). Any run with `bubtherm=1` returns different numbers unless it
  passes `thermal="fd"` explicitly.

  This is a fix. Measured on the fully coupled model
  (`bubtherm + medtherm + masstrans + vapor`, `R0 = 225 um, Req = 37.5 um`,
  NHKV) against a converged spectral `N = 100` reference, max deviation in
  `R/R0` over the trajectory:

  | scheme | N | cost | error |
  |---|---|---|---|
  | fd | 25 | 1.9 s | **2.8e-01** (the old default) |
  | fd | 50 | 6.6 s | 1.9e-01 |
  | **spectral** | **25** | 9.0 s | **5.0e-02** |
  | fd | 100 | 24.9 s | 1.1e-01 |
  | spectral | 50 | 30.6 s | 1.0e-02 |
  | fd | 200 | 100.5 s | 5.3e-02 |

  Spectral is Pareto-dominant: at `N = 25` it matches finite difference at
  `N = 200` for a ninth of the cost, and at matched cost it is roughly three
  times more accurate. The old default was not merely less accurate — at
  `Nt = 25` it was badly under-resolved on the coupled model.

  Spectral is *stiffer*, so it costs more at equal `N`: 1.6x to 4.9x, and the
  cause is step count rather than per-step work (RHS evaluations go 9,986 to
  51,428 on the coupled case). The Chebyshev second-derivative operator has
  eigenvalues scaling like `N**4` against `N**2` for finite difference.

  Note that **neither scheme is converged at `Nt = 25`** on the fully coupled
  model. Raising `Nt` is a separate decision from choosing the scheme, and this
  change does not make it for you.

  **If you have published numbers from a thermal model under 0.2.0, pass
  `thermal="fd"` to reproduce them.**

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
  | `import imr_pymc` | `from imr_fast import pymc_op` |
  | `import thermal_fd`, `thermal_spectral` | `from imr_fast import thermal_fd`, `thermal_spectral` |
  | `_imr_config`, `_imr_materials`, ... | `imr_fast._config`, `imr_fast._materials`, ... |

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

  `imr_pymc` became `pymc_op` rather than `imr_fast.pymc`, which would have
  shadowed the real `pymc` inside the package.

  `imr_fast._rhs` and `imr_fast._stress` name both a module and a re-exported
  function, so attribute access gets the function -- the `datetime.datetime`
  pattern. Import the module as `from imr_fast._rhs import ...`, not
  `from imr_fast import _rhs`. A test pins that the modules stay reachable.

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
- `pymc_op.log_marginal_likelihood(trace)`, the supported way to read the
  evidence out of a `sample_smc` run (#25). Reading it by hand does not
  survive adaptive tempering: chains do not all take the same number of
  stages, arviz stores unequal ones raggedly, and the obvious
  `np.asarray(...).ravel()[-1]` then returns a list rather than a float. It
  also kept a single chain and discarded the rest. 5 of 18 calibration runs
  tempered raggedly.
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
- **The SMC log marginal likelihood was never checked against a known
  answer** (#25). Its test asserted `np.isfinite`, which any number satisfies;
  removing the Gaussian normalisation constant entirely still passed it, at
  an 815-nat error. It is now validated against exact Gauss-Legendre
  quadrature of the same posterior -- the prior is Uniform(0, 1) per
  parameter, so the evidence is a plain integral over the unit cube that owes
  nothing to SMC. Measured error against quadrature is +0.02 nats at 256
  draws and +0.26 at 64; the limit is seed-to-seed scatter rather than bias.
  No behaviour changed -- the numbers were right, nothing had established it.
- The last two `np.errstate` suppressions in the package, and the reason one of
  them was never needed (#35). The Mie-Gruneisen sound-speed `sqrt` was
  suppressed on the strength of a comment blaming rejected LSODA trial steps;
  those negative values belonged to the *spurious* density root fixed in #18.
  Both discriminants now read in closed form -- `_mu_of_A`'s collapses to
  `1 + 4*A*(s + nog)`, the sound-speed radicand to
  `(1 + (s + 2*nog)*mu) / (1 - s*mu)**3` -- which shows the two share one
  boundary exactly, so a negative argument at the second is unreachable. The
  wall-temperature secant keeps its suppression, narrowed from the whole
  fallback ladder to the residual evaluation that actually needs it. Trajectory
  changes are at roundoff; the pinned suite is unmoved.
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
