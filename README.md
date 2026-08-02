# PyIMR

Fast, validated solvers for **inertial microcavitation rheometry (IMR)** with a
typed, dimensional material API.

The package is designed for inference campaigns that require many forward
evaluations. It retains closed-form hot paths for common constitutive laws and
also supports composable hyperelastic and generalized-Newtonian materials,
plus distributed Giesekus and linear PTT memory.

```bash
python -m pip install -e ".[test]"
python -m pytest                  # everything, including numerical validation
python -m pytest -m "not slow"    # skip the high-resolution convergence studies
```

`pytest` prints a table of the measured deviations after the run, not only
pass/fail — a check that still passes but has moved an order of magnitude is
worth seeing.

## Contents

| file | purpose |
|---|---|
| `pyimr/__init__.py` | forward solver, material models, and strict result API |
| `pyimr/sensitivity.py` | production-RHS forward sensitivities |
| `pyimr/inference.py` | prepared likelihood, batch, and multistart tools |
| `pyimr/pymc_op.py` | PyMC bridge: NUTS on the exact tangents, plus SMC for model comparison. Needs `pip install 'PyIMR[inference]'` |
| `pyimr/design.py` | Laplace/Fisher expected information gain for ranking experiment designs |
| `pyimr/assimilation.py` | ensemble and variational state estimation on the prepared flow |
| `pyimr/optimize.py` | Bayesian optimization, and expected-information-gain design search on top of it |
| `pyimr/data.py` | trace-side estimators: equilibrium radius, natural frequency, collapse features |
| `pyimr/thermal_fd.py`, `thermal_spectral.py` | finite-difference and Chebyshev operators for the thermal PDEs |
| `tests/test_validation_*.py` | IMRv2 trajectories, closed forms, reduction limits, and derivative checks |
| `CHANGELOG.md` | released versions and every breaking change |
| API reference | `pip install 'PyIMR[docs]'` then `python -m pdoc pyimr pyimr.sensitivity pyimr.inference pyimr.data pyimr.design pyimr.pymc_op` |
| `docs/discretization.md` | stress quadrature and the two thermal backends, with cost/accuracy measurements |
| `docs/validation.md` | what the suite pins, and the per-case deviations from IMRv2 |
| `docs/upstream.md` | defects found in IMRv2, and what PyIMR does instead |
| `benchmarks/run.py` | reproducible timings; `--json` and `--baseline` to compare runs |

## Upgrading from 0.2.0

`0.3.0` changes results for two configurations. Neither raises; both silently
return different numbers from the same call. See `CHANGELOG.md` for the full
list.

- **`radial = 5`** (KM enthalpy / Mie-Gruneisen) now takes the correct root of
  the Mie-Gruneisen density quadratic. Collapse depth moves from
  `R/R0 = 0.0536` to `0.0818`, which is a `4.8e-01` pointwise change. The new
  value agrees with the Tait branch to `3e-4`; the old one did not. **Numbers
  published from `radial = 5` under 0.2.0 carry the upstream defect.**
- **`Giesekus` and `LinearPTT`** default to `points = 240` with
  `quadrature = "gauss"` instead of `points = 480` with trapezoid. Results move
  by ~`1.6e-02` because the old default was that far from converged; the new
  one is accurate to `2.2e-07` against a 1920-point reference. Pass
  `points=480, quadrature="trapezoid"` to reproduce 0.2.0.

## Solver scope

`pyimr.simulate` supports:

| option | setting |
|---|---|
| radial dynamics | Rayleigh-Plesset; Keller-Miksis pressure; KM enthalpy/Tait; Gilmore/Tait; KM enthalpy/Mie-Gruneisen; Gilmore/Mie-Gruneisen |
| bubble thermodynamics | polytropic closure or a gas thermal PDE |
| thermal discretization | second-order finite difference (default) or Chebyshev collocation |
| medium thermodynamics | optional liquid thermal layer |
| mass transfer | optional vapor transport |
| forcing | constant offset, Gaussian, histotripsy, Heaviside step, or sampled pressure history |
| materials | closed-form, composable instantaneous, or distributed-memory models |

Unsupported option values and inconsistent thermal/mass-transfer combinations
fail during configuration. Integration failures and material-domain violations
raise `SimulationError`.

## Public API

All inputs are dimensional. The material is explicit and typed; there is no
integer constitutive selector or shared bag of material parameters.

```python
import numpy as np
from pyimr import (
    NeoHookeanKelvinVoigt,
    SimulationConfig,
    prepare,
    simulate,
)

t = np.linspace(0.0, 120e-6, 300)
config = SimulationConfig(
    R0=225e-6,
    Req=37.5e-6,
    material=NeoHookeanKelvinVoigt(
        shear_modulus_pa=2500.0,
        viscosity_pa_s=0.1,
    ),
)
result = simulate(t, config)

result.time_s
result.radius_ratio
result.radius_m
result.wall_velocity_m_s
result.internal_pressure_pa
result.stress_integral_pa
result.stats
```

The returned arrays are read-only. Thermal configurations also return gas and
liquid temperature fields and, when enabled, vapor mass fraction. Materials
with memory expose their internal stress state. Inactive fields are `None`.

For repeated solves with one configuration, preparation hoists constant work
such as state layout, grids, finite-difference operators, constitutive
quadrature, and Jacobian sparsity:

```python
problem = prepare(config)
first = problem.solve(t)
second = problem.solve(t)  # immutable setup is reused; solve state is fresh
```

The same prepared problem can integrate any set of continuous parameter
directions with the state:

```python
sensitivity = problem.solve_with_sensitivities(
    t,
    (
        "R0",
        "material.shear_modulus_pa",
        "material.viscosity_pa_s",
        "physics.polytropic_exponent",
    ),
)

sensitivity.simulation
sensitivity.radius_m       # shape: (time, parameter)
sensitivity.state          # complete internal-state derivatives
```

Parameter paths follow the frozen configuration objects. Returned derivatives
are with respect to dimensional parameter values. The sensitivity result also
contains wall-velocity, pressure, stress, gas-temperature,
medium-temperature, and vapor-fraction derivatives when those fields are
active.

### Composable instantaneous materials

An `InstantaneousMaterial` can contain an elastic law, a viscous law, or both:

```python
from pyimr import CarreauYasuda, Gent, InstantaneousMaterial

material = InstantaneousMaterial(
    elastic=Gent(
        shear_modulus_pa=2500.0,
        extensibility=250.0,
    ),
    viscous=CarreauYasuda(
        zero_shear_viscosity_pa_s=0.5,
        infinite_shear_viscosity_pa_s=0.02,
        time_constant_s=20e-6,
        transition_exponent=2.0,
        power_index=0.45,
    ),
)
config = SimulationConfig(R0=225e-6, Req=37.5e-6, material=material)
```

Elastic laws:

- `NeoHookean`
- `MooneyRivlin`
- `Yeoh`
- `Fung`
- `Gent`
- `ArrudaBoyce`
- `Ogden`

Generalized-Newtonian laws:

- `Newtonian`
- `PowerLaw`
- `CarreauYasuda`
- `Cross`
- `PowellEyring`
- `ModifiedPowellEyring`
- `HerschelBulkley`
- `Bingham`

`Carreau` is `CarreauYasuda(transition_exponent=2)`; simplified Cross is
`Cross(transition_exponent=1)`.

`Ogden` takes matched tuples and is the only elastic law here that depends on
the principal stretches rather than on `I1` alone, so exponents may be negative
or fractional:

```python
from pyimr import Ogden

material = Ogden(shear_moduli_pa=(1800.0, 600.0, -300.0), exponents=(1.3, 4.0, -2.0))
```

A single term with `exponents=(2.0,)` *is* neo-Hookean, and reduces to it
exactly rather than asymptotically. The small-strain shear modulus is
`sum(shear_moduli_pa * exponents) / 2`, which must be positive; individual
moduli may be negative, as in the example above.

**`BlatzKo` is deliberately absent.** The standard Blatz-Ko strain energy is
distinguished by its dependence on `I3`, and this solver assumes an
incompressible spherical deformation with stretches `(l^-2, l, l)`, so `I3 = 1`
identically. In that limit Blatz-Ko is `MooneyRivlin(c10=0.0, c01=mu/2)` --
verified equal to machine precision -- so a separate class would be an alias,
not a new capability. A genuinely compressible Blatz-Ko needs the
incompressibility assumption relaxed throughout the radial dynamics.

The Powell-Eyring pair uses the standard laws,
`eta_inf + (eta_0 - eta_inf) * asinh(x)/x` and its `log1p(x)/x` variant with
`x = lambda*|gdot|`. Both reduce exactly to `Newtonian(eta_0)` as
`lambda -> 0`. IMRv2's `f_viscosity.m` instead uses `sinh(x)/x^nc`, which is
shear-*thickening* and diverges exponentially, and a `log(1+x)/x^nc` variant
with no finite zero-shear limit unless `nc == 1`; neither was copied.

Prepared Gauss-Legendre rules evaluate the finite-interval stress integrals.
The solver evaluates the stress-rate terms and acceleration coefficient
analytically, including the viscosity tangent. No finite-difference derivative
is used inside the radial dynamics. The specialized
`NeoHookeanKelvinVoigt` path and the equivalent composable material agree to
solver tolerance.

`PowerLaw`, `HerschelBulkley`, and `Bingham` require a positive
`regularization_rate_per_s`. The latter two use a smooth yield-stress
regularization so the implicit radial equations retain a finite tangent at zero
strain rate.

Gent lock-up is a material-domain error. If a trajectory reaches
$I_1-3\ge J_m$, the solve stops and raises `SimulationError` instead of
continuing with nonphysical stress.

### Closed-form memory models

The finite-dimensional hot paths are:

- `Zener`
- `QuadraticZener`
- `OldroydB`
- `LinearMaxwell`

For example:

```python
from pyimr import Zener

material = Zener(
    shear_modulus_pa=2500.0,
    viscosity_pa_s=0.1,
    relaxation_time_s=40e-6,
    retardation_time_s=8e-6,
)
```

### Distributed nonlinear memory

`Giesekus` and `LinearPTT` evolve radial and hoop stress on a prepared,
wall-clustered Lagrangian grid:

```python
from pyimr import Giesekus

material = Giesekus(
    viscosity_pa_s=0.1,
    relaxation_time_s=40e-6,
    retardation_time_s=8e-6,
    mobility=0.2,
)
config = SimulationConfig(R0=225e-6, Req=37.5e-6, material=material)
```

`result.stress_state` contains radial stress followed by hoop stress on
`result.stress_reference_radius_ratio`. Prepared coupled heat/mass-transfer
problems use sparse BDF, while non-stiff configurations retain LSODA.

The constitutive equations at each material point are ordinary differential
equations -- there are no spatial derivatives -- so the only spatial
approximation is the quadrature for the stress integral. `quadrature="gauss"`
(the default, 240 points) places the material points at Gauss-Legendre nodes
and converges spectrally; it is about five orders of magnitude more accurate
than the 480-point `quadrature="trapezoid"` grid it replaced, and 1.4x faster.
See [docs/discretization.md](docs/discretization.md).

At zero mobility or zero extensibility, the distributed models converge to the
analytic Oldroyd-B solution. Use `OldroydB` directly when that closure applies:
it is substantially smaller and faster.

### Physical parameters, initial state, and forcing

```python
from pyimr import InitialState, PhysicalParameters

config = SimulationConfig(
    R0=225e-6,
    Req=37.5e-6,
    material=NeoHookeanKelvinVoigt(2500.0, 0.1),
    physics=PhysicalParameters(polytropic_exponent=1.47),
    initial=InitialState(
        wall_velocity_m_s=2.0,
        internal_pressure_pa=1.5e5,
    ),
)
```

Sampled forcing values are pressure perturbations relative to the far-field
baseline. A shape-preserving cubic interpolant is used between samples and the
perturbation is zero outside their time span.

```python
from pyimr import SampledForcing

config = SimulationConfig(
    R0=225e-6,
    Req=37.5e-6,
    material=NeoHookeanKelvinVoigt(2500.0, 0.1),
    sampled_forcing=SampledForcing(
        time_s=tuple(measured_time_s),
        pressure_pa=tuple(measured_pressure_perturbation_pa),
    ),
)
```

### Collapse-state shooting

Memory materials can initialize from a resolved equilibrium-to-maximum-radius
precursor rather than an assumed unstressed state:

```python
from pyimr import CollapseInitialization

config = SimulationConfig(
    R0=225e-6,
    Req=37.5e-6,
    material=Zener(2500.0, 0.1, 40e-6, 8e-6),
    collapse=CollapseInitialization(),
)
problem = prepare(config)
problem.collapse_stats
```

Preparation brackets the precursor velocity, shoots to `R/R0 == 1`, and
retains the complete memory state at the maximum. `CollapseStats` records the
root, achieved maximum, integration work, and immutable stress state.
Sensitivities differentiate the event time and shooting root implicitly.
Oldroyd-B and distributed Giesekus/PTT states use the same mechanism; the
Zener precursor retains the upstream IMRv2 formulation.

## Sensitivities and inference

The tangent-linear solver differentiates the production RHS rather than a
reduced surrogate. It covers radial models 1--5, every typed material,
thermal and mass-transfer states, distributed nonlinear memory, forcing,
geometry, initial conditions, and continuous physical parameters.

The mechanical path, including distributed memory, uses a cached compiled
directional kernel. After its one-time compilation, six simultaneous NHKV
gradients take about 1.9 times one prepared forward solve on the development
machine. Thermal and sampled-forcing branches use the same forward-mode
reference implementation and augmented sparse BDF structure where appropriate.

### Gradient accuracy

Tangent accuracy is not uniform across configurations, and the difference is
large enough to matter for anything built on the gradients. Measured against
centered differences on the production RHS:

| configuration | relative error | limited by |
|---|---|---|
| mechanical, `radial = 1`–`5` | ~7e-07 | the finite-difference check |
| coupled heat/mass transfer, `thermal = "spectral"` | ~5e-05 | time integration |
| coupled heat/mass transfer, `thermal = "fd"` | ~8e-05 | time integration |

The thermal tangents are **correct, not defective**: their error is flat in the
finite-difference step across a 16x range, which rules out truncation in the
check, and it responds to the integrator tolerance, which an error in the
tangent equations would not. What bounds them is the accuracy to which the
augmented state/tangent system is integrated. On the coupled fd case at
`h = 0.05`:

| `rtol` / `atol` | relative error |
|---|---|
| `1e-9` / `1e-11` (default) | 8.43e-05 |
| `1e-12` / `1e-14` | 1.53e-06 |

So three orders of tolerance buys a factor of about 55, at a large cost in
runtime — the coupled tangent solve is already the slowest operation in the
package. Tightening further stops helping, because the centered-difference
reference becomes round-off limited before the tangent does.

The mechanical tangents show clean `h^2` convergence before reaching their
noise floor; the thermal ones do not, because they are already at it.

Gradient-based optimizers and Laplace/EIG calculations on coupled thermal
models are therefore working with about four to five significant digits, not
the seven the mechanical path gives.

Prepared inference uses normalized bounded coordinates, dimensional Gaussian
radius likelihoods, analytic sensitivity Jacobians, deterministic
Latin-hypercube starts, and optional process-parallel batch evaluation:

```python
from pyimr.inference import (
    InferenceParameter,
    RadiusObservation,
    prepare_inference,
)

inference = prepare_inference(
    config,
    RadiusObservation(
        measured_time_s,
        measured_radius_m,
        standard_deviation_m=2e-6,
    ),
    (
        InferenceParameter(
            "material.shear_modulus_pa", 500.0, 5000.0, "log"
        ),
        InferenceParameter(
            "material.viscosity_pa_s", 0.01, 1.0, "log"
        ),
    ),
)

batch = inference.evaluate_batch(unit_parameter_matrix, workers=4)
fit = inference.fit_multistart(64, seed=7, workers=4)
fit.endpoints  # every successful and unsuccessful endpoint is retained
fit.best
```

Unit parameter vectors always lie in `[0, 1]`; the configured linear or
logarithmic transform maps them to physical bounds. Multistart results never
discard alternative basins.

### Thermal discretization

`thermal="spectral"` -- the default since 0.3.0 -- puts both the gas and liquid
grids on Chebyshev collocation. `thermal="fd"` is the cheaper second-order
finite difference, and is what every pinned IMRv2 trajectory was generated
against, so the pinned suite sets it explicitly regardless of the default.

On the fully coupled model, spectral at `Nt = 25` matches finite difference at
`Nt = 200` for a ninth of the cost. **Neither is converged there**, and
choosing the scheme is a separate decision from choosing the resolution -- the
default makes only the first. Both cost/accuracy tables are in
**[docs/discretization.md](docs/discretization.md)**.

## Trace estimators

`pyimr.data` covers the step before inference: getting from a measured `R(t)`
history to the quantities a fit needs.

```python
from pyimr import data

Req = data.equilibrium_radius(R0_m, initial_gas_pressure_pa)
omega_n, beta = data.natural_frequency(R0_m, Req, 2500.0, 0.1)
collapse_times_s, peak_radii_m, peak_times_s = data.collapse_features(
    measured_time_s, measured_radius_m
)
data.resolution_convergence(config, times_s, [(10, 10), (20, 20), (40, 40)])
```

`equilibrium_radius` inverts the solver's own pressure/radius relation exactly.
`natural_frequency` linearises Rayleigh-Plesset about `Req` in a Kelvin-Voigt
medium; it reproduces Minnaert exactly in the gas-only limit and matches the
simulated rebound frequency to 2.7%. `collapse_features` locates interior
extrema with sub-sample parabolic refinement, replacing the manual index
windows of IMR-vanilla `calc_3tmins_3Rmaxs`.

IMR-vanilla's `calc_omega_N` is deliberately not ported: it is a scratch script
whose formula treats the gas pressure at `Rmax` as the equilibrium value,
inflating the stiffness by `alpha**(-3*kappa)` and overpredicting by 42x on the
reference case. Video processing (`calcRofT/`) is also out of scope -- that is
image analysis, and scikit-image covers it.

## Validation

The suite pins IMRv2 trajectories across radial equations, forcing, vapor, heat
transfer, mass transfer and the specialized constitutive models, and separately
checks closed forms, reduction limits, and every analytic tangent against
independent centered differences.

Two statistics are reported per pinned case, because they measure different
things (#23): the pointwise maximum sits at a collapse in every case and is
dominated by sub-nanosecond integrator phase, while the median carries no such
sensitivity and is what the suite bounds tightly. The per-case deviation tables
and the argument behind that split are in
**[docs/validation.md](docs/validation.md)**.

## Boundaries

- `PhysicalParameters` defaults reproduce the pinned reference trajectories.
  The default polytropic exponent is 1.4; IMRv2 itself ships with 1.47.
- `SimulationResult.stress_state` contains nondimensional internal variables.
  Public dimensional outputs carry units in their names or documentation.
- `radial = 6` (Gilmore/Mie-Gruneisen) **is supported here**, and is the one
  configuration IMRv2 cannot run at all -- upstream returns complex radii
  (`max|imag(R/R0)| = 4.069`) without raising. The cause is a wrong root of the
  Mie-Gruneisen density quadratic; see [docs/upstream.md](docs/upstream.md).
- `radial = 5` and `radial = 6` **deliberately diverge from IMRv2**. Upstream's
  Mie-Gruneisen branch is physically wrong; the corrections are validated
  against the independent Tait branches and the weakly-compressible limit
  rather than against upstream. `tests/ref_radial5.csv` is retained as a record
  of upstream behaviour, not as a target.
- Collapse shooting requires a material with memory and cannot be combined
  with an explicit initial stress state or nonzero observed wall velocity.
  IMRv2 does permit collapse for memoryless materials; see `PLAN.md` W8.

## Reference implementation

PyIMR diverges from IMRv2 in several places, always deliberately. Eight
defects were found at `dea31cd`, each reproduced with MATLAB R2025a via
`tools/gen_imrv2_cases.m`, and each correction validated against something
other than upstream -- a closed form, an independent equation of state, or a
reduction limit. The wrong Mie-Gruneisen root, the non-functional
non-Newtonian viscosity suite, the stubbed collapse initialization and the
rest are in **[docs/upstream.md](docs/upstream.md)**.

Those defects are why several PyIMR models are validated by reduction limit
rather than against a pinned upstream trajectory: for those models, no working
upstream implementation exists to pin against.

## Tangent equations

Forward sensitivities integrate:

$$
\frac{ds_k}{dt}=J_y s_k+\frac{\partial f}{\partial c_k}.
$$

All requested parameter directions share one augmented integration. Prepared
parameter scaling keeps error control dimensionless; public derivatives are
converted back to dimensional parameter units.

## Citation

- Estrada, Barajas, Henann, Johnsen & Franck, *High strain-rate soft material
  characterization via inertial cavitation*, JMPS (2018).
  <https://doi.org/10.1016/j.jmps.2017.12.006>
- Warnez & Johnsen, *Numerical modeling of bubble dynamics in viscoelastic media
  with relaxation*, Physics of Fluids 27, 063103 (2015).
  <https://doi.org/10.1063/1.4922598>
- IMRv2: <https://github.com/InertialMicrocavitationRheometry/IMRv2>

## License

MIT — see [LICENSE](LICENSE).
