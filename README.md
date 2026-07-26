# imr-fast

Fast, validated solvers for **inertial microcavitation rheometry (IMR)** with a
typed, dimensional material API.

The package is designed for inference campaigns that require many forward
evaluations. It retains closed-form hot paths for common constitutive laws and
also supports composable hyperelastic and generalized-Newtonian materials,
plus distributed Giesekus and linear PTT memory.

```bash
python -m pip install -e ".[test]"
python -m pytest
python tests/run_validation.py
```

## Contents

| file | purpose |
|---|---|
| `imr_fast.py` | forward solver, material models, and strict result API |
| `imr_sensitivity.py` | production-RHS forward sensitivities |
| `imr_inference.py` | prepared likelihood, batch, and multistart tools |
| `imr_data.py` | trace-side estimators: equilibrium radius, natural frequency, collapse features |
| `thermal_fd.py`, `thermal_spectral.py` | finite-difference and Chebyshev operators for the thermal PDEs |
| `tests/run_validation.py` | IMRv2 trajectories, closed forms, reduction limits, and derivative checks |
| `CHANGELOG.md` | released versions and every breaking change |

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

`imr_fast.simulate` supports:

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
from imr_fast import (
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
from imr_fast import CarreauYasuda, Gent, InstantaneousMaterial

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

For example:

```python
from imr_fast import Zener

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
from imr_fast import Giesekus

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
approximation is the quadrature for the stress integral

$$
I=\int_R^\infty \frac{2(\tau_{rr}-\tau_{\theta\theta})}{r}\,dr .
$$

Mapping to the Lagrangian coordinate makes the geometric factor constant in
time, so the integral is a fixed-weight sum and its time derivative is exact
rather than a discrete difference. `quadrature="gauss"` (the default) places
the material points at Gauss-Legendre nodes and converges spectrally;
`quadrature="trapezoid"` reproduces the older uniform grid.

| points | trapezoid | Gauss-Legendre |
|---:|---:|---:|
| 60 | 3.4e-01 | 3.9e-04 |
| 120 | 2.2e-01 | 1.5e-07 |
| 240 | 7.4e-02 | 6.0e-08 |
| 480 | 2.6e-02 | 5.1e-08 |

Maximum absolute `R/R0` deviation from a converged reference, nonlinear
Giesekus at mobility 0.2. The default is 240 Gauss points, which is both 1.4x
faster and about five orders of magnitude more accurate than the 480-point
trapezoid grid used previously. That older default carried percent-level
error which the Oldroyd-B reduction limit alone did not reveal, because the
linear limit is far easier to integrate than the nonlinear problem.

At zero mobility or zero extensibility, the distributed models converge to the
analytic Oldroyd-B solution. Use `OldroydB` directly when that closure applies:
it is substantially smaller and faster.

### Physical parameters, initial state, and forcing

```python
from imr_fast import InitialState, PhysicalParameters

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
from imr_fast import SampledForcing

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
from imr_fast import CollapseInitialization

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

Prepared inference uses normalized bounded coordinates, dimensional Gaussian
radius likelihoods, analytic sensitivity Jacobians, deterministic
Latin-hypercube starts, and optional process-parallel batch evaluation:

```python
from imr_inference import (
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

Unlike the distributed stress -- whose Lagrangian form has no spatial
derivatives at all -- the thermal fields genuinely need a spatial
discretization, so collocation buys something real here.
`thermal="spectral"` switches both the gas and liquid grids to Chebyshev
collocation:

```python
config = SimulationConfig(
    R0=225e-6, Req=37.5e-6,
    material=NeoHookeanKelvinVoigt(2500.0, 0.1),
    bubtherm=1, Nt=25, thermal="spectral",
)
```

The spherical Laplacian is built on the even extension of the gas grid, so
regularity at the bubble centre is exact and the removable `2/y` singularity
never has to be special-cased. Boundary flux weights come from the relevant
row of the first-derivative matrix rather than a hardcoded three-point
stencil, which is what makes the wall closure work on a non-uniform grid.

On the spherical Laplacian alone at `N = 17`, the Chebyshev operator errs by
`1.9e-11` against `2.6e-02` for the finite-difference stencil.

On the trajectory, measure collapse depth and timing on an output grid fine
enough to resolve the minimum. A coarse grid samples the sharp collapse at
slightly different phases as the solution shifts, which produces an apparent
error floor near `1e-04` that has nothing to do with the discretization:

| | minimum radius | collapse time |
|---|---:|---:|
| spectral, `Nt = 25` | 4.4e-06 | 0.0005 ns |
| finite difference, `Nt = 400` | 4.7e-06 | 0.0119 ns |

**Spectral at 25 points matches or beats finite difference at 400** on both,
and keeps converging -- there is no floor.

`thermal="fd"` remains the default because it is what every pinned IMRv2
trajectory was generated against; switching would invalidate the pinned
suite.

## Trace estimators

`imr_data` covers the step before inference: getting from a measured `R(t)`
history to the quantities a fit needs.

```python
import imr_data

Req = imr_data.equilibrium_radius(R0_m, initial_gas_pressure_pa)
omega_n, beta = imr_data.natural_frequency(R0_m, Req, 2500.0, 0.1)
collapse_times_s, peak_radii_m, peak_times_s = imr_data.collapse_features(
    measured_time_s, measured_radius_m
)
imr_data.resolution_convergence(config, times_s, [(10, 10), (20, 20), (40, 40)])
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

The regression suite covers pinned IMRv2 trajectories across radial equations,
forcing, vapor, heat transfer, mass transfer, and the specialized constitutive
models. It also checks:

- composable neo-Hookean/Newtonian dynamics against the closed-form
  Kelvin-Voigt path;
- elastic and viscous reduction limits;
- analytic stress-rate and acceleration tangents against centered finite
  differences;
- Giesekus and linear PTT convergence to Oldroyd-B;
- sparse coupled thermal-memory integration;
- mechanical, thermal, distributed, and collapse-shooting sensitivities against
  independent centered differences;
- likelihood Jacobians and retained multistart endpoints.

Representative maximum absolute radius-ratio deviations from pinned IMRv2
trajectories are:

| case | maximum deviation |
|---|---:|
| Zener, Deborah number 2, stretch 6 | 2.6e-05 |
| neo-Hookean Kelvin-Voigt parameter grid | 6.3e-05 |
| quadratic Kelvin-Voigt | 1.7e-05 |
| Oldroyd-B | 6.2e-05 |
| thermal and mass-transfer branches | 1.6e-05 |
| compressible radial-equation families | 1.6e-05 |

These are numerical comparisons with the reference implementation, not
estimates of physical-model error.

## Boundaries

- `PhysicalParameters` defaults reproduce the pinned reference trajectories.
  The default polytropic exponent is 1.4; IMRv2 itself ships with 1.47.
- `SimulationResult.stress_state` contains nondimensional internal variables.
  Public dimensional outputs carry units in their names or documentation.
- `radial = 6` (Gilmore/Mie-Gruneisen) **is supported here**, and is the one
  configuration IMRv2 cannot run at all -- upstream returns complex radii
  (`max|imag(R/R0)| = 4.069`) without raising. The cause is a wrong root of the
  Mie-Gruneisen density quadratic; see the reference-implementation notes.
- `radial = 5` and `radial = 6` **deliberately diverge from IMRv2**. Upstream's
  Mie-Gruneisen branch is physically wrong; the corrections are validated
  against the independent Tait branches and the weakly-compressible limit
  rather than against upstream. `tests/ref_radial5.csv` is retained as a record
  of upstream behaviour, not as a target.
- Collapse shooting requires a material with memory and cannot be combined
  with an explicit initial stress state or nonzero observed wall velocity.
  IMRv2 does permit collapse for memoryless materials; see `PLAN.md` W8.

## Reference implementation

Defects found in IMRv2 at `dea31cd`, all reproduced with MATLAB R2025a via
`tools/gen_imrv2_cases.m` and `tools/probe_viscosity.m`. Full list in
`PLAN.md`.

- **Giesekus and linear PTT cannot be run.** `f_call_params.m` dispatches
  `stress` 6 and 7 and forces spectral collocation for both, but its own input
  gate rejects `stress > 5`. imr-fast implements both.
- **The non-Newtonian viscosity suite is non-functional.** `nu_model` 3--7
  leave `intf`/`dintf`/`ddintf` unassigned and raise; `nu_model = 2`
  (Carreau-Yasuda) calls a four-argument helper with three arguments and
  raises. Only `nu_model = 1` (Carreau) runs, and it fails its own Newtonian
  reduction by `6.7e-01` -- its stress integral is quadratic in the strain rate
  where the Newtonian term it must reduce to is linear.
- **Collapse initialization is a stub for most materials.** `f_call_params.m`
  applies a precursor only for the Zener family; it leaves the initial stress
  empty for memoryless materials and returns zeros under an explicit
  `% TODO initial max stress for UCM and Oldroyd-B`. The flag is accepted and
  silently ignored. imr-fast implements the precursor for Oldroyd-B and the
  distributed models, and refuses the flag outright for memoryless materials.
- **The collapse precursor locates the maximum by discrete argmax.**
  `f_init_stress.m` takes `max(abs(X(:,1)))` over ode23tb output points rather
  than root-solving the wall velocity. `R` is locally quadratic at the maximum
  and the stress locally linear, so this costs O(sqrt(tol)) in peak position
  and carries that straight into the initial stress. Upstream's `Szero` is
  `-0.1600469117` against `-0.1599451098` here -- a 1.02e-04 offset equivalent
  to sampling 1.9e-03 before the peak, where the radius is only 2.06e-06 lower.
  That single number accounts for the whole 1.55e-03 deviation on the pinned
  collapse-Zener trajectory; injecting upstream's own `Szero` reproduces it at
  2.08e-05. imr-fast root-finds `v = 0` instead, which is O(tol).
- **The Mie-Gruneisen branch takes the wrong root of its own density
  quadratic.** `a*mu^2 + b*mu + A = 0` has roots tending to `0` and `-1/nog`
  as `A -> 0`; `f_radial_eq.m` takes `(-b + sqrt(d))/(2a)`, which is the
  `-1/nog` branch -- a 32.5% density deficit at ambient pressure, a 48-60%
  enthalpy error, and a negative `c^2`. That negative `c^2` is why `radial = 6`
  returns complex radii: it is the only branch that evaluates the sound speed
  from the EoS. The branch also omits the stress term from `Pb`, which
  `radial = 3` and `4` both include.

  With the correct root, density and sound speed recover their ambient values
  (`rho/rho0 - 1 = 4.3e-05`, `c/c0 - 1 = 2.8e-04`), the analytic enthalpy
  matches both `h ~ P - 1` and a direct numerical integral of `1/rho`, and
  `radial = 5` agrees with the independent Tait form `radial = 3` to
  **5.2e-04** -- against **4.8e-01** as shipped. Upstream's `radial = 5`
  collapses to `R/R0 = 0.0536` where its own Tait branch gives `0.0821`.
- **The `radial` constraint is stated three mutually inconsistent ways.**
- **`f_init_stress.m` uses an undefined `z1`** in the `De == 0 || De == Inf`
  branch. Unreachable for the memory models that call it, so latent rather
  than active.

These are the reason several imr-fast models are validated by reduction limit
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
