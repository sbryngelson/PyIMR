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
| `tests/run_validation.py` | IMRv2 trajectories, closed forms, reduction limits, and derivative checks |

## Solver scope

`imr_fast.simulate` supports:

| option | setting |
|---|---|
| radial dynamics | Rayleigh-Plesset; Keller-Miksis pressure; KM enthalpy/Tait; Gilmore/Tait; KM enthalpy/Mie-Gruneisen |
| bubble thermodynamics | polytropic closure or a gas thermal PDE |
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
- `HerschelBulkley`
- `Bingham`

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
`result.stress_reference_radius_ratio`. The default 480 points prioritize the
validated Oldroyd-B reduction; lower resolutions are useful for exploratory
sweeps. Prepared coupled heat/mass-transfer problems use sparse BDF, while
non-stiff configurations retain LSODA.

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
- Gilmore/Mie-Gruneisen (`radial = 6`) is unavailable because the corresponding
  IMRv2 branch returns complex radii. Measured on the standard reference case:
  `max|imag(R/R0)| = 4.069`, non-real at 299 of 300 points, returned without
  any upstream error.
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
- **`radial = 6` returns complex trajectories** without raising, and the
  `radial` constraint is stated three mutually inconsistent ways.
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
