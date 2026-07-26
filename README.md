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
| `imr_grad.py` | RP/polytropic forward sensitivities for a six-term linear stress library |
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
- forward sensitivities against finite differences.

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
- `imr_grad.py` differentiates its own RP/polytropic six-term linear material
  library. It does not yet cover the complete `imr_fast.simulate` solver.
- Gilmore/Mie-Gruneisen is intentionally unavailable because the corresponding
  IMRv2 branch produces non-real states in the tested pressure range.
- Shooting-computed collapse initial stress is not implemented. Explicit
  constitutive initial states are accepted; they matter when relaxation is
  comparable to the observation window.

## Reference implementation

`IMRv2/src/f_init_stress.m` uses an undefined `z1` in the
`De == 0 || De == Inf` branch. That branch is unreachable for the memory models
that call the function, so this is a latent upstream defect rather than an
active discrepancy.

## Gradients

`imr_grad.simulate_grad` returns `(R, dR/dc)` using analytic forward
sensitivities for a constitutive law linear in six coefficients:

$$
\frac{ds_k}{dt}=J_y s_k+\frac{\partial f}{\partial c_k}.
$$

This is a separate specialized solver intended for efficient gradient-based
inference. General full-solver sensitivities remain future work.

## Citation

- Estrada, Barajas, Henann, Johnsen & Franck, *High strain-rate soft material
  characterization via inertial cavitation*, JMPS (2018).
  <https://doi.org/10.1016/j.jmps.2017.12.006>
- Warnez & Johnsen, *Numerical modeling of bubble dynamics in viscoelastic media
  with relaxation*, Physics of Fluids 27, 063103 (2015).
  <https://doi.org/10.1063/1.4922598>
- IMRv2: <https://github.com/InertialMicrocavitationRheometry/IMRv2>
