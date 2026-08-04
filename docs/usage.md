# Usage

All inputs are dimensional. Returned arrays are read-only.

## Solving

```python
import numpy as np
from pyimr import NeoHookeanKelvinVoigt, SimulationConfig, simulate

t = np.linspace(0.0, 120e-6, 300)
config = SimulationConfig(
    R0=225e-6,
    Req=37.5e-6,
    material=NeoHookeanKelvinVoigt(shear_modulus_pa=2500.0, viscosity_pa_s=0.1),
)
result = simulate(t, config)
```

`result` carries `time_s`, `radius_ratio`, `radius_m`, `wall_velocity_m_s`,
`internal_pressure_pa`, `stress_integral_pa` and `stats`. Thermal
configurations also return gas and liquid temperature fields and, when enabled,
vapor mass fraction. Materials with memory expose their internal stress state.
Inactive fields are `None`.

Unsupported option values and inconsistent thermal/mass-transfer combinations
fail during configuration. Integration failures and material-domain violations
raise `SimulationError`.

## Preparing

For repeated solves with one configuration, preparation hoists constant work --
state layout, grids, finite-difference operators, constitutive quadrature,
Jacobian sparsity:

```python
problem = prepare(config)
first = problem.solve(t)
second = problem.solve(t)  # immutable setup is reused; solve state is fresh
```

## Physical parameters, initial state, forcing

```python
from pyimr import InitialState, PhysicalParameters, SampledForcing

config = SimulationConfig(
    R0=225e-6, Req=37.5e-6,
    material=NeoHookeanKelvinVoigt(2500.0, 0.1),
    physics=PhysicalParameters(polytropic_exponent=1.47),
    initial=InitialState(wall_velocity_m_s=2.0, internal_pressure_pa=1.5e5),
)
```

Sampled forcing values are pressure perturbations relative to the far-field
baseline. A shape-preserving cubic interpolant is used between samples, and the
perturbation is zero outside their time span:

```python
config = SimulationConfig(
    R0=225e-6, Req=37.5e-6,
    material=NeoHookeanKelvinVoigt(2500.0, 0.1),
    sampled_forcing=SampledForcing(
        time_s=tuple(measured_time_s),
        pressure_pa=tuple(measured_pressure_perturbation_pa),
    ),
)
```

## Sensitivities

The tangent-linear solver differentiates the production RHS rather than a
reduced surrogate. It covers radial models 1-5, every typed material, thermal
and mass-transfer states, distributed nonlinear memory, forcing, geometry,
initial conditions, and continuous physical parameters.

```python
sensitivity = problem.solve_with_sensitivities(
    t,
    ("R0", "material.shear_modulus_pa", "material.viscosity_pa_s", "physics.polytropic_exponent"),
)

sensitivity.simulation
sensitivity.radius_m       # shape: (time, parameter)
sensitivity.state          # complete internal-state derivatives
```

Parameter paths follow the frozen configuration objects, and derivatives are
with respect to dimensional values. All requested directions share one augmented
integration; prepared parameter scaling keeps error control dimensionless.

The mechanical path, including distributed memory, uses a cached compiled
directional kernel. After its one-time compilation, six simultaneous NHKV
gradients take about 1.9 times one prepared forward solve.

Tangent accuracy is not uniform across configurations. The mechanical path is
limited by the finite-difference check it is measured against; the coupled
thermal paths are limited by how accurately the augmented state/tangent system
is integrated, which is a real bound on anything built on those gradients rather
than a defect in the tangent equations. Measured error by configuration is in
[accuracy.md](accuracy.md).

## Inference

Prepared inference uses normalized bounded coordinates, dimensional Gaussian
radius likelihoods, analytic sensitivity Jacobians, deterministic
Latin-hypercube starts, and optional process-parallel batch evaluation:

```python
from pyimr.inference import InferenceParameter, RadiusObservation, prepare_inference

inference = prepare_inference(
    config,
    RadiusObservation(measured_time_s, measured_radius_m, standard_deviation_m=2e-6),
    (
        InferenceParameter("material.shear_modulus_pa", 500.0, 5000.0, "log"),
        InferenceParameter("material.viscosity_pa_s", 0.01, 1.0, "log"),
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

## Tolerances and resolution

Observables do not converge at the same rate, so the right tolerance depends on
which one you fit -- internal pressure is roughly two orders behind radius at
the same setting. `rtol=1e-6, atol=1e-8` is ample for likelihood evaluation
against experimental radius data; keep `1e-10, 1e-12` for sensitivities and
`1e-9` or tighter for validation. The per-observable table is in
[accuracy.md](accuracy.md).

Tolerance does not bound how long one solve can take. `max_steps` (default
`1_000_000`) turns a trajectory that will not finish into a `SimulationError` at
a point of your choosing, which is what makes a grid sweep affordable.

`Nt` and tolerance requirements depend on record length, material stiffness and
which observable is fitted, so a setting adequate for one collapse can be badly
wrong over five. `pyimr.resolution` measures on your own problem:

```python
from pyimr.resolution import choose_resolution

setting = choose_resolution(config, times, target=1e-3, field="radius_ratio")
# Resolution(thermal='fd', Nt=5, rtol=1e-06, atol=1e-08,
#            achieved=3.9e-05, seconds=0.0037)

config = setting.apply(config)
```

It builds a reference and checks it is converged, searches both `spectral` and
`fd` for the cheapest grid meeting `target`, then loosens tolerance as far as
that grid allows. Roughly 18 solves -- worth paying before a sampling or
sensitivity campaign, not worth paying for a single run.

`target` is relative to each field's own peak magnitude. That matters because
observables do not converge together: at identical settings, relative error was
3.4e-07 for radius and 2.8e-05 for internal pressure. It raises rather than
guessing if the reference is not converged or the target is out of reach, since
a number built on either is indistinguishable from a real answer.

## Model selection

Constitutive models for soft matter nest -- `NHKV` is `qKV` at zero strain
stiffening and `SLS` at zero relaxation time -- so comparing best fits always
favours the flexible ones. `pyimr.selection` scores them by evidence instead:

```python
from pyimr.selection import STANDARD_MODELS, compare, log_evidence, redundancy_over_grid, solve_grid

evidences = {}
for candidate in STANDARD_MODELS.values():
    points, normalized, radii, stresses = solve_grid(candidate, solve, count=12)
    redundancies = redundancy_over_grid(candidate, STANDARD_MODELS, points, stresses, solve)
    evidences[candidate.name], _ = log_evidence(
        radii, normalized, redundancies, observed, deviations, dimension=candidate.dimension
    )
posterior = compare(evidences)
```

`pyimr.noise` supplies the strain-rate weighting and the marginalized noise
scale; `pyimr.prior` the redundancy and Occam penalties. Use ONE grid `count`
for every model compared -- mixed resolutions let grid luck decide which lands
nearest the truth.

Always report the best chi-squared per sample alongside the posterior. Model
selection only means something where some candidate actually fits; otherwise the
winner is the least-bad member of an inadequate set, and the posteriors look
just as confident. Worked studies are in `examples/`.

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
data.resolution_convergence(config, times_s, [10, 20, 40])
```

`equilibrium_radius` inverts the solver's own pressure/radius relation exactly.
`natural_frequency` linearises Rayleigh-Plesset about `Req` in a Kelvin-Voigt
medium; it reproduces Minnaert exactly in the gas-only limit and matches the
simulated rebound frequency closely. `collapse_features` locates interior
extrema with sub-sample parabolic refinement, replacing the manual index windows
of IMR-vanilla `calc_3tmins_3Rmaxs`. `resolution_convergence` reports a table for
a ladder you supply, where `pyimr.resolution` searches for a setting; both scale
deviations by the field's own peak, and both move `Mt` with `Nt` only when the
medium is actually solved. Pass `(Nt, Mt)` pairs to set both.

IMR-vanilla's `calc_omega_N` is deliberately not ported: it is a scratch script
whose formula treats the gas pressure at `Rmax` as the equilibrium value,
inflating the stiffness by `alpha**(-3*kappa)`; see [upstream.md](upstream.md).
Video processing (`calcRofT/`) is also out of scope -- that is image analysis,
and scikit-image covers it.

[Back to the README](../README.md)
