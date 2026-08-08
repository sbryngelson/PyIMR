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
reduced surrogate. It covers every operator but `gilmore/mie-gruneisen`, every typed material, thermal
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

### The grid ranks models. It does not estimate parameters.

The grid is a screening tool, and quoting its best-fit point as a converged
estimate is a mistake. On the 15C record, refining a qSLS grid from 12 to 20 per
axis moved the log evidence by 17.3 and moved the best-fit point from
`g = 658, alpha = 1.52` to `g = 144, alpha = 8.86`. The "best" cell is simply
whichever node lands nearest the ridge.

Refining further does not rescue it. The posterior standard deviations there are
`0.0018, 0.0025, 0.0055, 0.078` in unit coordinates against a grid spacing of
`0.053` at 20 per axis, so three of the four directions are narrower than the
spacing -- the posterior occupies about `1e-7` of the prior volume, and resolving
it on a uniform grid would need of order `1e10` points.

The ranking survives this: 17.3 of discretisation error sits well inside a
qSLS-to-SLS gap of 101.6 log units. The parameter estimates and the evidence
*value* do not.

For parameter estimates, use `pyimr.pymc_op.sample_smc`, which concentrates
where the mass is and returns an evidence with an error bar. NUTS
(`sample_posterior`) also works and is not pathological here -- zero divergences,
tree depth well under its cap -- but costs about 87 ms per gradient and 31
leapfrog steps per iteration, so roughly 1.5 hours per chain against SMC's
half hour (#216).

Always report the best chi-squared per sample alongside the posterior. Model
selection only means something where some candidate actually fits; otherwise the
winner is the least-bad member of an inadequate set, and the posteriors look
just as confident. Worked studies are in `examples/`.

### Models the grid cannot reach

`solve_grid` is a Cartesian product at one count on every axis, so it costs
`count**dimension` and runs out at four or five parameters. At `count = 12`, six
axes is 5,971,968 solves against 168,072 for the whole of `STANDARD_MODELS`.

Candidates past that limit live in `EXTENDED_MODELS` and are scored by expansion
about a fit instead of by quadrature over a grid:

```python
from pyimr.selection import EXTENDED_MODELS, candidate_log_evidence, fit_candidate

candidate = EXTENDED_MODELS["qSLS2"]
fit = fit_candidate(candidate, solve, observed, deviation)
evidence = candidate_log_evidence(candidate, solve, observed, deviation, fit.unit)
```

`fit_candidate` is multistart least squares in the same unit coordinates, and it is what
locates the point the expansion needs: `PreparedInference.fit_multistart` works from material
attribute paths, and a candidate's axes need not be material fields at all --- `tau_ratio` is
a ratio of two of them. It returns the distinct modes it found, not only the winner, because
the expansion is about one mode and the evidence is over all of them; sum with `logsumexp`
over `fit.modes`. Check `fit.failure_fraction` --- near one, the search spent its budget
somewhere the material will not integrate and the result is not a fit --- and note that the
differenced Jacobian costs `p + 1` solves an iteration, so an under-converged search reports
a `chi_squared` worse than the truth's rather than announcing itself.

It costs `1 + 2*dimension` solves. The Jacobian is differenced in the *unit*
coordinates the prior is uniform on, which is both where the Occam factor has to
be measured and what lets it handle candidates whose axes are not material fields
-- `qSLS2` has `tau_ratio`, a ratio of two of them, and `oldroydb` likewise.

You must supply the fitted point. Nothing in the package yet locates one for a
candidate, which is the current gap between having these models and being able to
rank them -- see [open work](open-work.md).

### The forward operator is a model choice too

`DYNAMICS_MODELS` names the fourteen operators, as `(dynamics, liquid_eos)` pairs. Six
dynamics --- Rayleigh--Plesset, the Keller--Miksis pressure form, and the enthalpy forms
`keller-enthalpy`, `herring`, `gilmore` and `lezzi-prosperetti-2` --- of which the last four
take one of three equations of state: `tait`, `mie-gruneisen` or `nasg`. `keller-enthalpy` and
`herring` are the `lambda = 0` and `lambda = 1` members of the Prosperetti--Lezzi (1986)
first-order family, not separate theories; `gilmore` is Kirkwood--Bethe, with the local wall
sound speed; `lezzi-prosperetti-2` is their 1987 second-order equation (8.7) at the authors'
recommended `(lambda, theta) = (0.5, 0)`, and is the only one implicit in the acceleration.
It requires a steady far field, because the second-order far-field terms are dropped. They are not
`CandidateModel`s --- a candidate is a material, and this is the operator it is pushed
through --- so they compare by holding the candidate fixed and varying the `solve` callback:

```python
from pyimr.selection import DYNAMICS_MODELS

for dynamics, liquid_eos in DYNAMICS_MODELS:
    ...  # build a solve callback at this operator, then fit and score as above
```

The parameter space is identical across the set, so the Occam terms cancel and the difference
in log evidence is a Bayes factor between operators. Every candidate in this package assumes
`dynamics="keller-miksis"`; on the records analysed in `docs/writeup`, two other operators beat it. See
[open work](open-work.md) for what that comparison does and does not establish --- in
particular, it must be run in identified coordinates, or the ranking follows the prior box
rather than the data.

### The Occam factor can pay for parameters the data cannot see

`laplace_log_evidence` takes `cap_at_prior`. Leave it off and the plain expansion
rewards a model for a parameter its design does not probe: an unidentified
direction sends an eigenvalue of `J^T J` to zero, so `-log det / 2` goes to
`+infinity`. Measured on synthetic one-mode data, the two-mode model beat the
one-mode by 29.7 nats exactly where its second arm did nothing, and lost by 2.6
only where the arm did real work -- backwards, and not by a little.

`cap_at_prior=True` bounds each eigendirection's contribution at one, which is
what a uniform prior on a bounded cube implies: a posterior cannot be wider than
the prior it came from. It is a strict generalisation, agreeing with the plain
form to round-off wherever every direction is already sharper than the prior, so
it changes an answer only where the plain form was not entitled to one.
`candidate_log_evidence` turns it on, because it compares across dimensions.

## Designing when the criteria disagree

`optimize_design` maximises one number. That is the right question only while
the design criteria happen to agree, and on the qSLS study they do not.
Measured over the `(R_max, stretch)` plane:

| criterion | best design |
| --- | --- |
| identify `g` against `alpha` (E-optimality) | 277 µm, stretch 5.0 |
| separate qSLS from SLS — is stiffening real? | 1200 µm, stretch 7.09 |
| separate qSLS from qKV — is relaxation real? | 100 µm, stretch 20 |

The last is the opposite corner from the first, for a physical reason: detecting
a relaxation mode needs a collapse fast enough to excite it, while separating
`g` from `alpha` needs a gentle one, because that term of `Ze` dies as
`lambda^-4`. No design is best at both, so the answer is a *set* of designs.

`explore_tradeoff` traces it. Pass an objective returning several numbers, all
maximised; return a non-finite entry for a design that cannot be run.

```python
from pyimr.pareto import explore_tradeoff

result = explore_tradeoff(criteria, [[50e-6, 1200e-6], [3.0, 20.0]],
                          evaluations=34, initial=10)

result.front_points        # the designs nothing else beats on every criterion
result.front_values        # their scores
result.best_for(0)         # the design that wins criterion 0 outright
```

Two things worth knowing about how it works. The search is ParEGO: one shared
archive with a fresh weight vector each iteration, so every expensive evaluation
informs every weight — which is what makes it affordable when scoring one design
costs hundreds of solves. And the scalarisation is Chebyshev rather than a
weighted sum, because a weighted sum only ever returns points on the convex hull
of the front. Where two mechanisms genuinely compete the front bulges inward, and
there a weighted sum returns the same two endpoints at *every* weight while the
whole interior stays unreachable.

When you score models against each other, let the rival re-fit. T-optimality is
a minimum over the rival's parameters, not a difference at its own fitted values;
holding them fixed overstates discrimination badly. On this study it inflated
qSLS-versus-SLS separation by about an order of magnitude — the honest figure at
the present design is 0.668 noise units, meaning SLS imitates qSLS to well within
the noise.

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
