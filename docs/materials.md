# Materials

The material is explicit and typed. There is no integer constitutive selector
and no shared bag of parameters: every law names its own arguments, and every
argument carries its units.

## Composable instantaneous materials

An `InstantaneousMaterial` can contain an elastic law, a viscous law, or both:

```python
from pyimr import CarreauYasuda, Gent, InstantaneousMaterial

material = InstantaneousMaterial(
    elastic=Gent(shear_modulus_pa=2500.0, extensibility=250.0),
    viscous=CarreauYasuda(
        zero_shear_viscosity_pa_s=0.5,
        infinite_shear_viscosity_pa_s=0.02,
        time_constant_s=20e-6,
        transition_exponent=2.0,
        power_index=0.45,
    ),
)
```

Elastic: `NeoHookean`, `MooneyRivlin`, `Yeoh`, `Fung`, `Gent`, `ArrudaBoyce`,
`Ogden`.

Generalized-Newtonian: `Newtonian`, `PowerLaw`, `CarreauYasuda`, `Cross`,
`PowellEyring`, `ModifiedPowellEyring`, `HerschelBulkley`, `Bingham`.

`Carreau` is `CarreauYasuda(transition_exponent=2)`; simplified Cross is
`Cross(transition_exponent=1)`.

### Ogden

`Ogden` takes matched tuples and is the only elastic law here that depends on
the principal stretches rather than on `I1` alone, so exponents may be negative
or fractional:

```python
material = Ogden(shear_moduli_pa=(1800.0, 600.0, -300.0), exponents=(1.3, 4.0, -2.0))
```

A single term with `exponents=(2.0,)` *is* neo-Hookean, and reduces to it
exactly rather than asymptotically. The small-strain shear modulus is
`sum(shear_moduli_pa * exponents) / 2`, which must be positive; individual
moduli may be negative, as above.

### Why there is no `BlatzKo`

The standard Blatz-Ko strain energy is distinguished by its dependence on `I3`,
and this solver assumes an incompressible spherical deformation with stretches
`(l^-2, l, l)`, so `I3 = 1` identically. In that limit Blatz-Ko is
`MooneyRivlin(c10=0.0, c01=mu/2)` -- verified equal to machine precision -- so a
separate class would be an alias, not a new capability. A genuinely
compressible Blatz-Ko needs the incompressibility assumption relaxed throughout
the radial dynamics.

### Powell-Eyring

The pair uses the standard laws, `eta_inf + (eta_0 - eta_inf) * asinh(x)/x` and
its `log1p(x)/x` variant with `x = lambda*|gdot|`. Both reduce exactly to
`Newtonian(eta_0)` as `lambda -> 0`. IMRv2's `f_viscosity.m` instead uses
`sinh(x)/x^nc`, which is shear-*thickening* and diverges exponentially, and a
`log(1+x)/x^nc` variant with no finite zero-shear limit unless `nc == 1`;
neither was copied. See [upstream.md](upstream.md).

### Domain limits

`PowerLaw`, `HerschelBulkley` and `Bingham` require a positive
`regularization_rate_per_s`. The latter two use a smooth yield-stress
regularization so the implicit radial equations retain a finite tangent at zero
strain rate.

Gent lock-up is a material-domain error: if a trajectory reaches
$I_1-3\ge J_m$, the solve raises `SimulationError` rather than continuing with
nonphysical stress.

Prepared Gauss-Legendre rules evaluate the finite-interval stress integrals. The
solver evaluates the stress-rate terms and acceleration coefficient
analytically, including the viscosity tangent -- no finite-difference derivative
is used inside the radial dynamics. The specialized `NeoHookeanKelvinVoigt` path
and the equivalent composable material agree to solver tolerance.

## Closed-form memory

The finite-dimensional hot paths are `Zener`, `QuadraticZener`, `OldroydB` and
`LinearMaxwell`:

```python
material = Zener(
    shear_modulus_pa=2500.0,
    viscosity_pa_s=0.1,
    relaxation_time_s=40e-6,
    retardation_time_s=8e-6,
)
```

## Distributed nonlinear memory

`Giesekus` and `LinearPTT` evolve radial and hoop stress on a prepared,
wall-clustered Lagrangian grid:

```python
material = Giesekus(
    viscosity_pa_s=0.1,
    relaxation_time_s=40e-6,
    retardation_time_s=8e-6,
    mobility=0.2,
)
```

`result.stress_state` contains radial stress followed by hoop stress on
`result.stress_reference_radius_ratio`.

The constitutive equations at each material point are ordinary differential
equations -- there are no spatial derivatives -- so the only spatial
approximation is the quadrature for the stress integral. `quadrature="gauss"`
(the default, 240 points) places the material points at Gauss-Legendre nodes and
converges spectrally: about five orders of magnitude more accurate than the
trapezoid grid it replaced, and cheaper. The table is in
[discretization.md](discretization.md).

At zero mobility or zero extensibility these converge to the analytic Oldroyd-B
solution. Use `OldroydB` directly when that closure applies -- it is
substantially smaller and faster.

## Collapse-state shooting

Memory materials can initialize from a resolved equilibrium-to-maximum-radius
precursor rather than an assumed unstressed state:

```python
from pyimr import CollapseInitialization

config = SimulationConfig(
    R0=225e-6, Req=37.5e-6,
    material=Zener(2500.0, 0.1, 40e-6, 8e-6),
    collapse=CollapseInitialization(),
)
problem = prepare(config)
problem.collapse_stats
```

Preparation brackets the precursor velocity, shoots to `R/R0 == 1`, and retains
the complete memory state at the maximum. `CollapseStats` records the root,
achieved maximum, integration work, and immutable stress state. Sensitivities
differentiate the event time and shooting root implicitly. Oldroyd-B and
distributed Giesekus/PTT states use the same mechanism; the Zener precursor
retains the upstream IMRv2 formulation.

[Back to the README](../README.md)
