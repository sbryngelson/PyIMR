# Discretization

Two spatial discretizations matter, and they matter for opposite reasons.
The distributed stress needs only a quadrature rule, because its constitutive
equations carry no spatial derivatives at all. The thermal fields need a real
operator. Both defaults are measured here
rather than asserted.

## Distributed stress quadrature

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

## Thermal discretization

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
and keeps converging -- there is no floor. It is also far cheaper at that
accuracy. Collapse-depth error against a converged reference, with wall time:

| | error | seconds |
|---|---:|---:|
| finite difference, `Nt = 25` | 1.15e-03 | 0.61 |
| finite difference, `Nt = 400` | 1.82e-05 | 26.67 |
| spectral, `Nt = 25` | **1.80e-06** | **0.64** |

Ten times more accurate than `fd(400)` at a fortieth of the cost.

### Which to use

`thermal="spectral"` is the **default** (#26). The case for it is
stronger on the fully coupled model than on `bubtherm` alone. Against a
converged reference, on `bubtherm + medtherm + masstrans + vapor`, max
deviation in `R/R0`:

| scheme | N | cost | error |
|---|---:|---:|---:|
| finite difference | 25 | 1.9 s | **2.8e-01** |
| finite difference | 50 | 6.6 s | 1.9e-01 |
| **spectral** | **25** | 9.0 s | **5.0e-02** |
| finite difference | 100 | 24.9 s | 1.1e-01 |
| spectral | 50 | 30.6 s | 1.0e-02 |
| finite difference | 200 | 100.5 s | 5.3e-02 |

Spectral at `N = 25` matches finite difference at `N = 200` for a ninth of the
cost, and at matched cost is about three times more accurate. The reference is
spectral at `N = 100`, which is only trustworthy because **finite difference
converges toward it monotonically** — two independent discretisations agreeing
is the evidence, not either one alone.

Spectral is *stiffer*, so it costs more at equal `N` — 1.6x to 4.9x. The cause
is step count, not per-step work: RHS evaluations go from 9,986 to 51,428 on
the coupled case, because the Chebyshev second-derivative operator has
eigenvalues scaling like `N**4` against `N**2` for finite difference.

Pass `thermal="fd"` when you want the cheaper scheme. It is also what every
pinned IMRv2 trajectory was
generated against — IMRv2 is a finite-difference code and has no spectral
branch — so the pinned suite sets it explicitly regardless of the default.

**Neither scheme is converged at `Nt = 25` on the fully coupled model.**
Choosing the scheme and choosing the resolution are separate decisions, and the
default makes only the first.

> Earlier revisions of this section recommended `thermal="fd"` for the coupled
> model, on the grounds that spectral "does not converge" there — the
> `Giesekus(mobility=0)` to `Oldroyd-B` reduction limit sat at `1.4e-02` and
> worsened under refinement. That was a misattribution, resolved in #47: the
> gap is physics, not discretisation. `_dissipation` carries the polymer at its
> quasi-steady value while `_distributed_dissipation` carries its actual
> stress, so the two differ by `O(De)` and the gap grows with `Mt` under
> **finite difference too** (`2.0e-03`, `2.7e-03`, `7.2e-03` at `Mt = 9, 17,
> 33`). It was never evidence about the spectral operator.

[Back to the README](../README.md)
