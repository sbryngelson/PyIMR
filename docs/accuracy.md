# Accuracy

What error a given setting buys, measured. Every number here is a measurement on
this machine at the stated configuration, so treat it as a scale rather than a
constant, and re-measure on your own problem with `pyimr.resolution` when the
answer matters.

## Tolerance and the observable you fit

Observables do not converge at the same rate, so the right tolerance depends on
which one you fit. Relative error against `rtol=1e-11`, coupled thermal:

| `rtol` | radius | wall velocity | internal pressure | stress integral | temperature |
|---|---|---|---|---|---|
| 1e-9 | 1.5e-10 | 2.1e-09 | 1.0e-08 | 2.2e-09 | 1.7e-09 |
| 1e-7 | 1.3e-08 | 2.5e-07 | 1.6e-06 | 2.8e-07 | 2.1e-07 |
| 1e-6 | 3.4e-07 | 4.4e-06 | 2.8e-05 | 4.8e-06 | 3.7e-06 |

Internal pressure is roughly 80x less accurate than radius at the same setting,
so fitting it needs about two orders tighter for the same relative accuracy.

For likelihood evaluation against experimental radius data, `rtol=1e-6,
atol=1e-8` is ample: 3.4e-07 against measurement noise nearer 2e-02, and model
selection on real data is unchanged from `1e-9` -- same winners, same
chi-squared per sample. Keep `1e-10, 1e-12` for sensitivities, where derivatives
amplify error, and `1e-9` or tighter when validating against reference
trajectories.

## Gradient accuracy

Tangent accuracy is not uniform across configurations, and the difference is
large enough to matter for anything built on the gradients. Measured against
centered differences on the production RHS:

| configuration | relative error | limited by |
|---|---|---|
| mechanical, every operator | ~7e-07 | the finite-difference check |
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
runtime -- the coupled tangent solve is already the slowest operation in the
package. Tightening further stops helping, because the centered-difference
reference becomes round-off limited before the tangent does.

The mechanical tangents show clean `h^2` convergence before reaching their noise
floor; the thermal ones do not, because they are already at it.

Gradient-based optimizers and Laplace/EIG calculations on coupled thermal models
are therefore working with about four to five significant digits, not the seven
the mechanical path gives.

## Cost is not bounded by tolerance

A parameter set whose bubble collapses to a fraction of a percent of its maximum
and then creeps rather than rebounds takes hundreds of thousands of steps at any
tolerance, while the healthy points of the same sweep finish in under 7e3.
`SimulationConfig.max_steps` (default `1_000_000`) turns that into a
`SimulationError` at a point of your choosing, which is what makes a grid sweep
affordable.
