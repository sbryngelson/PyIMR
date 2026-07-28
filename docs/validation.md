# Validation

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

Two statistics are reported per pinned trajectory, because they measure
different things (issue #23).

The **pointwise maximum** sits at a collapse in every pinned case. There
`|dR/dt| ~ 3.3e5 /s`, so it is dominated by a sub-nanosecond timing difference
rather than by radius accuracy: shifting our own solution by **25 ps** removes
about 77% of it, taking Keller-Miksis neo-Hookean from `8.6e-06` to `2.0e-06`.
The residual matches the deviation away from collapses, `2-10e-06`, so the true
pointwise agreement is several times better than the maxima below suggest. The
maximum cannot be tightened without measuring integrator phase.

The **median** carries no such sensitivity, and is what the suite bounds
tightly. It is far more responsive to real error: a `3e-5` relative
perturbation of the polytropic pressure moves Keller-Miksis NHKV's median by a
factor of 38 and its maximum by only 1.35x. Bounds are per case rather than
uniform, because baseline medians span `3e-08` to `1.6e-06` and one threshold
would be set by the worst case.

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

[Back to the README](../README.md)
