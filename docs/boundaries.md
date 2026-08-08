# Boundaries

Where PyIMR stops, and where it deliberately parts company with IMRv2.

## Conventions

- `PhysicalParameters` defaults reproduce the pinned reference trajectories. The
  default polytropic exponent is 1.4; IMRv2 itself ships with 1.47.
- `SimulationResult.stress_state` contains nondimensional internal variables.
  Public dimensional outputs carry units in their names or documentation.
- Collapse shooting requires a material with memory, and cannot be combined with
  an explicit initial stress state or a nonzero observed wall velocity. IMRv2
  does permit collapse for memoryless materials, which PyIMR deliberately does not.

## Deliberate divergences

- `gilmore` with `liquid_eos="mie-gruneisen"` (upstream `radial = 6`) **is supported here**, and is the one
  configuration IMRv2 cannot run at all -- upstream returns complex radii
  without raising. The cause is a wrong root of the Mie-Gruneisen density
  quadratic; see [upstream.md](upstream.md).
- Both Mie-Gruneisen operators (upstream `radial = 5` and `6`) **deliberately diverge from IMRv2**. Upstream's
  Mie-Gruneisen branch is physically wrong; the corrections are validated
  against the independent Tait branches and the weakly-compressible limit rather
  than against upstream. `tests/ref_radial5.csv` is retained as a record of
  upstream behaviour, not as a target.

## Reference implementation

PyIMR reproduces IMRv2 except where upstream is wrong. One divergence is
numerical rather than a defect fix: the `Zener` acceleration coefficient, which
makes three Zener reference trajectories regression pins rather than
cross-checks (#174).

Eight defects were found at `dea31cd`, each reproduced with MATLAB R2025a via
`tools/gen_imrv2_cases.m`, and each correction validated against something other
than upstream -- a closed form, an independent equation of state, or a reduction
limit. The wrong Mie-Gruneisen root, the non-functional non-Newtonian viscosity
suite, the stubbed collapse initialization and the rest are in
[upstream.md](upstream.md).

Those defects are why several PyIMR models are validated by reduction limit
rather than against a pinned upstream trajectory: for those models, no working
upstream implementation exists to pin against.

## Tangent equations

Forward sensitivities integrate

$$
\frac{ds_k}{dt}=J_y s_k+\frac{\partial f}{\partial c_k}.
$$

All requested parameter directions share one augmented integration. Prepared
parameter scaling keeps error control dimensionless; public derivatives are
converted back to dimensional parameter units.

[Back to the README](../README.md)
