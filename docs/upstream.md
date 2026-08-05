# Reference implementation

Defects found in IMRv2 at `dea31cd`, all reproduced with MATLAB R2025a via
`tools/gen_imrv2_cases.m` and `tools/probe_viscosity.m`. The full list is below;
the original scoping notes are in git history, in a `PLAN.md` retired in #218.

- **Giesekus and linear PTT cannot be run.** `f_call_params.m` dispatches
  `stress` 6 and 7 and forces spectral collocation for both, but its own input
  gate rejects `stress > 5`. PyIMR implements both.
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
  silently ignored. PyIMR implements the precursor for Oldroyd-B and the
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
  2.08e-05. PyIMR root-finds `v = 0` instead, which is O(tol).
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

These are the reason several PyIMR models are validated by reduction limit
rather than against a pinned upstream trajectory: for those models, no working
upstream implementation exists to pin against.

[Back to the README](../README.md)

- **`calc_omega_N` (IMR-vanilla) treats the gas pressure at `Rmax` as the
  equilibrium value.** That inflates the linearised stiffness by
  `alpha**(-3*kappa)` and overpredicts the natural frequency by 42x on the
  reference case, which is why PyIMR's `data.natural_frequency` is a
  reimplementation rather than a port.
- **`radial = 6` (Gilmore/Mie-Gruneisen) returns complex radii.** Upstream
  reaches `max|imag(R/R0)| = 4.069` without raising, from a wrong root of the
  Mie-Gruneisen density quadratic.

## Which branches replicate upstream, and which correct it

Moved here from the package docstring, where four of its claims had gone stale
without anyone noticing — it still said `radial = 6` was "NOT supported,
confirmed dead/broken upstream" long after #18 implemented it, and still pointed
at a `tests/run_validation.py` that #32 split up.

**`bubtherm = 1`** implements IMRv2's `elseif bubtherm` branch of `f_imr_fd.m`:
gas-phase thermal PDE, dry gas (`kv0 = 0`, `vapor = 0`). With `medtherm = 0` the
wall is an isothermal-equivalent clamp (`thetadot[-1] = 0`). Its `Pdot` uses bare
`P` (`kappa*P`), **not** `(P - Pv)` — that is IMRv2's actual equation for this
branch rather than a simplification, and the `bubtherm = 0` polytropic branch's
`Pdot` does use `(P - Pv)`. The two are deliberately not reconciled: they are
genuinely different equations in the source.

**`medtherm = 1`** adds the liquid boundary layer — a stretched exterior grid
(`Mt` points, `Lt` controlling the stretching) and an advection + diffusion +
viscous-dissipation right-hand side for `Tm`. The wall temperature `theta[-1]`
is not a free state; it is an algebraic boundary value enforcing heat-flux
continuity across the interface, and is **solved in closed form** (#57): the
residual is a quadratic in `sqrt((alpha + beta)^2 + 2*alpha*theta)`. Upstream
iterates a secant here. `thetadot[-1] = 0` and `Tmdot[0] = 0` always, because
both slots are algebraic rather than evolved. Forward sensitivities
differentiate the boundary solve.

**`masstrans = 1`** (needs `bubtherm = 1`, `vapor = 1`) implements the
`if bubtherm && masstrans` branch: a wall vapour mass fraction field `kv(y, t)`,
a `kv`-weighted mixture conductivity and diffusivity, extra mass-transfer terms
in `Pdot`/`Uvel`/`thetadot`, and a `kvdot` equation. `kv[-1]` is set
algebraically each RHS call from vapour-liquid equilibrium using a `T[-1]`
computed from the **stale, pre-update** `kv[-1]` — IMRv2's own one-step lag,
replicated exactly rather than reconciled.

With `medtherm = 0` and mass transfer on, `theta[-1]` never evolves, so
`T[-1] == 1` identically and no wall solve is needed. With both on, `theta[-1]`
comes from a coupled root-find (`_wall_theta_bw_full`) that enforces
vapour-mass-flux continuity alongside heat flux; no closed form exists there,
because the vapour fraction puts `Tw` inside `pvsat`. `alpha_m` in that solve
uses the stale `kv[-1]` too, same lag. Forward sensitivities cover it.

## Deliberate numerical divergence

`Zener` and `QuadraticZener` use `4*LAM/Re8` for the acceleration coefficient
where IMRv2 uses `4/Re8`. On compressible trajectories with differing
retardation and relaxation times the two differ by roughly 5e-02. IMRv2's own
stress carries `-4*LAM/Re8*Rdot/R`, so the coefficient it pairs with that stress
is internally inconsistent, and the reduction limit to `LinearMaxwell` converges
only with the `LAM` factor restored. Three Zener reference trajectories were
regenerated from PyIMR as a result, and pin regressions rather than
cross-checking upstream (#174, IMRv2#18).
