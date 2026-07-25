# imr-fast

Fast, validated, differentiable solvers for **inertial microcavitation rheometry
(IMR)**, plus a suite of constitutive models with numerically-computed stress
integrals.

Built to make large-scale IMR inference tractable: the reference MATLAB
implementation ([IMRv2](https://github.com/InertialMicrocavitationRheometry/IMRv2))
takes ~0.4–1 s per solve plus ~5 s startup, which is prohibitive for campaigns
needing 10⁴–10⁶ forward evaluations.

```bash
python3 tests/run_validation.py     # full validation suite
```

## Contents

| file | what it is |
|---|---|
| `imr_fast.py` | forward solver — **11 ms/solve, 40–90× faster than IMRv2** |
| `imr_grad.py` | differentiable solver via forward sensitivity equations — exact analytic ∂R/∂c |
| `constitutive.py` | constitutive suite: stress integrals computed numerically from strain-energy / viscosity functions |
| `imr_nonlinear.py` | Giesekus & Phan-Thien-Tanner via a Lagrangian material-point ODE system |
| `tests/run_validation.py` | validation against IMRv2 reference trajectories + closed forms + model-reduction limits |

## Scope

`imr_fast` covers a deliberately narrow slice of IMRv2, matching the
configuration it was built for:

| option | setting |
|---|---|
| `radial` | **1** Rayleigh–Plesset, **2** Keller–Miksis (pressure form) |
| `bubtherm` | 0 — polytropic, κ=1.4 (no thermal PDE) |
| `masstrans` | 0 |
| `vapor` | **0 or 1** (saturation pressure at `T8`) |
| `wave_type` | **0** constant offset, **1** Gaussian, **2** histotripsy, **3** Heaviside step (`pA`, `TW`, `DT`, `omega`, `mn`) |
| `stress` | **1** neo-Hookean KV, **2** quadratic KV (`alphax`), **3** linear Maxwell/Jeffreys/Zener, **5** UCM/Oldroyd-B |

**It will silently give wrong answers outside this scope** — Gilmore, thermal
PDE, mass transfer, PTT, or Giesekus. Re-validate before extending.

## Deliberate scope: ODE integrators only

Everything here is a **system of ODEs** — Rayleigh–Plesset or Keller–Miksis for the
wall motion, plus internal stress variables for the memory models. That is what
makes it fast, and what makes exact forward-sensitivity gradients practical.

**Out of scope by design:** models requiring spatial discretization inside the
bubble or the surrounding medium — the thermal PDE (`bubtherm=1`, `medtherm=1`)
and mass transfer (`masstrans=1`). These are a different class of solver, and
adding them would sacrifice both the speed and the gradient machinery that are the
point of this package. Use IMRv2 for those.

Two consequences worth knowing:

- **`collapse=1`** (shooting-computed initial stress) is **not** implemented, and
  cannot be added in isolation: IMRv2 requires `bubtherm=1`, `medtherm=1`,
  `masstrans=1` *and* `vapor=1` before it will accept `collapse=1`, so there is no
  way to validate a standalone implementation. Runs here start from an unstressed
  state (`Szero=0`). If your problem has a relaxation time comparable to or longer
  than the observation window, that initial condition matters — check it.
- **PTT and Giesekus** *are* implemented, in `imr_nonlinear.py` — see below. They
  are often described as requiring PDEs in the surrounding medium (Warnez &
  Johnsen 2015), which is why IMRv2's ODE path stops at `stress=5`. That framing
  is Eulerian. What they actually lack is **analytic closure** of the stress
  integral, which is not the same thing as needing a PDE.

### Note on the upstream reference

`IMRv2/src/f_init_stress.m` line 44 uses a variable `z1` that is never defined in
that file. It sits in the `De == 0 || De == Inf` branch, which is unreachable for
the `stress=3,4` cases that actually call the function — so it is a latent bug in
dead code rather than an active one, but it is worth reporting upstream.

## Validation status

Against IMRv2 reference trajectories:

| case | max abs. deviation |
|---|---|
| Zener truth, De=2, stretch 6 | 2.6e-05 |
| NHKV across (G, µ) grid | 2.5e-05 – 6.3e-05 |
| qKV (`stress=2`), alphax = 0.10 / 0.25 | 9.1e-06 / 1.7e-05 |
| UCM–Oldroyd-B (`stress=5`), De = 0.5 / 2.0 | 2.2e-05 / 6.2e-05 |
| Keller–Miksis (`radial=2`) + NHKV | 8.6e-06 |
| Keller–Miksis (`radial=2`) + Zener | 2.3e-05 |
| Gaussian forcing, pA = 5e4 / 2e5 Pa | 1.2e-05 / 1.1e-05 |
| constant-offset / Heaviside / histotripsy forcing | 3.6e-05 / 1.4e-05 / 1.1e-05 |
| `vapor=1` (saturation pressure at 298.15 K) | 1.5e-05 |

Deviations are at integrator-tolerance level, not model error.

Constitutive suite:

| check | error |
|---|---|
| elastic vs neo-Hookean closed form, R/Req ∈ [1.2, 50] | 6e-07 – 2.1e-05 |
| viscous vs −4µṘ/R | 1.4e-06 |
| Gent / Mooney / Fung / Yeoh → neo-Hookean limits | 0 – 2.7e-08 |
| power-law(n=1), Carreau(λ=0), Cross(λ=0) → Newtonian | exact |
| gradients vs finite differences | 1e-05 – 5.4e-04 |

## Constitutive models

Stress integrals are computed **numerically from the strain-energy function**, so
adding a model requires no hand derivation and no derivation can silently be wrong.

**Elastic** — stretch-coordinate form (finite interval, no truncation):

$$S_{el}=\int_1^{\lambda_w}\frac{2\,(\tau_{rr}-\tau_{\theta\theta})(\lambda)}{\lambda(\lambda^3-1)}\,d\lambda,\qquad \lambda_w=R/R_{eq}$$

which integrates analytically to $-(G/2)(5-4R_{st}-R_{st}^4)$ for neo-Hookean.

Available: `neo_hookean`, `mooney_rivlin`, `yeoh`, `fung`, `gent`, `arruda_boyce`.

**Viscous** — $u=R/r$ substitution (also a finite interval):

$$S_{vis}=-12\frac{\dot R}{R}\int_0^1 \mu(\dot\gamma)\,u^2\,du,\qquad \dot\gamma=2\sqrt3\,|\dot R|u^3/R$$

giving exactly $-4\mu\dot R/R$ for Newtonian.

Available: `newtonian`, `power_law`, `carreau_yasuda`, `cross`,
`herschel_bulkley`, `bingham`.

### Gent lock-up

Gent materials lock when $I_1-3\ge J_m$. At bubble stretch $\lambda_w$,
$I_1\approx2\lambda_w^2$, so $J_m=5$ is unphysical for **any** stretch ≥ 2, and
IMR's typical stretch 2–10 range needs $J_m\gtrsim200$. `lock_free()` detects this
rather than returning garbage.

## Nonlinear memory: Giesekus and PTT

Giesekus's quadratic term and PTT's trace function prevent the stress integral
from closing into a finite set of moments — unlike UCM/Oldroyd-B, which collapses
to two variables (Z₁, Z₂). But with a **known** velocity field the constitutive
law contains only the *material* derivative, so in a Lagrangian frame each
material point obeys an ODE in time, with no spatial derivatives and no coupling
between points:

$$\mathbf L=\mathrm{diag}(-2a,a,a),\quad a=\frac{R^2\dot R}{r^3},\qquad
\dot\tau_{ii}=-\frac{\tau_{ii}}{De}+2L_{ii}\tau_{ii}+2\eta_p D_{ii}/De-(\text{nonlinear})$$

`imr_nonlinear.simulate_nonlinear` integrates this on a wall-clustered Lagrangian
grid (the near-wall region needs the resolution: incompressibility maps
$r_0\in[1,1.1]$ onto $r\in[0.1,0.69]$ at collapse) and evaluates
$S=\int_R^\infty \frac{2}{r}(\tau_{rr}-\tau_{\theta\theta})\,dr$ by quadrature.

Validated by reduction: `alpha=0` (Giesekus) and `eps=0` (PTT) both reproduce the
IMRv2-validated UCM to **3.5e-03** at `NM=480`, converging in `NM`
(1.4e-01 → 6.7e-02 → 1.2e-02 → 3.5e-03 for NM = 30, 60, 120, 480).

**Cost:** 2·NM extra ODE states — ~0.12 s/solve at NM=480, about 11× the
closed-form models. Use `imr_fast` when the model admits closure.

## Gradients

`imr_grad.simulate_grad` returns `(R, dR/dc)` using forward sensitivity equations
— exact analytic derivatives, no autodiff dependency:

$$\frac{ds_k}{dt}=J_y\,s_k+\frac{\partial f}{\partial c_k},\qquad \frac{\partial f_2}{\partial c_k}=\frac{\phi_k}{R}$$

for a constitutive law linear in the coefficients, $S=\sum_k c_k\phi_k(R,\dot R)$.
Cost is ~29 ms for the value plus all 6 gradients (2.6× the value-only solve —
cheaper than finite-differencing, and it makes `least_squares(jac=...)` viable).

## Citation

The reference implementation and the underlying physics:

- Estrada, Barajas, Henann, Johnsen & Franck, *High strain-rate soft material
  characterization via inertial cavitation*, JMPS (2018).
  <https://doi.org/10.1016/j.jmps.2017.12.006>
- Warnez & Johnsen, *Numerical modeling of bubble dynamics in viscoelastic media
  with relaxation*, Physics of Fluids 27, 063103 (2015) — the viscoelastic
  formulation, and the ODE-reducible vs PDE-requiring distinction.
  <https://doi.org/10.1063/1.4922598>
- IMRv2: <https://github.com/InertialMicrocavitationRheometry/IMRv2>
