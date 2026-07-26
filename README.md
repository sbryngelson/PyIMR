# imr-fast

Fast, validated, differentiable solvers for **inertial microcavitation rheometry
(IMR)**, plus a suite of constitutive models with numerically-computed stress
integrals.

Built to make large-scale IMR inference tractable: the reference MATLAB
implementation ([IMRv2](https://github.com/InertialMicrocavitationRheometry/IMRv2))
takes ~0.4–1 s per solve plus ~5 s startup, which is prohibitive for campaigns
needing 10⁴–10⁶ forward evaluations.

```bash
python -m pip install -e ".[test]"
python -m pytest
python tests/run_validation.py
```

## Contents

| file | what it is |
|---|---|
| `imr_fast.py` | main forward solver and strict `SimulationConfig`/`SimulationResult` API |
| `imr_grad.py` | RP/polytropic forward sensitivity solver for a six-term linear stress library |
| `constitutive.py` | standalone elastic and generalized-Newtonian stress-integral models |
| `imr_nonlinear.py` | separate RP/polytropic Giesekus and PTT solver |
| `tests/run_validation.py` | validation against IMRv2 reference trajectories + closed forms + model-reduction limits |

## Scope

The main `imr_fast.simulate` solver currently supports:

| option | setting |
|---|---|
| `radial` | **1** RP, **2** KM pressure, **3** KM enthalpy/Tait, **4** Gilmore/Tait, **5** KM enthalpy/Mie–Grüneisen |
| `bubtherm` | **0** polytropic or **1** gas thermal PDE |
| `medtherm` | **0** or **1** liquid thermal layer; requires `bubtherm=1` |
| `masstrans` | **0** or **1** vapor mass transfer; requires `bubtherm=vapor=1` |
| `vapor` | **0 or 1** (saturation pressure at `T8`) |
| `wave_type` | **0** constant offset, **1** Gaussian, **2** histotripsy, **3** Heaviside step (`pA`, `TW`, `DT`, `omega`, `mn`) |
| `stress` | **0** none, **1** NHKV, **2** qKV, **3** linear Maxwell/Jeffreys/Zener, **4** quadratic Zener family, **5** UCM/Oldroyd-B |

Every combination exercised by the regression suite is checked against a pinned
IMRv2 trajectory. Unsupported option values and inconsistent thermal/mass
combinations raise `ValueError`.

### Public APIs

The original array-returning function remains available:

```python
import numpy as np
from imr_fast import simulate

t = np.linspace(0.0, 120e-6, 300)
radius_ratio = simulate(t, R0=225e-6, Req=37.5e-6, G=2500.0, mu=0.1)
```

New code should prefer the validated, structured API:

```python
from imr_fast import SimulationConfig, simulate_result

config = SimulationConfig(R0=225e-6, Req=37.5e-6, G=2500.0, mu=0.1)
result = simulate_result(t, config)

result.time_s       # immutable input-time copy
result.radius_ratio # R/R0
result.radius_m     # dimensional radius
```

The structured API raises `SimulationError` if the integrator fails. The legacy
API retains its historical all-NaN failure return unless
`raise_on_failure=True` is requested.

### Important boundaries

- Physical constants such as `kappa`, far-field pressure/density, surface
  tension, sound speed, and liquid equation-of-state parameters are currently
  fixed to the reference-validation values. In particular, `kappa=1.4`, whereas
  IMRv2 ships with an overridable default of 1.47.
- The main solver returns radius only. Pressure, velocity, temperature, and
  concentration are integrated internally but are not yet part of its public
  result.
- The models in `constitutive.py` are validated stress-integral functions, not
  selectable main-solver models.
- `imr_grad.py` differentiates a separate RP/polytropic six-term stress-library
  solver. Its gradients do not yet cover the complete `imr_fast.simulate` model.
- `imr_nonlinear.py` is a separate RP/polytropic solver. Its PTT and Giesekus
  implementations do not yet support acoustic forcing or the thermal branches.
- `radial=6` (Gilmore/Mie–Grüneisen) is intentionally unavailable because the
  corresponding upstream IMRv2 branch produces non-real states in the tested
  pressure range.
- **`collapse=1`** (shooting-computed initial stress) is **not** implemented, and
  cannot be added in isolation: IMRv2 requires `bubtherm=1`, `medtherm=1`,
  `masstrans=1` *and* `vapor=1` before it will accept `collapse=1`, so there is no
  way to validate a standalone implementation. Runs here start from an unstressed
  state (`Szero=0`). If the relaxation time is comparable to or longer
  than the observation window, that initial condition matters — check it.

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
| gas / liquid thermal PDE branches | 9.4e-06 / 9.6e-06 |
| vapor mass / coupled heat-mass branches | 1.6e-05 / 6.0e-06 |
| KM-enthalpy/Tait, Gilmore/Tait, KM/Mie–Grüneisen | 8.6e-06 / 8.8e-06 / 1.6e-05 |

These are numerical agreement measurements against the reference
implementation, not an estimate of physical model error.

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
