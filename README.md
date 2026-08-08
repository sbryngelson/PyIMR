# PyIMR

[![CI](https://github.com/sbryngelson/PyIMR/actions/workflows/ci.yml/badge.svg)](https://github.com/sbryngelson/PyIMR/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/sbryngelson/PyIMR/branch/master/graph/badge.svg)](https://codecov.io/gh/sbryngelson/PyIMR)
[![PyPI](https://img.shields.io/pypi/v/pyimr)](https://pypi.org/project/PyIMR/)
[![Python](https://img.shields.io/pypi/pyversions/pyimr)](https://pypi.org/project/PyIMR/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![DOI](https://zenodo.org/badge/1311535002.svg)](https://doi.org/10.5281/zenodo.21792446)

Fast, validated solvers for **inertial microcavitation rheometry** — measuring
how soft materials behave at strain rates a rheometer cannot reach, by watching
a bubble collapse inside them.

Built for inference campaigns that need thousands of forward solves: closed-form
hot paths for the common constitutive laws, exact forward sensitivities from the
production right-hand side, and Bayesian model comparison on top.

```bash
pip install pyimr
```

## Getting started

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

result.radius_ratio     # R(t)/R0
result.internal_pressure_pa
```

Every input is dimensional and every material is explicit and typed — no integer
constitutive selector, no shared bag of parameters.

## What it models

| | |
|---|---|
| bubble dynamics | `dynamics=` Rayleigh-Plesset, Keller-Miksis, Keller enthalpy, Herring, Gilmore, Lezzi-Prosperetti (2nd order); the last four take `liquid_eos=` Tait, Mie-Gruneisen or Noble-Abel stiffened gas |
| thermodynamics | polytropic closure, or gas and liquid thermal PDEs with vapor transport |
| discretization | Chebyshev collocation (default) or second-order finite difference |
| forcing | constant, Gaussian, histotripsy, Heaviside step, or a sampled pressure history |
| materials | hyperelastic, generalized-Newtonian, viscoelastic memory, distributed nonlinear memory |

Materials compose: pick an elastic law and a viscous law and combine them, or
reach for a closed-form memory model when it applies. Neo-Hookean, Mooney-Rivlin,
Yeoh, Fung, Gent, Arruda-Boyce and Ogden on the elastic side; Carreau-Yasuda,
Cross, Powell-Eyring, Herschel-Bulkley and Bingham on the viscous; Zener,
Oldroyd-B, Giesekus and linear PTT for memory. See
**[docs/materials.md](docs/materials.md)**.

## Beyond a forward solve

**Sensitivities.** The tangent-linear solver differentiates the production RHS,
not a surrogate — across materials, thermal states, forcing and geometry. Six
simultaneous gradients cost about 1.9 forward solves.

**Inference.** Prepared likelihoods with analytic Jacobians, deterministic
multistart, and process-parallel batch evaluation. A PyMC bridge runs NUTS on
the exact tangents.

**Model selection.** Constitutive models nest, so comparing best fits always
favours the flexible ones. `pyimr.selection` scores by evidence instead, with
redundancy and Occam penalties.

**Knowing your resolution.** `pyimr.resolution` measures the cheapest grid and
tolerance meeting an accuracy target on *your* problem, and raises rather than
guessing when the target is out of reach.

See **[docs/usage.md](docs/usage.md)**.

## Validation

The suite pins IMRv2 trajectories across radial equations, forcing, vapor, heat
transfer, mass transfer and the specialized constitutive models, and separately
checks closed forms, reduction limits, and every analytic tangent against
independent centered differences.

PyIMR reproduces IMRv2 except where upstream is wrong. Eight defects were found
and each correction validated against something other than upstream — a closed
form, an independent equation of state, or a reduction limit.

```bash
pytest                  # everything, including numerical validation
pytest -m "not slow"    # skip the high-resolution convergence studies
```

The suite prints a table of measured deviations after the run, not just
pass/fail: a check that still passes but has moved an order of magnitude is
visible.

## Documentation

| | |
|---|---|
| [Usage](docs/usage.md) | solving, sensitivities, inference, model selection, resolution |
| [Materials](docs/materials.md) | every constitutive law, and what each one requires |
| [Accuracy](docs/accuracy.md) | what error each tolerance and discretization actually buys |
| [Discretization](docs/discretization.md) | stress quadrature and the two thermal backends |
| [Validation](docs/validation.md) | what the suite pins, and per-case deviations from IMRv2 |
| [Upstream](docs/upstream.md) | defects found in IMRv2, and what PyIMR does instead |
| [Boundaries](docs/boundaries.md) | where PyIMR stops, and where it diverges deliberately |
| [Open work](docs/open-work.md) | what is known to be unfinished, and what blocks what |

API reference: `pip install 'PyIMR[docs]'`, then `python -m pdoc pyimr`.

## Citation

If you use PyIMR, please cite it via [CITATION.cff](CITATION.cff), along with:

- Estrada, Barajas, Henann, Johnsen & Franck, *High strain-rate soft material
  characterization via inertial cavitation*, JMPS (2018).
  <https://doi.org/10.1016/j.jmps.2017.12.006>
- Warnez & Johnsen, *Numerical modeling of bubble dynamics in viscoelastic media
  with relaxation*, Physics of Fluids 27, 063103 (2015).
  <https://doi.org/10.1063/1.4922598>

Built on [IMRv2](https://github.com/InertialMicrocavitationRheometry/IMRv2).

## License

MIT — see [LICENSE](LICENSE).
