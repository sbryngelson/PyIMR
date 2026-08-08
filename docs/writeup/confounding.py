"""Before designing against three model axes, ask whether they are separable at all.

The plan is an experiment that discriminates the bubble-dynamics operator, the constitutive
law and the thermal treatment at once. Optimising a design for that is premature until two
things are known, and both are cheap:

  1. Does each model difference survive REFITTING the material? A discrepancy lying in the
     span of `dR/dtheta` is absorbed by adjusting `g`, `mu`, `lambda1`, `alpha`, and is then
     invisible at every design. This is what makes model error hard to see: it does not
     announce itself, it gets fitted away.
  2. Are the differences separable from EACH OTHER? Two axes whose difference directions are
     nearly parallel cannot be told apart by any experiment, however it is designed --- the
     evidence would move, but it could not say which axis moved it.

Each axis is perturbed from one base configuration and the resulting trace difference is
whitened, projected off the material sensitivity span, and compared with the others. The
angles are the answer; the design problem only becomes well posed for the axes that survive.
"""

import json

import numpy as np

import records

DATASET = "gelatin_15C"
FIT = {"g": 204.3, "mu": 0.04651, "lambda1": 1.964e-7, "alpha": 5.301}
PATHS = ("material.shear_modulus_pa", "material.viscosity_pa_s",
         "material.relaxation_time_s", "material.stiffening")


def _material(**over):
  import pyimr

  values = {**FIT, **over}
  if "share" in values:
    return pyimr.TwoModeQuadraticZener(values["g"], values["mu"], values["lambda1"], 0.0,
                                       values["alpha"], values["tau"], values["share"])
  return pyimr.QuadraticZener(values["g"], values["mu"], values["lambda1"], 0.0, values["alpha"])


def main():
  import pyimr

  times, mean, spread, maximum, stretch = records.load(DATASET)

  def trace(material=None, **config):
    settings = dict(dynamics="keller-miksis", rtol=1e-9, atol=1e-11, max_steps=600_000)
    settings.update(config)
    problem = pyimr.SimulationConfig(maximum, maximum / stretch, material or _material(), **settings)
    return np.asarray(pyimr.simulate(times, problem).radius_ratio, dtype=float)

  base = trace()
  axes = {
    "dynamics (KM -> KM/Mie-G)": trace(dynamics="keller-enthalpy", liquid_eos="mie-gruneisen") - base,
    "constitutive (one -> two modes)": trace(_material(tau=2e-6, share=0.2)) - base,
    "thermal (cold -> bubble+medium)": trace(bubtherm=1, medtherm=1) - base,
  }

  # the material sensitivity span, in the same whitened units
  problem = pyimr.prepare(pyimr.SimulationConfig(maximum, maximum / stretch, _material(),
                                                 dynamics="keller-miksis", rtol=1e-9, atol=1e-11, max_steps=600_000))
  jacobian = np.asarray(problem.solve_with_sensitivities(times, PATHS).radius_ratio, dtype=float)
  jacobian = jacobian * np.array([FIT["g"], FIT["mu"], FIT["lambda1"], FIT["alpha"]]) / spread[:, None]
  basis, _ = np.linalg.qr(jacobian)

  print(f"{DATASET}: three model axes, each perturbed from the same base\n")
  print(f"{'axis':>34} {'size':>9} {'after refit':>12} {'absorbed':>10}")
  left = {}
  for name, difference in axes.items():
    whitened = difference / spread
    residual = whitened - basis @ (basis.T @ whitened)
    left[name] = residual
    size, remains = np.linalg.norm(whitened), np.linalg.norm(residual)
    print(f"{name:>34} {size:9.2f} {remains:12.2f} {1 - remains / max(size, 1e-30):10.1%}")
  print("  size and `after refit` are in units of the noise, over the whole record.")

  print("\n  angles between what each axis leaves behind:")
  names = list(axes)
  for i, a in enumerate(names):
    for b in names[i + 1:]:
      x, y = left[a], left[b]
      cosine = float(x @ y / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-30))
      note = "  CONFOUNDED" if abs(cosine) > 0.9 else ""
      print(f"    {a.split()[0]:>14} vs {b.split()[0]:<14} cos = {cosine:+.3f}, "
            f"{np.degrees(np.arccos(np.clip(abs(cosine), 0, 1))):5.1f} deg{note}")

  print("\n  An axis whose residual is small is absorbed by refitting the material and cannot")
  print("  be seen at any design. Two axes at a small angle move the evidence together and")
  print("  cannot be separated. Only what is left is worth designing for.")
  records.HERE.joinpath("confounding.json").write_text(json.dumps(
    {name: {"size": float(np.linalg.norm(d / spread)), "after_refit": float(np.linalg.norm(left[name]))}
     for name, d in axes.items()}, indent=1))


if __name__ == "__main__":
  main()
