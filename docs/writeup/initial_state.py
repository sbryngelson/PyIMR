"""Is the operator choice partly an initial-radius measurement in disguise?

`confounding.py` asks how much of each model difference survives refitting the material, and
finds the bubble-dynamics axis 77.7% absorbed. That calculation holds the initial state fixed
at its measured value, which asserts `R0` and `Req` are known exactly. They are not: `R_max`
comes off the images and `Req` is inferred, and the reviewer's claim is that this uncertainty
may matter as much as the choice between neighbouring compressible radial equations.

The claim is checkable rather than believable, because `R0` and `Req` are already
differentiable (`CONFIG_PATHS`). Two questions, both answered by projection:

  1. Widen the refit span to include the initial state. What is left of each model axis then?
     A difference that the initial state can absorb is not evidence about the operator.
  2. Treat `Req` as an axis of its own. How large would its error have to be to mimic the
     dynamics axis, and does it point the same way? If the angle is small the two are
     confounded, and every operator margin in the document is partly a statement about `Req`.

The equivalent-error number is a linear extrapolation, so the perturbation is checked for
linearity first -- quoted without that check it would be a plausible number of unknown worth.
"""

import json

import numpy as np

import records

DATASET = "gelatin_15C"
FIT = {"g": 204.3, "mu": 0.04651, "lambda1": 1.964e-7, "alpha": 5.301}
MATERIAL_PATHS = ("material.shear_modulus_pa", "material.viscosity_pa_s",
                  "material.relaxation_time_s", "material.stiffening")
INITIAL_PATHS = ("R0", "Req")
FRACTIONS = (0.005, 0.01, 0.02)          # for the linearity check; 1% is the quoted axis
SETTINGS = dict(dynamics="keller-miksis", rtol=1e-9, atol=1e-11, max_steps=600_000)


def _material(**over):
  import pyimr

  values = {**FIT, **over}
  if "share" in values:
    return pyimr.TwoModeQuadraticZener(values["g"], values["mu"], values["lambda1"], 0.0,
                                       values["alpha"], values["tau"], values["share"])
  return pyimr.QuadraticZener(values["g"], values["mu"], values["lambda1"], 0.0, values["alpha"])


def _span(columns, tolerance=1e-10):
  """An orthonormal basis for the columns, with rank reported rather than assumed.

  QR would hand back as many vectors as columns whatever their rank, so a sensitivity that
  already lies in the material span would silently widen the basis with a direction made of
  round-off and absorb a little more of everything. The SVD makes that visible instead.
  """
  u, s, _ = np.linalg.svd(np.asarray(columns, dtype=float), full_matrices=False)
  keep = s > tolerance * max(s[0], 1e-300)
  return u[:, keep], int(keep.sum())


def main():
  import pyimr

  times, mean, spread, maximum, stretch = records.load(DATASET)
  equilibrium = maximum / stretch

  def trace(material=None, R0=maximum, Req=equilibrium, **config):
    settings = {**SETTINGS, **config}
    problem = pyimr.SimulationConfig(R0, Req, material or _material(), **settings)
    return np.asarray(pyimr.simulate(times, problem).radius_ratio, dtype=float)

  base = trace()
  axes = {
    "dynamics (KM -> KM/Mie-G)": trace(dynamics="keller-enthalpy", liquid_eos="mie-gruneisen") - base,
    "constitutive (one -> two modes)": trace(_material(tau=2e-6, share=0.2)) - base,
    "thermal (cold -> bubble+medium)": trace(bubtherm=1, medtherm=1) - base,
  }

  # the two refit spans, in whitened units and scaled to fractional changes so the columns
  # are comparable across parameters that differ by ten orders of magnitude
  problem = pyimr.prepare(pyimr.SimulationConfig(maximum, equilibrium, _material(), **SETTINGS))
  paths = MATERIAL_PATHS + INITIAL_PATHS
  values = np.array([FIT["g"], FIT["mu"], FIT["lambda1"], FIT["alpha"], maximum, equilibrium])
  jacobian = np.asarray(problem.solve_with_sensitivities(times, paths).radius_ratio, dtype=float)
  jacobian = jacobian * values / spread[:, None]

  material_basis, material_rank = _span(jacobian[:, :len(MATERIAL_PATHS)])
  widened_basis, widened_rank = _span(jacobian)
  print(f"{DATASET}: what survives refitting, with and without the initial state free\n")
  print(f"  refit span ranks: material {material_rank} of {len(MATERIAL_PATHS)}, "
        f"material+initial {widened_rank} of {len(paths)}")
  if widened_rank <= material_rank:
    print("  WARNING the initial-state columns add no rank: they already lie in the material")
    print("    span, so the widened numbers below are not a wider refit and mean nothing.")

  print(f"\n{'axis':>34} {'size':>8} {'material':>10} {'+initial':>10} {'extra':>8}")
  left = {}
  for name, difference in axes.items():
    whitened = difference / spread
    narrow = whitened - material_basis @ (material_basis.T @ whitened)
    wide = whitened - widened_basis @ (widened_basis.T @ whitened)
    left[name] = narrow
    size = np.linalg.norm(whitened)
    print(f"{name:>34} {size:8.2f} {np.linalg.norm(narrow):10.2f} {np.linalg.norm(wide):10.2f} "
          f"{1 - np.linalg.norm(wide) / max(np.linalg.norm(narrow), 1e-30):8.1%}")
  print("  size and both residuals are in units of the noise, over the whole record.")
  print("  `extra` is the fraction of what the material left behind that the initial state takes.")

  # A near-zero `extra` is the interesting reading here, and it is also exactly what a broken
  # projection would report. So the operation is checked in both directions on the same
  # vectors, at no extra solves: handed the difference itself the basis must absorb all of it,
  # handed a random column it must absorb none.
  probe = axes["dynamics (KM -> KM/Mie-G)"] / spread
  narrow = np.linalg.norm(left["dynamics (KM -> KM/Mie-G)"])
  everything, _ = _span(np.column_stack([jacobian[:, :len(MATERIAL_PATHS)], probe]))
  nothing, _ = _span(np.column_stack([jacobian[:, :len(MATERIAL_PATHS)],
                                      np.random.default_rng(0).standard_normal(probe.size)]))
  absorbed = np.linalg.norm(probe - everything @ (everything.T @ probe))
  ignored = np.linalg.norm(probe - nothing @ (nothing.T @ probe))
  print(f"  controls: the difference itself leaves {absorbed:.1e}, a random column leaves "
        f"{ignored:.2f} of {narrow:.2f}")
  assert absorbed < 1e-8 * narrow and ignored > 0.95 * narrow, "the projection is not projecting"

  # Req as an axis of its own, solved rather than linearised, and checked for linearity before
  # anything is extrapolated from it
  print("\n  Req perturbed alone, at fixed R0:\n")
  print(f"{'fraction':>10} {'size':>8} {'after material refit':>22} {'size/fraction':>15}")
  requiv = {}
  for fraction in FRACTIONS:
    whitened = (trace(Req=equilibrium * (1.0 + fraction)) - base) / spread
    residual = whitened - material_basis @ (material_basis.T @ whitened)
    requiv[fraction] = residual
    print(f"{fraction:10.3%} {np.linalg.norm(whitened):8.2f} {np.linalg.norm(residual):22.2f} "
          f"{np.linalg.norm(residual) / fraction:15.1f}")

  slopes = [np.linalg.norm(requiv[f]) / f for f in FRACTIONS]
  linear = max(slopes) < 1.15 * min(slopes)
  print(f"  {'linear' if linear else 'superlinear'}: residual per unit fraction spans "
        f"{min(slopes):.1f} to {max(slopes):.1f}")
  if not linear:
    print("    So none of these rows may be scaled to another fraction: the equivalent error")
    print("    below is solved for by re-solving, not read off one of them. Extrapolating the")
    print("    1% row linearly overstates it by a third.")

  # solved for, not extrapolated: the slope above is not constant, and a linear reading from
  # the 1% point overstates the answer by a third because the growth is superlinear
  reference, dynamics = requiv[0.01], left["dynamics (KM -> KM/Mie-G)"]
  target = float(np.linalg.norm(dynamics))

  def residual_at(fraction):
    whitened = (trace(Req=equilibrium * (1.0 + fraction)) - base) / spread
    return float(np.linalg.norm(whitened - material_basis @ (material_basis.T @ whitened)))

  low, high = 1e-4, 0.05
  if residual_at(high) < target:
    equivalent = float("nan")
    print(f"\n  No Req error up to {high:.0%} reaches the dynamics residual of {target:.2f}.")
  else:
    for _ in range(20):
      middle = 0.5 * (low + high)
      low, high = (middle, high) if residual_at(middle) < target else (low, middle)
    equivalent = 0.5 * (low + high)
  print(f"\n  A Req error of {equivalent:.2%} leaves the same residual as switching the operator")
  print("  from Keller-Miksis to its Mie-Grueneisen enthalpy form, after the material has")
  print("  absorbed what it can from each.")

  print("\n  angles between what Req leaves behind and what each model axis leaves behind:")
  for name, residual in left.items():
    cosine = float(reference @ residual / max(np.linalg.norm(reference) * np.linalg.norm(residual), 1e-30))
    note = "  CONFOUNDED" if abs(cosine) > 0.9 else ""
    print(f"    Req vs {name.split()[0]:<14} cos = {cosine:+.3f}, "
          f"{np.degrees(np.arccos(np.clip(abs(cosine), 0, 1))):5.1f} deg{note}")

  print("\n  A small angle means the operator margin and the initial-radius error move the")
  print("  evidence together: no experiment can say which of them it saw.")
  records.HERE.joinpath("initial_state.json").write_text(json.dumps({
    "ranks": {"material": material_rank, "widened": widened_rank},
    "axes": {name: {"size": float(np.linalg.norm(d / spread)),
                    "after_material": float(np.linalg.norm(left[name])),
                    "after_widened": float(np.linalg.norm(
                      (d / spread) - widened_basis @ (widened_basis.T @ (d / spread))))}
             for name, d in axes.items()},
    "req": {f"{f}": float(np.linalg.norm(requiv[f])) for f in FRACTIONS},
    "linear": bool(linear), "equivalent_req_error": float(equivalent),
  }, indent=1))


if __name__ == "__main__":
  main()
