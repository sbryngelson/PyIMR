r"""An estimate of the stretch that does not use the constitutive model at all.

\Cref{sec:reqprior} finds every record wanting a larger equilibrium radius than the stored
stretch gives, and the stored value's provenance is a ratio the dataset table calls
$R_{\max}/R_\infty$ --- the radius at infinity, which the loader then hands to the solver as
$R_{\rm eq}$. Those agree only if the bubble has fully settled, and these records have not: at
\SI{23}{} and \SI{33}{\celsius} the trace is still climbing steeply at the last sample, ending
at $0.239$ and $0.291$ against stored $1/\text{stretch}$ of $0.136$ and $0.146$. So $R_\infty$
came from the paper's table rather than from these files, and whether it is the same quantity
the solver wants cannot be settled from the fits, which is what makes an independent estimate
worth having.

Energy conservation supplies one. A bubble released at rest at $R_{\max}$ and arriving at rest
at $R_{\min}$ has had the ambient pressure's work go into compressing the gas,
%
    P_\infty (V_{\max} - V_{\min}) = \int_{V_{\min}}^{V_{\max}} P_g \, dV,
    \qquad P_g = P_{g0}\left(R_{\rm eq}/R\right)^{3\kappa},

which for $\kappa \neq 1$ integrates in closed form and can be solved for $R_{\rm eq}$ given the
two radii the record shows directly. No shear modulus, no viscosity, no relaxation time: the
only inputs are the observed collapse depth, the ambient pressure and the polytropic exponent.

IT IS A BOUND, AND THE DIRECTION IS KNOWN. Viscosity and elasticity both remove energy from the
collapse, so a real bubble arrives shallower than this ideal one for the same gas content.
Matching an observed depth without dissipation therefore requires MORE gas than the truth --- a
larger $R_{\rm eq}$, hence a smaller stretch. What comes out is a lower bound on the stretch,
and a lower bound is exactly what is useful here: the stored values are alleged to be too LARGE,
and a bound above them would refute that outright.

MEASURED, THE BOUND IS TOO WEAK TO SETTLE ANYTHING, and that is the result. It lands at $2.0$
to $2.7$ against a stored $6.8$ to $7.4$ and a fitted $6.4$ to $6.8$: consistent with both, and
therefore evidence for neither. Gelatin dissipates enough that an ideal gas cushion is nowhere
near the binding constraint on how deep the collapse goes, so the inequality is true and
useless.

Kept because the next person to want a model-free handle on the stretch will think of this one
first. It is the obvious independent check, it is cheap, and it does not work.
"""

import json

import numpy as np
from scipy.optimize import brentq

import records


def equilibrium_from_depth(maximum, minimum, kappa, ambient, surface_tension, vapor):
  """Solve the energy balance for `R_eq`, in metres."""
  def imbalance(req):
    # P_g0 from the Laplace condition at equilibrium, less the vapour that does not compress
    pressure = ambient + 2.0 * surface_tension / req - vapor
    work_on_gas = (4.0 * np.pi / (3.0 * kappa - 3.0)) * pressure * req ** (3.0 * kappa) * (
      minimum ** (3.0 - 3.0 * kappa) - maximum ** (3.0 - 3.0 * kappa))
    driving = (4.0 * np.pi / 3.0) * (ambient - vapor) * (maximum**3 - minimum**3)
    return driving - work_on_gas
  # bracket generously: the answer is a few tens of microns for a few hundred micron bubble
  low, high = minimum * 1.001, maximum * 0.999
  if imbalance(low) * imbalance(high) > 0: return float("nan")
  return float(brentq(imbalance, low, high, xtol=1e-12, rtol=1e-12))


def main():
  import pyimr
  physics = pyimr.PhysicalParameters()
  ambient = physics.far_field_pressure_pa
  surface = physics.surface_tension_n_m
  vapor = 0.0                      # cold records; vapour would only reduce the driving further

  print(f"\n  ambient {ambient:.4g} Pa, surface tension {surface:.4g} N/m\n")
  print(f"  {'record':13s} {'stored':>8s} {'fitted':>8s} " +
        " ".join(f"{'kappa=' + str(k):>12s}" for k in (1.2, 1.4, 1.667)))
  table = {}
  fitted = json.load(open(records.HERE / "stretch_offset.json"))
  for dataset in records.DATASETS:
    times, mean, spread, maximum, stretch = records.load(dataset)
    minimum = maximum * float(mean.min())
    row = {}
    for kappa in (1.2, 1.4, 1.667):
      req = equilibrium_from_depth(maximum, minimum, kappa, ambient, surface, vapor)
      row[str(kappa)] = float(maximum / req) if np.isfinite(req) else float("nan")
    table[dataset] = {"stored": float(stretch),
                      "fitted": fitted[dataset]["implied_stretch"], "bounds": row}
    print(f"  {dataset:13s} {stretch:8.3f} {table[dataset]['fitted']:8.3f} " +
          " ".join(f"{row[str(k)]:12.3f}" for k in (1.2, 1.4, 1.667)))

  print("\n  A lower bound BELOW the stored value leaves the question open -- dissipation could")
  print("  account for the gap. A lower bound ABOVE it would refute the offset outright.")
  for dataset, row in table.items():
    best = max(v for v in row["bounds"].values() if np.isfinite(v))
    verdict = ("REFUTES the offset" if best > row["stored"]
               else "consistent with the offset" if best < row["fitted"]
               else "between the fitted and stored values")
    print(f"    {dataset:13s} largest bound {best:6.3f} vs stored {row['stored']:.3f}"
          f" and fitted {row['fitted']:.3f}: {verdict}")

  json.dump(table, open(records.HERE / "stretch_from_collapse.json", "w"), indent=1)


if __name__ == "__main__":
  main()
