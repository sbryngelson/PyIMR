r"""Can a shape mode be the \SI{72}{\kilo\hertz} anomaly? Not one belonging to the bubble.

`shape_anomaly.py` finds the afterbounce misfit collapsing onto a single bounce that sits at a
common frequency near \SI{72}{\kilo\hertz} rather than a common index, and moving against both
$R_{\max}$ and the sampling interval. A spherical model damps monotonically whatever its
material, so a resonance draining the radial mode is the natural reading, and the natural
candidate is a spherical-harmonic shape mode, which a strictly radial model cannot carry.

THE CANDIDATE IS REFUTED WITHOUT NEEDING ITS COEFFICIENTS, WHICH IS THE POINT OF THIS MODULE.
A shape mode of degree $n$ on a bubble of radius $R_0$ in a medium of density $\rho$ has
%
\begin{equation}
\omega_n^2 \;=\; \underbrace{\frac{(n-1)(n+1)(n+2)\,\sigma}{\rho R_0^3}}_{\text{capillary}}
\;+\; C_n \underbrace{\frac{G}{\rho R_0^2}}_{\text{elastic}},
\label{eq:shapemode}
\end{equation}
%
and $C_n$ is an $O(n)$ number this module never needs. Every term scales as $R_0^{-3/2}$ or as
$\sqrt{G}/R_0$, so ANY mixture of them varies across three datasets by at least as much as the
smaller of those two, and both are large here: $R_0$ spreads \SI{15.8}{\percent} and the
fitted modulus spreads by more than a factor of two. The anomalous frequency spreads
\SI{3.2}{\percent}. No choice of $n$, and no coefficient in front of it, converts a
\SIrange{26}{44}{\percent} predicted variation into a measured \SI{3.2}{\percent} one.

SO THE USEFUL OUTPUT IS NEGATIVE AND SPECIFIC. This reports what each candidate scale actually
does across the three datasets, in both magnitude and TREND, because the trend is the sharper
test: the fitted modulus falls with temperature and $R_0$ rises with it, so every
bubble-intrinsic frequency here falls from \SI{15}{\celsius} to \SI{33}{\celsius}. The
anomalous frequency rises. A quantity that moves the wrong way is not rescued by refitting the
quantity that sets it.

WHAT THAT LEAVES. If no scale belonging to the bubble is constant across these datasets, then
either the near-constancy is the coincidence its $p = 0.016$ admits it might be, or the
frequency belongs to the apparatus rather than to the bubble. Those are distinguishable, and
not by this dataset: they are distinguished by varying $R_0$ deliberately at fixed apparatus,
which is a design statement and belongs with \cref{part:design}.
"""

import json

import numpy as np

import records

SIGMA, RHO, P8, KAPPA = 0.07, 1064.0, 101325.0, 1.4
RATIO = 38.5
DEGREES = (2, 3, 4, 5, 6, 8)
# the anomalous bounce, from shape_anomaly.py
ANOMALY_KHZ = {"gelatin_15C": 71.76, "gelatin_23C": 72.27, "gelatin_33C": 74.12}


def _capillary(n, r0):
  return float(np.sqrt((n - 1) * (n + 1) * (n + 2) * SIGMA / (RHO * r0**3)) / (2 * np.pi))


def _elastic_scale(shear, r0):
  """The elastic frequency scale, without its O(n) coefficient: sqrt(G/rho)/R0."""
  return float(np.sqrt(shear / RHO) / r0 / (2 * np.pi))


def _radial(shear, r0):
  """The RADIAL natural frequency: gas stiffness plus elasticity, the Minnaert form.

  This is not a shape mode and it is the candidate the shape modes were distracting from. An
  afterbounce train decays toward small amplitude, where its frequency must approach this one,
  which is on its own an explanation for why the later periods cluster at all.
  """
  stiffness = 3.0 * KAPPA * P8 + 4.0 * shear
  return float(np.sqrt(stiffness / RHO) / r0 / (2 * np.pi))


def main():
  fits = json.load(open(records.HERE / "per_trial_fits.json"))
  rows = {}
  for dataset in records.DATASETS:
    _, _, _, maximum, stretch = records.load(dataset)
    r0 = maximum / stretch
    galpha = float(fits[dataset]["median"]["galpha"])
    rows[dataset] = {"r0": r0, "shear": float(np.sqrt(galpha * RATIO)), "galpha": galpha}

  print("  Every bubble-intrinsic frequency scales as R0^-3/2 or as sqrt(G)/R0.")
  print("  Both inputs vary strongly across these three datasets.\n")
  print(f"  {'dataset':>13s} {'R0 (um)':>9s} {'G (Pa)':>9s} {'anomaly (kHz)':>14s}")
  for dataset, v in rows.items():
    print(f"  {dataset:>13s} {v['r0']*1e6:9.2f} {v['shear']:9.1f} {ANOMALY_KHZ[dataset]:14.2f}")

  def spread(values):
    a = np.array(values, dtype=float)
    return 100.0 * (a.max() - a.min()) / a.mean()

  print("\n  ---- capillary shape modes, eq:shapemode first term ----\n")
  print(f"  {'degree n':>10s} " + " ".join(f"{d.replace('gelatin_',''):>9s}"
                                           for d in records.DATASETS)
        + f" {'spread':>9s} {'trend':>7s}")
  summary = {"rows": {k: {a: float(b) for a, b in v.items()} for k, v in rows.items()},
             "anomaly_khz": ANOMALY_KHZ, "capillary": {}, "elastic": {}}
  for n in DEGREES:
    vals = [_capillary(n, rows[d]["r0"]) / 1e3 for d in records.DATASETS]
    trend = "rises" if vals[-1] > vals[0] else "falls"
    summary["capillary"][n] = vals
    print(f"  {n:>10d} " + " ".join(f"{v:9.1f}" for v in vals)
          + f" {spread(vals):8.1f}% {trend:>7s}")

  print("\n  ---- the elastic scale sqrt(G/rho)/R0, coefficient omitted ----\n")
  vals = [_elastic_scale(rows[d]["shear"], rows[d]["r0"]) / 1e3 for d in records.DATASETS]
  summary["elastic"]["scale_khz"] = vals
  trend = "rises" if vals[-1] > vals[0] else "falls"
  print(f"  {'G-scale':>10s} " + " ".join(f"{v:9.2f}" for v in vals)
        + f" {spread(vals):8.1f}% {trend:>7s}")

  print("\n  ---- the RADIAL natural frequency, gas plus elasticity ----\n")
  rad = [_radial(rows[d]["shear"], rows[d]["r0"]) / 1e3 for d in records.DATASETS]
  summary["radial_khz"] = rad
  print(f"  {'f0':>10s} " + " ".join(f"{v:9.2f}" for v in rad)
        + f" {spread(rad):8.1f}% {'rises' if rad[-1] > rad[0] else 'falls':>7s}")

  anom = [ANOMALY_KHZ[d] for d in records.DATASETS]
  frac = [a / r for a, r in zip(anom, rad, strict=True)]
  summary["anomaly_over_f0"] = frac
  print(f"\n  {'ANOMALY':>10s} " + " ".join(f"{v:9.2f}" for v in anom)
        + f" {spread(anom):8.1f}% {'rises':>7s}")

  print("\n  ---- what it says ----\n")
  print(f"  R0 spreads {spread([rows[d]['r0'] for d in records.DATASETS]):.1f} percent and G "
        f"spreads {spread([rows[d]['shear'] for d in records.DATASETS]):.1f} percent, so every")
  print(f"  candidate above spreads at least {min(spread(summary['capillary'][n]) for n in DEGREES):.0f} "
        f"percent. The anomaly spreads {spread(anom):.1f} percent.")
  falls = sum(1 for n in DEGREES if summary["capillary"][n][-1] < summary["capillary"][n][0])
  print(f"  And {falls} of {len(DEGREES)} capillary degrees FALL with temperature, as does the")
  print("  elastic scale, because R0 rises and G falls. The anomaly rises. A candidate that")
  print("  moves the wrong way is not rescued by adjusting the coefficient in front of it.")
  print("\n  The radial natural frequency is the one candidate of the right SIZE: the anomaly")
  print(f"  sits at {min(frac):.2f} to {max(frac):.2f} of it. That ratio spreads {spread(frac):.0f} percent, so it is")
  print("  not constant either, but an afterbounce train decaying toward small amplitude must")
  print("  approach f0, which explains why the later periods cluster without any resonance.")
  print("\n  No frequency belonging to this bubble is constant to 3 percent across these three")
  print("  datasets. Either the constancy is the coincidence p = 0.016 allows, or the frequency")
  print("  is the apparatus. Varying R0 at fixed apparatus separates them; nothing here does.")
  summary["verdict"] = {"anomaly_spread_pct": float(spread(anom)),
                        "r0_spread_pct": float(spread([rows[d]["r0"] for d in records.DATASETS])),
                        "shear_spread_pct": float(spread([rows[d]["shear"]
                                                          for d in records.DATASETS]))}
  json.dump(summary, open(records.HERE / "shape_modes.json", "w"), indent=1)


if __name__ == "__main__":
  main()
