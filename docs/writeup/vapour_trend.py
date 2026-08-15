r"""Is the temperature trend physics, or a temperature-graded systematic?

\#277: \texttt{SimulationConfig.vapor} defaults off and no script here ever sets it, and
\texttt{T8} is never set either, so every fit runs at \SI{298.15}{\kelvin} whatever temperature
the gel was at. Two consequences, and the second is what makes this worth a run rather than a
footnote.

THE ERROR IS GRADED WITH TEMPERATURE. With vapour off the bubble contains only its
non-condensable charge, about \SI{25}{\pascal} at $R_{\max}$, where the saturated vapour pressure
is $1702$, $2770$ and \SI{4916}{\pascal} at $15$, $23$ and \SI{33}{\celsius}. The driving
pressure is therefore wrong by roughly $1.7$, $2.7$ and \SI{4.9}{\percent} --- not a constant
offset that a refit absorbs into the material, but a ramp that grows exactly along the axis the
abstract calls the clearest physical signal in the comparison.

AND THE FITTED VISCOSITY ALREADY LOOKS WRONG. The identified $\mu$ RISES with temperature,
$0.0437 \to 0.0520 \to \SI{0.0662}{\pascal\second}$, while $g\alpha$ falls as a warming gel
should. \texttt{temperature\_design.json} puts the Arrhenius slope at
$b_\mu = \SI{-2039.7}{\kelvin}$, an activation energy of \SI{-17.0}{\kilo\joule\per\mole}. No
liquid, polymer solution or gel has a negative activation energy for viscosity; water is
$+16$. Either $\mu$ is not a viscosity, or the trend is an artifact.

THE TEST IS CHEAP BECAUSE THE FIRST-ORDER EFFECT DOES NOT NEED THE THERMAL STACK. Setting
\texttt{vapor=1} with \texttt{bubtherm=0} is permitted and adds $p_v(T_8)$ to the bubble
contents, which is the graded part of the error. Three configurations per dataset separate the
two things being conflated: the published one, then the far-field temperature corrected with
vapour still off, then vapour on. If the negative activation energy is an artifact of a
temperature-graded pressure error, it should shrink or reverse in the third.

WHAT THIS IS NOT. It is not the thermal comparison of \cref{sec:thermal}, which needs
\texttt{bubtherm} and \texttt{masstrans} and therefore latent heat and condensation. It is the
leading term only, and it is reported as such. A trend that survives it is not thereby physics;
a trend that does not survive it is not physics.
"""

import json

import numpy as np

import records
from identified import BOX

RATIO = 38.5
AXES = ("mu", "galpha", "lambda1")
KELVIN = {"gelatin_15C": 288.15, "gelatin_23C": 296.15, "gelatin_33C": 306.15}
PUBLISHED_T8 = 298.15                    # the default, used on all three datasets
GAS_CONSTANT = 8.314462618
STARTS, EVALUATIONS = 32, 800
CASES = (("published", False, False),    # (label, correct T8, vapour on)
         ("T8 corrected", True, False),
         ("T8 + vapour", True, True))


def solver(times, maximum, stretch, t8, vapour):
  import pyimr

  def solve(material, _config=None):
    config = pyimr.SimulationConfig(
      maximum, maximum / stretch, material, dynamics="keller-miksis",
      rtol=1e-8, atol=1e-10, max_steps=400_000,
      vapor=1 if vapour else 0, T8=t8)
    return np.asarray(pyimr.simulate(times, config).radius_ratio, dtype=float), None

  return solve


def candidate():
  import pyimr
  from pyimr.selection import CandidateModel

  def build(t):
    product = t["galpha"]
    return pyimr.QuadraticZener(float(np.sqrt(product * RATIO)), t["mu"], t["lambda1"], 0.0,
                                float(np.sqrt(product / RATIO)))

  return CandidateModel("qSLS", build, AXES)


def one(job):
  """One dataset under one configuration, fitted in the identified coordinates."""
  from pyimr.noise import lack_of_fit
  from pyimr.selection import evaluate_at, fit_candidate, physical_from_unit

  dataset, label, correct_t8, vapour = job
  times, mean, spread, maximum, stretch = records.load(dataset)
  t8 = KELVIN[dataset] if correct_t8 else PUBLISHED_T8
  cand = candidate()
  solve = solver(times, maximum, stretch, t8, vapour)
  try:
    fit = fit_candidate(cand, solve, mean, spread, bounds=BOX, starts=STARTS,
                        max_evaluations=EVALUATIONS)
  except Exception as error:                                                 # noqa: BLE001
    return (dataset, label), {"failed": f"{type(error).__name__}: {error}"}
  fitted = dict(zip(AXES, (float(v) for v in physical_from_unit(AXES, fit.unit, BOX)), strict=True))
  model = np.asarray(evaluate_at(cand, solve, fitted)[0], dtype=float)
  ratio = lack_of_fit(mean, model, spread, records.trial_count(dataset), cand.dimension).ratio
  return (dataset, label), {"t8": t8, "vapour": bool(vapour), "chi2_per_n": float(fit.chi_squared),
                            "lack_of_fit": float(ratio), **fitted}


def arrhenius(values):
  """Slope of `log theta` against `1/T`, and the activation energy it implies."""
  t = np.array([KELVIN[d] for d in records.DATASETS], dtype=float)
  y = np.log(np.array(values, dtype=float))
  x = 1.0 / t - 1.0 / t.mean()
  slope = float(np.polyfit(x, y, 1)[0])
  return slope, slope * GAS_CONSTANT / 1000.0        # K, kJ/mol


def main():
  print("  saturated vapour pressure the model currently omits:")
  from pyimr._prepare import pvsat
  for d, k in KELVIN.items():
    print(f"    {d:>14s}  T8 = {k:.2f} K   p_v = {float(pvsat(k)):.0f} Pa")
  print(f"  published runs use T8 = {PUBLISHED_T8} K and p_v = 0 on all three.\n")

  jobs = [(d, label, t8, v) for d in records.DATASETS for label, t8, v in CASES]
  print(f"  {len(jobs)} fits ...", flush=True)
  with records.pool(len(jobs)) as pool:
    got = dict(pool.map(one, jobs))

  summary = {"kelvin": KELVIN, "cases": {}}
  for label, _, _ in CASES:
    print(f"\n  ==== {label} ====\n")
    print(f"  {'dataset':>14s} {'T8 (K)':>7s} {'mu':>9s} {'g*alpha':>9s} {'lambda1':>10s} "
          f"{'chi2/N':>7s} {'lack of fit':>11s}")
    mus, gas = [], []
    for d in records.DATASETS:
      v = got[(d, label)]
      if "failed" in v:
        print(f"  {d:>14s}  {v['failed'][:60]}")
        continue
      mus.append(v["mu"]); gas.append(v["galpha"])
      print(f"  {d:>14s} {v['t8']:7.2f} {v['mu']:9.4f} {v['galpha']:9.0f} {v['lambda1']:10.3e} "
            f"{v['chi2_per_n']:7.3f} {v['lack_of_fit']:11.2f}")
    if len(mus) == 3:
      b_mu, ea_mu = arrhenius(mus)
      b_ga, ea_ga = arrhenius(gas)
      summary["cases"][label] = {
        "mu": mus, "galpha": gas, "b_mu": b_mu, "activation_kj_mol": ea_mu,
        "b_galpha": b_ga, "galpha_kj_mol": ea_ga,
        "mu_rises": bool(mus[-1] > mus[0])}
      print(f"\n  Arrhenius slope for mu: {b_mu:+.1f} K, "
            f"activation energy {ea_mu:+.1f} kJ/mol"
            f"   {'<-- NEGATIVE, unphysical' if ea_mu < 0 else '<-- positive'}")
      print(f"  for g*alpha:            {b_ga:+.1f} K "
            f"({'falls' if gas[-1] < gas[0] else 'rises'} as the gel warms, as it should)")

  print("\n  ---- what it says ----\n")
  base = summary["cases"].get("published", {})
  best = summary["cases"].get("T8 + vapour", {})
  if base and best:
    print(f"  activation energy for mu: {base['activation_kj_mol']:+.1f} kJ/mol published, "
          f"{best['activation_kj_mol']:+.1f} with T8 corrected and vapour on")
    shrink = abs(best["activation_kj_mol"]) < abs(base["activation_kj_mol"])
    flip = best["activation_kj_mol"] > 0
    summary["shrinks"], summary["reverses"] = bool(shrink), bool(flip)
    print(f"  it {'shrinks' if shrink else 'does not shrink'}; "
          f"it {'reverses to physical' if flip else 'stays negative'}")
    print("\n  A negative activation energy that survives the graded pressure error is not")
    print("  explained by it, and mu is a sink for something else. One that shrinks or reverses")
    print("  says the temperature trend was partly reading an artifact.")
  json.dump(summary, open(records.HERE / "vapour_trend.json", "w"), indent=1)


if __name__ == "__main__":
  main()
