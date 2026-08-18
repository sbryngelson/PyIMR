r"""What FUNCTION of the trajectory is the discrepancy?

\Cref{sec:screenres} asks whether adding a candidate physics would remove $\hat\delta$, and its
resolution floor of \SI{20}{\percent} means most of its answers are unreadable. This asks a
different and easier question. $\hat\delta$ is a curve in time along a trajectory the solver
computes in full, so rather than simulating alternative models and testing whether they absorb
it, project it onto terms that can be EVALUATED along that trajectory. If one of them, or a
short combination, reproduces the curve on every record, the missing term is named rather than
screened, and it can be named even if nobody thought to add it to a model.

THE DICTIONARY IS THE PHYSICS THE EQUATION IS BUILT FROM. Mach number and its square, the
acoustic radiation term, the wall acceleration, the surface-tension term going as $1/R$, the
polytropic gas pressure, the Hencky strain rate, the constitutive stress integral itself and the
internal pressure. Each is a real term in some bubble-dynamics equation, and each is computed
along the record's own fitted trajectory rather than postulated.

EVERY TERM IS PROJECTED THE WAY $\hat\delta$ IS. $\hat\delta$ lives in the complement of the
model's own sensitivities, so a dictionary entry is orthogonalised against the same span before
it is scored; otherwise a term the parameters can already absorb would appear to explain a
residual from which that part was removed by construction.

AND THE SCORE NEEDS A NULL, since ten smooth regressors against three hundred correlated
samples will fit something. The same regression is run against surrogates carrying the curve's
spectrum and its orthogonality, which is the floor an explanation has to clear.
"""

import json

import numpy as np

import records
from shape_error import seed_for
from universal_delta import GRID, _surrogate, one as delta_one

DRAWS = 400
ORDER = (*records.DATASETS, *records.PAAM)
KAPPA, SOUND, SURF = 1.4, 1484.0, 0.056


def dictionary(dataset):
  """Physical terms along the record's own fitted trajectory, on the common grid."""
  import pyimr

  times, _, _, maximum, stretch = records.load(dataset)
  fitted = json.load(open(records.HERE / "paam_lackoffit.json"))[dataset]["fitted"]
  material = pyimr.QuadraticZener(fitted["g"], fitted["mu"], fitted["lambda1"], 0.0,
                                  fitted["alpha"])
  config = pyimr.SimulationConfig(maximum, maximum / stretch, material,
                                  dynamics="keller-miksis", rtol=1e-10, atol=1e-12,
                                  max_steps=4_000_000)
  out = pyimr.simulate(times, config)
  radius = np.asarray(out.radius_m, dtype=float)
  velocity = np.asarray(out.wall_velocity_m_s, dtype=float)
  stress = np.asarray(out.stress_integral_pa, dtype=float)
  pressure = np.asarray(out.internal_pressure_pa, dtype=float)
  acceleration = np.gradient(velocity, times)
  equilibrium = maximum / stretch

  terms = {
    "Mach": velocity / SOUND,
    "Mach^2": (velocity / SOUND) ** 2,
    "radiation R'|R'|/c": velocity * np.abs(velocity) / SOUND,
    "wall acceleration R R''/c^2": radius * acceleration / SOUND**2,
    "surface tension 2 sigma / R": 2 * SURF / radius,
    "gas (Req/R)^3kappa": (equilibrium / radius) ** (3 * KAPPA),
    "strain rate R'/R": velocity / radius,
    "stress integral": stress,
    "internal pressure": pressure,
    "trace R/Rmax": radius / maximum,
  }
  tau = times / (maximum * np.sqrt(1064.0 / 101325.0))
  built = {}
  for name, series in terms.items():
    got = np.interp(GRID, tau, np.nan_to_num(series, posinf=0.0, neginf=0.0))
    got = got - got.mean()
    norm = np.linalg.norm(got)
    if norm > 1e-30 and np.all(np.isfinite(got)): built[name] = got / norm
  return built


def _explained(target, design):
  """Share of the target's squared norm reached by the best combination of the columns."""
  basis, _ = np.linalg.qr(design)
  projected = basis @ (basis.T @ target)
  return float((projected @ projected) / (target @ target))


def one(dataset):
  _, payload = delta_one(dataset)
  curve = np.array(payload["curve"])
  span = np.array(payload["span"])
  terms = dictionary(dataset)
  if not terms: return dataset, {"failed": "no terms"}

  # orthogonalise every dictionary entry against the sensitivity span, as delta_hat is
  basis, _ = np.linalg.qr(span.T)
  clean = {}
  for name, column in terms.items():
    residual = column - basis @ (basis.T @ column)
    norm = np.linalg.norm(residual)
    if norm > 1e-9: clean[name] = residual / norm
  names = sorted(clean)
  design = np.column_stack([clean[n] for n in names])

  single = {n: float(np.dot(curve, clean[n]) ** 2) for n in names}
  joint = _explained(curve, design)

  rng = np.random.default_rng(seed_for(dataset + "dict"))
  null_single, null_joint = [], []
  for _ in range(DRAWS):
    fake = _surrogate(rng, curve)
    fake = fake - basis @ (basis.T @ fake)
    fake = fake / (np.linalg.norm(fake) + 1e-30)
    null_single.append(max(float(np.dot(fake, clean[n]) ** 2) for n in names))
    null_joint.append(_explained(fake, design))
  return dataset, {"single": single, "joint": joint,
                   "null_best_single_95": float(np.percentile(null_single, 95)),
                   "null_joint_95": float(np.percentile(null_joint, 95))}


def main():
  with records.pool(len(ORDER)) as pool:
    got = dict(pool.map(one, list(ORDER)))
  good = {d: v for d, v in got.items() if "failed" not in v}
  names = sorted({n for v in good.values() for n in v["single"]})

  print("  share of delta-hat each trajectory term explains, orthogonalised as delta-hat is\n")
  print(f"  {'term':28s} " + " ".join(f"{d.replace('paam_','')[:8]:>9s}" for d in good))
  for name in names:
    cells = []
    for d, v in good.items():
      got_one = v["single"].get(name, 0.0)
      cells.append(f"{got_one:8.1%}" + ("*" if got_one > v["null_best_single_95"] else " "))
    print(f"  {name:28s} " + " ".join(cells))
  print(f"  {'best-single null (95%)':28s} "
        + " ".join(f"{v['null_best_single_95']:8.1%} " for v in good.values()))
  print(f"  {'ALL TEN TOGETHER':28s} "
        + " ".join(f"{v['joint']:8.1%}" + ("*" if v["joint"] > v["null_joint_95"] else " ")
                   for v in good.values()))
  print(f"  {'joint null (95%)':28s} "
        + " ".join(f"{v['null_joint_95']:8.1%} " for v in good.values()))

  print("\n  ---- what it says ----\n")
  winners = {}
  for d, v in good.items():
    above = [n for n, x in v["single"].items() if x > v["null_best_single_95"]]
    winners[d] = above
    print(f"  {d:>16s}: {', '.join(above) if above else 'no term clears its own floor'}")
  common = set.intersection(*(set(w) for w in winners.values())) if winners else set()
  print(f"\n  terms clearing the floor on EVERY record: {sorted(common) if common else 'none'}")
  joint_beats = [d for d, v in good.items() if v["joint"] > v["null_joint_95"]]
  print(f"  the full dictionary beats its own null on {len(joint_beats)} of {len(good)} records.")
  json.dump(good, open("delta_dictionary.json", "w"), indent=1)


if __name__ == "__main__":
  main()
