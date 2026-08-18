r"""Is the missing term the same curve on two unrelated materials?

\Cref{sec:enrich} rests on $\hat\delta$ being one target rather than eight: the three gelatin
discrepancies correlate $0.49$ to $0.84$, "where unrelated curves of this length would sit near
$0.08$". That $0.08$ is $1/\sqrt{N}$, the null for two INDEPENDENT vectors, and these curves are
smooth by construction -- \cref{sec:enrich} itself notes $65$ to \SI{69}{\percent} of each sits
in the first collapse. `correlated_nulls.py` has already shown what that mistake is worth
elsewhere in this document, so the number is recomputed here against surrogates that carry each
curve's own autocorrelation.

AND THE INTERESTING COMPARISON WAS NEVER AVAILABLE BEFORE. Gelatin is a physical gel of triple
helices; polyacrylamide is a covalent network. They share an instrument and nothing else. If
$\hat\delta$ has the same shape on both, then whatever every model here is missing is not the
chemistry of either material -- it is common to the experiment or to the modelling of it, its
shape is measured rather than postulated, and it is a target any theory can be tested against.
If the two materials disagree, the discrepancy is the material's, \cref{sec:enrich}'s premise
holds only within gelatin, and the enrichment programme is a per-material exercise.

THE CURVES MUST BE COMPARED IN PHASE, NOT IN SECONDS. Each record has its own $R_{\max}$ and its
own clock, so $\hat\delta$ is resampled onto a common normalised time before any correlation is
taken -- the same argument \cref{sec:transfer} makes for transferring $\sigma$ and $\tau$.
"""

import json

import numpy as np

import records
from enrichment_screen import directions
from shape_error import seed_for

# a common NORMALISED-TIME grid, over the window every record actually covers
GRID = np.linspace(0.0, 3.9, 300)
DRAWS = 4000
ORDER = (*records.DATASETS, *records.PAAM)


def one(dataset):
  """`delta_hat` against the record's own NORMALISED TIME, not its index fraction.

  Mapping each curve's index range onto [0, 1] stretches a record observed over four
  collapse times onto one observed over five, misaligning them by a quarter of the window.
  Done that way the correlations cluster by window rather than by material, which is an
  artefact of the resampling and was mistaken for a result once already.
  """
  _, split, jac, _ = directions(dataset)
  curve = np.asarray(split.identifiable, dtype=float)
  times, _, _, maximum, _ = records.load(dataset)
  tau = times / (maximum * np.sqrt(1064.0 / 101325.0))
  resampled = np.interp(GRID, tau[: curve.size], curve)
  resampled = resampled - resampled.mean()
  # the sensitivity span this residual was projected out of, resampled the same way, so a
  # null can be built carrying the SAME constraint the measurement carries
  span = []
  for column in np.asarray(jac, dtype=float).T:
    got = np.interp(GRID, tau[: column.size], column)
    span.append((got - got.mean()) / (np.linalg.norm(got - got.mean()) + 1e-30))
  return dataset, {"curve": (resampled / np.linalg.norm(resampled)).tolist(),
                   "span": np.array(span).tolist(),
                   "tau_max": float(tau[curve.size - 1])}


def _surrogate(rng, curve):
  """Phase randomisation: same power spectrum, same autocorrelation, no shared mechanism."""
  spectrum = np.fft.rfft(curve)
  angles = rng.uniform(0.0, 2 * np.pi, spectrum.size)
  angles[0] = 0.0
  shuffled = np.abs(spectrum) * np.exp(1j * angles)
  out = np.fft.irfft(shuffled, n=curve.size)
  out = out - out.mean()
  return out / np.linalg.norm(out)


def main():
  with records.pool(len(ORDER)) as pool:
    raw = dict(pool.map(one, list(ORDER)))
  print("  record windows, in collapse times: "
        + ", ".join(f"{d.replace('paam_','')[:8]} {raw[d]['tau_max']:.1f}" for d in ORDER))
  print(f"  compared over the common {GRID[0]:.1f} to {GRID[-1]:.1f}\n")
  curves = {d: np.array(v["curve"]) for d, v in raw.items()}
  spans = {d: np.array(v["span"]) for d, v in raw.items()}

  def constrained(rng, dataset):
    """A surrogate with the curve's spectrum AND its orthogonality to the sensitivities."""
    out = _surrogate(rng, curves[dataset])
    basis, _ = np.linalg.qr(spans[dataset].T)
    out = out - basis @ (basis.T @ out)
    return out / (np.linalg.norm(out) + 1e-30)

  print("  pairwise |correlation| of the identifiable discrepancy, on a common phase grid\n")
  print(f"  {'':>16s} " + " ".join(f"{d.replace('paam_','')[:8]:>9s}" for d in ORDER))
  matrix = {}
  for a in ORDER:
    cells = []
    for b in ORDER:
      r = float(abs(np.dot(curves[a], curves[b])))
      matrix[f"{a}|{b}"] = r
      cells.append("        -" if a == b else f"{r:9.3f}")
    print(f"  {a:>16s} " + " ".join(cells))

  rng = np.random.default_rng(seed_for("universal"))
  null = np.empty(DRAWS)
  keys = list(ORDER)
  for d in range(DRAWS):
    a, b = rng.choice(len(keys), size=2, replace=False)
    null[d] = abs(np.dot(_surrogate(rng, curves[keys[a]]), _surrogate(rng, curves[keys[b]])))
  cut95 = float(np.percentile(null, 95))

  within_gel, within_paam, across = [], [], []
  for i, a in enumerate(ORDER):
    for b in ORDER[i + 1:]:
      r = matrix[f"{a}|{b}"]
      ga, gb = a in records.DATASETS, b in records.DATASETS
      (within_gel if ga and gb else within_paam if not (ga or gb) else across).append(r)

  tight = np.empty(DRAWS)
  for d in range(DRAWS):
    a, b = rng.choice(len(keys), size=2, replace=False)
    tight[d] = abs(np.dot(constrained(rng, keys[a]), constrained(rng, keys[b])))
  cut_tight = float(np.percentile(tight, 95))

  print("\n  ---- against a null that keeps each curve's own spectrum ----\n")
  print("  and one that ALSO projects out the sensitivity span, as delta-hat is:")
  print(f"    mean {tight.mean():.3f}, 95th percentile {cut_tight:.3f}")
  print(f"  phase-randomised surrogate pairs: mean {null.mean():.3f}, "
        f"95th percentile {cut95:.3f}")
  print(f"  the paper compares against 1/sqrt(N) = {1/np.sqrt(GRID.size):.3f}\n")
  for label, group in (("gelatin with gelatin", within_gel), ("PAAm with PAAm", within_paam),
                       ("gelatin with PAAm", across)):
    beat = sum(1 for r in group if r > cut95)
    print(f"  {label:>22s}: {len(group):2d} pairs, |r| {min(group):.3f} to {max(group):.3f}, "
          f"median {np.median(group):.3f}, {beat} above the null")

  print(f"\n  cross-material pairs above the constrained null: "
        f"{sum(1 for r in across if r > cut_tight)} of {len(across)}")

  print("\n  ---- what it says ----\n")
  if np.median(across) > cut95 and np.median(within_gel) > cut95:
    print("  The discrepancy is the SAME CURVE on two unrelated networks, beyond a null that")
    print("  already grants it the smoothness. What every model here is missing is therefore not")
    print("  the chemistry of either material, and its shape is a measured target rather than a")
    print("  postulated one.")
  elif np.median(within_gel) > cut95 >= np.median(across):
    print("  It is one curve WITHIN gelatin and a different one across materials, so")
    print("  sec:enrich's premise holds per material and the enrichment programme is")
    print("  per material too.")
  else:
    print("  The cross-dataset agreement does not survive a null that carries the curves' own")
    print("  smoothness. sec:enrich's 'one target, measured rather than guessed' rests on a")
    print("  1/sqrt(N) null and has to be withdrawn.")
  json.dump({"pairs": matrix, "null_mean": float(null.mean()), "null_95": cut95,
             "within_gelatin": within_gel, "within_paam": within_paam, "across": across},
            open(records.HERE / "universal_delta.json", "w"), indent=1)


if __name__ == "__main__":
  main()
