r"""Calibrate the universality statistic at the level it is quoted.

\Cref{tab:universal} reports the MEDIAN |cos| over 15 across-material pairs, but compares it
against the 95th point of a SINGLE-PAIR surrogate null. The pairs share the same eight curves,
so they are dependent, and the null distribution of a median of dependent pairs is not the
null distribution of a pair. Here the full ensemble is drawn -- eight phase-randomised
surrogates at once -- and the same group statistics the table reports are recomputed per draw.

The single-pair comparison turns out CONSERVATIVE: the ensemble null's 95th point for the
across-material median is far below 0.508, and within-gelatin, which sec:universal hedges as
sitting nearly on the pair-level null, is p ~ 1e-3 at the level the table actually quotes.
"""

import json

import numpy as np

import records
from shape_error import seed_for

DRAWS = 4000
PAIR_CUT = 0.5082  # the single-pair phase-randomised 95th point tab:universal quotes


def _surrogate(rng, curve):
  spectrum = np.fft.rfft(curve)
  angles = rng.uniform(0.0, 2 * np.pi, spectrum.size)
  angles[0] = 0.0
  out = np.fft.irfft(np.abs(spectrum) * np.exp(1j * angles), n=curve.size)
  out = out - out.mean()
  return out / np.linalg.norm(out)


def main():
  stored = json.load(open(records.HERE / "universal_curve.json"))
  curves = {d: np.asarray(v, dtype=float) for d, v in stored["curves"].items()}
  curves = {d: c / np.linalg.norm(c) for d, c in curves.items()}
  gel = [d for d in curves if d in records.DATASETS]
  paam = [d for d in curves if d not in records.DATASETS]

  def stats(ensemble):
    across = [abs(ensemble[a] @ ensemble[b]) for a in gel for b in paam]
    within_gel = [abs(ensemble[a] @ ensemble[b]) for i, a in enumerate(gel) for b in gel[i + 1:]]
    within_paam = [abs(ensemble[a] @ ensemble[b])
                   for i, a in enumerate(paam) for b in paam[i + 1:]]
    return (float(np.median(across)), sum(1 for r in across if r > PAIR_CUT),
            float(np.median(within_gel)), float(np.median(within_paam)))

  rng = np.random.default_rng(seed_for("ensemble_null"))
  null = np.empty((DRAWS, 4))
  for draw in range(DRAWS):
    null[draw] = stats({d: _surrogate(rng, curves[d]) for d in curves})
  observed = stats(curves)

  out = {}
  print(f"  {DRAWS} ensembles of {len(curves)} surrogates; statistics at the level tab:universal"
        " quotes them\n")
  for k, label in enumerate(("across median (15 dependent pairs)", f"across pairs > {PAIR_CUT}",
                             "within-gelatin median (3 pairs)", "within-PAAm median (10 pairs)")):
    p = float(np.mean(null[:, k] >= observed[k]))
    out[label] = {"observed": observed[k], "null_mean": float(null[:, k].mean()),
                  "null_95": float(np.percentile(null[:, k], 95)),
                  "null_99": float(np.percentile(null[:, k], 99)), "p": p}
    print(f"  {label:36s} observed {observed[k]:6.3f}  null mean {out[label]['null_mean']:.3f}"
          f"  95th {out[label]['null_95']:.3f}  99th {out[label]['null_99']:.3f}"
          f"  p {p if p > 0 else 1.0 / DRAWS:.2}{'' if p > 0 else ' (bound)'}")

  print("\n  The pair-level 0.508 is the wrong bar for a median of fifteen dependent pairs,")
  print("  and it errs conservative: the correct 95th point is "
        f"{out['across median (15 dependent pairs)']['null_95']:.3f}.")
  json.dump(out, open(records.HERE / "ensemble_null.json", "w"), indent=1)


if __name__ == "__main__":
  main()
