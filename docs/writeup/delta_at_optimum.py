r"""Does the universal curve rest on fits that satisfy its own orthogonality condition?

$\hat\delta = (I-P)d$ is orthogonal to the sensitivity span by construction, so it is defined
wherever it is evaluated. That is not the same as being MEANINGFUL wherever it is evaluated.
`pyimr.noise.discrepancy` says so in as many words: at a converged interior fit the normal
equations force $Pd = 0$ exactly, and when they do not --- because the search stopped early or
stopped against a bound --- $\hat\delta$ carries the distance to the optimum along with the
model error, and the bias bound of \cref{sec:bias} does not apply.

The package exposes that as `at_optimum`, comparing $\lVert Pd\rVert / \lVert d\rVert$ against a
tolerance. Two scripts here consult it. The one that produces the eight $\hat\delta$ behind
\cref{sec:universal} does not, and neither does anything downstream of it, so the condition the
central result depends on has never been printed. This prints it.

It matters more than a normal convergence check because of WHAT would leak. Every record is
fitted with the same model over the same box, so a residual left over from stopping in the same
place would be a component shared by all eight --- which is exactly the signature the
universality test is looking for. A common $Pd$ would therefore manufacture a common curve.
"""

import json

import numpy as np

import records
from enrichment_screen import directions

ORDER = (*records.DATASETS, *records.PAAM)


def one(dataset):
  # `directions` returns (name, split, jacobian, candidates); `split.absorbed` is ||P d|| and
  # `split.size` is ||delta_hat||, so the total residual norm is recovered from the two.
  _, split, jacobian, _ = directions(dataset)
  identifiable = np.asarray(split.identifiable, dtype=float)
  taken, size = float(split.absorbed), float(split.size)
  total = float(np.hypot(taken, size))
  return dataset, {"at_optimum": bool(split.at_optimum),
                   "leak": taken / (total + 1e-30),
                   "absorbed_norm": taken, "identifiable_norm": size,
                   "summary": str(split)[:150]}


def main():
  with records.pool(len(ORDER)) as pool:
    got = dict(pool.map(one, list(ORDER)))

  print("  is each fit at an interior optimum, as delta_hat's construction assumes?\n")
  print(f"  {'record':>16s} {'at_optimum':>11s} {'||Pd||/||d||':>13s} {'||Pd||':>10s}"
        f" {'||delta_hat||':>14s}")
  for dataset in ORDER:
    v = got[dataset]
    print(f"  {dataset:>16s} {str(v['at_optimum']):>11s} {v['leak']:13.4f}"
          f" {v['absorbed_norm']:10.2f} {v['identifiable_norm']:14.2f}")

  failed = [d for d in ORDER if not got[d]["at_optimum"]]
  print(f"\n  fits failing the condition: {len(failed)} of {len(ORDER)}"
        + (f" ({', '.join(failed)})" if failed else ""))

  worst = max(got[d]["leak"] for d in ORDER)
  print(f"  largest ||Pd||/||d|| over the eight: {worst:.4f}, against a tolerance of 0.01")
  print("\n  ---- what it says ----\n")
  if failed:
    print("  The universal curve is computed from fits that do NOT satisfy the orthogonality")
    print("  its own construction assumes. Every record shares a model and a box, so a residual")
    print("  left over from stopping in the same place is a COMMON component, which is exactly")
    print("  the signature sec:universal measures. This has to be resolved before the claim.")
  else:
    print("  Every fit is at an interior optimum to within the package's own tolerance, so the")
    print("  eight delta_hat are model error rather than distance-to-optimum, and the shared")
    print("  curve cannot be a shared failure to converge.")
  json.dump(got, open(records.HERE / "delta_at_optimum.json", "w"), indent=1)


if __name__ == "__main__":
  main()
