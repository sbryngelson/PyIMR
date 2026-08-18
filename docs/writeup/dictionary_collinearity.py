r"""How separable are the dictionary's columns, and what does that do to its shares?

`delta_dictionary.py` reports how much of $\hat\delta$ each physical term explains, and
\cref{sec:screenres} reads those shares as attributions --- functions of $R$ alone score
\SIrange{29}{70}{\percent}, functions of $\dot R$ and $\ddot R$ below \SI{9}{\percent}. That
reading needs the columns to be distinguishable from each other, and two of the high scorers
are $2\sigma/R$ and $R/R_{\max}$, which are a quantity and its reciprocal along the same
trajectory. If they are nearly collinear after the sensitivity span is projected out, then the
share attaches to the PAIR and naming either one is an artifact of which was asked first.

So the Gram matrix of the cleaned dictionary is reported rather than assumed well conditioned:
its condition number, its worst pair, and a variance inflation factor per column. A VIF above
about ten is the usual signal that a coefficient is not separately estimable; here the shares
are single-column projections rather than regression coefficients, so the pairwise cosine is
the number that matters most and it is reported in full.
"""

import json

import numpy as np

import records
from delta_dictionary import ORDER, dictionary
from universal_delta import one as delta_one


def one(dataset):
  payload = delta_one(dataset)[1]
  span = np.array(payload["span"])
  basis, _ = np.linalg.qr(span.T)
  clean = {}
  for name, column in dictionary(dataset).items():
    residual = column - basis @ (basis.T @ column)
    norm = np.linalg.norm(residual)
    if norm > 1e-9: clean[name] = residual / norm
  names = sorted(clean)
  design = np.column_stack([clean[n] for n in names])
  gram = design.T @ design
  inverse = np.linalg.pinv(gram)
  return dataset, {
    "names": names,
    "condition": float(np.linalg.cond(gram)),
    "vif": {n: float(inverse[i, i]) for i, n in enumerate(names)},
    "gram": gram.tolist()}


def main():
  with records.pool(len(ORDER)) as pool:
    got = dict(pool.map(one, list(ORDER)))

  names = got[ORDER[0]]["names"]
  print("  worst pairwise |cos| between cleaned dictionary columns, per record\n")
  print(f"  {'record':>16s} {'cond(G)':>12s} {'worst pair':>46s} {'|cos|':>7s} {'max VIF':>9s}")
  worst_pairs, out = {}, {}
  for dataset in ORDER:
    gram = np.abs(np.array(got[dataset]["gram"]))
    np.fill_diagonal(gram, 0.0)
    i, j = np.unravel_index(np.argmax(gram), gram.shape)
    pair = f"{names[i]} / {names[j]}"
    worst_pairs[pair] = worst_pairs.get(pair, 0) + 1
    top = max(got[dataset]["vif"].items(), key=lambda kv: kv[1])
    out[dataset] = {"condition": got[dataset]["condition"], "worst_pair": pair,
                    "worst_cos": float(gram[i, j]), "vif": got[dataset]["vif"]}
    print(f"  {dataset:>16s} {got[dataset]['condition']:12.1f} {pair:>46s} "
          f"{gram[i, j]:7.3f} {top[1]:9.1f}")

  print("\n  how often each pair is the worst: "
        + ", ".join(f"{p} ({n})" for p, n in sorted(worst_pairs.items(), key=lambda kv: -kv[1])))

  print("\n  median VIF per column across records\n")
  for name in names:
    values = [got[d]["vif"][name] for d in ORDER]
    flag = "  <- not separately estimable" if np.median(values) > 10 else ""
    print(f"    {name:34s} {np.median(values):9.1f}{flag}")

  # the specific pair sec:screenres's reading depends on
  key = [n for n in names if "surface tension" in n or "trace" in n]
  if len(key) == 2:
    cosines = []
    for dataset in ORDER:
      gram = np.array(got[dataset]["gram"])
      a, b = names.index(key[0]), names.index(key[1])
      cosines.append(abs(gram[a, b]))
    print(f"\n  {key[0]} against {key[1]}: |cos| "
          f"{min(cosines):.3f} to {max(cosines):.3f}, median {np.median(cosines):.3f}")
    print("  These are the two the shares are read off, so their overlap is the number that")
    print("  decides whether either can be named separately.")

  json.dump(out, open(records.HERE / "dictionary_collinearity.json", "w"), indent=1)


if __name__ == "__main__":
  main()
