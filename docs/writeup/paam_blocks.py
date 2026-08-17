r"""The PAAm afterbounces again, with the pooled temperature sweep taken apart first.

`paam_afterbounce.py` reads `PA05_temp_exp_data.csv` as one dataset of $117$ events and finds
its ensemble ratio sequence monotone throughout, which it reports as PAAm being better behaved
than gelatin. That reading has a hole in it, and it is the same hole `dip_coherence.py` opened
under the gelatin anomaly: averaging over events that do not share a condition dilutes any
feature that is not common to all of them.

The file is a POOLED temperature sweep. Its $117$ columns are three blocks of $30$, $57$ and
$30$ at \SI{10}{\celsius}, \SI{21}{\celsius} and \SI{33}{\celsius}, which is what the
per-temperature event counts state and what the collapse time confirms at $F = 14.1$ against a
permutation null of the same partition. Averaging a soft record with a stiff one is exactly the
operation that turns a dip into a slope, so the monotone verdict has to be re-taken per block
before it means anything.

Both readings then get the shift null, so a dip that appears once the blocks are separated is
still held to the standard that retracted the gelatin one.
"""

import json

import numpy as np

import records
import dip_coherence
from bounce_sweep import maxima
from dip_coherence import DRAWS, KEEP, _dip

# gelatin, read through `dip_coherence`'s own path rather than this one. The PAAm records come
# out of `records.trials`, which is new code on a new abscissa, and a sequence extractor that
# quietly disagreed with the old one would make every PAAm record look monotone for a reason
# that has nothing to do with PAAm. These three are where a dip is known to be.
CONTROL = {"gelatin 15C": "Ga_t15_exp_data.csv", "gelatin 23C": "Ga_t23_exp_data.csv",
           "gelatin 33C": "Ga_t33_exp_data.csv"}


def ratio_matrix(dataset):
  """Each event's afterbounce ratio sequence, from the record's own columns."""
  times, events = records.trials(dataset)
  tau = times / times[-1]                    # `maxima` needs an abscissa, not a unit
  seqs = []
  for row in events:
    _, amps, _ = maxima(row, tau, order=2)
    if len(amps) >= 5: seqs.append([amps[k + 1] / amps[k] for k in range(len(amps) - 1)])
  n = min(KEEP, min(len(s) for s in seqs))
  return np.array([s[:n] for s in seqs], dtype=float)


POOLED = "paam_PA05_temp_pooled"


def main():
  matrices = {d: ratio_matrix(d) for d in records.PAAM}
  # the pooled read, rebuilt from its own blocks so the two differ only by the averaging
  blocks = [matrices[d] for d in ("paam_PA05_10C", "paam_PA05_21C", "paam_PA05_33C")]
  width = min(m.shape[1] for m in blocks)
  matrices[POOLED] = np.vstack([m[:, :width] for m in blocks])
  matrices |= {k: dip_coherence.ratio_matrix(v) for k, v in CONTROL.items()}

  print("  ensemble afterbounce ratios, per record and for the pooled sweep\n")
  print(f"  {'record':>22s} {'n':>4s}  " + " ".join(f"{'b' + str(k + 1):>7s}" for k in range(KEEP)))
  summary = {}
  rng = np.random.default_rng(20260817)
  for label, matrix in matrices.items():
    events, n = matrix.shape
    mean = matrix.mean(axis=0)
    se = matrix.std(axis=0, ddof=1) / np.sqrt(events)
    dips = [k for k in range(1, n) if mean[k] < mean[k - 1]]
    observed = _dip(mean)
    null = np.empty(DRAWS)
    for d in range(DRAWS):
      shifts = rng.integers(0, n, size=events)
      null[d] = _dip(np.array([np.roll(matrix[i], int(shifts[i]))
                               for i in range(events)]).mean(axis=0))
    z = (observed - null.mean()) / null.std(ddof=1)
    summary[label] = {"events": events, "bounces": n, "ratios": mean.tolist(),
                      "se": se.tolist(), "dips": [d + 1 for d in dips],
                      "observed_dip": observed, "null_mean": float(null.mean()),
                      "null_sd": float(null.std(ddof=1)), "z": float(z),
                      "p": float((null >= observed).mean())}
    print(f"  {label:>22s} {events:4d}  " + " ".join(f"{v:7.4f}" for v in mean))
    print(f"  {'+- se':>22s}      " + " ".join(f"{v:7.4f}" for v in se))

  print("\n  and the shift null on each\n")
  print(f"  {'record':>22s} {'dips at':>12s} {'ensemble dip':>13s} {'null mean':>10s} "
        f"{'z':>7s} {'p':>7s}")
  for label, v in summary.items():
    print(f"  {label:>22s} {str(v['dips']) if v['dips'] else '  monotone':>12s} "
          f"{v['observed_dip']:13.4f} {v['null_mean']:10.4f} {v['z']:7.2f} {v['p']:7.4f}")

  print("\n  ---- the control ----\n")
  fired = [k for k in CONTROL if summary[k]["dips"]]
  print(f"  the same extractor finds dips on {len(fired)} of {len(CONTROL)} gelatin records "
        f"({', '.join(fired) if fired else 'none'}), so a monotone PAAm verdict is the data's")
  if not fired:
    print("  NOTHING BELOW IS INTERPRETABLE: the extractor cannot find the dips it is known to")
    print("  find, so every monotone reading here may be the extractor's rather than PAAm's")

  print("\n  ---- what it says ----\n")
  pooled, parts = summary[POOLED], [summary[d] for d in
                                    ("paam_PA05_10C", "paam_PA05_21C", "paam_PA05_33C")]
  print(f"  Pooled, the sweep {'dips at ' + str(pooled['dips']) if pooled['dips'] else 'is monotone'}"
        f" (p {pooled['p']:.3f}). Split, its blocks give "
        f"{[ (v['dips'] or 'monotone') for v in parts ]}.")
  if not pooled["dips"] and any(v["dips"] for v in parts):
    print("  So the monotone reading of the pooled file was an averaging artefact, and the")
    print("  same operation that manufactured the gelatin dip had been hiding a PAAm one.")
  elif not pooled["dips"] and not any(v["dips"] for v in parts):
    print("  The blocks are monotone too, so the pooled verdict was not an averaging artefact")
    print("  and PAAm afterbounces really are consistent with the spherical form where")
    print("  gelatin's are not. That is a difference between MATERIALS, not between rigs.")
  json.dump(summary, open(records.HERE / "paam_blocks.json", "w"), indent=1)


if __name__ == "__main__":
  main()
