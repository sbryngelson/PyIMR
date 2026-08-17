r"""Is the discrepancy on the physics shelf for PAAm either? The screen, on both materials.

`enrichment_screen.py` scores each candidate enrichment by the share of the measured
discrepancy it could remove, and on gelatin every one of them leaves $75$ to \SI{85}{\percent}
unexplained. That is the second link in the chain, and like the first it was measured on three
records of one material.

The screen transfers to PAAm unchanged: it needs a fit, a Jacobian, and one solve per
candidate, all of which `records` now supplies for the PAAm records. The identified coordinate
$g\alpha$ and the stated ratio $38.5$ are carried across, which is legitimate because the ratio
is UNIDENTIFIED by construction -- `identified.py` establishes that the record cannot see it,
and tests two decades either side.

WHAT WOULD BREAK THE CLAIM. If some enrichment removes most of the PAAm discrepancy, then the
gelatin result is about gelatin's chemistry rather than about the model form, and the paper
should say so. If PAAm leaves the same three quarters unexplained, the shelf is empty for two
materials and the argument is about the method.
"""

import json


import records
from enrichment_screen import directions

ORDER = (*records.DATASETS, *records.PAAM)


def main():
  from pyimr.noise import enrichment_overlap

  with records.pool(len(ORDER)) as pool:
    got = list(pool.map(directions, list(ORDER)))
  screens = {d: enrichment_overlap(s.identifiable, j, c) for d, s, j, c in got}
  spans = {d: enrichment_overlap(s.identifiable, j, c, one_sided=False) for d, s, j, c in got}
  names = sorted({k for _, _, _, c in got for k in c})

  print("\n  share of the discrepancy each candidate could remove, at its own fit\n")
  print(f"  {'candidate':28s} " + " ".join(f"{d.replace('paam_', ''):>13s}" for d, *_ in got))
  for name in names:
    cells = []
    for dataset, *_ in got:
      value = screens[dataset].removable.get(name)
      if value is None: cells.append("            -")
      else: cells.append(f"{value:12.1%}" + ("*" if not screens[dataset].reachable[name] else " "))
    print(f"  {name:28s} " + " ".join(cells))
  print(f"  {'ALL TOGETHER (cone)':28s} "
        + " ".join(f"{screens[d].joint:13.1%}" for d, *_ in got))
  print(f"  {'ALL TOGETHER (span)':28s} "
        + " ".join(f"{spans[d].joint:13.1%}" for d, *_ in got))
  print("\n  * anti-aligned: no admissible size of it removes anything")

  print("\n  ---- what it says ----\n")
  for material, group in (("gelatin", records.DATASETS), ("PAAm", records.PAAM)):
    joints = [screens[d].joint for d in group if d in screens]
    if joints:
      print(f"  {material:>8s}: every candidate together leaves "
            f"{1 - max(joints):.0%} to {1 - min(joints):.0%} of the discrepancy unexplained")
  json.dump({d: {"removable": screens[d].removable, "reachable": screens[d].reachable,
                 "overlap": screens[d].overlap, "joint": screens[d].joint,
                 "joint_span": spans[d].joint, "identifiable": float(s.size)}
             for d, s, _, _ in got},
            open(records.HERE / "paam_enrichment.json", "w"), indent=1)


if __name__ == "__main__":
  main()
