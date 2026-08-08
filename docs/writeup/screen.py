"""Which model questions are still open, and therefore worth an experiment?

A design that separates models the records have already separated spends runs on a settled
matter. Before optimizing anything, ask what is actually still in question.

The answer bears directly on the design work. `identify.py` builds a batch to separate qSLS
from SLS across a plane of geometries; that is only worth doing if the pair is live. And the
complementary case matters as much: rivals far enough apart to be decided are also the easy
ones, so screening them out is what leaves a derivative-based criterion in the regime where it
is correct.

Three axes are screened separately because they are separate questions, and per record because
the material differs with temperature -- a constitutive form decided at one temperature is not
thereby decided at another. A model decided on ALL of them is decided.
"""

import json

import numpy as np

import records

DECISIVE = 5.0                                       # nats; the margin below which we stop asking
# Every evidence here was computed under an independent likelihood, and `check_residuals` puts
# N_eff near 10 of 201 -- so each of those margins is inflated by roughly N/N_eff. Screening
# is exactly where that matters: it turns a margin into a yes-or-no, and a 25-fold inflation
# moves margins across the threshold. Both readings are reported, because which one is right
# is the open item at the top of `open-work.md` and not something this study can settle.
INFLATION = 201.0 / 10.0


def _constitutive(dataset):
  record = json.loads(records.HERE.joinpath("results.json").read_text())[dataset]
  models = {k: v["evidence"] for k, v in record["models"].items()
            if np.isfinite(v.get("evidence", float("nan")))}
  return models


def _operators(dataset):
  """From `req_prior.py`, at the pinned width: the run that fits Req rather than asserting it
  is the one whose margins we believe, and its pinned column is what the rest of the document
  quotes."""
  path = records.HERE / "req_prior.json"
  if not path.exists(): return {}
  stored = json.loads(path.read_text())
  got = {}
  for key, row in stored.items():
    record, operator, width = key.split("|")
    if record != dataset or width != "None" or "failed" in row: continue
    got[operator] = row["log_evidence"]
  return got


def _thermal(dataset):
  path = records.HERE / "thermal.json"
  if not path.exists(): return {}
  stored = json.loads(path.read_text())
  return {k.split("|")[1]: v["log_evidence"] for k, v in stored.items()
          if k.startswith(f"{dataset}|") and "failed" not in v}


def _report(label, table, live_everywhere):
  from pyimr.discriminate import screen_models

  print(f"\n=== {label} ===")
  for dataset in records.DATASETS:
    models = table.get(dataset, {})
    if len(models) < 2:
      print(f"  {dataset}: fewer than two scored models, nothing to screen")
      continue
    names = list(models)
    values = np.array([models[n] for n in names], dtype=float)
    screen = screen_models(values, decisive=DECISIVE)
    live = [names[i] for i in screen.live]
    live_everywhere[label] = live_everywhere.get(label, set(names)) & set(live)
    print(f"  {dataset}: {len(live)} of {len(names)} live -- "
          + ", ".join(f"{n} ({screen.weights[names.index(n)]:.3f})" for n in live))
    if len(names) - len(live) > 0:
      worst = names[int(np.argmin(screen.margins))]
      print(f"      {len(names) - len(live)} decided, the furthest by "
            f"{screen.margins.min():.0f} nats ({worst})")
    # the same screen with the margins deflated by the residual correlation
    deflated = screen_models(values / INFLATION, decisive=DECISIVE)
    if deflated.live.size != screen.live.size:
      print(f"      deflated by N/N_eff = {INFLATION:.0f}: {deflated.live.size} live instead "
            f"of {screen.live.size} -- " + ", ".join(names[i] for i in deflated.live))


def main():
  live_everywhere = {}
  _report("constitutive", {d: _constitutive(d) for d in records.DATASETS}, live_everywhere)
  _report("bubble dynamics", {d: _operators(d) for d in records.DATASETS}, live_everywhere)
  _report("thermal treatment", {d: _thermal(d) for d in records.DATASETS}, live_everywhere)

  print("\n\n  still live on EVERY record -- the only questions a batch should target:")
  for label, names in live_everywhere.items():
    print(f"    {label:20} {', '.join(sorted(names)) if names else '(none: every rival decided somewhere)'}")
  print(f"\n  `decisive` is {DECISIVE:.0f} nats. A rival further behind than that on a record is")
  print("  settled there, and designing to separate it spends runs on a closed question.")
  print("  Note what this does NOT say: a live pair is one the data cannot yet separate, which")
  print("  is not the same as one the data supports. `lackoffit.py` rejects the winner itself.")
  records.HERE.joinpath("screen.json").write_text(json.dumps(
    {label: sorted(names) for label, names in live_everywhere.items()}, indent=1))


if __name__ == "__main__":
  main()
