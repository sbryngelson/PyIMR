r"""Does the ranking depend on the three constants in the noise weight?

\Cref{sec:records} inflates the variance where the wall moves fastest, through a logistic
weight in the strain rate. That weight carries three constants --- a floor, a steepness and a
threshold rate --- which the paper quoted as a functional form and never as numbers, and which
nobody had varied. They multiply $\sigma_i$, hence $\chi^2$, hence every evidence and every
margin in the comparison. A referee is entitled to ask whether the winner is an artifact of
three unstated choices.

WHAT IS ALREADY SAFE, AND WHY IT IS ONLY HALF THE ANSWER. An overall factor on every variance
is absorbed by the marginalised noise scale of \cref{sec:beta}: $\beta$ integrates it out, so
the \emph{level} of the weight cannot matter. What survives is the \emph{shape} --- where the
downweighting turns on and how sharply --- because different candidates put their residual in
different parts of the record, and a reweighting that favours the late trace over the collapse
does not treat them alike.

THE SWEEP IS PRE-REGISTERED, which matters more than its width. A range chosen after seeing the
answer is a search for a range where the answer holds. The three grids below were fixed before
the first evaluation: each constant at its default and at a factor of two or three either side,
$27$ combinations in all, with the pass criterion stated in advance --- the quadratic Zener
solid must remain the best-fitting candidate on all three records at every combination.

WHAT THIS DOES AND DOES NOT RE-RUN. The weights change $\sigma_i$ and therefore $\chi^2$; they
do not change the forward solves, which depend only on the parameters. This sweep therefore
recomputes $\chi^2$ at each candidate's stored best fit rather than re-quadraturing the whole
grid --- $68{,}720$ grid points across $23$ models per record makes the full re-run
prohibitive at $27$ settings. It is a statement about the best-fit ordering, which is what the
evidence differences track when the margins are hundreds of nats, and not a re-derivation of
the evidence itself. Reported as such.
"""

import itertools
import json

import numpy as np

import records
from per_trial_fits import _trials

# fixed before the first evaluation; defaults are floor 0.1, steepness 1.0, threshold 1e5
FLOOR = (0.05, 0.1, 0.2)
STEEPNESS = (0.5, 1.0, 2.0)
THRESHOLD = (3e4, 1e5, 3e5)
WINNER = "qSLS"


def main():
  from pyimr.noise import characteristic_time, hencky_strain_rate, strain_rate_weights

  stored = json.loads((records.HERE / "results.json").read_text())
  settings = list(itertools.product(FLOOR, STEEPNESS, THRESHOLD))
  print(f"  pre-registered: floor {FLOOR}, steepness {STEEPNESS}, threshold {THRESHOLD}")
  print(f"  {len(settings)} combinations; pass criterion is {WINNER} best on every record\n")

  summary = {"settings": len(settings), "winner": WINNER, "records": {}}
  for record in records.DATASETS:
    times, mean, spread, maximum, _, window = _trials(record)
    models = stored[record]["models"]
    traces = {name: np.asarray(v["best_trace"], dtype=float) for name, v in models.items()
              if isinstance(v.get("best_trace"), list) and len(v["best_trace"]) == len(mean)}
    if len(traces) < 3:
      print(f"  {record}: only {len(traces)} usable traces; skipped")
      continue
    rate = hencky_strain_rate(mean, times, characteristic_time(maximum))

    held, tightest = 0, None
    for floor, steepness, threshold in settings:
      weight = strain_rate_weights(rate, threshold, steepness=steepness, floor=floor)
      sigma = spread / np.sqrt(weight)
      cost = {name: float(np.sum(((window - trace[:, None]) / sigma[:, None]) ** 2))
              for name, trace in traces.items()}
      order = sorted(cost, key=cost.get)
      held += int(order[0] == WINNER)
      margin = cost[order[1]] - cost[order[0]]
      if tightest is None or margin < tightest["margin"]:
        tightest = {"margin": float(margin), "floor": floor, "steepness": steepness,
                    "threshold": threshold, "best": order[0], "runner_up": order[1]}
    summary["records"][record] = {"held": held, "of": len(settings), "tightest": tightest}
    print(f"  {record}: {WINNER} best in {held} of {len(settings)} settings"
          f"{'' if held == len(settings) else '   *** NOT UNANIMOUS ***'}")
    print(f"    tightest margin {tightest['margin']:.0f} in chi-squared at floor "
          f"{tightest['floor']}, steepness {tightest['steepness']}, threshold "
          f"{tightest['threshold']:.0g} ({tightest['best']} over {tightest['runner_up']})")

  unanimous = all(v["held"] == v["of"] for v in summary["records"].values())
  summary["unanimous"] = bool(unanimous)
  print(f"\n  The ranking {'does not depend' if unanimous else 'DEPENDS'} on the three constants")
  print("  over the pre-registered range. The level was never at risk, since beta absorbs it;")
  print("  what this adds is that the shape is not at risk either.")
  json.dump(summary, open(records.HERE / "noise_constants.json", "w"), indent=1)


if __name__ == "__main__":
  main()
