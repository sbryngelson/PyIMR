r"""The operator margins under the likelihood the residuals have, not under a scalar deflation.

\Cref{sec:operator} reports the operator ranking at face value. \Cref{sec:screening} deflates
every margin by $N/N_{\rm eff} \approx 20$ and finds five or six of six operators live. The
conclusions then call compressibility ``the most robust statement in this work'' at $74$ to
$179$ nats --- and $74/20.1 = 3.7$, below this document's own five-nat threshold. Applying the
deflation to the tabulated margins puts \emph{Rayleigh--Plesset} back among the live models at
\SI{23}{} and \SI{33}{\celsius}. Two of the paper's own statements cannot both stand.

THE SCALAR DEFLATION IS THE WRONG INSTRUMENT, which is why this recomputes rather than
rescales. $N/N_{\rm eff}$ is derived for a sum of correlated terms and applied as though every
model difference were the same shape. It is not: whitening by an exponential covariance charges
for a smooth separation and pays for a fast one, and \texttt{correlated\_design.py} measures the
operator differences landing on the amplified side at most candidate geometries. So the honest
number is neither the face value nor the face value over twenty --- it is the evidence computed
with $\Sigma$ in it.

TWO PASSES, because $\tau$ is not known in advance --- and ONE $\tau$ PER RECORD, shared by every
operator, which is the whole difficulty. A first attempt let each operator take $\tau$ from its
own residual, on the reasoning that the correlation is a property of what each model leaves
behind. That reasoning is wrong and the output said so: at \SI{23}{\celsius} the incompressible
model came out BEST by $62$ nats while at \SI{33}{\celsius} it lost by $10887$. Evidences
computed under different covariances are not a Bayes factor --- $\log|\Sigma|$ differs between
them, and a model allowed to choose its own noise model chooses one that flatters it, with
nothing charging it for the choice. So $\tau$ is estimated once, from the best independent fit
on that record, and held.

It is also SWEPT, because the answer turns out to depend on it: whitening by an exponential
covariance amplifies fine structure by roughly $1/(1-\rho^2)$, which is $16$ at $\rho = 0.97$,
so a comparison at one $\tau$ is a comparison at one arbitrary amplification.

The identified coordinates of \cref{sec:discrepancy} are used throughout so no fit rests against
a box wall, which is what made the natural-coordinate comparison a statement about the prior.

WHAT WOULD SETTLE IT EITHER WAY. If the correlated margins keep compressibility above five nats
on every record, the conclusion stands and the scalar deflation was too harsh. If they do not,
the conclusion has to be narrowed to the records that survive. Either is a result; quoting the
face value while a section twenty pages earlier deflates it is not.
"""

import json

import numpy as np

import records
from identified import BOX, candidate_at_ratio

RATIO = 38.5
STARTS, EVALUATIONS = 10, 260
OPERATORS = (("rayleigh-plesset", None), ("keller-miksis", None), ("keller-enthalpy", "tait"),
             ("gilmore", "tait"), ("keller-enthalpy", "mie-gruneisen"), ("gilmore", "mie-gruneisen"))
INFLATION = 201.0 / 10.0                 # the scalar the document applies, for comparison
DECISIVE_NATS = 5.0                      # the threshold sec:batch screens on
RHOS = (0.80, 0.90, 0.95)                # swept: the amplification 1/(1-rho^2) is steep


def _name(operator): return operator[0] if operator[1] is None else f"{operator[0]}/{operator[1]}"


def tau_from(dataset, operator, fitted):
  """`(tau, rho)` from one operator's residual on one record, under independence."""
  from pyimr.noise import correlation_time_from
  from pyimr.selection import evaluate_at

  times, mean, spread, maximum, stretch = records.load(dataset)
  solve = records.solver(times, maximum, stretch, dynamics=operator[0], liquid_eos=operator[1])
  model = np.asarray(evaluate_at(candidate_at_ratio(RATIO), solve, fitted)[0], dtype=float)
  tau, rho = correlation_time_from((model - mean) / spread, times)
  return (float(tau) if np.isfinite(tau) else None), float(rho)


def one(job):
  """Log evidence for one operator on one record, at a stated correlation time."""
  dataset, operator, tau = job
  times, mean, spread, maximum, stretch = records.load(dataset)
  candidate = candidate_at_ratio(RATIO)
  solve = records.solver(times, maximum, stretch, dynamics=operator[0], liquid_eos=operator[1])
  noise = {} if tau is None else {"correlation_time_s": float(tau), "times": times}
  try:
    got = records.score(candidate, solve, mean, spread, bounds=BOX, starts=STARTS,
                        evaluations=EVALUATIONS, trials=records.trial_count(dataset), **noise)
  except ValueError as error:
    return job, {"failed": str(error)}
  return job, {"log_evidence": got["log_evidence"], "chi2_per_n": got["chi2_per_n"],
               "fitted": got["fitted"]}


def _tau_for(dataset, rho):
  """The `tau` an AR(1) of lag-one `rho` has on this record's sampling."""
  from pyimr.noise import correlation_time_from                                # noqa: F401

  times = records.load(dataset)[0]
  return -float(np.median(np.diff(times))) / float(np.log(rho))


def main():
  # every independent fit first, IN THE POOL: the serial version of this took longer than the
  # sweep it exists to set up, and produced no output for half an hour while doing it
  first = [(d, o, None) for d in records.DATASETS for o in OPERATORS]
  print(f"  {len(first)} independent fits ...", flush=True)
  with records.pool(len(first)) as pool:
    plain = dict(pool.map(one, first))

  shared = {}
  for dataset in records.DATASETS:
    usable = {o: plain[(dataset, o, None)] for o in OPERATORS
              if "failed" not in plain[(dataset, o, None)]
              and np.isfinite(plain[(dataset, o, None)]["log_evidence"])}
    if not usable:
      shared[dataset] = {"tau_s": None, "rho": None, "from": None}
      continue
    best = max(usable, key=lambda o: usable[o]["log_evidence"])
    tau, rho = tau_from(dataset, best, usable[best]["fitted"])
    shared[dataset] = {"tau_s": tau, "rho": rho, "from": _name(best)}
    print(f"  {dataset:13s} rho {rho:.3f} from {_name(best)}  ->  tau {tau:.3e} s", flush=True)

  ladders = {d: [("independent", None)]
             + [(f"rho={r:g}", _tau_for(d, r)) for r in RHOS]
             + ([("fitted", shared[d]["tau_s"])] if shared[d]["tau_s"] else [])
             for d in records.DATASETS}

  rest = [(d, o, tau) for d in records.DATASETS for label, tau in ladders[d]
          if label != "independent" for o in OPERATORS]
  print(f"\n  {len(rest)} correlated fits ...", flush=True)
  with records.pool(len(rest)) as pool:
    table = dict(pool.map(one, rest))
  table |= plain

  summary = {}
  for dataset in records.DATASETS:
    print(f"\n=== {dataset} ===\n")
    header = " ".join(f"{label:>13s}" for label, _ in ladders[dataset])
    print(f"  {'operator':32s} {header}")
    columns = {}
    for label, tau in ladders[dataset]:
      values = {}
      for operator in OPERATORS:
        got = table[(dataset, operator, tau)]
        if "failed" in got or not np.isfinite(got["log_evidence"]): continue
        values[_name(operator)] = got["log_evidence"]
      best = max(values.values()) if values else float("nan")
      columns[label] = {k: v - best for k, v in values.items()}
    entry = {}
    for operator in OPERATORS:
      name = _name(operator)
      cells = " ".join(f"{columns[label].get(name, float('nan')):13.2f}"
                       for label, _ in ladders[dataset])
      print(f"  {name:32s} {cells}")
      entry[name] = {label: columns[label].get(name) for label, _ in ladders[dataset]}
    summary[dataset] = {"shared": shared[dataset], "margins": entry}

  print(f"\n  compressibility, against the {DECISIVE_NATS:g}-nat threshold sec:batch screens on\n")
  first_record = list(records.DATASETS)[0]
  header = " ".join(f"{label:>13s}" for label, _ in ladders[first_record])
  print(f"  {'record':13s} {header}")
  for dataset in records.DATASETS:
    row = summary[dataset]["margins"].get("rayleigh-plesset", {})
    cells = " ".join(
      f"{(row.get(label) if row.get(label) is not None else float('nan')):13.2f}"
      for label, _ in ladders[dataset])
    print(f"  {dataset:13s} {cells}")
  print("\n  A margin above -5 is a model this document's own screen calls LIVE. The first")
  print("  column is what sec:operator reports; the rest is what happens once the correlation")
  print("  the residuals actually carry goes into the likelihood rather than into a divisor.")

  json.dump(summary, open(records.HERE / "operator_correlated.json", "w"), indent=1)


if __name__ == "__main__":
  main()
