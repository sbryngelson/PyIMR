r"""The afterbounce sweep read as a frequency sweep, and which channel carries $\rho$.

`bounce_identify.py` finds the amplitude ratios $96$ to \SI{99}{\percent} degenerate in
$\mu$ against $\rho$, and the reason is structural rather than numerical. For a damped
oscillator $A_{k+1}/A_k = e^{-\pi/Q}$, so the ratio measures $Q$ and nothing else, and
linearising the bubble about equilibrium gives
%
\begin{equation}
\omega_0 \sim \frac{\sqrt{K}}{\sqrt{\rho}\,R_0}, \qquad
Q \sim \frac{\sqrt{\rho K}\,R_0}{4\mu},
\label{eq:oscillator}
\end{equation}
%
with $K$ an effective stiffness carrying the gas and the shear modulus together. In logs
$\partial\log Q$ is $(-1, +\tfrac12)$ in $(\log\mu, \log\rho)$: one scalar, fixed opposite
exponents, cosine $-1$ by construction. The measured column ratios are $-1.3$ to $-1.9$
against the predicted $-2$, with the sign right on $17$ of $18$ bounces, so the mechanism is
the one \eqref{eq:oscillator} names.

THE SECOND NUMBER IS THE ONE THAT WAS DISCARDED. $\omega_0$ carries $\rho$ with the OPPOSITE
exponent and carries $\mu$ not at all at leading order, so the $2\times2$ in $(\mu,\rho)$ is
non-singular for the PAIR where it is singular for either alone. `bounce_sweep.py` computes
the periods and then reports only the ratios, on the grounds that a period carries the
event's clock. It does, and that is exactly where $\rho$ lives.

WHICH MAKES THE CLOCK THE PRICE. There are three families here and they are not
interchangeable. Amplitude ratios $A_{k+1}/A_k$ are clock-free and measure $Q$. Absolute
periods $T_k$ carry $\rho$ and are corrupted by any timebase error, which is the systematic
this package already suspects at $5$--$9\%$. Period RATIOS $T_{k+1}/T_k$ are clock-free
again, and are identically one for a linear oscillator, so they are a pure measure of how
the frequency moves as the amplitude decays. This asks which combinations condition the
problem and at what cost in clock exposure, rather than assuming the answer.

WHAT WOULD MAKE THE READING SPURIOUS. \Cref{eq:oscillator} is a linearisation and these
amplitudes are not small, so an amplitude-dependent frequency is expected from the
nonlinearity alone and would imitate a relaxation spectrum. The model is a single-relaxation
qSLS, so running the identical extraction on the model IS the sham: whatever spectral
breadth appears in the model's own $(\omega_k, Q_k)$ is manufactured by the analysis, and
only the measured-minus-model difference is a statement about the material.
"""

import json

import numpy as np

import records
from bounce_sweep import DATA, FILES, MAX_RATIO, RATIO, sequence

AXES = ("mu", "galpha", "lambda1", "rho")
STEP = 0.02
BASE_RHO = 1064.0
SPAN = 5.0                       # both axes normalised to a common known window
KEEP = 5


def _families(amps, times):
  """`(amplitude ratios, periods, period ratios)`, the three observables and their clocks."""
  ratios = [amps[k + 1] / amps[k] for k in range(len(amps) - 1)]
  periods = [times[k + 1] - times[k] for k in range(len(times) - 1)]
  quotients = [periods[k + 1] / periods[k] for k in range(len(periods) - 1)]
  return ratios, periods, quotients


def measured(dataset):
  """Per trial, then aggregated, with the between-trial spread each family earns."""
  table = np.loadtxt(DATA / FILES[dataset], delimiter=",", ndmin=2)
  tau, trials = table[:, 0], table[:, 1:].T
  keep = ~(trials > MAX_RATIO).any(axis=0) & (trials.std(axis=0, ddof=1) > 0.0)
  tau, trials = tau[keep], trials[:, keep]
  phase = (tau - tau[0]) / (tau[-1] - tau[0]) * SPAN
  got = []
  for row in trials:
    s = sequence(row, phase)
    if s is not None: got.append(_families(s["amplitudes"], s["times"]))
  n = min(len(g[2]) for g in got)                      # quotients are the shortest family
  n = min(n, KEEP)
  out = {"events": len(got)}
  for j, name in enumerate(("ratio", "period", "quotient")):
    stack = np.array([g[j][:n] for g in got], dtype=float)
    out[name] = stack.mean(axis=0).tolist()
    out[name + "_spread"] = stack.std(axis=0, ddof=1).tolist()
  return out


def _model(dataset, scales, n):
  import pyimr

  times, _, _, maximum, stretch = records.load(dataset)
  m = json.load(open(records.HERE / "per_trial_fits.json"))[dataset]["median"]
  product = m["galpha"] * scales.get("galpha", 1.0)
  material = pyimr.QuadraticZener(float(np.sqrt(product * RATIO)),
                                  m["mu"] * scales.get("mu", 1.0),
                                  m["lambda1"] * scales.get("lambda1", 1.0), 0.0,
                                  float(np.sqrt(product / RATIO)))
  physics = pyimr.PhysicalParameters(medium_density_kg_m3=BASE_RHO * scales.get("rho", 1.0))
  fine = np.linspace(times[0], times[-1], 20001)
  config = pyimr.SimulationConfig(maximum, maximum / stretch, material,
                                  dynamics="keller-miksis", rtol=1e-9, atol=1e-11,
                                  max_steps=2_000_000, physics=physics)
  trace = np.asarray(pyimr.simulate(fine, config).radius_ratio, dtype=float)
  phase = (fine - fine[0]) / (fine[-1] - fine[0]) * SPAN
  s = sequence(trace, phase)
  if s is None: return None
  fam = _families(s["amplitudes"], s["times"])
  return None if len(fam[2]) < n else tuple(np.array(f[:n], dtype=float) for f in fam)


def one(dataset):
  obs = measured(dataset)
  n = len(obs["quotient"])
  base = _model(dataset, {}, n)
  if base is None: return dataset, None

  weights = {name: np.array(obs[name + "_spread"], dtype=float)
             for name in ("ratio", "period", "quotient")}
  cols = {name: [] for name in ("ratio", "period", "quotient")}
  for axis in AXES:
    up = _model(dataset, {axis: 1 + STEP}, n)
    dn = _model(dataset, {axis: 1 - STEP}, n)
    if up is None or dn is None: return dataset, None
    for j, name in enumerate(("ratio", "period", "quotient")):
      cols[name].append((up[j] - dn[j]) / (2 * STEP) / weights[name])
  return dataset, {
    "jacobians": {k: np.column_stack(v).tolist() for k, v in cols.items()},
    "measured": obs,
    "model": {name: base[j].tolist()
              for j, name in enumerate(("ratio", "period", "quotient"))}}


def _diagnose(jac):
  """Condition number, the mu-rho cosine, and the weakest direction of a stacked Jacobian."""
  norms = np.linalg.norm(jac, axis=0)
  live = norms > 0
  cos = float("nan")
  if live[0] and live[3]:
    cos = float(jac[:, 0] @ jac[:, 3] / (norms[0] * norms[3]))
  values, vectors = np.linalg.eigh(jac.T @ jac)
  return {"cos_mu_rho": cos,
          "condition": float(values[-1] / max(values[0], 1e-300)),
          "null": [float(v) for v in vectors[:, 0]]}


COMBOS = (("ratios only (clock-free)", ("ratio",)),
          ("periods only (clock-exposed)", ("period",)),
          ("period ratios only (clock-free)", ("quotient",)),
          ("ratios + period ratios (CLOCK-FREE)", ("ratio", "quotient")),
          ("ratios + periods (clock-exposed)", ("ratio", "period")),
          ("all three", ("ratio", "period", "quotient")))


def main():
  print("  A_{k+1}/A_k measures Q, in which mu and rho have fixed opposite exponents.")
  print("  The period measures omega, in which rho's exponent flips sign and mu drops out.\n")

  with records.pool(len(records.DATASETS)) as pool:
    got = dict(pool.map(one, list(records.DATASETS)))

  summary = {}
  for dataset in records.DATASETS:
    v = got[dataset]
    if v is None:
      print(f"  {dataset}: not enough resolved bounces"); continue
    jacs = {k: np.array(a, dtype=float) for k, a in v["jacobians"].items()}
    print(f"  ==== {dataset}, {v['measured']['events']} events ====\n")
    print(f"  {'observable':>36s} {'cos(mu,rho)':>12s} {'condition':>12s}")
    rows = {}
    for label, names in COMBOS:
      d = _diagnose(np.vstack([jacs[k] for k in names]))
      rows[label] = d
      flag = "  <-- degenerate" if abs(d["cos_mu_rho"]) > 0.9 else ""
      print(f"  {label:>36s} {d['cos_mu_rho']:12.3f} {d['condition']:12.3g}{flag}")
    # the frequency sweep itself, measured against the single-relaxation model that is its sham
    print(f"\n  {'bounce':>10s} " + " ".join(f"{k+1:>9d}" for k in range(len(v['model']['ratio']))))
    for name in ("ratio", "period", "quotient"):
      print(f"  {name+' meas':>10s} " + " ".join(f"{x:9.3f}" for x in v["measured"][name]))
      print(f"  {name+' model':>10s} " + " ".join(f"{x:9.3f}" for x in v["model"][name]))
    summary[dataset] = {"combinations": rows, "measured": v["measured"], "model": v["model"]}
    print()

  print("  ---- what it says ----\n")
  for dataset, s in summary.items():
    a = s["combinations"]["ratios only (clock-free)"]["cos_mu_rho"]
    b = s["combinations"]["ratios + period ratios (CLOCK-FREE)"]["cos_mu_rho"]
    c = s["combinations"]["ratios + periods (clock-exposed)"]["cos_mu_rho"]
    print(f"  {dataset:>13s}: cos(mu,rho) {a:+.3f} alone, {b:+.3f} adding period ratios, "
          f"{c:+.3f} adding periods")
  print("\n  If only the clock-EXPOSED combination breaks the degeneracy, then separating mu")
  print("  from rho costs exactly the assumption this package cannot make, and the honest")
  print("  statement is that rho must be measured rather than fitted.")
  json.dump(summary, open(records.HERE / "frequency_space.json", "w"), indent=1)


if __name__ == "__main__":
  main()
