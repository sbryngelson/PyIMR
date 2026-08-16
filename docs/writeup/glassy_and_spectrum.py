r"""Two rheological candidates for the collapse-time deficit, tested at forward-solve cost.

\Cref{sec:limitations} records that these datasets want a collapse $5$--$9\%$ slower than
gelatin's density gives, and that \texttt{RHO}$=1064$ has been absorbing it. Resolved on a fine
grid the wall reaches $\sim10^{8}\,\mathrm{s^{-1}}$ at the collapse, so the fitted arm runs at a
Weissenberg number of $6$ to $14$ there --- unrelaxed --- while sitting below $0.1$ over
\SI{99.8}{\percent} of the record. Essentially all the constitutive information in a trace comes
from that sliver, and two rheological readings of what the model is missing there both predict a
collapse that is too fast.

FIRST, THE INSTANTANEOUS MODULUS. A Zener arm shows $G_\infty = \mu/\lambda_1$ when it cannot
relax. The fits give $300$, $544$ and \SI{520}{\kilo\pascal}, which is $271$ to $1101$ times the
equilibrium $g\alpha$ and is therefore not a soft model in the naive sense. It is still far below
where a hydrated biopolymer should sit at \SI{10}{\nano\second}, which is the glassy plateau,
order \SI{1}{\giga\pascal}. A material that stiffens to a gigapascal at the collapse resists it;
one that stops at half a megapascal does not, and the fit would have to buy the missing
resistance somewhere --- an inflated density being exactly the kind of somewhere available.

The test needs no new constitutive law. $G_\infty$ is a ratio of two fitted numbers, so scanning
$(\mu, \lambda_1)$ moves it directly, and the question is whether the contour that delivers the
missing $5$--$9\%$ sits at a physically sensible glassy modulus or at an absurd one.

SECOND, THE WIDTH OF THE SPECTRUM. Gelatin is unequal chains debonding stochastically
(\cref{sec:limitations}), which is a broad continuous spectrum rather than a mode. A trace sweeps
three decades of rate in microseconds, so a single relaxation time can be right at one rate only.
\Cref{sec:enrich} tested a second mode and found it landed $15.7$ times the first --- barely more
than a decade, and free to place itself where it liked. The question here is different: pinned
DECADES apart so the pair actually spans the rate range, does a broad pair beat a narrow one?

WHAT THEY FOUND, SO THE NEXT READER DOES NOT REPEAT THEM. Both candidates are refuted, and the
first is refuted in a way worth keeping. The collapse timing does not depend on $\lambda_1$ AT
ALL: across each row below $G_\infty$ moves by a factor of sixteen and the shift does not change
in the second decimal. It depends only on $\mu$. So the deficit is not a high-rate elastic
stiffness the model lacks, it is dissipation, and about four times the fitted viscosity closes
it. Width fares no better --- a second mode a thousandfold slower at a share of $0.3$ buys
\SI{1.94}{\percent} against the $5$--$9$ wanted, and faster second modes buy nothing measurable.

That $\lambda_1$ does not reach the collapse timing is also why it is the loosest coordinate in
this package, at a between-trial spread of $51$ to \SI{117}{\percent}: the feature carrying most
of the information is one it cannot move.

WHAT NEITHER TEST CAN DO. \texttt{TwoModeQuadraticZener} splits one elastic target between two
arms rather than adding an independent stiff one, so the second part is a test of \emph{width},
not of a glassy mode. The first part reaches the glassy question only through $\mu/\lambda_1$,
which also moves the low-rate dissipation, so its answer is a scale rather than a fit. Both are
screens: they say whether a candidate is worth building a constitutive law for.
"""

import json

import numpy as np

import records

DATASET, RATIO = "gelatin_15C", 38.5
SPAN, SAMPLES = 5.0, 20001            # resolved: the 201-sample record hides the collapse
TARGET = (0.05, 0.09)                 # the deficit sec:limitations measures
MU_SCALES = (1.0, 2.0, 4.0, 8.0, 16.0)
LAM_SCALES = (1.0, 0.5, 0.25, 0.125, 0.0625)
# the paper's own fitted second mode sat at 15.7x the first; these span the rate range instead
SECOND_RATIOS = (15.7, 100.0, 1000.0, 0.01, 0.001)
SHARES = (0.05, 0.15, 0.30)


def _fit():
  return json.load(open(records.HERE / "per_trial_fits.json"))[DATASET]["median"]


def collapse(material):
  """`(phase of the first minimum, depth)` on a grid fine enough to resolve them."""
  import pyimr

  times, _, _, maximum, stretch = records.load(DATASET)
  fine = np.linspace(times[0], times[-1], SAMPLES)
  config = pyimr.SimulationConfig(maximum, maximum / stretch, material,
                                  dynamics="keller-miksis", rtol=1e-9, atol=1e-11,
                                  max_steps=2_000_000)
  try:
    trace = np.asarray(pyimr.simulate(fine, config).radius_ratio, dtype=float)
  except Exception:                                                          # noqa: BLE001
    return None, None
  if not np.all(np.isfinite(trace)): return None, None
  i = next((j for j in range(20, len(trace) - 1)
            if trace[j] <= trace[j - 1] and trace[j] < trace[j + 1]), int(np.argmin(trace)))
  return (fine[i] - fine[0]) / (fine[-1] - fine[0]) * SPAN, float(trace[i])


def one_glassy(job):
  import pyimr

  mu_scale, lam_scale = job
  m = _fit()
  product = m["galpha"]
  material = pyimr.QuadraticZener(float(np.sqrt(product * RATIO)), m["mu"] * mu_scale,
                                  m["lambda1"] * lam_scale, 0.0, float(np.sqrt(product / RATIO)))
  phase, depth = collapse(material)
  return job, {"phase": phase, "depth": depth,
               "g_inst": m["mu"] * mu_scale / (m["lambda1"] * lam_scale)}


def one_spectrum(job):
  import pyimr

  ratio, share = job
  m = _fit()
  product = m["galpha"]
  material = pyimr.TwoModeQuadraticZener(
    float(np.sqrt(product * RATIO)), m["mu"], m["lambda1"], 0.0,
    float(np.sqrt(product / RATIO)), m["lambda1"] * ratio, share)
  phase, depth = collapse(material)
  return job, {"phase": phase, "depth": depth}


def main():
  m = _fit()
  base = pyimr_material(m)
  base_phase, base_depth = collapse(base)
  print(f"  baseline: first minimum at {base_phase:.4f} t_c, depth {base_depth:.5f}, "
        f"G_inst = {m['mu']/m['lambda1']:.3e} Pa")
  print(f"  the deficit to close is {100*TARGET[0]:.0f} to {100*TARGET[1]:.0f} percent LATER\n")

  print("  ==== A: does raising the instantaneous modulus slow the collapse enough? ====\n")
  jobs = [(a, b) for a in MU_SCALES for b in LAM_SCALES]
  with records.pool(len(jobs)) as pool:
    got = dict(pool.map(one_glassy, jobs))
  print("  shift in the first-minimum phase, percent, against G_inst\n")
  print(f"  {'mu x':>6s} " + " ".join(f"{f'lam x{b:g}':>12s}" for b in LAM_SCALES))
  reached = []
  summary = {"baseline_phase": base_phase, "baseline_g_inst": m["mu"] / m["lambda1"], "glassy": {}}
  for a in MU_SCALES:
    cells = []
    for b in LAM_SCALES:
      v = got[(a, b)]
      if v["phase"] is None: cells.append(f"{'--':>12s}"); continue
      shift = v["phase"] / base_phase - 1.0
      summary["glassy"][f"{a:g}|{b:g}"] = {"shift": shift, "g_inst": v["g_inst"],
                                           "depth": v["depth"]}
      if TARGET[0] <= shift <= TARGET[1]: reached.append((a, b, v["g_inst"], shift))
      cells.append(f"{100*shift:+7.2f}% {v['g_inst']:.0e}".rjust(12))
    print(f"  {a:6g} " + " ".join(cells))
  print()
  if reached:
    gs = [g for *_, g, _ in reached]
    summary["glassy_window_pa"] = [float(min(gs)), float(max(gs))]
    print(f"  the deficit is reached at G_inst between {min(gs):.2e} and {max(gs):.2e} Pa")
    print("  a hydrated biopolymer's glassy plateau is order 1e9 Pa")
  else:
    summary["glassy_window_pa"] = None
    print("  no combination in this range delivers the deficit")

  print("\n  ==== B: does a spectrum spanning decades beat one spanning 15.7x? ====\n")
  jobs2 = [(r, w) for r in SECOND_RATIOS for w in SHARES]
  with records.pool(len(jobs2)) as pool:
    got2 = dict(pool.map(one_spectrum, jobs2))
  print(f"  {'lam2/lam1':>10s} " + " ".join(f"{f'share {w:g}':>13s}" for w in SHARES))
  summary["spectrum"] = {}
  for r in SECOND_RATIOS:
    cells = []
    for w in SHARES:
      v = got2[(r, w)]
      if v["phase"] is None: cells.append(f"{'--':>13s}"); continue
      shift = v["phase"] / base_phase - 1.0
      summary["spectrum"][f"{r:g}|{w:g}"] = {"shift": shift, "depth": v["depth"]}
      cells.append(f"{100*shift:+8.2f}%".rjust(13))
    print(f"  {r:10g} " + " ".join(cells))
  print("\n  Shift in the first-minimum phase. The paper's own fitted second mode sat at 15.7x.")

  print("\n  ---- what it says ----\n")
  print("  A: if the deficit needs a G_inst near a real glassy plateau, the missing physics is")
  print("     high-rate stiffening and a constitutive law for it is worth building. If it needs")
  print("     an absurd one, the deficit is not the glassy modulus.")
  print("  B: if width alone moves the collapse, a broad spectrum is worth implementing; if it")
  print("     barely moves at any spacing, sec:enrich's negative result extends to the class.")
  json.dump(summary, open(records.HERE / "glassy_and_spectrum.json", "w"), indent=1)


def pyimr_material(m):
  import pyimr

  product = m["galpha"]
  return pyimr.QuadraticZener(float(np.sqrt(product * RATIO)), m["mu"], m["lambda1"], 0.0,
                              float(np.sqrt(product / RATIO)))


if __name__ == "__main__":
  main()
