r"""Designing under the likelihood the records actually have.

\Cref{sec:gain} closes with the deflation and does not apply it: ``the criterion, the
certificate and \eqref{eq:runs} are all statements about $M$, whatever $M$ was computed from'',
and then every batch in the chapter is computed from a diagonal $M$. That is not a caveat that
multiplies through. The measured deflations at $\rho = 0.908$ are

    constant offset  x0.053     smooth half-cycle  x0.049
    five cycles      x0.176     point-to-point jitter  x20.7 ,

so a correlated likelihood does not merely charge more for the same design --- it changes which
FEATURE of a difference is worth buying. Under independence a design is rewarded for making the
rival predictions far apart; under correlation it is rewarded for making them differ FAST,
because a smooth separation is most of what the noise already does.

That is a testable mechanism and not just a rescaling, so this measures three things. Whether
the geometry ranking moves at all. Whether the deflation a geometry suffers is predicted by the
high-frequency content of its own model difference, which is the mechanism above stated as a
correlation. And what the change does to \eqref{eq:runs}, which is where a collaborator feels
it: $s$ falls, and the budget rises by the same factor.

THE CORRELATION LENGTH TRANSFERS BY PHASE, like the noise profile of \cref{sec:noisedesign} and
for the same reason: $\tau$ measured on a record is a property of the collapse, not of the
clock, so $\tau/t_c$ is what carries. Transferring $\tau$ in absolute seconds would put the
correlation at a different point of the collapse at every candidate size, which is the error
that section already identifies for $\sigma$.
"""

import json

import numpy as np

import records
from noise_design import RELATIVE_NOISE, profile, raw_information



LAG_ONE = 0.918                    # sec:fitquality, the qSLS residual on the 15 C record
HIGH_PASS = 8                      # cycles over the record: above this counts as "fast"


def correlation_length(dataset):
  """`tau / t_c`: the AR(1) correlation time as a fraction of the collapse time."""
  from pyimr.noise import characteristic_time

  times, mean, spread, maximum, stretch = records.load(dataset)
  spacing = float(np.median(np.diff(times)))
  return (-spacing / np.log(LAG_ONE)) / characteristic_time(maximum)


def _gains(columns, scale, phase, tau_over_tc):
  """`(diagonal, correlated)` gain, and the discrimination variance under each."""
  from pyimr.noise import whitening

  out = {}
  for name, correlation in (("diagonal", None), ("correlated", tau_over_tc)):
    model = whitening(phase, scale, correlation_time=correlation)
    whitened = model.apply(columns)
    fisher = whitened.T @ whitened
    fisher = 0.5 * (fisher + fisher.T)
    prior = 1.0 / np.sqrt(12.0)
    posterior = np.eye(fisher.shape[0]) + prior**2 * fisher
    # the model column is last: its variance after the material is fitted out is the Schur
    # complement, which is what eq:runs calls s
    block = np.linalg.inv(posterior)
    out[name] = {"gain": 0.5 * float(np.linalg.slogdet(posterior)[1]),
                 "separation": float(1.0 / block[-1, -1] - 1.0)}
  return out


def fast_share(columns, phase):
  """Fraction of the model difference's energy above `HIGH_PASS` cycles across the record."""
  difference = columns[:, -1]
  spectrum = np.abs(np.fft.rfft(difference - difference.mean())) ** 2
  total = float(spectrum.sum())
  return float(spectrum[HIGH_PASS:].sum() / total) if total > 0 else 0.0


def main():
  from design_operator import PERFORMED, RADII, STRETCH

  designs = [(float(r), float(s)) for r in RADII for s in STRETCH] + list(PERFORMED.values())
  with records.pool(len(designs)) as pool:
    got = list(pool.map(raw_information, designs))
  usable = [(d, payload) for d, payload in got if payload is not None]
  print(f"  {len(usable)} of {len(designs)} candidates integrate")

  # all three profiles, not just one: sec:noisedesign shows 15 C is the record whose raw
  # profile keeps the constant's geometry, so testing on it alone asks the easiest question
  table, summary = {}, {}
  for source in records.DATASETS:
    tau_over_tc = correlation_length(source)
    tau, spread = profile(source)
    rows = []
    for design, (columns, phase) in usable:
      scale = np.interp(phase, tau, spread)
      entry = _gains(columns, scale, phase, tau_over_tc)
      entry["fast_share"] = fast_share(columns, phase)
      entry["flat"] = _gains(columns, np.full(columns.shape[0], RELATIVE_NOISE), phase,
                             tau_over_tc)
      entry["radius_m"], entry["stretch"] = design
      rows.append(entry)
    table[source] = rows
    summary[source] = {"tau_over_tc": tau_over_tc}

  print("\n  which geometry each likelihood wants, per noise profile\n")
  print(f"  {'profile':13s} {'tau/t_c':>8s} {'diagonal':>22s} {'correlated':>22s} {'moved':>7s}")
  for source, rows in table.items():
    picks = {}
    for name in ("diagonal", "correlated"):
      pick = max(rows, key=lambda r: r[name]["gain"])
      picks[name] = (pick["radius_m"], pick["stretch"], pick[name]["gain"])
    moved = picks["diagonal"][:2] != picks["correlated"][:2]
    summary[source] |= {"diagonal": picks["diagonal"], "correlated": picks["correlated"],
                        "moved": bool(moved)}
    print(f"  {source:13s} {summary[source]['tau_over_tc']:8.4f} "
          f"{picks['diagonal'][0] * 1e6:7.0f} um at {picks['diagonal'][1]:5.2f} "
          f"{picks['correlated'][0] * 1e6:12.0f} um at {picks['correlated'][1]:5.2f} "
          f"{'YES' if moved else 'no':>7s}")

  print("\n  is the change explained by how FAST each model difference is?\n")
  print(f"  {'profile':13s} {'median ratio':>13s} {'range':>22s} {'rank corr w/ fast share':>24s}")
  for source, rows in table.items():
    fast = np.array([r["fast_share"] for r in rows])
    ratio = np.array([r["correlated"]["separation"] / max(r["diagonal"]["separation"], 1e-300)
                      for r in rows])
    keep = np.isfinite(ratio) & (ratio > 0)
    order = float(np.corrcoef(np.argsort(np.argsort(fast[keep])),
                              np.argsort(np.argsort(ratio[keep])))[0, 1])
    summary[source] |= {"ratio_median": float(np.median(ratio[keep])),
                        "ratio_low": float(ratio[keep].min()),
                        "ratio_high": float(ratio[keep].max()), "fast_rank": order}
    print(f"  {source:13s} {np.median(ratio[keep]):13.4f} "
          f"{ratio[keep].min():9.4f} to {ratio[keep].max():<9.4f} {order:+24.3f}")
  print("  Positive is the mechanism: correlation charges for smooth separations and pays for")
  print("  fast ones, so a design whose difference has fine structure gains rather than loses.")

  print("\n  what it costs in bubbles, eq:runs at 5 nats and 95 percent\n")
  print(f"  {'profile':13s} {'noise model':>12s} {'s diagonal':>11s} {'s correlated':>13s} "
        f"{'N diag':>8s} {'N corr':>8s}")
  numerator = (1.645 + np.sqrt(1.645**2 + 2 * 5.0)) ** 2
  for source, rows in table.items():
    pick = max(rows, key=lambda r: r["diagonal"]["gain"])
    for label, cell in (("flat 0.018", pick["flat"]), ("measured", pick)):
      a, b = cell["diagonal"]["separation"], cell["correlated"]["separation"]
      runs = (numerator / a if a > 0 else float("inf"),
              numerator / b if b > 0 else float("inf"))
      summary[source] |= {f"runs_{label.split()[0]}": list(runs)}
      print(f"  {source:13s} {label:>12s} {a:11.4f} {b:13.4f} {runs[0]:8.2f} {runs[1]:8.2f}")
  print("  The flat scale is the one every batch in sec:gain is built on; the measured one is")
  print("  what sec:noisedesign replaces it with. They differ by more than the likelihood does.")

  json.dump(summary, open(records.HERE / "correlated_design.json", "w"), indent=1)


if __name__ == "__main__":
  main()
