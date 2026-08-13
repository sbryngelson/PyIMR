r"""What the pipeline's clock already removes, and what aligning on the measured one removes next.

\Cref{sec:latent} finds the dominant trial mode aligned with time dilation at $\lvert\cos\rvert$
of $0.95$ to $0.98$ in the acquisition data, and at $0.52$ to $0.58$ in the records this
document fits --- with the implied timescale scatter falling from the $22$--$28\%$ measured per
event to $1$--$2\%$. A factor of fifteen is removed somewhere, and until now nobody here knew
where.

THE FILE SAYS WHERE, EXACTLY. Reading its own fields against each other,
%
    R_norm = R(t) / R_max        to the last bit, and
    d(t_norm)/dt = 10 / R_max ,

so the stored clock is $t\,/\,(R_{\max}/\SI{10}{\metre\per\second})$, and
$\sqrt{p_\infty/\rho} = \SI{10.07}{\metre\per\second}$. The pipeline is dividing each event's
time by \emph{that event's own} inertial timescale $R_{\max}\sqrt{\rho/p_\infty}$. The
normalisation is per-event and correct, which is why the $R_{\max}$ scatter does not reach us.

SO WHAT IS THE RESIDUE? Not $R_{\max}$: that is divided out. What remains is that the
\emph{actual} collapse takes a time which is not exactly the inertial estimate --- it depends on
the stretch and on the material, both of which vary between events. The file stores that too, as
\texttt{tc\_norm}, the measured collapse time in units of the inertial one. If the residual
dilation is that quantity, its scatter should match the $1$--$2\%$ the traces imply, and
realigning each event on its own \texttt{tc\_norm} should remove what the inertial clock left
behind.

THAT IS THE TEST, and it is a clean one because it is a prediction with a number attached rather
than a search. Three clocks are compared on the same events: absolute time, the inertial clock
the pipeline uses, and the measured collapse time. What the mode does across those three says
whether the remaining quarter of \#222 is a clock effect the processing missed or something that
no choice of clock reaches.
"""

import json
import pathlib

import numpy as np

import records

SOURCE = pathlib.Path("~/data_pa5_tempsweeps_master_20260210.mat").expanduser()
BINS = {"12-18C": (12.0, 18.0), "20-26C": (20.0, 26.0), "30-36C": (30.0, 36.0)}
GRID = np.linspace(0.15, 3.0, 200)          # in units of each clock, past the first collapse
RAYLEIGH = 10.0                              # m/s, the constant the file's clock uses


def _scalar(entry, field):
  try:
    return float(getattr(entry, field))
  except Exception:                                                          # noqa: BLE001
    return float("nan")


def _cos(a, b):
  return abs(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))


def clocks(entry, population):
  """`{name: (clock, trace)}` for one event, under each of the three alignments.

  `population` is the bin's mean `R_max`, so the first clock is the one a pipeline would use
  if it normalised by a POPULATION value rather than each event's own -- which is exactly what
  sec:reqprior shows this document doing with the stretch, and the reason to price it here.
  """
  radius = np.asarray(getattr(entry, "R_norm"), dtype=float)
  inertial = np.asarray(getattr(entry, "t_norm"), dtype=float)
  absolute = np.asarray(getattr(entry, "t"), dtype=float)
  collapse = _scalar(entry, "tc_norm")
  good = np.isfinite(radius) & np.isfinite(inertial) & np.isfinite(absolute)
  if good.sum() < 60 or not np.isfinite(collapse) or collapse <= 0: return None
  radius, inertial, absolute = radius[good], inertial[good], absolute[good]
  # every clock is measured from the maximum, so each starts where its own trace does
  start = inertial.min()
  size = _scalar(entry, "R0")
  if not np.isfinite(size) or size <= 0: return None
  # the population clock: the same t_norm, but rescaled back to the BIN's mean size, so every
  # event is measured against one shared inertial time instead of its own
  return {
    "population": ((inertial - start) * size / population, radius),
    "inertial": (inertial - start, radius),
    "collapse": ((inertial - start) / collapse, radius),
  }


def mode_of(stack, grid):
  """Leading deviation mode of a set of resampled traces, and its overlap with dilation."""
  matrix = np.column_stack(stack)
  deviation = matrix - matrix.mean(axis=1, keepdims=True)
  left, values, _ = np.linalg.svd(deviation, full_matrices=False)
  mean = matrix.mean(axis=1)
  dilation = -grid * np.gradient(mean, grid)
  unit = dilation / np.linalg.norm(dilation)
  fraction = (unit @ deviation) / np.linalg.norm(dilation)
  fitted = np.outer(dilation, fraction)
  return {
    "events": matrix.shape[1],
    "share": float(values[0] ** 2 / (values**2).sum()),
    "cos_dilation": _cos(left[:, 0], dilation),
    "variance_explained": 1.0 - float(np.sum((deviation - fitted) ** 2) / np.sum(deviation**2)),
    "implied_cv": float(fraction.std(ddof=1)),
    "total_variance": float(np.sum(deviation**2) / deviation.shape[1]),
  }


def main():
  import scipy.io as sio

  if not SOURCE.exists():
    print(f"  {SOURCE} not present; nothing to do")
    return
  loaded = sio.loadmat(SOURCE, squeeze_me=True, struct_as_record=False)
  entries = loaded["dataPa5TempSweep"]
  temperature = np.array([_scalar(e, "T") for e in entries])
  flag = np.array([_scalar(e, "errorFlag") for e in entries])
  collapse = np.array([_scalar(e, "tc_norm") for e in entries])
  radius = np.array([_scalar(e, "R0") for e in entries])
  usable = (flag == 0) & np.isfinite(temperature) & np.isfinite(collapse) & (collapse > 0)

  print("\n  the measured collapse time, in units of the inertial one the clock uses\n")
  print(f"  {'bin':>8s} {'n':>4s} {'tc_norm':>16s} {'cv':>7s} {'R_max cv':>9s}")
  summary = {"tc_norm": {}}
  for name, (low, high) in BINS.items():
    pick = usable & (temperature >= low) & (temperature < high)
    if pick.sum() < 4: continue
    value, size = collapse[pick], radius[pick]
    summary["tc_norm"][name] = {"mean": float(value.mean()),
                                "cv": float(value.std(ddof=1) / value.mean()),
                                "r_max_cv": float(size.std(ddof=1) / size.mean())}
    print(f"  {name:>8s} {pick.sum():4d} {value.mean():9.4f} +- {value.std(ddof=1):.4f} "
          f"{summary['tc_norm'][name]['cv']:7.4f} {summary['tc_norm'][name]['r_max_cv']:9.3f}")
  print("  Compare the last two columns: the pipeline's clock converts a 22-28% scatter in the")
  print("  SIZE into a 1-2% scatter in what is left of the TIME. That is the factor of fifteen.")

  print("\n  the leading mode under each clock\n")
  print(f"  {'bin':>8s} {'clock':>10s} {'n':>4s} {'share':>7s} {'|cos| dilation':>15s} "
        f"{'var expl':>9s} {'implied cv':>11s} {'total var':>11s}")
  table = {}
  for name, (low, high) in BINS.items():
    chosen = [e for e, ok, t in zip(entries, usable, temperature, strict=True)
              if ok and low <= t < high]
    if len(chosen) < 8: continue
    population = float(np.mean([_scalar(e, "R0") for e in chosen]))
    built = {k: [] for k in ("population", "inertial", "collapse")}
    for entry in chosen:
      got = clocks(entry, population)
      if got is None: continue
      for key, (clock, trace) in got.items():
        order = np.argsort(clock)
        built[key].append(np.interp(GRID, clock[order], trace[order]))
    row = {}
    for key in ("population", "inertial", "collapse"):
      if len(built[key]) < 8: continue
      row[key] = mode_of(built[key], GRID)
      got = row[key]
      print(f"  {name:>8s} {key:>10s} {got['events']:4d} {got['share']:7.3f} "
            f"{got['cos_dilation']:15.3f} {got['variance_explained']:9.3f} "
            f"{got['implied_cv']:11.4f} {got['total_variance']:11.3e}")
    table[name] = row
    print()
  summary["modes"] = table

  print("  Reading it: 'population' normalises every event by the bin's MEAN size, which is what")
  print("  a pipeline does when it applies a population value to each trial; 'inertial' is the")
  print("  per-event clock this file actually uses and what reaches this document; 'collapse'")
  print("  realigns each event on its own MEASURED collapse time.\n")

  print(f"  {'bin':>8s} {'population -> inertial':>26s} {'inertial -> collapse':>24s}")
  for name, row in table.items():
    if not {"population", "inertial", "collapse"} <= set(row): continue
    first = row["inertial"]["total_variance"] / row["population"]["total_variance"]
    second = row["collapse"]["total_variance"] / row["inertial"]["total_variance"]
    print(f"  {name:>8s} {'total variance x' + format(first, '.3f'):>26s} "
          f"{'x' + format(second, '.3f'):>24s}")
    summary.setdefault("reductions", {})[name] = {"per_event_clock": first,
                                                  "measured_collapse": second}

  json.dump(summary, open(records.HERE / "realign.json", "w"), indent=1)


if __name__ == "__main__":
  main()
