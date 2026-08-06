"""One matplotlib style for every figure in this document.

Two things make a figure look like it belongs in the page rather than pasted onto it, and
neither is about colour.

The first is that its type matches the body. The document is Computer Modern, so the
figures are too --- matplotlib ships `cmr10` and a matching maths set, which gets there
without the fragility of shelling out to LaTeX for every label.

The second is that it is drawn at the size it is printed. A figure authored at
\\SI{8.4}{in} and placed at `\\textwidth` is scaled by $6.5/8.4$, and its \\SI{9}{pt} labels
arrive on the page at \\SI{7}{pt}. Before this module the figures here were authored
between \\SI{3.5}{in} and \\SI{8.4}{in} and printed at fractions from $0.5$ to $1$, so label
sizes on the page ran from \\SI{6.3}{pt} to \\SI{8.4}{pt} --- a spread that is invisible
figure by figure and obvious once they are seen together. `size()` returns a figure size
already scaled by the fraction of the text width the figure will occupy, so nothing is
resized on the way in and every label lands at `FONT_SIZE`.
"""

import matplotlib as mpl

TEXT_WIDTH = 6.5           # \textwidth for letterpaper with 1in margins, in inches
FONT_SIZE = 8.0

# Colourblind-safe and print-safe: Wong's palette, ordered so the first two are the pair
# most often used for a contrast.
PALETTE = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000")


def size(fraction=1.0, height=2.6):
  """A figure size whose width is `fraction` of the text width, so it is printed unscaled."""
  return (TEXT_WIDTH * float(fraction), float(height))


def use():
  """Apply the document style. Call once, before any figure is created."""
  mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    # cmr10 has no U+2212, so matplotlib warns and drops minus signs on tick labels unless
    # the formatter is told to use mathtext and the unicode minus is disabled.
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
    "font.size": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE - 0.5,
    "ytick.labelsize": FONT_SIZE - 0.5,
    "legend.fontsize": FONT_SIZE - 1.0,
    "figure.titlesize": FONT_SIZE + 0.5,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.prop_cycle": mpl.cycler(color=list(PALETTE)),
    "lines.linewidth": 1.1,
    "lines.markersize": 3.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.8,
    "ytick.major.size": 2.8,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.25,
    "legend.frameon": False,
    "legend.handlelength": 1.6,
    "legend.borderaxespad": 0.4,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,       # embed as TrueType so the text stays selectable and searchable
  })
