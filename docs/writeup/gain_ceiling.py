r"""A ceiling on what any single observation at a design can be worth.

The expected gain averages a per-parameter quantity over the prior,
%
    U(d) = \int A_d(\theta) \pi_d(\theta) d\theta,
    \qquad A_d(\theta) = D_{KL}[\,L_d(\cdot\mid\theta) \,\|\, m_d\,],

so its supremum bounds it: $U(d) \le B(d) \equiv \sup_\theta A_d(\theta)$. The bound is free in
the sense that matters -- it needs no new solves, only the Jacobian the design work already
computes -- and it says something the criterion does not. $U$ is what a design is worth on
average against the prior; $B$ is the most a single observation there could ever be worth, at
the most favourable parameter value in the box. A design whose $B$ is small cannot be rescued
by a luckier $\theta$.

For a Gaussian likelihood with a linearised forward model everything collapses to $p \times p$.
Writing $F = J^{\mathsf T}\Sigma^{-1}J$ for the whitened Fisher matrix and $\Sigma_\theta = s^2 I$
for the prior in the unit coordinates the design box already uses,
%
    A_d(\theta) = \tfrac{1}{2}\left[
        \log\det(I + s^2 F) - \operatorname{tr}\!\left(s^2 F (I + s^2 F)^{-1}\right)
        + \delta^{\mathsf T} F (I + s^2 F)^{-1} \delta \right],

with $\delta = \theta - \bar\theta$. The first two terms are constant in $\theta$; the third is a
positive semidefinite quadratic, so its supremum over a box sits at a vertex and costs $2^{p-1}$
evaluations. Averaging the quadratic against the prior returns
$\operatorname{tr}(s^2F(I+s^2F)^{-1})$ exactly, which cancels the second term and recovers
$U(d) = \tfrac12\log\det(I + s^2F)$ -- printed as a check, since a ceiling that does not sit
above the quantity it bounds is arithmetic rather than analysis.

MEASURED, IT IS VALID AND WEAK, and both halves are worth stating. The ceiling holds on all
$227$ candidates and $B/U$ runs from $1.51$ to $2.21$ with median $1.71$, so no design's value
is concentrated at the edge of the prior -- the average is a fair summary everywhere, which is
itself reassuring about batches chosen on $U$ alone.

But as a screen it barely bites. A candidate can be discarded without any assumption about
$\theta$ when its ceiling falls below the best design's average, and only $7$ of $227$ qualify.
The reason is visible in the spread: $U$ ranges over $3.83$ to $9.03$ across the whole design
space, a factor of $2.4$, while the ceiling carries $1.7$ of slack. Slack comparable to the
spread leaves nothing to eliminate. Worse for screening, $B/U$ correlates at $-0.874$ with $U$,
so the bound is tightest on the designs that need it least.

It would earn its place on a design space whose candidates differ more, or where the criterion
were expensive and the ceiling cheap. Here they cost the same Jacobian, so it is a check rather
than a filter -- and as a check it is worth having, since a ceiling below the quantity it bounds
would expose an error in either.

Idea taken from T. Chen's surrogate-BOED note, where $B(d)$ appears as the constant in an
EIG error bound. It is useful here on its own, without the surrogate.
"""

import json

import numpy as np

import records
from design_operator import PERFORMED, RADII, STRETCH, information

PRIOR = 1.0 / np.sqrt(12.0)      # a unit-box prior's standard deviation, per axis


def ceiling(matrix, scale=PRIOR):
  """`(U, B)` in nats for one design's whitened Fisher matrix."""
  size = matrix.shape[0]
  scaled = scale**2 * matrix
  identity = np.eye(size)
  average = float(np.linalg.slogdet(identity + scaled)[1])
  shrunk = np.linalg.solve(identity + scaled, matrix)     # F (I + s^2 F)^-1
  shrunk = 0.5 * (shrunk + shrunk.T)
  spent = float(np.trace(scale**2 * matrix @ np.linalg.inv(identity + scaled)))
  # a positive semidefinite quadratic over a box is maximised at a vertex; the sign of the
  # first coordinate is free because the form is even, so half the vertices suffice
  half = 0.5
  best = 0.0
  for mask in range(2 ** (size - 1)):
    signs = np.array([1.0] + [1.0 if (mask >> k) & 1 else -1.0 for k in range(size - 1)])
    corner = half * signs
    best = max(best, float(corner @ shrunk @ corner))
  return 0.5 * average, 0.5 * (average - spent + best)


def main():
  designs = [(float(r), float(s)) for r in RADII for s in STRETCH] + list(PERFORMED.values())
  with records.pool(len(designs)) as pool:
    got = list(pool.map(information, designs))
  usable = [(d, m) for d, m in got if m is not None]
  print(f"\n  {len(usable)} of {len(designs)} candidates integrate\n")

  table = {}
  for design, matrix in usable:
    gain, bound = ceiling(matrix)
    table[f"{design[0]:.6e}|{design[1]:.6e}"] = {"U": gain, "B": bound,
                                                 "radius_m": design[0], "stretch": design[1]}
  values = list(table.values())
  gains = np.array([v["U"] for v in values])
  bounds = np.array([v["B"] for v in values])
  assert np.all(bounds >= gains - 1e-9), "a ceiling below the quantity it bounds is a bug"

  order = np.argsort(-gains)
  print(f"  {'rank':>4s} {'R_max (um)':>11s} {'stretch':>8s} {'U (nats)':>10s} "
        f"{'B (nats)':>10s} {'B/U':>7s}")
  for rank, index in enumerate(order[:8], start=1):
    v = values[index]
    print(f"  {rank:4d} {v['radius_m'] * 1e6:11.1f} {v['stretch']:8.2f} {v['U']:10.3f} "
          f"{v['B']:10.3f} {v['B'] / v['U']:7.2f}")

  print("\n  the ceiling is loosest where the gain is smallest:")
  for rank, index in enumerate(order[-4:], start=len(order) - 3):
    v = values[index]
    print(f"  {rank:4d} {v['radius_m'] * 1e6:11.1f} {v['stretch']:8.2f} {v['U']:10.3f} "
          f"{v['B']:10.3f} {v['B'] / v['U']:7.2f}")

  ratio = bounds / gains
  print(f"\n  B/U over all candidates: median {np.median(ratio):.2f}, "
        f"range {ratio.min():.2f} to {ratio.max():.2f}")
  print(f"  the design maximising B is {'the same as' if order[0] == int(np.argmax(bounds)) else 'NOT'}"
        " the one maximising U")
  best_b = int(np.argmax(bounds))
  print(f"    argmax U: R_max {values[order[0]]['radius_m'] * 1e6:.1f} um at stretch "
        f"{values[order[0]]['stretch']:.2f}")
  print(f"    argmax B: R_max {values[best_b]['radius_m'] * 1e6:.1f} um at stretch "
        f"{values[best_b]['stretch']:.2f}")

  json.dump(table, open(records.HERE / "gain_ceiling.json", "w"), indent=1)


if __name__ == "__main__":
  main()
