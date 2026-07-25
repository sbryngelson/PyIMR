"""Nonlinear-memory constitutive models (Giesekus, Phan-Thien-Tanner) for IMR,
as a pure ODE system on a Lagrangian radial grid.

WHY THIS IS AN ODE SOLVER
-------------------------
Giesekus and PTT are often described as requiring PDEs in the surrounding medium
(e.g. Warnez & Johnsen 2015), which is why IMRv2's ODE path (`f_stress.m`) stops
at `stress=5`. That framing is Eulerian. What these models actually lack is
*analytic closure* of the stress integral: the Giesekus quadratic term and the
PTT trace function mean no finite set of moments closes, unlike UCM/Oldroyd-B
which collapses to two variables (Z1, Z2).

With a known velocity field the constitutive law contains only the MATERIAL
derivative, so in a Lagrangian frame each material point obeys an ODE in time
with no spatial derivatives and no coupling between points:

    L = diag(-2a, a, a),  a = R^2 Rdot / r^3          (spherical, incompressible)
    tau + lam*tau^(1) + (nonlinear term) = 2*etaP*D
    tau^(1)_ii = d(tau_ii)/dt - 2*L_ii*tau_ii

Giesekus (mobility alpha):     nonlinear term = (alpha*lam/etaP) tau.tau
PTT linear (parameter eps):    f(tr tau) = 1 + (eps*lam/etaP) tr(tau) multiplies tau

Both reduce to UCM/Oldroyd-B when alpha -> 0 / eps -> 0, which is the validation
reference (UCM itself is validated against IMRv2).

The stress integral is evaluated by quadrature over the Lagrangian grid each step:
    S_polymer = int_R^inf (2/r)(tau_rr - tau_thth) dr

COST: 2*NM extra ODE states. NM=480 gives ~3e-3 agreement with UCM at alpha=0;
this is far slower than the closed-form models in imr_fast.py -- use those when
the model admits closure.
"""
import numpy as np
from scipy.integrate import solve_ivp
from imr_fast import params, KAPPA


def _grid(NM, xmax=60.0):
    """Lagrangian material points, labelled by reference radius r0/R0 >= 1,
    clustered at the bubble wall where the stress field is sharpest."""
    u = np.linspace(0.0, 1.0, NM)
    return 1.0 + (xmax - 1.0) * u ** 4


def simulate_nonlinear(tv, R0, Req, G, mu, De, LAM, model='giesekus',
                       alpha=0.0, eps=0.0, NM=480, radial=1,
                       rtol=1e-7, atol=1e-9):
    """R(t)/R0 for Giesekus or linear-PTT media.

    De   Deborah number (lam1*Uc/R0)
    LAM  retardation ratio lam2/lam1 (solvent fraction of the viscosity)
    model 'giesekus' (uses alpha) or 'ptt' (uses eps); alpha=eps=0 -> UCM/Oldroyd-B
    NM   number of Lagrangian material points
    """
    p = params(R0, Req, G, mu)
    Deb, Re8 = De, p['Re8']
    etaP = (1.0 - LAM) / Re8                      # polymer viscosity (nondim)
    x0 = _grid(NM)
    ia = alpha / etaP if etaP > 0 else 0.0
    ie = eps / etaP if etaP > 0 else 0.0
    giesekus = (model == 'giesekus')

    def rhs(_tn, y):
        R = max(y[0], 1e-8)
        V = y[1]
        trr = y[2:2 + NM]
        tth = y[2 + NM:2 + 2 * NM]
        r = np.cbrt(np.maximum(x0 ** 3 + R ** 3 - 1.0, 1e-30))
        a = V * R ** 2 / r ** 3
        if giesekus:
            dtrr = -trr / Deb - 4 * a * trr - ia * trr ** 2 - 4 * etaP * a / Deb
            dtth = -tth / Deb + 2 * a * tth - ia * tth ** 2 + 2 * etaP * a / Deb
        else:                                      # linear PTT
            f = 1.0 + ie * (trr + 2.0 * tth)
            dtrr = -f * trr / Deb - 4 * a * trr - 4 * etaP * a / Deb
            dtth = -f * tth / Deb + 2 * a * tth + 2 * etaP * a / Deb
        S = np.trapezoid(2.0 * (trr - tth) / r, r) - 4.0 * LAM / Re8 * V / R
        P = p['Pb'] * R ** (-3 * KAPPA)
        if radial == 1:
            Rdd = (P - 1 - p['iWe'] / R + S - 1.5 * V ** 2) / R
        else:
            raise ValueError("only radial=1 supported for nonlinear-memory models")
        return np.concatenate(([V, Rdd], dtrr, dtth))

    tn = np.asarray(tv) / p['t0']
    y0 = np.zeros(2 + 2 * NM)
    y0[0] = 1.0
    so = solve_ivp(rhs, (tn[0], tn[-1]), y0, t_eval=tn, method='LSODA',
                   rtol=rtol, atol=atol)
    if not so.success or so.y.shape[1] != len(tn):
        return np.full(len(tn), np.nan)
    return so.y[0]
