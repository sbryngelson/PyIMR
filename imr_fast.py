"""Fast Python IMR solver -- a validated slice of IMRv2.

Covers:
  radial   1 (Rayleigh-Plesset), 2 (Keller-Miksis, pressure form)
  bubtherm 0 (polytropic, kappa)     masstrans/vapor 0
  stress   1 (neo-Hookean Kelvin-Voigt)
           2 (quadratic Kelvin-Voigt, strain-stiffening, alphax)
           3 (linear Maxwell / Jeffreys / Zener, 1 internal variable)
           5 (UCM / Oldroyd-B, 2 internal variables)
  forcing  none (free collapse)

Equations transcribed from IMRv2/src/{f_radial_eq,f_stress,f_call_params}.m.
Validated against IMRv2 reference trajectories -- see tests/run_validation.py.
NOT valid outside the above (Gilmore, thermal PDE, mass transfer, vapor,
acoustic forcing, PTT, Giesekus). Re-validate before extending.
"""
import numpy as np
from scipy.integrate import solve_ivp

P8 = 101325.0      # far-field pressure (Pa)
RHO = 1064.0       # far-field density (kg/m^3)
SURF = 0.07        # surface tension (N/m)
KAPPA = 1.4        # polytropic exponent
C8 = 1484.0        # far-field sound speed (m/s)


def params(R0, Req, G, mu, lam1=0.0, lam2=0.0, alphax=0.25):
    """Nondimensional groups, matching f_call_params.m."""
    Uc = np.sqrt(P8 / RHO)
    t0 = R0 / Uc
    Ca = P8 / G if G > 0 else np.inf
    Re8 = P8 * R0 / (mu * Uc)
    We = P8 * R0 / (2 * SURF)
    P0 = (P8 + 2 * SURF / Req) * (Req / R0) ** (3 * KAPPA)
    return dict(t0=t0, Ca=Ca, Re8=Re8, iWe=1.0 / We, req=Req / R0, Pb=P0 / P8,
                De=(lam1 * Uc / R0 if lam1 > 0 else 0.0),
                LAM=(lam2 / lam1 if lam1 > 0 else 0.0),
                Cstar=C8 / Uc, alphax=alphax)


def _stress(stress, p, R, Rd, Z):
    """Return (S, Sdot, dZdt) matching f_stress.m."""
    Rst = p['req'] / R
    Ca, Re8, De, LAM, ax = p['Ca'], p['Re8'], p['De'], p['LAM'], p['alphax']
    if stress == 1:                                     # neo-Hookean KV
        S = -(5 - 4 * Rst - Rst ** 4) / (2 * Ca) - 4.0 / Re8 * Rd / R
        Sdot = -2 * Rd / R * (Rst + Rst ** 4) / Ca + 4.0 / Re8 * (Rd / R) ** 2
        return S, Sdot, None
    if stress == 2:                                     # quadratic KV
        S = ((3 * ax - 1) * (5 - Rst ** 4 - 4 * Rst) / (2 * Ca)
             - 4.0 / Re8 * Rd / R
             + (2 * ax / Ca) * (27 / 40 + Rst ** 8 / 8 + Rst ** 5 / 5 + Rst ** 2 - 2 / Rst))
        Sdot = ((Rd / R) * ((3 * ax - 1) / (2 * Ca)) * (4 * Rst ** 4 + 4 * Rst)
                + 4 * (Rd / R) ** 2 / Re8
                - 2 * ax / Ca * Rd / R * (Rst ** 8 + Rst ** 5 + 2 * Rst ** 2 + 2 / Rst))
        return S, Sdot, None
    if stress == 3:                                     # Zener / Jeffreys / Maxwell
        Z1 = Z[0]
        S = Z1 / R ** 3 - 4 * LAM / Re8 * Rd / R
        Ze = -0.5 * (R ** 3 / Ca) * (5 - Rst ** 4 - 4 * Rst)
        Z1d = -(Z1 - Ze) / De + 4 * (LAM - 1) / (Re8 * De) * R ** 2 * Rd
        Sdot = Z1d / R ** 3 - 3 * Rd / R ** 4 * Z1 + 4 * LAM / Re8 * (Rd / R) ** 2
        return S, Sdot, np.array([Z1d])
    if stress == 5:                                     # UCM / Oldroyd-B
        Z1, Z2 = Z[0], Z[1]
        Z1d = -(1 / De - 2 * Rd / R) * Z1 + 2 * (LAM - 1) / (Re8 * De) * R ** 2 * Rd
        Z2d = -(1 / De + Rd / R) * Z2 + 2 * (LAM - 1) / (Re8 * De) * R ** 2 * Rd
        S = (Z1 + Z2) / R ** 3 - 4 * LAM / Re8 * Rd / R
        Sdot = ((Z1d + Z2d) / R ** 3 - 3 * Rd / R ** 4 * (Z1 + Z2)
                + 4 * LAM / Re8 * Rd ** 2 / R ** 2)
        return S, Sdot, np.array([Z1d, Z2d])
    raise ValueError(f"stress={stress} not supported")


def _nZ(stress):
    """Number of internal stress variables."""
    return {1: 0, 2: 0, 3: 1, 5: 2}[stress]


def _JdotA(stress, p):
    """f_call_params.m: 4/Re8 for stress 1-4, 4*LAM/Re8 for stress 5-6."""
    return 4.0 / p['Re8'] if stress in (1, 2, 3, 4) else 4.0 * p['LAM'] / p['Re8']


def _rhs(_t, y, p, stress, radial):
    R = max(y[0], 1e-8)
    Rd = y[1]
    Z = y[2:] if _nZ(stress) else None
    S, Sdot, dZ = _stress(stress, p, R, Rd, Z)
    P = p['Pb'] * R ** (-3 * KAPPA)
    iWe = p['iWe']
    if radial == 1:                                     # Rayleigh-Plesset
        Rdd = (P - 1 - iWe / R + S - 1.5 * Rd ** 2) / R
    elif radial == 2:                                   # Keller-Miksis (pressure form)
        Cs = p['Cstar']
        Pdot = -3 * KAPPA * p['Pb'] * R ** (-3 * KAPPA - 1) * Rd
        num = ((1 + Rd / Cs) * (P - 1 - iWe / R + S)
               + R / Cs * (Pdot + iWe * Rd / R ** 2 + Sdot)
               - 1.5 * (1 - Rd / (3 * Cs)) * Rd ** 2)
        den = (1 - Rd / Cs) * R + _JdotA(stress, p) / Cs
        Rdd = num / den
    else:
        raise ValueError(f"radial={radial} not supported")
    out = [Rd, Rdd]
    if dZ is not None:
        out.extend(dZ.tolist())
    return out


def simulate(tv, R0, Req, G, mu, stress=1, lam1=0.0, lam2=0.0, alphax=0.25,
             radial=1, rtol=1e-8, atol=1e-10):
    """Bubble radius R(t)/R0 at times tv (seconds). Returns NaN array on failure."""
    p = params(R0, Req, G, mu, lam1, lam2, alphax)
    tn = np.asarray(tv) / p['t0']
    y0 = [1.0, 0.0] + [0.0] * _nZ(stress)
    s = solve_ivp(_rhs, (tn[0], tn[-1]), y0, t_eval=tn, args=(p, stress, radial),
                  method='LSODA', rtol=rtol, atol=atol)
    if not s.success or s.y.shape[1] != len(tn):
        return np.full(len(tn), np.nan)
    return s.y[0]
