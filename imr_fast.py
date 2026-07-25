"""Fast Python IMR solver -- a validated slice of IMRv2.

Covers:
  radial   1 (Rayleigh-Plesset), 2 (Keller-Miksis, pressure form)
  bubtherm 0 (polytropic, kappa) or 1 (gas thermal PDE, dry/vapor=0 only)
           medtherm 0     masstrans 0     vapor 0 or 1
  stress   1 (neo-Hookean Kelvin-Voigt)
           2 (quadratic Kelvin-Voigt, strain-stiffening, alphax)
           3 (linear Maxwell / Jeffreys / Zener, 1 internal variable)
           5 (UCM / Oldroyd-B, 2 internal variables)
  forcing  wave_type 0 (constant offset), 1 (Gaussian), 2 (histotripsy),
           3 (Heaviside step); pA = amplitude

Equations transcribed from IMRv2/src/{f_radial_eq,f_stress,f_call_params,
f_imr_fd}.m. Validated against IMRv2 reference trajectories -- see
tests/run_validation.py.

KAPPA=1.4 is a fixed module constant (not exposed as a per-call override),
matching what the reference trajectories were generated with; IMRv2's own
shipped default is 1.47 (default_case.m), overridable via 'kappa' in
varargin. Runs that need a different kappa are out of scope until this is
exposed as a parameter.

bubtherm=1 implements ONLY IMRv2's "elseif bubtherm" branch (f_imr_fd.m):
gas-phase thermal PDE, isothermal-wall-equivalent clamp (thetadot[-1]=0,
i.e. medtherm=0), dry gas (kv0=0, vapor=0). Its Pdot uses bare P (kappa*P),
NOT (P-Pv) -- this is IMRv2's actual equation for this branch, not a
simplification; the bubtherm=0 polytropic branch's Pdot uses (P-Pv) and the
two are NOT reconciled/harmonized, since they are genuinely different
equations in the source. NOT valid outside the above (medtherm, masstrans,
Gilmore, PTT, Giesekus). Re-validate before extending.
"""
import numpy as np
from scipy.integrate import solve_ivp
from thermal_fd import finite_diff_mat

P8 = 101325.0      # far-field pressure (Pa)
RHO = 1064.0       # far-field density (kg/m^3)
SURF = 0.07        # surface tension (N/m)
KAPPA = 1.4        # polytropic exponent (see module docstring)
C8 = 1484.0        # far-field sound speed (m/s)
KAPOVER = (KAPPA - 1.0) / KAPPA

# gas / vapor thermal-conductivity linear-in-T fit coefficients, IMRv2 defaults
# (default_case.m). K8 is IMRv2's reference conductivity: it mixes gas AND
# vapor coefficients even when vapor=0, because it is used purely as a
# normalization constant, not a physical mixture average at a given state.
_ATG, _BTG = 5.28e-5, 1.165e-2
_ATV, _BTV = 3.30e-5, 1.742e-2


def pvsat(T):
    """Saturation vapour pressure (Pa), f_pvsat.m."""
    return 1.17e11 * np.exp(-5200.0 / T)


def params(R0, Req, G, mu, lam1=0.0, lam2=0.0, alphax=0.25, vapor=0, T8=298.15,
           pA=0.0, omega=0.0, TW=0.0, DT=0.0, mn=0.0, wave_type=0, bubtherm=0):
    """Nondimensional groups, matching f_call_params.m.

    P0's exponent depends on bubtherm (f_call_params.m:158-163): 3*kappa
    (polytropic/adiabatic initial-state assumption) when bubtherm==0, but
    plain 3 (pure geometric volume relation) when bubtherm!=0 -- these are
    NOT the same formula, and using the wrong one silently gives a P(0)
    off by ~8.6x for these test parameters, not a subtle error.
    """
    Uc = np.sqrt(P8 / RHO)
    t0 = R0 / Uc
    Ca = P8 / G if G > 0 else np.inf
    Re8 = P8 * R0 / (mu * Uc)
    We = P8 * R0 / (2 * SURF)
    Pv = vapor * pvsat(T8)
    P0_exp = 3 if bubtherm else 3 * KAPPA
    P0 = (P8 + 2 * SURF / Req - Pv) * (Req / R0) ** P0_exp
    Pv_star = Pv / P8
    # thermal groups (f_call_params.m); cheap, computed unconditionally so
    # bubtherm=1 needs no separate params() path
    K8 = 0.5 * (_ATG * T8 + _BTG + _ATV * T8 + _BTV)
    chi = T8 * K8 / (P8 * R0 * Uc)
    alpha_g = _ATG * T8 / K8
    beta_g = _BTG / K8
    return dict(t0=t0, Ca=Ca, Re8=Re8, iWe=1.0 / We, req=Req / R0,
                Pb=P0 / P8 + Pv_star, Pv=Pv_star,
                De=(lam1 * Uc / R0 if lam1 > 0 else 0.0),
                LAM=(lam2 / lam1 if lam1 > 0 else 0.0),
                Cstar=C8 / Uc, alphax=alphax,
                ee=pA / P8, om=omega * t0, tw=TW / t0, dt=DT / t0, mn=mn,
                wave_type=wave_type, chi=chi, alpha_g=alpha_g, beta_g=beta_g)


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


def _pinf(tn, p):
    """Far-field pressure perturbation (Pf8, Pf8dot), f_pinfinity.m. Nondimensional."""
    wt, ee, om, tw, dt, mn = (p['wave_type'], p['ee'], p['om'], p['tw'], p['dt'], p['mn'])
    if ee == 0.0:
        return 0.0, 0.0
    if wt == 0:                                        # constant offset impulse
        return ee, 0.0
    if wt == 1:                                        # Gaussian
        e = np.exp(-((tn - dt) ** 2) / tw ** 2)
        return -ee * e, ee * (2 * (tn - dt) / tw ** 2) * e
    if wt == 2:                                        # histotripsy pulse
        if tn < dt - np.pi / om or tn > dt + np.pi / om:
            return 0.0, 0.0
        c = 0.5 + 0.5 * np.cos(om * (tn - dt))
        return (ee * c ** mn,
                -ee * mn * c ** (mn - 1) * 0.5 * om * np.sin(om * (tn - dt)))
    if wt == 3:                                        # Heaviside step
        return (-ee * (1.0 - (1.0 if tn > tw else 0.0)), 0.0)
    raise ValueError(f"wave_type={wt} not supported")


def _nZ(stress):
    """Number of internal stress variables."""
    return {1: 0, 2: 0, 3: 1, 5: 2}[stress]


def _JdotA(stress, p):
    """f_call_params.m: 4/Re8 for stress 1-4, 4*LAM/Re8 for stress 5-6."""
    return 4.0 / p['Re8'] if stress in (1, 2, 3, 4) else 4.0 * p['LAM'] / p['Re8']


def _rhs(tn, y, p, stress, radial, bubtherm=0, D1=None, D2=None, ygrid=None):
    R = max(y[0], 1e-8)
    Rd = y[1]
    Pv = p['Pv']

    if bubtherm:
        P = y[2]
        Nt = ygrid.size
        theta = y[3:3 + Nt]
        Zstart = 3 + Nt
    else:
        P = (p['Pb'] - Pv) * R ** (-3 * KAPPA) + Pv          # f_imr_fd.m:412
        Zstart = 2

    nz = _nZ(stress)
    Z = y[Zstart:Zstart + nz] if nz else None
    S, Sdot, dZ = _stress(stress, p, R, Rd, Z)
    Pf8, Pf8dot = _pinf(tn, p)
    iWe = p['iWe']

    thetadot = None
    if bubtherm:
        # f_imr_fd.m, "elseif bubtherm" branch, kv0=0 (dry gas) simplification
        alpha_g, beta_g, chi = p['alpha_g'], p['beta_g'], p['chi']
        T = (alpha_g - 1.0 + np.sqrt(1.0 + 2.0 * theta * alpha_g)) / alpha_g
        dtheta = D1 @ theta
        ddtheta = D2 @ theta
        Pdot = 3.0 / R * (chi * (KAPPA - 1.0) * dtheta[-1] / R - KAPPA * P * Rd)
        Uvel = (chi / R * (KAPPA - 1.0) * dtheta - ygrid * R * Pdot / 3.0) / (KAPPA * P)
        Kstar = alpha_g * T + beta_g
        diffusion = (chi * ddtheta / R ** 2 + Pdot) * (KAPOVER * Kstar * T / P)
        advection = -dtheta * (Uvel - ygrid * Rd) / R
        thetadot = advection + diffusion
        thetadot[-1] = 0.0
    else:
        Pdot = -3 * KAPPA * (p['Pb'] - Pv) * R ** (-3 * KAPPA - 1) * Rd

    if radial == 1:                                     # Rayleigh-Plesset
        Rdd = (P - 1 - Pf8 - iWe / R + S - 1.5 * Rd ** 2) / R
    elif radial == 2:                                   # Keller-Miksis (pressure form)
        Cs = p['Cstar']
        num = ((1 + Rd / Cs) * (P - 1 - Pf8 - iWe / R + S)
               + R / Cs * (Pdot + iWe * Rd / R ** 2 + Sdot - Pf8dot)
               - 1.5 * (1 - Rd / (3 * Cs)) * Rd ** 2)
        den = (1 - Rd / Cs) * R + _JdotA(stress, p) / Cs
        Rdd = num / den
    else:
        raise ValueError(f"radial={radial} not supported")

    out = [Rd, Rdd]
    if bubtherm:
        out.append(Pdot)
        out.extend(thetadot.tolist())
    if dZ is not None:
        out.extend(dZ.tolist())
    return out


def simulate(tv, R0, Req, G, mu, stress=1, lam1=0.0, lam2=0.0, alphax=0.25,
             radial=1, vapor=0, T8=298.15, pA=0.0, omega=0.0, TW=0.0, DT=0.0,
             mn=0.0, wave_type=0, bubtherm=0, Nt=25, rtol=1e-8, atol=1e-10):
    """Bubble radius R(t)/R0 at times tv (seconds). Returns NaN array on failure.

    bubtherm=1: dry gas (vapor must be 0) thermal PDE -- see module docstring
    for exact scope. Nt is the number of interior grid points.
    """
    if bubtherm and vapor:
        raise ValueError("bubtherm=1 currently requires vapor=0 (dry gas only)")
    p = params(R0, Req, G, mu, lam1, lam2, alphax, vapor, T8,
               pA, omega, TW, DT, mn, wave_type, bubtherm)
    tn = np.asarray(tv) / p['t0']
    if bubtherm:
        D1 = finite_diff_mat(Nt, 1, tm_check=0)
        D2 = finite_diff_mat(Nt, 2, tm_check=0)
        ygrid = np.linspace(0.0, 1.0, Nt)
        theta0 = [0.0] * Nt
        y0 = [1.0, 0.0, p['Pb']] + theta0 + [0.0] * _nZ(stress)
        args = (p, stress, radial, 1, D1, D2, ygrid)
    else:
        y0 = [1.0, 0.0] + [0.0] * _nZ(stress)
        args = (p, stress, radial)
    s = solve_ivp(_rhs, (tn[0], tn[-1]), y0, t_eval=tn, args=args,
                  method='LSODA', rtol=rtol, atol=atol)
    if not s.success or s.y.shape[1] != len(tn):
        return np.full(len(tn), np.nan)
    return s.y[0]
