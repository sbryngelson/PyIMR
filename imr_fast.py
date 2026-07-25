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

bubtherm=1 implements IMRv2's "elseif bubtherm" branch (f_imr_fd.m): gas-phase
thermal PDE, dry gas (kv0=0, vapor=0). With medtherm=0, the wall is an
isothermal-equivalent clamp (thetadot[-1]=0). Its Pdot uses bare P (kappa*P),
NOT (P-Pv) -- this is IMRv2's actual equation for this branch, not a
simplification; the bubtherm=0 polytropic branch's Pdot uses (P-Pv) and the
two are NOT reconciled/harmonized, since they are genuinely different
equations in the source.

medtherm=1 (requires bubtherm=1) adds the liquid boundary layer: a stretched
exterior grid (Mt points, Lt controls the stretching), advection+diffusion+
viscous-dissipation RHS for Tm, and the wall temperature theta[-1] is NOT a
free state -- it is solved every RHS call via a 1-D root-find (scipy.optimize
.newton, matching IMRv2's fzero-from-single-guess; f_bubble_wall_thermal_bc)
enforcing heat-flux continuity across the interface, warm-started from the
previous call's solution. thetadot[-1]=0 and Tmdot[0]=0 always (both slots
are formally-integrated-but-frozen; their real continuity is the warm-start
guess, external to the ODE state -- exactly mirroring IMRv2's own
theta_bw_guess closure variable). GRADIENTS ARE NOT VALIDATED for medtherm=1:
naive autodiff through the root-find would need the implicit function
theorem, not attempted here.

NOT valid outside the above (masstrans, Gilmore, PTT, Giesekus). Re-validate
before extending.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import newton
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

# liquid (medium) thermal properties, IMRv2 defaults (default_case.m); water-like
_KM = 0.55          # liquid thermal conductivity (W/m/K)
_CP = 4.181e3        # liquid specific heat (J/kg/K)
_LT = 2.0            # exterior-grid stretching length (default_case.m)


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
    # medium (liquid) thermal groups, f_call_params.m -- Dm=Km/(rho*Cp) is the
    # standard liquid thermal-diffusivity formula (confirmed from source, not
    # assumed); needed only when medtherm=1 but cheap to compute unconditionally
    Dm = _KM / (RHO * _CP)
    Foh = Dm / (Uc * R0)
    iota = _KM / (K8 * _LT)
    Br = Uc ** 2 / (_CP * T8)
    return dict(t0=t0, Ca=Ca, Re8=Re8, iWe=1.0 / We, req=Req / R0,
                Pb=P0 / P8 + Pv_star, Pv=Pv_star,
                De=(lam1 * Uc / R0 if lam1 > 0 else 0.0),
                LAM=(lam2 / lam1 if lam1 > 0 else 0.0),
                Cstar=C8 / Uc, alphax=alphax,
                ee=pA / P8, om=omega * t0, tw=TW / t0, dt=DT / t0, mn=mn,
                wave_type=wave_type, chi=chi, alpha_g=alpha_g, beta_g=beta_g,
                Foh=Foh, iota=iota, Br=Br, Lt=_LT)


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


def _dissipation(stress, p, R, Rd, yT2, yT3, iyT3, iyT4, iyT6):
    """taugradu, f_stress_dissipation.m, finite-difference (non-spectral) mode.
    fnu=0 always (nu_model not supported), so Br/(Re8+DRe*fnu) simplifies to
    Br/Re8 exactly."""
    Ca, Re8, Br, ax = p['Ca'], p['Re8'], p['Br'], p['alphax']
    Rst = p['req'] / R
    x2 = (yT3 - 1.0 + Rst ** 3) ** (2.0 / 3.0)
    ix2 = 1.0 / x2
    x4 = x2 ** 2
    base = (12.0 * (Br / Re8) * (Rd / R) ** 2 * iyT6
            + 2.0 * Br / Ca * iyT3 * (Rd / R) * (yT2 * ix2 - iyT4 * x4))
    if stress == 2:
        return base * (1.0 + ax * (x4 * iyT4 + 2.0 * yT2 * ix2 - 3.0))
    return base   # stress in {1, 3, 5}: identical to the base formula


def _wall_theta_bw(guess, theta_tail, Tm_tail, alpha_g, grad_Tm, grad_Trans):
    """Solve heat-flux continuity at the wall for theta[-1], f_bubble_wall_thermal_bc.m.
    theta_tail = [theta[-2], theta[-3]], Tm_tail = [Tm[1], Tm[2]] (0-indexed)."""
    def resid(theta_bw):
        Tw = (alpha_g - 1.0 + np.sqrt(1.0 + 2.0 * theta_bw * alpha_g)) / alpha_g
        lhs = grad_Tm[0] * Tw + grad_Tm[1] * Tm_tail[0] + grad_Tm[2] * Tm_tail[1]
        rhs = grad_Trans[0] * theta_bw + grad_Trans[1] * theta_tail[0] + grad_Trans[2] * theta_tail[1]
        return lhs + rhs
    return newton(resid, guess, tol=1e-13, maxiter=100)


def _rhs(tn, y, p, stress, radial, bubtherm=0, D1=None, D2=None, ygrid=None,
          medtherm=0, mt=None):
    R = max(y[0], 1e-8)
    Rd = y[1]
    Pv = p['Pv']

    if bubtherm:
        P = y[2]
        Nt = ygrid.size
        theta = y[3:3 + Nt].copy()
        if medtherm:
            Tm = y[3 + Nt:3 + Nt + mt['Mt']].copy()
            theta[-1] = _wall_theta_bw(mt['wall_guess'][0], [theta[-2], theta[-3]],
                                        [Tm[1], Tm[2]], p['alpha_g'],
                                        mt['grad_Tm'], mt['grad_Trans'])
            mt['wall_guess'][0] = theta[-1]
            Zstart = 3 + Nt + mt['Mt']
        else:
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
        # f_imr_fd.m, "elseif bubtherm" branch, kv0=0 (dry gas) simplification.
        # Identical whether medtherm is on or off -- only theta[-1]'s VALUE
        # differs (wall-BC solve above vs. frozen at its initial value).
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

    Tmdot = None
    if medtherm:
        # f_imr_fd.m "surrounding temperature" block. xi[-1]=-1 exactly (the
        # far-field point) makes yT[-1]=inf -- an algebraic singularity
        # inherent to IMRv2's own xi=1+(j-1)*deltaYm formula (confirmed:
        # xi_last = 1+Nm*(-2/Nm) = -1 identically, not introduced by this
        # port), producing a transient inf*0=nan in the last entry of
        # med_advection/taugradu. Harmless: Tmdot[-1]=0.0 unconditionally
        # overwrites it and that state slot's value never depends on it
        # (same frozen-slot pattern as theta[-1] without medtherm).
        Tm[0] = T[-1]                                    # wall Dirichlet clamp
        dTm = mt['D1m'] @ Tm
        ddTm = mt['D2m'] @ Tm
        xi, yT, yT2, yT3, iyT3, iyT4, iyT6 = (mt['xi'], mt['yT'], mt['yT2'],
                                               mt['yT3'], mt['iyT3'], mt['iyT4'], mt['iyT6'])
        Lt, Foh, iota = p['Lt'], p['Foh'], p['iota']
        with np.errstate(divide='ignore', invalid='ignore'):
            med_advection = ((1 + xi) ** 2 / (Lt * R)
                              * (Rd / yT2 * (1 - yT3) / 2 + Foh / R * ((xi + 1) / (2 * Lt) - 1 / yT))
                              * dTm)
            med_diffusion = Foh / R ** 2 * (xi + 1) ** 4 / Lt ** 2 * ddTm / 4
            taugradu = _dissipation(stress, p, R, Rd, yT2, yT3, iyT3, iyT4, iyT6)
        Tmdot = med_advection + med_diffusion + taugradu
        Tmdot[0] = 0.0
        Tmdot[-1] = 0.0

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
    if medtherm:
        out.extend(Tmdot.tolist())
    if dZ is not None:
        out.extend(dZ.tolist())
    return out


def simulate(tv, R0, Req, G, mu, stress=1, lam1=0.0, lam2=0.0, alphax=0.25,
             radial=1, vapor=0, T8=298.15, pA=0.0, omega=0.0, TW=0.0, DT=0.0,
             mn=0.0, wave_type=0, bubtherm=0, Nt=25, medtherm=0, Mt=25,
             rtol=1e-8, atol=1e-10):
    """Bubble radius R(t)/R0 at times tv (seconds). Returns NaN array on failure.

    bubtherm=1: dry gas (vapor must be 0) thermal PDE -- see module docstring
    for exact scope. Nt is the number of interior grid points.
    medtherm=1 (requires bubtherm=1): adds the liquid boundary layer, Mt
    exterior grid points. Gradients not validated for medtherm=1.
    """
    if bubtherm and vapor:
        raise ValueError("bubtherm=1 currently requires vapor=0 (dry gas only)")
    if medtherm and not bubtherm:
        raise ValueError("medtherm=1 requires bubtherm=1")
    p = params(R0, Req, G, mu, lam1, lam2, alphax, vapor, T8,
               pA, omega, TW, DT, mn, wave_type, bubtherm)
    tn = np.asarray(tv) / p['t0']
    if bubtherm:
        D1 = finite_diff_mat(Nt, 1, tm_check=0)
        D2 = finite_diff_mat(Nt, 2, tm_check=0)
        ygrid = np.linspace(0.0, 1.0, Nt)
        theta0 = [0.0] * Nt
        y0 = [1.0, 0.0, p['Pb']] + theta0
        mt = None
        if medtherm:
            Nm = Mt - 1
            deltaYm = -2.0 / Nm
            xi = 1.0 + np.arange(Mt) * deltaYm
            # xi[-1]=-1 exactly (see _rhs's medtherm block docstring) -> yT[-1]=inf;
            # harmless, that grid point's Tmdot is unconditionally zeroed in _rhs.
            with np.errstate(divide='ignore', invalid='ignore'):
                yT = (2.0 / (xi + 1.0) - 1.0) * p['Lt'] + 1.0
                yT2, yT3 = yT ** 2, yT ** 3
                iyT3, iyT4, iyT6 = yT ** -3, yT ** -4, yT ** -6
            coeff = np.array([-1.5, 2.0, -0.5])
            deltaY = 1.0 / (Nt - 1)
            grad_Tm = 2 * p['chi'] * p['iota'] / deltaYm * coeff
            grad_Trans = -coeff * p['chi'] / deltaY
            mt = dict(Mt=Mt, xi=xi, yT=yT, yT2=yT2, yT3=yT3, iyT3=iyT3, iyT4=iyT4,
                      iyT6=iyT6, D1m=finite_diff_mat(Mt, 1, tm_check=1),
                      D2m=finite_diff_mat(Mt, 2, tm_check=1),
                      grad_Tm=grad_Tm, grad_Trans=grad_Trans, wall_guess=[-1e-4])
            y0 = y0 + [1.0] * Mt
        y0 = y0 + [0.0] * _nZ(stress)
        args = (p, stress, radial, 1, D1, D2, ygrid, medtherm, mt)
    else:
        y0 = [1.0, 0.0] + [0.0] * _nZ(stress)
        args = (p, stress, radial)
    s = solve_ivp(_rhs, (tn[0], tn[-1]), y0, t_eval=tn, args=args,
                  method='LSODA', rtol=rtol, atol=atol)
    if not s.success or s.y.shape[1] != len(tn):
        return np.full(len(tn), np.nan)
    return s.y[0]
