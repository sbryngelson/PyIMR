"""Full validation suite. Run: python3 tests/run_validation.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import constitutive as C
import imr_fast
from imr_fast import params
from imr_grad import NP, simulate_grad


def solve_radius(times, R0, Req, G, mu, **options):
    config = imr_fast.SimulationConfig(R0=R0, Req=Req, G=G, mu=mu, **options)
    return imr_fast.simulate(times, config).radius_ratio


fail = 0
print("="*64); print("1. FORWARD SOLVER vs IMRv2 reference trajectories"); print("="*64)
_d = os.path.dirname(os.path.abspath(__file__))
_R0 = 225e-6; _t0 = _R0/np.sqrt(101325/1064)
_t2 = np.loadtxt(f"{_d}/imr2_t.csv"); _A = np.loadtxt(f"{_d}/imr2_s06.csv", delimiter=',')
_Gg = np.loadtxt(f"{_d}/imr2_G.csv"); _Mg = np.loadtxt(f"{_d}/imr2_M.csv")
_checks = [("Zener truth De=2 s=6",
            solve_radius(_t2,_R0,_R0/6,2500.,.1,stress=3,lam1=2*_t0,lam2=.4*_t0), _A[:,0])]
for _k in [0, 30, len(_Gg)*len(_Mg)-1]:
    _gi,_mi = _k//len(_Mg), _k % len(_Mg)
    _checks.append((f"NHKV G={_Gg[_gi]:.0f} mu={_Mg[_mi]:.4f}",
                    solve_radius(_t2,_R0,_R0/6,_Gg[_gi],_Mg[_mi],stress=1), _A[:,1+_k]))
for _lab,_py,_ml in _checks:
    _mx = np.nanmax(np.abs(_ml-_py)); _ok = _mx < 2e-3; fail += (not _ok)
    print(f"    {_lab:30s} max|dR|={_mx:.2e}  {'PASS' if _ok else 'FAIL'}")

print("\n"+"="*64); print("1b. EXTENDED FEATURES vs IMRv2 references"); print("="*64)
_d = os.path.dirname(os.path.abspath(__file__))
_R0=225e-6; _t0=_R0/np.sqrt(101325/1064)
_t = np.loadtxt(f"{_d}/ref_t.csv")
for lab, kw, ref in [
    ("qKV alphax=0.10",      dict(stress=2, alphax=0.10),                          "ref_qkv_a010.csv"),
    ("qKV alphax=0.25",      dict(stress=2, alphax=0.25),                          "ref_qkv_a025.csv"),
    ("UCM/OldB De=0.5",      dict(stress=5, lam1=0.5*_t0, lam2=0.1*_t0),           "ref_ucm_De005.csv"),
    ("UCM/OldB De=2.0",      dict(stress=5, lam1=2.0*_t0, lam2=0.4*_t0),           "ref_ucm_De020.csv"),
    ("Keller-Miksis NHKV",   dict(stress=1, radial=2),                             "ref_km_nhkv.csv"),
    ("Keller-Miksis Zener",  dict(stress=3, radial=2, lam1=2.0*_t0, lam2=0.4*_t0), "ref_km_zener.csv"),
    ("Gaussian forcing pA=5e4",  dict(wave_type=1, pA=5e4, TW=5e-6, DT=2e-5),  "ref_gauss_pA50.csv"),
    ("Gaussian forcing pA=2e5",  dict(wave_type=1, pA=2e5, TW=5e-6, DT=2e-5),  "ref_gauss_pA200.csv"),
    ("constant offset pA=3e4",   dict(wave_type=0, pA=3e4),                    "ref_imp_pA30.csv"),
    ("Heaviside step pA=5e4",    dict(wave_type=3, pA=5e4, TW=3e-5),           "ref_heav_pA50.csv"),
    ("histotripsy pulse",        dict(wave_type=2, pA=1e5, omega=2*np.pi/2e-5, DT=3e-5, mn=2), "ref_histo.csv"),
    ("vapor=1 (T=298.15K)",      dict(vapor=1, T8=298.15),                     "ref_vapor.csv"),
    ("bubtherm=1 (thermal PDE)", dict(bubtherm=1, Nt=25),                      "ref_bubtherm.csv"),
    ("medtherm=1 (liquid layer)", dict(bubtherm=1, medtherm=1, Nt=25, Mt=25),  "ref_medtherm.csv"),
    ("masstrans=1 (vapor transfer)", dict(bubtherm=1, vapor=1, masstrans=1, Nt=25), "ref_masstrans.csv"),
    ("masstrans=1+medtherm=1 (coupled)", dict(bubtherm=1, vapor=1, masstrans=1, medtherm=1, Nt=25, Mt=25),
                                                                               "ref_masstrans_medtherm.csv"),
    ("stress=0 (no stress)",     dict(stress=0),                                                "ref_stress0.csv"),
    ("stress=4 (qKV Zener)",     dict(stress=4, lam1=2.0*_t0, lam2=0.4*_t0, alphax=0.25),        "ref_stress4.csv"),
    ("radial=3 (KM enthalpy, Tait)",     dict(radial=3),                                        "ref_radial3.csv"),
    ("radial=4 (Gilmore, Tait)",         dict(radial=4),                                        "ref_radial4.csv"),
    ("radial=3+Zener",  dict(radial=3, stress=3, lam1=2.0*_t0, lam2=0.4*_t0),          "ref_radial3_zener.csv"),
    ("radial=4+Zener",  dict(radial=4, stress=3, lam1=2.0*_t0, lam2=0.4*_t0),          "ref_radial4_zener.csv"),
    ("radial=5 (KM enthalpy, Mie-Gruneisen)", dict(radial=5),                          "ref_radial5.csv"),
]:
    ml = np.loadtxt(f"{_d}/{ref}")
    py = solve_radius(_t, _R0, _R0/6, 2500., 0.1, **kw)
    mx = np.nanmax(np.abs(ml-py)); ok = mx < 2e-3; fail += (not ok)
    print(f"    {lab:24s} max|dR|={mx:.2e}  {'PASS' if ok else 'FAIL'}")

print("\n"+"="*64); print("2. CONSTITUTIVE SUITE"); print("="*64)
G = Req = 1.0
print("  elastic: neo-Hookean vs closed form -(G/2)(5-4Rst-Rst^4)")
for ra in [1.5, 3.0, 8.0, 20.0]:
    v = C.S_elastic(ra, Req, C.neo_hookean(G)); Rst = 1/ra
    c = -(G/2)*(5 - 4*Rst - Rst**4); rel = abs(v-c)/abs(c)
    ok = rel < 1e-4; fail += (not ok)
    print(f"    R/Req={ra:<5} rel={rel:.2e}  {'PASS' if ok else 'FAIL'}")
print("  viscous: Newtonian vs closed form -4*mu*Rdot/R")
for R, Rd, mu in [(1.,-1.,.1), (2.,-3.,.05)]:
    v = C.S_viscous(R, Rd, C.newtonian(mu)); c = -4*mu*Rd/R
    rel = abs(v-c)/abs(c); ok = rel < 1e-4; fail += (not ok)
    print(f"    R={R} Rdot={Rd:<5} rel={rel:.2e}  {'PASS' if ok else 'FAIL'}")
print("  model reduction limits (must collapse to neo-Hookean / Newtonian)")
nh = C.S_elastic(3., 1., C.neo_hookean(1.))
for lab, f in [("Gent Jm=1e8", C.gent(1., 1e8)), ("Mooney C01=0", C.mooney_rivlin(.5, 0.)),
               ("Fung b=1e-8", C.fung(1., 1e-8)), ("Yeoh c2=c3=0", C.yeoh(.5, 0., 0.))]:
    rel = abs(C.S_elastic(3., 1., f) - nh)/abs(nh); ok = rel < 1e-6; fail += (not ok)
    print(f"    {lab:16s} rel={rel:.1e}  {'PASS' if ok else 'FAIL'}")
nv = C.S_viscous(2., -3., C.newtonian(.05))
for lab, f in [("power-law n=1", C.power_law(.05, 1.)), ("Carreau lam=0", C.carreau_yasuda(.05,.05,0.,2.,.5)),
               ("Cross lam=0", C.cross(.05,.05,0.,1.))]:
    rel = abs(C.S_viscous(2.,-3.,f) - nv)/abs(nv); ok = rel < 1e-9; fail += (not ok)
    print(f"    {lab:16s} rel={rel:.1e}  {'PASS' if ok else 'FAIL'}")
print("  Gent finite-extensibility lock-up detection")
for Jm, st, exp in [(5., 3., False), (500., 3., True)]:
    got = C.lock_free(C.gent(1., Jm), st); ok = (got == exp); fail += (not ok)
    print(f"    Jm={Jm:<6} stretch={st}  lock_free={got} (expect {exp})  {'PASS' if ok else 'FAIL'}")

print("\n"+"="*64); print("2b. NONLINEAR MEMORY (Giesekus / PTT) reduction limits"); print("="*64)
_De, _LAM = 2.0, 0.2
_tv = np.linspace(0, 1.2e-4, 300)
_p0 = params(225e-6, 225e-6/6, 2500., .1)
_ucm = solve_radius(_tv, 225e-6, 225e-6/6, 2500., .1, stress=5,
                    lam1=_De*_p0['t0'], lam2=_LAM*_De*_p0['t0'])
print("  zero nonlinearity must reproduce UCM/Oldroyd-B")
for _name, _model in [
    ("giesekus", imr_fast.GiesekusModel()),
    ("linear PTT", imr_fast.LinearPTTModel()),
]:
    _R = solve_radius(
        _tv, 225e-6, 225e-6/6, 2500., .1,
        stress=_model, lam1=_De*_p0["t0"], lam2=_LAM*_De*_p0["t0"],
    )
    _mx = np.nanmax(np.abs(_R-_ucm))
    _ok = _mx < 5e-3; fail += (not _ok)      # discretisation-limited, converging in points
    print(f"    {_name:>10} -> UCM   max|dR|={_mx:.2e}  {'PASS' if _ok else 'FAIL'}")
_ucm_km = solve_radius(
    _tv, 225e-6, 225e-6/6, 2500., .1, stress=5, radial=2,
    lam1=_De*_p0["t0"], lam2=_LAM*_De*_p0["t0"],
)
_distributed_km = solve_radius(
    _tv, 225e-6, 225e-6/6, 2500., .1,
    stress=imr_fast.GiesekusModel(), radial=2,
    lam1=_De*_p0["t0"], lam2=_LAM*_De*_p0["t0"],
)
_mx = np.nanmax(np.abs(_distributed_km-_ucm_km))
_ok = _mx < 2e-3; fail += (not _ok)
print(f"    {'KM Giesekus':>10} -> UCM   max|dR|={_mx:.2e}  {'PASS' if _ok else 'FAIL'}")
_coupled_options = dict(
    bubtherm=1, medtherm=1, vapor=1, masstrans=1, Nt=9, Mt=9,
    lam1=_De*_p0["t0"], lam2=_LAM*_De*_p0["t0"],
)
_ucm_coupled = solve_radius(
    _tv, 225e-6, 225e-6/6, 2500., .1, stress=5, **_coupled_options,
)
_distributed_coupled = solve_radius(
    _tv, 225e-6, 225e-6/6, 2500., .1,
    stress=imr_fast.GiesekusModel(), **_coupled_options,
)
_mx = np.nanmax(np.abs(_distributed_coupled-_ucm_coupled))
_ok = _mx < 3e-3; fail += (not _ok)
print(f"    {'coupled':>10} -> UCM   max|dR|={_mx:.2e}  {'PASS' if _ok else 'FAIL'}")
print("  nonlinear parameter must produce distinct physics")
for _name, _model in [
    ("giesekus", imr_fast.GiesekusModel(mobility=0.2)),
    ("linear PTT", imr_fast.LinearPTTModel(extensibility=0.2)),
]:
    _R = solve_radius(
        _tv, 225e-6, 225e-6/6, 2500., .1,
        stress=_model, lam1=_De*_p0["t0"], lam2=_LAM*_De*_p0["t0"],
    )
    _mx = np.nanmax(np.abs(_R-_ucm)); _ok = _mx > 0.05; fail += (not _ok)
    print(f"    {_name:>10} parameter=0.2  max|dR| vs UCM={_mx:.2e}  {'PASS' if _ok else 'FAIL'}")

print("\n"+"="*64); print("3. GRADIENTS (forward sensitivities) vs finite differences"); print("="*64)
p = params(225e-6, 225e-6/6, 2500., .1)
c = np.zeros(NP); c[0] = -5/(2*p['Ca']); c[1] = 4/(2*p['Ca']); c[3] = 1/(2*p['Ca']); c[5] = -4/p['Re8']
tn = np.linspace(0, 1.2e-4, 300)/p['t0']
R, dR = simulate_grad(tn, p, c)
for k, nm in [(0,'1'), (1,'Rst'), (3,'Rst^4'), (5,'Rd/R')]:
    h = abs(c[k])*1e-4
    Rp,_ = simulate_grad(tn, p, c + h*np.eye(NP)[k]); Rm,_ = simulate_grad(tn, p, c - h*np.eye(NP)[k])
    fd = (Rp-Rm)/(2*h); m = np.isfinite(fd)
    rel = np.linalg.norm(dR[m,k]-fd[m])/np.linalg.norm(fd[m]); ok = rel < 1e-2; fail += (not ok)
    print(f"    d/d[{nm:>6}] rel={rel:.2e}  {'PASS' if ok else 'FAIL'}")

print("\n" + ("ALL VALIDATION PASSED" if fail == 0 else f"{fail} CHECK(S) FAILED"))
sys.exit(1 if fail else 0)
