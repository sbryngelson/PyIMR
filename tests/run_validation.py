"""Full validation suite. Run: python3 tests/run_validation.py"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import imr_fast, constitutive as C
from imr_grad import simulate_grad, NP
from imr_fast import params

fail = 0
print("="*64); print("1. FORWARD SOLVER vs IMRv2 reference trajectories"); print("="*64)
if not imr_fast.validate(refdir=os.path.dirname(os.path.abspath(__file__))): fail += 1

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
