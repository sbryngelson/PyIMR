import time

import numpy as np
from scipy.integrate import solve_ivp

from imr_fast import KAPPA, params

"""
DIFFERENTIABLE IMR solver via FORWARD SENSITIVITY EQUATIONS.
Exact analytic gradients dR/dc_k -- no autodiff dependency, and cheaper than AD
for this small system.  State y=[R,V]; augmented with s_k = dy/dc_k.

  f1 = V
  f2 = (P - 1 - iWe/R + S - 1.5 V^2)/R,   S = sum_k c_k phi_k(R,V)
  ds_k/dt = J_y s_k + df/dc_k,   df2/dc_k = phi_k / R
"""
# library: phi_k(R,V) with analytic partials
def library(R,V,req):
    Rst=req/R; q=V/R
    phi   = np.array([1.0, Rst, Rst**2, Rst**4, Rst**5, q])
    dphidR= np.array([0.0, -Rst/R, -2*Rst**2/R, -4*Rst**4/R, -5*Rst**5/R, -q/R])
    dphidV= np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0/R])
    return phi,dphidR,dphidV
NP=6
def make_rhs(p,c):
    req,Pb,iWe=p['req'],p['Pb'],p['iWe']
    def rhs(t,y):
        R=max(y[0],1e-8); V=y[1]
        phi,dphidR,dphidV=library(R,V,req)
        S=float(c@phi); dSdR=float(c@dphidR); dSdV=float(c@dphidV)
        P=Pb*R**(-3*KAPPA); dPdR=-3*KAPPA*Pb*R**(-3*KAPPA-1)
        f2=(P-1-iWe/R+S-1.5*V**2)/R
        # Jacobian wrt state
        J=np.array([[0.0,1.0],
                    [(dPdR+iWe/R**2+dSdR)/R - f2/R, (dSdV-3*V)/R]])
        out=np.empty(2+2*NP); out[0]=V; out[1]=f2
        s=y[2:].reshape(NP,2)
        ds=s@J.T
        ds[:,1]+=phi/R                      # df2/dc_k
        out[2:]=ds.ravel()
        return out
    return rhs
def simulate_grad(tn,p,c,rtol=1e-9,atol=1e-11):
    y0=np.zeros(2+2*NP); y0[0]=1.0
    so=solve_ivp(make_rhs(p,np.asarray(c,float)),(tn[0],tn[-1]),y0,t_eval=tn,
                 method='LSODA',rtol=rtol,atol=atol)
    if not so.success or so.y.shape[1]!=len(tn):
        return np.full(len(tn),np.nan),np.full((len(tn),NP),np.nan)
    R=so.y[0]; dR=so.y[2::2].T                # (nt, NP) sensitivities dR/dc_k
    return R,dR
if __name__=="__main__":
    R0=225e-6; s=6
    p=params(R0,R0/s,2500.,0.1)
    c=np.zeros(NP); c[0]=-5/(2*p['Ca']); c[1]=4/(2*p['Ca']); c[3]=1/(2*p['Ca']); c[5]=-4/p['Re8']
    tn=np.linspace(0,1.2e-4,400)/p['t0']
    R,dR=simulate_grad(tn,p,c)
    print("VALIDATE gradients vs central finite differences\n")
    print(f"{'k':>3} {'term':>8} {'analytic':>14} {'finite-diff':>14} {'rel err':>10}")
    names=['1','Rst','Rst^2','Rst^4','Rst^5','Rd/R']
    for k in range(NP):
        h=max(abs(c[k]),1e-3)*1e-5
        cp=c.copy(); cp[k]+=h; Rp,_=simulate_grad(tn,p,cp)
        cm=c.copy(); cm[k]-=h; Rm,_=simulate_grad(tn,p,cm)
        fd=(Rp-Rm)/(2*h)
        a=dR[:,k]; msk=np.isfinite(fd)&np.isfinite(a)
        rel=np.linalg.norm(a[msk]-fd[msk])/max(np.linalg.norm(fd[msk]),1e-30)
        print(f"{k:>3} {names[k]:>8} {np.linalg.norm(a[msk]):>14.5e} {np.linalg.norm(fd[msk]):>14.5e} {rel:>10.2e}")
    st=time.time()
    for _ in range(20): simulate_grad(tn,p,c)
    print(f"\nspeed: {(time.time()-st)/20*1000:.1f} ms/solve (value + all {NP} gradients)")
