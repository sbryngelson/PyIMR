import numpy as np
"""
CONSTITUTIVE SUITE for IMR constitutive-law discovery.
Stress integral S = int_R^inf (2/r)(tau_rr - tau_thth) dr, nondimensionalised by p_inf.

ELASTIC (stretch coordinate, exact & validated):
    S_el = int_1^{lam_w} 2*taudiff(lam)/(lam*(lam^3-1)) dlam
VISCOUS (generalized Newtonian, derived below):
    velocity v_r = Rdot R^2/r^2 ; D = diag(-2a,a,a), a = Rdot R^2/r^3
    tau_rr - tau_thth = 2 mu(gdot) (D_rr - D_thth) = -6 mu a
    gdot = sqrt(2 D:D) = 2*sqrt(3)*|a|
    S_vis = -12 Rdot R^2 int_R^inf mu(gdot(r))/r^4 dr
    (Newtonian mu=const -> -4 mu Rdot/R, the known result)
MEMORY: internal-variable ODEs (handled by the solver, not here).
"""
SQ3=np.sqrt(3.0)

# ---------------- elastic: dW/dlam_r, dW/dlam ----------------
def neo_hookean(G=1.0):        return lambda lr,l:(G*lr, G*l)
def mooney_rivlin(C10,C01):    return lambda lr,l:(2*C10*lr-2*C01/lr**3, 2*C10*l-2*C01/l**3)
def yeoh(c1,c2,c3):
    def f(lr,l):
        d=lr**2+2*l**2-3.0; c=2*(c1+2*c2*d+3*c3*d**2); return (c*lr,c*l)
    return f
def fung(G,b):
    def f(lr,l):
        c=G*np.exp(b*(lr**2+2*l**2-3.0)); return (c*lr,c*l)
    return f
def gent(G,Jm):
    """W=-(G Jm/2)ln(1-(I1-3)/Jm). LOCKS when I1-3 >= Jm (finite extensibility)."""
    def f(lr,l):
        d=1.0-(lr**2+2*l**2-3.0)/Jm
        c=G/np.where(d>1e-6,d,np.nan)          # NaN signals lock-up
        return (c*lr,c*l)
    return f
def arruda_boyce(G,N):
    """8-chain, series-truncated Pade-free form (first 5 terms)."""
    C=[0.5,1/20.,11/1050.,19/7000.,519/673750.]
    def f(lr,l):
        I1=lr**2+2*l**2
        c=2*sum((k+1)*C[k]*(I1**k)/(N**k) for k in range(5))
        return (c*lr,c*l)
    return f
def lock_free(dWd,lam_w):
    """True if the model stays finite up to wall stretch lam_w."""
    lr=1.0/lam_w**2
    v=dWd(np.array([lr]),np.array([lam_w]))
    return np.all(np.isfinite(v[0])) and np.all(np.isfinite(v[1]))

def S_elastic(R,Req,dWd,n=800):
    lw=R/Req
    if abs(lw-1.0)<1e-12: return 0.0
    s=np.linspace(0.,1.,n); lam=np.clip(1.0+(lw-1.0)*s**2,1.0+1e-12,None)
    lr=1.0/lam**2; dr_,dl_=dWd(lr,lam)
    num=2.0*(lr*dr_-lam*dl_); den=lam*(lam**3-1.0)
    f=num/den
    if not np.all(np.isfinite(f)): return np.nan
    f[0]=f[1]
    return np.trapezoid(f*(2.0*(lw-1.0)*s),s)

# ---------------- viscous: mu(gammadot) ----------------
def newtonian(mu):             return lambda g: np.full_like(np.asarray(g,float),mu)
def power_law(K,n_):           return lambda g: K*np.maximum(np.asarray(g,float),1e-12)**(n_-1)
def carreau_yasuda(mu0,muinf,lam_c,a,n_):
    return lambda g: muinf+(mu0-muinf)*(1+(lam_c*np.asarray(g,float))**a)**((n_-1)/a)
def cross(mu0,muinf,lam_c,m):
    return lambda g: muinf+(mu0-muinf)/(1+(lam_c*np.asarray(g,float))**m)
def herschel_bulkley(tau_y,K,n_,greg=1e-3):
    """yield stress: mu_eff = tau_y/gdot + K gdot^(n-1), regularised."""
    return lambda g: tau_y/np.maximum(np.asarray(g,float),greg)+K*np.maximum(np.asarray(g,float),1e-12)**(n_-1)
def bingham(tau_y,mu_p,greg=1e-3): return herschel_bulkley(tau_y,mu_p,1.0,greg)

def S_viscous(R,Rdot,mufun,n=600):
    """u = R/r substitution -> FINITE interval [0,1], no truncation.
       S_vis = -12*(Rdot/R)*int_0^1 mu(gdot(u)) u^2 du,  gdot = 2*sqrt(3)*|Rdot|*u^3/R
       Newtonian: int mu u^2 du = mu/3  =>  S = -4*mu*Rdot/R (exact)."""
    if Rdot==0.0: return 0.0
    u=np.linspace(0.,1.,n)
    g=2*SQ3*abs(Rdot)*u**3/R
    return -12.0*(Rdot/R)*np.trapezoid(mufun(g)*u**2,u)

if __name__=="__main__":
    print("=== VALIDATION ===")
    G=1.;Req=1.
    print("elastic: neo-Hookean vs closed form")
    for ra in [1.5,3.,8.]:
        v=S_elastic(ra,Req,neo_hookean(G)); Rst=1/ra; c=-(G/2)*(5-4*Rst-Rst**4)
        print(f"   R/Req={ra:<5} num={v:>11.7f} closed={c:>11.7f} rel={abs(v-c)/abs(c):.1e}")
    print("\nviscous: Newtonian vs closed form  S=-4*mu*Rdot/R")
    for R,Rd,mu in [(1.,-1.,0.1),(2.,-3.,0.05),(0.5,1.,0.2)]:
        v=S_viscous(R,Rd,newtonian(mu)); c=-4*mu*Rd/R
        print(f"   R={R:<4} Rdot={Rd:<5} num={v:>11.7f} closed={c:>11.7f} rel={abs(v-c)/abs(c):.1e}")
    print("\nviscous limits: power-law n=1 -> Newtonian; Carreau lam=0 -> Newtonian")
    base=S_viscous(2.,-3.,newtonian(0.05))
    for lab,f in [("power_law K=.05,n=1",power_law(.05,1.)),
                  ("carreau lam_c=0",carreau_yasuda(.05,.05,0.,2.,.5)),
                  ("cross lam_c=0",cross(.05,.05,0.,1.))]:
        v=S_viscous(2.,-3.,f); print(f"   {lab:22s} {v:>11.7f} vs {base:>11.7f} rel={abs(v-base)/abs(base):.1e}")
    print("\nGent lock-up check (finite extensibility):")
    for Jm in [5.,50.,500.]:
        for ra in [2.,3.,5.]:
            ok=lock_free(gent(1.,Jm),ra)
            print(f"   Jm={Jm:<6} stretch={ra:<4} {'OK' if ok else 'LOCKED (unphysical)'}")
    print("\nDistinct physics at R/Req=3 (elastic) and R=2,Rdot=-3 (viscous):")
    nh=S_elastic(3.,1.,neo_hookean(1.))
    for lab,f in [("Mooney .25/.25",mooney_rivlin(.25,.25)),("Yeoh stiffen",yeoh(.5,.05,.005)),
                  ("Fung b=0.2",fung(1.,.2)),("Gent Jm=500",gent(1.,500.)),("Arruda N=50",arruda_boyce(1.,50.))]:
        print(f"   {lab:16s} S_el={S_elastic(3.,1.,f):>12.5f}  ({S_elastic(3.,1.,f)/nh:>5.2f}x NH)")
    nv=S_viscous(2.,-3.,newtonian(.05))
    for lab,f in [("power-law n=0.5",power_law(.05,.5)),("Carreau shear-thin",carreau_yasuda(.05,.005,1.,2.,.4)),
                  ("Herschel-Bulkley",herschel_bulkley(.02,.05,.8)),("Bingham ty=.02",bingham(.02,.05))]:
        print(f"   {lab:20s} S_vis={S_viscous(2.,-3.,f):>12.5f}  ({S_viscous(2.,-3.,f)/nv:>5.2f}x Newt)")
