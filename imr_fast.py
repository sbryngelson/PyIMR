"""Fast Python IMR simulator — narrow slice of IMRv2 (radial=1 RP, bubtherm=0
polytropic, no vapor/mass-transfer/forcing), stress=1 (NHKV) and stress=3 (Zener/SLS).
Equations transcribed from IMRv2/src/{f_radial_eq,f_stress,f_call_params}.m.
MUST be validated against MATLAB output before use (see validate())."""
import numpy as np
from scipy.integrate import solve_ivp

P8=101325.0; RHO=1064.0; SURF=0.07; KAPPA=1.4

def params(R0,Req,G,mu,lam1=0.0,lam2=0.0):
    Uc=np.sqrt(P8/RHO); t0=R0/Uc
    Ca=P8/G if G>0 else np.inf
    Re8=P8*R0/(mu*Uc)
    We=P8*R0/(2*SURF); iWe=1.0/We
    req=Req/R0
    P0=(P8+2*SURF/Req)*(Req/R0)**(3*KAPPA)          # polytropic branch
    Pb=P0/P8
    De=lam1*Uc/R0 if lam1>0 else 0.0
    LAM=(lam2/lam1) if lam1>0 else 0.0
    return dict(t0=t0,Ca=Ca,Re8=Re8,iWe=iWe,req=req,Pb=Pb,De=De,LAM=LAM)

def _rhs(t,y,p,stress):
    R,Rd=y[0],y[1]; R=max(R,1e-8)
    Rst=p['req']/R
    P=p['Pb']*R**(-3*KAPPA)
    if stress==1:
        S=-(5-4*Rst-Rst**4)/(2*p['Ca'])-4.0/p['Re8']*Rd/R
        Rdd=(P-1-p['iWe']/R+S-1.5*Rd**2)/R
        return [Rd,Rdd]
    else:
        Z1=y[2]
        S=Z1/R**3-4*p['LAM']/p['Re8']*Rd/R
        Ze=-0.5*(R**3/p['Ca'])*(5-Rst**4-4*Rst)
        Z1d=-(Z1-Ze)/p['De']+4*(p['LAM']-1)/(p['Re8']*p['De'])*R**2*Rd
        Rdd=(P-1-p['iWe']/R+S-1.5*Rd**2)/R
        return [Rd,Rdd,Z1d]

def simulate(tv,R0,Req,G,mu,stress=1,lam1=0.0,lam2=0.0,rtol=1e-8,atol=1e-10):
    p=params(R0,Req,G,mu,lam1,lam2)
    tn=np.asarray(tv)/p['t0']
    y0=[1.0,0.0] if stress==1 else [1.0,0.0,0.0]
    s=solve_ivp(_rhs,(tn[0],tn[-1]),y0,t_eval=tn,args=(p,stress),
                method='LSODA',rtol=rtol,atol=atol)
    if not s.success or s.y.shape[1]!=len(tn): return np.full(len(tn),np.nan)
    return s.y[0]

def validate(refdir='tests'):
    """compare against saved MATLAB trajectories"""
    R0=225e-6; Uc=np.sqrt(P8/RHO); t0=R0/Uc
    t=np.loadtxt(f'{refdir}/imr2_t.csv'); A=np.loadtxt(f'{refdir}/imr2_s06.csv',delimiter=',')
    Gg=np.loadtxt(f'{refdir}/imr2_G.csv'); Mg=np.loadtxt(f'{refdir}/imr2_M.csv')
    out=[]
    # (a) Zener truth: De=2 -> lam1=2*t0, lam2=0.2*lam1, G=2500, mu=0.1, s=6
    ml=A[:,0]; py=simulate(t,R0,R0/6,2500.,0.1,stress=3,lam1=2*t0,lam2=0.4*t0)
    out.append(('Zener truth De=2 s=6',ml,py))
    # (b) NHKV at 3 grid points
    for k in [0, 30, len(Gg)*len(Mg)-1]:
        gi,mi=k//len(Mg),k%len(Mg)
        ml=A[:,1+k]; py=simulate(t,R0,R0/6,Gg[gi],Mg[mi],stress=1)
        out.append((f'NHKV G={Gg[gi]:.0f} mu={Mg[mi]:.4f}',ml,py))
    print(f"{'case':34s} {'max|dR|':>10} {'RMS dR':>10} {'verdict':>10}")
    ok=True
    for lab,ml,py in out:
        d=np.abs(ml-py); mx,rms=np.nanmax(d),np.sqrt(np.nanmean(d**2))
        v='PASS' if mx<2e-3 else 'FAIL'; ok&=(mx<2e-3)
        print(f"{lab:34s} {mx:>10.2e} {rms:>10.2e} {v:>10}")
    return ok

if __name__=='__main__':
    import time
    ok=validate()
    t=np.loadtxt(f'{refdir}/imr2_t.csv'); R0=225e-6
    n=20; st=time.time()
    for _ in range(n): simulate(t,R0,R0/6,600.,0.02,stress=1)
    dt=(time.time()-st)/n
    print(f"\nspeed: {dt*1000:.1f} ms/solve  ({1/dt:.0f} solves/s)  vs MATLAB ~400-1000 ms")
    print("VALIDATION", "PASSED" if ok else "FAILED")
