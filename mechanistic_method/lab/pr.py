import sys,os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import statm
from p import *
def theta_loo(th,trm,tem,how='kernel'):
    m=th.mean(0); s=th.std(0); Zt=(th-m)/s; n=trm.sum(); best=None
    for sc in [.003,.01,.03,.1,.3,1]:
        Kt=sc*K0[np.ix_(trm,trm)]; Hm=Kt@np.linalg.inv(Kt+np.eye(n)); loo=Zt-(Zt-Hm@Zt)/(1-np.diag(Hm))[:,None]; e=((loo-Zt)**2).mean()
        if best is None or e<best[0]: best=(e,sc,loo)
    _,sc,loo=best; pred=sc*K0[np.ix_(tem,trm)]@np.linalg.solve(sc*K0[np.ix_(trm,trm)]+np.eye(n),Zt)
    return pred*s+m,loo*s+m
def pr(ode='m01_base',res='search',sig=(.05,.1,.15,.3,.08),tag='',shrink=1.0):
    model=V[ode](); F=Fitter(model); xs=np.load(f'{LAB}/ode/{ode}_xs.npy'); fn={'search':statm.m_kernel_search,'evidence':statm.m_evidence,'equal':statm.m_kernel_alpha}[res]
    pos={r:i for i,r in enumerate(np.where(tr)[0])}; MU=np.zeros((tr.sum(),52)); SD=np.zeros((tr.sum(),52)); folds=folds7(tr)+[np.where(test)[0]]
    for f,ho in enumerate(folds):
        m=tr.copy()
        if f<7: m[ho]=False
        te=np.zeros_like(tr); te[ho]=True; m0=stats0(m); p=F.params(xs[f],np.ones(len(Y),bool))[0]
        sel=lambda rows:{k:(v[rows] if np.ndim(v) else v) for k,v in p.items()}
        th=fit_indiv(model,sel(m),m,m0,np.array(sig)); pred,loo=theta_loo(th,m,te)
        o_tr=model.simulate(with_theta(sel(m),loo*shrink),sub(A,m),Y[m][:,0],m0); o_te=model.simulate(with_theta(sel(te),pred*shrink),sub(A,te),Y[te][:,0],m0)
        l_tr=k4_to_52(np.log(o_tr[:,1:])); l_te=k4_to_52(np.log(o_te[:,1:]))
        mu,sd=fn(m,te,statm.Y52[m]-l_tr)
        if f<7: ii=[pos[r] for r in ho]; MU[ii]=mu+l_te; SD[ii]=sd
        else: mut=mu+l_te; sdt=sd
    P=post(f52_to_k4(MU),f52_to_k4(SD),fl[tr]); Pt=post(f52_to_k4(mut),f52_to_k4(np.tile(sdt,(test.sum(),1))),fl[test])
    nm=f'pr_{ode}_{res}{tag}'; np.savez(f'{LAB}/oof/{nm}__mu.npz',MU=MU,SD=SD,mut=mut,sdt=sdt)
    return register(nm,'hybrid',f'ODE {ode} with per-batch kinetic parameters predicted from process signals (two-stage), then kernel model ({res}) on log residuals, shift + rule truncation',P,Ytr,Pt)
if __name__=='__main__':
    a=sys.argv[1:]; pr(a[0],a[1])
