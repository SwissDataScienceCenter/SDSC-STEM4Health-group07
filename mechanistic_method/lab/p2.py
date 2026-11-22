import sys,os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import statm
from p import *
def theta_model2(th,trm,tem):
    m=th.mean(0); s=th.std(0); Zt=(th-m)/s; n=trm.sum(); best=None
    for sc in [.003,.01,.03,.1,.3,1]:
        Kt=sc*K0[np.ix_(trm,trm)]; Hm=Kt@np.linalg.inv(Kt+np.eye(n)); loo=Zt-(Zt-Hm@Zt)/(1-np.diag(Hm))[:,None]; e=((loo-Zt)**2).mean()
        if best is None or e<best[0]: best=(e,sc,loo)
    _,sc,loo=best; pred=sc*K0[np.ix_(tem,trm)]@np.linalg.solve(sc*K0[np.ix_(trm,trm)]+np.eye(n),Zt)
    return pred*s+m,loo*s+m,np.cov(((Zt-loo)*s).T)
def particle2(ode='m21_re8',N=400,resid='evidence',names=None,seed=0,tag='',kappa=1.0,nosub=True):
    model=V[ode](); F=Fitter(model); xs=np.load(f'{LAB}/ode/{ode}_xs.npy'); rng=np.random.default_rng(seed); names=names or (statm.ALLG+['base25'])
    pos={r:i for i,r in enumerate(np.where(tr)[0])}; P=np.zeros((tr.sum(),13,4)); Pt=None; folds=folds7(tr)+[np.where(test)[0]]; sig=np.array([.05,.1,.15,.3,.08])
    fn={'evidence':statm.m_evidence,'search':statm.m_kernel_search}[resid]
    for f,ho in enumerate(folds):
        m=tr.copy()
        if f<7: m[ho]=False
        te=np.zeros_like(tr); te[ho]=True; m0=stats0(m); p=F.params(xs[f],np.ones(len(Y),bool))[0]
        pm={k:(v[m] if np.ndim(v) else v) for k,v in p.items()}
        th=fit_indiv(model,pm,m,m0,sig)
        o=model.simulate(with_theta(pm,th),sub(A,m),Y[m][:,0],m0); r=np.log(o[:,1:])-np.log(Y[m][:,1:]); bias=np.median(r,0); so=(r-bias).std(0)*1.06
        mu_te,mu_loo,Sg=theta_model2(th,m,te); Lc=np.linalg.cholesky(Sg*kappa**2+1e-10*np.eye(Q))
        def predict(rows,mu_th):
            nh=len(rows); TH=np.repeat(mu_th,N,axis=0)+rng.standard_normal((nh*N,Q))@Lc.T; rep=np.repeat(rows,N)
            pr={k:(v[rep] if np.ndim(v) else v) for k,v in p.items()}; os_=model.simulate(with_theta(pr,TH),{k:v[rep] for k,v in A.items()},Y[rep][:,0],m0)
            mu=(np.log(os_[:,1:])-bias).reshape(nh,N,13,4); w=np.ones((nh,N)); ys=mu+rng.standard_normal(mu.shape)*so; flh=fl[rows]
            for k,thr in THR.items():
                lt=np.log(thr); z=(lt-mu[:,:,k-1,2])/so[k-1,2]; low=flh[:,FLAGCOL[k]][:,None]>0
                w*=np.maximum(np.where(low,norm.cdf(z),norm.sf(z)),1e-12)
                ys[:,:,k-1,2]=mu[:,:,k-1,2]+so[k-1,2]*truncnorm.rvs(np.where(low,-np.inf,z),np.where(low,z,np.inf),random_state=rng)
            yv=np.exp(ys); out=np.zeros((nh,13,4))
            for i in range(nh):
                for k in range(13):
                    for t_ in range(4):
                        y=yv[i,:,k,t_]; ww=w[i]/y; o_=np.argsort(y); c=np.cumsum(ww[o_]); out[i,k,t_]=y[o_][np.searchsorted(c,0.5*c[-1])]
            return out
        out_te=predict(np.where(te)[0],mu_te)
        if resid:
            out_tr=predict(np.where(m)[0],mu_loo); R=np.log(Y[m][:,1:])-np.log(out_tr); R=R-np.median(R,0)
            mu_r,_=fn(m,te,k4_to_52(np.clip(R,-0.5,0.5)),names=names); mr=f52_to_k4(mu_r)
            if nosub: mr[:,:,2]=0
            out_te=out_te*np.exp(mr)
        if f<7: P[[pos[r] for r in ho]]=out_te
        else: Pt=out_te
    nm=f'p2_{ode}_{resid}{tag}'
    return register(nm,'hybrid',f'Bayesian particle ODE {ode} (random effects from signals, feed-rule likelihood) + kernel model ({resid} weights) on its centred log residuals'+(' (cells and byproduct only)' if nosub else ''),P,Ytr,Pt)
if __name__=='__main__':
    a=sys.argv[1:]; particle2(a[0],resid=a[1],tag=a[2] if len(a)>2 else '',nosub=(a[2]!='_all') if len(a)>2 else True)
