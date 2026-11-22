import sys,os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from variants import *
from scipy.sparse import lil_matrix
from scipy.stats import norm, truncnorm
REP=['mumax','mS','Yov','kc']; Q=len(REP)+1
CORE=['lx0','ls0','lp0','vol0','dVr_e','volL','t','fb','fc','fs_m','fs_l']
def kern(z): z=sd_(z); return z@z.T/z.shape[1]
_cum=np.cumsum(D['dVr'],1); G['base25']=(_cum[:,4]-_cum[:,1])[:,None]
CORE=['lx0','ls0','lp0','vol0','base25','volL','t','fb','fc','fs_m','fs_l']
K0=sum(kern(G[g]) for g in CORE)
def with_theta(p,th):
    q=dict(p)
    for j,nm in enumerate(REP): q[nm]=p[nm]*np.exp(th[:,j])
    q['dx0']=th[:,Q-1]; return q
def fit_indiv(model,p,rows,m0,sig,nfev=40):
    n=rows.sum(); Ar=sub(A,rows); Yr=Y[rows]
    def res(x):
        th=x.reshape(n,Q); out=model.simulate(with_theta(p,th),Ar,Yr[:,0],m0)
        r=np.log(out[:,1:])-np.log(Yr[:,1:]); via=5*(r[:,:,0]-r[:,:,1])
        return np.c_[r.reshape(n,-1)/0.07,via/0.07,th/sig].ravel()
    m=13*5+Q; Sp=lil_matrix((n*m,n*Q),dtype=int)
    for b in range(n): Sp[b*m:(b+1)*m,b*Q:(b+1)*Q]=1
    r=least_squares(res,np.zeros(n*Q),jac_sparsity=Sp,loss='soft_l1',f_scale=2.0,max_nfev=nfev,x_scale=0.05)
    return r.x.reshape(n,Q)
def theta_model(th,trm,tem,how):
    """predict random effects of held-out batches from process signals; returns mean (nte,Q) and residual cov"""
    m=th.mean(0); s=th.std(0); Zt=(th-m)/s; n=trm.sum()
    if how=='none': return np.tile(m,(tem.sum(),1)),np.cov(th.T)
    if how=='z7':
        Z,_=zsets('z7'); Xa=np.c_[Z[trm]]; lam=10.0
        Hm=Xa@np.linalg.solve(Xa.T@Xa+lam*np.eye(Xa.shape[1]),Xa.T); fit=Hm@Zt; loo=Zt-(Zt-fit)/(1-np.diag(Hm))[:,None]
        pred=Z[tem]@np.linalg.solve(Xa.T@Xa+lam*np.eye(Xa.shape[1]),Xa.T@Zt)
    else:
        best=None
        for sc in [.003,.01,.03,.1,.3,1]:
            Kt=sc*K0[np.ix_(trm,trm)]; Hm=Kt@np.linalg.inv(Kt+np.eye(n)); loo=Zt-(Zt-Hm@Zt)/(1-np.diag(Hm))[:,None]; e=((loo-Zt)**2).mean()
            if best is None or e<best[0]: best=(e,sc,loo)
        _,sc,loo=best; pred=sc*K0[np.ix_(tem,trm)]@np.linalg.solve(sc*K0[np.ix_(trm,trm)]+np.eye(n),Zt)
    R=(Zt-loo)*s
    return pred*s+m,np.cov(R.T)
def particle(ode='m01_base',how='kernel',N=400,kappa=1.0,cal=True,flags=True,seed=0,tag=''):
    model=V[ode](); F=Fitter(model); xs=np.load(f'{LAB}/ode/{ode}_xs.npy'); rng=np.random.default_rng(seed)
    pos={r:i for i,r in enumerate(np.where(tr)[0])}; P=np.zeros((tr.sum(),13,4)); Pt=None; folds=folds7(tr)+[np.where(test)[0]]
    sig=np.array([.05,.1,.15,.3,.08])
    for f,ho in enumerate(folds):
        m=tr.copy()
        if f<7: m[ho]=False
        te=np.zeros_like(tr); te[ho]=True; m0=stats0(m); p=F.params(xs[f],np.ones(len(Y),bool))[0]
        pm={k:(v[m] if np.ndim(v) else v) for k,v in p.items()}
        th=fit_indiv(model,pm,m,m0,sig)
        o=model.simulate(with_theta(pm,th),sub(A,m),Y[m][:,0],m0); r=np.log(o[:,1:])-np.log(Y[m][:,1:])
        bias=np.median(r,0) if cal else np.zeros((13,4)); so=(r-bias).std(0)*1.06
        mu_th,Sg=theta_model(th,m,te,how); Lc=np.linalg.cholesky(Sg*kappa**2+1e-10*np.eye(Q))
        nh=len(ho); TH=np.repeat(mu_th,N,axis=0)+rng.standard_normal((nh*N,Q))@Lc.T
        rep=np.repeat(np.where(te)[0],N); pr={k:(v[rep] if np.ndim(v) else v) for k,v in p.items()}
        Ar={k:v[rep] for k,v in A.items()}; os_=model.simulate(with_theta(pr,TH),Ar,Y[rep][:,0],m0)
        mu=(np.log(os_[:,1:])-bias).reshape(nh,N,13,4); w=np.ones((nh,N)); ys=mu+rng.standard_normal(mu.shape)*so
        flh=fl[ho]
        if flags:
            for k,thr in THR.items():
                lt=np.log(thr); z=(lt-mu[:,:,k-1,2])/so[k-1,2]; low=flh[:,FLAGCOL[k]][:,None]>0
                pk=np.where(low,norm.cdf(z),norm.sf(z)); w*=np.maximum(pk,1e-12)
                a=np.where(low,-np.inf,z); b=np.where(low,z,np.inf)
                ys[:,:,k-1,2]=mu[:,:,k-1,2]+so[k-1,2]*truncnorm.rvs(a,b,random_state=rng)
        yv=np.exp(ys); out=np.zeros((nh,13,4))
        for i in range(nh):
            for k in range(13):
                for t_ in range(4):
                    y=yv[i,:,k,t_]; ww=w[i]/y; o_=np.argsort(y); c=np.cumsum(ww[o_]); out[i,k,t_]=y[o_][np.searchsorted(c,0.5*c[-1])]
        if f<7: P[[pos[r] for r in ho]]=out
        else: Pt=out
    nm=f'p_{ode}_{how}{tag}'
    return register(nm,'mechanistic-bayes',f'ODE {ode} with per-batch random effects (growth, uptake, overflow yield, re-consumption, inoculum) predicted from process signals ({how}); {N} particles weighted by substrate feed-rule likelihood; calibration={cal}, flags={flags}, kappa={kappa}',P,Ytr,Pt)
if __name__=='__main__':
    a=sys.argv[1:]; particle(a[0],a[1],kappa=float(a[2]) if len(a)>2 else 1.0,cal=(a[3]!='nocal') if len(a)>3 else True,flags=(a[4]!='noflags') if len(a)>4 else True,tag=a[5] if len(a)>5 else '')
