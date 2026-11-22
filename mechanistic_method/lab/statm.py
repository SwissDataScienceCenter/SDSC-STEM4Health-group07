import sys,os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import *
from scipy.optimize import minimize
bs,A,Y,tt,test=load(); tr=~test; G,fl,D=feature_groups(A,Y,tt)
Ytr=Y[tr][:,1:]; Y52=k4_to_52(np.log(Y[:,1:]))        # (61,52), nan for test
CORE=['lx0','ls0','lp0','vol0','dVr_e','volL','t','fb','fc','fs_m','fs_l']
ALLG=list(G)
_cum=np.cumsum(D['dVr'],1); G['base25']=(_cum[:,4]-_cum[:,1])[:,None]; G['base15']=(_cum[:,4]-_cum[:,0])[:,None]
CORE2=['lx0','ls0','lp0','vol0','base25','volL','t','fb','fc','fs_m','fs_l']; CORE3=CORE+['base25']
def kern(z): z=sd_(z); return z@z.T/z.shape[1]
KALL={g:kern(G[g]) for g in G}
def Xmat(gs): return np.c_[tuple(sd_(G[g]) for g in gs)]
# ---------- kernel ridge utils
def loo_pred(Ktr,Yt,alpha=1.0):
    n=len(Yt); Kc=Ktr+1e4; H=Kc@np.linalg.inv(Kc+alpha*np.eye(n)); return Yt-(Yt-H@Yt)/(1-np.diag(H))[:,None]
def fitpred(K,trm,tem,Yt,alpha=1.0):
    al=np.linalg.solve(K[np.ix_(trm,trm)]+1e4+alpha*np.eye(trm.sum()),Yt); return (K[np.ix_(tem,trm)]+1e4)@al
def mapel(P,Yt): return (np.abs(np.exp(P)-np.exp(Yt))/np.exp(Yt)).mean()*100
def search(K,trm,Yt,grid,names,passes=3,tol=0.01):
    Ks={g:K[g][np.ix_(trm,trm)] for g in names}; sc=lambda w: mapel(loo_pred(sum(w[g]**2*Ks[g] for g in names),Yt),Yt)
    w={g:0.0 for g in names}; best=sc(w)
    for _ in range(passes):
        ch=False
        for g in names:
            for v in grid:
                if v==w[g]: continue
                w2=dict(w); w2[g]=v; s=sc(w2)
                if s<best-tol: best,w,ch=s,w2,True
        if not ch: break
    return w
BL={'whole':{'cells':np.r_[0:26],'substrate':np.r_[26:39],'byproduct':np.r_[39:52]},
    'split':{'ce':np.r_[0:5,13:18],'cl':np.r_[5:13,18:26],'se':np.r_[26:32],'sl':np.r_[32:39],'be':np.r_[39:45],'bl':np.r_[45:52]},
    'target':{'vcd':np.r_[0:13],'tcd':np.r_[13:26],'substrate':np.r_[26:39],'byproduct':np.r_[39:52]}}
GRIDS=[[0,.15,.3,.6,1.2,2.4],[0,.25,.5,1,2]]
def m_kernel_search(trm,tem,Yt,names=CORE,blocks=('whole','split'),grids=(0,1),K=KALL):
    mus=[];sds=[]
    for gi in grids:
        for bname in blocks:
            mu=np.zeros((tem.sum(),52)); sd=np.zeros(52)
            for bn,sl in BL[bname].items():
                w=search(K,trm,Yt[:,sl],GRIDS[gi],names); Kw=sum(w[g]**2*K[g] for g in names)
                mu[:,sl]=fitpred(Kw,trm,tem,Yt[:,sl]); sd[sl]=(loo_pred(Kw[np.ix_(trm,trm)],Yt[:,sl])-Yt[:,sl]).std(0)
            mus.append(mu); sds.append(sd)
    return np.mean(mus,0),np.mean(sds,0)
def m_kernel_fixed(trm,tem,Yt,names=CORE,w=1.0,alpha=1.0):
    Kw=sum(w**2*KALL[g] for g in names)
    return fitpred(Kw,trm,tem,Yt,alpha),(loo_pred(Kw[np.ix_(trm,trm)],Yt,alpha)-Yt).std(0)
def m_kernel_alpha(trm,tem,Yt,names=CORE):
    """equal group weights, one global scale per block chosen by LOO"""
    mu=np.zeros((tem.sum(),52)); sd=np.zeros(52); K0=sum(KALL[g] for g in names)
    for bn,sl in BL['whole'].items():
        best=None
        for s in [.003,.01,.03,.1,.3,1,3]:
            e=mapel(loo_pred(s*K0[np.ix_(trm,trm)],Yt[:,sl]),Yt[:,sl])
            if best is None or e<best[0]: best=(e,s)
        s=best[1]; mu[:,sl]=fitpred(s*K0,trm,tem,Yt[:,sl]); sd[sl]=(loo_pred(s*K0[np.ix_(trm,trm)],Yt[:,sl])-Yt[:,sl]).std(0)
    return mu,sd
def m_evidence(trm,tem,Yt,names=CORE,blocks='whole',l1=0.0):
    """GP empirical Bayes: group variances + noise by maximising the marginal likelihood (no grid search)"""
    mu=np.zeros((tem.sum(),52)); sd=np.zeros(52); n=trm.sum()
    Ks=np.stack([KALL[g][np.ix_(trm,trm)] for g in names]); Kf=np.stack([KALL[g] for g in names])
    for bn,sl in BL[blocks].items():
        Yb=Yt[:,sl]; m=Yb.mean(0); s=Yb.std(0); Z=(Yb-m)/s; q=Z.shape[1]
        def nll(th):
            v=np.exp(th[:-1]); K=np.tensordot(v,Ks,1)+np.exp(th[-1])*np.eye(n)
            try: L=np.linalg.cholesky(K)
            except np.linalg.LinAlgError: return 1e10
            a=np.linalg.solve(L,Z); return 0.5*(a**2).sum()+q*np.log(np.diag(L)).sum()+l1*np.sqrt(v).sum()*q
        r=minimize(nll,np.r_[np.full(len(names),np.log(.05)),np.log(.5)],method='L-BFGS-B',bounds=[(-12,3)]*len(names)+[(-4,2)])
        v=np.exp(r.x[:-1]); nz=np.exp(r.x[-1]); K=np.tensordot(v,Kf,1)
        al=np.linalg.solve(K[np.ix_(trm,trm)]+nz*np.eye(n),Z); mu[:,sl]=(K[np.ix_(tem,trm)]@al)*s+m
        Kt=K[np.ix_(trm,trm)]; H=Kt@np.linalg.inv(Kt+nz*np.eye(n)); loo=Z-(Z-H@Z)/(1-np.diag(H))[:,None]; sd[sl]=((loo-Z)*s).std(0)
    return mu,sd
def m_mean(trm,tem,Yt): return np.tile(Yt.mean(0),(tem.sum(),1)),Yt.std(0)
def m_sklearn(make,gs=CORE):
    def f(trm,tem,Yt):
        X=Xmat(gs); m=make().fit(X[trm],Yt); P=m.predict(X[tem]); return P,(m.predict(X[trm])-Yt).std(0)*1.15
    return f
def m_ridgecv(trm,tem,Yt,gs=ALLG):
    X=Xmat(gs); K=X@X.T; mu=np.zeros((tem.sum(),52)); sd=np.zeros(52)
    for bn,sl in BL['whole'].items():
        best=None
        for a in [10,30,100,300,1000,3000]:
            e=mapel(loo_pred(K[np.ix_(trm,trm)],Yt[:,sl],a),Yt[:,sl])
            if best is None or e<best[0]: best=(e,a)
        a=best[1]; mu[:,sl]=fitpred(K,trm,tem,Yt[:,sl],a); sd[sl]=(loo_pred(K[np.ix_(trm,trm)],Yt[:,sl],a)-Yt[:,sl]).std(0)
    return mu,sd
def m_pls(trm,tem,Yt,gs=CORE):
    from sklearn.cross_decomposition import PLSRegression
    X=Xmat(gs); mu=np.zeros((tem.sum(),52)); sd=np.zeros(52); idx=np.where(trm)[0]
    for bn,sl in BL['whole'].items():
        best=None
        for nc in [1,2,3,4,6]:
            P=np.zeros_like(Yt[:,sl])
            for f in range(6):
                ho=np.arange(len(idx))[f::6]; ti=np.setdiff1d(np.arange(len(idx)),ho)
                P[ho]=PLSRegression(nc,scale=False).fit(X[idx[ti]],Yt[ti][:,sl]).predict(X[idx[ho]])
            e=mapel(P,Yt[:,sl])
            if best is None or e<best[0]: best=(e,nc,(P-Yt[:,sl]).std(0))
        mu[:,sl]=PLSRegression(best[1],scale=False).fit(X[trm],Yt[:,sl]).predict(X[tem]); sd[sl]=best[2]
    return mu,sd
def m_rrr(trm,tem,Yt,names=CORE,rank=3):
    """kernel search prediction projected on top principal components of the fitted values (reduced rank)"""
    mu,sd=m_kernel_search(trm,tem,Yt,blocks=('whole',),grids=(1,))
    mtr,_=m_kernel_search(trm,trm,Yt,blocks=('whole',),grids=(1,))
    m=Yt.mean(0); s=Yt.std(0); U,S_,Vt=np.linalg.svd((mtr-m)/s,full_matrices=False); Pj=Vt[:rank].T@Vt[:rank]
    return ((mu-m)/s)@Pj*s+m,sd
def oof(fn):
    P=np.zeros((tr.sum(),13,4)); pos={r:i for i,r in enumerate(np.where(tr)[0])}; MU=np.zeros((tr.sum(),52)); SD=np.zeros((tr.sum(),52))
    for ho in folds7(tr):
        m=tr.copy(); m[ho]=False; te=np.zeros_like(tr); te[ho]=True
        mu,sd=fn(m,te,Y52[m]); ii=[pos[r] for r in ho]; MU[ii]=mu; SD[ii]=sd
    mut,sdt=fn(tr,test,Y52[tr])
    return MU,SD,mut,sdt
def run(name,family,desc,fn,c=1.0,trunc=True,sdscale=0.7,save_mu=True):
    t0=time.time(); MU,SD,mut,sdt=oof(fn)
    P=post(f52_to_k4(MU),f52_to_k4(SD),fl[tr],c,trunc,sdscale); Pt=post(f52_to_k4(mut),f52_to_k4(np.tile(sdt,(test.sum(),1))),fl[test],c,trunc,sdscale)
    if save_mu: np.savez(f'{LAB}/oof/{name}__mu.npz',MU=MU,SD=SD,mut=mut,sdt=sdt)
    return register(name,family,desc,P,Ytr,Pt,extra=dict(secs=round(time.time()-t0,1)))
if __name__=='__main__':
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.linear_model import MultiTaskElasticNetCV
    from sklearn.kernel_ridge import KernelRidge
    which=sys.argv[1:]
    R={
     's01_mean':('baseline','per-day mean profile (log scale), no batch information',m_mean,dict(c=0,trunc=False)),
     's02_mean_shift':('baseline','per-day mean + lognormal MAPE shift',m_mean,dict(trunc=False)),
     's03_mean_trunc':('baseline','per-day mean + shift + substrate feed-rule truncation',m_mean,dict(sdscale=1.0)),
     's04_ridge_all':('statistical','ridge on all raw features, one penalty per target block',m_ridgecv,{}),
     's05_kernel_equal':('statistical','grouped kernel ridge, equal group weights, one scale per block',m_kernel_alpha,{}),
     's06_kernel_search_raw':('statistical','grouped kernel ridge, LOO weight search, no output step',m_kernel_search,dict(c=0,trunc=False)),
     's07_kernel_search_shift':('statistical','s06 + lognormal shift',m_kernel_search,dict(trunc=False)),
     's08_kernel_search':('statistical','s06 + shift + rule truncation (= solution.py)',m_kernel_search,{}),
     's09_kernel_search_allgroups':('statistical','s08 with all 19 feature groups in the search',lambda a,b,c:m_kernel_search(a,b,c,names=ALLG),{}),
     's10_kernel_search_pertarget':('statistical','s08 with separate weights per target (4 blocks)',lambda a,b,c:m_kernel_search(a,b,c,blocks=('target',)),{}),
     's11_evidence':('statistical','GP empirical Bayes: group variances by marginal likelihood, 3 blocks',m_evidence,{}),
     's12_evidence_split':('statistical','s11 with early/late split blocks',lambda a,b,c:m_evidence(a,b,c,blocks='split'),{}),
     's13_evidence_all':('statistical','s11 with all 19 groups',lambda a,b,c:m_evidence(a,b,c,names=ALLG),{}),
     's14_pls':('statistical','PLS regression, components by inner CV',m_pls,{}),
     's15_extratrees':('statistical','ExtraTrees multi-output on core features',m_sklearn(lambda:ExtraTreesRegressor(400,min_samples_leaf=3,max_features=.3,random_state=0)),{}),
     's16_rf':('statistical','random forest multi-output',m_sklearn(lambda:RandomForestRegressor(400,min_samples_leaf=3,max_features=.3,random_state=0)),{}),
     's17_knn':('statistical','k-nearest batches (k=10, distance weighted) on core features',m_sklearn(lambda:KNeighborsRegressor(10,weights='distance')),{}),
     's18_enet':('statistical','multi-task elastic net, penalty by inner CV',m_sklearn(lambda:MultiTaskElasticNetCV(l1_ratio=[.2,.5,.8],cv=6,alphas=30,max_iter=5000)),{}),
     's19_rbf':('statistical','RBF kernel ridge on core features',m_sklearn(lambda:KernelRidge(alpha=.3,kernel='rbf',gamma=1/400)),{}),
     's20_rrr':('statistical','kernel search projected to rank-3 (shared latent batch factors)',m_rrr,{}),
     's22_search_base25':('statistical','s08 with the base-addition signal as one windowed value (days 2-5) instead of 4 daily values',lambda a,b,c:m_kernel_search(a,b,c,names=CORE2),{}),
     's23_evidence_base25':('statistical','s11 with windowed base-addition signal',lambda a,b,c:m_evidence(a,b,c,names=CORE2),{}),
     's24_search_core_plus_base25':('statistical','s08 plus windowed base-addition signal',lambda a,b,c:m_kernel_search(a,b,c,names=CORE3),{}),
     's25_search_all_plus_base25':('statistical','s09 plus windowed base-addition signal',lambda a,b,c:m_kernel_search(a,b,c,names=ALLG+['base25']),{}),
     's26_evidence_all_plus_base25':('statistical','s13 plus windowed base-addition signal',lambda a,b,c:m_evidence(a,b,c,names=ALLG+['base25']),{}),
     's21_evidence_target':('statistical','s11 with one block per target',lambda a,b,c:m_evidence(a,b,c,blocks='target'),{}),
    }
    for nm in (which or list(R)):
        fam,desc,fn,kw=R[nm]; run(nm,fam,desc,fn,**kw)
