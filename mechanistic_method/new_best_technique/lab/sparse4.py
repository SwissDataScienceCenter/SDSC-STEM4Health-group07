"""Small models, version 4 (new file; sparse.py untouched).
Adds the exact sample time: cells and byproduct change fast early in the run, and samples are taken up to +-0.1 day off
schedule, so a sample taken late reads higher (or lower) for a reason that has nothing to do with the batch.
 rank1t : cells. log X_k = typical_k + offset_batch + g_k * (t_k - k); g_k fitted per day, offset from 4 signals as before.
 perdayt: byproduct. per-day ridge on the 3 signals plus that day's sample-time offset (and optionally the interval)."""
import sys,os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from sparse import *
TAU=tt[:,1:]-np.arange(1,14); DT=np.diff(tt,axis=1)-1.0
def wins(L,clip=3.0):
    ref=np.median(L,0); mad=1.4826*np.median(np.abs(L-ref),0); return np.clip(L,ref-clip*mad,ref+clip*mad)
def rank1t(fit,te,cols,targets,lam=0.03,lamg=0.03,iters=3,gdays=13):
    X=np.c_[tuple(FE[c] for c in cols)]; Lc=wins(np.log(Y[fit][:,1:][:,:,targets])); ref=Lc.mean(0); dev=Lc-ref; tau=TAU[fit]-TAU[fit].mean(0); nt=len(targets)
    g=np.zeros((13,nt)); d=dev.mean((1,2))
    for _ in range(iters):
        for k in range(gdays):
            for j in range(nt):
                r=dev[:,k,j]-d; g[k,j]=(tau[:,k]*r).sum()/((tau[:,k]**2).sum()+lamg*len(fit)*tau[:,k].var())
        d=(dev-g[None]*tau[:,:,None]).mean((1,2))
    co,dl=ridge_fit(X[fit],d,lam); dh=np.c_[X[te],np.ones(len(te))]@co
    res=dev-g[None]*tau[:,:,None]-dl[:,None,None]; sd=1.4826*np.median(np.abs(res-np.median(res,0)),0)
    taut=TAU[te]-TAU[fit].mean(0); return ref+dh[:,None,None]+g[None]*taut[:,:,None],sd
def perdayt(fit,te,cols,targets,lam=0.03,usedt=False):
    X=np.c_[tuple(FE[c] for c in cols)]; Lc=wins(np.log(Y[fit][:,1:][:,:,targets])); mu=np.zeros((len(te),13,len(targets))); sd=np.zeros((13,len(targets)))
    for k in range(13):
        ex=[TAU[:,k]]+([DT[:,k]] if usedt else []); ex=np.c_[tuple(ex)]; ex=(ex-ex[fit].mean(0))/ex[fit].std(0); Xk=np.c_[X,ex]
        for j in range(len(targets)):
            co,loo=ridge_fit(Xk[fit],Lc[:,k,j],lam); mu[:,k,j]=np.c_[Xk[te],np.ones(len(te))]@co; r=Lc[:,k,j]-loo; sd[k,j]=1.4826*np.median(np.abs(r-np.median(r)))
    return mu,sd
if __name__=='__main__':
    np.set_printoptions(precision=1,suppress=True,linewidth=220)
    z=np.load(f'{LAB}/oof/b09_hyb_stat_bayesresid_smoothcells.npz'); B9=np.zeros((n,13,4)); B9[tr]=z['oof']; B9[test]=z['test']
    def rep(nm,P,tg):
        e=score(P,tg); eb=score(np.sqrt(P*B9[:,:,tg]),tg); print('%-34s alone %s blend %s |'%(nm,np.round(e.mean(0),3),np.round(eb.mean(0),3)),e.mean(1) if len(tg)>1 else e[:,0])
    C=['lx0','lp0','V0','fbL']; rep('cells rank1 (v2)',run(rank1,[0,1],cols=C,shrink=1.0),[0,1])
    for lg in (0.01,0.03,0.1,0.3,1.0): rep(f'cells rank1 + time, lamg {lg}',run(rank1t,[0,1],cols=C,lamg=lg),[0,1])
    rep('cells rank1 + time days1-6',run(rank1t,[0,1],cols=C,gdays=6),[0,1])
    Pc=['base25','V0','lp0']; rep('byp perday (v2)',run(perday,[3],cols=Pc),[3]); rep('byp perday + time',run(perdayt,[3],cols=Pc),[3]); rep('byp perday + time + interval',run(perdayt,[3],cols=Pc,usedt=True),[3])
    rep('byp perday + time lam .1',run(perdayt,[3],cols=Pc,lam=0.1),[3])
