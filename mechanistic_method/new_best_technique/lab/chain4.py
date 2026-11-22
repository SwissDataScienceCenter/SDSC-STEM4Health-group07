"""Substrate simulator, version 4 experiments (new file; chain.py untouched).
Differences from chain.py:
  - selective shrinkage: the physical terms of each step (yesterday's substrate, feed B, sample interval, bolus) are
    almost unpenalised, the helper covariates (stage-1 predictions, unlogged volume) get their own, stronger penalty
  - optional pooling of neighbouring days when fitting a step (weights w on days k-1 and k+1) to cut coefficient noise
  - optional persistent batch effect in the innovations (a share rho2 of each step's variance is common to all steps)"""
import sys,os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from chain import *
D4=dict(DEF,vb=1,lamH=1e-4,lamS=3e-2,pool=0.0,rho2=0.0,N=6000)
def nhard(k): return 4 if k in BOL else 3
def ridge2(X,y,lamvec,w=None):
    n_=len(y); w=np.ones(n_) if w is None else w; m=np.average(X,0,weights=w); s=np.sqrt(np.average((X-m)**2,0,weights=w)); s[s<1e-12]=1
    Z=np.c_[(X-m)/s,np.ones(n_)]; R=w.sum()*np.diag(np.r_[lamvec,0]); Ai=np.linalg.inv((Z*w[:,None]).T@Z+R); co=Ai@(Z*w[:,None]).T@y
    h=np.einsum('ij,jk,ik->i',Z,Ai,Z)*w; r=(y-Z@co)/(1-h); return (m,s,co),r
def fit_step(k,fit,PB,cfg):
    X=fS(k,Y[fit,k,2],Y[fit,k,3],fit,PB,cfg,Y[fit,max(k-1,0),2]); y=Y[fit,k+1,2]; lamvec=np.r_[np.full(nhard(k),cfg['lamH']),np.full(X.shape[1]-nhard(k),cfg['lamS'])]
    mod,r=ridge2(X,y,lamvec); sd=1.4826*np.median(np.abs(r-np.median(r))); return mod,sd,r
def simulate4(fit,te,PB,cfg):
    rng=np.random.default_rng(cfg['seed']); N=cfg['N']; m=len(te); S=np.repeat(Y[te,0,2][:,None],N,1); P=np.repeat(Y[te,0,3][:,None],N,1)
    W=np.ones((m,N))/N; Sp=S; path=np.zeros((m,N,13)); idx=np.repeat(te,N); u=rng.standard_normal(m*N); rho=np.sqrt(cfg['rho2'])
    for k in range(13):
        mS,sd,_=fit_step(k,fit,PB,cfg); sd*=cfg['sdS']
        mu=pred(mS,fS(k,S.ravel(),P.ravel(),idx,PB,cfg,Sp.ravel()))+rho*sd*u; sde=sd*np.sqrt(1-cfg['rho2'])
        if cfg['flags'] and (k+1) in THR:
            thr=THR[k+1]; low=np.repeat(fl[te,FLAGCOL[k+1]]>0,N); c=norm.cdf((thr-mu)/sde); pm=np.where(low,c,1-c); q=rng.uniform(size=m*N); q=np.where(low,q*c,c+q*(1-c))
            S1=mu+sde*norm.ppf(np.clip(q,1e-12,1-1e-12)); S1=np.where(low,np.minimum(S1,thr),np.maximum(S1,thr)); W=W*np.maximum(pm,1e-300).reshape(m,N); W/=W.sum(1,keepdims=True)
        else: S1=mu+sde*rng.standard_normal(m*N)
        Sp=S; S=np.maximum(S1,0.03).reshape(m,N); path[:,:,k]=S
    out=np.zeros((m,13))
    for i in range(m):
        for k in range(13): out[i,k]=wmed(path[i,:,k],W[i]/path[i,:,k])
    return out,1/(W**2).sum(1)
def chain4(base,**kw):
    cfg={**D4,**kw}; PB=base_preds(base); out=PB[:,1:,2].copy(); trn=np.where(tr)[0]; em=1e9
    for te in folds7(tr): o,e=simulate4(np.setdiff1d(trn,te),te,PB,cfg); out[te]=o; em=min(em,e.min())
    te=np.where(test)[0]; out[te],_=simulate4(trn,te,PB,cfg); return out,em
if __name__=='__main__':
    np.set_printoptions(precision=1,suppress=True,linewidth=220)
    base='b12_stage1'; PB=base_preds(base); YS=Ytr[:,:,2]; b=PB[:,1:,2]; m=lambda p:(np.abs(p[tr]-YS)/YS*100).mean(0)
    cfgs=[dict(lamH=1e-3,lamS=1e-3),dict(lamS=1e-2),dict(lamS=3e-2),dict(lamS=0.1),dict(lamS=0.3),dict(lamS=1.0),dict(lamS=0.1,rho2=0.2),dict(lamS=0.1,rho2=0.4)] if len(sys.argv)<2 else [eval(a) for a in sys.argv[1:]]
    for kw in cfgs:
        o,e=chain4(base,**kw); er=m(o); print('%-36s alone %.3f | blend .7: %.3f | ess %.0f |'%(str(kw),er.mean(),m(o**.7*b**.3).mean(),e),er,flush=True)
