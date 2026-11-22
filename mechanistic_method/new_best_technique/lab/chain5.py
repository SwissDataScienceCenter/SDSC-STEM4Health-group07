"""Substrate simulator with switches for the test battery (new file; chain.py and chain4.py untouched)."""
import sys,os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from chain4 import *
D5=dict(D4,lamS=0.1,N=4000,thr79=1.95,thr1112=0.967,wp=1.0,point='wmed',flagdays=(7,8,9,11,12),quad=0,inter=0,sd10=1.0,sdlate=1.0,sdearly=1.0,xfeat=())
def thr_of(d,cfg): return cfg['thr79'] if d<=9 else cfg['thr1112']
def f5(k,S,P,idx,PB,cfg,Sprev):
    X=fS(k,S,P,idx,PB,cfg,Sprev); ex=[]
    if cfg['quad']: ex.append(S**2)
    if cfg['inter']: ex.append(S*(tt[idx,k+1]-tt[idx,k]))
    for name in cfg['xfeat']: ex.append(XF[name][idx])
    return np.c_[X,np.c_[tuple(ex)]] if ex else X
XF={}
def simulate5(fit,te,PB,cfg):
    rng=np.random.default_rng(cfg['seed']); N=cfg['N']; m=len(te); S=np.repeat(Y[te,0,2][:,None],N,1); P=np.repeat(Y[te,0,3][:,None],N,1)
    W=np.ones((m,N))/N; Sp=S; path=np.zeros((m,N,13)); idx=np.repeat(te,N)
    for k in range(13):
        X=f5(k,Y[fit,k,2],Y[fit,k,3],fit,PB,cfg,Y[fit,max(k-1,0),2]); y=Y[fit,k+1,2]
        mod,r=ridge2(X,y,np.r_[np.full(nhard(k),cfg['lamH']),np.full(X.shape[1]-nhard(k),cfg['lamS'])]); sd=1.4826*np.median(np.abs(r-np.median(r)))*cfg['sdS']
        sd*=cfg['sd10'] if k==10 else 1.0; sd*=cfg['sdlate'] if k>=10 else cfg['sdearly'] if k<4 else 1.0
        mu=pred(mod,f5(k,S.ravel(),P.ravel(),idx,PB,cfg,Sp.ravel()))
        if cfg['flags'] and (k+1) in cfg['flagdays']:
            thr=thr_of(k+1,cfg); low=np.repeat(fl[te,FLAGCOL[k+1]]>0,N); c=norm.cdf((thr-mu)/sd); pm=np.where(low,c,1-c); q=rng.uniform(size=m*N); q=np.where(low,q*c,c+q*(1-c))
            S1=mu+sd*norm.ppf(np.clip(q,1e-12,1-1e-12)); S1=np.where(low,np.minimum(S1,thr),np.maximum(S1,thr)); W=W*np.maximum(pm,1e-300).reshape(m,N); W/=W.sum(1,keepdims=True)
        else: S1=mu+sd*rng.standard_normal(m*N)
        Sp=S; S=np.maximum(S1,0.03).reshape(m,N); path[:,:,k]=S
    out=np.zeros((m,13))
    for i in range(m):
        for k in range(13):
            if cfg['point']=='mean': out[i,k]=(W[i]*path[i,:,k]).sum()
            elif cfg['point']=='hmean': out[i,k]=1/((W[i]/path[i,:,k]).sum())
            else: out[i,k]=wmed(path[i,:,k],W[i]/path[i,:,k]**cfg['wp'])
    return out
def chain5(base='b12_stage1',**kw):
    cfg={**D5,**kw}; PB=base_preds(base); out=PB[:,1:,2].copy(); trn=np.where(tr)[0]
    for te in folds7(tr): out[te]=simulate5(np.setdiff1d(trn,te),te,PB,cfg)
    te=np.where(test)[0]; out[te]=simulate5(trn,te,PB,cfg); return out
