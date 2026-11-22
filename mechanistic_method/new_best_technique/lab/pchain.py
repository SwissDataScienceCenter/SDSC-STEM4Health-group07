"""Day-by-day simulator for byproduct, run together with the substrate simulator.
Each particle carries (substrate, byproduct). Substrate steps as in chain.py (feed decisions constrain the paths).
Byproduct step: tomorrow's value from today's value, the simulated substrate of both days, feed B, the bolus, the sample
interval, the unlogged volume gain of that day, and the stage-1 cell predictions. Fits are robust (targets winsorised once).
New file; chain.py is not modified."""
import sys,os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from chain import *
PDEF=dict(DEF,vb=1,plog=0,lamP=3e-2,ps=1,pxb=1,panchor=0,pfs=1,pv=1,sdP=1.0,p2=0)
def fPb(k,S,S1,P,idx,PB,cfg,Pprev=None):
    tf=np.log if cfg['plog'] else (lambda a:a)
    c=[tf(P),F[idx,k,0]/V[idx,k],tt[idx,k+1]-tt[idx,k]]
    if cfg['pv']: c+=[dVr[idx,k]]
    if cfg['pfs'] and k in BOL: c+=[F[idx,k,1]/V[idx,k]]
    if cfg['ps']: c+=[S,S1]
    if cfg['pxb']: c+=[PB[idx,k,0],PB[idx,k+1,0]]
    if cfg['panchor']: c+=[tf(PB[idx,k+1,3])]
    if cfg['p2'] and k>=1: c+=[tf(Pprev)]
    return np.c_[tuple(c)]
def rfit(X,y,lam):
    mod,sd,r=ridge(X,y,lam); p=y-r                       # r = LOO residuals
    yc=np.clip(y,p-3*sd,p+3*sd); return ridge(X,yc,lam)[:2]
def simulate2(fit,te,PB,cfg):
    rng=np.random.default_rng(cfg['seed']); N=cfg['N']; m=len(te); tf=np.log if cfg['plog'] else (lambda a:a)
    S=np.repeat(Y[te,0,2][:,None],N,1); P=np.repeat(Y[te,0,3][:,None],N,1); W=np.ones((m,N))/N; Sp=S; Pp=P; pathP=np.zeros((m,N,13)); idx=np.repeat(te,N)
    for k in range(13):
        mS,sdS,_=ridge(fS(k,Y[fit,k,2],Y[fit,k,3],fit,PB,cfg,Y[fit,max(k-1,0),2]),Y[fit,k+1,2],cfg['lam'])
        mP,sdP=rfit(fPb(k,Y[fit,k,2],Y[fit,k+1,2],Y[fit,k,3],fit,PB,cfg,Y[fit,max(k-1,0),3]),tf(Y[fit,k+1,3]),cfg['lamP'])
        muS=pred(mS,fS(k,S.ravel(),P.ravel(),idx,PB,cfg,Sp.ravel())); sd=cfg['sdS']*sdS
        if cfg['flags'] and (k+1) in THR:
            thr=THR[k+1]; low=np.repeat(fl[te,FLAGCOL[k+1]]>0,N); c=norm.cdf((thr-muS)/sd); pm=np.where(low,c,1-c); u=rng.uniform(size=m*N); q=np.where(low,u*c,c+u*(1-c))
            S1=muS+sd*norm.ppf(np.clip(q,1e-12,1-1e-12)); S1=np.where(low,np.minimum(S1,thr),np.maximum(S1,thr)); W=W*np.maximum(pm,1e-300).reshape(m,N); W/=W.sum(1,keepdims=True)
        else: S1=muS+sd*rng.standard_normal(m*N)
        S1=np.maximum(S1,0.03)
        muP=pred(mP,fPb(k,S.ravel(),S1,P.ravel(),idx,PB,cfg,Pp.ravel()))+cfg['sdP']*sdP*rng.standard_normal(m*N)
        P1=np.exp(muP) if cfg['plog'] else np.maximum(muP,0.02)
        Sp=S; Pp=P; S=S1.reshape(m,N); P=P1.reshape(m,N); pathP[:,:,k]=P
    out=np.zeros((m,13))
    for i in range(m):
        for k in range(13): out[i,k]=wmed(pathP[i,:,k],W[i]/pathP[i,:,k])
    return out
def pchain(base,**kw):
    cfg={**PDEF,**kw}; PB=base_preds(base); out=PB[:,1:,3].copy(); trn=np.where(tr)[0]
    for te in folds7(tr): out[te]=simulate2(np.setdiff1d(trn,te),te,PB,cfg)
    te=np.where(test)[0]; out[te]=simulate2(trn,te,PB,cfg)
    return out
if __name__=='__main__':
    np.set_printoptions(precision=1,suppress=True,linewidth=220)
    base='b12_stage1'; PB=base_preds(base); YP=Ytr[:,:,3]; b=PB[:,1:,3]; m=lambda p:(np.abs(p[tr]-YP)/YP*100).mean(0)
    print('stage1 byproduct        %.3f'%m(b).mean(),m(b))
    cfgs=[dict(),dict(plog=1),dict(ps=0),dict(pxb=0),dict(lamP=0.1),dict(lamP=0.01),dict(pv=0),dict(panchor=1),dict(plog=1,panchor=1),dict(p2=1),dict(plog=1,p2=1)] if len(sys.argv)<2 else [eval(a) for a in sys.argv[1:]]
    for kw in cfgs:
        o=pchain(base,N=3000,**kw); e=m(o); e3=m(np.exp((2*np.log(b)+np.log(o))/3)); e2=m(np.sqrt(b*o))
        print('%-28s alone %.3f | blend 1/3: %.3f | blend 1/2: %.3f |'%(str(kw),e.mean(),e3.mean(),e2.mean()),e,'| b1/3',e3,flush=True)
