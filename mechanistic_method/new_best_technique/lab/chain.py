"""Empirical state-space chain for substrate (and byproduct).
One-step linear transitions S_k -> S_k+1 are fitted on measured values (they are tight: 3-5 %), then each held-out batch
is simulated forward from its day-0 value with N particles; particles that contradict the logged substrate-feed decisions
(days 7-9, 11-12) get weight 0. Point prediction = weighted median with weights w/y (MAPE-optimal).
Covariates of each step: known inputs (feed B per volume, substrate bolus, sample interval, unlogged volume gain) and the
base blend's predictions (out-of-fold for training batches)."""
import sys,os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import *
bs,A,Y,tt,test=load(); tr=~test; Ytr=Y[tr][:,1:]; n=len(bs)
G,fl,D=feature_groups(A,Y,tt); V,F,dVr=D['V'],D['F'],D['dVr']
def base_preds(name):
    z=np.load(f'{LAB}/oof/{name}.npz'); P=np.zeros((n,13,4)); P[tr]=z['oof']; P[test]=z['test']; return np.concatenate([Y[:,:1],P],1)
def wmed(y,w):
    o=np.argsort(y); c=np.cumsum(w[o]); return y[o][min(np.searchsorted(c,0.5*c[-1]),len(y)-1)]
def ridge(X,y,lam):
    m=X.mean(0); s=X.std(0); s[s<1e-12]=1; Z=np.c_[(X-m)/s,np.ones(len(y))]; R=lam*len(y)*np.diag(np.r_[np.ones(Z.shape[1]-1),0])
    Ainv=np.linalg.inv(Z.T@Z+R); co=Ainv@Z.T@y; h=np.einsum('ij,jk,ik->i',Z,Ainv,Z); r=(y-Z@co)/(1-h)
    sd=1.4826*np.median(np.abs(r-np.median(r)))
    return (m,s,co),sd,r
def pred(mod,X): m,s,co=mod; return np.c_[(X-m)/s,np.ones(len(X))]@co
BOL=(7,8,9,11,12)
def fS(k,S,P,idx,PB,cfg,Sprev=None,bol=True):
    c=[S,F[idx,k,0]/V[idx,k],tt[idx,k+1]-tt[idx,k]]
    if k in BOL and bol: c.append(F[idx,k,1]/V[idx,k])
    if cfg['xb']: c+=[PB[idx,k,0],PB[idx,k+1,0]]
    if cfg['pb']: c+=[PB[idx,k,3],PB[idx,k+1,3]]
    if cfg['sb']: c+=[PB[idx,k+1,2]]
    if cfg['sp']: c+=[P]
    if cfg['ar2'] and k>=cfg['ar2']: c+=[Sprev,F[idx,k-1,0]/V[idx,k-1],F[idx,k-1,1]/V[idx,k-1]]
    if cfg['vb']: c+=[dVr[idx,k]]
    return np.c_[tuple(c)]
def fP(k,S,S1,P,idx,PB,cfg):
    c=[P,F[idx,k,0]/V[idx,k],tt[idx,k+1]-tt[idx,k],dVr[idx,k]]
    if cfg['ps']: c+=[S,S1]
    if cfg['xb']: c+=[PB[idx,k,0],PB[idx,k+1,0]]
    if cfg['ppb']: c+=[PB[idx,k+1,3]]
    return np.c_[tuple(c)]
def bolus_coef(fit,PB,cfg):
    """g/L gained per unit of bolus volume fraction: one value for all bolus days, pooled by inverse variance"""
    num=den=0.0
    for k in BOL:
        fs=F[fit,k,1]/V[fit,k]
        if fs.std()<1e-9: continue
        X=np.c_[fS(k,Y[fit,k,2],Y[fit,k,3],fit,PB,cfg,Y[fit,max(k-1,0),2]),np.ones(len(fit))]; y=Y[fit,k+1,2]
        XtX=np.linalg.pinv(X.T@X); co=XtX@X.T@y; r=y-X@co; s2=1.4826**2*np.median(np.abs(r-np.median(r)))**2*len(y)/max(len(y)-X.shape[1],1)
        v=s2*XtX[3,3]; num+=co[3]/v; den+=1/v
    return num/den
DEF=dict(cb=0,logk=(),tnu=0,ar2=0,vb=0,xb=1,pb=1,sb=0,sp=0,ps=1,ppb=1,lam=1e-3,lamP=1e-2,N=4000,sdS=1.0,sdP=1.0,flags=1,seed=0)
def simulate(fit,te,PB,cfg):
    rng=np.random.default_rng(cfg['seed']); N=cfg['N']; m=len(te)
    S=np.repeat(Y[te,0,2][:,None],N,1); P=np.repeat(Y[te,0,3][:,None],N,1); W=np.ones((m,N)); Sp=S; pathS=np.zeros((m,N,13)); pathP=np.zeros((m,N,13))
    rep=lambda a:np.repeat(te,N)
    cb=bolus_coef(fit,PB,{**cfg,'ar2':0}) if cfg['cb'] else 0.0
    for k in range(13):
        lg=k in cfg['logk']; tf=(np.log if lg else (lambda a:a)); bol=lg or not cfg['cb'] or k>9
        offf=0 if bol else cb*F[fit,k,1]/V[fit,k]; offt=0 if bol else np.repeat(cb*F[te,k,1]/V[te,k],N)
        mS,sdS,_=ridge(fS(k,tf(Y[fit,k,2]),Y[fit,k,3],fit,PB,cfg,Y[fit,max(k-1,0),2],bol),tf(Y[fit,k+1,2])-offf,cfg['lam'])
        mP,sdP,_=ridge(fP(k,Y[fit,k,2],Y[fit,k+1,2],Y[fit,k,3],fit,PB,cfg),Y[fit,k+1,3],cfg['lamP'])
        idx=np.repeat(te,N)
        muS=pred(mS,fS(k,tf(S.ravel()),P.ravel(),idx,PB,cfg,Sp.ravel(),bol))+offt; sd=cfg['sdS']*sdS; thr_t=tf(THR[k+1]) if (k+1) in THR else None
        if cfg['flags'] and (k+1) in THR:
            thr=thr_t; low=np.repeat(fl[te,FLAGCOL[k+1]]>0,N); c=norm.cdf((thr-muS)/sd)
            pm=np.where(low,c,1-c); u=rng.uniform(size=m*N); q=np.where(low,u*c,c+u*(1-c))
            S1=muS+sd*norm.ppf(np.clip(q,1e-12,1-1e-12)); S1=np.where(low,np.minimum(S1,thr),np.maximum(S1,thr)); W=W*np.maximum(pm,1e-300).reshape(m,N); W/=W.sum(1,keepdims=True)
        else: S1=muS+sd*rng.standard_normal(m*N)
        S1=np.exp(S1) if lg else S1
        S1=np.maximum(S1,0.03)
        P1=np.maximum(pred(mP,fP(k,S.ravel(),S1,P.ravel(),idx,PB,cfg))+cfg['sdP']*sdP*rng.standard_normal(m*N),0.02)
        Sp=S; S=S1.reshape(m,N); P=P1.reshape(m,N)
        pathS[:,:,k]=S; pathP[:,:,k]=P
    if cfg.get('ret'): return pathS,W
    oS=np.zeros((m,13)); oP=np.zeros((m,13)); ess=np.zeros(m)
    for i in range(m):
        ess[i]=W[i].sum()**2/(W[i]**2).sum()
        for k in range(13): oS[i,k]=wmed(pathS[i,:,k],W[i]/pathS[i,:,k]); oP[i,k]=wmed(pathP[i,:,k],W[i]/pathP[i,:,k])
    return oS,oP,ess
def chain(base,**kw):
    cfg={**DEF,**kw}; PB=base_preds(base); out=PB[:,1:].copy(); trn=np.where(tr)[0]; essmin=1e9
    for te in folds7(tr):
        oS,oP,ess=simulate(np.setdiff1d(trn,te),te,PB,cfg); out[te,:,2]=oS; out[te,:,3]=oP; essmin=min(essmin,ess.min())
    te=np.where(test)[0]; oS,oP,ess=simulate(trn,te,PB,cfg); out[te,:,2]=oS; out[te,:,3]=oP
    return out[tr],out[test],min(essmin,ess.min())
if __name__=='__main__':
    np.set_printoptions(precision=2,suppress=True,linewidth=220)
    base=sys.argv[1]; PB=base_preds(base); E0=mape(PB[tr,1:],Ytr); print('base',E0.mean(0),E0.mean()); print(E0[:,2:].T)
    cfgs=[dict()] if len(sys.argv)<3 else [eval(a) for a in sys.argv[2:]]
    for kw in cfgs:
        o,ot,ess=chain(base,**kw); E=mape(o,Ytr); print(kw,'S %.3f ess %d'%(E[:,2].mean(),ess),E[:,2])
        for w in (0.5,0.7):
            b=np.exp(w*np.log(o[:,:,2])+(1-w)*np.log(PB[tr,1:,2])); print('     blend w=%.1f: %.3f'%(w,(np.abs(b-Ytr[:,:,2])/Ytr[:,:,2]*100).mean()))
