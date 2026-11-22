"""chem.py - targeted improvement of substrate (t=2) and byproduct (t=3) predictions.

Works purely on saved OOF/test predictions of existing members (lab/oof/*.npz), so it
is cheap and needs no ODE refit. Every learned choice is made INSIDE folds7, so the
reported OOF score is honest with respect to the post-processing itself.

Steps (cells, t=0/1, are left exactly as in the final recipe):
  1. per-target blend weights in log space (instead of the equal 1/3 geometric mean)
  2. per-target smoothing strength for substrate/byproduct (none / light / strong)
  3. a mass-balance residual stacker: ridge on log-residuals using
     cumulative substrate fed, predicted integral of viable cells, feed-D, flags, time
  4. a final re-projection of substrate onto the side of the feeding threshold
Usage:  cd lab && python chem.py  [member1 member2 ...]
"""
import sys,os,itertools; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from final import *            # load(), get(), smooth_cells, folds7, register, members H,SS,P2...
from smooth2 import lowpass
CH=(2,3)
CELLS='b09_hyb_stat_bayesresid_smoothcells'   # reference final submission (before/after comparison)
_,fl,D=feature_groups(A,Y,tt)

# ---------- 1. per-target log-space weights on a coarse simplex grid ----------
def simplex(m,step=0.1):
    g=np.round(np.arange(0,1+1e-9,step),3)
    return [np.r_[c,1-sum(c)] for c in itertools.product(g,repeat=m-1) if sum(c)<=1+1e-9]
def wblend(L,w): return np.tensordot(w,L,1)                      # L (m,n,13,4) log preds
def fit_w(L,Yl,t):
    best=None
    for w in simplex(len(L)):
        e=(np.abs(np.exp(wblend(L[:,:,:,t],w))-np.exp(Yl[:,:,t]))/np.exp(Yl[:,:,t])).mean()
        if best is None or e<best[0]: best=(e,w)
    return best[1]

# ---------- 2. smoothing options for chemistry ----------
SMO=[None,(1/3,1),(1/3,3),(1/3,6)]
def smooth_t(Lt,ref_t,opt):
    return Lt if opt is None else ref_t+lowpass(Lt-ref_t,*opt)

# ---------- 3. mass-balance features (all inputs, no targets) ----------
def chem_feats(Lpred):
    """per batch & sample point features for the residual stacker; Lpred = log blended preds (n,13,4)"""
    V,F=D['V'],D['F']; n=len(V)
    sfed=np.cumsum(F[:,:13,1]/V[:,:13],1)                         # cumulative substrate fed per volume
    fd  =np.cumsum(F[:,:13,3]/V[:,:13],1)                         # cumulative feed D (byproduct consumption driver)
    fb  =np.cumsum(F[:,:13,0]/V[:,:13],1)
    vcd =np.exp(Lpred[:,:,0]); dtt=np.diff(tt,axis=1)
    ivcd=np.cumsum(vcd*dtt,1)                                       # integral of viable cells ~ total uptake capacity
    dil =np.log(V[:,1:14]/V[:,:1])                                  # dilution since start
    time=np.tile(np.arange(13),(n,1))
    return np.stack([sfed,fd,fb,np.log(ivcd),dil,time,Lpred[:,:,0],Lpred[:,:,2],Lpred[:,:,3]],-1)  # (n,13,f)

def ridge_fit(X,r,alphas=(1,10,100,1000)):
    """closed-form LOO choose alpha; returns predictor. X (N,f) r (N,)"""
    mu,s=X.mean(0),X.std(0)+1e-9; Z=np.c_[np.ones(len(X)),(X-mu)/s]; best=None
    for a in alphas:
        Rg=a*np.eye(Z.shape[1]); Rg[0,0]=0; Hm=Z@np.linalg.solve(Z.T@Z+Rg,Z.T)
        loo=(r-Hm@r)/(1-np.diag(Hm)); e=np.mean(np.abs(loo))
        if best is None or e<best[0]: best=(e,a)
    a=best[1]; Rg=a*np.eye(Z.shape[1]); Rg[0,0]=0; beta=np.linalg.solve(Z.T@Z+Rg,Z.T@r)
    return lambda Xn: np.c_[np.ones(len(Xn)),(Xn-mu)/s]@beta

def stack(Ltr,Ytr_l,trb,Lte,teb,t,shrink=0.5):
    """learn residual of target t from mass-balance features; residuals pooled over batch x time.
       NB: training rows are OOF predictions, so residuals are honest. shrink keeps it conservative."""
    Xtr=chem_feats(Ltr)[...,:] ; Xtr=Xtr.reshape(-1,Xtr.shape[-1])
    # features need batch-aligned D arrays -> we index with global batch ids
    r=(Ytr_l[:,:,t]-Ltr[:,:,t]).ravel(); r=np.clip(r,-0.5,0.5)
    f=ridge_fit(Xtr,r); Xte=chem_feats(Lte); return Lte[:,:,t]+shrink*f(Xte.reshape(-1,Xte.shape[-1])).reshape(Lte.shape[:2])

# chem_feats uses global D -> wrap so rows refer to chosen batches
_Dfull=D
def with_rows(rows,fn,*a):
    global D; D={k:v[rows] for k,v in _Dfull.items()}; g_tt=globals()['tt']; globals()['tt']=g_tt[rows]
    try: return fn(*a)
    finally: D=_Dfull; globals()['tt']=g_tt

# ---------- 4. substrate threshold re-projection ----------
def project(P,flr):
    P=P.copy()
    for k,thr in THR.items():
        low=flr[:,FLAGCOL[k]]>0; s=P[:,k-1,2]
        P[low,k-1,2]=np.minimum(s[low],thr*0.999); P[~low,k-1,2]=np.maximum(s[~low],thr*1.001)
    return P

# ---------- driver: fit every choice on 6 folds, apply on the 7th ----------
def run(members,name='c01_chem_tuned'):
    O=[get(n) for n in members]; Lo=np.log(np.stack([o[0] for o in O])); Lt=np.log(np.stack([o[1] for o in O]))
    idx=np.where(tr)[0]; te_idx=np.where(test)[0]; pos={r:i for i,r in enumerate(idx)}; LY=np.log(Ytr)
    base=np.exp(Lo.mean(0)); out=np.empty_like(base)
    cells_oof,cells_te=get(CELLS)          # keep cell predictions from the smoothed final recipe
    def choose(trn):                                         # trn: local indices of the fitting rows
        cfg={}
        for t in CH:
            w=fit_w(Lo[:,trn],LY[trn],t); ref=LY[trn][:,:,t].mean(0); Lb=wblend(Lo[:,trn][:,:,:,t],w)
            sc=lambda L:(np.abs(np.exp(L)-Ytr[trn][:,:,t])/Ytr[trn][:,:,t]).mean()
            opt=min(SMO,key=lambda o:sc(smooth_t(Lb,ref,o))); cfg[t]=(w,opt)
        return cfg
    def apply(cfg,Lm,ref,rows_g,fit=None):
        L=np.log(np.exp(Lm.mean(0)))                         # start from equal blend (cells unchanged)
        for t,(w,opt) in cfg.items(): L[:,:,t]=smooth_t(wblend(Lm[:,:,:,t],w),ref[:,t],opt)
        return L
    for ho in folds7(tr):
        ii=np.array([pos[r] for r in ho]); m=np.ones(len(idx),bool); m[ii]=False; trn=np.where(m)[0]
        cfg=choose(trn); ref=LY[trn].mean(0)
        Ltrn=apply(cfg,Lo[:,trn],ref,idx[trn]); Lho=apply(cfg,Lo[:,ii],ref,idx[ii])
        for t in CH:   # stacker trained on the 6 fitting folds' (OOF) predictions
            Xs=with_rows(idx[trn],chem_feats,Ltrn); Xh=with_rows(idx[ii],chem_feats,Lho)
            r=np.clip(LY[trn][:,:,t]-Ltrn[:,:,t],-.5,.5).ravel(); f=ridge_fit(Xs.reshape(-1,Xs.shape[-1]),r)
            Lho[:,:,t]+=0.5*f(Xh.reshape(-1,Xh.shape[-1])).reshape(len(ii),13)
        P=np.exp(Lho); P[:,:,:2]=cells_oof[ii][:,:,:2]; out[ii]=project(P,fl[idx[ii]])
    # test: fit on all 49
    allr=np.arange(len(idx)); cfg=choose(allr); ref=LY.mean(0)
    Ltrn=apply(cfg,Lo,ref,idx); Lte=apply(cfg,Lt,ref,te_idx)
    for t in CH:
        Xs=with_rows(idx,chem_feats,Ltrn); Xh=with_rows(te_idx,chem_feats,Lte)
        f=ridge_fit(Xs.reshape(-1,Xs.shape[-1]),np.clip(LY[:,:,t]-Ltrn[:,:,t],-.5,.5).ravel())
        Lte[:,:,t]+=0.5*f(Xh.reshape(-1,Xh.shape[-1])).reshape(len(te_idx),13)
    Pt=np.exp(Lte); Pt[:,:,:2]=cells_te[:,:,:2]; Pt=project(Pt,fl[te_idx])
    print('chosen on all data:',{T[t]:(np.round(w,1).tolist(),o) for t,(w,o) in cfg.items()})
    Eb,Ec=mape(cells_oof,Ytr).mean(0),mape(out,Ytr).mean(0)
    for i,t in enumerate(T): print(f'  {t:10s} before {Eb[i]:.3f}  after {Ec[i]:.3f}')
    return register(name,'blend','per-target weights + chem smoothing + mass-balance residual stacker + threshold projection',out,Ytr,Pt)

if __name__=='__main__':
    members=sys.argv[1:] or [H,SS,SE,'p2_m21_re8_evidence_all_n1000']
    run(members)