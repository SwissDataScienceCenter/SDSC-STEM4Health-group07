"""Automatic grid search with cross-validation over the version-2 recipe.
Each target is searched separately (the score is a plain mean over targets, so they do not interact).
  select : configuration with the lowest out-of-fold MAPE on all 49 batches (what one would use for the test set)
  nested : the same search repeated inside each outer fold on the other six folds' out-of-fold predictions,
           then applied to the held-out fold. This is the honest estimate of what the search itself is worth.
Usage: python gridcv.py            -> prints tables, saves gridcv_result.json and oof/g01_gridcv.npz"""
import sys,os,json,itertools; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import *
import sparse as sp
bs,A,Y,tt,test=load(); tr=~test; Ytr=Y[tr][:,1:]; n=len(bs); trn=np.where(tr)[0]; FOLDS=folds7(tr)
def full(name): z=np.load(f'{LAB}/oof/{name}.npz'); P=np.zeros((n,13,4)); P[tr]=z['oof']; P[test]=z['test']; return P
PH={'one':[slice(0,13)],'three':[slice(0,4),slice(4,8),slice(8,13)]}
WG=[round(x,1) for x in np.arange(0,1.01,0.1)]
def ape(P,j): return np.abs(P[:,:,j if P.shape[2]==4 else 0]-Y[:,1:,j])/Y[:,1:,j]*100      # (n,13); nan rows for test
def search(cands,base,j,phases):
    """cands: dict name -> (n,13) predictions of the new member for target j; base: (n,13) version-1 predictions.
    For every phase pick (candidate, weight) minimising MAPE of base^(1-w) * cand^w. Returns selected + nested OOF + test preds."""
    names=list(cands); LB=np.log(base); LC={c:np.log(cands[c]) for c in names}; Yt=Y[:,1:,j]
    E=np.zeros((len(names),len(WG),n,13))                       # per-batch-day APE of every (cand, w)
    for a,c in enumerate(names):
        for b,w in enumerate(WG): E[a,b]=np.abs(np.exp((1-w)*LB+w*LC[c])-Yt)/Yt*100
    out=np.zeros((n,13)); nest=np.zeros((n,13)); picks=[]
    for s in phases:
        M=E[:,:,trn][:,:,:,s].mean((2,3)); a,b=np.unravel_index(np.argmin(M),M.shape); picks.append((names[a],WG[b],float(M[a,b])))
        out[:,s]=np.exp((1-WG[b])*LB[:,s]+WG[b]*LC[names[a]][:,s])
        for te in FOLDS:
            fit=np.setdiff1d(trn,te); Mf=E[:,:,fit][:,:,:,s].mean((2,3)); a2,b2=np.unravel_index(np.argmin(Mf),Mf.shape)
            nest[te,s]=np.exp((1-WG[b2])*LB[te,s]+WG[b2]*LC[names[a2]][te,s])
    m=lambda P: float((np.abs(P[trn]-Yt[trn])/Yt[trn]*100).mean())
    return out,nest,picks,m(out),m(nest)
def candidates():
    C={0:{},3:{},2:{}}
    for cols in (['lx0','lp0','V0','fbL'],['lx0','lp0','V0'],['lx0','lp0','V0','fbL','nC'],['lx0','lp0','V0','fbL','ls0'],['lx0','lp0','fbL'],['lx0','lp0','V0','fbL','base25']):
        for lam in (0.003,0.01,0.03,0.1,0.3,1.0):
            for sh in (0.0,0.5,1.0):
                C[0]['+'.join(cols)+f'|lam{lam}|sh{sh}']=sp.run(sp.rank1,[0,1],cols=cols,lam=lam,shrink=sh)
    for cols in (['base25','V0'],['base25','V0','lp0'],['base25','V0','lp0','lx0'],['base25','V0','lp0','base58'],['base25','V0','lp0','fl11','fl12'],['base13','base35','V0','lp0']):
        for lam in (0.003,0.01,0.03,0.1,0.3,1.0):
            C[3]['pd:'+'+'.join(cols)+f'|lam{lam}']=sp.run(sp.perday,[3],cols=cols,lam=lam)[:,:,0]
        for lam in (0.01,0.03,0.1): C[3]['r1:'+'+'.join(cols)+f'|lam{lam}']=sp.run(sp.rank1,[3],cols=cols,lam=lam,days=slice(2,12))[:,:,0]
    ch=np.log(full('c02_chain_ens')[:,:,2]); kf=np.log(full('k02_statespace_ens')[:,:,2])
    for m in (0,0.25,0.5,0.75,1.0): C[2][f'chain{1-m:.2f}+statespace{m:.2f}']=np.exp((1-m)*ch+m*kf)
    return C
if __name__=='__main__':
    B9=full('b09_hyb_stat_bayesresid_smoothcells'); V2=full('b13_v2_small_plus_substrate_chain') if os.path.exists(f'{LAB}/oof/b13_v2_small_plus_substrate_chain.npz') else None
    C=candidates(); res={}; G1=B9.copy(); G3=B9.copy(); N1=B9.copy(); N3=B9.copy()
    print('%-10s %-6s | v1 %6s | v2 fixed | grid select | grid nested | picks'%('target','phases',''))
    e=lambda P,j: float((np.abs(P[trn][:,:,j]-Ytr[:,:,j])/Ytr[:,:,j]*100).mean())
    for j,key in [(0,0),(1,0),(2,2),(3,3)]:
        for pn,ph in PH.items():
            cands={k:(v[:,:,j] if v.ndim==3 else v) for k,v in C[key].items()}
            out,nest,picks,ms,mn=search(cands,B9[:,:,j],j,ph)
            res[f'{T[j]}|{pn}']=dict(select=ms,nested=mn,picks=picks)
            (G1 if pn=='one' else G3)[:,:,j]=out; (N1 if pn=='one' else N3)[:,:,j]=nest
            print('%-10s %-6s | %.3f | %s | %.3f | %.3f | %s'%(T[j],pn,e(B9,j),('%.3f'%e(V2,j)) if V2 is not None else '  -  ',ms,mn,'; '.join(f'{p[0]} w={p[1]}' for p in picks)),flush=True)
    for nm,P in [('v1',B9),('v2 fixed',V2),('grid one-phase select',G1),('grid one-phase NESTED',N1),('grid three-phase select',G3),('grid three-phase NESTED',N3)]:
        if P is not None: E=mape(P[tr],Ytr); print('%-26s'%nm,' '.join('%.3f'%x for x in E.mean(0)),'| %.3f'%E.mean())
    json.dump(res,open(f'{LAB}/gridcv_result.json','w'),indent=1)
    np.savez(f'{LAB}/oof/g01_gridcv_onephase.npz',oof=G1[tr],test=G1[test]); np.savez(f'{LAB}/oof/g02_gridcv_threephase.npz',oof=G3[tr],test=G3[test])
    np.savez(f'{LAB}/oof/_gridcv_nested.npz',N1=N1[tr],N3=N3[tr])
