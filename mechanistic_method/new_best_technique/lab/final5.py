"""Version 5 recipe (new file; nothing existing is modified). Needs oof/b15_v4.npz and the version-1 predictions.
Changes against version 4, both found by the 178-test battery (tests100*.py) and confirmed by a 500-placebo permutation test:
 cells     : the small model now uses feed B on the peak feeding days 5-7 (instead of the average over days 4-12) and a
             flag for a sugar top-up on days 7-9. A top-up means the sugar ran low, which means more cells than expected.
             Signals: day-0 VCD, day-0 byproduct, initial volume, feed B days 5-7, top-up flag. Blend weight unchanged (0.5).
 byproduct : days 9-13 use a per-day small model that knows the exact sample time on days 12-13 (weight unchanged, 0.3).
 substrate : unchanged from version 4.
Writes oof/b16_v5.npz; pass 'csv' to write ../submission_v5.csv (refuses to overwrite)."""
import sys,os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import *
import sparse as sp, sparse4 as s4
bs,A,Y,tt,test=load(); tr=~test; Ytr=Y[tr][:,1:]; n=len(bs); trn=np.where(tr)[0]
G,fl,D=feature_groups(A,Y,tt); V,F=D['V'],D['F']
st=lambda v:(v-v.mean())/v.std()
sp.FE['fb57']=st(np.log(F[:,5:8,0].sum(1)/V[:,5])); sp.FE['any79']=st(fl[:,:3].sum(1).astype(float))
CELLS=['lx0','lp0','V0','fb57','any79']
def full(name): z=np.load(f'{LAB}/oof/{name}.npz'); P=np.zeros((n,13,4)); P[tr]=z['oof']; P[test]=z['test']; return P
def show(nm,P): E=mape(P[tr],Ytr); print('%-12s'%nm,' '.join('%.3f'%e for e in E.mean(0)),'| %.3f'%E.mean(),flush=True); return E
def perday_time(fit,te,cols,targets,lam=0.01,tdays=(11,12)):
    mu0,sd0=sp.perday(fit,te,cols=cols,targets=targets,lam=lam); mu1,sd1=s4.perdayt(fit,te,cols=cols,targets=targets,lam=lam)
    for k in tdays: mu0[:,k]=mu1[:,k]; sd0[k]=sd1[k]
    return mu0,sd0
def build(new_cells=True,new_byp=True,w=0.5):
    V1=full('b09_hyb_stat_bayesresid_smoothcells'); V4=full('b15_v4'); X=V4.copy()
    if new_cells: X[:,:,:2]=V1[:,:,:2]**(1-w)*sp.run(sp.rank1,[0,1],cols=CELLS,lam=0.03,shrink=1.0)**w
    if new_byp:
        late=sp.run(perday_time,[3],cols=['base25','V0','lp0','fl11','fl12'])[:,:,0]; X[:,8:,3]=V1[:,8:,3]**0.7*late[:,8:]**0.3
    return X
if __name__=='__main__':
    np.set_printoptions(precision=1,suppress=True,linewidth=220)
    show('v1',full('b09_hyb_stat_bayesresid_smoothcells')); show('v4',full('b15_v4')); show('v5 cells only',build(new_byp=False)); show('v5 byp only',build(new_cells=False)); X=build(); E=show('v5',X)
    for i,t in enumerate(T): print('   ',t,np.round(E[:,i],1))
    np.savez(f'{LAB}/oof/b16_v5.npz',oof=X[tr],test=X[test])
    V4=full('b15_v4'); te=np.where(test)[0]; bol=fl[:,:3].sum(1)>0
    print('test cells, v5 / v4 (mean ratio per batch):',{bs[b]:round(float((X[b,:,:2]/V4[b,:,:2]).mean()),3) for b in te})
    print('top-up test batches:',[bs[b] for b in te if bol[b]],'| feed-B days 5-7 signal: test range %.2f..%.2f, known range %.2f..%.2f (z-scores)'%(sp.FE['fb57'][test].min(),sp.FE['fb57'][test].max(),sp.FE['fb57'][tr].min(),sp.FE['fb57'][tr].max()))
    if 'csv' in sys.argv:
        out=PROJ+'submission_v5.csv'; assert not os.path.exists(out),'submission_v5.csv exists; not overwriting'
        from report import write_submission; write_submission('b16_v5','submission_v5.csv'); print('wrote',out)
