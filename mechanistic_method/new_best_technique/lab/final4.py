"""Version 4 recipe (new file; nothing existing is modified). Needs the stored version-1/version-2 predictions in oof/.
 cells      : as version 2 (0.5 x v1 + 0.5 x one-offset-per-batch model)
 byproduct  : three phases, as found by the grid search and identical on both splits
                days 1-4  : 0.5 x v1 + 0.5 x small model (volume gain d2-5 and d5-8, initial volume, day-0 byproduct;
                            plus the sample-time offset on days 1-2, where byproduct rises by ~1 per day)
                days 5-8  : small model alone (volume gain d2-5, initial volume)
                days 9-13 : 0.7 x v1 + 0.3 x small model (one offset per batch, incl. day-11/12 feed flags)
 substrate  : 0.3 x v1 + 0.7 x simulators, where simulators = version-2 chain ensemble, version-2 state-space ensemble and
              three chains with selective shrinkage (chain4.py). For batches that got a bolus on days 7-9 the
              simulators get weight 1.0 (the large models have only 1-3 such batches to learn each case from).
Writes oof/b15_v4.npz. Prints CV; pass 'csv' to also write ../submission_v4.csv (refuses to overwrite)."""
import sys,os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import *
import sparse as sp, sparse4 as s4, chain4 as c4
bs,A,Y,tt,test=load(); tr=~test; Ytr=Y[tr][:,1:]; n=len(bs); trn=np.where(tr)[0]
G,fl,D=feature_groups(A,Y,tt)
def full(name): z=np.load(f'{LAB}/oof/{name}.npz'); P=np.zeros((n,13,4)); P[tr]=z['oof']; P[test]=z['test']; return P
def show(nm,P): E=mape(P[tr],Ytr); print('%-44s'%nm,' '.join('%.3f'%e for e in E.mean(0)),'| %.3f'%E.mean(),flush=True); return E
def perday_sel(fit,te,cols,targets,lam=0.003,tdays=(0,1)):
    """per-day ridge; the sample-time offset is added as a feature only on the listed day indices"""
    mu0,sd0=sp.perday(fit,te,cols=cols,targets=targets,lam=lam); mu1,sd1=s4.perdayt(fit,te,cols=cols,targets=targets,lam=lam)
    for k in tdays: mu0[:,k]=mu1[:,k]; sd0[k]=sd1[k]
    return mu0,sd0
def build(time_days=(0,1),bolus_w=1.0,use_c4=True,verbose=True):
    V1=full('b09_hyb_stat_bayesresid_smoothcells'); V2=full('b13_v2_small_plus_substrate_chain'); X=V2.copy()
    # byproduct
    early=sp.run(perday_sel,[3],cols=['base25','V0','lp0','base58'],lam=0.003,tdays=time_days)[:,:,0]
    mid=sp.run(sp.perday,[3],cols=['base25','V0'],lam=0.003)[:,:,0]
    late=sp.run(sp.rank1,[3],cols=['base25','V0','lp0','fl11','fl12'],lam=0.01,days=slice(2,12))[:,:,0]
    b=V1[:,:,3]; X[:,0:4,3]=np.sqrt(b[:,0:4]*early[:,0:4]); X[:,4:8,3]=mid[:,4:8]; X[:,8:,3]=b[:,8:]**0.7*late[:,8:]**0.3
    # substrate
    lc=np.log(full('c02_chain_ens')[:,:,2]); lk=np.log(full('k02_statespace_ens')[:,:,2]); sims=[lc,lk]
    if use_c4:
        l4=[]
        for i,ls in enumerate((0.03,0.1,0.3)):
            o,e=c4.chain4('b12_stage1',lamS=ls,N=8000,seed=40+i); l4.append(np.log(o))
        sims.append(np.mean(l4,0))
    sim=np.mean(sims,0); bol=fl[:,:3].sum(1)>0; w=np.where(bol,bolus_w,0.7)[:,None]
    X[:,:,2]=np.exp(w*sim+(1-w)*np.log(V1[:,:,2]))
    if verbose: show('v1',V1); show('v2',V2); show('v4',X)
    return X
if __name__=='__main__':
    np.set_printoptions(precision=1,suppress=True,linewidth=220)
    if 'ablate' in sys.argv:
        for nm,kw in [('v4 without time feature',dict(time_days=())),('v4 time on days 1,2,13 (early model only uses 1,2)',dict(time_days=(0,1))),('v4 without bolus rule',dict(bolus_w=0.7)),('v4 without selective-shrinkage chains',dict(use_c4=False))]:
            show(nm,build(verbose=False,**kw))
    X=build(); E=mape(X[tr],Ytr)
    for i,t in enumerate(T): print('   ',t,np.round(E[:,i],1))
    np.savez(f'{LAB}/oof/b15_v4.npz',oof=X[tr],test=X[test])
    if 'csv' in sys.argv:
        out=PROJ+'submission_v4.csv'; assert not os.path.exists(out),'submission_v4.csv exists; not overwriting'
        from report import write_submission; write_submission('b15_v4','submission_v4.csv'); print('wrote',out)
