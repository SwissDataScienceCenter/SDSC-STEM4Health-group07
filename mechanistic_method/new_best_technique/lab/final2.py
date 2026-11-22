"""Second-generation blend (v2). Run after final.py:  python final2.py
  cells      = geometric mean of b09 and the one-number-per-batch model (sparse.rank1)
  byproduct  = geometric mean of b09 and the 3-signal per-day model (sparse.perday)
  substrate  = 0.3 * b09  +  0.7 * [chain ensemble x state-space ensemble]   (weights on the log scale)
All weights are fixed in advance; nothing is tuned per day. Every member is refitted inside each fold."""
import sys,os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import *
import sparse as sp, chain as ch, kf
bs,A,Y,tt,test=load(); tr=~test; Ytr=Y[tr][:,1:]; n=len(bs); trn=np.where(tr)[0]
CHAIN=[dict(vb=1),dict(),dict(vb=1,ar2=5,lam=1e-2),dict(vb=1,logk=(10,11,12)),dict(vb=1,sp=1,cb=1),dict(vb=1,xb=0,cb=1),dict(vb=1,lam=1e-2,cb=1),dict(vb=1,sdS=1.2,cb=1)]
KF=[dict(lam=1e-2,sdq=1.4),dict(lam=1e-2,sdq=1.4,vb=1,cb=1),dict(lam=3e-2,sdq=1.4,cb=1),dict(lam=3e-2,sdq=1.4,vb=1),dict(lam=1e-2,sdq=1.7,cb=1),dict(lam=1e-2,sdq=1.2,vb=1)]
def full(name): z=np.load(f'{LAB}/oof/{name}.npz'); P=np.zeros((n,13,4)); P[tr]=z['oof']; P[test]=z['test']; return P
def show(nm,P): E=mape(P[tr],Ytr); print('%-36s'%nm,' '.join('%.3f'%e for e in E.mean(0)),'| %.3f'%E.mean(),flush=True)
def save(name,P): np.savez(f'{LAB}/oof/{name}.npz',oof=P[tr],test=P[test])
def build(base='b09_hyb_stat_bayesresid_smoothcells',N=8000,wS=0.7,reg=True):
    B9=full(base); show(base,B9)
    SM=B9.copy(); SM[:,:,:2]=sp.run(sp.rank1,[0,1],cols=['lx0','lp0','V0','fbL'],shrink=1.0); SM[:,:,3:]=sp.run(sp.perday,[3],cols=['base25','V0','lp0'])
    show('small models (cells, byproduct)',SM)
    S1=np.sqrt(B9*SM); save('b12_stage1',S1); show('b12 = b09 x small',S1)
    LC=np.zeros((len(CHAIN),n,13)); LK=np.zeros((len(KF),n,13))
    for i,kw in enumerate(CHAIN):
        o,ot,ess=ch.chain('b12_stage1',N=N,seed=i,**kw); LC[i,tr]=np.log(o[:,:,2]); LC[i,test]=np.log(ot[:,:,2])
    for i,kw in enumerate(KF):
        o,ot,ess,_=kf.kf('b12_stage1',N=N,seed=10+i,**kw); LK[i,tr]=np.log(o); LK[i,test]=np.log(ot)
    C=S1.copy(); C[:,:,2]=np.exp(LC.mean(0)); K=S1.copy(); K[:,:,2]=np.exp(LK.mean(0)); show('substrate chain ensemble',C); show('substrate state-space ensemble',K)
    save('c02_chain_ens',C); save('k02_statespace_ens',K)
    Fin=S1.copy(); Fin[:,:,2]=np.exp(wS*0.5*(LC.mean(0)+LK.mean(0))+(1-wS)*np.log(S1[:,:,2])); show('b13 final',Fin)
    if reg:
        register('sp1_small_models','small','cells: one offset per batch from day-0 VCD, day-0 byproduct, initial volume, late feed-B dosing; byproduct: per-day ridge on unlogged volume gain days 2-5, initial volume, day-0 byproduct; substrate copied from b09',SM[tr],Ytr,SM[test],quiet=True)
        register('b12_stage1','blend','geometric mean of b09 and the small models (cells, byproduct)',S1[tr],Ytr,S1[test],quiet=True)
        register('c02_chain_ens','state-space','substrate: one-step linear transitions fitted on measured values, particles from day 0, feed decisions as hard constraints; 8 variants averaged; other targets from b12',C[tr],Ytr,C[test],quiet=True)
        register('k02_statespace_ens','state-space','substrate: latent state + assay noise by EM (Kalman smoother), particles with feed-decision likelihood; 6 variants averaged; other targets from b12',K[tr],Ytr,K[test],quiet=True)
        register('b13_v2_small_plus_substrate_chain','blend','b12 with substrate = 0.3 x b09 + 0.7 x (chain ensemble, state-space ensemble) on the log scale',Fin[tr],Ytr,Fin[test])
    return Fin
if __name__=='__main__':
    np.set_printoptions(precision=2,suppress=True,linewidth=220)
    Fin=build(reg='noreg' not in sys.argv); E=mape(Fin[tr],Ytr)
    for i,t in enumerate(T): print('   ',t,np.round(E[:,i],1))
