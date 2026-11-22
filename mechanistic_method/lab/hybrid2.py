import sys,os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from hybrid import *
def feat_hybrid(ode,names=CORE,tag=''):
    """ODE-simulated trajectories as an extra kernel group in the statistical model"""
    sims=np.load(f'{LAB}/ode/{ode}_sims.npy'); pos={r:i for i,r in enumerate(np.where(tr)[0])}; MU=np.zeros((tr.sum(),52)); SD=np.zeros((tr.sum(),52))
    def Kf(f): K=dict(KALL); K['ode']=kern(np.log(sims[f][:,1:].reshape(len(Y),-1))); return K
    for f,ho in enumerate(folds7(tr)):
        m=tr.copy(); m[ho]=False; te=np.zeros_like(tr); te[ho]=True
        mu,sd=m_kernel_search(m,te,Y52[m],names=list(names)+['ode'],K=Kf(f)); ii=[pos[r] for r in ho]; MU[ii]=mu; SD[ii]=sd
    mut,sdt=m_kernel_search(tr,test,Y52[tr],names=list(names)+['ode'],K=Kf(7))
    P=post(f52_to_k4(MU),f52_to_k4(SD),fl[tr]); Pt=post(f52_to_k4(mut),f52_to_k4(np.tile(sdt,(test.sum(),1))),fl[test])
    nm=f'hf_{ode}{tag}'; np.savez(f'{LAB}/oof/{nm}__mu.npz',MU=MU,SD=SD,mut=mut,sdt=sdt)
    return register(nm,'hybrid',f'kernel model (searched weights) with ODE {ode} simulated trajectories as an extra feature group',P,Ytr,Pt)
if __name__=='__main__':
    feat_hybrid('m01_base'); feat_hybrid('m11_consHill'); feat_hybrid('m01_base',names=ALLG,tag='_allg')
    RES['search_allg']=lambda a,b,c,**k:m_kernel_search(a,b,c,names=ALLG)
    RES['evidence_allg']=lambda a,b,c,**k:m_evidence(a,b,c,names=ALLG)
    for ode in ['m11_consHill','m03_feedD']:
        hybrid(ode,'search_allg'); hybrid(ode,'evidence_allg'); hybrid(ode,'evsplit')
