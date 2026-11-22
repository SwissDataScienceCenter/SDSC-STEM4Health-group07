import sys,os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from blend import *
from smooth2 import lowpass
def smooth_cells(P,ref,w=1/3,passes=3,targets=(0,1)):
    L=np.log(P).copy()
    for t in targets: L[:,:,t]=ref[:,t]+lowpass(L[:,:,t]-ref[:,t],w,passes)
    return np.exp(L)
def final(members,name,desc,smooth=True,targets=(0,1),passes=3):
    O=[get(n) for n in members]; P=np.exp(np.mean([np.log(o[0]) for o in O],0)); Pt=np.exp(np.mean([np.log(o[1]) for o in O],0))
    if smooth:
        idx=np.where(tr)[0]; pos={r:i for i,r in enumerate(idx)}; LY=np.log(Ytr)
        for ho in folds7(tr):
            ii=[pos[r] for r in ho]; m=np.ones(len(idx),bool); m[ii]=False; P[ii]=smooth_cells(P[ii],LY[m].mean(0),passes=passes,targets=targets)
        Pt=smooth_cells(Pt,LY.mean(0),passes=passes,targets=targets)
    r=register(name,'blend',desc,P,Ytr,Pt); E=mape(P,Ytr)
    for i,t in enumerate(T): print('   ',t,np.round(E[:,i],1))
    return r
H,SS,SE,B='h_m11_consHill_evidence_allb25','s25_search_all_plus_base25','s26_evidence_all_plus_base25','p_m21_re8_kernel_b25'
if __name__=='__main__':
    P2='p2_m21_re8_evidence_all_n1000'
    final([H,SS,B],'b04_hyb_stat_bayes_smoothcells','geometric mean of hybrid (ODE + evidence-weighted residual kernel), statistical kernel model (searched weights) and Bayesian particle ODE; cell-density deviations smoothed in time')
    final([H,SE,B],'b06_nosearch_smoothcells','as b04 but the statistical member uses marginal-likelihood weights: no grid search anywhere')
    final([H,SS,P2],'b09_hyb_stat_bayesresid_smoothcells','geometric mean of hybrid, statistical (searched weights) and Bayesian particle ODE with residual kernel; cells smoothed')
    final([H,SE,P2],'b10_nosearch_bayesresid_smoothcells','as b09 with marginal-likelihood weights in the statistical member (no grid search anywhere)')
    final([H,P2],'b11_mechanistic_only_bayesresid','geometric mean of the two ODE-based models (hybrid + Bayesian particle with residual kernel); cells smoothed')
