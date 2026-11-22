import sys,os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from statm import *
RES={'equal':m_kernel_alpha,'search':m_kernel_search,'evidence':m_evidence,
     'evsplit':lambda a,b,c,**k:m_evidence(a,b,c,blocks='split',**k)}
def hybrid(ode,res,names=CORE,tag=''):
    sims=np.load(f'{LAB}/ode/{ode}_sims.npy'); ls=k4_to_52(np.log(sims[:,:,1:].reshape(-1,13,4))).reshape(8,len(Y),52)
    pos={r:i for i,r in enumerate(np.where(tr)[0])}; MU=np.zeros((tr.sum(),52)); SD=np.zeros((tr.sum(),52)); fn=RES[res]
    for f,ho in enumerate(folds7(tr)):
        m=tr.copy(); m[ho]=False; te=np.zeros_like(tr); te[ho]=True
        mu,sd=fn(m,te,Y52[m]-ls[f][m],names=names); ii=[pos[r] for r in ho]; MU[ii]=mu+ls[f][te]; SD[ii]=sd
    mu,sd=fn(tr,test,Y52[tr]-ls[7][tr],names=names); mut=mu+ls[7][test]
    P=post(f52_to_k4(MU),f52_to_k4(SD),fl[tr]); Pt=post(f52_to_k4(mut),f52_to_k4(np.tile(sd,(test.sum(),1))),fl[test])
    nm=f'h_{ode}_{res}{tag}'; np.savez(f'{LAB}/oof/{nm}__mu.npz',MU=MU,SD=SD,mut=mut,sdt=sd)
    return register(nm,'hybrid',f'ODE {ode} + kernel model ({res} weights) on its log residuals, shift + rule truncation',P,Ytr,Pt)
if __name__=='__main__':
    for ode in sys.argv[2:]:
        for res in sys.argv[1].split(','): hybrid(ode,res)
