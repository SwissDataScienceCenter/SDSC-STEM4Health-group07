"""Random-subspace ensemble of the marginal-likelihood kernel model: each member sees a random half of the signal groups.
Averaging members stabilises the group weights (the single models differ mostly by which groups they happen to trust)."""
import sys,os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from statm import *
def m_subspace(trm,tem,Yt,M=30,frac=0.5,blocks='whole',seed=0,always=('base25',)):
    rng=np.random.default_rng(seed); names=[g for g in ALLG+['base25']]; mus=[];sds=[]
    for m in range(M):
        sub=list(rng.choice([g for g in names if g not in always],int(frac*len(names)),replace=False))+list(always)
        mu,sd=m_evidence(trm,tem,Yt,names=sub,blocks=blocks); mus.append(mu); sds.append(sd)
    return np.mean(mus,0),np.mean(sds,0)
if __name__=='__main__':
    t0=time.time()
    for nm,kw in [('s27_subspace_whole',dict(blocks='whole')),('s28_subspace_split',dict(blocks='split')),('s29_subspace_target',dict(blocks='target'))]:
        run(nm,'statistical','random-subspace ensemble (30 members, half of the signal groups each) of the marginal-likelihood kernel model, blocks='+kw['blocks'],lambda a,b,c,kw=kw:m_subspace(a,b,c,**kw)); print(time.time()-t0,flush=True)
