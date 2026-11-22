"""How much of the grid-search gain is real? Bootstrap check.
For 1000 resamples of the 49 batches: run the grid search on the resample, score the winner on the batches left out,
and compare with the fixed version-2 setting on the same left-out batches.
 apparent = gain on the batches used for the search (optimistic)
 oob      = gain on left-out batches (pessimistic when candidates are near-tied)
 .632     = 0.368*apparent + 0.632*oob (Efron's compromise)"""
import sys,os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from gridcv import *
def tables(cands,base,j):
    names=list(cands); LB=np.log(base); Yt=Y[:,1:,j]; E=np.zeros((len(names),len(WG),len(trn),13))
    for a,c in enumerate(names):
        LC=np.log(cands[c])
        for b,w in enumerate(WG): E[a,b]=(np.abs(np.exp((1-w)*LB+w*LC)-Yt)/Yt*100)[trn]
    return names,E
if __name__=='__main__':
    B9=full('b09_hyb_stat_bayesresid_smoothcells'); C=candidates(); rng=np.random.default_rng(0); nb=len(trn); R=1000
    FIX={0:('lx0+lp0+V0+fbL|lam0.03|sh1.0',0.5),2:('chain0.50+statespace0.50',0.7),3:('pd:base25+V0+lp0|lam0.03',0.5)}
    draws=[rng.integers(0,nb,nb) for _ in range(R)]
    print('gain of grid search over the fixed setting (negative = better), MAPE points on that target')
    print('%-10s %-6s | apparent |   oob   |  .632  | share of resamples where the search wins on left-out batches'%('target','phases'))
    tot={'one':np.zeros(3),'three':np.zeros(3)}
    for j,key in [(0,0),(1,0),(2,2),(3,3)]:
        cands={k:(v[:,:,j] if v.ndim==3 else v) for k,v in C[key].items()}; names,E=tables(cands,B9[:,:,j],j)
        fa,fb=names.index(FIX[key][0]),WG.index(FIX[key][1])
        for pn,ph in PH.items():
            app=[];oob=[]
            for ib in draws:
                ob=np.setdiff1d(np.arange(nb),ib); a_s=0; o_s=0
                for s in ph:
                    Eb=E[:,:,:,s].mean(3); M=Eb[:,:,ib].mean(2); a,b=np.unravel_index(np.argmin(M),M.shape); w=(s.stop-s.start)/13
                    a_s+=w*(M[a,b]-M[fa,fb]); o_s+=w*(Eb[a,b,ob].mean()-Eb[fa,fb,ob].mean())
                app.append(a_s); oob.append(o_s)
            app,oob=np.array(app),np.array(oob); r=np.array([app.mean(),oob.mean(),0.368*app.mean()+0.632*oob.mean()]); tot[pn]+=r/4
            print('%-10s %-6s |  %+.3f  | %+.3f  | %+.3f | %.0f %%'%(T[j],pn,*r,100*(oob<0).mean()),flush=True)
    for pn in tot: print('%-10s %-6s |  %+.3f  | %+.3f  | %+.3f |'%('MEAN',pn,*tot[pn]))
