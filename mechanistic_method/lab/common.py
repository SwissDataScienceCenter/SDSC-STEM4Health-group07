import numpy as np, pandas as pd, json, os, sys, time, warnings
from scipy.stats import norm
warnings.filterwarnings('ignore')
LAB=os.path.dirname(os.path.abspath(__file__)); PROJ=os.path.dirname(LAB)+'/'
T=['vcd','tcd','substrate','byproduct']
THR={7:1.95,8:1.95,9:1.95,11:0.967,12:0.967}; FLAGCOL={7:0,8:1,9:2,11:3,12:4}
def load():
    d=pd.read_csv(PROJ+'bioreaction_remaining.csv')
    bs=list(d.batch_id.unique()); n=len(bs); M=d.groupby('batch_id').size().max()
    A={k:np.zeros((n,M)) for k in ['dt','fb','fs','fc','fd']}
    A['V']=np.full((n,M),np.nan); A['k']=np.full((n,M),-1,int)
    Y=np.full((n,14,4),np.nan); tt=np.zeros((n,14)); test=np.zeros(n,bool)
    for b,name in enumerate(bs):
        g=d[d.batch_id==name].sort_values('elapsed_time',kind='stable').reset_index(drop=True)
        test[b]=g.to_predict.any(); t=g.elapsed_time.values
        A['dt'][b,1:len(g)]=np.diff(t)
        for c,cc in [('fb','feed_b'),('fs','feed_substrate'),('fc','feed_c'),('fd','feed_d')]: A[c][b,:len(g)]=g[cc].fillna(0).values
        A['V'][b,:len(g)]=g.volume.values
        si=np.where(g.volume.notna())[0]; assert len(si)==14
        A['k'][b,si]=np.arange(14); Y[b]=g.loc[si,T].values; tt[b]=t[si]
    return bs,A,Y,tt,test
def sub(A,m): return {k:v[m] for k,v in A.items()}
def daily(A,tt):
    """per-day aggregates: V (n,14), F (n,14,4) feeds between sample k and k+1 [fb,fs,fc,fd], counts and timing"""
    n,M=A['dt'].shape; V=np.zeros((n,14)); F=np.zeros((n,14,4)); N=np.zeros((n,14,4)); TF=np.full((n,14,4),np.nan)
    for b in range(n):
        k=-1; t=0.0
        for i in range(M):
            t+=A['dt'][b,i]
            if A['k'][b,i]>=0: k=A['k'][b,i]; V[b,k]=A['V'][b,i]
            else:
                for j,c in enumerate(['fb','fs','fc','fd']):
                    if A[c][b,i]>0:
                        F[b,k,j]+=A[c][b,i]; N[b,k,j]+=1
                        if np.isnan(TF[b,k,j]): TF[b,k,j]=t-tt[b,k]
    return V,F,N,TF
def sd_(z):
    z=np.asarray(z,float); z=z[:,None] if z.ndim==1 else z
    s=z.std(0); return (z-z.mean(0))/np.where(s>1e-12,s,1)
def feature_groups(A,Y,tt):
    V,F,N,TF=daily(A,tt)
    dVr=np.diff(V,axis=1)-F[:,:-1].sum(2)
    fl=np.c_[F[:,7:10,1]>0,F[:,11:13,1]/V[:,11:13]>0.005].astype(float)
    G={}
    G['lx0']=np.log(Y[:,0,0]); G['ls0']=np.log(Y[:,0,2]); G['lp0']=np.log(Y[:,0,3]); G['via0']=Y[:,0,0]/Y[:,0,1]
    G['vol0']=V[:,0]; G['dVr_e']=dVr[:,1:5]; G['dVr_l']=dVr[:,5:]; G['dVr0']=dVr[:,:1]; G['volL']=V[:,5:]-V[:,:1]
    G['t']=tt[:,1:]-np.arange(1,14); G['dt']=np.diff(tt,axis=1)
    G['fb']=F[:,1:13,0]/V[:,1:13]; G['fbt']=TF[:,1:13,0]
    G['fc']=F[:,2:13,2]; G['fcn']=N[:,2:13,2]; G['fct']=np.nan_to_num(TF[:,2:13,2])
    G['fd']=F[:,[5,8,10],3]/V[:,[5,8,10]]
    G['fs_m']=fl[:,:3]; G['fs_l']=fl[:,3:]
    G={k:(v[:,None] if v.ndim==1 else v) for k,v in G.items()}
    return G,fl,dict(V=V,F=F,N=N,TF=TF,dVr=dVr)
def folds7(tr):
    idx=np.where(tr)[0]; return [idx[j::7] for j in range(7)]
def k4_to_52(a): return a.transpose(0,2,1).reshape(len(a),52)
def f52_to_k4(a): return a.reshape(len(a),4,13).transpose(0,2,1)
_Q=(np.arange(400)+0.5)/400
def tpoint(mu,sd,lo,hi):
    a,b=norm.cdf((lo-mu)/sd),norm.cdf((hi-mu)/sd)
    if b-a<1e-9: return float(np.exp(lo if np.isfinite(lo) else hi))
    y=np.exp(mu+sd*norm.ppf(a+(b-a)*_Q)); c=np.cumsum(1/y)/(1/y).sum()
    return y[np.searchsorted(c,0.5)]
def post(mu,sd,fl,c=1.0,trunc=True,sdscale=0.7):
    """mu (n,13,4) log-scale predictions, sd (13,4) or (n,13,4) -> point predictions"""
    sd=np.broadcast_to(sd,mu.shape); P=np.exp(mu-c*sd**2)
    if trunc:
        for k,thr in THR.items():
            lt=np.log(thr)
            for i in range(len(mu)):
                low=fl[i,FLAGCOL[k]]>0
                P[i,k-1,2]=tpoint(mu[i,k-1,2],max(sd[i,k-1,2]*sdscale,1e-3),-np.inf if low else lt,lt if low else np.inf)
    return P
def mape(P,Yk4): return (np.abs(P-Yk4)/Yk4*100).mean(0)      # (13,4); Yk4 = Y[:,1:]
def register(name,family,desc,P_oof,Ytr,P_test=None,extra=None,quiet=False):
    E=mape(P_oof,Ytr); rec=dict(name=name,family=family,desc=desc,mean=float(E.mean()),time=time.strftime('%H:%M:%S'))
    for i,t in enumerate(T): rec[t]=float(E[:,i].mean())
    if extra: rec.update(extra)
    np.savez(f'{LAB}/oof/{name}.npz',oof=P_oof,test=P_test if P_test is not None else np.zeros(0))
    with open(f'{LAB}/results.jsonl','a') as f: f.write(json.dumps(rec)+'\n')
    if not quiet: print(f"[{name}] "+' '.join(f'{t}:{rec[t]:.2f}' for t in T)+f" | mean {rec['mean']:.3f}",flush=True)
    return rec
def results():
    rows=[json.loads(l) for l in open(f'{LAB}/results.jsonl')]
    d=pd.DataFrame(rows).drop_duplicates('name',keep='last'); return d.sort_values('mean').reset_index(drop=True)
