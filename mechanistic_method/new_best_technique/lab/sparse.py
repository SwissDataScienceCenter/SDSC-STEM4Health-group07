"""Deliberately small models: a handful of physically motivated signals, ridge, nothing searched.
cells:     one number per batch (its mean log-deviation from the typical profile over all days, VCD and TCD together)
           regressed on day-0 VCD, day-0 byproduct, initial volume and the late feed-B dosing factor; each day then gets
           loading_k * offset (loadings fitted on leave-one-out offsets).
byproduct: same construction with the unlogged volume gain of days 2-5 (base addition) and initial volume.
Every fit is repeated inside each fold."""
import sys,os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import *
bs,A,Y,tt,test=load(); tr=~test; Ytr=Y[tr][:,1:]; n=len(bs)
G,fl,D=feature_groups(A,Y,tt); V,F,dVr=D['V'],D['F'],D['dVr']
def feats():
    FB=np.log(F[:,1:13,0]/V[:,1:13]); Z=(FB-FB.mean(0))/FB.std(0)
    f=dict(lx0=np.log(Y[:,0,0]),lp0=np.log(Y[:,0,3]),ls0=np.log(Y[:,0,2]),V0=V[:,0],fbL=Z[:,3:].mean(1),fbE=Z[:,:3].mean(1),base25=dVr[:,1:5].sum(1),
           base13=dVr[:,1:3].sum(1),base35=dVr[:,3:5].sum(1),base58=dVr[:,5:8].sum(1),fl11=fl[:,3],fl12=fl[:,4],nC=D['N'][:,2:13,2].sum(1))
    return {k:(v-v.mean())/v.std() for k,v in f.items()}
FE=feats()
def ridge_fit(X,y,lam):
    Z=np.c_[X,np.ones(len(y))]; R=lam*len(y)*np.diag(np.r_[np.ones(X.shape[1]),0]); Ai=np.linalg.inv(Z.T@Z+R); co=Ai@Z.T@y
    h=np.einsum('ij,jk,ik->i',Z,Ai,Z); loo=y-(y-Z@co)/(1-h); return co,loo
def rank1(fit,te,cols,targets,days=slice(0,13),lam=0.03,shrink=0.0,clip=3.0):
    """returns mu (len(te),13,len(targets)) and sd (13,len(targets)) on log scale"""
    X=np.c_[tuple(FE[c] for c in cols)]; L=np.log(Y[fit][:,1:][:,:,targets]); ref=np.median(L,0)
    mad=1.4826*np.median(np.abs(L-ref),0); Lc=np.clip(L,ref-clip*mad,ref+clip*mad); ref=Lc.mean(0); dev=Lc-ref
    d=dev[:,days].mean((1,2)); co,dl=ridge_fit(X[fit],d,lam); dh=np.c_[X[te],np.ones(len(te))]@co
    beta=(dev*dl[:,None,None]).sum(0)/(dl**2).sum(); beta=(1-shrink)*beta+shrink*1.0
    res=dev-beta*dl[:,None,None]; sd=1.4826*np.median(np.abs(res-np.median(res,0)),0)
    return ref+beta*dh[:,None,None],sd
def perday(fit,te,cols,targets,lam=0.03,clip=3.0):
    X=np.c_[tuple(FE[c] for c in cols)]; L=np.log(Y[fit][:,1:][:,:,targets]); ref=np.median(L,0)
    mad=1.4826*np.median(np.abs(L-ref),0); Lc=np.clip(L,ref-clip*mad,ref+clip*mad)
    mu=np.zeros((len(te),13,len(targets))); sd=np.zeros((13,len(targets)))
    for k in range(13):
        for j in range(len(targets)):
            co,loo=ridge_fit(X[fit],Lc[:,k,j],lam); mu[:,k,j]=np.c_[X[te],np.ones(len(te))]@co; r=Lc[:,k,j]-loo; sd[k,j]=1.4826*np.median(np.abs(r-np.median(r)))
    return mu,sd
def run(fn,targets,**kw):
    """OOF + test point predictions exp(mu - sd^2) for the given targets"""
    trn=np.where(tr)[0]; out=np.zeros((n,13,len(targets)))
    for te in folds7(tr):
        mu,sd=fn(np.setdiff1d(trn,te),te,targets=targets,**kw); out[te]=np.exp(mu-sd**2)
    te=np.where(test)[0]; mu,sd=fn(trn,te,targets=targets,**kw); out[te]=np.exp(mu-sd**2)
    return out
def score(P,targets): return (np.abs(P[tr]-Ytr[:,:,targets])/Ytr[:,:,targets]*100).mean(0)
if __name__=='__main__':
    np.set_printoptions(precision=2,suppress=True,linewidth=220)
    z=np.load(f'{LAB}/oof/b09_hyb_stat_bayesresid_smoothcells.npz'); B9=np.zeros((n,13,4)); B9[tr]=z['oof']; B9[test]=z['test']
    def rep(nm,P,targets):
        e=score(P,targets); eb=score(np.sqrt(P*B9[:,:,targets]),targets); print('%-44s alone %s blend %s'%(nm,e.mean(0),eb.mean(0)))
    print('b09',score(B9,[0,1,2,3]).mean(0))
    C=['lx0','lp0','V0','fbL']
    for lam in (0.01,0.03,0.1): rep(f'cells rank1 lam{lam}',run(rank1,[0,1],cols=C,lam=lam),[0,1])
    rep('cells rank1 shrink .5',run(rank1,[0,1],cols=C,shrink=0.5),[0,1]); rep('cells rank1 const',run(rank1,[0,1],cols=C,shrink=1.0),[0,1])
    rep('cells rank1 no fbL',run(rank1,[0,1],cols=['lx0','lp0','V0']),[0,1]); rep('cells rank1 +nC',run(rank1,[0,1],cols=C+['nC']),[0,1]); rep('cells rank1 +ls0',run(rank1,[0,1],cols=C+['ls0']),[0,1])
    rep('cells perday',run(perday,[0,1],cols=C),[0,1])
    Pc=['base25','V0']
    for lam in (0.01,0.03,0.1): rep(f'byp perday lam{lam}',run(perday,[3],cols=Pc,lam=lam),[3])
    rep('byp rank1 d3-12',run(rank1,[3],cols=Pc,days=slice(2,12)),[3]); rep('byp rank1 all',run(rank1,[3],cols=Pc),[3])
    rep('byp perday +flags',run(perday,[3],cols=Pc+['fl11','fl12']),[3]); rep('byp perday base13,35',run(perday,[3],cols=['base13','base35','V0']),[3]); rep('byp perday +base58',run(perday,[3],cols=Pc+['base58']),[3])
    rep('byp perday +lp0',run(perday,[3],cols=Pc+['lp0']),[3]); rep('byp perday +lx0',run(perday,[3],cols=Pc+['lx0']),[3])
