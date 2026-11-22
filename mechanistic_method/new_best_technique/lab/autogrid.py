"""Automatic model + hyper-parameter search (scikit-learn GridSearchCV) for cells and byproduct.
Outer loop: the usual 7 batch-wise folds. Inside each outer fold, GridSearchCV picks the hyper-parameters with a 6-fold
inner CV on the training batches only, so the out-of-fold predictions are honest (nested CV).
 cells     : target = one log offset per batch (mean deviation of VCD and TCD from the typical profile)
 byproduct : target = log byproduct, one search per day
 'auto'    : per target, the family with the best inner-CV score in that outer fold
Writes oof/a_<family>_<features>.npz (cells + byproduct; substrate copied from version 2) and autogrid_result.json."""
import sys,os,json,warnings; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import *
import sparse as sp
from joblib import Parallel,delayed
from sklearn.model_selection import GridSearchCV,KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,Lasso,ElasticNet,HuberRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
bs,A,Y,tt,test=load(); tr=~test; Ytr=Y[tr][:,1:]; n=len(bs); trn=np.where(tr)[0]; tei=np.where(test)[0]
G,fl,D=feature_groups(A,Y,tt); V,F,dVr,N=D['V'],D['F'],D['dVr'],D['N']
def feature_sets():
    cur=np.c_[tuple(sp.FE[k] for k in ['lx0','lp0','ls0','V0','fbL','fbE','base25','base58','fl11','fl12','nC'])]
    to=tt[:,1:]-np.arange(1,14); wide=np.c_[cur,np.log(F[:,1:13,0]/V[:,1:13]),dVr,N[:,2:13,2],fl[:,:3],to,Y[:,0,0]/Y[:,0,1]]
    return {'curated':cur,'wide':wide}
FAM={'ridge':(lambda:Ridge(),{'ridge__alpha':list(np.logspace(-2,3,12))}),
     'lasso':(lambda:Lasso(max_iter=20000),{'lasso__alpha':list(np.logspace(-4,-1,12))}),
     'enet':(lambda:ElasticNet(max_iter=20000),{'elasticnet__alpha':list(np.logspace(-4,-1,8)),'elasticnet__l1_ratio':[.2,.5,.8]}),
     'huber':(lambda:HuberRegressor(max_iter=2000),{'huberregressor__alpha':list(np.logspace(-2,3,8))}),
     'svr':(lambda:SVR(),{'svr__C':[.01,.03,.1,.3,1,3],'svr__gamma':['scale',.03,.01,.003],'svr__epsilon':[.01,.03]}),
     'krr':(lambda:KernelRidge(kernel='rbf'),{'kernelridge__alpha':[.03,.1,.3,1,3],'kernelridge__gamma':[.001,.003,.01,.03]}),
     'pls':(lambda:PLSRegression(scale=False),{'plsregression__n_components':[1,2,3,4]}),
     'gbr':(lambda:GradientBoostingRegressor(subsample=.7,random_state=0),{'gradientboostingregressor__n_estimators':[50,150],'gradientboostingregressor__max_depth':[1,2],'gradientboostingregressor__learning_rate':[.03,.1]}),
     'rf':(lambda:RandomForestRegressor(200,random_state=0),{'randomforestregressor__min_samples_leaf':[2,4,8],'randomforestregressor__max_features':[.3,.6]})}
def clip3(y):
    m=np.median(y); s=1.4826*np.median(np.abs(y-m)); return np.clip(y,m-3*s,m+3*s)
def one(fam,X,fit,pred,y):
    """grid-search on batches `fit`, predict batches `pred`; returns prediction, inner-CV score, robust residual sd, chosen params"""
    warnings.filterwarnings('ignore'); mk,grid=FAM[fam]; yc=clip3(y)
    gs=GridSearchCV(make_pipeline(StandardScaler(),mk()),grid,scoring='neg_mean_absolute_error',cv=KFold(6,shuffle=True,random_state=0),n_jobs=1).fit(X[fit],yc)
    p=np.ravel(gs.predict(X[pred])); sd=-gs.best_score_*1.2533                      # MAE -> sd for a normal
    return p,gs.best_score_,sd,{k.split('__')[1]:v for k,v in gs.best_params_.items()}
def job(fam,fs,X,kind,k,fold):
    fit=np.setdiff1d(trn,fold) if fold is not None else trn; pred=fold if fold is not None else tei
    L=np.log(Y[fit][:,1:])
    if kind=='cells':
        ref=L[:,:,:2].mean(0); y=(L[:,:,:2]-ref).mean((1,2)); p,sc,sd,bp=one(fam,X,fit,pred,y)
        res=(L[:,:,:2]-ref).std(0); res=np.sqrt(np.maximum(res**2-y.var(),0)+sd**2)   # per-day sd: day scatter around the offset + offset error
        out=np.exp(ref[None]+p[:,None,None]-res[None]**2)
    else:
        p,sc,sd,bp=one(fam,X,fit,pred,L[:,k,3]); out=np.exp(p-sd**2)
    return fam,fs,kind,k,(None if fold is None else tuple(fold)),pred,out,sc,bp
if __name__=='__main__':
    FS=feature_sets(); folds=folds7(tr)+[None]; jobs=[]
    for fs,X in FS.items():
        for fam in FAM:
            for fold in folds:
                jobs.append((fam,fs,X,'cells',-1,fold)); jobs+= [(fam,fs,X,'byp',k,fold) for k in range(13)]
    t0=time.time(); R=Parallel(n_jobs=10,batch_size=4)(delayed(job)(*j) for j in jobs); print('searches',len(R),'in %.0f s'%(time.time()-t0),flush=True)
    base=np.load(f'{LAB}/oof/b13_v2_small_plus_substrate_chain.npz'); V2=np.zeros((n,13,4)); V2[tr]=base['oof']; V2[test]=base['test']
    P={}; SC={}; BP={}
    for fam,fs,kind,k,fold,pred,out,sc,bp in R:
        key=(fam,fs); P.setdefault(key,V2.copy())
        if kind=='cells': P[key][pred,:,:2]=out
        else: P[key][pred,k,3]=out
        SC[(fam,fs,kind,k,fold)]=sc
        if fold is None: BP.setdefault(f'{fam}|{fs}',{})[f'{kind}{k+1 if k>=0 else ""}']=bp
    # 'auto': per outer fold and target, the family/feature-set with the best inner-CV score
    AUTO=V2.copy(); pick={}
    for fold in folds:
        fk=None if fold is None else tuple(fold); pred=tei if fold is None else fold
        for kind,k in [('cells',-1)]+[('byp',k) for k in range(13)]:
            best=max(P,key=lambda key: SC[(key[0],key[1],kind,k,fk)])
            if kind=='cells': AUTO[pred,:,:2]=P[best][pred,:,:2]
            else: AUTO[pred,k,3]=P[best][pred,k,3]
            if fold is None: pick[f'{kind}{k+1 if k>=0 else ""}']='|'.join(best)
    P[('auto','best')]=AUTO
    e=lambda X,j: float((np.abs(X[tr][:,:,j]-Ytr[:,:,j])/Ytr[:,:,j]*100).mean())
    rows=[]
    print('%-8s %-8s | alone: vcd tcd byp | blended 1/3 into v2: vcd tcd byp'%('family','features'))
    for key in sorted(P,key=lambda key:e(P[key],0)+e(P[key],1)+e(P[key],3)):
        X=P[key]; Bm=np.exp((2*np.log(V2)+np.log(X))/3)
        print('%-8s %-8s | %.3f %.3f %.3f | %.3f %.3f %.3f'%(*key,e(X,0),e(X,1),e(X,3),e(Bm,0),e(Bm,1),e(Bm,3)),flush=True)
        rows.append(dict(family=key[0],features=key[1],vcd=e(X,0),tcd=e(X,1),byproduct=e(X,3),vcd_blend=e(Bm,0),tcd_blend=e(Bm,1),byproduct_blend=e(Bm,3)))
        np.savez(f'{LAB}/oof/a_{key[0]}_{key[1]}.npz',oof=X[tr],test=X[test])
    print('v2 fixed                 | %.3f %.3f %.3f'%(e(V2,0),e(V2,1),e(V2,3))); print('auto picks on all data:',pick)
    json.dump(dict(rows=rows,auto_picks=pick,best_params=BP),open(f'{LAB}/autogrid_result.json','w'),indent=1,default=str)
