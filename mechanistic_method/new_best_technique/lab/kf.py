"""Latent state-space model for substrate.
True substrate x_k follows x_k+1 = a_k x_k + c_k.u_k + process noise (variance q_k); the recorded value is
y_k = x_k + assay noise (variance r_k). Parameters by EM (Kalman smoother + closed-form M-step, ridge on covariates).
Held-out batches are simulated with particles from day 0; the logged substrate-feed decisions act as a likelihood on the
recorded value (bolus on days 7-9 iff y < 1.95; 0.6 % vs 0.4 % on days 11-12 iff y < 0.967).
Point prediction = weighted median of simulated recorded values with weights w/y (MAPE-optimal)."""
import sys,os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from chain import *
def covs(k,idx,PB,cfg):
    c=[F[idx,k,0]/V[idx,k],tt[idx,k+1]-tt[idx,k]]
    if k in (7,8,9,11,12) and not (cfg['cb'] and k<=9): c.append(F[idx,k,1]/V[idx,k])
    if cfg['xb']: c+=[PB[idx,k,0],PB[idx,k+1,0]]
    if cfg['pb']: c+=[PB[idx,k,3],PB[idx,k+1,3]]
    if cfg['vb']: c+=[dVr[idx,k]]
    return np.c_[tuple(c)]
KDEF=dict(cb=0,xb=1,pb=1,vb=0,lam=1e-3,N=4000,iters=300,rfloor=0.005,qfloor=1e-5,tie=0,seed=0,flags=1,sdq=1.0,rinit=0.03,nu=0)
def em(fit,PB,cfg):
    y=Y[fit][:,:,2]; nb=len(fit); mlev=y.mean(0)
    cb=bolus_coef(fit,PB,{**DEF,'xb':cfg['xb'],'pb':cfg['pb'],'vb':cfg['vb']}) if cfg['cb'] else 0.0; OFF=cb*F[fit,:13,1]/V[fit,:13]; OFF[:,10:]=0
    U=[]; st=[]
    for k in range(13):
        u=covs(k,fit,PB,cfg); m=u.mean(0); s=u.std(0); s[s<1e-12]=1; st.append((m,s)); U.append(np.c_[(u-m)/s,np.ones(nb)])
    th=[]; q=np.zeros(13); r=(cfg['rinit']*mlev)**2
    for k in range(13):
        Z=np.c_[y[:,k],U[k]]; co=np.linalg.lstsq(Z,y[:,k+1]-OFF[:,k],rcond=None)[0]; th.append(co); q[k]=max(np.var(y[:,k+1]-OFF[:,k]-Z@co)-r[k+1]-co[0]**2*r[k],0.2*np.var(y[:,k+1]-OFF[:,k]-Z@co))
    wq=np.ones((nb,13))
    for it in range(cfg['iters']):
        m=np.zeros((nb,14)); P=np.zeros((nb,14)); mp=np.zeros((nb,14)); Pp=np.zeros((nb,14))
        m[:,0]=y[:,0]; P[:,0]=r[0]
        for k in range(13):
            a=th[k][0]; qk=q[k]/wq[:,k]
            mp[:,k+1]=a*m[:,k]+U[k]@th[k][1:]+OFF[:,k]; Pp[:,k+1]=a*a*P[:,k]+qk
            K=Pp[:,k+1]/(Pp[:,k+1]+r[k+1]); m[:,k+1]=mp[:,k+1]+K*(y[:,k+1]-mp[:,k+1]); P[:,k+1]=(1-K)*Pp[:,k+1]
        ms=m.copy(); Ps=P.copy(); C=np.zeros((nb,13))
        for k in range(12,-1,-1):
            J=P[:,k]*th[k][0]/Pp[:,k+1]; ms[:,k]=m[:,k]+J*(ms[:,k+1]-mp[:,k+1]); Ps[:,k]=P[:,k]+J*J*(Ps[:,k+1]-Pp[:,k+1]); C[:,k]=J*Ps[:,k+1]
        for k in range(13):
            w=wq[:,k]; Z=np.c_[ms[:,k],U[k]]; p=Z.shape[1]
            Ezz=(Z*w[:,None]).T@Z; Ezz[0,0]+=(w*Ps[:,k]).sum(); x1=ms[:,k+1]-OFF[:,k]; Ezx=(Z*w[:,None]).T@x1; Ezx[0]+=(w*C[:,k]).sum()
            R=cfg['lam']*nb*np.diag(np.r_[0,np.ones(p-2),0]); t=np.linalg.solve(Ezz+R,Ezx); th[k]=t
            e2=(x1-Z@t)**2+Ps[:,k+1]-2*t[0]*C[:,k]+t[0]**2*Ps[:,k]
            if cfg['nu']>0: wq[:,k]=(cfg['nu']+1)/(cfg['nu']+e2/q[k])
            q[k]=max((w*e2).mean(),cfg['qfloor'])
        d=((y-ms)**2+Ps).mean(0)
        if cfg['tie']:
            Xr=np.c_[np.ones(14),mlev**2]; from scipy.optimize import nnls
            co,_=nnls(Xr/d[:,None],np.ones(14)); r=np.maximum(Xr@co,(cfg['rfloor']*mlev)**2)
        else: r=np.maximum(d,(cfg['rfloor']*mlev)**2)
    return dict(th=th,q=q,r=r,st=st,cb=cb)
def simulate(mod,te,PB,cfg):
    rng=np.random.default_rng(cfg['seed']); N=cfg['N']; m=len(te); th,q,r,st=mod['th'],mod['q'],mod['r'],mod['st']
    x=Y[te,0,2][:,None]+np.sqrt(r[0])*rng.standard_normal((m,N)); W=np.ones((m,N)); ys=np.zeros((m,N,13))
    for k in range(13):
        u=(covs(k,te,PB,cfg)-st[k][0])/st[k][1]; mu=th[k][0]*x+(np.c_[u,np.ones(m)]@th[k][1:]+(mod['cb']*F[te,k,1]/V[te,k] if k<=9 else 0))[:,None]
        qk=cfg['sdq']**2*q[k]; rk=r[k+1]; tot=np.sqrt(qk+rk)
        if cfg['flags'] and (k+1) in THR:
            thr=THR[k+1]; low=(fl[te,FLAGCOL[k+1]]>0)[:,None]; c=norm.cdf((thr-mu)/tot); pm=np.where(low,c,1-c)
            uu=rng.uniform(size=(m,N)); qq=np.where(low,uu*c,c+uu*(1-c)); yk=mu+tot*norm.ppf(np.clip(qq,1e-12,1-1e-12)); yk=np.where(low,np.minimum(yk,thr),np.maximum(yk,thr))
            W=W*np.maximum(pm,1e-300); W/=W.sum(1,keepdims=True)
        else: yk=mu+tot*rng.standard_normal((m,N))
        x=mu+qk/(qk+rk)*(yk-mu)+np.sqrt(qk*rk/(qk+rk))*rng.standard_normal((m,N))
        ys[:,:,k]=np.maximum(yk,0.03)
    out=np.zeros((m,13)); ess=1/((W/W.sum(1,keepdims=True))**2).sum(1)
    for i in range(m):
        for k in range(13): out[i,k]=wmed(ys[i,:,k],W[i]/ys[i,:,k])
    return out,ess
def kf(base,**kw):
    cfg={**KDEF,**kw}; PB=base_preds(base); out=PB[:,1:,2].copy(); trn=np.where(tr)[0]; essmin=1e9
    for te in folds7(tr):
        mod=em(np.setdiff1d(trn,te),PB,cfg); o,ess=simulate(mod,te,PB,cfg); out[te]=o; essmin=min(essmin,ess.min())
    te=np.where(test)[0]; mod=em(trn,PB,cfg); o,ess=simulate(mod,te,PB,cfg); out[te]=o
    return out[tr],out[test],essmin,mod
if __name__=='__main__':
    np.set_printoptions(precision=2,suppress=True,linewidth=220)
    base=sys.argv[1]; PB=base_preds(base); YS=Ytr[:,:,2]; e0=(np.abs(PB[tr,1:,2]-YS)/YS*100).mean(0); print('base',e0,e0.mean())
    cfgs=[dict(),dict(vb=1),dict(lam=1e-2),dict(rfloor=0.01),dict(rfloor=0.02),dict(qfloor=1e-3)] if len(sys.argv)<3 else [eval(a) for a in sys.argv[2:]]
    for kw in cfgs:
        o,ot,ess,mod=kf(base,**kw); e=(np.abs(o-YS)/YS*100).mean(0); print(kw,'S %.3f ess %d'%(e.mean(),ess),e)
        for w in (0.5,0.7):
            b=np.exp(w*np.log(o)+(1-w)*np.log(PB[tr,1:,2])); print('     blend w=%.1f: %.3f'%(w,(np.abs(b-YS)/YS*100).mean()))
        print('     sd_meas/level',np.sqrt(mod['r'])/Y[tr][:,:,2].mean(0)); print('     sd_proc',np.sqrt(mod['q']),' a',np.array([t[0] for t in mod['th']]))
