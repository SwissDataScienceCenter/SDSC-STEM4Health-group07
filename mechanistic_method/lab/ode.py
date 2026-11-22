import sys,os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import *
from scipy.optimize import least_squares
bs,A,Y,tt,test=load(); tr=~test; G,fl,D=feature_groups(A,Y,tt); Ytr=Y[tr][:,1:]
M_=A['dt'].shape[1]; NSUBI=np.maximum(1,np.ceil(A['dt'].max(0)/0.34)).astype(int)
# ---- sensor sets (standardised over all 61 batches)
def zsets(name):
    V,F,N,dVr=D['V'],D['F'],D['N'],D['dVr']
    z7=np.c_[sd_(np.log(F[:,4:10,0]/V[:,4:10])).mean(1),F[:,5:11,2].sum(1),dVr[:,2]+dVr[:,3],V[:,0],fl[:,:3].max(1),fl[:,3],fl[:,4]]
    if name=='none': return np.zeros((len(V),0)),[]
    if name=='z7': return sd_(z7),['feedB','feedC','base','V0','low7_9','low11','low12']
    if name=='z4': return sd_(z7[:,:4]),['feedB','feedC','base','V0']
    if name=='z10':
        fbe=sd_(np.log(F[:,1:4,0]/V[:,1:4])).mean(1); fbl=sd_(np.log(F[:,10:13,0]/V[:,10:13])).mean(1); fce=F[:,2:5,2].sum(1)
        return sd_(np.c_[z7,fbe,fbl,fce]),['feedB','feedC','base','V0','low7_9','low11','low12','feedB_early','feedB_late','feedC_early']
    raise KeyError(name)
BASE_SPEC={'mumax':(.75,.3,3,'log'),'Ks':(.05,.02,5,'log'),'KI':(36.,3,2000,'log'),'nI':(1.8,.3,5,'log'),
      'kd0':(.001,1e-5,.2,'log'),'kd1':(4e-4,1e-6,.01,'log'),'klys':(.8,1e-2,10,'log'),
      'mS':(.13,.02,.6,'log'),'Km':(.15,.02,8,'log'),'qov':(.42,.05,20,'log'),'KIo':(3.9,.1,1e3,'log'),'no':(1.8,.3,6,'log'),
      'Yov':(1.15,.1,3,'log'),'kc':(.017,1e-4,2,'log'),'Kg':(2.,.05,100,'log'),
      'cb':(62.,25,150,'log'),'cs':(265.,100,600,'log'),
      'rx':(.2,0,1,'lin'),'rxp':(.12,-2,2,'lin'),'rxv':(.09,-3,3,'lin'),'rs':(.4,0,1.2,'lin'),'rp':(1.,0,1.2,'lin')}
class Model:
    name='base'; desc='base ODE'; zset='none'; RE=('mumax','mS','Yov','kc'); lam=10.0; states=('Xv','Xd','S','P','I','t'); nodil=('t',)
    loss='soft_l1'; parent=None
    def spec(self): return dict(BASE_SPEC)
    def init(self,p,Y0,V0,m0):
        lx=m0['lx']+p['rx']*(np.log(Y0[:,0])-m0['lx'])+p['rxp']*(np.log(Y0[:,3])-m0['lp'])+p['rxv']*(V0-m0['v'])/m0['vs']*0.1+p.get('dx0',0.0)
        Xv=np.exp(lx); z=np.zeros(len(Xv))
        return dict(Xv=Xv,Xd=Xv*(Y0[:,1]/Y0[:,0]-1),S=np.exp(m0['ls']+p['rs']*(np.log(Y0[:,2])-m0['ls'])),P=np.exp(m0['lp']+p['rp']*(np.log(Y0[:,3])-m0['lp'])),I=z.copy(),t=z.copy())
    # kinetic pieces (overridable)
    def mu(self,p,s): return p['mumax']*s['S']/(p['Ks']+s['S'])/(1+(s['I']/p['KI'])**p['nI'])
    def kd(self,p,s): return p['kd0']+p['kd1']*s['I']
    def ov(self,p,s): return p['qov']*s['S']/(0.05+s['S'])/(1+(s['I']/p['KIo'])**p['no'])
    def qS(self,p,s,mu,ov): return p['mS']*s['S']/(p['Km']+s['S'])+ov
    def cons(self,p,s): return p['kc']*s['P']*p['Kg']/(p['Kg']+s['S'])
    def extra(self,p,s,d,mu,kd,ov): pass
    def rates(self,p,s):
        s=dict(s); s['S']=np.maximum(s['S'],1e-9); s['P']=np.maximum(s['P'],1e-9); s['I']=np.maximum(s['I'],1e-9)
        mu=self.mu(p,s); kd=self.kd(p,s); ov=self.ov(p,s); Xv=s['Xv']
        d=dict(Xv=(mu-kd)*Xv,Xd=kd*Xv-p['klys']*s['Xd'],S=-self.qS(p,s,mu,ov)*Xv,P=(p['Yov']*ov-self.cons(p,s))*Xv,I=Xv,t=np.ones_like(Xv))
        self.extra(p,s,d,mu,kd,ov); return d
    def feed(self,p,s,fb,fs,fc,fd,Vn): s['S']=s['S']+(p['cb']*fb+p['cs']*fs)/Vn
    def simulate(self,p,A,Y0,m0):
        n=len(Y0); V=A['V'][:,0].copy(); s=self.init(p,Y0,V,m0); out=np.zeros((n,14,4)); rows=np.arange(n); ks=list(s)
        for i in range(M_):
            dt=A['dt'][:,i]
            if dt.max()>0:
                h=dt/NSUBI[i]
                for _ in range(NSUBI[i]):
                    k1=self.rates(p,s); k2=self.rates(p,{k:s[k]+0.5*h*k1[k] for k in ks}); k3=self.rates(p,{k:s[k]+0.5*h*k2[k] for k in ks}); k4=self.rates(p,{k:s[k]+h*k3[k] for k in ks})
                    s={k:s[k]+h/6*(k1[k]+2*k2[k]+2*k3[k]+k4[k]) for k in ks}
            fb,fs,fc,fd=A['fb'][:,i],A['fs'][:,i],A['fc'][:,i],A['fd'][:,i]; f=fb+fs+fc+fd
            if f.max()>0:
                Vn=V+f; r=V/Vn
                for k in ks:
                    if k not in self.nodil: s[k]=s[k]*r
                self.feed(p,s,fb,fs,fc,fd,Vn); V=Vn
            k=A['k'][:,i]; m=k>=0
            if m.any():
                out[rows[m],k[m],0]=s['Xv'][m]; out[rows[m],k[m],1]=s['Xv'][m]+s['Xd'][m]; out[rows[m],k[m],2]=s['S'][m]; out[rows[m],k[m],3]=s['P'][m]
                V=np.where(m,A['V'][:,i],V)
        return np.maximum(out,1e-6)
def stats0(m): return dict(lx=np.log(Y[m,0,0]).mean(),ls=np.log(Y[m,0,2]).mean(),lp=np.log(Y[m,0,3]).mean(),v=A['V'][m,0].mean(),vs=A['V'][m,0].std())
class Fitter:
    def __init__(self,model):
        self.m=model; self.spec=model.spec(); self.names=list(self.spec); self.Z,self.zn=zsets(model.zset); self.NG=len(self.names); self.NC=len(model.RE)*len(self.zn)
    def x0(self,warm=None):
        tr_=lambda s,v: np.log(v) if s[3]=='log' else v
        x=np.array([tr_(self.spec[n],self.spec[n][0]) for n in self.names]+[0.]*self.NC)
        lo=np.array([tr_(self.spec[n],self.spec[n][1]) for n in self.names]+[-.5]*self.NC); hi=np.array([tr_(self.spec[n],self.spec[n][2]) for n in self.names]+[.5]*self.NC)
        if warm:
            for i,n in enumerate(self.names):
                if n in warm['p']: x[i]=tr_(self.spec[n],warm['p'][n])
            if 'C' in warm and warm.get('zn')==self.zn and warm.get('RE')==list(self.m.RE): x[self.NG:]=np.array(warm['C']).ravel()
            elif 'C' in warm and warm.get('RE')==list(self.m.RE):
                C=np.zeros((len(self.m.RE),len(self.zn))); Cw=np.array(warm['C'])
                for j,z in enumerate(self.zn):
                    if z in warm['zn']: C[:,j]=Cw[:,warm['zn'].index(z)]
                x[self.NG:]=C.ravel()
        return np.clip(x,lo+1e-6,hi-1e-6),lo,hi
    def params(self,x,rows):
        p={n:(np.exp(v) if self.spec[n][3]=='log' else v) for n,v in zip(self.names,x[:self.NG])}
        C=x[self.NG:].reshape(len(self.m.RE),len(self.zn))
        if self.NC:
            for j,nm in enumerate(self.m.RE): p[nm]=p[nm]*np.exp(self.Z[rows]@C[j])
        return p,C
    def sim(self,x,rows,m0): return self.m.simulate(self.params(x,rows)[0],sub(A,rows),Y[rows][:,0],m0)
    def fit(self,rows,m0,x0,lo,hi,max_nfev=150):
        Yr=Y[rows]
        def res(x):
            p,C=self.params(x,rows); out=self.m.simulate(p,sub(A,rows),Yr[:,0],m0)
            r=np.log(out[:,1:])-np.log(Yr[:,1:]); via=5*((r[:,:,0]-r[:,:,1]))
            return np.r_[r.ravel(),via.ravel(),np.sqrt(self.m.lam)*C.ravel()]
        kw=dict(loss='soft_l1',f_scale=0.15) if self.m.loss=='soft_l1' else dict(loss=self.m.loss)
        return least_squares(res,x0,bounds=(lo,hi),max_nfev=max_nfev,**kw).x
    def dump(self,x):
        p={n:float(np.exp(v) if self.spec[n][3]=='log' else v) for n,v in zip(self.names,x[:self.NG])}
        return dict(p=p,C=x[self.NG:].reshape(len(self.m.RE),len(self.zn)).tolist(),zn=self.zn,RE=list(self.m.RE))
def calibrate(o_tr,Yk,use):
    r=np.log(o_tr[:,1:])-np.log(Yk[:,1:]); b=np.median(r,0) if use else np.zeros(r.shape[1:]); return b,(r-b).std(0)
def run(model,fold_nfev=60):
    t0=time.time(); F=Fitter(model); warm=None
    wf=f'{LAB}/ode/{model.parent}.json' if model.parent else None
    if wf and os.path.exists(wf): warm=json.load(open(wf))
    all_=np.ones(len(Y),bool); m0=stats0(tr); x0,lo,hi=F.x0(warm)
    x=F.fit(tr,m0,x0,lo,hi,max_nfev=150 if warm else 300)
    json.dump(F.dump(x),open(f'{LAB}/ode/{model.name}.json','w'),indent=1)
    sims=np.zeros((8,len(Y),14,4)); sims[7]=F.sim(x,all_,m0); folds=folds7(tr); xs=np.zeros((8,len(x))); xs[7]=x
    for f,ho in enumerate(folds):
        m=tr.copy(); m[ho]=False; m0f=stats0(m); xf=F.fit(m,m0f,np.clip(x,lo+1e-6,hi-1e-6),lo,hi,max_nfev=fold_nfev); sims[f]=F.sim(xf,all_,m0f); xs[f]=xf
    np.save(f'{LAB}/ode/{model.name}_sims.npy',sims); np.save(f'{LAB}/ode/{model.name}_xs.npy',xs)
    pos={r:i for i,r in enumerate(np.where(tr)[0])}
    for tag,cal,trc in [('pure',False,False),('cal',True,True)]:
        P=np.zeros((tr.sum(),13,4))
        for f,ho in enumerate(folds):
            m=tr.copy(); m[ho]=False; b,sd=calibrate(sims[f][m],Y[m],cal)
            P[[pos[r] for r in ho]]=post(np.log(sims[f][ho][:,1:])-b,sd,fl[ho],1.0,trc,1.0)
        b,sd=calibrate(sims[7][tr],Y[tr],cal); Pt=post(np.log(sims[7][test][:,1:])-b,sd,fl[test],1.0,trc,1.0)
        ins=mape(post(np.log(sims[7][tr][:,1:])-b,sd,fl[tr],1.0,trc,1.0),Ytr).mean()
        register(f'{model.name}_{tag}','mechanistic',model.desc+(' | pure ODE output' if tag=='pure' else ' | + per-day calibration + rule truncation'),P,Ytr,Pt,extra=dict(insample=float(ins),secs=round(time.time()-t0)))
