"""Test battery: 150+ single changes to version 4, each scored against version 4 on the same batches, plus verifications.
For every test: CV of the target with and without the change, the paired difference per batch (mean, standard error, t),
how many of the 49 batches improve, and the difference on two fixed halves of the batches (a change that is real should
help in both). Run in lab/ (primary split) and in the second-split copy; results go to tests100_<tag>.json (new file)."""
import sys,os,json,warnings,traceback; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')
from common import *
import sparse as sp, sparse4 as s4, chain5 as c5, chain4 as c4, kf
from smooth2 import lowpass
from joblib import Parallel,delayed
bs,A,Y,tt,test=load(); tr=~test; Ytr=Y[tr][:,1:]; n=len(bs); trn=np.where(tr)[0]; tei=np.where(test)[0]; FOLDS=folds7(tr)
G,fl,D=feature_groups(A,Y,tt); V,F,dVr,Ncnt=D['V'],D['F'],D['dVr'],D['N']
def full(name): z=np.load(f'{LAB}/oof/{name}.npz'); P=np.zeros((n,13,4)); P[tr]=z['oof']; P[test]=z['test']; return P
V1=full('b09_hyb_stat_bayesresid_smoothcells'); V2=full('b13_v2_small_plus_substrate_chain'); V4=full('b15_v4')
rngh=np.random.default_rng(123); perm=rngh.permutation(49); DISC=np.zeros(49,bool); DISC[perm[:25]]=True
RES=[]; VER=[]
def pb(P,js):
    P=P[trn]; P=P[:,:,None] if P.ndim==2 else P; return np.mean([np.abs(P[:,:,i]-Ytr[:,:,j])/Ytr[:,:,j]*100 for i,j in enumerate(js)],axis=(0,2))
JS={'cells':[0,1],'substrate':[2],'byproduct':[3]}
def rec(tid,group,desc,target,cand,base):
    js=JS[target.split('(')[0]]; a=pb(cand,js); b=pb(base,js); d=a-b; se=d.std()/7
    RES.append(dict(id=tid,group=group,desc=desc,target=target,base=float(b.mean()),cv=float(a.mean()),delta=float(d.mean()),se=float(se),t=float(d.mean()/se) if se>0 else 0.0,
                    wins=int((d<0).sum()),d_disc=float(d[DISC].mean()),d_conf=float(d[~DISC].mean())))
def attempt(tid,group,desc,target,fn,base):
    try: rec(tid,group,desc,target,fn(),base)
    except Exception as e: RES.append(dict(id=tid,group=group,desc=desc,target=target,error=repr(e)[:200]))
# ---------------------------------------------------------------- extra signals
st=lambda v:(v-v.mean())/(v.std() if v.std()>0 else 1)
import pandas as pd
_d=pd.read_csv(PROJ+'bioreaction_remaining.csv'); small=[]
for b_ in bs:
    g=_d[(_d.batch_id==b_)&_d.feed_c.notna()]; t=g.elapsed_time.values; a=g.feed_c.values; small.append(t[a<0.0013])
to=tt[:,1:]-np.arange(1,14)
EX=dict(tbar=to[:,:12].mean(1),t1=to[:,0],via0=Y[:,0,0]/Y[:,0,1],any79=fl[:,:3].sum(1),dV0=dVr[:,0],fb1=np.log(F[:,1,0]/V[:,1]),fb13=np.log(F[:,1:4,0].sum(1)/V[:,1]),
        fb67=np.log(F[:,6:8,0].sum(1)/V[:,6]),fb412=np.log(F[:,4:13,0].sum(1)/V[:,4]),lpV=np.log(Y[:,0,3]*V[:,0]),lxV=np.log(Y[:,0,0]*V[:,0]),
        c_last=np.array([e[-1] for e in small]),c_first=np.array([e[0] for e in small]),n23=np.array([((e>=tt[i,2])&(e<tt[i,4])).sum() for i,e in enumerate(small)]).astype(float),
        nlate=np.array([(e>=tt[i,6]).sum() for i,e in enumerate(small)]).astype(float),lx0sq=np.log(Y[:,0,0])**2,lx0lp0=np.log(Y[:,0,0])*np.log(Y[:,0,3]),
        Sh11=np.log(V4[:,10,2]),Sh12=np.log(V4[:,11,2]),Shmid=np.log(V4[:,5:9,2]).mean(1),dhat=np.log(V4[:,:,:2]).mean((1,2)),fd5=F[:,5,3]/V[:,5],dVr5_9=dVr[:,5:9].sum(1),dVr9_13=dVr[:,9:13].sum(1))
for k_,v_ in EX.items(): sp.FE[k_]=st(v_.astype(float))
XFD={k_:sp.FE[k_] for k_ in ('lx0','fbL','base25','dhat')}
C4=['lx0','lp0','V0','fbL']
# ---------------------------------------------------------------- cells
def cells(cols=C4,lam=0.03,shrink=1.0,w=0.5,days=slice(0,13),separate=False,fn=None,**kw):
    if fn is not None: sm=sp.run(fn,[0,1],cols=cols,**kw)
    elif separate: sm=np.concatenate([sp.run(sp.rank1,[j],cols=cols,lam=lam,shrink=shrink,days=days) for j in (0,1)],2)
    else: sm=sp.run(sp.rank1,[0,1],cols=cols,lam=lam,shrink=shrink,days=days)
    return V1[:,:,:2]**(1-w)*sm**w
BC=V4[:,:,:2]
def rank1_huber(fit,te,cols,targets):
    from sklearn.linear_model import HuberRegressor
    X=np.c_[tuple(sp.FE[c] for c in cols)]; L=np.log(Y[fit][:,1:][:,:,targets]); ref=L.mean(0); dev=L-ref; d=dev.mean((1,2)); m=HuberRegressor(alpha=1.0,max_iter=1000).fit(X[fit],d)
    res=dev-m.predict(X[fit])[:,None,None]; sd=1.4826*np.median(np.abs(res-np.median(res,0)),0)*1.05; return ref+m.predict(X[te])[:,None,None],sd
def cells_mass():
    Vb=V[tr].mean(0); Yk=sp.Y; Ym=Y.copy(); Ym[:,:,0]*=V/Vb; Ym[:,:,1]*=V/Vb
    try: sp.Y=Ym; sm=sp.run(sp.rank1,[0,1],cols=C4,shrink=1.0)
    finally: sp.Y=Yk
    return V1[:,:,:2]**0.5*(sm*(Vb/V)[:,1:,None])**0.5
def struct(mode):
    out=BC.copy(); lv=np.log(Y[:,1:,1]/Y[:,1:,0])
    def app(idx,fit):
        m=lv[fit].mean(0)
        if mode in ('tcd','both'): a=BC[idx,:,0]*np.exp(m); out[idx,:,1]=a if mode=='tcd' else np.sqrt(a*BC[idx,:,1])
        if mode in ('vcd','both'): a=BC[idx,:,1]*np.exp(-m); out[idx,:,0]=a if mode=='vcd' else np.sqrt(a*BC[idx,:,0])
    for te in FOLDS: app(te,np.setdiff1d(trn,te))
    app(tei,trn); return out
def resmooth(passes=1,a=None):
    out=BC.copy(); L=np.log(BC)
    def app(idx,fit):
        ref=np.log(Y[fit][:,1:,:2]).mean(0)
        for j in (0,1):
            dd=L[idx,:,j]-ref[:,j]; dd=lowpass(dd,1/3,passes) if a is None else (1-a)*dd+a*dd.mean(1,keepdims=True); out[idx,:,j]=np.exp(ref[:,j]+dd)
    for te in FOLDS: app(te,np.setdiff1d(trn,te))
    app(tei,trn); return out
def alt_small(name):
    P=full(name); return V1[:,:,:2]**0.5*P[:,:,:2]**0.5
i=0
for f in ['ls0','via0','tbar','t1','nC','base25','base13','base58','fbE','fl11','fl12','any79','dV0','fb1','fb13','fb67','fb412','lpV','lxV','c_last','c_first','n23','nlate','Shmid','Sh11','fd5','dVr5_9','dVr9_13']:
    i+=1; attempt(f'C{i:02d}','cells: add one signal',f'offset model + {f}','cells',lambda f=f:cells(cols=C4+[f]),BC)
for nm,cols in [('without late feed-B dosing',['lx0','lp0','V0']),('without initial volume',['lx0','lp0','fbL']),('without day-0 byproduct',['lx0','V0','fbL']),('day-0 VCD only',['lx0']),('fb67 instead of fbL',['lx0','lp0','V0','fb67']),('fb412 instead of fbL',['lx0','lp0','V0','fb412'])]:
    i+=1; attempt(f'C{i:02d}','cells: remove/replace signal','offset model '+nm,'cells',lambda cols=cols:cells(cols=cols),BC)
for lam in (0.003,0.3,1.0): i+=1; attempt(f'C{i:02d}','cells: settings',f'ridge strength {lam} (v4: 0.03)','cells',lambda lam=lam:cells(lam=lam),BC)
for sh in (0.0,0.5): i+=1; attempt(f'C{i:02d}','cells: settings',f'per-day loadings, shrink {sh} (v4: same offset every day)','cells',lambda sh=sh:cells(shrink=sh),BC)
for nm,ds in [('days 3-11',slice(2,11)),('days 2-9',slice(1,9)),('days 5-13',slice(4,13))]: i+=1; attempt(f'C{i:02d}','cells: settings','offset defined on '+nm,'cells',lambda ds=ds:cells(days=ds),BC)
i+=1; attempt(f'C{i:02d}','cells: settings','separate offsets for VCD and TCD','cells',lambda:cells(separate=True),BC)
i+=1; attempt(f'C{i:02d}','cells: model form','Huber (outlier-robust) regression for the offset','cells',lambda:cells(fn=rank1_huber),BC)
i+=1; attempt(f'C{i:02d}','cells: model form','add squared day-0 VCD and VCD x byproduct terms','cells',lambda:cells(cols=C4+['lx0sq','lx0lp0']),BC)
i+=1; attempt(f'C{i:02d}','cells: model form','sample-time correction per day (sparse4.rank1t)','cells',lambda:cells(fn=s4.rank1t,lamg=1.0),BC)
i+=1; attempt(f'C{i:02d}','cells: model form','model total cells (density x volume), convert back with the known volume','cells',cells_mass,BC)
for nm in ('a_svr_curated','a_krr_curated','a_rf_curated','a_huber_curated'): i+=1; attempt(f'C{i:02d}','cells: model form',f'grid-searched {nm[2:]} as the small model','cells',lambda nm=nm:alt_small(nm),BC)
for w in (0.3,0.4,0.6,0.7): i+=1; attempt(f'C{i:02d}','cells: blend',f'weight of small model {w} (v4: 0.5)','cells',lambda w=w:cells(w=w),BC)
def hblend(names,w=0.5):
    b=np.exp(np.mean([np.log(full(x)[:,:,:2]) for x in names],0)); sm=sp.run(sp.rank1,[0,1],cols=C4,shrink=1.0); return b**(1-w)*sm**w
i+=1; attempt(f'C{i:02d}','cells: blend','hybrid ODE only + small model','cells',lambda:hblend(['h_m11_consHill_evidence_allb25']),BC)
i+=1; attempt(f'C{i:02d}','cells: blend','hybrid + Bayesian ODE + small model','cells',lambda:hblend(['h_m11_consHill_evidence_allb25','p2_m21_re8_evidence_all_n1000']),BC)
i+=1; attempt(f'C{i:02d}','cells: blend','five large models + small model','cells',lambda:hblend(['h_m11_consHill_evidence_allb25','s25_search_all_plus_base25','p2_m21_re8_evidence_all_n1000','s26_evidence_all_plus_base25','p_m21_re8_kernel_b25']),BC)
for nm,md in [('TCD = predicted VCD / typical viability','tcd'),('VCD = predicted TCD x typical viability','vcd'),('average of direct and viability-based','both')]: i+=1; attempt(f'C{i:02d}','cells: structure',nm,'cells',lambda md=md:struct(md),BC)
i+=1; attempt(f'C{i:02d}','cells: post-processing','smooth the blended cell deviations once more','cells',lambda:resmooth(1),BC)
i+=1; attempt(f'C{i:02d}','cells: post-processing','pull cell deviations half-way to a constant offset','cells',lambda:resmooth(a=0.5),BC)
for c in (0.995,1.005,0.99,1.01): i+=1; attempt(f'C{i:02d}','cells: post-processing',f'multiply cell predictions by {c}','cells',lambda c=c:BC*c,BC)
# ---------------------------------------------------------------- byproduct
def perday_sel(fit,te,cols,targets,lam=0.003,tdays=(0,1)):
    mu0,sd0=sp.perday(fit,te,cols=cols,targets=targets,lam=lam); 
    if len(tdays):
        mu1,sd1=s4.perdayt(fit,te,cols=cols,targets=targets,lam=lam)
        for k in tdays: mu0[:,k]=mu1[:,k]; sd0[k]=sd1[k]
    return mu0,sd0
E0=dict(cols=['base25','V0','lp0','base58'],lam=0.003,tdays=(0,1),w=0.5); M0=dict(cols=['base25','V0'],lam=0.003,w=1.0); L0=dict(kind='r1',cols=['base25','V0','lp0','fl11','fl12'],lam=0.01,w=0.3,tdays=())
def byp(e=None,m=None,l=None,b1=4,b2=8,scale=1.0):
    e={**E0,**(e or {})}; m={**M0,**(m or {})}; l={**L0,**(l or {})}; b=V1[:,:,3]; out=b.copy()
    pe=sp.run(perday_sel,[3],cols=e['cols'],lam=e['lam'],tdays=e['tdays'])[:,:,0]; pm=sp.run(sp.perday,[3],cols=m['cols'],lam=m['lam'])[:,:,0]
    pl=sp.run(sp.rank1,[3],cols=l['cols'],lam=l['lam'],days=slice(2,12))[:,:,0] if l['kind']=='r1' else sp.run(perday_sel,[3],cols=l['cols'],lam=l['lam'],tdays=l['tdays'])[:,:,0]
    out[:,:b1]=b[:,:b1]**(1-e['w'])*pe[:,:b1]**e['w']; out[:,b1:b2]=b[:,b1:b2]**(1-m['w'])*pm[:,b1:b2]**m['w']; out[:,b2:]=b[:,b2:]**(1-l['w'])*pl[:,b2:]**l['w']; return out*scale
BP=V4[:,:,3]
def byp_mass():
    Vb=V[tr].mean(0); Yk=sp.Y; Ym=Y.copy(); Ym[:,:,3]*=V/Vb
    try: sp.Y=Ym; pm=sp.run(sp.perday,[3],cols=['base25','V0'],lam=0.003)[:,:,0]*(Vb/V)[:,1:]
    finally: sp.Y=Yk
    out=BP.copy(); out[:,4:8]=pm[:,4:8]; return out
def byp_smooth():
    out=BP.copy(); L=np.log(BP)
    def app(idx,fit): ref=np.log(Y[fit][:,1:,3]).mean(0); out[idx]=np.exp(ref+lowpass(L[idx]-ref,1/3,1))
    for te in FOLDS: app(te,np.setdiff1d(trn,te))
    app(tei,trn); return out
i=0
PT=[('early: without the days 5-8 volume gain',dict(e=dict(cols=['base25','V0','lp0']))),('early: + day-0 VCD',dict(e=dict(cols=E0['cols']+['lx0']))),('early: + early feed-B dosing',dict(e=dict(cols=E0['cols']+['fbE']))),
    ('early: + feed-C pulse count days 2-3',dict(e=dict(cols=E0['cols']+['n23']))),('early: + day-0 substrate',dict(e=dict(cols=E0['cols']+['ls0']))),('early: volume gain split d2-3 / d4-5',dict(e=dict(cols=['base13','base35','V0','lp0','base58']))),
    ('early: no sample-time feature',dict(e=dict(tdays=()))),('early: sample time on day 1 only',dict(e=dict(tdays=(0,)))),('early: sample time on days 1-4',dict(e=dict(tdays=(0,1,2,3)))),('early: weight 0.3',dict(e=dict(w=0.3))),('early: weight 0.7',dict(e=dict(w=0.7))),
    ('early: ridge 0.03',dict(e=dict(lam=0.03))),('early: + predicted cell offset',dict(e=dict(cols=E0['cols']+['dhat']))),
    ('mid: + day-0 byproduct',dict(m=dict(cols=['base25','V0','lp0']))),('mid: + day-0 VCD',dict(m=dict(cols=['base25','V0','lx0']))),('mid: + days 5-8 volume gain',dict(m=dict(cols=['base25','V0','base58']))),('mid: + predicted cell offset',dict(m=dict(cols=['base25','V0','dhat']))),
    ('mid: + feed-C pulses late',dict(m=dict(cols=['base25','V0','nlate']))),('mid: ridge 0.03',dict(m=dict(lam=0.03))),('mid: weight 0.75 (rest large models)',dict(m=dict(w=0.75))),('mid: weight 0.5',dict(m=dict(w=0.5))),('mid: + predicted substrate days 6-9',dict(m=dict(cols=['base25','V0','Shmid']))),
    ('late: large models only',dict(l=dict(w=0.0))),('late: weight 0.2',dict(l=dict(w=0.2))),('late: weight 0.5',dict(l=dict(w=0.5))),('late: without feed flags',dict(l=dict(cols=['base25','V0','lp0']))),('late: per-day model instead of one offset',dict(l=dict(kind='pd',lam=0.01))),
    ('late: per-day model with sample time on day 13',dict(l=dict(kind='pd',lam=0.01,tdays=(12,)))),('late: + predicted substrate day 11/12',dict(l=dict(cols=L0['cols']+['Sh11','Sh12']))),('late: + predicted cell offset',dict(l=dict(cols=L0['cols']+['dhat']))),
    ('late: + feed-C pulses late',dict(l=dict(cols=L0['cols']+['nlate']))),('late: + late unlogged volume change',dict(l=dict(cols=L0['cols']+['dVr9_13']))),
    ('phases: mid = days 5-7',dict(b2=7)),('phases: mid = days 4-8',dict(b1=3)),('phases: mid = days 5-9',dict(b2=9)),('phases: mid = days 6-8',dict(b1=5)),('scale all by 1.01',dict(scale=1.01)),('scale all by 0.99',dict(scale=0.99)),('scale all by 1.02',dict(scale=1.02))]
for nm,kw in PT: i+=1; attempt(f'P{i:02d}','byproduct',nm,'byproduct',lambda kw=kw:byp(**kw),BP)
i+=1; attempt(f'P{i:02d}','byproduct','mid: model total mass (conc x volume), convert back','byproduct',byp_mass,BP)
i+=1; attempt(f'P{i:02d}','byproduct','smooth byproduct deviations over time once','byproduct',byp_smooth,BP)
i+=1; attempt(f'P{i:02d}','byproduct','version 2 byproduct (one weight 0.5 for all days)','byproduct',lambda:V2[:,:,3],BP)
i+=1; attempt(f'P{i:02d}','byproduct','version 1 byproduct (large models only)','byproduct',lambda:V1[:,:,3],BP)
# ---------------------------------------------------------------- substrate: blend level
lc=np.log(full('c02_chain_ens')[:,:,2]); lk=np.log(full('k02_statespace_ens')[:,:,2]); cache=f'{LAB}/oof/_l4_cache.npz'
if os.path.exists(cache): l4=np.load(cache)['l4']
else:
    l4=np.mean(Parallel(n_jobs=3)(delayed(lambda ls,sd:np.log(c4.chain4('b12_stage1',lamS=ls,N=8000,seed=sd)[0]))(ls,40+j) for j,ls in enumerate((0.03,0.1,0.3))),0); np.savez(cache,l4=l4)
bol=fl[:,:3].sum(1)>0; lb=np.log(V1[:,:,2]); BS=V4[:,:,2]
def sub(sims=(lc,lk,l4),w=0.7,bw=1.0,wk=None,scale=1.0):
    sim=np.mean(sims,0); W=np.where(bol,bw,w)[:,None]*np.ones((1,13))
    if wk is not None: W=np.where(bol[:,None],bw,np.array(wk)[None,:])
    return np.exp(W*sim+(1-W)*lb)*scale
i=0
ST=[('rebuilt v4 substrate from parts (should equal v4 up to simulation noise)',{}),('simulator weight 0.5 (v4: 0.7)',dict(w=0.5)),('simulator weight 0.6',dict(w=0.6)),('simulator weight 0.8',dict(w=0.8)),('simulator weight 0.9',dict(w=0.9)),('simulators only',dict(w=1.0)),
    ('no special rule for top-up batches',dict(bw=0.7)),('top-up batches: simulator weight 0.85',dict(bw=0.85)),('chain ensemble only',dict(sims=(lc,))),('state-space ensemble only',dict(sims=(lk,))),('selective-shrinkage chains only',dict(sims=(l4,))),
    ('chain + state-space (as v2)',dict(sims=(lc,lk))),('days 1-6 weight 0.5, days 7-13 weight 0.85',dict(wk=[.5]*6+[.85]*7)),('days 1-6 weight 0.85, days 7-13 weight 0.5',dict(wk=[.85]*6+[.5]*7)),('days 11-13 simulators only',dict(wk=[.7]*10+[1.0]*3)),
    ('scale by 0.99',dict(scale=0.99)),('scale by 1.01',dict(scale=1.01))]
for nm,kw in ST: i+=1; attempt(f'S{i:02d}','substrate: blend',nm,'substrate',lambda kw=kw:sub(**kw),BS)
# ---------------------------------------------------------------- substrate: one simulator, one change at a time
CH=[('reference simulator, other random seed 1',dict(seed=1)),('reference simulator, other random seed 2',dict(seed=2)),('reference simulator, other random seed 3',dict(seed=3)),('1,000 particles (ref: 4,000)',dict(N=1000)),('16,000 particles',dict(N=16000)),
    ('threshold days 7-9 = 1.92 (ref 1.95)',dict(thr79=1.92)),('threshold days 7-9 = 1.98',dict(thr79=1.98)),('threshold days 11-12 = 0.960 (ref 0.967)',dict(thr1112=0.960)),('threshold days 11-12 = 0.964',dict(thr1112=0.964)),('threshold days 11-12 = 0.969',dict(thr1112=0.969)),
    ('point = plain median of the paths',dict(wp=0.0)),('point = median weighted by 1/sqrt(y)',dict(wp=0.5)),('point = median weighted by 1/y^1.5',dict(wp=1.5)),('point = median weighted by 1/y^2',dict(wp=2.0)),('point = mean of the paths',dict(point='mean')),('point = harmonic mean of the paths',dict(point='hmean')),
    ('scatter x0.8',dict(sdS=0.8)),('scatter x0.9',dict(sdS=0.9)),('scatter x1.1',dict(sdS=1.1)),('scatter x1.3',dict(sdS=1.3)),('scatter of the day 10->11 step x0.8',dict(sd10=0.8)),('scatter of the day 10->11 step x1.3',dict(sd10=1.3)),('late scatter x0.85',dict(sdlate=0.85)),('late scatter x1.2',dict(sdlate=1.2)),('early scatter x0.8',dict(sdearly=0.8)),('early scatter x1.2',dict(sdearly=1.2)),
    ('without predicted cells as helper',dict(xb=0)),('without predicted byproduct as helper',dict(pb=0)),('without unlogged volume gain as helper',dict(vb=0)),('with the large models\' substrate prediction as helper',dict(sb=1)),('helpers taken from version 4 instead of stage 1',dict(base='b15_v4')),
    ('physical terms shrunk more (0.01)',dict(lamH=1e-2)),('helpers shrunk less (0.03)',dict(lamS=0.03)),('helpers shrunk more (0.3)',dict(lamS=0.3)),('helpers shrunk more (1.0)',dict(lamS=1.0)),
    ('VERIFY: feed decisions ignored (must be much worse)',dict(flags=0)),('only the day 11-12 decisions used',dict(flagdays=(11,12))),('only the day 7-9 decisions used',dict(flagdays=(7,8,9))),
    ('step includes yesterday squared',dict(quad=1)),('step includes yesterday x interval',dict(inter=1)),('step also uses the day before yesterday',dict(ar2=5)),
    ('+ day-0 VCD as helper in every step',dict(xfeat=('lx0',))),('+ late feed-B dosing as helper',dict(xfeat=('fbL',))),('+ volume gain d2-5 as helper',dict(xfeat=('base25',))),('+ predicted cell offset as helper',dict(xfeat=('dhat',)))]
def runc(kw): c5.XF.update(XFD); return c5.chain5(**kw)
outs=Parallel(n_jobs=10)(delayed(runc)(kw) for kw in [dict()]+[kw for _,kw in CH]); REF=outs[0]
for j,(nm,kw) in enumerate(CH): rec(f'S{i+j+1:02d}','substrate: single simulator',nm,'substrate(single simulator)',outs[j+1],REF)
i+=len(CH)
KV=[('state-space: process scatter as fitted (ref x1.4)',dict(sdq=1.0)),('state-space: process scatter x2',dict(sdq=2.0)),('state-space: helpers shrunk more (0.1)',dict(lam=0.1)),('state-space: assay-noise floor 2 %',dict(rfloor=0.02)),('state-space: pooled bolus coefficient',dict(cb=1)),('state-space: other seed',dict(seed=5))]
def runk(kw): return kf.kf('b12_stage1',**{**dict(lam=1e-2,sdq=1.4,vb=1,N=4000),**kw})[:2]
ko=Parallel(n_jobs=7)(delayed(runk)(kw) for kw in [dict()]+[kw for _,kw in KV])
def kfull(o): X=np.zeros((n,13)); X[tr]=o[0]; X[test]=o[1]; return X
for j,(nm,kw) in enumerate(KV): rec(f'S{i+j+1:02d}','substrate: single simulator',nm,'substrate(single simulator)',kfull(ko[j+1]),kfull(ko[0]))
# ---------------------------------------------------------------- verifications
def ver(vid,desc,ok,detail=''): VER.append(dict(id=vid,desc=desc,ok=bool(ok),detail=str(detail)))
try:
    tpl=pd.read_csv(PROJ+'bioreaction_template.csv',dtype={'batch_id':str}); s4f=pd.read_csv(PROJ+'submission_v4.csv',dtype={'batch_id':str})
    ver('V01','submission_v4.csv has the template\'s columns, rows, batch ids and sample times',list(s4f.columns)==list(tpl.columns) and len(s4f)==len(tpl) and (s4f.batch_id==tpl.batch_id).all() and np.allclose(s4f.elapsed_time,tpl.elapsed_time),f'{len(s4f)} rows')
    ver('V02','submission_v4.csv has no missing, zero or negative values',s4f[T].notna().all().all() and (s4f[T]>0).all().all())
    ver('V03','submission_v4.csv equals the stored version-4 test predictions',np.allclose(s4f[T].values.reshape(12,13,4),V4[test],rtol=1e-9),'max rel diff %.1e'%np.abs(s4f[T].values.reshape(12,13,4)/V4[test]-1).max())
    ver('V04','predicted VCD <= predicted TCD everywhere in the file',(s4f.vcd<=s4f.tcd).all(),'min TCD/VCD %.4f'%(s4f.tcd/s4f.vcd).min())
except Exception as e: ver('V01-04','submission file checks',False,repr(e))
viol=[(bs[b],d) for b in tei for d in THR if (V4[b,d-1,2]<THR[d])!=(fl[b,FLAGCOL[d]]>0)]; ver('V05','test substrate predictions are on the side of the threshold that the feed log implies (60 batch-days)',len(viol)==0,viol)
violt=sum((V4[b,d-1,2]<THR[d])!=(fl[b,FLAGCOL[d]]>0) for b in trn for d in THR); ver('V06','same check on out-of-fold predictions of the 49 known batches (245 batch-days)',violt==0,f'{violt} violations')
bad=sum((Y[b,d,2]<THR[d])!=(fl[b,FLAGCOL[d]]>0) for b in trn for d in THR); ver('V07','the feed rule holds exactly on the measured values of all known batches',bad==0,f'{bad} exceptions')
ver('V08','every known batch is in exactly one fold; no test batch is in any fold',sorted(np.concatenate(FOLDS).tolist())==sorted(trn.tolist()) and not set(np.concatenate(FOLDS))&set(tei))
ver('V09','stored version-4 CV reproduces',abs(mape(V4[tr],Ytr).mean()-float(np.mean([pb(V4[:,:,j],[j]).mean() for j in range(4)])))<1e-9,'%.3f'%mape(V4[tr],Ytr).mean())
ver('V10','version 4 cells are identical to version 2 cells',np.allclose(V4[:,:,:2],V2[:,:,:2]))
a1=sp.run(sp.rank1,[0,1],cols=C4,shrink=1.0); a2=sp.run(sp.rank1,[0,1],cols=C4,shrink=1.0); ver('V11','small cell model is deterministic (two runs identical)',np.array_equal(a1,a2))
# leak check: shuffle the outcomes among batches; a model that only uses inputs must fall back to the no-information level
rs=np.random.default_rng(7); Yk=sp.Y; Ysh=Y.copy(); p_=rs.permutation(trn); Ysh[trn,1:]=Y[p_,1:]
try:
    sp.Y=Ysh; sh=sp.run(sp.rank1,[0,1],cols=C4,shrink=1.0)
finally: sp.Y=Yk
esh=np.mean([np.abs(sh[trn][:,:,j]-Ysh[trn,1:,j])/Ysh[trn,1:,j]*100 for j in (0,1)]); emn=np.mean([pb(full('s01_mean')[:,:,j],[j]).mean() for j in (0,1)])
ver('V12','leak check: with outcomes shuffled among batches the small cell model is no better than "every batch is typical"',esh>emn-0.15,'shuffled %.2f vs no-information %.2f vs real %.2f'%(esh,emn,np.mean([pb(a1[:,:,j],[j]).mean() for j in (0,1)])))
flg=[r for r in RES if 'feed decisions ignored' in r.get('desc','')]; ver('V13','ignoring the feed decisions makes the simulator much worse (the constraint code is active)',bool(flg) and flg[0]['delta']>2.0,'+%.2f'%flg[0]['delta'] if flg else '')
sd_seed=np.std([r['delta'] for r in RES if 'other random seed' in r.get('desc','') and 'state' not in r['desc']]); ver('V14','simulation noise is small: CV moves by less than 0.05 between random seeds',sd_seed<0.05,'sd of CV across seeds %.3f'%sd_seed)
rb=[r for r in RES if r.get('desc','').startswith('rebuilt v4 substrate')]; ver('V15','version-4 substrate can be rebuilt from its parts',bool(rb) and abs(rb[0]['delta'])<0.03,'difference %.3f'%rb[0]['delta'] if rb else '')
lo=np.log(Ytr).min(0)-0.25; hi=np.log(Ytr).max(0)+0.25; outside=int(((np.log(V4[test])<lo)|(np.log(V4[test])>hi)).sum()); ver('V16','all test predictions lie within the range seen in known batches (25 % margin)',outside==0,f'{outside} of 624 outside')
r_tr=np.log(V4[tr]).std(0).mean(0); r_te=np.log(V4[test]).std(0).mean(0); ver('V17','test predictions spread about as much between batches as out-of-fold predictions do (ratio 0.6-1.6 per target)',((r_te/r_tr>0.6)&(r_te/r_tr<1.6)).all(),np.round(r_te/r_tr,2))
j1=np.log(V4[:,4,3]/V4[:,3,3]); j2=np.log(V4[:,8,3]/V4[:,7,3]); t1_=np.log(Ytr[:,4,3]/Ytr[:,3,3]); t2_=np.log(Ytr[:,8,3]/Ytr[:,7,3])
ver('V18','byproduct phase boundaries (day 4->5, 8->9) show no artificial jump: predicted day-to-day change matches the measured typical change',abs(j1[test].mean()-t1_.mean())<0.05 and abs(j2[test].mean()-t2_.mean())<0.05,'d4->5 pred %.3f vs measured %.3f; d8->9 pred %.3f vs measured %.3f'%(j1[test].mean(),t1_.mean(),j2[test].mean(),t2_.mean()))
jb=[(bs[b],d) for b in tei for d in (7,8,9) if F[b,d,1]>0 and not V4[b,d,2]>V4[b,d-1,2]+0.5]; ver('V19','after a logged top-up the predicted substrate rises by more than 0.5 g/L next day (3 test batches)',len(jb)==0,jb)
ver('V20','version-1 CV matches its leaderboard score (7.028 vs 7.018)',abs(mape(V1[tr],Ytr).mean()-7.018)<0.15,'%.3f'%mape(V1[tr],Ytr).mean())
tag=sys.argv[1] if len(sys.argv)>1 else 'split1'
json.dump(dict(tests=RES,verifications=VER,baseline=dict(v4=[float(x) for x in mape(V4[tr],Ytr).mean(0)])),open(f'{LAB}/tests100_{tag}.json','w'),indent=1)
ok=[r for r in RES if 'error' not in r]; print(len(RES),'tests,',len(RES)-len(ok),'errors;',len(VER),'verifications,',sum(v['ok'] for v in VER),'passed')
for v in VER: print(('PASS' if v['ok'] else 'FAIL'),v['id'],v['desc'],'|',v['detail'])
for r in RES:
    if 'error' in r: print('ERR',r['id'],r['desc'],r['error'])
