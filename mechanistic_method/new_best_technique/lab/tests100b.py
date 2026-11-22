"""Follow-up tests on the leads found by tests100.py (new file). Same scoring. Writes tests100b_<tag>.json."""
import sys,os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
_argv=sys.argv; sys.argv=[_argv[0],'_tmp']          # reuse the harness; its own output goes to tests100__tmp.json
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'tests100.py')).read().split('# ---------------------------------------------------------------- cells')[0])
sys.argv=_argv; RES.clear(); VER.clear()
C4=['lx0','lp0','V0','fbL']; BC=V4[:,:,:2]; BP=V4[:,:,3]; BS=V4[:,:,2]
def cells(cols=C4,lam=0.03,shrink=1.0,w=0.5):
    sm=sp.run(sp.rank1,[0,1],cols=cols,lam=lam,shrink=shrink); return V1[:,:,:2]**(1-w)*sm**w
# graded version of the top-up clue: how far the feed decisions pull the simulated substrate away from the unconstrained simulation
def runc(kw): return c5.chain5(**kw)
sf,s0=Parallel(n_jobs=2)(delayed(runc)(kw) for kw in (dict(N=8000,seed=11),dict(N=8000,seed=11,flags=0)))
sur=np.log(sf/s0); extra=dict(sur79=sur[:,6:10].mean(1),sur1113=sur[:,10:].mean(1),surall=sur[:,6:].mean(1),b7=fl[:,0],b8=fl[:,1],b9=fl[:,2],b79x=fl[:,:3].sum(1)*np.log(Y[:,0,0]))
for k_,v_ in extra.items(): sp.FE[k_]=st(v_.astype(float))
i=0
CT=[('top-up flag (any of days 7-9)',C4+['any79']),('top-up flag, ridge 0.1',None),('top-up flag, ridge 0.3',None),('separate flags for day 7, 8, 9',C4+['b7','b8','b9']),('graded: substrate pulled down by the decisions, days 7-10',C4+['sur79']),
    ('graded: days 11-13',C4+['sur1113']),('graded: days 7-13',C4+['surall']),('top-up flag + graded days 7-10',C4+['any79','sur79']),('top-up flag + day-11 flag',C4+['any79','fl11']),('top-up flag + day-11 + day-12 flags',C4+['any79','fl11','fl12']),
    ('top-up flag, without late feed-B dosing',['lx0','lp0','V0','any79']),('top-up flag + early volume gain d2-3',C4+['any79','base13']),('top-up flag + fb67 instead of fbL',['lx0','lp0','V0','fb67','any79'])]
for nm,cols in CT:
    i+=1
    if cols is None: lam=float(nm.split()[-1]); attempt(f'F{i:02d}','follow-up: cells',nm,'cells',lambda lam=lam:cells(cols=C4+['any79'],lam=lam),BC)
    else: attempt(f'F{i:02d}','follow-up: cells',nm,'cells',lambda cols=cols:cells(cols=cols),BC)
for w in (0.4,0.6): i+=1; attempt(f'F{i:02d}','follow-up: cells',f'top-up flag, small-model weight {w}','cells',lambda w=w:cells(cols=C4+['any79'],w=w),BC)
i+=1; attempt(f'F{i:02d}','follow-up: cells','top-up flag, per-day loadings (shrink 0.5)','cells',lambda:cells(cols=C4+['any79'],shrink=0.5),BC)
# placebo: a random flag on 7 batches should not help
rp=np.random.default_rng(5)
for r_ in range(5):
    v=np.zeros(n); v[rp.choice(trn,7,replace=False)]=1; v[rp.choice(tei,3,replace=False)]=1; sp.FE[f'plc{r_}']=st(v); i+=1
    attempt(f'F{i:02d}','follow-up: cells',f'PLACEBO: random flag on 7 known + 3 test batches (#{r_+1})','cells',lambda r_=r_:cells(cols=C4+[f'plc{r_}']),BC)
# byproduct
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'tests100.py')).read().split('# ---------------------------------------------------------------- byproduct')[1].split('BP=V4[:,:,3]')[0])
LP=dict(kind='pd',lam=0.01,tdays=(12,))
BT=[('late: per-day model + sample time on day 13',dict(l=LP)),('late: same, without sample time (control)',dict(l=dict(kind='pd',lam=0.01))),('late: per-day + sample time on days 12-13',dict(l=dict(kind='pd',lam=0.01,tdays=(11,12)))),('late: per-day + sample time days 11-13',dict(l=dict(kind='pd',lam=0.01,tdays=(10,11,12)))),
    ('late: per-day + time d13, weight 0.2',dict(l=dict(LP,w=0.2))),('late: per-day + time d13, weight 0.4',dict(l=dict(LP,w=0.4))),('late: per-day + time d13, ridge 0.03',dict(l=dict(LP,lam=0.03))),('late: per-day + time d13 + top-up flag',dict(l=dict(LP,cols=L0['cols']+['any79']))),
    ('late: offset model + top-up flag',dict(l=dict(cols=L0['cols']+['any79']))),('mid: + top-up flag',dict(m=dict(cols=['base25','V0','any79']))),('early: + top-up flag',dict(e=dict(cols=E0['cols']+['any79']))),
    ('late: per-day + time d13, scale 1.01',dict(l=LP,scale=1.01)),('late: per-day + time d13; mid weight 0.75',dict(l=LP,m=dict(w=0.75)))]
for nm,kw in BT: i+=1; attempt(f'F{i:02d}','follow-up: byproduct',nm,'byproduct',lambda kw=kw:byp(**kw),BP)
# substrate: final-level effect of adding variants to the simulator pool
lc=np.log(full('c02_chain_ens')[:,:,2]); lk=np.log(full('k02_statespace_ens')[:,:,2]); l4=np.load(f'{LAB}/oof/_l4_cache.npz')['l4']; bol=fl[:,:3].sum(1)>0; lb=np.log(V1[:,:,2])
def fin(sims,w=0.7,bw=1.0): sim=np.mean(sims,0); W=np.where(bol,bw,w)[:,None]; return np.exp(W*sim+(1-W)*lb)
XFD={k_:sp.FE[k_] for k_ in ('lx0','fbL','base25','dhat')}
def runx(kw): c5.XF.update(XFD); return np.log(c5.chain5(**kw))
V_=[dict(N=8000,seed=21,xfeat=('base25',)),dict(N=8000,seed=22,xfeat=('base25',),lamS=0.3),dict(N=8000,seed=23,wp=0.5),dict(N=8000,seed=24,wp=0.0),dict(N=8000,seed=25,point='mean'),dict(N=8000,seed=26),dict(N=8000,seed=27,wp=0.5,xfeat=('base25',))]
lx=Parallel(n_jobs=7)(delayed(runx)(kw) for kw in V_)
np.savez(f'{LAB}/oof/_l5_cache.npz',lx=np.array(lx))
SX=[('pool + 2 simulators with volume gain d2-5 as helper',[lc,lk,l4,np.mean(lx[:2],0)]),('pool + simulator with 1/sqrt(y) weighting',[lc,lk,l4,lx[2]]),('pool + simulator with plain median',[lc,lk,l4,lx[3]]),('pool + simulator with mean of paths',[lc,lk,l4,lx[4]]),
    ('pool + one more reference simulator (control)',[lc,lk,l4,lx[5]]),('pool + helper & 1/sqrt(y) simulator',[lc,lk,l4,lx[6]]),('pool + all four alternatives',[lc,lk,l4,np.mean([lx[0],lx[1],lx[2],lx[6]],0)])]
for nm,sims in SX: i+=1; attempt(f'F{i:02d}','follow-up: substrate',nm,'substrate',lambda sims=sims:fin(sims),BS)
tag=sys.argv[1] if len(sys.argv)>1 else 'split1'
json.dump(dict(tests=RES),open(f'{LAB}/tests100b_{tag}.json','w'),indent=1)
for r in RES:
    print(('%-4s %-58s %6.3f -> %6.3f  %+6.3f  t %5.1f  wins %2d  halves %+6.3f %+6.3f'%(r['id'],r['desc'][:58],r['base'],r['cv'],r['delta'],r['t'],r['wins'],r['d_disc'],r['d_conf'])) if 'error' not in r else ('ERR '+r['id']+' '+r['error']))
