"""Fourth round (new file): build the improved small cell model around the two leads (feed B on the peak days 5-7, top-up flag)
and check it from several sides. Writes tests100d_<tag>.json."""
import sys,os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
_argv=sys.argv; sys.argv=[_argv[0],'_tmp']
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'tests100.py')).read().split('# ---------------------------------------------------------------- cells')[0])
sys.argv=_argv; RES.clear(); BC=V4[:,:,:2]; C3=['lx0','lp0','V0']
sp.FE['fb57']=st(np.log(F[:,5:8,0].sum(1)/V[:,5])); sp.FE['fb67']=st(np.log(F[:,6:8,0].sum(1)/V[:,6])); sp.FE['fb47']=st(np.log(F[:,4:8,0].sum(1)/V[:,4]))
N5=C3+['fb57','any79']
def small(cols,lam=0.03,shrink=1.0): return sp.run(sp.rank1,[0,1],cols=cols,lam=lam,shrink=shrink)
def cells(cols,lam=0.03,w=0.5,shrink=1.0): return V1[:,:,:2]**(1-w)*small(cols,lam,shrink)**w
i=0
T_=[('NEW small model: day-0 VCD, day-0 byproduct, initial volume, feed B days 5-7, top-up flag',dict(cols=N5)),('  + early volume gain d2-3',dict(cols=N5+['base13'])),('  + early volume gain d2-5',dict(cols=N5+['base25'])),('  + feed-C pulse count',dict(cols=N5+['nC'])),
    ('  + day-11 flag',dict(cols=N5+['fl11'])),('  + late feed-B factor as well',dict(cols=N5+['fbL'])),('  + day-0 substrate',dict(cols=N5+['ls0'])),('  + sample-time offset',dict(cols=N5+['tbar'])),('  without day-0 byproduct',dict(cols=['lx0','V0','fb57','any79'])),('  without initial volume',dict(cols=['lx0','lp0','fb57','any79'])),
    ('  without day-0 VCD',dict(cols=['lp0','V0','fb57','any79'])),('  ridge 0.01',dict(cols=N5,lam=0.01)),('  ridge 0.1',dict(cols=N5,lam=0.1)),('  ridge 0.3',dict(cols=N5,lam=0.3)),('  small-model weight 0.4',dict(cols=N5,w=0.4)),('  small-model weight 0.6',dict(cols=N5,w=0.6)),('  small-model weight 0.7',dict(cols=N5,w=0.7)),
    ('  small-model weight 0.8',dict(cols=N5,w=0.8)),('  small model alone',dict(cols=N5,w=1.0)),('  per-day loadings (shrink 0.5)',dict(cols=N5,shrink=0.5)),('  NEW + d2-3 volume gain, weight 0.6',dict(cols=N5+['base13'],w=0.6)),('  NEW + d2-3 volume gain, weight 0.7',dict(cols=N5+['base13'],w=0.7)),
    ('  window days 6-7 instead of 5-7',dict(cols=C3+['fb67','any79'])),('  window days 4-7 instead of 5-7',dict(cols=C3+['fb47','any79']))]
for nm,kw in T_: i+=1; attempt(f'N{i:02d}','new cell model',nm,'cells',lambda kw=kw:cells(**kw),BC)
# placebos: shuffle the two new signals among batches (keeps their distribution, breaks the link)
rp=np.random.default_rng(11)
for r_ in range(8):
    p_=rp.permutation(n); sp.FE['p_fb']=sp.FE['fb57'][p_]; sp.FE['p_fl']=sp.FE['any79'][p_]; i+=1
    attempt(f'N{i:02d}','new cell model',f'PLACEBO: feed-B window and top-up flag shuffled among batches (#{r_+1})','cells',lambda:cells(C3+['p_fb','p_fl']),BC)
# how much of the offset does each model explain (leave-one-out R2), for the record
L=np.log(Ytr); dev=L-L.mean(0); d=dev[:,:,:2].mean((1,2)); R2={}
for nm,cols in [('v4 small model',C3+['fbL']),('new small model',N5),('new + d2-3 volume gain',N5+['base13']),('day-0 VCD only',['lx0'])]:
    X=np.c_[tuple(sp.FE[c][tr] for c in cols)]; co,loo=sp.ridge_fit(X,d,0.03); R2[nm]=float(1-((d-loo)**2).mean()/d.var())
tag=sys.argv[1] if len(sys.argv)>1 else 'split1'; json.dump(dict(tests=RES,r2=R2),open(f'{LAB}/tests100d_{tag}.json','w'),indent=1)
print('leave-one-out R2 of the batch offset:',{k:round(v,3) for k,v in R2.items()})
for r in RES: print(('%-4s %-88s %6.3f -> %6.3f  %+6.3f  t %5.1f  wins %2d  halves %+6.3f %+6.3f'%(r['id'],r['desc'][:88],r['base'],r['cv'],r['delta'],r['t'],r['wins'],r['d_disc'],r['d_conf'])) if 'error' not in r else 'ERR '+r['id']+r['error'])
