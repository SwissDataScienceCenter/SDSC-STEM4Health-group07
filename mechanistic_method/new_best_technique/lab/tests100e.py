"""Permutation test for the new cell signals (new file). 500 placebos: the feed-B (days 5-7) signal and the top-up flag are
shuffled among batches (jointly, and one at a time) and the whole small model is refitted. Writes tests100e_<tag>.json."""
import sys,os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
_argv=sys.argv; sys.argv=[_argv[0],'_tmp']
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'tests100.py')).read().split('# ---------------------------------------------------------------- cells')[0])
sys.argv=_argv; C3=['lx0','lp0','V0']; sp.FE['fb57']=st(np.log(F[:,5:8,0].sum(1)/V[:,5]))
def cv(cols,w=0.5): sm=sp.run(sp.rank1,[0,1],cols=cols,shrink=1.0); return float(pb(V1[:,:,:2]**(1-w)*sm**w,[0,1]).mean())
base=cv(C3); real=dict(both=cv(C3+['fb57','any79']),fb=cv(C3+['fb57']),flag=cv(C3+['any79']),v4=cv(C3+['fbL']))
rp=np.random.default_rng(2026); R=500; out=dict(both=[],fb=[],flag=[])
for r_ in range(R):
    p_=rp.permutation(n); q_=rp.permutation(n); sp.FE['p_fb']=sp.FE['fb57'][p_]; sp.FE['p_fl']=sp.FE['any79'][q_]
    out['both'].append(cv(C3+['p_fb','p_fl'])); out['fb'].append(cv(C3+['p_fb'])); out['flag'].append(cv(C3+['fb57','p_fl']))
res=dict(base_three_signals=base,real=real)
print('cells CV (blend with large models). three basic signals only: %.3f ; v4 small model: %.3f'%(base,real['v4']))
for k,ref,nm in [('both',real['both'],'feed B days 5-7 + top-up flag'),('fb',real['fb'],'feed B days 5-7 alone'),('flag',real['both'],'top-up flag, given feed B days 5-7')]:
    a=np.array(out[k]); p=float((a<=ref).mean()); res[k]=dict(real=ref,placebo_mean=float(a.mean()),placebo_min=float(a.min()),placebo_p5=float(np.quantile(a,.05)),p_value=p)
    print('%-38s real %.3f | 500 placebos: mean %.3f, best %.3f, 5%% quantile %.3f | share of placebos as good as real: %.3f'%(nm,ref,a.mean(),a.min(),np.quantile(a,.05),p))
tag=sys.argv[1] if len(sys.argv)>1 else 'split1'; json.dump(res,open(f'{LAB}/tests100e_{tag}.json','w'),indent=1)
