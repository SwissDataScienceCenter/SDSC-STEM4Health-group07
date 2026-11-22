"""Third round (new file): is the 'feed B on days 6-7' window special, or do neighbouring windows work too? Writes tests100c_<tag>.json."""
import sys,os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
_argv=sys.argv; sys.argv=[_argv[0],'_tmp']
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'tests100.py')).read().split('# ---------------------------------------------------------------- cells')[0])
sys.argv=_argv; RES.clear(); BC=V4[:,:,:2]; C3=['lx0','lp0','V0']
def cells(cols,lam=0.03,w=0.5): sm=sp.run(sp.rank1,[0,1],cols=cols,lam=lam,shrink=1.0); return V1[:,:,:2]**(1-w)*sm**w
i=0; WIN=[(a,b) for a in range(2,12) for b in range(a,min(a+4,13))]
for a,b in WIN:
    nm=f'fbw{a}_{b}'; sp.FE[nm]=st(np.log(F[:,a:b+1,0].sum(1)/V[:,a])); i+=1
    attempt(f'W{i:02d}','feed-B window',f'feed B days {a}-{b} + top-up flag (in place of late feed-B factor)','cells',lambda nm=nm:cells(C3+[nm,'any79']),BC)
for a,b in [(6,7),(5,8),(4,12)]:
    i+=1; attempt(f'W{i:02d}','feed-B window',f'feed B days {a}-{b} WITHOUT the top-up flag','cells',lambda a=a,b=b:cells(C3+[f'fbw{a}_{b}']),BC)
# feed B relative to the batch's own earlier dosing (dose change), and per-volume z-scores of single days
Zf=np.log(F[:,1:13,0]/V[:,1:13]); Zf=(Zf-Zf.mean(0))/Zf.std(0)
for k in range(3,12):
    nm=f'fbz{k}'; sp.FE[nm]=st(Zf[:,k-1]); i+=1; attempt(f'W{i:02d}','feed-B window',f'feed B single day {k} + top-up flag','cells',lambda nm=nm:cells(C3+[nm,'any79']),BC)
i+=1; attempt(f'W{i:02d}','feed-B window','late factor (v4) + days 6-7 + top-up flag','cells',lambda:cells(C3+['fbL','fbw6_7','any79']),BC)
for lam in (0.01,0.1): i+=1; attempt(f'W{i:02d}','feed-B window',f'days 6-7 + top-up flag, ridge {lam}','cells',lambda lam=lam:cells(C3+['fbw6_7','any79'],lam=lam),BC)
tag=sys.argv[1] if len(sys.argv)>1 else 'split1'; json.dump(dict(tests=RES),open(f'{LAB}/tests100c_{tag}.json','w'),indent=1)
for r in RES: print('%-4s %-66s %6.3f -> %6.3f  %+6.3f  t %5.1f  wins %2d  halves %+6.3f %+6.3f'%(r['id'],r['desc'][:66],r['base'],r['cv'],r['delta'],r['t'],r['wins'],r['d_disc'],r['d_conf']))
