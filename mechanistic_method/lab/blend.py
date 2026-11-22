import sys,os,itertools; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import *
bs,A,Y,tt,test=load(); tr=~test; Ytr=Y[tr][:,1:]
def get(n): z=np.load(f'{LAB}/oof/{n}.npz'); return z['oof'],z['test']
def blend(names,name=None,desc=None,quiet=False):
    O=[get(n) for n in names]; P=np.exp(np.mean([np.log(o[0]) for o in O],0)); Pt=np.exp(np.mean([np.log(o[1]) for o in O],0))
    if name: return register(name,'blend',desc or ('geometric mean of '+', '.join(names)),P,Ytr,Pt,quiet=quiet)
    E=mape(P,Ytr); return E.mean(),E.mean(0)
if __name__=='__main__':
    if sys.argv[1]=='scan':
        cands=sys.argv[2:]; rows=[]
        for r in (2,3,4):
            for c in itertools.combinations(cands,r):
                m,e=blend(c); rows.append((m,*e,' + '.join(c)))
        for row in sorted(rows)[:15]: print('%.3f | %.2f %.2f %.2f %.2f | %s'%row)
    else: blend(sys.argv[2:],sys.argv[1])
