"""Build a submission by taking each target's column from a chosen file (new file; refuses to overwrite).
The leaderboard mean is the plain average of the four per-target scores and each per-target score depends only on its own
column, so the mixed file's score is known in advance from the per-target scores of the source files.
Usage:  python lab/mixcols.py OUT.csv vcd=FILE tcd=FILE substrate=FILE byproduct=FILE     (paths relative to the project root)"""
import sys,os
import pandas as pd
PROJ=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/'; T=['vcd','tcd','substrate','byproduct']
def main(argv):
    out=argv[0]; src=dict(a.split('=',1) for a in argv[1:]); assert sorted(src)==sorted(T),'give one file for each of '+', '.join(T)
    op=out if os.path.isabs(out) else PROJ+out; assert not os.path.exists(op),out+' exists; not overwriting'
    tpl=pd.read_csv(PROJ+'bioreaction_template.csv',dtype={'batch_id':str}); res=None
    for t in T:
        d=pd.read_csv(src[t] if os.path.isabs(src[t]) else PROJ+src[t],dtype={'batch_id':str})
        assert list(d.columns)==list(tpl.columns) and len(d)==len(tpl) and (d.batch_id==tpl.batch_id).all(),src[t]+' does not match the template'
        if res is None: res=d.copy()
        res[t]=d[t].values
    assert res[T].notna().all().all() and (res[T]>0).all().all()
    res.to_csv(op,index=False); print('wrote',out,'|',' | '.join('%s from %s'%(t,src[t]) for t in T))
if __name__=='__main__': main(sys.argv[1:])
