"""Builds submission_pa/pb/pc.csv: three complete model versions, each column a different model (new file; refuses to overwrite).
 pa: vcd,tcd ODE-based models only (b11) | substrate v4, days 5-8 x1.02 | byproduct b06 x1.02
 pb: vcd particle ODE p_m01_base_kernel | tcd particle ODE p2_m02_cov7_evidence | substrate simulator weight 0.5 | byproduct hybrid h_m11_consHill_search_b25
 pc: vcd kernel ridge a_krr_wide | tcd particle ODE p_m01_base_kernel_k14 | substrate grid-search g02 | byproduct v1 x1.02, days 5-8 a further x1.02
 x_sub_w05 = substrate with simulator weight 0.5 (plus the top-up rule for batches topped up on days 7-9)."""
import sys,os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import *
from report import write_submission
bs,A,Y,tt,test=load(); tr=~test
def ld(nm): z=np.load(f'{LAB}/oof/{nm}.npz'); return z['oof'],z['test']
O1,T1=ld('b09_hyb_stat_bayesresid_smoothcells'); O4,T4=ld('b15_v4')
def col(nm,j): o,t=ld(nm); return o[:,:,j],t[:,:,j]
def scale(pair,m): return pair[0]*m,pair[1]*m
d58=np.ones(13); d58[4:8]=1.02
PLAN={'pa':[col('b11_mechanistic_only_bayesresid',0),col('b11_mechanistic_only_bayesresid',1),scale(col('b15_v4',2),d58),scale(col('b06_nosearch_smoothcells',3),1.02)],
      'pb':[col('p_m01_base_kernel',0),col('p2_m02_cov7_evidence',1),col('x_sub_w05',2),col('h_m11_consHill_search_b25',3)],
      'pc':[col('a_krr_wide',0),col('p_m01_base_kernel_k14',1),col('g02_gridcv_threephase',2),scale(col('b09_hyb_stat_bayesresid_smoothcells',3),1.02*d58)]}
DESC={'pa':'vcd,tcd: ODE-based models only (b11) | substrate: v4, days 5-8 x1.02 | byproduct: b06 x1.02',
      'pb':'vcd: particle ODE p_m01_base_kernel | tcd: particle ODE p2_m02_cov7_evidence | substrate: simulator weight 0.5 | byproduct: hybrid h_m11_consHill_search_b25',
      'pc':'vcd: kernel ridge a_krr_wide | tcd: particle ODE p_m01_base_kernel_k14 | substrate: grid-search g02 | byproduct: v1 x1.02, days 5-8 a further x1.02'}
if __name__=='__main__':
    tpl=pd.read_csv(PROJ+'bioreaction_template.csv',dtype={'batch_id':str})
    for k,cols in PLAN.items():
        fn=f'submission_{k}.csv'; assert not os.path.exists(PROJ+fn),fn+' exists; not overwriting'
        np.savez(f'{LAB}/oof/x_{k}.npz',oof=np.stack([c[0] for c in cols],2),test=np.stack([c[1] for c in cols],2)); write_submission(f'x_{k}',fn)
        b=pd.read_csv(PROJ+fn,dtype={'batch_id':str}); assert list(b.columns)==list(tpl.columns) and len(b)==156 and (b.batch_id==tpl.batch_id).all() and (b[T]>0).all().all()
        print(fn,'|',DESC[k])
