import sys,os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import *
def write_submission(name,out):
    bs,A,Y,tt,test=load(); Pt=np.load(f'{LAB}/oof/{name}.npz')['test']; names=[b for b,t in zip(bs,test) if t]
    tpl=pd.read_csv(PROJ+'bioreaction_template.csv',dtype={'batch_id':str}); tpl['k']=tpl.groupby('batch_id').cumcount()+1
    tsamp={b:tt[bs.index(b)] for b in names}
    assert np.allclose(tpl.elapsed_time.values,[tsamp[b][k] for b,k in zip(tpl.batch_id,tpl.k)],atol=1e-6)
    tpl[T]=np.array([Pt[names.index(b),k-1] for b,k in zip(tpl.batch_id,tpl.k)]); assert tpl[T].notna().all().all() and (tpl[T]>0).all().all()
    tpl.drop(columns='k').to_csv(PROJ+out,index=False)
NOTES = """
## How to read this

- **baseline**: per-day typical profile, no batch information.
- **statistical**: regularised regression on process signals (initial values, feeds, volumes, sample times, substrate-feed rule flags).
- **mechanistic**: ODE mass-balance model. `_pure` = raw simulation, `_cal` = with per-day correction factors and the substrate feed-rule step.
- **mechanistic-bayes**: ODE with per-batch random effects, sampled as particles and weighted by the substrate feed-rule likelihood.
- **hybrid**: ODE simulation plus a statistical model on what the ODE leaves unexplained.
- **blend**: geometric mean of members from different families.

## What the data says (findings that drove the modelling)

1. Assay noise on cell counts is about 6-7 %, so no model can go much below ~5.2 % MAPE on VCD/TCD.
2. The predictable part of a batch's cell density is essentially ONE number: a constant multiplicative offset from the typical profile, present from day 1 (true inoculum density). Process signals explain about 55-60 % of its variance: day-0 VCD (noisy), day-0 byproduct and volume (inoculum carry-over), feed-B dosing per volume, feed-C pulse count.
3. Volume gained on days 2-5 that the logged feeds do not explain (unlogged base addition) explains about half of the batch-level byproduct offset; adding initial volume brings it to ~60 %.
4. The substrate feed log reveals the measured substrate: bolus on days 7-9 only when substrate < ~1.95; on days 11-12 the bolus is 0.6 % of volume when substrate < ~0.967, else 0.4 %.
5. Substrate on day 11 has ~0.17 g/L one-step uncertainty on a ~0.9 g/L level even with perfect knowledge of day 10, so ~12-15 % error there is close to irreducible.
6. ODE structure matters little once batch information is used: 20 kinetic variants all land within 0.2 of each other. Batch order carries no signal.
7. Temporal smoothing of predicted cell-density deviations gives a small, repeatable gain.

## Literature used

- Overflow metabolism and the production-to-consumption lactate switch, with glucose and lactate uptake complementary (Ma et al. 2009; Luginsland et al. 2023; Sietaram et al. 2024) -> overflow/re-consumption kinetics, faster re-consumption at low substrate.
- Lactate switch detectable from pH/base and feeding data (Li et al. 2026) -> unlogged volume gain as a base-addition sensor.
- Hybrid models with mass balances plus data-driven parts beat both pure mechanistic and pure statistical models at small data (Narayanan et al. 2019, 2022; Bayer et al. 2022) -> ODE + residual model.
- Mixed-effects / ensemble state-and-parameter estimation for batch-to-batch variation (Hickman et al. 2018; Yu et al. 2026) -> per-batch random effects and particle weighting.
"""
def table2():
    p=f'{LAB}/results_partition2.jsonl'
    if not os.path.exists(p): return []
    d2=pd.DataFrame([json.loads(l) for l in open(p)]).drop_duplicates('name',keep='last').set_index('name'); d1=results().set_index('name')
    if 'b04_b03_smoothcells' in d2.index: d2.loc['b04_hyb_stat_bayes_smoothcells']=d2.loc['b04_b03_smoothcells']
    keep=['s01_mean','s25_search_all_plus_base25','s26_evidence_all_plus_base25','m02_cov7_cal','m21_re8_cal','h_m11_consHill_evidence_allb25','p_m21_re8_kernel_b25','p2_m21_re8_evidence_all_n1000','b04_hyb_stat_bayes_smoothcells','b06_nosearch_smoothcells','b09_hyb_stat_bayesresid_smoothcells','b10_nosearch_bayesresid_smoothcells','b11_mechanistic_only_bayesresid']
    L=['','## Robustness check on a second, independent 7-fold partition','','Everything was re-fitted from scratch on a different random split of the 49 batches. Numbers are ~0.1 higher than on the primary split, which is the size of the selection optimism to expect.','','| approach | primary split | second split |','|---|---|---|']
    for n in keep:
        if n in d1.index and n in d2.index: L.append(f'| `{n}` | {d1.loc[n,"mean"]:.3f} | {d2.loc[n,"mean"]:.3f} |')
    return L
if __name__=='__main__':
    d=results(); best=d.iloc[0]
    L=['# Experiment log - Decode the Bioreactor','',
       f'Updated {time.strftime("%Y-%m-%d %H:%M")}. **{len(d)} approaches evaluated.** Score = mean MAPE (%) over the 4 targets, out-of-fold, on the same 7 batch-wise folds of the 49 labelled batches (every model refitted per fold). Leaderboard reference: #1 = 7.96, Team 18 = 8.03. Nothing here was tuned against the leaderboard; no submission has been made.','',
       f'**Current best: `{best["name"]}` - CV mean {best["mean"]:.3f}** (vcd {best.vcd:.2f}, tcd {best.tcd:.2f}, substrate {best.substrate:.2f}, byproduct {best.byproduct:.2f}). Its test predictions are in `submission_best.csv`.','',
       '- `submission_nosearch.csv` = `b10_nosearch_bayesresid_smoothcells` (same recipe, no grid search anywhere).','- `submission_mechanistic_blend.csv` = `b11_mechanistic_only_bayesresid` (ODE-based members only).','',
       'Caveat: 49 batches give a CV standard error of roughly 0.3, and picking the minimum of many approaches is optimistic. Differences below ~0.2 are not meaningful; the top ~15 rows are statistically tied.']
    L+=NOTES.split('\n')+table2()+['','## All approaches, best first','','| # | approach | family | vcd | tcd | substrate | byproduct | mean | what it is |','|---|---|---|---|---|---|---|---|---|']
    for i,r in d.iterrows(): L.append(f'| {i+1} | `{r["name"]}` | {r.family} | {r.vcd:.2f} | {r.tcd:.2f} | {r.substrate:.2f} | {r.byproduct:.2f} | **{r["mean"]:.3f}** | {r.desc} |')
    open(PROJ+'experiments_log.md','w').write('\n'.join(L)+'\n')
    write_submission(best['name'],'submission_best.csv')
    for n,o in [('b10_nosearch_bayesresid_smoothcells','submission_nosearch.csv'),('b11_mechanistic_only_bayesresid','submission_mechanistic_blend.csv')]:
        if (d.name==n).any(): write_submission(n,o)
    print(len(d),'approaches; best',best['name'],round(best['mean'],3))
    print(d[['name','family','mean']].head(int(sys.argv[1]) if len(sys.argv)>1 else 12).round(3).to_string())
    print(d.groupby('family')['mean'].agg(['count','min']).round(3))
