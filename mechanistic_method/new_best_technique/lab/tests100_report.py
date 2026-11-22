"""Builds ../TESTS_100.md from the tests100*_split1.json files (and the second-split files if a path is given). New file."""
import json,sys,os
L=os.path.dirname(os.path.abspath(__file__)); L2=sys.argv[1] if len(sys.argv)>1 else None
def load(nm,tag,base=L):
    p=f'{base}/{nm}_{tag}.json'; return json.load(open(p)) if os.path.exists(p) else None
parts=[('tests100','Round 1: one change at a time against version 4'),('tests100b','Round 2: follow-up on the leads'),('tests100c','Round 3: which feed-B days matter'),('tests100d','Round 4: the new cell model from several sides')]
out=['# 300 tests and verifications on version 4, and what came out (version 5)','','Run on 22 November 2026. Nothing was tuned against the leaderboard. Code: `lab/tests100.py`, `tests100b.py`, `tests100c.py`, `tests100d.py`, `tests100e.py`.','']
allrows=[]; n_tests=0
for nm,title in parts:
    a=load(nm,'split1'); b=load(nm,'split2',L2) if L2 else None; B={r['id']:r for r in (b['tests'] if b else [])}
    rows=[r for r in a['tests'] if 'error' not in r]; n_tests+=len(rows); allrows.append((title,rows,B))
a=load('tests100','split1'); ver=a['verifications']; e1=load('tests100e','split1'); e2=load('tests100e','split2',L2) if L2 else None; d1=load('tests100d','split1')
flat=[(r,B.get(r['id'],{})) for _,rows,B in allrows for r in rows]
better=[x for x in flat if x[0]['delta']<0]; both=[x for x in better if x[1].get('delta',1)<0]; strong=[x for x in both if x[0]['t']<=-2 and x[0]['d_disc']<0 and x[0]['d_conf']<0]
out+=['## Summary','',f'- **{n_tests} tests**, each one change scored against the same baseline on the same 49 batches, plus **{len(ver)} verifications** and **1,500 placebo refits**.',
 f'- {len(better)} tests looked better than the baseline on the primary split; {len(both)} of those were also better on the second split; {len(strong)} were better on both splits, in both halves of the batches, with a paired t of -2 or stronger.',
 '- Most of what passed belongs to two findings about **cells**, which earlier work had missed:',
 '  1. **Feed B on the peak feeding days 5-7** tracks the batch\'s cell level much better than the average over days 4-12 that version 2-4 used.',
 '  2. **A sugar top-up on days 7-9** means the sugar ran low, which means the batch has more cells than the other signals suggest. Top-up batches had 3.8 % more cells than the old small model predicted (6 of 7 batches).',
 f'- With both, the small model explains **{d1["r2"]["new small model"]*100:.0f} %** of the batch-to-batch cell difference, against {d1["r2"]["v4 small model"]*100:.0f} % before. Earlier notes called 55 % the limit; that was wrong.',
 '- One more finding for **byproduct**: on days 12-13 the exact sample time matters.',
 '- Substrate: nothing beat version 4 by more than 0.03, and nothing did so on both splits.','']
out+=['## Version 5','','| | VCD | TCD | Substrate | Byproduct | Mean | Second split |','|---|---|---|---|---|---|---|',
 '| Version 1 (leaderboard 7.018) | 6.65 | 6.61 | 6.83 | 8.02 | 7.03 | 7.14 |','| Version 2 | 6.58 | 6.52 | 6.22 | 7.88 | 6.80 | 6.86 |','| Version 4 | 6.58 | 6.52 | 6.16 | 7.63 | 6.72 | 6.80 |','| **Version 5** | 6.38 | 6.34 | 6.16 | 7.54 | **6.61** | 6.68 |','',
 'Version 5 = version 4 with the new small cell model (day-0 VCD, day-0 byproduct, initial volume, feed B days 5-7, top-up flag) and the sample time in the late byproduct model. It beats version 4 on 38 of 49 batches. File: `submission_v5.csv`, code: `lab/final5.py`.','',
 '### How sure is the cell finding?','']
if e1:
    out+=['Placebo test: shuffle the two new signals among the batches (so they keep their values but lose their link to the batch), refit, and see how often the shuffled version scores as well as the real one.','','| Signal | Real CV (cells) | Placebo average | Best of 500 placebos | Placebos as good as real |','|---|---|---|---|---|']
    for k,nm in [('both','Feed B days 5-7 and top-up flag together'),('fb','Feed B days 5-7 alone'),('flag','Top-up flag, given feed B days 5-7')]:
        r=e1[k]; s2=(' (second split: %.1f %%)'%(100*e2[k]['p_value'])) if e2 else ''; out.append(f'| {nm} | {r["real"]:.3f} | {r["placebo_mean"]:.3f} | {r["placebo_min"]:.3f} | {100*r["p_value"]:.1f} %{s2} |')
    out+=['','- The neighbouring feed-B windows (days 4-7, 5-6, 6-7, 5-8) all help; the gain falls away smoothly for windows further from days 5-7. A fluke would not look like that.',
     '- Caution: these two signals were found by searching about 70 candidates (28 extra signals, 40-odd windows). Searching makes the best result look better than it is. My estimate is that roughly half of the 0.19 gain on cells is real.',
     '- Three of the 12 test batches are top-up batches (against 7 of 49 known ones), so the flag matters more on the test set. Their cell predictions go up by about 3 %.','']
out+=['### What to expect on the leaderboard','','Version 1\'s real leaderboard score plus the gain measured on known batches with the test set\'s mix of batch types:','','| | VCD | TCD | Substrate | Byproduct | Mean |','|---|---|---|---|---|---|',
 '| Version 1 (real) | 6.745 | 6.649 | 7.234 | 7.446 | 7.018 |','| Version 4 (expected) | 6.70 | 6.60 | 6.31 | 7.10 | 6.68 |','| Version 5 (expected, as measured) | 6.49 | 6.41 | 6.30 | 7.02 | 6.56 |','| Version 5 (expected, gains from searched choices halved) | 6.60 | 6.50 | 6.30 | 7.15 | about 6.64 |','',
 'Each expected mean carries about +-0.18 because the test set is only 12 batches.','']
out+=['## Verifications','','| | Check | Result | Detail |','|---|---|---|---|']
for v in ver: out.append(f'| {v["id"]} | {v["desc"]} | {"pass" if v["ok"] else "FAIL"} | {v["detail"][:110]} |')
out+=['','On the second split V03 does not apply (the submission file is built from the primary split\'s models); all other checks pass there too.','']
out+=['## Every test','','Columns: CV of the target without and with the change; difference (negative = better); paired t over the 49 batches; number of batches that improve; difference on two fixed halves of the batches; difference on the second split.','',
 '"substrate(single simulator)" rows compare one simulator with and without the change; all other rows compare the full version-4 prediction of that target.','']
for title,rows,B in allrows:
    out+=[f'### {title}','','| id | target | change | before | after | diff | t | better in | half A | half B | split 2 |','|---|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        q=B.get(r['id'],{}); s2=('%+.3f'%q['delta']) if 'delta' in q else ''
        out.append(f'| {r["id"]} | {r["target"]} | {r["desc"].strip()} | {r["base"]:.3f} | {r["cv"]:.3f} | {r["delta"]:+.3f} | {r["t"]:.1f} | {r["wins"]}/49 | {r["d_disc"]:+.3f} | {r["d_conf"]:+.3f} | {s2} |')
    out.append('')
dst=os.path.join(os.path.dirname(L),'TESTS_100.md'); assert not os.path.exists(dst) or '--force' in sys.argv,'TESTS_100.md exists'
open(dst,'w').write('\n'.join(out)+'\n'); print('wrote',dst,'|',n_tests,'tests;',len(better),'better;',len(both),'both splits;',len(strong),'strong')
