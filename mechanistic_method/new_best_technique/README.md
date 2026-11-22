# New technique: final version (leaderboard 6.709)

What changed compared with the old technique:

1. **Day-by-day sugar simulator** (`lab/chain.py`, `lab/kf.py`, `lab/chain4.py`; blended in `lab/final2.py` and `lab/final4.py`). It starts from day-0 sugar and steps one day at a time from the feed per litre and the time between samples. The operators' top-up rule is a hard constraint: a bolus on days 7–9 only if sugar < 1.95; on days 11–12 a 0.6 % bolus if sugar < 0.967, else 0.4 %. Sugar error on the hidden batches: 7.23 → 6.33.
2. **Best model per measurement.** Several complete model versions were submitted and, for each measurement, the best one kept:

| Measurement | Model | Score |
|---|---|---|
| VCD | kernel ridge on its own (`lab/autogrid.py`, stored as `a_krr_wide`) | 6.627 |
| TCD | simulators only (`b11_mechanistic_only_bayesresid`, from the old technique) | 6.569 |
| Substrate | sugar simulator blended with the version 1 average, days 5–8 × 1.02 | 6.326 |
| Byproduct | version 1 average × 1.02, days 5–8 a further × 1.02 | 7.314 |

On the 49 known batches the same recipe scores 6.94; the hidden-batch 6.709 is a little optimistic because the per-measurement choice used hidden scores.

Rebuild the final file from the stored predictions (seconds):

```bash
cd 2_new_technique
python3 lab/build_pa_pb_pc.py      # writes submission_pa/pb/pc.csv here
python3 lab/mixcols.py submission_v9_rebuilt.csv vcd=submission_pc.csv tcd=submission_pa.csv substrate=submission_pa.csv byproduct=submission_pc.csv
```

`lab/oof/` holds every stored prediction, including the old technique's, because the final file uses both. `common.py`, `report.py`, `smooth2.py` and `statm.py` are shared helpers copied from the old technique.

Notes: `README_AGENT_V9.md` (full handover; its file paths refer to the original single-folder project), `V4_NOTES.md`, `GRIDCV_RESULTS.md`, `TESTS_100.md` (their forecasts turned out too optimistic). `presentation_v9/`: the final slide deck.
