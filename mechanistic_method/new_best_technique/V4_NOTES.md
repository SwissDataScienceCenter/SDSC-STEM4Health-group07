# Version 4: what changed and what to expect

Made on 22 November 2026. Nothing was tuned against the leaderboard. File: `submission_v4.csv`.

## Scores on the 49 known batches

| | VCD | TCD | Substrate | Byproduct | Mean | Second split |
|---|---|---|---|---|---|---|
| Version 1 (on the leaderboard: 7.018) | 6.65 | 6.61 | 6.83 | 8.02 | 7.03 | 7.14 |
| Version 2 | 6.58 | 6.52 | 6.22 | 7.88 | 6.80 | 6.86 |
| Version 3 | 6.58 | 6.52 | 6.22 | 7.64 | 6.74 | 6.81 |
| **Version 4** | 6.58 | 6.52 | 6.16 | 7.63 | **6.72** | 6.80 |

Version 4 beats version 2 on 35 of the 49 batches and version 1 on 33.

## What is new in version 4

1. **Batches with a mid-run sugar top-up use the simulators only.** Three of the 12 test batches got a top-up on day 7, 8 or 9, against 7 of the 49 known ones. The large models have only one to three such batches to learn each case from, and they were the weakest part of version 1 (10.5 % substrate error on those batches against 6 % on the rest). The day-by-day simulators handle a top-up as a known addition. On those batches substrate error goes 10.5 % (v1) -> 8.6 % (v2) -> 8.2 % (v4).
2. **Three more substrate simulators** that shrink the helper inputs harder than the physical ones (`lab/chain4.py`). Small gain.
3. **Byproduct blended in three phases** (this is version 3's change): days 1-4 half and half; days 5-8 the small model alone; days 9-13 mostly the large models.
4. **Exact sample time for byproduct on days 1-2.** Byproduct rises by about 1 g/L per day at the start, and samples are taken up to 0.1 day off schedule. Small gain on day 1.

Cells are unchanged from version 2.

## What to expect on the leaderboard

Take the real leaderboard score of version 1 and add the gain measured on known batches with the same mix as the test set (3 top-up batches, 5 flagged low on day 11, 2 low on day 12 only, 2 neither):

| | VCD | TCD | Substrate | Byproduct | Mean |
|---|---|---|---|---|---|
| Version 1 (real) | 6.745 | 6.649 | 7.234 | 7.446 | 7.018 |
| Version 2 (expected) | 6.70 | 6.60 | 6.42 | 7.31 | 6.76 |
| Version 4 (expected) | 6.70 | 6.60 | 6.30 | 7.10 | 6.67 |

- Each expected mean carries about +-0.17, because the test set is 12 batches.
- The version 4 figure is somewhat optimistic: the byproduct phases and the top-up rule were chosen on the same known batches. A fair figure is about 6.70.
- The byproduct phase weights are the least certain part. Version 2 is the more conservative file.

## Tried for version 4 and dropped

- A day-by-day simulator for byproduct (`lab/pchain.py`): worse alone and in the blend.
- Feed B as a read-out of the day's measured cell count: no relation (pooled partial correlation -0.03).
- A persistent batch effect in the substrate simulator: worse.
- Sample-time correction for cells: no gain (the effect is small next to the 6-7 % counting noise).
- Stratifying cells or byproduct by feed pattern: nothing stands out.

## Files added (nothing existing was changed)

`submission_v4.csv`, `V4_NOTES.md`, `lab/final4.py`, `lab/chain4.py`, `lab/sparse4.py`, `lab/pchain.py`, `lab/oof/b15_v4.npz`.

Rebuild: `cd lab && python final4.py` (about 2 minutes; needs the stored version 1 and 2 predictions in `lab/oof/`).
