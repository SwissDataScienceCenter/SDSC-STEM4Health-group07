# Automatic grid search with cross-validation: what it found

Run on 22 November 2026, after version 2. Nothing here was tuned against the leaderboard.

## Short answer

The grid search did not find a clearly better model than version 2. It confirmed version 2's settings for cells and substrate. It found one possible small gain, for byproduct, which is saved as a separate file.

| File | What it is | CV | Second split |
|---|---|---|---|
| `submission_v2.csv` | Version 2, settings fixed in advance | 6.80 | 6.86 |
| `submission_v3_byproduct_phases.csv` | Version 2 with the grid-searched byproduct blend | 6.74 (honest estimate about 6.78) | 6.81 |

Cells and substrate are identical in the two files. Only byproduct differs.

## Search 1: the version-2 recipe (`lab/gridcv.py`)

About 1,200 settings per target: which clues the small model uses, how strongly it is regularised, and how much weight it gets against the large models. Tried with one weight for the whole run and with separate weights for early, middle and late days.

| | VCD | TCD | Substrate | Byproduct | Mean |
|---|---|---|---|---|---|
| Version 2 (fixed) | 6.58 | 6.52 | 6.22 | 7.88 | 6.80 |
| Grid winner, one weight per target | 6.56 | 6.50 | 6.21 | 7.79 | 6.76 |
| Grid winner, three phases | 6.55 | 6.48 | 6.14 | 7.64 | 6.70 |
| One-weight search scored on held-out folds (nested) | 6.83 | 6.81 | 6.34 | 7.94 | 6.98 |
| Three-phase search scored on held-out folds (nested) | 6.90 | 6.85 | 6.39 | 7.97 | 7.03 |

The winners look better, but they were picked on the same 49 batches they are scored on. Two checks on how much is real:

- **Nested CV** (search on six folds, score on the seventh): worse than version 2 on every target.
- **Bootstrap** (`lab/gridcv_boot.py`; search on a random resample, score on the batches left out, 1,000 times): the search loses to the fixed setting for cells and substrate. For byproduct with three phases it gains about 0.09.

Why the search does not help: many settings are within 0.05 of each other, and 49 batches cannot tell them apart. The search then mostly picks up luck.

The one result that holds up is byproduct in three phases. Both splits pick the same thing:

| Days | Winner | Weight of the small model |
|---|---|---|
| 1-4 | small model with volume gain on days 2-5 and 5-8, initial volume, day-0 byproduct | 0.5 |
| 5-8 | small model alone (volume gain days 2-5, initial volume) | 1.0 |
| 9-13 | large models mostly; small model with the day-11/12 feed flags | 0.3 |

This fits the process: while the cells are producing byproduct, the unlogged volume gain says most of what can be known; once they consume it, the large models do better. It improved 36 of 49 batches on byproduct.

## Search 2: standard models with tuned settings (`lab/autogrid.py`)

Nine model families (ridge, lasso, elastic net, Huber, support vector regression, kernel ridge, PLS, gradient boosting, random forest), two sets of clues, settings chosen by scikit-learn's GridSearchCV inside each fold. 2,016 searches.

| Best of each kind | VCD | TCD | Byproduct |
|---|---|---|---|
| Version 2 | 6.58 | 6.52 | 7.88 |
| Best searched model alone (kernel ridge for cells, ridge for byproduct) | 6.84 | 6.78 | 8.69 |
| Best searched model added to version 2 as a third member | 6.57 | 6.52 | 7.87 |
| "Auto": let the inner CV pick the family per fold | 7.36 | 7.31 | 9.19 |

No family beats version 2, alone or added to it. Non-linear models (support vectors, trees, boosting) are no better than the four-clue straight-line model for cells, so there is no hidden non-linear pattern. This agrees with the earlier finding that cells are at the limit of what the inputs can tell.

## What to submit

- `submission_v2.csv` is still the main recommendation.
- `submission_v3_byproduct_phases.csv` is expected to be about 0.02 better on the mean, with real uncertainty either way. Use it if you can submit twice, or if you prefer the grid-searched byproduct.

## Files added

- `lab/gridcv.py`, `lab/gridcv_boot.py`, `lab/autogrid.py`
- `lab/gridcv_result.json`, `lab/autogrid_result.json`
- `lab/oof/g01_*.npz`, `g02_*.npz`, `b14_*.npz`, `a_*.npz` (stored predictions)
- `submission_v3_byproduct_phases.csv`, this file

No existing file was changed by this step.
