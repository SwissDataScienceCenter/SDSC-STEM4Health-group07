# 300 tests and verifications on version 4, and what came out (version 5)

Run on 22 November 2026. Nothing was tuned against the leaderboard. Code: `lab/tests100.py`, `tests100b.py`, `tests100c.py`, `tests100d.py`, `tests100e.py`.

## Summary

- **302 tests**, each one change scored against the same baseline on the same 49 batches, plus **20 verifications** and **1,500 placebo refits**.
- 156 tests looked better than the baseline on the primary split; 144 of those were also better on the second split; 38 were better on both splits, in both halves of the batches, with a paired t of -2 or stronger.
- Most of what passed belongs to two findings about **cells**, which earlier work had missed:
  1. **Feed B on the peak feeding days 5-7** tracks the batch's cell level much better than the average over days 4-12 that version 2-4 used.
  2. **A sugar top-up on days 7-9** means the sugar ran low, which means the batch has more cells than the other signals suggest. Top-up batches had 3.8 % more cells than the old small model predicted (6 of 7 batches).
- With both, the small model explains **65 %** of the batch-to-batch cell difference, against 55 % before. Earlier notes called 55 % the limit; that was wrong.
- One more finding for **byproduct**: on days 12-13 the exact sample time matters.
- Substrate: nothing beat version 4 by more than 0.03, and nothing did so on both splits.

## Version 5

| | VCD | TCD | Substrate | Byproduct | Mean | Second split |
|---|---|---|---|---|---|---|
| Version 1 (leaderboard 7.018) | 6.65 | 6.61 | 6.83 | 8.02 | 7.03 | 7.14 |
| Version 2 | 6.58 | 6.52 | 6.22 | 7.88 | 6.80 | 6.86 |
| Version 4 | 6.58 | 6.52 | 6.16 | 7.63 | 6.72 | 6.80 |
| **Version 5** | 6.38 | 6.34 | 6.16 | 7.54 | **6.61** | 6.68 |

Version 5 = version 4 with the new small cell model (day-0 VCD, day-0 byproduct, initial volume, feed B days 5-7, top-up flag) and the sample time in the late byproduct model. It beats version 4 on 38 of 49 batches. File: `submission_v5.csv`, code: `lab/final5.py`.

### How sure is the cell finding?

Placebo test: shuffle the two new signals among the batches (so they keep their values but lose their link to the batch), refit, and see how often the shuffled version scores as well as the real one.

| Signal | Real CV (cells) | Placebo average | Best of 500 placebos | Placebos as good as real |
|---|---|---|---|---|
| Feed B days 5-7 and top-up flag together | 6.363 | 6.589 | 6.382 | 0.0 % (second split: 1.0 %) |
| Feed B days 5-7 alone | 6.468 | 6.579 | 6.405 | 1.0 % (second split: 6.6 %) |
| Top-up flag, given feed B days 5-7 | 6.363 | 6.477 | 6.330 | 0.8 % (second split: 0.6 %) |

- The neighbouring feed-B windows (days 4-7, 5-6, 6-7, 5-8) all help; the gain falls away smoothly for windows further from days 5-7. A fluke would not look like that.
- Caution: these two signals were found by searching about 70 candidates (28 extra signals, 40-odd windows). Searching makes the best result look better than it is. My estimate is that roughly half of the 0.19 gain on cells is real.
- Three of the 12 test batches are top-up batches (against 7 of 49 known ones), so the flag matters more on the test set. Their cell predictions go up by about 3 %.

### What to expect on the leaderboard

Version 1's real leaderboard score plus the gain measured on known batches with the test set's mix of batch types:

| | VCD | TCD | Substrate | Byproduct | Mean |
|---|---|---|---|---|---|
| Version 1 (real) | 6.745 | 6.649 | 7.234 | 7.446 | 7.018 |
| Version 4 (expected) | 6.70 | 6.60 | 6.31 | 7.10 | 6.68 |
| Version 5 (expected, as measured) | 6.49 | 6.41 | 6.30 | 7.02 | 6.56 |
| Version 5 (expected, gains from searched choices halved) | 6.60 | 6.50 | 6.30 | 7.15 | about 6.64 |

Each expected mean carries about +-0.18 because the test set is only 12 batches.

## Verifications

| | Check | Result | Detail |
|---|---|---|---|
| V01 | submission_v4.csv has the template's columns, rows, batch ids and sample times | pass | 156 rows |
| V02 | submission_v4.csv has no missing, zero or negative values | pass |  |
| V03 | submission_v4.csv equals the stored version-4 test predictions | pass | max rel diff 0.0e+00 |
| V04 | predicted VCD <= predicted TCD everywhere in the file | pass | min TCD/VCD 1.0075 |
| V05 | test substrate predictions are on the side of the threshold that the feed log implies (60 batch-days) | pass | [] |
| V06 | same check on out-of-fold predictions of the 49 known batches (245 batch-days) | pass | 0 violations |
| V07 | the feed rule holds exactly on the measured values of all known batches | pass | 0 exceptions |
| V08 | every known batch is in exactly one fold; no test batch is in any fold | pass |  |
| V09 | stored version-4 CV reproduces | pass | 6.721 |
| V10 | version 4 cells are identical to version 2 cells | pass |  |
| V11 | small cell model is deterministic (two runs identical) | pass |  |
| V12 | leak check: with outcomes shuffled among batches the small cell model is no better than "every batch is typical" | pass | shuffled 8.83 vs no-information 8.72 vs real 6.67 |
| V13 | ignoring the feed decisions makes the simulator much worse (the constraint code is active) | pass | +3.81 |
| V14 | simulation noise is small: CV moves by less than 0.05 between random seeds | pass | sd of CV across seeds 0.022 |
| V15 | version-4 substrate can be rebuilt from its parts | pass | difference 0.000 |
| V16 | all test predictions lie within the range seen in known batches (25 % margin) | pass | 0 of 624 outside |
| V17 | test predictions spread about as much between batches as out-of-fold predictions do (ratio 0.6-1.6 per target) | pass | [1.24 1.24 0.93 0.75] |
| V18 | byproduct phase boundaries (day 4->5, 8->9) show no artificial jump: predicted day-to-day change matches the measured typical change | pass | d4->5 pred -0.018 vs measured -0.026; d8->9 pred -0.151 vs measured -0.150 |
| V19 | after a logged top-up the predicted substrate rises by more than 0.5 g/L next day (3 test batches) | pass | [] |
| V20 | version-1 CV matches its leaderboard score (7.028 vs 7.018) | pass | 7.028 |

On the second split V03 does not apply (the submission file is built from the primary split's models); all other checks pass there too.

## Every test

Columns: CV of the target without and with the change; difference (negative = better); paired t over the 49 batches; number of batches that improve; difference on two fixed halves of the batches; difference on the second split.

"substrate(single simulator)" rows compare one simulator with and without the change; all other rows compare the full version-4 prediction of that target.

### Round 1: one change at a time against version 4

| id | target | change | before | after | diff | t | better in | half A | half B | split 2 |
|---|---|---|---|---|---|---|---|---|---|---|
| C01 | cells | offset model + ls0 | 6.550 | 6.568 | +0.018 | 0.3 | 31/49 | -0.017 | +0.054 | -0.033 |
| C02 | cells | offset model + via0 | 6.550 | 6.584 | +0.034 | 1.7 | 13/49 | +0.004 | +0.066 | +0.072 |
| C03 | cells | offset model + tbar | 6.550 | 6.573 | +0.023 | 0.5 | 21/49 | +0.141 | -0.099 | +0.058 |
| C04 | cells | offset model + t1 | 6.550 | 6.568 | +0.018 | 0.5 | 25/49 | +0.097 | -0.064 | +0.031 |
| C05 | cells | offset model + nC | 6.550 | 6.566 | +0.016 | 0.5 | 27/49 | +0.044 | -0.012 | +0.003 |
| C06 | cells | offset model + base25 | 6.550 | 6.533 | -0.017 | -0.5 | 26/49 | -0.013 | -0.021 | -0.026 |
| C07 | cells | offset model + base13 | 6.550 | 6.509 | -0.041 | -1.2 | 29/49 | -0.055 | -0.026 | -0.077 |
| C08 | cells | offset model + base58 | 6.550 | 6.557 | +0.007 | 1.1 | 22/49 | +0.010 | +0.005 | +0.021 |
| C09 | cells | offset model + fbE | 6.550 | 6.600 | +0.050 | 2.0 | 21/49 | +0.031 | +0.070 | +0.005 |
| C10 | cells | offset model + fl11 | 6.550 | 6.530 | -0.020 | -0.4 | 25/49 | -0.129 | +0.094 | -0.028 |
| C11 | cells | offset model + fl12 | 6.550 | 6.562 | +0.013 | 0.9 | 23/49 | -0.001 | +0.027 | +0.016 |
| C12 | cells | offset model + any79 | 6.550 | 6.456 | -0.094 | -2.2 | 28/49 | -0.078 | -0.110 | -0.085 |
| C13 | cells | offset model + dV0 | 6.550 | 6.571 | +0.022 | 1.3 | 25/49 | +0.016 | +0.028 | +0.022 |
| C14 | cells | offset model + fb1 | 6.550 | 6.562 | +0.013 | 0.7 | 27/49 | +0.036 | -0.012 | +0.002 |
| C15 | cells | offset model + fb13 | 6.550 | 6.603 | +0.054 | 1.9 | 22/49 | +0.023 | +0.085 | +0.003 |
| C16 | cells | offset model + fb67 | 6.550 | 6.491 | -0.059 | -1.1 | 32/49 | -0.110 | -0.005 | -0.082 |
| C17 | cells | offset model + fb412 | 6.550 | 6.521 | -0.029 | -1.5 | 28/49 | -0.079 | +0.023 | -0.054 |
| C18 | cells | offset model + lpV | 6.550 | 6.547 | -0.003 | -1.1 | 24/49 | -0.006 | +0.000 | -0.002 |
| C19 | cells | offset model + lxV | 6.550 | 6.549 | -0.000 | -0.2 | 26/49 | +0.002 | -0.003 | -0.001 |
| C20 | cells | offset model + c_last | 6.550 | 6.608 | +0.059 | 1.7 | 26/49 | +0.098 | +0.018 | +0.074 |
| C21 | cells | offset model + c_first | 6.550 | 6.564 | +0.014 | 0.8 | 23/49 | +0.003 | +0.026 | +0.032 |
| C22 | cells | offset model + n23 | 6.550 | 6.568 | +0.018 | 0.5 | 24/49 | +0.032 | +0.004 | -0.002 |
| C23 | cells | offset model + nlate | 6.550 | 6.567 | +0.018 | 0.6 | 27/49 | +0.032 | +0.003 | +0.007 |
| C24 | cells | offset model + Shmid | 6.550 | 6.568 | +0.018 | 1.0 | 27/49 | +0.023 | +0.013 | +0.007 |
| C25 | cells | offset model + Sh11 | 6.550 | 6.552 | +0.003 | 0.1 | 29/49 | -0.063 | +0.071 | +0.004 |
| C26 | cells | offset model + fd5 | 6.550 | 6.551 | +0.002 | 0.1 | 22/49 | -0.013 | +0.016 | -0.012 |
| C27 | cells | offset model + dVr5_9 | 6.550 | 6.567 | +0.017 | 0.9 | 21/49 | -0.011 | +0.046 | +0.014 |
| C28 | cells | offset model + dVr9_13 | 6.550 | 6.557 | +0.007 | 0.6 | 23/49 | +0.006 | +0.009 | +0.012 |
| C29 | cells | offset model without late feed-B dosing | 6.550 | 6.570 | +0.020 | 0.4 | 22/49 | +0.062 | -0.024 | -0.051 |
| C30 | cells | offset model without initial volume | 6.550 | 6.581 | +0.031 | 0.6 | 24/49 | +0.104 | -0.045 | +0.042 |
| C31 | cells | offset model without day-0 byproduct | 6.550 | 6.716 | +0.167 | 2.0 | 24/49 | +0.247 | +0.083 | +0.138 |
| C32 | cells | offset model day-0 VCD only | 6.550 | 6.710 | +0.161 | 1.7 | 24/49 | +0.305 | +0.010 | +0.065 |
| C33 | cells | offset model fb67 instead of fbL | 6.550 | 6.491 | -0.059 | -1.5 | 33/49 | -0.096 | -0.020 | -0.089 |
| C34 | cells | offset model fb412 instead of fbL | 6.550 | 6.535 | -0.014 | -0.9 | 28/49 | -0.046 | +0.018 | -0.026 |
| C35 | cells | ridge strength 0.003 (v4: 0.03) | 6.550 | 6.544 | -0.006 | -1.0 | 27/49 | -0.014 | +0.002 | -0.003 |
| C36 | cells | ridge strength 0.3 (v4: 0.03) | 6.550 | 6.609 | +0.059 | 1.5 | 23/49 | +0.102 | +0.014 | +0.034 |
| C37 | cells | ridge strength 1.0 (v4: 0.03) | 6.550 | 6.746 | +0.196 | 2.3 | 20/49 | +0.282 | +0.107 | +0.121 |
| C38 | cells | per-day loadings, shrink 0.0 (v4: same offset every day) | 6.550 | 6.580 | +0.031 | 1.3 | 23/49 | +0.067 | -0.007 | +0.044 |
| C39 | cells | per-day loadings, shrink 0.5 (v4: same offset every day) | 6.550 | 6.556 | +0.007 | 0.5 | 25/49 | +0.030 | -0.018 | +0.014 |
| C40 | cells | offset defined on days 3-11 | 6.550 | 6.549 | -0.001 | -0.0 | 27/49 | -0.033 | +0.032 | +0.005 |
| C41 | cells | offset defined on days 2-9 | 6.550 | 6.549 | -0.001 | -0.0 | 29/49 | -0.011 | +0.010 | -0.019 |
| C42 | cells | offset defined on days 5-13 | 6.550 | 6.580 | +0.030 | 2.3 | 17/49 | +0.009 | +0.052 | +0.036 |
| C43 | cells | separate offsets for VCD and TCD | 6.550 | 6.550 | +0.000 | 0.3 | 24/49 | -0.000 | +0.000 | -0.000 |
| C44 | cells | Huber (outlier-robust) regression for the offset | 6.550 | 6.539 | -0.010 | -0.6 | 25/49 | -0.012 | -0.009 | -0.025 |
| C45 | cells | add squared day-0 VCD and VCD x byproduct terms | 6.550 | 6.583 | +0.033 | 2.9 | 18/49 | +0.030 | +0.036 | +0.007 |
| C46 | cells | sample-time correction per day (sparse4.rank1t) | 6.550 | 6.540 | -0.010 | -0.7 | 27/49 | -0.017 | -0.002 | -0.009 |
| C47 | cells | model total cells (density x volume), convert back with the known volume | 6.550 | 6.554 | +0.004 | 0.2 | 20/49 | +0.017 | -0.010 | +0.022 |
| C48 | cells | grid-searched svr_curated as the small model | 6.550 | 6.563 | +0.013 | 0.2 | 25/49 | -0.134 | +0.166 |  |
| C49 | cells | grid-searched krr_curated as the small model | 6.550 | 6.596 | +0.046 | 0.6 | 24/49 | -0.107 | +0.206 |  |
| C50 | cells | grid-searched rf_curated as the small model | 6.550 | 6.607 | +0.057 | 0.9 | 25/49 | +0.030 | +0.085 |  |
| C51 | cells | grid-searched huber_curated as the small model | 6.550 | 6.662 | +0.112 | 1.5 | 21/49 | +0.058 | +0.169 |  |
| C52 | cells | weight of small model 0.3 (v4: 0.5) | 6.550 | 6.560 | +0.010 | 0.3 | 25/49 | +0.068 | -0.050 | +0.008 |
| C53 | cells | weight of small model 0.4 (v4: 0.5) | 6.550 | 6.551 | +0.002 | 0.1 | 25/49 | +0.031 | -0.029 | -0.001 |
| C54 | cells | weight of small model 0.6 (v4: 0.5) | 6.550 | 6.554 | +0.004 | 0.3 | 24/49 | -0.025 | +0.035 | +0.008 |
| C55 | cells | weight of small model 0.7 (v4: 0.5) | 6.550 | 6.568 | +0.018 | 0.5 | 24/49 | -0.039 | +0.078 | +0.026 |
| C56 | cells | hybrid ODE only + small model | 6.550 | 6.556 | +0.006 | 0.2 | 24/49 | +0.056 | -0.046 | +0.025 |
| C57 | cells | hybrid + Bayesian ODE + small model | 6.550 | 6.539 | -0.011 | -0.2 | 26/49 | +0.012 | -0.035 | -0.013 |
| C58 | cells | five large models + small model | 6.550 | 6.571 | +0.022 | 0.8 | 20/49 | +0.054 | -0.012 | +0.002 |
| C59 | cells | TCD = predicted VCD / typical viability | 6.550 | 6.552 | +0.002 | 0.3 | 24/49 | +0.007 | -0.003 | -0.001 |
| C60 | cells | VCD = predicted TCD x typical viability | 6.550 | 6.548 | -0.001 | -0.2 | 25/49 | -0.006 | +0.004 | +0.003 |
| C61 | cells | average of direct and viability-based | 6.550 | 6.550 | +0.000 | 0.1 | 21/49 | -0.000 | +0.000 | +0.001 |
| C62 | cells | smooth the blended cell deviations once more | 6.550 | 6.552 | +0.002 | 0.5 | 25/49 | +0.004 | -0.001 | +0.002 |
| C63 | cells | pull cell deviations half-way to a constant offset | 6.550 | 6.557 | +0.007 | 0.6 | 22/49 | +0.011 | +0.004 | +0.008 |
| C64 | cells | multiply cell predictions by 0.995 | 6.550 | 6.567 | +0.017 | 0.5 | 24/49 | +0.031 | +0.004 | +0.017 |
| C65 | cells | multiply cell predictions by 1.005 | 6.550 | 6.560 | +0.010 | 0.3 | 22/49 | -0.009 | +0.031 | +0.008 |
| C66 | cells | multiply cell predictions by 0.99 | 6.550 | 6.612 | +0.062 | 0.9 | 22/49 | +0.080 | +0.045 | +0.063 |
| C67 | cells | multiply cell predictions by 1.01 | 6.550 | 6.595 | +0.046 | 0.6 | 22/49 | +0.000 | +0.093 | +0.039 |
| P01 | byproduct | early: without the days 5-8 volume gain | 7.627 | 7.649 | +0.022 | 0.9 | 20/49 | +0.008 | +0.037 | +0.026 |
| P02 | byproduct | early: + day-0 VCD | 7.627 | 7.629 | +0.002 | 0.2 | 26/49 | +0.005 | -0.000 | +0.010 |
| P03 | byproduct | early: + early feed-B dosing | 7.627 | 7.614 | -0.013 | -1.3 | 29/49 | +0.005 | -0.032 | -0.007 |
| P04 | byproduct | early: + feed-C pulse count days 2-3 | 7.627 | 7.629 | +0.002 | 0.2 | 21/49 | -0.006 | +0.010 | -0.004 |
| P05 | byproduct | early: + day-0 substrate | 7.627 | 7.640 | +0.013 | 0.8 | 26/49 | -0.009 | +0.035 | +0.033 |
| P06 | byproduct | early: volume gain split d2-3 / d4-5 | 7.627 | 7.631 | +0.004 | 0.6 | 23/49 | +0.006 | +0.002 | +0.023 |
| P07 | byproduct | early: no sample-time feature | 7.627 | 7.638 | +0.011 | 0.8 | 19/49 | +0.015 | +0.006 | +0.013 |
| P08 | byproduct | early: sample time on day 1 only | 7.627 | 7.629 | +0.002 | 0.5 | 20/49 | +0.002 | +0.001 | +0.001 |
| P09 | byproduct | early: sample time on days 1-4 | 7.627 | 7.626 | -0.001 | -0.1 | 25/49 | +0.012 | -0.015 | -0.013 |
| P10 | byproduct | early: weight 0.3 | 7.627 | 7.645 | +0.018 | 1.1 | 21/49 | +0.019 | +0.016 | +0.039 |
| P11 | byproduct | early: weight 0.7 | 7.627 | 7.634 | +0.007 | 0.4 | 24/49 | +0.018 | -0.004 | -0.010 |
| P12 | byproduct | early: ridge 0.03 | 7.627 | 7.628 | +0.001 | 0.3 | 27/49 | +0.001 | +0.001 | +0.004 |
| P13 | byproduct | early: + predicted cell offset | 7.627 | 7.613 | -0.014 | -1.1 | 24/49 | +0.004 | -0.033 | +0.011 |
| P14 | byproduct | mid: + day-0 byproduct | 7.627 | 7.662 | +0.035 | 0.9 | 23/49 | +0.040 | +0.029 | +0.011 |
| P15 | byproduct | mid: + day-0 VCD | 7.627 | 7.636 | +0.009 | 0.3 | 23/49 | +0.038 | -0.023 | +0.007 |
| P16 | byproduct | mid: + days 5-8 volume gain | 7.627 | 7.662 | +0.035 | 1.1 | 21/49 | +0.058 | +0.012 | -0.021 |
| P17 | byproduct | mid: + predicted cell offset | 7.627 | 7.645 | +0.018 | 0.7 | 25/49 | +0.045 | -0.009 | -0.021 |
| P18 | byproduct | mid: + feed-C pulses late | 7.627 | 7.634 | +0.007 | 0.1 | 26/49 | -0.045 | +0.060 | +0.016 |
| P19 | byproduct | mid: ridge 0.03 | 7.627 | 7.631 | +0.004 | 0.9 | 26/49 | +0.002 | +0.006 | +0.007 |
| P20 | byproduct | mid: weight 0.75 (rest large models) | 7.627 | 7.649 | +0.022 | 0.8 | 24/49 | +0.033 | +0.011 | +0.049 |
| P21 | byproduct | mid: weight 0.5 | 7.627 | 7.702 | +0.075 | 1.3 | 22/49 | +0.098 | +0.051 | +0.132 |
| P22 | byproduct | mid: + predicted substrate days 6-9 | 7.627 | 7.649 | +0.022 | 1.2 | 23/49 | +0.023 | +0.020 | +0.096 |
| P23 | byproduct | late: large models only | 7.627 | 7.693 | +0.066 | 0.8 | 22/49 | -0.037 | +0.173 | +0.088 |
| P24 | byproduct | late: weight 0.2 | 7.627 | 7.636 | +0.009 | 0.3 | 22/49 | -0.017 | +0.036 | +0.009 |
| P25 | byproduct | late: weight 0.5 | 7.627 | 7.661 | +0.034 | 0.7 | 25/49 | +0.045 | +0.023 | +0.017 |
| P26 | byproduct | late: without feed flags | 7.627 | 7.670 | +0.043 | 0.8 | 20/49 | +0.044 | +0.042 | +0.010 |
| P27 | byproduct | late: per-day model instead of one offset | 7.627 | 7.637 | +0.010 | 0.4 | 24/49 | +0.007 | +0.013 | +0.028 |
| P28 | byproduct | late: per-day model with sample time on day 13 | 7.627 | 7.557 | -0.070 | -1.8 | 30/49 | -0.058 | -0.082 | -0.034 |
| P29 | byproduct | late: + predicted substrate day 11/12 | 7.627 | 7.622 | -0.005 | -0.3 | 24/49 | -0.003 | -0.007 | +0.023 |
| P30 | byproduct | late: + predicted cell offset | 7.627 | 7.617 | -0.010 | -0.6 | 26/49 | -0.015 | -0.003 | -0.037 |
| P31 | byproduct | late: + feed-C pulses late | 7.627 | 7.636 | +0.009 | 0.8 | 24/49 | -0.010 | +0.030 | +0.028 |
| P32 | byproduct | late: + late unlogged volume change | 7.627 | 7.613 | -0.014 | -0.7 | 27/49 | -0.036 | +0.009 | -0.007 |
| P33 | byproduct | phases: mid = days 5-7 | 7.627 | 7.646 | +0.019 | 0.6 | 29/49 | -0.027 | +0.067 | +0.029 |
| P34 | byproduct | phases: mid = days 4-8 | 7.627 | 7.659 | +0.032 | 1.1 | 24/49 | +0.029 | +0.035 | +0.051 |
| P35 | byproduct | phases: mid = days 5-9 | 7.627 | 7.720 | +0.093 | 2.7 | 16/49 | +0.110 | +0.076 | +0.087 |
| P36 | byproduct | phases: mid = days 6-8 | 7.627 | 7.643 | +0.016 | 0.6 | 21/49 | +0.035 | -0.005 | +0.016 |
| P37 | byproduct | scale all by 1.01 | 7.627 | 7.585 | -0.042 | -0.6 | 28/49 | -0.010 | -0.075 | -0.075 |
| P38 | byproduct | scale all by 0.99 | 7.627 | 7.797 | +0.170 | 2.3 | 18/49 | +0.141 | +0.201 | +0.189 |
| P39 | byproduct | scale all by 1.02 | 7.627 | 7.652 | +0.025 | 0.2 | 25/49 | +0.061 | -0.012 | -0.028 |
| P40 | byproduct | mid: model total mass (conc x volume), convert back | 7.627 | 7.635 | +0.008 | 0.4 | 29/49 | +0.010 | +0.006 | +0.016 |
| P41 | byproduct | smooth byproduct deviations over time once | 7.627 | 7.699 | +0.072 | 1.3 | 24/49 | +0.044 | +0.101 | +0.129 |
| P42 | byproduct | version 2 byproduct (one weight 0.5 for all days) | 7.627 | 7.876 | +0.249 | 2.6 | 13/49 | +0.260 | +0.238 | +0.224 |
| P43 | byproduct | version 1 byproduct (large models only) | 7.627 | 8.022 | +0.395 | 2.3 | 19/49 | +0.325 | +0.468 | +0.527 |
| S01 | substrate | rebuilt v4 substrate from parts (should equal v4 up to simulation noise) | 6.157 | 6.157 | +0.000 | 0.0 | 0/49 | +0.000 | +0.000 | +0.000 |
| S02 | substrate | simulator weight 0.5 (v4: 0.7) | 6.157 | 6.176 | +0.020 | 0.3 | 20/49 | -0.021 | +0.062 | +0.029 |
| S03 | substrate | simulator weight 0.6 | 6.157 | 6.158 | +0.002 | 0.1 | 20/49 | -0.017 | +0.022 | +0.002 |
| S04 | substrate | simulator weight 0.8 | 6.157 | 6.177 | +0.020 | 0.6 | 18/49 | +0.044 | -0.004 | +0.017 |
| S05 | substrate | simulator weight 0.9 | 6.157 | 6.212 | +0.056 | 0.8 | 17/49 | +0.102 | +0.007 | +0.049 |
| S06 | substrate | simulators only | 6.157 | 6.264 | +0.108 | 1.1 | 16/49 | +0.176 | +0.036 | +0.095 |
| S07 | substrate | no special rule for top-up batches | 6.157 | 6.215 | +0.059 | 0.9 | 4/49 | +0.072 | +0.045 | +0.032 |
| S08 | substrate | top-up batches: simulator weight 0.85 | 6.157 | 6.181 | +0.024 | 0.8 | 4/49 | +0.026 | +0.022 | +0.012 |
| S09 | substrate | chain ensemble only | 6.157 | 6.207 | +0.051 | 1.2 | 17/49 | +0.098 | +0.001 | -0.013 |
| S10 | substrate | state-space ensemble only | 6.157 | 6.163 | +0.007 | 0.1 | 22/49 | -0.065 | +0.081 | +0.052 |
| S11 | substrate | selective-shrinkage chains only | 6.157 | 6.174 | +0.017 | 0.4 | 21/49 | +0.049 | -0.017 | +0.023 |
| S12 | substrate | chain + state-space (as v2) | 6.157 | 6.163 | +0.006 | 0.2 | 27/49 | -0.017 | +0.030 | +0.003 |
| S13 | substrate | days 1-6 weight 0.5, days 7-13 weight 0.85 | 6.157 | 6.130 | -0.026 | -0.5 | 20/49 | +0.026 | -0.081 | -0.001 |
| S14 | substrate | days 1-6 weight 0.85, days 7-13 weight 0.5 | 6.157 | 6.239 | +0.082 | 1.3 | 16/49 | +0.025 | +0.142 | +0.062 |
| S15 | substrate | days 11-13 simulators only | 6.157 | 6.151 | -0.005 | -0.1 | 21/49 | +0.114 | -0.130 | -0.004 |
| S16 | substrate | scale by 0.99 | 6.157 | 6.204 | +0.047 | 0.7 | 23/49 | -0.009 | +0.106 | +0.048 |
| S17 | substrate | scale by 1.01 | 6.157 | 6.265 | +0.109 | 1.5 | 21/49 | +0.154 | +0.062 | +0.099 |
| S18 | substrate(single simulator) | reference simulator, other random seed 1 | 6.309 | 6.300 | -0.009 | -0.3 | 20/49 | +0.006 | -0.023 | -0.031 |
| S19 | substrate(single simulator) | reference simulator, other random seed 2 | 6.309 | 6.272 | -0.037 | -1.1 | 31/49 | -0.008 | -0.067 | -0.047 |
| S20 | substrate(single simulator) | reference simulator, other random seed 3 | 6.309 | 6.326 | +0.017 | 0.7 | 25/49 | +0.025 | +0.009 | -0.013 |
| S21 | substrate(single simulator) | 1,000 particles (ref: 4,000) | 6.309 | 6.318 | +0.010 | 0.3 | 25/49 | +0.002 | +0.018 | +0.001 |
| S22 | substrate(single simulator) | 16,000 particles | 6.309 | 6.294 | -0.015 | -0.9 | 26/49 | +0.015 | -0.047 | -0.015 |
| S23 | substrate(single simulator) | threshold days 7-9 = 1.92 (ref 1.95) | 6.309 | 6.345 | +0.036 | 1.3 | 25/49 | -0.014 | +0.088 | +0.043 |
| S24 | substrate(single simulator) | threshold days 7-9 = 1.98 | 6.309 | 6.284 | -0.025 | -0.9 | 20/49 | +0.011 | -0.062 | -0.016 |
| S25 | substrate(single simulator) | threshold days 11-12 = 0.960 (ref 0.967) | 6.309 | 6.305 | -0.003 | -0.2 | 24/49 | +0.009 | -0.016 | -0.000 |
| S26 | substrate(single simulator) | threshold days 11-12 = 0.964 | 6.309 | 6.308 | -0.001 | -0.2 | 23/49 | +0.003 | -0.005 | +0.001 |
| S27 | substrate(single simulator) | threshold days 11-12 = 0.969 | 6.309 | 6.310 | +0.001 | 0.3 | 24/49 | -0.003 | +0.006 | +0.001 |
| S28 | substrate(single simulator) | point = plain median of the paths | 6.309 | 6.243 | -0.065 | -0.6 | 22/49 | +0.006 | -0.139 | -0.052 |
| S29 | substrate(single simulator) | point = median weighted by 1/sqrt(y) | 6.309 | 6.255 | -0.054 | -0.8 | 23/49 | -0.009 | -0.100 | -0.036 |
| S30 | substrate(single simulator) | point = median weighted by 1/y^1.5 | 6.309 | 6.341 | +0.033 | 0.7 | 23/49 | +0.023 | +0.043 | +0.024 |
| S31 | substrate(single simulator) | point = median weighted by 1/y^2 | 6.309 | 6.367 | +0.058 | 0.7 | 22/49 | +0.061 | +0.054 | +0.060 |
| S32 | substrate(single simulator) | point = mean of the paths | 6.309 | 6.235 | -0.073 | -0.6 | 25/49 | -0.015 | -0.135 | -0.055 |
| S33 | substrate(single simulator) | point = harmonic mean of the paths | 6.309 | 6.273 | -0.036 | -0.8 | 24/49 | -0.023 | -0.048 | -0.049 |
| S34 | substrate(single simulator) | scatter x0.8 | 6.309 | 6.360 | +0.052 | 0.9 | 22/49 | +0.066 | +0.037 | +0.063 |
| S35 | substrate(single simulator) | scatter x0.9 | 6.309 | 6.336 | +0.027 | 0.8 | 22/49 | +0.012 | +0.044 | +0.032 |
| S36 | substrate(single simulator) | scatter x1.1 | 6.309 | 6.313 | +0.005 | 0.2 | 24/49 | -0.008 | +0.018 | -0.005 |
| S37 | substrate(single simulator) | scatter x1.3 | 6.309 | 6.343 | +0.035 | 0.4 | 24/49 | +0.001 | +0.070 | +0.035 |
| S38 | substrate(single simulator) | scatter of the day 10->11 step x0.8 | 6.309 | 6.320 | +0.012 | 0.4 | 22/49 | +0.012 | +0.011 | +0.017 |
| S39 | substrate(single simulator) | scatter of the day 10->11 step x1.3 | 6.309 | 6.295 | -0.014 | -0.2 | 25/49 | -0.012 | -0.015 | -0.021 |
| S40 | substrate(single simulator) | late scatter x0.85 | 6.309 | 6.317 | +0.008 | 0.3 | 23/49 | +0.010 | +0.006 | +0.012 |
| S41 | substrate(single simulator) | late scatter x1.2 | 6.309 | 6.307 | -0.002 | -0.1 | 26/49 | -0.008 | +0.005 | -0.009 |
| S42 | substrate(single simulator) | early scatter x0.8 | 6.309 | 6.330 | +0.021 | 1.1 | 23/49 | +0.009 | +0.034 | +0.028 |
| S43 | substrate(single simulator) | early scatter x1.2 | 6.309 | 6.303 | -0.006 | -0.3 | 28/49 | -0.014 | +0.003 | -0.015 |
| S44 | substrate(single simulator) | without predicted cells as helper | 6.309 | 6.420 | +0.112 | 0.5 | 24/49 | -0.273 | +0.512 | +0.104 |
| S45 | substrate(single simulator) | without predicted byproduct as helper | 6.309 | 6.285 | -0.024 | -0.2 | 23/49 | +0.101 | -0.154 | +0.246 |
| S46 | substrate(single simulator) | without unlogged volume gain as helper | 6.309 | 6.425 | +0.116 | 2.5 | 16/49 | +0.080 | +0.154 | +0.127 |
| S47 | substrate(single simulator) | with the large models' substrate prediction as helper | 6.309 | 6.470 | +0.161 | 0.9 | 23/49 | -0.159 | +0.495 | +0.047 |
| S48 | substrate(single simulator) | helpers taken from version 4 instead of stage 1 | 6.309 | 6.308 | -0.001 | -0.0 | 26/49 | -0.028 | +0.027 | +0.082 |
| S49 | substrate(single simulator) | physical terms shrunk more (0.01) | 6.309 | 6.295 | -0.013 | -0.3 | 26/49 | +0.019 | -0.047 | +0.023 |
| S50 | substrate(single simulator) | helpers shrunk less (0.03) | 6.309 | 6.351 | +0.043 | 1.4 | 20/49 | +0.061 | +0.024 | -0.007 |
| S51 | substrate(single simulator) | helpers shrunk more (0.3) | 6.309 | 6.270 | -0.039 | -1.0 | 28/49 | -0.079 | +0.003 | +0.031 |
| S52 | substrate(single simulator) | helpers shrunk more (1.0) | 6.309 | 6.348 | +0.040 | 0.3 | 25/49 | -0.107 | +0.192 | +0.210 |
| S53 | substrate(single simulator) | VERIFY: feed decisions ignored (must be much worse) | 6.309 | 10.118 | +3.810 | 5.2 | 8/49 | +2.891 | +4.767 | +3.773 |
| S54 | substrate(single simulator) | only the day 11-12 decisions used | 6.309 | 6.939 | +0.631 | 2.2 | 18/49 | +0.105 | +1.178 | +0.664 |
| S55 | substrate(single simulator) | only the day 7-9 decisions used | 6.309 | 8.749 | +2.440 | 4.5 | 9/49 | +1.618 | +3.297 | +2.423 |
| S56 | substrate(single simulator) | step includes yesterday squared | 6.309 | 6.324 | +0.015 | 1.2 | 24/49 | -0.007 | +0.038 | +0.004 |
| S57 | substrate(single simulator) | step includes yesterday x interval | 6.309 | 6.310 | +0.001 | 2.6 | 21/49 | +0.001 | +0.001 | +0.001 |
| S58 | substrate(single simulator) | step also uses the day before yesterday | 6.309 | 6.254 | -0.055 | -0.3 | 34/49 | -0.205 | +0.101 | +0.055 |
| S59 | substrate(single simulator) | + day-0 VCD as helper in every step | 6.309 | 6.314 | +0.005 | 0.2 | 28/49 | -0.060 | +0.072 | +0.031 |
| S60 | substrate(single simulator) | + late feed-B dosing as helper | 6.309 | 6.274 | -0.035 | -0.3 | 24/49 | +0.003 | -0.074 | -0.063 |
| S61 | substrate(single simulator) | + volume gain d2-5 as helper | 6.309 | 6.251 | -0.057 | -1.6 | 26/49 | -0.054 | -0.061 | -0.029 |
| S62 | substrate(single simulator) | + predicted cell offset as helper | 6.309 | 6.309 | -0.000 | -0.0 | 22/49 | -0.003 | +0.003 | -0.017 |
| S63 | substrate(single simulator) | state-space: process scatter as fitted (ref x1.4) | 6.278 | 6.353 | +0.075 | 0.9 | 18/49 | +0.095 | +0.056 | +0.101 |
| S64 | substrate(single simulator) | state-space: process scatter x2 | 6.278 | 6.373 | +0.096 | 1.0 | 22/49 | +0.034 | +0.159 | +0.178 |
| S65 | substrate(single simulator) | state-space: helpers shrunk more (0.1) | 6.278 | 6.411 | +0.134 | 0.6 | 24/49 | +0.018 | +0.254 | +0.114 |
| S66 | substrate(single simulator) | state-space: assay-noise floor 2 % | 6.278 | 6.330 | +0.053 | 0.6 | 23/49 | +0.040 | +0.066 | +0.096 |
| S67 | substrate(single simulator) | state-space: pooled bolus coefficient | 6.278 | 6.305 | +0.027 | 0.3 | 21/49 | -0.018 | +0.074 | +0.022 |
| S68 | substrate(single simulator) | state-space: other seed | 6.278 | 6.307 | +0.029 | 1.2 | 21/49 | +0.018 | +0.041 | -0.021 |

### Round 2: follow-up on the leads

| id | target | change | before | after | diff | t | better in | half A | half B | split 2 |
|---|---|---|---|---|---|---|---|---|---|---|
| F01 | cells | top-up flag (any of days 7-9) | 6.550 | 6.456 | -0.094 | -2.2 | 28/49 | -0.078 | -0.110 | -0.085 |
| F02 | cells | top-up flag, ridge 0.1 | 6.550 | 6.484 | -0.065 | -1.8 | 28/49 | -0.035 | -0.097 | -0.064 |
| F03 | cells | top-up flag, ridge 0.3 | 6.550 | 6.554 | +0.004 | 0.1 | 23/49 | +0.059 | -0.053 | -0.015 |
| F04 | cells | separate flags for day 7, 8, 9 | 6.550 | 6.467 | -0.083 | -1.7 | 29/49 | -0.083 | -0.083 | -0.027 |
| F05 | cells | graded: substrate pulled down by the decisions, days 7-10 | 6.550 | 6.507 | -0.042 | -1.0 | 30/49 | -0.000 | -0.086 | -0.055 |
| F06 | cells | graded: days 11-13 | 6.550 | 6.520 | -0.030 | -1.0 | 25/49 | +0.001 | -0.062 | -0.113 |
| F07 | cells | graded: days 7-13 | 6.550 | 6.513 | -0.037 | -1.1 | 26/49 | +0.001 | -0.077 | -0.109 |
| F08 | cells | top-up flag + graded days 7-10 | 6.550 | 6.465 | -0.084 | -1.7 | 31/49 | -0.054 | -0.116 | -0.080 |
| F09 | cells | top-up flag + day-11 flag | 6.550 | 6.462 | -0.088 | -1.5 | 26/49 | -0.152 | -0.021 | -0.089 |
| F10 | cells | top-up flag + day-11 + day-12 flags | 6.550 | 6.483 | -0.066 | -1.0 | 27/49 | -0.141 | +0.011 | -0.068 |
| F11 | cells | top-up flag, without late feed-B dosing | 6.550 | 6.504 | -0.046 | -0.8 | 23/49 | +0.025 | -0.120 | -0.113 |
| F12 | cells | top-up flag + early volume gain d2-3 | 6.550 | 6.419 | -0.131 | -2.2 | 29/49 | -0.126 | -0.135 | -0.168 |
| F13 | cells | top-up flag + fb67 instead of fbL | 6.550 | 6.351 | -0.198 | -3.3 | 37/49 | -0.201 | -0.196 | -0.207 |
| F14 | cells | top-up flag, small-model weight 0.4 | 6.550 | 6.473 | -0.077 | -1.8 | 31/49 | -0.035 | -0.121 | -0.070 |
| F15 | cells | top-up flag, small-model weight 0.6 | 6.550 | 6.449 | -0.101 | -2.1 | 28/49 | -0.111 | -0.091 | -0.091 |
| F16 | cells | top-up flag, per-day loadings (shrink 0.5) | 6.550 | 6.470 | -0.080 | -1.8 | 28/49 | -0.053 | -0.108 | -0.067 |
| F17 | cells | PLACEBO: random flag on 7 known + 3 test batches (#1) | 6.550 | 6.569 | +0.020 | 0.5 | 25/49 | +0.058 | -0.020 | +0.071 |
| F18 | cells | PLACEBO: random flag on 7 known + 3 test batches (#2) | 6.550 | 6.600 | +0.051 | 2.3 | 15/49 | +0.007 | +0.097 | +0.006 |
| F19 | cells | PLACEBO: random flag on 7 known + 3 test batches (#3) | 6.550 | 6.555 | +0.006 | 0.4 | 26/49 | +0.017 | -0.006 | +0.016 |
| F20 | cells | PLACEBO: random flag on 7 known + 3 test batches (#4) | 6.550 | 6.512 | -0.038 | -1.1 | 30/49 | -0.044 | -0.032 | -0.050 |
| F21 | cells | PLACEBO: random flag on 7 known + 3 test batches (#5) | 6.550 | 6.584 | +0.035 | 1.1 | 24/49 | -0.009 | +0.080 | +0.064 |
| F22 | byproduct | late: per-day model + sample time on day 13 | 7.627 | 7.557 | -0.070 | -1.8 | 30/49 | -0.058 | -0.082 | -0.034 |
| F23 | byproduct | late: same, without sample time (control) | 7.627 | 7.637 | +0.010 | 0.4 | 24/49 | +0.007 | +0.013 | +0.028 |
| F24 | byproduct | late: per-day + sample time on days 12-13 | 7.627 | 7.538 | -0.089 | -2.1 | 32/49 | -0.091 | -0.088 | -0.051 |
| F25 | byproduct | late: per-day + sample time days 11-13 | 7.627 | 7.532 | -0.095 | -2.0 | 32/49 | -0.105 | -0.085 | -0.051 |
| F26 | byproduct | late: per-day + time d13, weight 0.2 | 7.627 | 7.582 | -0.045 | -1.3 | 28/49 | -0.067 | -0.023 | -0.018 |
| F27 | byproduct | late: per-day + time d13, weight 0.4 | 7.627 | 7.558 | -0.069 | -1.2 | 28/49 | -0.048 | -0.090 | -0.033 |
| F28 | byproduct | late: per-day + time d13, ridge 0.03 | 7.627 | 7.562 | -0.065 | -1.8 | 31/49 | -0.059 | -0.071 | -0.027 |
| F29 | byproduct | late: per-day + time d13 + top-up flag | 7.627 | 7.557 | -0.070 | -1.7 | 29/49 | -0.067 | -0.072 | -0.055 |
| F30 | byproduct | late: offset model + top-up flag | 7.627 | 7.639 | +0.012 | 1.5 | 19/49 | +0.005 | +0.020 | +0.016 |
| F31 | byproduct | mid: + top-up flag | 7.627 | 7.642 | +0.015 | 0.5 | 24/49 | +0.020 | +0.010 | +0.056 |
| F32 | byproduct | early: + top-up flag | 7.627 | 7.615 | -0.012 | -0.9 | 22/49 | +0.003 | -0.028 | -0.010 |
| F33 | byproduct | late: per-day + time d13, scale 1.01 | 7.627 | 7.524 | -0.103 | -1.3 | 28/49 | -0.065 | -0.143 | -0.103 |
| F34 | byproduct | late: per-day + time d13; mid weight 0.75 | 7.627 | 7.579 | -0.048 | -1.1 | 29/49 | -0.024 | -0.072 | +0.015 |
| F35 | substrate | pool + 2 simulators with volume gain d2-5 as helper | 6.157 | 6.144 | -0.013 | -0.8 | 26/49 | -0.001 | -0.025 | -0.011 |
| F36 | substrate | pool + simulator with 1/sqrt(y) weighting | 6.157 | 6.150 | -0.006 | -0.4 | 26/49 | +0.009 | -0.023 | -0.002 |
| F37 | substrate | pool + simulator with plain median | 6.157 | 6.147 | -0.009 | -0.5 | 26/49 | +0.013 | -0.032 | -0.009 |
| F38 | substrate | pool + simulator with mean of paths | 6.157 | 6.150 | -0.006 | -0.4 | 24/49 | +0.010 | -0.023 | -0.003 |
| F39 | substrate | pool + one more reference simulator (control) | 6.157 | 6.156 | -0.001 | -0.1 | 22/49 | +0.011 | -0.013 | +0.002 |
| F40 | substrate | pool + helper & 1/sqrt(y) simulator | 6.157 | 6.139 | -0.017 | -0.9 | 29/49 | +0.005 | -0.040 | -0.014 |
| F41 | substrate | pool + all four alternatives | 6.157 | 6.144 | -0.013 | -0.8 | 29/49 | +0.003 | -0.029 | -0.010 |

### Round 3: which feed-B days matter

| id | target | change | before | after | diff | t | better in | half A | half B | split 2 |
|---|---|---|---|---|---|---|---|---|---|---|
| W01 | cells | feed B days 2-2 + top-up flag (in place of late feed-B factor) | 6.550 | 6.535 | -0.015 | -0.2 | 26/49 | +0.042 | -0.074 | -0.114 |
| W02 | cells | feed B days 2-3 + top-up flag (in place of late feed-B factor) | 6.550 | 6.489 | -0.061 | -0.8 | 31/49 | -0.060 | -0.062 | -0.167 |
| W03 | cells | feed B days 2-4 + top-up flag (in place of late feed-B factor) | 6.550 | 6.504 | -0.045 | -0.7 | 29/49 | -0.016 | -0.076 | -0.133 |
| W04 | cells | feed B days 2-5 + top-up flag (in place of late feed-B factor) | 6.550 | 6.464 | -0.086 | -1.5 | 30/49 | -0.120 | -0.049 | -0.158 |
| W05 | cells | feed B days 3-3 + top-up flag (in place of late feed-B factor) | 6.550 | 6.469 | -0.080 | -1.1 | 29/49 | -0.094 | -0.066 | -0.185 |
| W06 | cells | feed B days 3-4 + top-up flag (in place of late feed-B factor) | 6.550 | 6.499 | -0.051 | -0.9 | 29/49 | -0.018 | -0.085 | -0.130 |
| W07 | cells | feed B days 3-5 + top-up flag (in place of late feed-B factor) | 6.550 | 6.459 | -0.090 | -1.7 | 31/49 | -0.132 | -0.047 | -0.150 |
| W08 | cells | feed B days 3-6 + top-up flag (in place of late feed-B factor) | 6.550 | 6.437 | -0.113 | -1.9 | 34/49 | -0.156 | -0.067 | -0.179 |
| W09 | cells | feed B days 4-4 + top-up flag (in place of late feed-B factor) | 6.550 | 6.514 | -0.035 | -0.6 | 29/49 | +0.026 | -0.100 | -0.088 |
| W10 | cells | feed B days 4-5 + top-up flag (in place of late feed-B factor) | 6.550 | 6.467 | -0.083 | -1.8 | 30/49 | -0.111 | -0.053 | -0.128 |
| W11 | cells | feed B days 4-6 + top-up flag (in place of late feed-B factor) | 6.550 | 6.441 | -0.109 | -2.0 | 34/49 | -0.145 | -0.072 | -0.163 |
| W12 | cells | feed B days 4-7 + top-up flag (in place of late feed-B factor) | 6.550 | 6.406 | -0.144 | -2.8 | 33/49 | -0.179 | -0.107 | -0.176 |
| W13 | cells | feed B days 5-5 + top-up flag (in place of late feed-B factor) | 6.550 | 6.449 | -0.101 | -1.9 | 30/49 | -0.183 | -0.015 | -0.142 |
| W14 | cells | feed B days 5-6 + top-up flag (in place of late feed-B factor) | 6.550 | 6.398 | -0.152 | -2.4 | 30/49 | -0.231 | -0.069 | -0.200 |
| W15 | cells | feed B days 5-7 + top-up flag (in place of late feed-B factor) | 6.550 | 6.363 | -0.187 | -3.2 | 34/49 | -0.254 | -0.117 | -0.205 |
| W16 | cells | feed B days 5-8 + top-up flag (in place of late feed-B factor) | 6.550 | 6.435 | -0.115 | -2.2 | 31/49 | -0.191 | -0.036 | -0.144 |
| W17 | cells | feed B days 6-6 + top-up flag (in place of late feed-B factor) | 6.550 | 6.417 | -0.132 | -1.8 | 28/49 | -0.110 | -0.156 | -0.199 |
| W18 | cells | feed B days 6-7 + top-up flag (in place of late feed-B factor) | 6.550 | 6.351 | -0.198 | -3.3 | 37/49 | -0.201 | -0.196 | -0.207 |
| W19 | cells | feed B days 6-8 + top-up flag (in place of late feed-B factor) | 6.550 | 6.451 | -0.099 | -1.9 | 29/49 | -0.130 | -0.066 | -0.133 |
| W20 | cells | feed B days 6-9 + top-up flag (in place of late feed-B factor) | 6.550 | 6.423 | -0.126 | -2.4 | 36/49 | -0.182 | -0.068 | -0.139 |
| W21 | cells | feed B days 7-7 + top-up flag (in place of late feed-B factor) | 6.550 | 6.429 | -0.121 | -2.1 | 35/49 | -0.062 | -0.181 | -0.130 |
| W22 | cells | feed B days 7-8 + top-up flag (in place of late feed-B factor) | 6.550 | 6.486 | -0.063 | -1.4 | 28/49 | -0.042 | -0.086 | -0.102 |
| W23 | cells | feed B days 7-9 + top-up flag (in place of late feed-B factor) | 6.550 | 6.471 | -0.079 | -1.6 | 28/49 | -0.087 | -0.071 | -0.094 |
| W24 | cells | feed B days 7-10 + top-up flag (in place of late feed-B factor) | 6.550 | 6.460 | -0.090 | -1.8 | 28/49 | -0.089 | -0.091 | -0.094 |
| W25 | cells | feed B days 8-8 + top-up flag (in place of late feed-B factor) | 6.550 | 6.541 | -0.009 | -0.2 | 22/49 | +0.065 | -0.086 | -0.084 |
| W26 | cells | feed B days 8-9 + top-up flag (in place of late feed-B factor) | 6.550 | 6.516 | -0.034 | -0.7 | 26/49 | -0.040 | -0.026 | -0.078 |
| W27 | cells | feed B days 8-10 + top-up flag (in place of late feed-B factor) | 6.550 | 6.495 | -0.055 | -1.1 | 26/49 | -0.056 | -0.054 | -0.081 |
| W28 | cells | feed B days 8-11 + top-up flag (in place of late feed-B factor) | 6.550 | 6.501 | -0.049 | -1.0 | 27/49 | -0.015 | -0.085 | -0.073 |
| W29 | cells | feed B days 9-9 + top-up flag (in place of late feed-B factor) | 6.550 | 6.500 | -0.050 | -0.8 | 28/49 | -0.058 | -0.041 | -0.056 |
| W30 | cells | feed B days 9-10 + top-up flag (in place of late feed-B factor) | 6.550 | 6.465 | -0.085 | -1.4 | 28/49 | -0.065 | -0.105 | -0.071 |
| W31 | cells | feed B days 9-11 + top-up flag (in place of late feed-B factor) | 6.550 | 6.482 | -0.068 | -1.3 | 27/49 | -0.010 | -0.128 | -0.056 |
| W32 | cells | feed B days 9-12 + top-up flag (in place of late feed-B factor) | 6.550 | 6.479 | -0.071 | -1.4 | 28/49 | -0.004 | -0.140 | -0.063 |
| W33 | cells | feed B days 10-10 + top-up flag (in place of late feed-B factor) | 6.550 | 6.493 | -0.057 | -1.0 | 25/49 | +0.014 | -0.131 | -0.097 |
| W34 | cells | feed B days 10-11 + top-up flag (in place of late feed-B factor) | 6.550 | 6.509 | -0.041 | -0.7 | 22/49 | +0.040 | -0.125 | -0.079 |
| W35 | cells | feed B days 10-12 + top-up flag (in place of late feed-B factor) | 6.550 | 6.507 | -0.042 | -0.7 | 24/49 | +0.030 | -0.118 | -0.077 |
| W36 | cells | feed B days 11-11 + top-up flag (in place of late feed-B factor) | 6.550 | 6.507 | -0.042 | -0.7 | 24/49 | +0.023 | -0.111 | -0.097 |
| W37 | cells | feed B days 11-12 + top-up flag (in place of late feed-B factor) | 6.550 | 6.506 | -0.044 | -0.7 | 24/49 | +0.023 | -0.114 | -0.088 |
| W38 | cells | feed B days 6-7 WITHOUT the top-up flag | 6.550 | 6.491 | -0.059 | -1.5 | 33/49 | -0.096 | -0.020 | -0.089 |
| W39 | cells | feed B days 5-8 WITHOUT the top-up flag | 6.550 | 6.543 | -0.006 | -0.2 | 30/49 | -0.090 | +0.081 | -0.045 |
| W41 | cells | feed B single day 3 + top-up flag | 6.550 | 6.469 | -0.080 | -1.1 | 29/49 | -0.094 | -0.066 | -0.185 |
| W42 | cells | feed B single day 4 + top-up flag | 6.550 | 6.514 | -0.035 | -0.6 | 29/49 | +0.026 | -0.100 | -0.088 |
| W43 | cells | feed B single day 5 + top-up flag | 6.550 | 6.449 | -0.101 | -1.9 | 30/49 | -0.183 | -0.015 | -0.142 |
| W44 | cells | feed B single day 6 + top-up flag | 6.550 | 6.417 | -0.132 | -1.8 | 28/49 | -0.110 | -0.156 | -0.199 |
| W45 | cells | feed B single day 7 + top-up flag | 6.550 | 6.429 | -0.121 | -2.1 | 35/49 | -0.062 | -0.181 | -0.130 |
| W46 | cells | feed B single day 8 + top-up flag | 6.550 | 6.541 | -0.009 | -0.2 | 22/49 | +0.065 | -0.086 | -0.084 |
| W47 | cells | feed B single day 9 + top-up flag | 6.550 | 6.500 | -0.050 | -0.8 | 28/49 | -0.058 | -0.041 | -0.056 |
| W48 | cells | feed B single day 10 + top-up flag | 6.550 | 6.493 | -0.057 | -1.0 | 25/49 | +0.014 | -0.131 | -0.097 |
| W49 | cells | feed B single day 11 + top-up flag | 6.550 | 6.507 | -0.042 | -0.7 | 24/49 | +0.023 | -0.111 | -0.097 |
| W50 | cells | late factor (v4) + days 6-7 + top-up flag | 6.550 | 6.353 | -0.197 | -3.0 | 32/49 | -0.192 | -0.203 | -0.213 |
| W51 | cells | days 6-7 + top-up flag, ridge 0.01 | 6.550 | 6.343 | -0.206 | -3.2 | 36/49 | -0.214 | -0.198 | -0.209 |
| W52 | cells | days 6-7 + top-up flag, ridge 0.1 | 6.550 | 6.382 | -0.167 | -3.2 | 35/49 | -0.153 | -0.182 | -0.189 |

### Round 4: the new cell model from several sides

| id | target | change | before | after | diff | t | better in | half A | half B | split 2 |
|---|---|---|---|---|---|---|---|---|---|---|
| N01 | cells | NEW small model: day-0 VCD, day-0 byproduct, initial volume, feed B days 5-7, top-up flag | 6.550 | 6.363 | -0.187 | -3.2 | 34/49 | -0.254 | -0.117 | -0.205 |
| N02 | cells | + early volume gain d2-3 | 6.550 | 6.362 | -0.187 | -2.8 | 35/49 | -0.253 | -0.118 | -0.232 |
| N03 | cells | + early volume gain d2-5 | 6.550 | 6.369 | -0.180 | -2.9 | 32/49 | -0.226 | -0.132 | -0.216 |
| N04 | cells | + feed-C pulse count | 6.550 | 6.380 | -0.169 | -3.1 | 31/49 | -0.219 | -0.117 | -0.191 |
| N05 | cells | + day-11 flag | 6.550 | 6.374 | -0.175 | -2.9 | 31/49 | -0.261 | -0.086 | -0.197 |
| N06 | cells | + late feed-B factor as well | 6.550 | 6.332 | -0.218 | -3.4 | 33/49 | -0.298 | -0.135 | -0.252 |
| N07 | cells | + day-0 substrate | 6.550 | 6.410 | -0.139 | -2.1 | 33/49 | -0.249 | -0.025 | -0.224 |
| N08 | cells | + sample-time offset | 6.550 | 6.336 | -0.213 | -2.4 | 33/49 | -0.121 | -0.309 | -0.186 |
| N09 | cells | without day-0 byproduct | 6.550 | 6.629 | +0.080 | 0.8 | 28/49 | +0.126 | +0.031 | +0.016 |
| N10 | cells | without initial volume | 6.550 | 6.413 | -0.137 | -2.0 | 29/49 | -0.162 | -0.112 | -0.165 |
| N11 | cells | without day-0 VCD | 6.550 | 6.423 | -0.127 | -1.3 | 28/49 | -0.324 | +0.079 | -0.155 |
| N12 | cells | ridge 0.01 | 6.550 | 6.354 | -0.196 | -3.2 | 35/49 | -0.270 | -0.117 | -0.211 |
| N13 | cells | ridge 0.1 | 6.550 | 6.395 | -0.154 | -3.2 | 35/49 | -0.197 | -0.110 | -0.182 |
| N14 | cells | ridge 0.3 | 6.550 | 6.476 | -0.073 | -1.6 | 31/49 | -0.077 | -0.070 | -0.114 |
| N15 | cells | small-model weight 0.4 | 6.550 | 6.397 | -0.152 | -3.0 | 33/49 | -0.175 | -0.128 | -0.167 |
| N16 | cells | small-model weight 0.6 | 6.550 | 6.339 | -0.211 | -3.1 | 35/49 | -0.323 | -0.094 | -0.234 |
| N17 | cells | small-model weight 0.7 | 6.550 | 6.326 | -0.224 | -2.8 | 34/49 | -0.377 | -0.064 | -0.254 |
| N18 | cells | small-model weight 0.8 | 6.550 | 6.323 | -0.227 | -2.4 | 30/49 | -0.415 | -0.031 | -0.267 |
| N19 | cells | small model alone | 6.550 | 6.338 | -0.212 | -1.8 | 29/49 | -0.463 | +0.050 | -0.265 |
| N20 | cells | per-day loadings (shrink 0.5) | 6.550 | 6.377 | -0.173 | -2.9 | 35/49 | -0.232 | -0.111 | -0.181 |
| N21 | cells | NEW + d2-3 volume gain, weight 0.6 | 6.550 | 6.340 | -0.209 | -2.8 | 33/49 | -0.316 | -0.099 | -0.261 |
| N22 | cells | NEW + d2-3 volume gain, weight 0.7 | 6.550 | 6.332 | -0.218 | -2.5 | 31/49 | -0.359 | -0.071 | -0.280 |
| N23 | cells | window days 6-7 instead of 5-7 | 6.550 | 6.351 | -0.198 | -3.3 | 37/49 | -0.201 | -0.196 | -0.207 |
| N24 | cells | window days 4-7 instead of 5-7 | 6.550 | 6.406 | -0.144 | -2.8 | 33/49 | -0.179 | -0.107 | -0.176 |
| N25 | cells | PLACEBO: feed-B window and top-up flag shuffled among batches (#1) | 6.550 | 6.367 | -0.183 | -2.5 | 31/49 | -0.189 | -0.176 | -0.240 |
| N26 | cells | PLACEBO: feed-B window and top-up flag shuffled among batches (#2) | 6.550 | 6.613 | +0.064 | 0.8 | 27/49 | -0.011 | +0.141 | +0.050 |
| N27 | cells | PLACEBO: feed-B window and top-up flag shuffled among batches (#3) | 6.550 | 6.577 | +0.027 | 0.4 | 24/49 | +0.074 | -0.021 | -0.054 |
| N28 | cells | PLACEBO: feed-B window and top-up flag shuffled among batches (#4) | 6.550 | 6.612 | +0.062 | 1.0 | 19/49 | +0.126 | -0.005 | -0.001 |
| N29 | cells | PLACEBO: feed-B window and top-up flag shuffled among batches (#5) | 6.550 | 6.665 | +0.116 | 1.8 | 20/49 | +0.134 | +0.096 | +0.013 |
| N30 | cells | PLACEBO: feed-B window and top-up flag shuffled among batches (#6) | 6.550 | 6.611 | +0.062 | 1.2 | 21/49 | +0.115 | +0.006 | -0.024 |
| N31 | cells | PLACEBO: feed-B window and top-up flag shuffled among batches (#7) | 6.550 | 6.607 | +0.057 | 1.2 | 23/49 | +0.071 | +0.042 | -0.017 |
| N32 | cells | PLACEBO: feed-B window and top-up flag shuffled among batches (#8) | 6.550 | 6.642 | +0.092 | 1.6 | 21/49 | +0.154 | +0.028 | +0.069 |

