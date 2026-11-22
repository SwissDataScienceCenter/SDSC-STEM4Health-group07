# Experiment log - Decode the Bioreactor

Updated 2026-11-21 16:06. **140 approaches evaluated.** Score = mean MAPE (%) over the 4 targets, out-of-fold, on the same 7 batch-wise folds of the 49 labelled batches (every model refitted per fold). Leaderboard reference: #1 = 7.96, Team 18 = 8.03. Nothing here was tuned against the leaderboard; no submission has been made.

**Current best: `b09_hyb_stat_bayesresid_smoothcells` - CV mean 7.028** (vcd 6.65, tcd 6.61, substrate 6.83, byproduct 8.02). Its test predictions are in `submission_best.csv`.

- `submission_nosearch.csv` = `b10_nosearch_bayesresid_smoothcells` (same recipe, no grid search anywhere).
- `submission_mechanistic_blend.csv` = `b11_mechanistic_only_bayesresid` (ODE-based members only).

Caveat: 49 batches give a CV standard error of roughly 0.3, and picking the minimum of many approaches is optimistic. Differences below ~0.2 are not meaningful; the top ~15 rows are statistically tied.

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


## Robustness check on a second, independent 7-fold partition

Everything was re-fitted from scratch on a different random split of the 49 batches. Numbers are ~0.1 higher than on the primary split, which is the size of the selection optimism to expect.

| approach | primary split | second split |
|---|---|---|
| `s01_mean` | 10.104 | 10.129 |
| `s25_search_all_plus_base25` | 7.472 | 7.549 |
| `s26_evidence_all_plus_base25` | 7.663 | 7.714 |
| `m02_cov7_cal` | 8.017 | 8.172 |
| `m21_re8_cal` | 7.890 | 7.964 |
| `h_m11_consHill_evidence_allb25` | 7.394 | 7.507 |
| `p_m21_re8_kernel_b25` | 7.652 | 7.767 |
| `p2_m21_re8_evidence_all_n1000` | 7.428 | 7.493 |
| `b04_hyb_stat_bayes_smoothcells` | 7.055 | 7.189 |
| `b06_nosearch_smoothcells` | 7.091 | 7.230 |
| `b09_hyb_stat_bayesresid_smoothcells` | 7.028 | 7.137 |
| `b10_nosearch_bayesresid_smoothcells` | 7.076 | 7.183 |
| `b11_mechanistic_only_bayesresid` | 7.213 | 7.322 |

## All approaches, best first

| # | approach | family | vcd | tcd | substrate | byproduct | mean | what it is |
|---|---|---|---|---|---|---|---|---|
| 1 | `b09_hyb_stat_bayesresid_smoothcells` | blend | 6.65 | 6.61 | 6.83 | 8.02 | **7.028** | geometric mean of hybrid, statistical (searched weights) and Bayesian particle ODE with residual kernel; cells smoothed |
| 2 | `b04_hyb_stat_bayes_smoothcells` | blend | 6.72 | 6.66 | 6.84 | 8.00 | **7.055** | geometric mean of hybrid (ODE + evidence-weighted residual kernel), statistical kernel model (searched weights) and Bayesian particle ODE; cell-density deviations smoothed in time |
| 3 | `b10_nosearch_bayesresid_smoothcells` | blend | 6.69 | 6.64 | 6.92 | 8.06 | **7.076** | as b09 with marginal-likelihood weights in the statistical member (no grid search anywhere) |
| 4 | `b08_six_member_smoothcells` | blend | 6.73 | 6.67 | 6.92 | 8.00 | **7.079** | geometric mean of 2 hybrids, 2 statistical and 2 Bayesian particle models; cells smoothed |
| 5 | `b03_hyb_stat_bayes` | blend | 6.77 | 6.73 | 6.84 | 8.00 | **7.085** | geometric mean of hybrid (ODE + evidence-weighted residual kernel), statistical kernel model (searched weights) and Bayesian particle ODE |
| 6 | `b06_nosearch_smoothcells` | blend | 6.75 | 6.70 | 6.92 | 8.00 | **7.091** | as b04 but the statistical member uses marginal-likelihood weights: no grid search anywhere |
| 7 | `b02_hyb_stat_bayes` | blend | 6.78 | 6.74 | 6.92 | 8.12 | **7.140** | geometric mean of h_m11_consHill_evidence, s09_kernel_search_allgroups, p_m21_re8_kernel |
| 8 | `b01_hyb_stat_bayes` | blend | 6.73 | 6.69 | 6.99 | 8.36 | **7.193** | geometric mean of h_m02_cov7_search, s09_kernel_search_allgroups, p_m01_base_kernel |
| 9 | `b11_mechanistic_only_bayesresid` | blend | 6.71 | 6.67 | 7.20 | 8.28 | **7.213** | geometric mean of the two ODE-based models (hybrid + Bayesian particle with residual kernel); cells smoothed |
| 10 | `b07_mechanistic_only_smoothcells` | blend | 6.81 | 6.77 | 7.26 | 8.34 | **7.291** | geometric mean of the two ODE-based models only (hybrid + Bayesian particle), cells smoothed |
| 11 | `p2_m21_re8_evidence_all` | hybrid | 7.12 | 6.98 | 7.06 | 8.39 | **7.388** | Bayesian particle ODE m21_re8 (random effects from signals, feed-rule likelihood) + kernel model (evidence weights) on its centred log residuals |
| 12 | `h_m11_consHill_evidence_allb25` | hybrid | 6.83 | 6.77 | 7.64 | 8.34 | **7.394** | ODE m11_consHill + kernel model (evidence weights) on its log residuals, shift + rule truncation |
| 13 | `h_m11_consHill_evidence_allg` | hybrid | 6.83 | 6.77 | 7.67 | 8.34 | **7.402** | ODE m11_consHill + kernel model (evidence_allg weights) on its log residuals, shift + rule truncation |
| 14 | `h_m11_consHill_evidence` | hybrid | 6.81 | 6.75 | 7.74 | 8.33 | **7.408** | ODE m11_consHill + kernel model (evidence weights) on its log residuals, shift + rule truncation |
| 15 | `h_m11_consHill_evsplit` | hybrid | 6.75 | 6.69 | 7.85 | 8.38 | **7.418** | ODE m11_consHill + kernel model (evsplit weights) on its log residuals, shift + rule truncation |
| 16 | `h_m03_feedD_evidence_allg` | hybrid | 6.83 | 6.77 | 7.59 | 8.48 | **7.420** | ODE m03_feedD + kernel model (evidence_allg weights) on its log residuals, shift + rule truncation |
| 17 | `h_m11_consHill_search_allb25` | hybrid | 6.76 | 6.72 | 7.88 | 8.33 | **7.422** | ODE m11_consHill + kernel model (search weights) on its log residuals, shift + rule truncation |
| 18 | `h_m03_feedD_evidence` | hybrid | 6.82 | 6.75 | 7.68 | 8.44 | **7.422** | ODE m03_feedD + kernel model (evidence weights) on its log residuals, shift + rule truncation |
| 19 | `h_m05_logistic_evidence` | hybrid | 6.87 | 6.82 | 7.59 | 8.42 | **7.425** | ODE m05_logistic + kernel model (evidence weights) on its log residuals, shift + rule truncation |
| 20 | `p2_m21_re8_evidence_all_n1000` | hybrid | 7.06 | 7.07 | 7.19 | 8.39 | **7.428** | Bayesian particle ODE m21_re8 (random effects from signals, feed-rule likelihood) + kernel model (evidence weights) on its centred log residuals |
| 21 | `h_m03_feedD_evsplit` | hybrid | 6.75 | 6.69 | 7.80 | 8.47 | **7.428** | ODE m03_feedD + kernel model (evsplit weights) on its log residuals, shift + rule truncation |
| 22 | `h_m02_cov7_evidence_allb25` | hybrid | 6.83 | 6.77 | 7.68 | 8.45 | **7.433** | ODE m02_cov7 + kernel model (evidence weights) on its log residuals, shift + rule truncation |
| 23 | `h_m02_cov7_search` | hybrid | 6.70 | 6.66 | 8.03 | 8.36 | **7.437** | ODE m02_cov7 + kernel model (search weights) on its log residuals, shift + rule truncation |
| 24 | `h_m02_cov7_evidence` | hybrid | 6.81 | 6.75 | 7.77 | 8.44 | **7.441** | ODE m02_cov7 + kernel model (evidence weights) on its log residuals, shift + rule truncation |
| 25 | `h_m11_consHill_evidence_b25` | hybrid | 6.81 | 6.74 | 7.71 | 8.51 | **7.441** | ODE m11_consHill + kernel model (evidence weights) on its log residuals, shift + rule truncation |
| 26 | `h_m11_consHill_search` | hybrid | 6.72 | 6.67 | 8.01 | 8.37 | **7.441** | ODE m11_consHill + kernel model (search weights) on its log residuals, shift + rule truncation |
| 27 | `h_m11_consHill_search_b25` | hybrid | 6.72 | 6.66 | 7.93 | 8.48 | **7.444** | ODE m11_consHill + kernel model (search weights) on its log residuals, shift + rule truncation |
| 28 | `h_m02_cov7_search_allb25` | hybrid | 6.78 | 6.74 | 7.96 | 8.29 | **7.444** | ODE m02_cov7 + kernel model (search weights) on its log residuals, shift + rule truncation |
| 29 | `p2_m21_re8_evidence_cb` | hybrid | 7.12 | 6.98 | 7.31 | 8.39 | **7.450** | Bayesian particle ODE m21_re8 (random effects from signals, feed-rule likelihood) + kernel model (evidence weights) on its centred log residuals (cells and byproduct only) |
| 30 | `h_m02_cov7_search_b25` | hybrid | 6.71 | 6.66 | 7.92 | 8.50 | **7.450** | ODE m02_cov7 + kernel model (search weights) on its log residuals, shift + rule truncation |
| 31 | `h_m11_consHill_search_allg` | hybrid | 6.77 | 6.73 | 7.88 | 8.42 | **7.452** | ODE m11_consHill + kernel model (search_allg weights) on its log residuals, shift + rule truncation |
| 32 | `h_m03_feedD_search_allg` | hybrid | 6.80 | 6.76 | 7.76 | 8.49 | **7.453** | ODE m03_feedD + kernel model (search_allg weights) on its log residuals, shift + rule truncation |
| 33 | `p2_m01_base_evidence` | hybrid | 7.00 | 6.99 | 7.36 | 8.47 | **7.454** | Bayesian particle ODE m01_base (random effects from signals, feed-rule likelihood) + kernel model (evidence weights) on its centred log residuals |
| 34 | `h_m02_cov7_evidence_b25` | hybrid | 6.80 | 6.73 | 7.75 | 8.54 | **7.456** | ODE m02_cov7 + kernel model (evidence weights) on its log residuals, shift + rule truncation |
| 35 | `h_m05_logistic_search` | hybrid | 6.83 | 6.79 | 7.85 | 8.37 | **7.459** | ODE m05_logistic + kernel model (search weights) on its log residuals, shift + rule truncation |
| 36 | `h_m03_feedD_search` | hybrid | 6.73 | 6.68 | 7.98 | 8.47 | **7.467** | ODE m03_feedD + kernel model (search weights) on its log residuals, shift + rule truncation |
| 37 | `s25_search_all_plus_base25` | statistical | 7.13 | 7.09 | 7.40 | 8.27 | **7.472** | s09 plus windowed base-addition signal |
| 38 | `p2_m02_cov7_evidence_cb` | hybrid | 6.96 | 6.83 | 7.67 | 8.46 | **7.481** | Bayesian particle ODE m02_cov7 (random effects from signals, feed-rule likelihood) + kernel model (evidence weights) on its centred log residuals (cells and byproduct only) |
| 39 | `h_m14_re6_evidence` | hybrid | 6.90 | 6.87 | 7.75 | 8.48 | **7.499** | ODE m14_re6 + kernel model (evidence weights) on its log residuals, shift + rule truncation |
| 40 | `p2_m01_base_evidence_cb` | hybrid | 7.00 | 6.99 | 7.67 | 8.34 | **7.499** | Bayesian particle ODE m01_base (random effects from signals, feed-rule likelihood) + kernel model (evidence weights) on its centred log residuals (cells and byproduct only) |
| 41 | `h_m21_re8_evidence` | hybrid | 6.93 | 6.90 | 7.77 | 8.46 | **7.513** | ODE m21_re8 + kernel model (evidence weights) on its log residuals, shift + rule truncation |
| 42 | `h_m14_re6_search` | hybrid | 6.92 | 6.92 | 7.93 | 8.49 | **7.564** | ODE m14_re6 + kernel model (search weights) on its log residuals, shift + rule truncation |
| 43 | `hf_m11_consHill` | hybrid | 7.21 | 7.16 | 7.72 | 8.19 | **7.568** | kernel model (searched weights) with ODE m11_consHill simulated trajectories as an extra feature group |
| 44 | `hf_m01_base_allg` | hybrid | 7.13 | 7.10 | 7.40 | 8.68 | **7.580** | kernel model (searched weights) with ODE m01_base simulated trajectories as an extra feature group |
| 45 | `s22_search_base25` | statistical | 7.31 | 7.25 | 7.60 | 8.18 | **7.584** | s08 with the base-addition signal as one windowed value (days 2-5) instead of 4 daily values |
| 46 | `h_m08_tvall_evidence` | hybrid | 6.91 | 6.84 | 8.13 | 8.49 | **7.593** | ODE m08_tvall + kernel model (evidence weights) on its log residuals, shift + rule truncation |
| 47 | `s09_kernel_search_allgroups` | statistical | 7.13 | 7.10 | 7.44 | 8.71 | **7.596** | s08 with all 19 feature groups in the search |
| 48 | `s24_search_core_plus_base25` | statistical | 7.31 | 7.27 | 7.64 | 8.17 | **7.597** | s08 plus windowed base-addition signal |
| 49 | `h_m21_re8_search` | hybrid | 6.99 | 6.98 | 7.89 | 8.55 | **7.604** | ODE m21_re8 + kernel model (search weights) on its log residuals, shift + rule truncation |
| 50 | `h_m08_tvall_search` | hybrid | 6.83 | 6.79 | 8.33 | 8.50 | **7.612** | ODE m08_tvall + kernel model (search weights) on its log residuals, shift + rule truncation |
| 51 | `p_m21_re8_kernel` | mechanistic-bayes | 7.20 | 7.20 | 7.42 | 8.66 | **7.624** | ODE m21_re8 with per-batch random effects (growth, uptake, overflow yield, re-consumption, inoculum) predicted from process signals (kernel); 400 particles weighted by substrate feed-rule likelihood; calibration=True, flags=True, kappa=1.0 |
| 52 | `hf_m01_base` | hybrid | 7.31 | 7.27 | 7.61 | 8.40 | **7.647** | kernel model (searched weights) with ODE m01_base simulated trajectories as an extra feature group |
| 53 | `p_m21_re8_kernel_b25` | mechanistic-bayes | 7.26 | 7.26 | 7.39 | 8.70 | **7.652** | ODE m21_re8 with per-batch random effects (growth, uptake, overflow yield, re-consumption, inoculum) predicted from process signals (kernel); 400 particles weighted by substrate feed-rule likelihood; calibration=True, flags=True, kappa=1.0 |
| 54 | `p_m02_cov7_kernel` | mechanistic-bayes | 7.15 | 7.11 | 7.73 | 8.65 | **7.660** | ODE m02_cov7 with per-batch random effects (growth, uptake, overflow yield, re-consumption, inoculum) predicted from process signals (kernel); 400 particles weighted by substrate feed-rule likelihood; calibration=True, flags=True, kappa=1.0 |
| 55 | `s26_evidence_all_plus_base25` | statistical | 7.27 | 7.26 | 7.73 | 8.39 | **7.663** | s13 plus windowed base-addition signal |
| 56 | `s08_kernel_search` | statistical | 7.31 | 7.26 | 7.69 | 8.39 | **7.664** | s06 + shift + rule truncation (= solution.py) |
| 57 | `p_m02_cov7_kernel_b25` | mechanistic-bayes | 7.17 | 7.13 | 7.69 | 8.69 | **7.668** | ODE m02_cov7 with per-batch random effects (growth, uptake, overflow yield, re-consumption, inoculum) predicted from process signals (kernel); 400 particles weighted by substrate feed-rule likelihood; calibration=True, flags=True, kappa=1.0 |
| 58 | `s23_evidence_base25` | statistical | 7.38 | 7.34 | 7.77 | 8.24 | **7.682** | s11 with windowed base-addition signal |
| 59 | `h_m20_re6_cov10_evidence` | hybrid | 7.12 | 7.11 | 7.66 | 8.84 | **7.684** | ODE m20_re6_cov10 + kernel model (evidence weights) on its log residuals, shift + rule truncation |
| 60 | `h_m20_re6_cov10_search` | hybrid | 7.16 | 7.18 | 7.70 | 8.72 | **7.689** | ODE m20_re6_cov10 + kernel model (search weights) on its log residuals, shift + rule truncation |
| 61 | `h_m02_cov7_equal` | hybrid | 6.90 | 6.86 | 8.26 | 8.79 | **7.705** | ODE m02_cov7 + kernel model (equal weights) on its log residuals, shift + rule truncation |
| 62 | `s07_kernel_search_shift` | statistical | 7.31 | 7.26 | 7.93 | 8.39 | **7.725** | s06 + lognormal shift |
| 63 | `pr_m01_base_evidence` | hybrid | 6.83 | 6.77 | 8.38 | 8.99 | **7.740** | ODE m01_base with per-batch kinetic parameters predicted from process signals (two-stage), then kernel model (evidence) on log residuals, shift + rule truncation |
| 64 | `s12_evidence_split` | statistical | 7.51 | 7.47 | 7.69 | 8.37 | **7.758** | s11 with early/late split blocks |
| 65 | `s10_kernel_search_pertarget` | statistical | 7.25 | 7.21 | 7.91 | 8.69 | **7.766** | s08 with separate weights per target (4 blocks) |
| 66 | `s06_kernel_search_raw` | statistical | 7.36 | 7.30 | 7.94 | 8.49 | **7.773** | grouped kernel ridge, LOO weight search, no output step |
| 67 | `pr_m01_base_search` | hybrid | 6.83 | 6.76 | 8.53 | 8.98 | **7.774** | ODE m01_base with per-batch kinetic parameters predicted from process signals (two-stage), then kernel model (search) on log residuals, shift + rule truncation |
| 68 | `s05_kernel_equal` | statistical | 6.98 | 6.93 | 8.06 | 9.23 | **7.799** | grouped kernel ridge, equal group weights, one scale per block |
| 69 | `s13_evidence_all` | statistical | 7.27 | 7.26 | 7.84 | 8.84 | **7.802** | s11 with all 19 groups |
| 70 | `s11_evidence` | statistical | 7.40 | 7.37 | 7.86 | 8.65 | **7.820** | GP empirical Bayes: group variances by marginal likelihood, 3 blocks |
| 71 | `s21_evidence_target` | statistical | 7.39 | 7.37 | 7.86 | 8.65 | **7.820** | s11 with one block per target |
| 72 | `p2_m21_re8_search` | hybrid | 7.12 | 7.04 | 8.68 | 8.47 | **7.828** | Bayesian particle ODE m21_re8 (random effects from signals, feed-rule likelihood) + kernel model (search weights) on its centred log residuals |
| 73 | `h_m01_base_search` | hybrid | 7.21 | 7.16 | 8.13 | 8.94 | **7.860** | ODE m01_base + kernel model (search weights) on its log residuals, shift + rule truncation |
| 74 | `p_m20_re6_cov10_kernel` | mechanistic-bayes | 7.59 | 7.54 | 7.12 | 9.23 | **7.869** | ODE m20_re6_cov10 with per-batch random effects (growth, uptake, overflow yield, re-consumption, inoculum) predicted from process signals (kernel); 400 particles weighted by substrate feed-rule likelihood; calibration=True, flags=True, kappa=1.0 |
| 75 | `p_m01_base_kernel` | mechanistic-bayes | 7.08 | 7.06 | 7.80 | 9.55 | **7.872** | ODE m01_base with per-batch random effects (growth, uptake, overflow yield, re-consumption, inoculum) predicted from process signals (kernel); 400 particles weighted by substrate feed-rule likelihood; calibration=True, flags=True, kappa=1.0 |
| 76 | `m19_combo_cal` | mechanistic | 7.35 | 7.34 | 7.91 | 8.91 | **7.877** | ODE with feed-D-promoted and switch-like byproduct re-consumption, 6 kinetic parameters scaled by 7 process signals | + per-day calibration + rule truncation |
| 77 | `p_m01_base_kernel_k07` | mechanistic-bayes | 7.07 | 7.02 | 7.95 | 9.50 | **7.885** | ODE m01_base with per-batch random effects (growth, uptake, overflow yield, re-consumption, inoculum) predicted from process signals (kernel); 400 particles weighted by substrate feed-rule likelihood; calibration=True, flags=True, kappa=0.7 |
| 78 | `m21_re8_cal` | mechanistic | 7.36 | 7.35 | 8.03 | 8.81 | **7.890** | m02 with covariates on 8 parameters (adds death rate and substrate feed concentration) | + per-day calibration + rule truncation |
| 79 | `m14_re6_cal` | mechanistic | 7.35 | 7.34 | 8.07 | 8.81 | **7.891** | m02 with covariates also on overflow flux and growth-inhibition constant | + per-day calibration + rule truncation |
| 80 | `p_m02_cov7_z7` | mechanistic-bayes | 7.58 | 7.55 | 7.69 | 8.86 | **7.920** | ODE m02_cov7 with per-batch random effects (growth, uptake, overflow yield, re-consumption, inoculum) predicted from process signals (z7); 400 particles weighted by substrate feed-rule likelihood; calibration=True, flags=True, kappa=1.0 |
| 81 | `p_m02_cov7_none` | mechanistic-bayes | 7.55 | 7.50 | 7.70 | 8.93 | **7.920** | ODE m02_cov7 with per-batch random effects (growth, uptake, overflow yield, re-consumption, inoculum) predicted from process signals (none); 400 particles weighted by substrate feed-rule likelihood; calibration=True, flags=True, kappa=1.0 |
| 82 | `m20_re6_cov10_cal` | mechanistic | 7.55 | 7.53 | 7.68 | 9.08 | **7.961** | m14 with 10 process signals | + per-day calibration + rule truncation |
| 83 | `m18_lam100_cal` | mechanistic | 7.48 | 7.45 | 8.25 | 8.71 | **7.974** | m02 with strong penalty on the covariate table | + per-day calibration + rule truncation |
| 84 | `h_m01_base_equal` | hybrid | 6.96 | 6.90 | 8.41 | 9.62 | **7.974** | ODE m01_base + kernel model (equal weights) on its log residuals, shift + rule truncation |
| 85 | `m15_l2_cal` | mechanistic | 7.52 | 7.48 | 8.16 | 8.74 | **7.976** | m02 fitted with plain least squares on log errors | + per-day calibration + rule truncation |
| 86 | `m11_consHill_cal` | mechanistic | 7.49 | 7.46 | 8.22 | 8.81 | **7.994** | m02 with switch-like byproduct re-consumption (Hill in substrate, Monod in byproduct) | + per-day calibration + rule truncation |
| 87 | `m03_feedD_cal` | mechanistic | 7.49 | 7.44 | 8.14 | 8.96 | **8.009** | m02 + feed D promotes byproduct re-consumption (decaying D state) | + per-day calibration + rule truncation |
| 88 | `m05_logistic_cal` | mechanistic | 7.66 | 7.66 | 7.99 | 8.76 | **8.014** | m02 with logistic capacity (1-Xv/Xmax) replacing IVCD inhibition | + per-day calibration + rule truncation |
| 89 | `m02_cov7_cal` | mechanistic | 7.51 | 7.48 | 8.30 | 8.78 | **8.017** | m01 + 4 kinetic parameters scaled by 7 process signals | + per-day calibration + rule truncation |
| 90 | `m16_growthcoupled_cal` | mechanistic | 7.51 | 7.48 | 8.30 | 8.78 | **8.018** | m02 with growth-coupled uptake qS = mu/Yxs + maintenance + overflow | + per-day calibration + rule truncation |
| 91 | `m10_ovHill_cal` | mechanistic | 7.51 | 7.48 | 8.30 | 8.79 | **8.018** | m02 with overflow flux driven by a Hill function of substrate as well as culture age | + per-day calibration + rule truncation |
| 92 | `m06_nutrient_cal` | mechanistic | 7.60 | 7.61 | 8.15 | 8.74 | **8.024** | m02 + unmeasured nutrient supplied by feed B limits growth | + per-day calibration + rule truncation |
| 93 | `h_m01_base_evidence` | hybrid | 7.35 | 7.31 | 8.37 | 9.08 | **8.028** | ODE m01_base + kernel model (evidence weights) on its log residuals, shift + rule truncation |
| 94 | `m09_death_cal` | mechanistic | 7.51 | 7.48 | 8.32 | 8.81 | **8.029** | m02 + death accelerated by low substrate and by byproduct | + per-day calibration + rule truncation |
| 95 | `m13_cov4_cal` | mechanistic | 7.47 | 7.43 | 8.26 | 8.99 | **8.037** | m02 with 4 signals only (no substrate-rule flags as covariates) | + per-day calibration + rule truncation |
| 96 | `m17_lam1_cal` | mechanistic | 7.52 | 7.50 | 8.30 | 8.84 | **8.040** | m02 with weak penalty on the covariate table | + per-day calibration + rule truncation |
| 97 | `m12_cov10_cal` | mechanistic | 7.70 | 7.66 | 7.85 | 9.01 | **8.054** | m02 with 10 process signals (adds early/late feed-B dosing, early feed-C) | + per-day calibration + rule truncation |
| 98 | `s18_enet` | statistical | 7.55 | 7.52 | 8.18 | 9.02 | **8.065** | multi-task elastic net, penalty by inner CV |
| 99 | `p_m01_base_z7` | mechanistic-bayes | 7.58 | 7.55 | 7.96 | 9.20 | **8.074** | ODE m01_base with per-batch random effects (growth, uptake, overflow yield, re-consumption, inoculum) predicted from process signals (z7); 400 particles weighted by substrate feed-rule likelihood; calibration=True, flags=True, kappa=1.0 |
| 100 | `p_m01_base_kernel_k14` | mechanistic-bayes | 7.30 | 7.26 | 7.87 | 9.86 | **8.074** | ODE m01_base with per-batch random effects (growth, uptake, overflow yield, re-consumption, inoculum) predicted from process signals (kernel); 400 particles weighted by substrate feed-rule likelihood; calibration=True, flags=True, kappa=1.4 |
| 101 | `m07_tvmu_cal` | mechanistic | 7.53 | 7.49 | 8.69 | 8.59 | **8.075** | m02 + phase-dependent growth: piecewise-linear time multiplier on mu (knots every 2 d) | + per-day calibration + rule truncation |
| 102 | `m04_Pinhib_cal` | mechanistic | 7.60 | 7.59 | 8.26 | 8.87 | **8.078** | m02 + byproduct inhibits growth | + per-day calibration + rule truncation |
| 103 | `m08_tvall_cal` | mechanistic | 7.49 | 7.43 | 8.43 | 8.96 | **8.080** | m02 + phase-dependent growth, substrate uptake and byproduct re-consumption (piecewise-linear in time) | + per-day calibration + rule truncation |
| 104 | `s20_rrr` | statistical | 7.03 | 6.97 | 9.69 | 8.92 | **8.152** | kernel search projected to rank-3 (shared latent batch factors) |
| 105 | `s04_ridge_all` | statistical | 7.31 | 7.26 | 8.45 | 9.90 | **8.229** | ridge on all raw features, one penalty per target block |
| 106 | `s14_pls` | statistical | 7.17 | 7.12 | 9.28 | 9.50 | **8.267** | PLS regression, components by inner CV |
| 107 | `p_m01_base_kernel_noflags` | mechanistic-bayes | 7.04 | 7.00 | 9.78 | 9.62 | **8.359** | ODE m01_base with per-batch random effects (growth, uptake, overflow yield, re-consumption, inoculum) predicted from process signals (kernel); 400 particles weighted by substrate feed-rule likelihood; calibration=True, flags=False, kappa=1.0 |
| 108 | `p2_m21_re8_evidence` | hybrid | 7.12 | 6.98 | 11.23 | 8.46 | **8.449** | Bayesian particle ODE m21_re8 (random effects from signals, feed-rule likelihood) + kernel model (evidence weights) on its centred log residuals |
| 109 | `p_m01_base_kernel_nocal` | mechanistic-bayes | 7.54 | 7.50 | 8.31 | 10.67 | **8.504** | ODE m01_base with per-batch random effects (growth, uptake, overflow yield, re-consumption, inoculum) predicted from process signals (kernel); 400 particles weighted by substrate feed-rule likelihood; calibration=False, flags=True, kappa=1.0 |
| 110 | `m19_combo_pure` | mechanistic | 7.69 | 7.69 | 8.81 | 10.08 | **8.567** | ODE with feed-D-promoted and switch-like byproduct re-consumption, 6 kinetic parameters scaled by 7 process signals | pure ODE output |
| 111 | `m08_tvall_pure` | mechanistic | 7.83 | 7.78 | 9.03 | 9.74 | **8.596** | m02 + phase-dependent growth, substrate uptake and byproduct re-consumption (piecewise-linear in time) | pure ODE output |
| 112 | `s16_rf` | statistical | 7.49 | 7.48 | 9.57 | 10.14 | **8.673** | random forest multi-output |
| 113 | `m03_feedD_pure` | mechanistic | 7.81 | 7.76 | 9.07 | 10.07 | **8.678** | m02 + feed D promotes byproduct re-consumption (decaying D state) | pure ODE output |
| 114 | `s15_extratrees` | statistical | 7.57 | 7.55 | 9.40 | 10.27 | **8.697** | ExtraTrees multi-output on core features |
| 115 | `p2_m02_cov7_evidence` | hybrid | 6.96 | 6.83 | 12.47 | 8.55 | **8.701** | Bayesian particle ODE m02_cov7 (random effects from signals, feed-rule likelihood) + kernel model (evidence weights) on its centred log residuals |
| 116 | `m11_consHill_pure` | mechanistic | 7.80 | 7.76 | 9.36 | 10.11 | **8.759** | m02 with switch-like byproduct re-consumption (Hill in substrate, Monod in byproduct) | pure ODE output |
| 117 | `m21_re8_pure` | mechanistic | 7.67 | 7.72 | 9.55 | 10.13 | **8.771** | m02 with covariates on 8 parameters (adds death rate and substrate feed concentration) | pure ODE output |
| 118 | `m14_re6_pure` | mechanistic | 7.68 | 7.73 | 9.56 | 10.14 | **8.776** | m02 with covariates also on overflow flux and growth-inhibition constant | pure ODE output |
| 119 | `p_m01_base_none` | mechanistic-bayes | 7.95 | 7.99 | 7.84 | 11.33 | **8.779** | ODE m01_base with per-batch random effects (growth, uptake, overflow yield, re-consumption, inoculum) predicted from process signals (none); 400 particles weighted by substrate feed-rule likelihood; calibration=True, flags=True, kappa=1.0 |
| 120 | `m20_re6_cov10_pure` | mechanistic | 7.81 | 7.85 | 9.13 | 10.36 | **8.787** | m14 with 10 process signals | pure ODE output |
| 121 | `m05_logistic_pure` | mechanistic | 7.87 | 7.86 | 9.39 | 10.03 | **8.789** | m02 with logistic capacity (1-Xv/Xmax) replacing IVCD inhibition | pure ODE output |
| 122 | `s17_knn` | statistical | 7.49 | 7.45 | 9.83 | 10.53 | **8.825** | k-nearest batches (k=10, distance weighted) on core features |
| 123 | `m06_nutrient_pure` | mechanistic | 7.99 | 7.95 | 9.32 | 10.07 | **8.834** | m02 + unmeasured nutrient supplied by feed B limits growth | pure ODE output |
| 124 | `m18_lam100_pure` | mechanistic | 7.79 | 7.78 | 9.75 | 10.05 | **8.844** | m02 with strong penalty on the covariate table | pure ODE output |
| 125 | `m15_l2_pure` | mechanistic | 7.81 | 7.81 | 9.68 | 10.10 | **8.850** | m02 fitted with plain least squares on log errors | pure ODE output |
| 126 | `m02_cov7_pure` | mechanistic | 7.82 | 7.80 | 9.75 | 10.13 | **8.877** | m01 + 4 kinetic parameters scaled by 7 process signals | pure ODE output |
| 127 | `m10_ovHill_pure` | mechanistic | 7.82 | 7.81 | 9.75 | 10.13 | **8.877** | m02 with overflow flux driven by a Hill function of substrate as well as culture age | pure ODE output |
| 128 | `m16_growthcoupled_pure` | mechanistic | 7.82 | 7.80 | 9.75 | 10.13 | **8.877** | m02 with growth-coupled uptake qS = mu/Yxs + maintenance + overflow | pure ODE output |
| 129 | `m12_cov10_pure` | mechanistic | 7.97 | 7.95 | 9.33 | 10.33 | **8.894** | m02 with 10 process signals (adds early/late feed-B dosing, early feed-C) | pure ODE output |
| 130 | `m17_lam1_pure` | mechanistic | 7.83 | 7.82 | 9.75 | 10.18 | **8.897** | m02 with weak penalty on the covariate table | pure ODE output |
| 131 | `m09_death_pure` | mechanistic | 7.83 | 7.81 | 9.82 | 10.15 | **8.902** | m02 + death accelerated by low substrate and by byproduct | pure ODE output |
| 132 | `m07_tvmu_pure` | mechanistic | 7.90 | 7.84 | 9.92 | 9.96 | **8.905** | m02 + phase-dependent growth: piecewise-linear time multiplier on mu (knots every 2 d) | pure ODE output |
| 133 | `m04_Pinhib_pure` | mechanistic | 8.11 | 8.03 | 9.34 | 10.18 | **8.912** | m02 + byproduct inhibits growth | pure ODE output |
| 134 | `m13_cov4_pure` | mechanistic | 7.75 | 7.71 | 10.20 | 10.16 | **8.955** | m02 with 4 signals only (no substrate-rule flags as covariates) | pure ODE output |
| 135 | `m01_base_cal` | mechanistic | 8.14 | 8.21 | 8.78 | 11.15 | **9.072** | ODE: growth slowed by IVCD, overflow byproduct, re-consumption; global parameters only | + per-day calibration + rule truncation |
| 136 | `s03_mean_trunc` | baseline | 8.65 | 8.70 | 9.96 | 11.46 | **9.694** | per-day mean + shift + substrate feed-rule truncation |
| 137 | `m01_base_pure` | mechanistic | 8.38 | 8.38 | 10.60 | 12.12 | **9.870** | ODE: growth slowed by IVCD, overflow byproduct, re-consumption; global parameters only | pure ODE output |
| 138 | `s02_mean_shift` | baseline | 8.65 | 8.70 | 11.28 | 11.46 | **10.024** | per-day mean + lognormal MAPE shift |
| 139 | `s01_mean` | baseline | 8.70 | 8.73 | 11.47 | 11.51 | **10.104** | per-day mean profile (log scale), no batch information |
| 140 | `s19_rbf` | statistical | 12.94 | 13.02 | 9.85 | 10.07 | **11.470** | RBF kernel ridge on core features |
