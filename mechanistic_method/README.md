# Bioreactor prediction project

## Start here

This folder contains the data, model code, saved predictions, and instructions needed to generate the challenge submissions.

The method is a **hybrid mechanistic and machine learning ensemble**. It combines a biological ODE simulator with a learned correction, a direct statistical kernel model, and a Bayesian particle ODE model with a learned correction. Their predictions are combined using an equal geometric mean, followed by smoothing of the cell-density deviations.

The challenge has 49 labelled batches for learning and 12 batches with missing measurements. Each test batch has 13 requested sampling points after the initial observation. Each output CSV has 156 rows and four predicted values per row, giving 624 predicted values.

The four outputs are VCD (living cell concentration), TCD (total cell concentration), substrate, and byproduct. The code is built for the supplied challenge format, including 14 observations per batch when the starting observation is counted.

## Choose how to run

| Your goal | Command | What happens |
|---|---|---|
| Generate the original submission files quickly | `python run.py` | Reads saved predictions and recorded scores, writes the CSV files, and checks them |
| Fit the original models again | `python run.py --retrain` | Fits models in a new run folder, computes predictions and local scores, then writes and checks the CSV files |
| Work through a notebook | Open `Run_Bioreactor.ipynb` | Guides you through setup, either run mode, output preview, and downloads |

Use the quick mode first. It recreates the supplied results for these exact input files. It does not train models or recalculate their validation scores. If the data files have changed, the launcher rejects cached export so that old predictions cannot be mistaken for predictions from new inputs.

Full retraining uses the original modelling procedure, including the validation limitations described below. It can take tens of minutes or longer, depending on the computer. The original research estimated 30 to 40 minutes on 10 cores, but its shell script was not timed from start to finish. This launcher runs the stages sequentially, so that estimate is not a runtime promise for this bundle.

## Computer setup

Use Python 3.11 or newer. Python 3.12 was used to verify the quick run. The model uses the CPU and does not require a GPU. Package installation needs internet access; generating the results uses the local files.

Extract the entire ZIP. Open a terminal inside the extracted `bioreactor_ready_to_run` folder. Keep the supplied names and folder structure.

### macOS or Linux

Create an isolated Python environment and run the quick mode:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run.py
```

To train the models again using that environment:

```bash
.venv/bin/python run.py --retrain
```

### Windows PowerShell

With Python 3.12 installed:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run.py
```

If you installed Python 3.11 instead, use `py -3.11` in the first command.

To train the models again:

```powershell
.\.venv\Scripts\python.exe run.py --retrain
```

### Google Colab or local Jupyter

Open `Run_Bioreactor.ipynb` and run its cells in order. In Colab, the notebook asks you to upload the project ZIP if the project is not already present. In local Jupyter, open the notebook from the extracted project folder.

Leave `MODE = "regenerate"` for the quick run. To train, change that single setting to `MODE = "retrain"`. The notebook installs the requirements, runs the same Python launcher, displays the checks and a preview, and provides the three CSV files. It does not submit anything to the leaderboard.

## What you receive after a successful run

Open the `outputs` folder.

| File | Purpose |
|---|---|
| `submission_best.csv` | Predictions from the lowest recorded error in the selected run |
| `submission_nosearch.csv` | Alternative that uses marginal likelihood for the direct statistical member's feature weights; see the search limitation below |
| `submission_mechanistic_blend.csv` | Blend of the two ODE based members, both with statistical corrections |
| `experiments_log.md` | Scores and descriptions from that run's result registry |
| `run_output.txt` | Execution output |
| `validation.json` | Checks on row counts, columns, times, missing values, and positive finite predictions |
| `run_info.json` | Run mode, Python package versions, input checksums, selected model, and whether the score was cached or recomputed |

The checks confirm that every requested row matches the template and corresponds to a `to_predict=True` row in the data. They check all four predicted columns. These checks establish the submission format; measuring prediction accuracy requires the true outcomes or suitable validation data.

Each run also has a separate folder inside `runs`. The latest successful results are copied into `outputs`. If a later run fails, read its error and log before using `outputs`: that folder can still contain older results, and an interruption during the final copy can leave an incomplete set. A successful run prints the final results location after publishing all files.

## Files included in this bundle

| File or folder | Role |
|---|---|
| `run.py` | Main launcher for quick generation and full retraining |
| `Run_Bioreactor.ipynb` | Notebook interface to the launcher |
| `requirements.txt` | Pinned versions of NumPy, pandas, SciPy, and scikit-learn |
| `bioreaction_remaining.csv` | Original challenge data |
| `bioreaction_template.csv` | Required submission rows and columns |
| `cache_manifest.json` | Input checksums for the bundled cached predictions |
| `lab/common.py` | Data loading, features, validation folds, score calculation, output adjustments, and result registration |
| `lab/statm.py` | Statistical prediction models |
| `lab/ode.py` and `lab/variants.py` | Biological simulator, fitting, and kinetic variants |
| `lab/hybrid.py`, `lab/hybrid2.py`, and `lab/hybrid3.py` | Biological simulation with learned residual corrections |
| `lab/p.py`, `lab/p2.py`, and `lab/pr.py` | Batch random effects, particle simulations, and residual variants |
| `lab/blend.py`, `lab/smooth2.py`, and `lab/final.py` | Model combinations and cell smoothing |
| `lab/report.py` | Report and CSV export from an existing result registry |
| `lab/ode/` | Original saved ODE parameters and simulations |
| `lab/oof/` | Original saved validation and test predictions |
| `lab/results.jsonl` | Original recorded experiment results |
| `lab/results_partition2.jsonl` | Original recorded results for the second partition |
| `lab/reproduce.sh` | Convenience wrapper that calls the new Python launcher in retraining mode |
| `reference/original_experiments_log.md` | Historical experiment report supplied with the original solution |

The earlier standalone `submission.csv` and `submission_mechanistic.csv` were superseded and are not the outputs to use for this final recipe. They and the original leaderboard notebook are not required to generate the current results.

## How full retraining works

The launcher creates a fresh run folder with the two data files and Python source code. Its parameter and prediction folders begin empty. It fits the following stages in order:

1. The baseline biological model and the required variants `m02_cov7`, `m11_consHill`, `m14_re6`, and `m21_re8`.
2. The direct statistical members with searched weights and marginal likelihood weights.
3. The hybrid corrections for the `m11_consHill` simulations.
4. The particle member used in comparison blends.
5. The final particle member with 1,000 simulated possibilities per batch and a residual correction.
6. The final model blends and cell smoothing.
7. The report and submission export, followed by output validation.

This rebuilds the final recipe and its dependencies. It does not repeat every historical experiment or the second partition. Each stage must finish successfully before the next begins. Original cached parameters, predictions, and historical score files are not copied into a training run.

The modelling source files have been retained from the original archive. The Python launcher improves run isolation, dependency checks, logging, and output validation. Numerical fitting results can vary with package versions and hardware, so full retraining is not promised to reproduce every original number exactly.

`lab/reproduce.sh` now delegates to this launcher. For Bash users it accepts `PY` as the Python executable, as the original script did. The Python commands above work without Bash.

## Verification performed for this bundle

The quick generation route was executed using Python 3.12.14, NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0, and scikit-learn 1.8.0. All three generated CSV files matched the original saved outputs numerically, and each passed the checks for 156 rows, 12 batches, and 624 finite positive predicted values. Python sources and notebook code cells were checked for syntax.

Full model retraining and the Colab upload and download interface were not run from start to finish while preparing this bundle. The recorded 7.028% result comes from the original experiment files. The bibliography below was checked for source identity and metadata; this was not a new replication of the cited papers or of the original biological analyses.

## Submitting the results separately

The runner and starter notebook only generate local files. To submit later, use the organiser's notebook or its `submit` helper with your actual team name. After that helper is available, this example loads the generated predictions:

```python
import pandas as pd

sub = pd.read_csv("outputs/submission_best.csv", dtype={"batch_id": str})
submit(sub, user="Team XX")
```

Run that only when you intend to submit. If the organiser requires authentication, follow its submission instructions. The example depends on the organiser's helper, which is not defined by this starter notebook.

## Research results and interpretation

The original archive records 140 distinct experiment names across six families. The table below was checked directly against `lab/results.jsonl`, keeping the last record for each name, as the original reporting code does. The file contains 159 records because some experiments were recorded more than once. These are historical results. Running the quick export does not repeat these experiments.

| Family | Recorded experiments | Best mean MAPE (%) |
|---|---:|---:|
| Baseline | 3 | 9.69 |
| Mechanistic ODE | 42 | 7.88 |
| Bayesian mechanistic ODE | 14 | 7.62 |
| Statistical | 23 | 7.47 |
| Hybrid | 48 | 7.39 |
| Blend | 10 | 7.03 |
| Total | 140 | 7.03 |

These counts include changes to features, calibration, and blending, rather than 140 unrelated algorithms. The 42 mechanistic results represent 21 kinetic variants, each evaluated with two output treatments. The best baseline includes a percentage error adjustment and substrate feed constraints; the simple typical-profile baseline scored 10.10%.

### Final results recorded in the package

| Output file | Primary split MAPE (%) | Second split MAPE (%) |
|---|---:|---:|
| `submission_best.csv` | 7.028 | 7.137 |
| `submission_nosearch.csv` | 7.076 | 7.183 |
| `submission_mechanistic_blend.csv` | 7.213 | 7.322 |

The best recipe is `b09_hyb_stat_bayesresid_smoothcells`. Its primary split errors are 6.65% for VCD, 6.61% for TCD, 6.83% for substrate, and 8.02% for byproduct. The overall score averages the absolute percentage errors across the four targets, sampling times, and batches. It is not a claim that every prediction is within 7.03% of the truth.

The second split scores come from `lab/results_partition2.jsonl`. They describe another partition of the same 49 labelled batches, not a new external dataset. The quick export uses previously saved predictions. It does not rerun either validation partition.

### What the experiments suggest

The following interpretations preserve the original research notes. The score tables above can be checked directly in the saved records. Estimates about measurement noise, explained variance, and biological causes were not independently re-established while preparing this bundle.

1. **Measurement noise may limit further gains.** The original notes estimate cell-count assay noise at about 6% to 7% and suggest a VCD/TCD MAPE floor near 5.2%. Treat the floor as a hypothesis based on those analyses, not a proven limit. The saved experiments place the strongest individual methods near 7.4% overall MAPE and the strongest blends near 7.0% to 7.2%. This is an observed result for the methods tried, not a guarantee that better methods cannot exist. The notes also suggest day-11 substrate uncertainty around 0.17 on a typical level near 0.9, corresponding to roughly 12% to 15% error; that proposed uncertainty limit is likewise unproven.

2. **A persistent batch difference appears useful for predicting cell density.** The notes describe the predictable cell-density component as approximately a constant multiplicative offset from the typical profile, visible from day 1. Initial VCD, initial byproduct and volume, feed B dosing, and feed C pulse count reportedly explain about 55% to 60% of this component. This is a compact empirical description of these batches, not a claim that one number captures all cell biology.

3. **Unexplained volume gain may act as an indirect byproduct signal.** The model calculates volume gain that is not accounted for by the logged feeds. The original analysis attributes the gain on days 2 to 5 to possible unlogged base addition and reports that it explains about half of the batch byproduct offset, rising to about 60% when initial volume is included. The proposed connection to lactate metabolism is biologically motivated [4], but the attribution to base addition and these variance estimates are dataset-specific interpretations.

4. **Feeding actions contain information about the missing substrate measurement.** The code assumes substrate feed on days 7 to 9 signals a value below about 1.95. For days 11 and 12, it interprets a dose near 0.6% of volume as below about 0.967 and a dose near 0.4% as above that threshold. It uses these rules to constrain substrate estimates and weight simulated batch trajectories. These thresholds are fixed assumptions in `lab/common.py`, not universal bioreactor rules.

5. **Different biological equations gave similar scores once batch information was included.** The 21 kinetic variants explored overflow production and consumption [1], [2], [3], logistic growth, inhibition, hidden nutrients, changes with culture phase, and feed D effects. More precisely, the 20 calibrated variants with batch covariates ranged from 7.877% to 8.080%, a spread of about 0.203 percentage points. The global baseline without those covariates scored 9.072%. Therefore, the original statement that all 21 variants were within 0.2 points is too broad. The records support the narrower finding that the batch covariates mattered more than the choice among these 20 calibrated structures.

6. **Combining biological structure and statistical corrections helped.** The best hybrid family score was 7.388%, compared with 7.472% for the best direct statistical model and 7.877% for the best mechanistic model. Blending complementary members reduced the recorded error further. This result supports the choice of a hybrid approach in this dataset; related hybrid modelling ideas are discussed in [5], [6], and [7]. Small differences between selected individual methods should not be treated as decisive evidence of superiority.

7. **Batch variation and multiple plausible trajectories provided another useful source of predictions.** The Bayesian member learns variations in growth, uptake, byproduct yield, consumption, and starting cell density. It simulates 1,000 possibilities per batch in the final recipe, weights them against the feeding record, and applies a statistical residual correction. The original research describes gains in substrate estimation from this approach. References [8] and [9] provide related context for modelling variation and transferring process knowledge; they do not independently validate this implementation or its scores.

8. **Some attempted refinements did not show a useful gain.** The supplied research summary reports no useful batch-order signal, no reliable deterministic rule for feed B, and no additional gain from further tuning of output shifts. These are findings within the attempted analyses, not proofs that such signals can never exist. The default percentage error adjustment remains part of the method. Smoothing cell-density deviations over time was retained because the original experiments reported a small improvement.

### Validation and interpretation limits

The primary validation uses seven groups of seven batches. Each group is held out while models fit the other 42 batches. Holding out complete batches is appropriate for assessing performance on other batches, but several details limit how strongly these scores can be interpreted.

1. **Selection among many experiments can make the winning score optimistic.** The same 49 labelled batches guided many modelling choices. A second partition is useful but does not provide independent external confirmation. The original notes estimate a CV standard error around 0.3 percentage points and advise caution about differences below roughly 0.2 points. Those uncertainty estimates were not independently established here.

2. **ODE validation fits are initialized using all labelled batches.** In `lab/ode.py`, the code first fits all 49 batches, then uses those parameters as the starting point for each fold's fit. The fold objective excludes its held-out targets, but its starting point has seen them. This is a possible source of optimistic bias whose size has not been measured. Feature standardization also uses the inputs from all 61 batches, including the unlabelled test batches. These details mean the original validation should not be described as fully isolated.

3. **The method reconstructs measurements from the available batch record.** Later feed and volume information can inform estimates for earlier sampling times. That is different from a live forecast that only has information available up to the prediction time. Availability in the supplied dataset does not by itself establish that every use complies with the competition rules; this bundle does not certify rule compliance.

4. **The alternative labelled `nosearch` still performs some scale selection.** Its direct statistical member sets feature-group weights by marginal likelihood instead of the main weight grid search. However, `theta_model2` in `lab/p2.py` still selects a regularization scale from a fixed list using leave-one-out error. The original description "no grid search anywhere" is inaccurate.

5. **Local validation does not determine the leaderboard result.** A planning estimate of about 7.0% to 7.2% local CV, or a leaderboard difference of about 0.4 percentage points with only 12 test batches, is an expectation from the original notes. It is not a validated prediction interval or a promised score. No new leaderboard submission or score is established by generating these files.

The numerical records cited in this section are preserved in `lab/results.jsonl` and `lab/results_partition2.jsonl`. The original interpretation notes remain in `reference/original_experiments_log.md`. The bracketed literature numbers refer to the bibliography elsewhere in this README.

## References and literature context

The references below provide biological and modelling context. The project scores, inferred feeding thresholds, noise estimates, and feature associations come from this challenge dataset and its experiment records. These papers do not independently validate those numerical claims.

Bibliographic details were checked against publisher pages and research article records on 21 November 2026. Citation counts are omitted because they change over time and depend on the indexing service. Reference numbering follows the supplied research notes.

1. Ma, N., et al. (2009). A single nutrient feed supports both chemically defined NS0 and CHO fed-batch processes: Improved productivity and lactate metabolism. *Biotechnology Progress*, 25, 1353 to 1363. [DOI: 10.1002/btpr.238](https://doi.org/10.1002/btpr.238).

2. Luginsland, M., Kontoravdi, C., Racher, A., Jaques, C., and Kiparissides, A. (2024). Elucidating lactate metabolism in industrial CHO cultures through the combined use of flux balance and principal component analyses. *Biochemical Engineering Journal*, 202, 109184. [DOI: 10.1016/j.bej.2023.109184](https://doi.org/10.1016/j.bej.2023.109184). The journal citation uses 2024; the supplied notes used 2023, which also appears in the DOI.

3. Sietaram, D., et al. (2025). A Multi Clone Kinetic Model for characterizing Chinese hamster ovary cell line variability. *Journal of Industrial Microbiology and Biotechnology*, 52, kuaf029. [DOI: 10.1093/jimb/kuaf029](https://doi.org/10.1093/jimb/kuaf029). The publisher gives 10 September 2025 as the publication date; the supplied 2024 year was corrected.

4. Qin, L., Savery, J., Cheeks, M., Farid, S. S., and Goldrick, S. (2026). A Hybrid Mechanistic Model Capturing Lactate Run-Away and Back-Metabolism in Mammalian Cell Culture. *Biotechnology and Bioengineering*, e70417. First published 22 October 2026. [DOI: 10.1002/bit.70417](https://doi.org/10.1002/bit.70417). The first author's name is Li Qin, and the publisher's citation uses Qin, L.; cite this paper as Qin et al., rather than Li et al.

5. Narayanan, H., Sokolov, M., Morbidelli, M., and Butte, A. (2019). A new generation of predictive models: The added value of hybrid models for manufacturing processes of therapeutic proteins. *Biotechnology and Bioengineering*, 116, 2540 to 2549. [DOI: 10.1002/bit.27097](https://doi.org/10.1002/bit.27097).

6. Narayanan, H., et al. (2022). Hybrid Models Based on Machine Learning and an Increasing Degree of Process Knowledge: Application to Cell Culture Processes. *Industrial & Engineering Chemistry Research*, 61(25), 8658 to 8672. [DOI: 10.1021/acs.iecr.1c04507](https://doi.org/10.1021/acs.iecr.1c04507).

7. Bayer, B., Duerkop, M., Portner, R., and Moller, J. (2023). Comparison of mechanistic and hybrid modeling approaches for characterization of a CHO cultivation process: Requirements, pitfalls and solution paths. *Biotechnology Journal*, 18, e2200381. [DOI: 10.1002/biot.202200381](https://doi.org/10.1002/biot.202200381). First published online 15 November 2022; the publisher's journal citation uses 2023.

8. Hickman, D. A., et al. (2019). Nonlinear mixed-effects models for kinetic parameter estimation with batch reactor data. *Chemical Engineering Journal*, 377, 119817. [DOI: 10.1016/j.cej.2018.08.203](https://doi.org/10.1016/j.cej.2018.08.203). The publisher lists the journal issue date as 1 December 2019; the supplied 2018 year also appears in the DOI.

9. Yu, L., del Rio Chanona, A., and Kontoravdi, C. (2026). Model-Enabled Knowledge Transfer Across Cell Lines, Culture Scales and Conditions. *Biotechnology and Bioengineering*, 123(10), 2613 to 2628. [DOI: 10.1002/bit.70269](https://doi.org/10.1002/bit.70269).

References [1], [2], and [3] provide context for lactate production, consumption, and kinetic modelling. Reference [4] studies the use of pH measurements in a mechanistic framework for lactate behaviour. References [5], [6], and [7] compare ways of combining process knowledge with learning from data. Reference [8] supports the general use of batch specific random effects in kinetic estimation, but its case study is a chemical reactor rather than this cell culture dataset. Reference [9] uses an ensemble Kalman filter for state and parameter estimation. It provides related context for model adaptation, while the particle weighting used in this package is a different implementation.
