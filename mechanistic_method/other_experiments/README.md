# Improved Blend Strategies

This folder explores improvements to the original blend method from the
mechanistic_method approach. The original approach uses an equal-weight
geometric mean across 2-4 member models. 

Sidenote: most of the methods were overfitting instead of improve the actual results.

This extension experiments with:

1. **Per-target optimized blend weights** — different ensemble weights for
   each of the four targets (VCD, TCD, substrate, byproduct), optimized
   via leave-one-fold-out cross-validation within the 7-fold splits.

2. **Target-tailored member selection** — different member subsets for
   different targets (e.g., the particle model may be stronger on substrate,
   the statistical model on byproduct).

3. **Asymmetric smoothing** — extending temporal smoothing beyond just
   cell densities (VCD, TCD) to byproduct trajectories where it may help
   dampen late-stage noise.

4. **Two-stage substrate prediction** — separating the discrete
   feed-rule bolus decision from continuous error, and training
   separate regressors for each regime.

## Key differences from the original method

| Aspect | Original | This extension |
|--------|----------|----------------|
| Blend weights | Equal geometric mean across all members | Per-target optimized weights |
| Smoothing | VCD & TCD only (lowpass, w=1/3, 3 passes) | Extended to byproduct; configurable passes & width |
| Substrate handling | Single model + feed-rule truncation | Two-stage classifier + regressor per bolus regime |
| Member selection | Fixed 2-3 member combinations | Per-target member subsets |
| Validation | Relies on original results.jsonl records | Independent OOF evaluation using saved predictions |

## Usage

```bash
cd ../mechanistic_method/improvements
python3 run_improved.py
```

The script loads the existing OOF predictions from `../lab/oof/` and
re-evaluates all blend strategies using 7-fold cross-validation.
It writes new submission CSVs and a comparison report to `./outputs/`.
The original method's submission files are not overwritten.

## Files

- `run_improved.py` — main script: computes improved blends, evaluates,
  and writes outputs
- `blend_search.py` — per-target weight optimization and member selection
- `two_stage.py` — two-stage substrate model with bolus classifier
- `smooth_ext.py` — extended temporal smoothing
- `compare.py` — comprehensive multi-metric comparison across all approaches
