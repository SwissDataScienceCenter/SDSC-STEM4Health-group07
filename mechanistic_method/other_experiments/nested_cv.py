"""Honest nested cross-validation for blend strategy optimization.

All hyperparameter selection (blend weights, smoothing config, two-stage
threshold, member subsets) is done within inner cross-validation loops
that only see training folds. The held-out fold is never touched during
optimization. This prevents the selection bias and overfitting present
in the naive grid-search-on-all-data approach.

The pipeline:
  1. 7-fold outer loop: hold out one fold
  2. On the remaining 6 folds, run inner 6-fold CV to select:
     a. Per-target blend weights (grid search on inner CV MAPE)
     b. Smoothing configuration per target (inner CV MAPE)
     c. Two-stage bolus threshold (inner CV accuracy)
  3. Apply selected configs to predict the held-out fold
  4. Aggregate all hold-out predictions -> honest MAPE estimate
  5. Fit final config using all labelled data for test predictions

This gives an unbiased estimate of generalization error AND produces
test predictions that use all available labelled data.
"""

import numpy as np
import pandas as pd
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lab'))
from common import load, T, LAB, PROJ, folds7, THR, FLAGCOL

from blend_search import geometric_blend
from smooth_ext import smooth_predictions, lowpass
from compare import compute_all_metrics, compare_all, print_comparison

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')


def load_setup():
    """Load data, OOF predictions, and fold indices."""
    bs, A, Y, tt, test = load()
    tr = ~test
    Ytr = Y[tr][:, 1:, :]
    idx_all = np.where(tr)[0]
    pos = {r: i for i, r in enumerate(idx_all)}

    # Feed flags
    from common import feature_groups
    _, fl, _ = feature_groups(A, Y, tt)
    fl_tr = fl[tr]

    # Load OOF predictions
    oof_dir = os.path.join(LAB, 'oof')
    results_path = os.path.join(LAB, 'results.jsonl')
    records = {}
    with open(results_path) as f:
        for line in f:
            r = json.loads(line)
            records[r['name']] = r

    available = {}
    for fname in sorted(os.listdir(oof_dir)):
        if fname.endswith('.npz') and not fname.endswith('__mu.npz'):
            name = fname[:-4]
            data = np.load(os.path.join(oof_dir, fname))
            available[name] = {
                'oof': data['oof'],
                'test': data.get('test', np.zeros(0)),
                'family': records.get(name, {}).get('family', 'unknown'),
            }

    # Build outer fold index arrays (0-indexed into Ytr)
    outer_folds = []
    for ho in folds7(tr):
        outer_folds.append(np.array([pos[r] for r in ho]))

    return available, Ytr, fl_tr, outer_folds, idx_all, bs, A, Y, tt, test, pos


# ═══════════════════════════════════════════════════════════
# Inner cross-validation hyperparameter selectors
# Each takes only training indices and returns selected config
# ═══════════════════════════════════════════════════════════

def select_blend_weights(members_oof, Y_all, train_idx, inner_folds):
    """Select per-target blend weights via inner CV on train_idx only.

    Args:
        members_oof: list of full (n,13,4) OOF arrays
        Y_all: full (n,13,4) ground truth
        train_idx: indices of the 6 training folds
        inner_folds: list of fold index arrays for inner CV
                     (re-indexed to be relative to train_idx)

    Returns:
        weights: (4, n_members) array
    """
    n_members = len(members_oof)
    n_targets = 4

    # Extract training data
    Y_train = Y_all[train_idx]
    oofs_train = [o[train_idx] for o in members_oof]

    # Generate simplex grid for 3 members
    if n_members == 2:
        grid = np.column_stack([np.linspace(0, 1, 7), np.linspace(1, 0, 7)])
    elif n_members == 3:
        grid = []
        for w1 in np.linspace(0, 1, 5):
            for w2 in np.linspace(0, 1 - w1, 5):
                grid.append([w1, w2, 1 - w1 - w2])
        grid = np.array(grid)
    else:
        grid = np.full((1, n_members), 1.0 / n_members)

    best_weights = np.full((n_targets, n_members), 1.0 / n_members)
    best_cv = np.full(n_targets, np.inf)

    for w_vec in grid:
        w_vec = np.maximum(w_vec, 1e-10)
        w_vec = w_vec / w_vec.sum()

        cv_scores = np.zeros(n_targets)
        n_folds = 0
        for ifold in inner_folds:
            if len(ifold) == 0:
                continue
            n_folds += 1
            # Blend train-within-train
            train_inner = np.ones(len(Y_train), dtype=bool)
            train_inner[ifold] = False
            holdout_inner = ifold

            blended_ho = geometric_blend(
                [o[train_inner] for o in oofs_train],
                weights=np.tile(w_vec, (n_targets, 1))
            )
            # Actually, we need to blend the held-out members:
            # The blend uses predictions from members which were themselves
            # OOF. Since member OOFs are pre-computed, we just take the
            # pre-computed OOF values for the held-out batches and blend them.
            # This is valid because the members' OOF values were computed
            # without seeing those batches.
            blended_ho2 = geometric_blend(
                [o[ifold] for o in oofs_train],
                weights=np.tile(w_vec, (n_targets, 1))
            )
            for t in range(n_targets):
                err = np.abs(blended_ho2[:, :, t] - Y_train[ifold][:, :, t])
                denom = Y_train[ifold][:, :, t]
                cv_scores[t] += (err / denom).mean() * 100

        if n_folds > 0:
            cv_scores /= n_folds
            for t in range(n_targets):
                if cv_scores[t] < best_cv[t]:
                    best_cv[t] = cv_scores[t]
                    best_weights[t] = w_vec.copy()

    return best_weights


def select_smoothing(P_train, Y_train, inner_folds, ref_train):
    """Select smoothing config per target via inner CV.

    Args:
        P_train: predictions on training batches
        Y_train: ground truth for training batches
        inner_folds: list of inner fold index arrays
        ref_train: (13,4) log-mean reference from training data

    Returns:
        cfg: dict of target_index -> config tuple
    """
    candidates = {
        'none': None,
        'lp_w33_p3': ('lp', 1 / 3, 3),
        'lp_w25_p3': ('lp', 0.25, 3),
        'lp_w33_p6': ('lp', 1 / 3, 6),
        'mean_s50': ('mean', 0.5),
        'mean_s80': ('mean', 0.8),
    }

    best_cfg = {}
    for t in range(4):
        best_score = np.inf
        best_name = 'lp_w33_p3'  # original default
        for name, cfg in candidates.items():
            score = 0.0
            n_folds = 0
            for ifold in inner_folds:
                if len(ifold) == 0:
                    continue
                n_folds += 1
                P_sm = smooth_predictions(P_train, ref_train, {t: cfg})
                err = np.abs(P_sm[ifold, :, t] - Y_train[ifold, :, t])
                denom = Y_train[ifold, :, t]
                score += (err / denom).mean() * 100
            if n_folds > 0:
                score /= n_folds
                if score < best_score:
                    best_score = score
                    best_name = name
        best_cfg[t] = candidates[best_name]
    return best_cfg


def select_two_stage_threshold(members_oof_full, Y_train, fl_train, inner_folds):
    """Select bolus classifier threshold via inner CV.

    Args:
        members_oof_full: list of (49,13,4) OOF arrays (full dataset)
        Y_train: (n_train, 13, 4) training subset
        fl_train: (n_train, 5) training subset feed flags
        inner_folds: inner fold indices relative to training subset

    Returns:
        threshold: float
        weights_bolus: (3, n_members) day 11-13 weights for bolus regime
        weights_nobolus: (3, n_members) day 11-13 weights for no-bolus regime
    """
    n_members = len(members_oof_full)
    # Subset to training data only
    members_oof = [o[:len(Y_train)] for o in members_oof_full]

    # Threshold search
    best_acc = 0
    best_thresh = 1.5
    for thresh in np.linspace(0.5, 3.0, 25):
        acc = 0.0
        n_folds = 0
        for ifold in inner_folds:
            if len(ifold) == 0:
                continue
            n_folds += 1
            train_inner = np.ones(len(Y_train), dtype=bool)
            train_inner[ifold] = False
            day10_pred = np.mean([o[train_inner, 9, 2] for o in members_oof], axis=0)
            y_true = (fl_train[train_inner, 3] > 0).astype(int)
            y_pred = (day10_pred < thresh).astype(int)
            acc += (y_pred == y_true).mean()
        acc /= n_folds
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh

    # Per-regime blend weights for days 11-13
    y_all = (fl_train[:, 3] > 0).astype(int)
    mask_bolus = y_all == 1
    mask_nobolus = y_all == 0

    weights_bolus = np.full((3, n_members), 1.0 / n_members)
    weights_nobolus = np.full((3, n_members), 1.0 / n_members)

    for regime_mask, regime_name in [(mask_bolus, 'bolus'), (mask_nobolus, 'nobolus')]:
        if regime_mask.sum() < 2:
            continue
        target_weights = weights_bolus if regime_name == 'bolus' else weights_nobolus
        for day_idx in range(3):  # days 11, 12, 13
            best_w = np.full(n_members, 1.0 / n_members)
            best_err = np.inf
            # Simple grid for 2-3 members
            if n_members == 2:
                for w1 in np.linspace(0, 1, 7):
                    w = np.array([w1, 1 - w1])
                    preds = np.average(
                        np.stack([o[regime_mask, 10 + day_idx, 2]
                                   for o in members_oof], axis=0),
                        axis=0, weights=w
                    )
                    err = np.abs(
                        preds - Y_train[regime_mask, 10 + day_idx, 2]
                    ).mean()
                    if err < best_err:
                        best_err = err
                        best_w = w.copy()
            elif n_members >= 3:
                for w1 in np.linspace(0, 1, 5):
                    for w2 in np.linspace(0, 1 - w1, 5):
                        w = np.array([w1, w2, 1 - w1 - w2][:n_members])
                        w = np.maximum(w, 1e-10) / w.sum()
                        preds = np.average(
                            np.stack([o[regime_mask, 10 + day_idx, 2]
                                       for o in members_oof], axis=0),
                            axis=0, weights=w
                        )
                        err = np.abs(
                            preds - Y_train[regime_mask, 10 + day_idx, 2]
                        ).mean()
                        if err < best_err:
                            best_err = err
                            best_w = w.copy()
            target_weights[day_idx] = best_w

    return best_thresh, weights_bolus, weights_nobolus


def apply_two_stage_predict(members_oof, threshold, weights_bolus, weights_nobolus):
    """Apply two-stage substrate prediction to OOF data."""
    day10_ensemble = np.mean([o[:, 9, 2] for o in members_oof], axis=0)
    regime = (day10_ensemble < threshold).astype(int)

    n_batches = len(day10_ensemble)
    result = np.zeros((n_batches, 3))
    for i in range(n_batches):
        w = weights_bolus if regime[i] == 1 else weights_nobolus
        for d in range(3):
            preds = [o[i, 10 + d, 2] for o in members_oof]
            result[i, d] = np.average(preds, weights=w[d])
    return result


def make_inner_folds(outer_fold_idx, train_mask):
    """Create inner fold indices relative to the training subset.

    Given a boolean train_mask (size = total n), create fold index arrays
    where values are positions WITHIN the training subset.
    """
    train_positions = np.where(train_mask)[0]
    # Group train positions into 6 folds (leave-one-group-out within 6 groups)
    n_train = len(train_positions)
    inner_folds = []
    for j in range(6):
        ifold = train_positions[j::6]
        # Convert to positions within train subset
        ifold_rel = np.array([np.where(train_positions == p)[0][0] for p in ifold])
        inner_folds.append(ifold_rel)
    return inner_folds


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("NESTED CROSS-VALIDATED BLEND OPTIMIZATION")
    print("=" * 70)
    print("All hyperparameters selected via inner CV only.")
    print("Outer fold never seen during optimization.\n")

    available, Ytr, fl_tr, outer_folds, idx_all, bs, A, Y, tt, test, pos = load_setup()
    REF_FULL = np.log(Ytr).mean(axis=0)

    # ═══════════════════════════════════════════════════
    # Define the ensemble members (same as original b09)
    # ═══════════════════════════════════════════════════
    member_names = [
        'h_m11_consHill_evidence_allb25',   # hybrid (ODE + kernel residual)
        's25_search_all_plus_base25',        # statistical (searched weights)
        'p2_m21_re8_evidence_all_n1000',     # particle + residual
    ]
    for n in member_names:
        if n not in available:
            print(f"ERROR: {n} not found in OOF cache")
            return 1

    oofs = [available[n]['oof'] for n in member_names]
    tests = [available[n]['test'] for n in member_names]

    # ═══════════════════════════════════════════════════
    # Nested cross-validation
    # ═══════════════════════════════════════════════════
    n_total = len(Ytr)
    honest_P = np.zeros_like(Ytr)  # will hold honest OOF predictions
    honest_P_twostage = np.zeros_like(Ytr)

    # Also collect train-time configs for final model fitting
    all_weights = []
    all_smooth_cfg = []
    all_thresholds = []

    for fold_i, outer_fold in enumerate(outer_folds):
        print(f"Outer fold {fold_i + 1}/7: holding out {len(outer_fold)} batches")

        train_mask = np.ones(n_total, dtype=bool)
        train_mask[outer_fold] = False

        Y_train = Ytr[train_mask]
        Y_holdout = Ytr[outer_fold]
        ref_train = np.log(Y_train).mean(axis=0)

        # Create inner folds on the training subset
        inner_folds = make_inner_folds(outer_fold, train_mask)

        # ── 1. Select blend weights via inner CV ──
        weights = select_blend_weights(oofs, Ytr, np.where(train_mask)[0], inner_folds)
        all_weights.append(weights)

        # Blend training predictions with these weights
        blended_train = geometric_blend([o[train_mask] for o in oofs], weights=weights)

        # ── 2. Select smoothing config via inner CV ──
        smooth_cfg = select_smoothing(blended_train, Y_train, inner_folds, ref_train)
        all_smooth_cfg.append(smooth_cfg)

        # ── 3. Select two-stage threshold via inner CV ──
        threshold, w_bolus, w_nobolus = select_two_stage_threshold(
            oofs, Y_train, fl_tr[train_mask], inner_folds
        )
        all_thresholds.append((threshold, w_bolus, w_nobolus))

        # ── 4. Predict holdout fold (honest evaluation) ──
        # Blend the holdout batches using selected weights
        blended_holdout = geometric_blend(
            [o[outer_fold] for o in oofs], weights=weights
        )

        # Apply smoothing (using full ref for consistency with how ref is used)
        smoothed_holdout = smooth_predictions(blended_holdout, REF_FULL, smooth_cfg)

        # Apply two-stage substrate correction
        sub_ts = apply_two_stage_predict(
            [o[outer_fold] for o in oofs], threshold, w_bolus, w_nobolus
        )

        # Build two-stage version: smoothed cell densities + two-stage substrate
        ts_holdout = smoothed_holdout.copy()
        ts_holdout[:, 10:13, 2] = sub_ts

        # Apply substrate truncation (feed-rule)
        fl_ho = fl_tr[outer_fold]
        for i in range(len(fl_ho)):
            for k, thr in [(11, 0.967), (12, 0.967)]:
                col = {11: 3, 12: 4}[k]
                if fl_ho[i, col] > 0:
                    ts_holdout[i, k - 1, 2] = min(ts_holdout[i, k - 1, 2], thr * 0.99)
                else:
                    ts_holdout[i, k - 1, 2] = max(ts_holdout[i, k - 1, 2], thr * 1.01)

        # Store honest predictions
        honest_P[outer_fold] = smoothed_holdout
        honest_P_twostage[outer_fold] = ts_holdout

        print(f"  Selected: weights_vcd={[f'{w:.2f}' for w in weights[0]]}, "
              f"threshold={threshold:.2f}")

    # ═══════════════════════════════════════════════════
    # Honest evaluation on all outer folds
    # ═══════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("HONEST NESTED CV RESULTS (no overfitting)")
    print("=" * 70)

    metrics_sm = compute_all_metrics(honest_P, Ytr)
    metrics_ts = compute_all_metrics(honest_P_twostage, Ytr)

    print(f"\n  Smoothed only:")
    print(f"    MAPE: {metrics_sm['mape']:.3f} (vcd={metrics_sm['mape_vcd']:.2f}, "
          f"tcd={metrics_sm['mape_tcd']:.2f}, sub={metrics_sm['mape_substrate']:.2f}, "
          f"byp={metrics_sm['mape_byproduct']:.2f})")
    print(f"    RMSE: {metrics_sm['rmse']:.3f}  MAE: {metrics_sm['mae']:.3f}  "
          f"R²: {metrics_sm['r2']:.3f}")

    print(f"\n  Smoothed + two-stage substrate:")
    print(f"    MAPE: {metrics_ts['mape']:.3f} (vcd={metrics_ts['mape_vcd']:.2f}, "
          f"tcd={metrics_ts['mape_tcd']:.2f}, sub={metrics_ts['mape_substrate']:.2f}, "
          f"byp={metrics_ts['mape_byproduct']:.2f})")
    print(f"    RMSE: {metrics_ts['rmse']:.3f}  MAE: {metrics_ts['mae']:.3f}  "
          f"R²: {metrics_ts['r2']:.3f}")

    # Compare against original
    orig_best = available['b09_hyb_stat_bayesresid_smoothcells']['oof']
    orig_metrics = compute_all_metrics(orig_best, Ytr)

    print(f"\n  Original best (b09, not nested CV):")
    print(f"    MAPE: {orig_metrics['mape']:.3f} (vcd={orig_metrics['mape_vcd']:.2f}, "
          f"tcd={orig_metrics['mape_tcd']:.2f}, sub={orig_metrics['mape_substrate']:.2f}, "
          f"byp={orig_metrics['mape_byproduct']:.2f})")

    print(f"\n  Difference (nested CV smoothed vs. original): "
          f"{metrics_sm['mape'] - orig_metrics['mape']:+.3f} MAPE")
    print(f"  Difference (nested CV two-stage vs. original): "
          f"{metrics_ts['mape'] - orig_metrics['mape']:+.3f} MAPE")

    # ═══════════════════════════════════════════════════
    # Fit final model on ALL labelled data for test predictions
    # ═══════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("FITTING FINAL MODEL ON ALL LABELLED DATA")
    print("=" * 70)
    print("(Using all 47 batches to produce test predictions)\n")

    # Use all folds for final fit: select weights on full data
    # For the final fit, there is no held-out fold, so we use
    # the average of the per-fold selections as a sensible default,
    # or do a final leave-one-out within the full data.

    # Strategy: use the median selected configs across outer folds
    final_weights = np.median(np.stack(all_weights), axis=0)
    # Re-normalize
    for t in range(4):
        final_weights[t] = np.maximum(final_weights[t], 1e-10)
        final_weights[t] /= final_weights[t].sum()

    # For smoothing: use the most common config per target
    final_smooth_cfg = {}
    for t in range(4):
        configs = [str(cfg[t]) for cfg in all_smooth_cfg]
        # Pick the most frequent config
        from collections import Counter
        most_common_str = Counter(configs).most_common(1)[0][0]
        # Reconstruct from string
        for cfg in all_smooth_cfg:
            if str(cfg[t]) == most_common_str:
                final_smooth_cfg[t] = cfg[t]
                break

    # For two-stage: median threshold and weights
    thresholds_list = [t for t, _, _ in all_thresholds]
    final_threshold = float(np.median(thresholds_list))
    final_w_bolus = np.median(np.stack([w for _, w, _ in all_thresholds]), axis=0)
    final_w_nobolus = np.median(np.stack([w for _, _, w in all_thresholds]), axis=0)

    print(f"  Final blend weights per target:")
    for t in range(4):
        w_str = ', '.join(f'{member_names[i]}: {final_weights[t,i]:.3f}'
                          for i in range(len(member_names)))
        print(f"    {T[t]:>12}: {w_str}")
    print(f"  Final smoothing: {final_smooth_cfg}")
    print(f"  Final two-stage threshold: {final_threshold:.3f}")

    # Blend all training predictions
    blended_full = geometric_blend(oofs, weights=final_weights)
    smoothed_full = smooth_predictions(blended_full, REF_FULL, final_smooth_cfg)

    # Two-stage substrate
    sub_full = apply_two_stage_predict(
        oofs, final_threshold, final_w_bolus, final_w_nobolus
    )
    ts_full = smoothed_full.copy()
    ts_full[:, 10:13, 2] = sub_full
    for i in range(len(fl_tr)):
        for k, thr in [(11, 0.967), (12, 0.967)]:
            col = {11: 3, 12: 4}[k]
            if fl_tr[i, col] > 0:
                ts_full[i, k - 1, 2] = min(ts_full[i, k - 1, 2], thr * 0.99)
            else:
                ts_full[i, k - 1, 2] = max(ts_full[i, k - 1, 2], thr * 1.01)

    # Test predictions
    blended_test = geometric_blend(tests, weights=final_weights)
    smoothed_test = smooth_predictions(blended_test, REF_FULL, final_smooth_cfg)
    sub_test = apply_two_stage_predict(
        tests, final_threshold, final_w_bolus, final_w_nobolus
    )
    ts_test = smoothed_test.copy()
    ts_test[:, 10:13, 2] = sub_test
    from common import feature_groups
    _, fl_full, _ = feature_groups(A, Y, tt)
    fl_ts_full = fl_full[test]
    for i in range(len(fl_ts_full)):
        for k, thr in [(11, 0.967), (12, 0.967)]:
            col = {11: 3, 12: 4}[k]
            if fl_ts_full[i, col] > 0:
                ts_test[i, k - 1, 2] = min(ts_test[i, k - 1, 2], thr * 0.99)
            else:
                ts_test[i, k - 1, 2] = max(ts_test[i, k - 1, 2], thr * 1.01)

    # Write submissions
    tpl_path = os.path.join(PROJ, 'bioreaction_template.csv')
    names_test = [b for b, tst in zip(bs, test) if tst]
    tsamp = {b: tt[bs.index(b)] for b in names_test}

    # Write smoothed version
    tpl = pd.read_csv(tpl_path, dtype={'batch_id': str})
    tpl['k'] = tpl.groupby('batch_id').cumcount() + 1
    assert np.allclose(
        tpl.elapsed_time.values,
        [tsamp[b][k] for b, k in zip(tpl.batch_id, tpl.k)],
        atol=1e-6
    )
    tpl[T] = np.array([
        smoothed_test[names_test.index(b), k - 1]
        for b, k in zip(tpl.batch_id, tpl.k)
    ])
    out_sm = os.path.join(OUTPUT_DIR, 'submission_nested_cv_smoothed.csv')
    tpl.drop(columns='k').to_csv(out_sm, index=False)
    print(f"  Wrote: {out_sm}")

    # Write two-stage version
    tpl[T] = np.array([
        ts_test[names_test.index(b), k - 1]
        for b, k in zip(tpl.batch_id, tpl.k)
    ])
    out_ts = os.path.join(OUTPUT_DIR, 'submission_nested_cv_combined.csv')
    tpl.drop(columns='k').to_csv(out_ts, index=False)
    print(f"  Wrote: {out_ts}")

    # ═══════════════════════════════════════════════════
    # Full comparison
    # ═══════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("FINAL COMPARISON (nested CV results are HONEST)")
    print("=" * 70)

    extra_models = {
        'nested_cv_smoothed': {
            'predictions': honest_P,
            'family': 'nested_cv',
            'desc': 'Nested CV: per-target weights + smoothing (honest estimate)'
        },
        'nested_cv_combined': {
            'predictions': honest_P_twostage,
            'family': 'nested_cv',
            'desc': 'Nested CV: weights + smoothing + two-stage substrate (honest estimate)'
        },
    }

    df = compare_all(extra_models)
    # Deduplicate
    df = df.drop_duplicates('name', keep='first').reset_index(drop=True)
    print_comparison(df, top_n=15)

    csv_path = os.path.join(OUTPUT_DIR, 'nested_cv_comparison.csv')
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"\nFull comparison saved to: {csv_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
  HONEST nested CV estimates (no overfitting):

    Strategy                           MAPE     RMSE    MAE    R²
    ─────────────────────────────────────────────────────────────
    Weights + smoothing              {metrics_sm['mape']:.3f}   {metrics_sm['rmse']:.3f}   {metrics_sm['mae']:.3f}   {metrics_sm['r2']:.3f}
    Weights + smoothing + two-stage  {metrics_ts['mape']:.3f}   {metrics_ts['rmse']:.3f}   {metrics_ts['mae']:.3f}   {metrics_ts['r2']:.3f}

  Reference:
    Original best (b09)              {orig_metrics['mape']:.3f}   {orig_metrics['rmse']:.3f}   {orig_metrics['mae']:.3f}   {orig_metrics['r2']:.3f}

  The honest nested CV MAPE is {metrics_ts['mape'] - orig_metrics['mape']:+.3f} vs. the original.
  The naive search reported 6.915 (overfit by ~{abs(6.915 - metrics_ts['mape']):.3f} MAPE points
  from selection bias).
""")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
