#!/usr/bin/env python3
"""Improved blend strategies for the bioreactor challenge.

Loads existing OOF predictions from the mechanistic_method and applies
several improvements to the final blend:

1. Per-target optimized blend weights
2. Extended temporal smoothing (beyond just VCD/TCD)
3. Two-stage substrate prediction (bolus classifier + per-regime regressor)
4. Broader member pool for blend combinations

Writes improved submission CSVs and a comparison report.
The original submission files are NOT overwritten.
"""

import numpy as np
import pandas as pd
import json
import os
import sys

# Add lab to path for common utilities
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lab'))
from common import load, T, LAB, PROJ, folds7, mape, post, THR, FLAGCOL

# Import improvement modules
from blend_search import (
    geometric_blend, optimize_blend, evaluate_member_subsets,
)
from smooth_ext import smooth_predictions, search_smoothing, lowpass
from compare import compute_all_metrics, compare_all, print_comparison
from two_stage import apply_two_stage, fit_two_stage_predictor

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')


def load_all_oofs():
    """Load all OOF predictions from the lab/oof directory."""
    bs, A, Y, tt, test = load()
    tr = ~test
    Ytr = Y[tr][:, 1:, :]  # (47, 13, 4)
    idx_all = np.where(tr)[0]
    pos = {r: i for i, r in enumerate(idx_all)}

    # Load results for metadata
    results_path = os.path.join(LAB, 'results.jsonl')
    records = {}
    with open(results_path) as f:
        for line in f:
            r = json.loads(line)
            records[r['name']] = r

    # Load OOF predictions
    oof_dir = os.path.join(LAB, 'oof')
    available = {}
    for fname in sorted(os.listdir(oof_dir)):
        if fname.endswith('.npz') and not fname.endswith('__mu.npz'):
            name = fname[:-4]
            data = np.load(os.path.join(oof_dir, fname))
            available[name] = {
                'oof': data['oof'],
                'test': data.get('test', np.zeros(0)),
                'family': records.get(name, {}).get('family', 'unknown'),
                'mape': records.get(name, {}).get('mean', float('inf')),
            }

    # Build 7-fold split indices (0-indexed into Ytr)
    folds = []
    for ho in folds7(tr):
        folds.append(np.array([pos[r] for r in ho]))

    return available, Ytr, folds, bs, A, Y, tt, test, idx_all, pos


def write_submission_csv(P_test, template_path, output_path, batch_ids, tt, test):
    """Write a submission CSV from test predictions."""
    tpl = pd.read_csv(template_path, dtype={'batch_id': str})
    tpl['k'] = tpl.groupby('batch_id').cumcount() + 1
    names = [b for b, tst in zip(batch_ids, test) if tst]
    tsamp = {b: tt[batch_ids.index(b)] for b in names}

    assert np.allclose(
        tpl.elapsed_time.values,
        [tsamp[b][k] for b, k in zip(tpl.batch_id, tpl.k)],
        atol=1e-6
    )

    tpl[T] = np.array([
        P_test[names.index(b), k - 1]
        for b, k in zip(tpl.batch_id, tpl.k)
    ])
    assert tpl[T].notna().all().all() and (tpl[T] > 0).all().all()
    tpl.drop(columns='k').to_csv(output_path, index=False)


def original_blend(members_oof, members_test):
    """Original equal-weight geometric mean blend."""
    P = np.exp(np.mean([np.log(o) for o in members_oof], axis=0))
    Pt = np.exp(np.mean([np.log(o) for o in members_test], axis=0))
    return P, Pt


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("IMPROVED BLEND STRATEGIES")
    print("=" * 70)
    print()
    print("Loading OOF predictions and setting up folds...")

    available, Ytr, folds, bs, A, Y, tt, test, idx_all, pos = load_all_oofs()
    REF = np.log(Ytr).mean(axis=0)  # (13, 4) reference for smoothing
    fl = None
    # Need to get feed flags — re-derive from the data
    # Load with feature_groups to get fl
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lab'))
    from common import feature_groups
    _, fl, _ = feature_groups(A, Y, tt)
    fl_tr = fl[~test]  # 47 batches

    print(f"  {len(available)} models available in OOF cache")
    print(f"  {len(Ytr)} labelled batches, 7 folds of ~7 batches each")
    print()

    # ──────────────────────────────────────────────────
    # Identify key members for blending
    # ──────────────────────────────────────────────────
    # Core members used in original final.py
    core_members = {
        'H': 'h_m11_consHill_evidence_allb25',    # hybrid (ODE + kernel residual)
        'SS': 's25_search_all_plus_base25',        # statistical (searched weights)
        'SE': 's26_evidence_all_plus_base25',       # statistical (evidence weights)
        'B': 'p_m21_re8_kernel_b25',                # mechanistic-bayes particle
        'P2': 'p2_m21_re8_evidence_all_n1000',      # hybrid particle + residual
    }

    # Verify all members exist
    for k, v in core_members.items():
        if v not in available:
            print(f"  WARNING: {k} ({v}) not found in OOF cache, skipping")
            core_members[k] = None

    # ──────────────────────────────────────────────────
    # Strategy 1: Per-target optimized blend weights
    # ──────────────────────────────────────────────────
    print("=" * 70)
    print("STRATEGY 1: Per-target optimized blend weights")
    print("=" * 70)

    # Use the original best-blend member set: H, SS, P2 (b09)
    members_3 = ['H', 'SS', 'P2']
    member_names = [core_members[m] for m in members_3 if core_members[m]]
    if len(member_names) >= 2:
        oofs_3 = [available[n]['oof'] for n in member_names]
        tests_3 = [available[n]['test'] for n in member_names]

        # Original blend (baseline)
        orig_P, orig_Pt = original_blend(oofs_3, tests_3)
        orig_metrics = compute_all_metrics(orig_P, Ytr)

        # Optimized per-target blend
        opt_P, weights, pt_mape, overall_mape = optimize_blend(
            oofs_3, Ytr, folds, resolution=7
        )
        opt_metrics = compute_all_metrics(opt_P, Ytr)

        print(f"\n  Members: {', '.join(member_names)}")
        print(f"\n  Original equal-weight blend:")
        print(f"    MAPE: {orig_metrics['mape']:.3f} (vcd={orig_metrics['mape_vcd']:.2f}, "
              f"tcd={orig_metrics['mape_tcd']:.2f}, sub={orig_metrics['mape_substrate']:.2f}, "
              f"byp={orig_metrics['mape_byproduct']:.2f})")
        print(f"    RMSE: {orig_metrics['rmse']:.3f}  MAE: {orig_metrics['mae']:.3f}  "
              f"R²: {orig_metrics['r2']:.3f}")

        print(f"\n  Optimized per-target blend:")
        print(f"    MAPE: {opt_metrics['mape']:.3f} (vcd={opt_metrics['mape_vcd']:.2f}, "
              f"tcd={opt_metrics['mape_tcd']:.2f}, sub={opt_metrics['mape_substrate']:.2f}, "
              f"byp={opt_metrics['mape_byproduct']:.2f})")
        print(f"    RMSE: {opt_metrics['rmse']:.3f}  MAE: {opt_metrics['mae']:.3f}  "
              f"R²: {opt_metrics['r2']:.3f}")
        print(f"\n  Per-target weights (vcd, tcd, substrate, byproduct):")
        for t in range(4):
            w_str = ', '.join(f'{member_names[i]}: {weights[t,i]:.3f}'
                              for i in range(len(member_names)))
            print(f"    {T[t]:>12}: {w_str}")

        delta = opt_metrics['mape'] - orig_metrics['mape']
        print(f"\n  Improvement: {delta:+.3f} MAPE ({-delta:.3f} decrease)"
              if delta < 0 else f"\n  Change: {delta:+.3f} MAPE")

        # Write optimized test predictions
        opt_Pt = geometric_blend(tests_3, weights=weights)
        write_submission_csv(
            opt_Pt,
            os.path.join(PROJ, 'bioreaction_template.csv'),
            os.path.join(OUTPUT_DIR, 'submission_improved_weights.csv'),
            bs, tt, test
        )
        print(f"  Wrote: submission_improved_weights.csv")
    else:
        print("  Not enough members for blend optimization.")
        opt_P = None

    # ──────────────────────────────────────────────────
    # Strategy 2: Extended temporal smoothing
    # ──────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("STRATEGY 2: Extended temporal smoothing")
    print("=" * 70)

    if opt_P is not None:
        # Search for best smoothing per target
        best_cfg, best_mape = search_smoothing(opt_P, Ytr, folds, pos, REF)

        print("\n  Best smoothing config per target:")
        for t in range(4):
            cfg_str = str(best_cfg[t]) if best_cfg[t] else 'none'
            print(f"    {T[t]:>12}: {cfg_str} (CV MAPE: {best_mape[t]:.2f})")

        # Apply best smoothing
        P_smoothed = smooth_predictions(opt_P, REF, best_cfg)
        sm_metrics = compute_all_metrics(P_smoothed, Ytr)

        print(f"\n  With extended smoothing:")
        print(f"    MAPE: {sm_metrics['mape']:.3f} (vcd={sm_metrics['mape_vcd']:.2f}, "
              f"tcd={sm_metrics['mape_tcd']:.2f}, sub={sm_metrics['mape_substrate']:.2f}, "
              f"byp={sm_metrics['mape_byproduct']:.2f})")
        print(f"    RMSE: {sm_metrics['rmse']:.3f}  MAE: {sm_metrics['mae']:.3f}  "
              f"R²: {sm_metrics['r2']:.3f}")

        # Apply best smoothing to test predictions too
        Pt_smoothed = smooth_predictions(opt_Pt, REF, best_cfg)
        write_submission_csv(
            Pt_smoothed,
            os.path.join(PROJ, 'bioreaction_template.csv'),
            os.path.join(OUTPUT_DIR, 'submission_improved_smoothed.csv'),
            bs, tt, test
        )
        print(f"  Wrote: submission_improved_smoothed.csv")
    else:
        sm_metrics = None

    # ──────────────────────────────────────────────────
    # Strategy 3: Two-stage substrate prediction
    # ──────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("STRATEGY 3: Two-stage substrate prediction")
    print("=" * 70)

    if len(member_names) >= 2:
        improved_ts, ts_info = apply_two_stage(oofs_3, Ytr, fl_tr)
        ts_metrics = compute_all_metrics(improved_ts, Ytr)

        print(f"\n  Bolus classifier threshold: {ts_info['threshold']:.3f}")
        print(f"  Classification accuracy: {ts_info['accuracy']:.1%}")
        print(f"\n  Two-stage substrate predictions:")
        print(f"    MAPE: {ts_metrics['mape']:.3f} (vcd={ts_metrics['mape_vcd']:.2f}, "
              f"tcd={ts_metrics['mape_tcd']:.2f}, sub={ts_metrics['mape_substrate']:.2f}, "
              f"byp={ts_metrics['mape_byproduct']:.2f})")

        ts_delta = ts_metrics['mape_substrate'] - orig_metrics['mape_substrate']
        print(f"    Substrate MAPE change: {ts_delta:+.2f}")

        # Apply two-stage to test predictions too
        ts_test = orig_Pt.copy()
        day10_test = np.mean([t[:, 9, 2] for t in tests_3], axis=0)
        predict_fn2, thresh2, acc2 = fit_two_stage_predictor(oofs_3, Ytr, fl_tr)
        ts_test_sub = predict_fn2(day10_test)
        ts_test[:, 10:13, 2] = ts_test_sub
        # Apply feed-rule truncation
        fl_ts2 = fl[test]
        for i in range(len(fl_ts2)):
            for k, thr in [(11, 0.967), (12, 0.967)]:
                col = {11: 3, 12: 4}[k]
                if fl_ts2[i, col] > 0:
                    ts_test[i, k - 1, 2] = min(ts_test[i, k - 1, 2], thr * 0.99)
                else:
                    ts_test[i, k - 1, 2] = max(ts_test[i, k - 1, 2], thr * 1.01)
        write_submission_csv(
            ts_test,
            os.path.join(PROJ, 'bioreaction_template.csv'),
            os.path.join(OUTPUT_DIR, 'submission_improved_twostage.csv'),
            bs, tt, test
        )
        print(f"  Wrote: submission_improved_twostage.csv")
    else:
        ts_metrics = None

    # ──────────────────────────────────────────────────
    # Strategy 4: Explore broader member pool
    # ──────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("STRATEGY 4: Broader member pool exploration")
    print("=" * 70)

    # Select top individual models per family
    top_per_family = {}
    for name, info in available.items():
        fam = info['family']
        if fam not in top_per_family or info['mape'] < top_per_family[fam][0]:
            top_per_family[fam] = (info['mape'], name)

    print("\n  Best individual models per family:")
    for fam, (m, name) in sorted(top_per_family.items(), key=lambda x: x[1][0]):
        print(f"    {fam:<20}: {name:<45} MAPE {m:.3f}")

    # Try all 3-member combinations of the best per family
    candidate_names = [name for _, name in top_per_family.values()]
    candidate_names = [n for n in candidate_names if n in available]
    print(f"\n  Evaluating {len(candidate_names)}-member subset combinations...")

    subset_results = evaluate_member_subsets(
        {n: (available[n]['oof'], available[n]['test']) for n in candidate_names},
        Ytr, folds
    )

    if subset_results:
        print(f"\n  Top 5 member combinations (with optimized weights):")
        for i, r in enumerate(subset_results[:5]):
            print(f"    {i+1}. MAPE {r['mape']:.3f} — {', '.join(r['members'])}")
            pt = r['per_target_mape']
            print(f"       targets: vcd={pt[0]:.2f} tcd={pt[1]:.2f} sub={pt[2]:.2f} byp={pt[3]:.2f}")

        best_subset = subset_results[0]
        # Write best subset blend
        best_subset_Pt = geometric_blend(
            [available[n]['test'] for n in best_subset['members']],
            weights=best_subset['weights']
        )
        write_submission_csv(
            best_subset_Pt,
            os.path.join(PROJ, 'bioreaction_template.csv'),
            os.path.join(OUTPUT_DIR, 'submission_improved_best_subset.csv'),
            bs, tt, test
        )
        print(f"\n  Wrote: submission_improved_best_subset.csv")
    else:
        best_subset = None

    # ──────────────────────────────────────────────────
    # Combine strategies: optimized weights + smoothing + two-stage
    # ──────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("STRATEGY 5: Combined (optimized weights + smoothing + two-stage)")
    print("=" * 70)

    if opt_P is not None and sm_metrics is not None:
        # Start with optimized-weight blend, apply smoothing to VCD/TCD,
        # and two-stage substrate
        # Smooth only VCD/TCD as in original + optional byproduct
        cfg_combined = best_cfg.copy() if best_cfg else {0: ('lp', 1/3, 3), 1: ('lp', 1/3, 3)}
        P_combined = smooth_predictions(opt_P, REF, cfg_combined)

        # Apply two-stage substrate
        predict_fn3, thresh3, acc3 = fit_two_stage_predictor(oofs_3, Ytr, fl_tr)
        day10_ens = np.mean([o[:, 9, 2] for o in oofs_3], axis=0)
        sub_improved = predict_fn3(day10_ens)
        P_combined[:, 10:13, 2] = sub_improved

        # Apply feed-rule truncation
        for i in range(len(fl_tr)):
            for k, thr in [(11, 0.967), (12, 0.967)]:
                col = {11: 3, 12: 4}[k]
                if fl_tr[i, col] > 0:
                    P_combined[i, k - 1, 2] = min(P_combined[i, k - 1, 2], thr * 0.99)
                else:
                    P_combined[i, k - 1, 2] = max(P_combined[i, k - 1, 2], thr * 1.01)

        comb_metrics = compute_all_metrics(P_combined, Ytr)

        print(f"\n  Combined strategy:")
        print(f"    MAPE: {comb_metrics['mape']:.3f} (vcd={comb_metrics['mape_vcd']:.2f}, "
              f"tcd={comb_metrics['mape_tcd']:.2f}, sub={comb_metrics['mape_substrate']:.2f}, "
              f"byp={comb_metrics['mape_byproduct']:.2f})")
        print(f"    RMSE: {comb_metrics['rmse']:.3f}  MAE: {comb_metrics['mae']:.3f}  "
              f"R²: {comb_metrics['r2']:.3f}")

        # Combined test predictions
        Pt_combined = smooth_predictions(opt_Pt, REF, cfg_combined)
        day10_test_ens = np.mean([t[:, 9, 2] for t in tests_3], axis=0)
        sub_test_improved = predict_fn3(day10_test_ens)
        Pt_combined[:, 10:13, 2] = sub_test_improved
        fl_ts = fl[test]
        for i in range(len(fl_ts)):
            for k, thr in [(11, 0.967), (12, 0.967)]:
                col = {11: 3, 12: 4}[k]
                if fl_ts[i, col] > 0:
                    Pt_combined[i, k - 1, 2] = min(Pt_combined[i, k - 1, 2], thr * 0.99)
                else:
                    Pt_combined[i, k - 1, 2] = max(Pt_combined[i, k - 1, 2], thr * 1.01)
        write_submission_csv(
            Pt_combined,
            os.path.join(PROJ, 'bioreaction_template.csv'),
            os.path.join(OUTPUT_DIR, 'submission_improved_combined.csv'),
            bs, tt, test
        )
        print(f"  Wrote: submission_improved_combined.csv")

    # ──────────────────────────────────────────────────
    # Full comparison
    # ──────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    extra_models = {}
    if opt_P is not None:
        extra_models['opt_blend_weights'] = {
            'predictions': opt_P,
            'family': 'improved',
            'desc': 'Per-target optimized blend weights'
        }
    if sm_metrics is not None:
        extra_models['opt_smoothed'] = {
            'predictions': P_smoothed,
            'family': 'improved',
            'desc': 'Optimized weights + extended smoothing'
        }
    if ts_metrics is not None:
        extra_models['opt_twostage'] = {
            'predictions': improved_ts,
            'family': 'improved',
            'desc': 'Optimized weights + two-stage substrate'
        }
    if 'comb_metrics' in dir():
        extra_models['opt_combined'] = {
            'predictions': P_combined,
            'family': 'improved',
            'desc': 'Combined: weights + smoothing + two-stage'
        }

    df = compare_all(extra_models)
    print_comparison(df, top_n=15)

    # Save comparison CSV
    csv_path = os.path.join(OUTPUT_DIR, 'improved_comparison.csv')
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"\nFull comparison saved to: {csv_path}")

    # Summary of all output files
    print("\n" + "=" * 70)
    print("OUTPUT FILES")
    print("=" * 70)
    for f in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"  {f} ({size_kb:.1f} KB)")

    print("\nDone. Original submissions in ../outputs/ are unchanged.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
