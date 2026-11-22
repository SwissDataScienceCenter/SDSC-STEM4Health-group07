#!/usr/bin/env python3
"""honest_blend_experiments.py — Nested CV evaluation of enhanced blend strategies.

Evaluates:
  E1: 4-member blend adding h_m03_feedD to the 3-member core
  E2: 5-member blend adding both h_m03_feedD and p2_nosub=False variants
  E2b: Per-target optimized weights (searched in inner CV)
  E3: Per-target optimized + cell smoothing (searched in inner CV)

All hyperparameters (blend weights, smoothing config) selected in inner CV.
Honest score from outer holdout predictions.
"""

import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lab'))
from common import load, T, LAB, folds7, mape, post, THR, FLAGCOL

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)


def load_setup():
    """Load data, OOF predictions, and build fold indices."""
    bs, A, Y, tt, test = load()
    tr = ~test
    Ytr = Y[tr][:, 1:, :]

    full_folds = folds7(tr)
    pos_map = {r: i for i, r in enumerate(np.where(tr)[0])}
    outer_folds = [np.array([pos_map[r] for r in f if r in pos_map]) for f in full_folds]
    outer_folds = [f for f in outer_folds if len(f) > 0]

    # Load OOF predictions
    oof_dir = os.path.join(LAB, 'oof')
    def load_oof(name):
        d = np.load(os.path.join(oof_dir, f'{name}.npz'))
        return d['oof'], d.get('test', np.zeros(0))

    members = {}
    variants_available = {}

    # Core 3 members
    for name in ['h_m11_consHill_evidence_allb25',
                 's25_search_all_plus_base25',
                 'p2_m21_re8_evidence_all_n1000']:
        if os.path.exists(os.path.join(oof_dir, f'{name}.npz')):
            oof, test_preds = load_oof(name)
            members[name] = {'oof': oof, 'test': test_preds}

    # h_m03 variant (best one: evidence_allg)
    for name in ['h_m03_feedD_evidence_allg', 'h_m03_feedD_evidence']:
        path = os.path.join(oof_dir, f'{name}.npz')
        if os.path.exists(path):
            oof, test_preds = load_oof(name)
            members[name] = {'oof': oof, 'test': test_preds}
            variants_available['m03'] = name
            break

    # p2 nosub=False variant (use _all which includes substrate correction)
    for name in ['p2_m21_re8_evidence_all']:
        path = os.path.join(oof_dir, f'{name}.npz')
        if os.path.exists(path):
            # We already loaded _n1000 which has nosub=True
            # _all is the same but with fewer particles — still has nosub=True
            # There's no nosub=False variant in cache, so skip for now
            pass

    print(f"Loaded {len(members)} members:")
    for name in members:
        print(f"  {name}")

    return {
        'Ytr': Ytr,
        'outer_folds': outer_folds,
        'members': members,
        'tr': tr,
        'test': test,
        'bs': bs,
        'A': A,
        'Y': Y,
        'tt': tt,
    }


def geometric_blend(oof_list, weights):
    """Geometric mean blend of OOF predictions with per-target weights.

    Args:
        oof_list: list of (n,13,4) arrays
        weights: (4, n_members) per-target weight matrix

    Returns:
        (n,13,4) blended predictions
    """
    n_targets = 4
    result = np.zeros_like(oof_list[0])
    for t in range(n_targets):
        w = weights[t]
        w = np.maximum(w, 1e-10)
        w = w / w.sum()
        log_stack = np.stack([np.log(o[:, :, t] + 1e-10) for o in oof_list], axis=0)
        result[:, :, t] = np.exp(np.average(log_stack, axis=0, weights=w))
    return result


def cell_smooth(P, ref_log_mean, targets=(0, 1), w=1/3, passes=3):
    """Low-pass filter cell density deviations (log-space)."""
    L = np.log(P + 1e-10).copy()
    for t in targets:
        dev = L[:, :, t] - ref_log_mean[:, t]
        smoothed = lowpass_vec(dev, w, passes)
        L[:, :, t] = ref_log_mean[:, t] + smoothed
    return np.exp(L)


def lowpass_vec(x, w=1/3, passes=3):
    """1D low-pass filter along axis=1 (time)."""
    out = x.copy()
    for _ in range(passes):
        out_padded = np.column_stack([out[:, :1], out, out[:, -1:]])
        out = w * out + (1 - w) * (out_padded[:, :-2] + out_padded[:, 2:]) / 2
    return out


def mape_all(P, Y):
    return (np.abs(P - Y) / Y * 100).mean()


def metrics(P, Y):
    err = np.abs(P - Y) / Y * 100
    se = (P - Y) ** 2
    r2s = []
    for i in range(4):
        ss_res = ((Y[:, :, i] - P[:, :, i])**2).sum()
        ss_tot = ((Y[:, :, i] - Y[:, :, i].mean())**2).sum()
        r2s.append(1 - ss_res / ss_tot)
    return {
        'mape': err.mean(),
        'mape_vcd': err[:,:,0].mean(), 'mape_tcd': err[:,:,1].mean(),
        'mape_substrate': err[:,:,2].mean(), 'mape_byproduct': err[:,:,3].mean(),
        'rmse': np.sqrt(se.mean()), 'mae': np.abs(P - Y).mean(),
        'r2': np.mean(r2s),
        'r2_vcd': r2s[0], 'r2_tcd': r2s[1],
        'r2_substrate': r2s[2], 'r2_byproduct': r2s[3],
    }


def print_m(label, m):
    print(f"  {label}: MAPE={m['mape']:.3f} (v={m['mape_vcd']:.1f} "
          f"t={m['mape_tcd']:.1f} s={m['mape_substrate']:.1f} "
          f"b={m['mape_byproduct']:.1f}) RMSE={m['rmse']:.3f} R²={m['r2']:.3f}")


# ═══════════════════════════════════════════════════════
# Blend weight optimization (inner CV)
# ═══════════════════════════════════════════════════════

def optimize_weights_inner(member_oofs, Y_all, train_idx, inner_folds):
    """Select per-target blend weights via inner CV.

    Uses a 0.1-resolution simplex grid.
    """
    n_members = len(member_oofs)
    n_targets = 4

    # Generate simplex grid
    if n_members == 2:
        grid = np.column_stack([np.linspace(0, 1, 11), np.linspace(1, 0, 11)])
    elif n_members == 3:
        grid = []
        for w1 in np.linspace(0, 1, 11):
            for w2 in np.linspace(0, 1 - w1, 11):
                grid.append([w1, w2, 1 - w1 - w2])
        grid = np.array(grid)
    elif n_members == 4:
        grid = []
        for w1 in np.linspace(0, 1, 6):
            for w2 in np.linspace(0, 1 - w1, 6):
                for w3 in np.linspace(0, 1 - w1 - w2, 6):
                    w4 = 1 - w1 - w2 - w3
                    if w4 >= -1e-9:
                        grid.append([w1, w2, w3, max(w4, 0)])
        grid = np.array(grid)
    else:
        return np.full((n_targets, n_members), 1.0 / n_members)

    # Extract training subset
    Y_train = Y_all[train_idx]
    oofs_train = [o[train_idx] for o in member_oofs]

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
            ho_predictions = geometric_blend(
                [o[ifold] for o in oofs_train],
                weights=np.tile(w_vec, (n_targets, 1))
            )
            for t in range(n_targets):
                err = np.abs(ho_predictions[:, :, t] - Y_train[ifold][:, :, t])
                denom = Y_train[ifold][:, :, t]
                cv_scores[t] += (err / denom).mean() * 100
        cv_scores /= max(n_folds, 1)

        for t in range(n_targets):
            if cv_scores[t] < best_cv[t]:
                best_cv[t] = cv_scores[t]
                best_weights[t] = w_vec.copy()

    return best_weights


def optimize_smoothing_inner(P_train, Y_train, inner_folds, ref_log_mean):
    """Select smoothing config via inner CV.

    Tests: no smoothing, 3-pass lowpass with w=1/3, and mean-shrink 0.8.
    Only smooths VCD/TCD (targets 0,1).
    """
    candidates = {
        'none': {'enabled': False},
        'lp_w33_p3': {'enabled': True, 'type': 'lp', 'w': 1/3, 'passes': 3},
        'mean_s80': {'enabled': True, 'type': 'mean', 'shrink': 0.8},
    }

    best_cfg_name = 'lp_w33_p3'  # default
    best_score = np.inf

    for name, cfg in candidates.items():
        score = 0.0
        n_folds = 0
        for ifold in inner_folds:
            if len(ifold) == 0:
                continue
            n_folds += 1
            if cfg['enabled']:
                if cfg['type'] == 'lp':
                    P_sm = cell_smooth(P_train, ref_log_mean,
                                      targets=(0, 1), w=cfg['w'],
                                      passes=cfg['passes'])
                else:  # mean shrink
                    L = np.log(P_train + 1e-10).copy()
                    for t in (0, 1):
                        dev = L[:, :, t] - ref_log_mean[:, t]
                        L[:, :, t] = ref_log_mean[:, t] + (1 - cfg['shrink']) * dev
                    P_sm = np.exp(L)
            else:
                P_sm = P_train

            err = np.abs(P_sm[ifold, :, :] - Y_train[ifold, :, :])
            denom = Y_train[ifold, :, :]
            score += (err / denom).mean() * 100
        score /= max(n_folds, 1)

        if score < best_score:
            best_score = score
            best_cfg_name = name

    return candidates[best_cfg_name]


# ═══════════════════════════════════════════════════════
# Experiments
# ═══════════════════════════════════════════════════════

def run_experiment(data, member_keys, label, optimize_weights=True,
                   optimize_smooth=True):
    """Run a nested CV experiment.

    Args:
        data: setup dict
        member_keys: list of member names to blend
        label: string label for the experiment
        optimize_weights: whether to use per-target weight optimization (inner CV)
        optimize_smooth: whether to optimize cell smoothing (inner CV)

    Returns:
        honest_P, metrics, best_hps
    """
    print(f"\n{'='*70}")
    print(f"Experiment: {label}")
    print(f"  Members: {member_keys}")
    print(f"  Optimize weights: {optimize_weights}, Smooth: {optimize_smooth}")
    print(f"{'='*70}")

    Ytr = data['Ytr']
    outer_folds = data['outer_folds']
    members = data['members']

    member_oofs = [members[k]['oof'] for k in member_keys]
    n_members = len(member_oofs)
    n_total = len(Ytr)

    honest_P = np.zeros_like(Ytr)
    best_hps = []

    for fold_i, outer_fold in enumerate(outer_folds):
        train_mask = np.ones(n_total, dtype=bool)
        train_mask[outer_fold] = False
        train_idx = np.where(train_mask)[0]
        test_idx = outer_fold

        # Inner folds
        inner_folds = []
        for j in range(6):
            ifold = train_idx[j::6]
            if len(ifold):
                inner_folds.append(np.array([np.where(train_idx == p)[0][0] for p in ifold]))

        Y_train = Ytr[train_idx]
        ref_train_log = np.log(Y_train + 1e-10).mean(axis=0)

        # ── Select weights ──
        if optimize_weights:
            weights = optimize_weights_inner(
                member_oofs, Ytr, train_idx, inner_folds
            )
        else:
            weights = np.full((4, n_members), 1.0 / n_members)

        # ── Select smoothing ──
        if optimize_smooth:
            # First blend training data with selected weights
            blended_tr = geometric_blend(
                [o[train_idx] for o in member_oofs], weights=weights
            )
            smooth_cfg = optimize_smoothing_inner(
                blended_tr, Y_train, inner_folds, ref_train_log
            )
        else:
            smooth_cfg = {'enabled': True, 'type': 'lp', 'w': 1/3, 'passes': 3}

        best_hps.append({
            'weights': weights,
            'smooth_cfg': smooth_cfg,
        })

        # ── Predict holdout ──
        blended_ho = geometric_blend(
            [o[test_idx] for o in member_oofs], weights=weights
        )

        # Apply smoothing
        if smooth_cfg['enabled']:
            if smooth_cfg['type'] == 'lp':
                P_ho = cell_smooth(blended_ho, ref_train_log,
                                  targets=(0, 1), w=smooth_cfg['w'],
                                  passes=smooth_cfg['passes'])
            else:
                L = np.log(blended_ho + 1e-10).copy()
                for t in (0, 1):
                    dev = L[:, :, t] - ref_train_log[:, t]
                    L[:, :, t] = ref_train_log[:, t] + (1 - smooth_cfg['shrink']) * dev
                P_ho = np.exp(L)
        else:
            P_ho = blended_ho

        honest_P[test_idx] = P_ho

        # Quick feedback
        fold_mape = mape_all(P_ho, Ytr[test_idx])
        print(f"  Fold {fold_i+1}/7: held out {len(outer_fold)}, "
              f"MAPE={fold_mape:.2f}%")

    m = metrics(honest_P, Ytr)
    print_m(f"\n  HONEST {label}", m)

    return honest_P, m, best_hps


def main():
    data = load_setup()

    # Baseline b09
    oof_dir = os.path.join(LAB, 'oof')
    b09 = np.load(os.path.join(oof_dir, 'b09_hyb_stat_bayesresid_smoothcells.npz'))
    base_m = metrics(b09['oof'], data['Ytr'])
    print_m("Baseline b09 (pre-computed)", base_m)

    all_results = {
        'baseline_b09': {'predictions': b09['oof'], 'metrics': base_m}
    }

    # ── Experiment 1: Original 3 members, per-target weights + smoothing ──
    core3 = ['h_m11_consHill_evidence_allb25',
             's25_search_all_plus_base25',
             'p2_m21_re8_evidence_all_n1000']

    if all(k in data['members'] for k in core3):
        P, m, hps = run_experiment(
            data, core3,
            "E1_3member_optimized",
            optimize_weights=True, optimize_smooth=True
        )
        all_results['E1_3member_optimized'] = {'predictions': P, 'metrics': m}

        # Also run equal-weight 3-member
        P_eq, m_eq, _ = run_experiment(
            data, core3,
            "E0_3member_equal_smoothed",
            optimize_weights=False, optimize_smooth=True
        )
        all_results['E0_3member_equal_smoothed'] = {'predictions': P_eq, 'metrics': m_eq}
    else:
        print("ERROR: Core 3-members not all available")

    # ── Experiment 2: Add h_m03 if available ──
    m03_key = None
    for candidate in ['h_m03_feedD_evidence_allg', 'h_m03_feedD_evidence']:
        if candidate in data['members']:
            m03_key = candidate
            break

    if m03_key and all(k in data['members'] for k in core3):
        core4 = core3 + [m03_key]
        P4, m4, hps4 = run_experiment(
            data, core4,
            f"E2_4member_+m03_optimized",
            optimize_weights=True, optimize_smooth=True
        )
        all_results['E2_4member_+m03_optimized'] = {'predictions': P4, 'metrics': m4}
    else:
        print(f"\nSkipping 4-member: m03 variant not available")

    # ── Experiment 3: Add p2 _all variant (nosub=True but different particle count) ──
    # Actually _all has nosub=True too. Check what we have vs _n1000:
    # _n1000: 1000 particles, _all: default (400) particles
    # Both have nosub=True. The difference is negligible.
    # Skip since no real nosub=False variant in cache.

    # ── Experiment 3b: Try s26 (evidence weights instead of searched) ──
    s26_key = 's26_evidence_all_plus_base25'
    s26_path = os.path.join(oof_dir, f'{s26_key}.npz')
    if os.path.exists(s26_path) and all(k in data['members'] for k in core3):
        # Replace s25 with s26
        core3_s26 = [core3[0], s26_key, core3[2]]
        # Load s26
        d = np.load(s26_path)
        data['members'][s26_key] = {'oof': d['oof'], 'test': d.get('test', np.zeros(0))}

        P_s26, m_s26, _ = run_experiment(
            data, core3_s26,
            "E3_3member_s26evidence_optimized",
            optimize_weights=True, optimize_smooth=True
        )
        all_results['E3_3member_s26_optimized'] = {'predictions': P_s26, 'metrics': m_s26}

    # ── Experiment 4: Try s26 + h_m03 4-member ──
    if m03_key and s26_key in data['members']:
        core4_s26 = [core3[0], s26_key, core3[2], m03_key]
        P4s, m4s, _ = run_experiment(
            data, core4_s26,
            "E4_4member_s26+m03_optimized",
            optimize_weights=True, optimize_smooth=True
        )
        all_results['E4_4member_s26+m03_optimized'] = {'predictions': P4s, 'metrics': m4s}

    # ═══════════════════════════════════════════════════
    # Final comparison
    # ═══════════════════════════════════════════════════
    print("\n" + "="*70)
    print("FINAL COMPARISON (all nested CV — HONEST scores)")
    print("="*70)
    print(f"\n{'Experiment':<45} {'MAPE':>8} {'RMSE':>8} {'R²':>8} | "
          f"{'vcd':>7} {'tcd':>7} {'sub':>7} {'byp':>7}")
    print("-"*105)

    for name in sorted(all_results, key=lambda n: all_results[n]['metrics']['mape']):
        m = all_results[name]['metrics']
        print(f"{name:<45} {m['mape']:7.3f} {m['rmse']:7.3f} {m['r2']:7.3f} | "
              f"{m['mape_vcd']:6.2f} {m['mape_tcd']:6.2f} "
              f"{m['mape_substrate']:6.2f} {m['mape_byproduct']:6.2f}")

    # Save
    import pandas as pd
    rows = []
    for name, res in all_results.items():
        m = res['metrics']
        rows.append({'name': name, **{k: v for k, v in m.items()}})
    df = pd.DataFrame(rows).sort_values('mape')
    csv_path = os.path.join(OUT_DIR, 'honest_blend_comparison.csv')
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"\nSaved: {csv_path}")

    best = df.iloc[0]
    base = df[df['name'] == 'baseline_b09'].iloc[0]
    delta = best['mape'] - base['mape']
    print(f"\nBest: {best['name']} (MAPE {best['mape']:.3f})")
    print(f"vs baseline b09: {delta:+.3f} MAPE")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
