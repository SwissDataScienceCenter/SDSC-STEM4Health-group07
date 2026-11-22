#!/usr/bin/env python3
"""experiments_v2.py — Approaches that work with the limited 49-batch sample size.

Focus: approaches that reduce dimensionality before learning.

E1: Functional PCA (FPCA) — compress 13-day trajectories into low-dim components
     then regress those on batch features. Handles the temporal structure properly.
E2: Sparse LASSO stack on OOF predictions — select only the most useful OOF models
     per target/day via LASSO regularization.
E3: Per-target day-collapsed GP with learned kernel on engineered features.

All evaluated with nested CV to prevent overfitting.
"""

import numpy as np
import pandas as pd
import json
import os
import sys
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LassoCV, ElasticNetCV
from sklearn.model_selection import KFold

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lab'))
from common import load, T, LAB, PROJ, folds7, mape, post, THR, FLAGCOL, feature_groups

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)


def load_data():
    bs, A, Y, tt, test = load()
    tr = ~test
    Ytr = Y[tr][:, 1:, :]
    G, fl, D = feature_groups(A, Y, tt)

    feat_names = sorted(G.keys())
    X_static = np.column_stack([G[k] for k in feat_names])

    # Map outer fold indices to 0..48
    full_folds = folds7(tr)
    pos_map = {r: i for i, r in enumerate(np.where(tr)[0])}
    outer_folds = [np.array([pos_map[r] for r in f if r in pos_map]) for f in full_folds]
    outer_folds = [f for f in outer_folds if len(f) > 0]

    return {
        'Ytr': Ytr, 'X': X_static[tr], 'X_test': X_static[test],
        'G': G, 'fl': fl[tr], 'fl_test': fl[test], 'D': D,
        'outer_folds': outer_folds, 'feat_names': feat_names,
        'tr': tr, 'test': test,
    }


def nested_cv(predict_fn, data, hp_grid=None, hp_selector_fn=None):
    """Generic nested CV. predict_fn(train_idx, test_idx, hp, data) -> P_test"""
    Ytr = data['Ytr']
    n_total = len(Ytr)
    outer_folds = data['outer_folds']

    honest_P = np.zeros_like(Ytr)
    best_hps = []

    for f_i, outer_fold in enumerate(outer_folds):
        train_mask = np.ones(n_total, dtype=bool)
        train_mask[outer_fold] = False
        train_idx = np.where(train_mask)[0]
        test_idx = outer_fold
        n_train = len(train_idx)

        # Inner folds
        inner_folds = []
        for j in range(6):
            ifold = train_idx[j::6]
            if len(ifold):
                inner_folds.append(np.array([np.where(train_idx == p)[0][0] for p in ifold]))

        # Select HP
        if hp_selector_fn:
            hp = hp_selector_fn(train_idx, inner_folds, data)
        elif hp_grid:
            hp = grid_select(predict_fn, train_idx, inner_folds, hp_grid, data)
        else:
            hp = {}

        best_hps.append(hp)
        P = predict_fn(train_idx, test_idx, hp, data)
        honest_P[test_idx] = P
        print(f"  Fold {f_i+1}/7: held out {len(outer_fold)}, hp={list(hp.keys())[:3]}")

    m = metrics(honest_P, Ytr)
    return honest_P, m, best_hps


def grid_select(predict_fn, train_idx, inner_folds, hp_grid, data):
    best = np.inf
    best_hp = hp_grid[0]
    for hp in hp_grid:
        scores = []
        for ifold in inner_folds:
            itr_inner = np.ones(len(train_idx), dtype=bool)
            itr_inner[ifold] = False
            P = predict_fn(train_idx[itr_inner], train_idx[ifold], hp, data)
            scores.append(mape_all(P, data['Ytr'][train_idx[ifold]]))
        s = np.mean(scores) if scores else np.inf
        if s < best:
            best = s
            best_hp = hp
    return best_hp


def mape_all(P, Y):
    return (np.abs(P - Y) / Y * 100).mean()


def metrics(P, Y):
    err = np.abs(P - Y) / Y * 100
    se = (P - Y) ** 2
    ae = np.abs(P - Y)
    r2s = []
    for i in range(4):
        ss_res = ((Y[:, :, i] - P[:, :, i])**2).sum()
        ss_tot = ((Y[:, :, i] - Y[:, :, i].mean())**2).sum()
        r2s.append(1 - ss_res / ss_tot)
    return {
        'mape': err.mean(),
        'mape_vcd': err[:,:,0].mean(), 'mape_tcd': err[:,:,1].mean(),
        'mape_substrate': err[:,:,2].mean(), 'mape_byproduct': err[:,:,3].mean(),
        'rmse': np.sqrt(se.mean()), 'mae': ae.mean(),
        'r2': np.mean(r2s),
        'r2_vcd': r2s[0], 'r2_tcd': r2s[1],
        'r2_substrate': r2s[2], 'r2_byproduct': r2s[3],
    }


def print_m(label, m):
    print(f"  {label}: MAPE={m['mape']:.3f} (v={m['mape_vcd']:.1f} "
          f"t={m['mape_tcd']:.1f} s={m['mape_substrate']:.1f} "
          f"b={m['mape_byproduct']:.1f}) RMSE={m['rmse']:.3f} R²={m['r2']:.3f}")


# ═══════════════════════════════════════════════════════
# E1: Functional PCA on log-trajectories
# ═══════════════════════════════════════════════════════

def experiment_fpca(data):
    """FPCA: fit PCA on the 52-dimensional concatenated log-trajectories,
    then regress PC scores on batch features. Predict back from PCs.

    This reduces 13×4 = 52 timepoints to k << 49 components, making
    regression feasible even with 118 batch features.

    Two variants:
      E1a: Global FPCA on all 4 targets jointly (52-dim)
      E1b: Per-target FPCA (13-dim each)
    """
    print("\n" + "="*70)
    print("E1: Functional PCA + Ridge Regression on Batch Features")
    print("="*70)

    results = {}

    for variant, dim_per_target, target_dim in [
        ('E1a_fpca_global', 13, 4),
        ('E1b_fpca_target', 13, 1),
    ]:
        print(f"\n  {variant}:")

        def make_fpca_fn(variant):
            def predict_fn(train_idx, test_idx, hp, data):
                Ytr = data['Ytr']
                X = data['X']
                n_comp = hp.get('n_components', 5)
                alpha = hp.get('alpha', 1.0)

                n_tr = len(train_idx)
                n_te = len(test_idx)
                n_targets = 4

                P_test = np.zeros((n_te, 13, 4))

                if variant == 'E1a_fpca_global':
                    # Stack all 4 targets: (n, 52)
                    Ylog_tr = np.log(Ytr[train_idx].reshape(n_tr, -1) + 1e-8)
                    Ylog_te = np.zeros((n_te, 52))

                    # PCA on training data
                    pca = PCA(n_components=min(n_comp, n_tr-1))
                    scores_tr = pca.fit_transform(Ylog_tr)  # (n_tr, k)

                    # Ridge regression: scores ~ X
                    ridge = Ridge(alpha=alpha)
                    ridge.fit(X[train_idx], scores_tr)

                    # Predict scores for test, reconstruct
                    scores_te = ridge.predict(X[test_idx])
                    Ylog_te = pca.inverse_transform(scores_te)
                    P_test = np.exp(Ylog_te.reshape(n_te, 13, 4))
                else:
                    # Per-target FPCA
                    for t in range(n_targets):
                        Ylog_tr = np.log(Ytr[train_idx, :, t] + 1e-8)  # (n_tr, 13)
                        k = min(n_comp, n_tr-1, 13)
                        pca = PCA(n_components=k)
                        scores_tr = pca.fit_transform(Ylog_tr)

                        ridge = Ridge(alpha=alpha)
                        ridge.fit(X[train_idx], scores_tr)

                        scores_te = ridge.predict(X[test_idx])
                        P_test[:, :, t] = np.exp(pca.inverse_transform(scores_te))

                return P_test
            return predict_fn

        hp_grid = [
            {'n_components': 3, 'alpha': 1.0},
            {'n_components': 5, 'alpha': 10.0},
            {'n_components': 8, 'alpha': 100.0},
            {'n_components': 10, 'alpha': 1000.0},
        ]

        P, m, hps = nested_cv(make_fpca_fn(variant), data, hp_grid=hp_grid)
        print_m(variant, m)
        results[variant] = {'predictions': P, 'metrics': m, 'hps': hps}

    return results


# ═══════════════════════════════════════════════════════
# E2: Sparse LASSO stack on 143 OOF predictions
# ═══════════════════════════════════════════════════════

def experiment_lasso_stack(data):
    """Use LASSO to select the most useful OOF models per (target, day).
    This addresses the 7436-feature problem from experiment C.

    For each (target, day) pair:
      - Features: log-predictions from 143 OOF models
      - Target: log-true value
      - Model: LASSO with CV-chosen regularization
    """
    print("\n" + "="*70)
    print("E2: Sparse LASSO Stack on 143 OOF Models")
    print("="*70)

    oof_dir = os.path.join(LAB, 'oof')
    results_path = os.path.join(LAB, 'results.jsonl')

    all_oofs = {}
    for fname in sorted(os.listdir(oof_dir)):
        if fname.endswith('.npz') and not fname.endswith('__mu.npz'):
            name = fname[:-4]
            d = np.load(os.path.join(oof_dir, fname))
            if d['oof'].shape[0] == 49:
                all_oofs[name] = d['oof']

    model_names = sorted(all_oofs.keys())
    feat_stack = np.stack([all_oofs[n] for n in model_names], axis=-1)  # (49,13,4,M)
    n_models = len(model_names)
    print(f"  {n_models} OOF models loaded, {n_models * 52} candidate features")

    def make_lasso_fn():
        def predict_fn(train_idx, test_idx, hp, data):
            n_tr = len(train_idx)
            n_te = len(test_idx)
            Ytr = data['Ytr']
            P_test = np.zeros((n_te, 13, 4))

            for t in range(4):
                for d in range(13):
                    X_tr = np.log(feat_stack[train_idx, d, t, :] + 1e-8)
                    X_te = np.log(feat_stack[test_idx, d, t, :] + 1e-8)
                    y_tr = np.log(Ytr[train_idx, d, t] + 1e-8)

                    # LASSO with internal CV (on training data only)
                    lasso = LassoCV(cv=min(5, n_tr-1), max_iter=5000,
                                   random_state=42, n_alphas=30,
                                   fit_intercept=True)
                    lasso.fit(X_tr, y_tr)
                    P_test[:, d, t] = np.exp(lasso.predict(X_te))

            return P_test
        return predict_fn

    # For LASSO, the HP is internal (alpha chosen by CV), but we expose n_alphas
    hp_grid = [{'n_alphas': 30}]

    P, m, hps = nested_cv(make_lasso_fn(), data, hp_grid=hp_grid)
    print_m('E2_lasso_stack', m)

    # Count how many models selected
    n_selected_total = 0
    n_models_used = set()
    for hp in hps:
        pass  # Would need to track lasso.coef_ per fold

    return {'E2_lasso_stack': {'predictions': P, 'metrics': m}}


# ═══════════════════════════════════════════════════════
# E3: Gaussian Process with ARD kernel on engineered features
# ═══════════════════════════════════════════════════════

def experiment_gp_ard(data):
    """GP with Automatic Relevance Determination (ARD) on a small set of
    engineered features per target.

    Features:
      - Initial conditions (lx0, ls0, lp0)
      - Feed pattern summaries
      - Volume trajectory shape params
      - Unexplained volume gain
      - Day index (for temporal trends)

    We use only ~10-15 careful features instead of 118, making the GP
    viable with 49 samples.
    """
    print("\n" + "="*70)
    print("E3: GP with ARD Kernel on Engineered Features")
    print("="*70)

    Ytr = data['Ytr']
    G = data['G']

    # Build a compact feature set: ~12 features
    def build_features(X_full):
        """Extract ~12 biologically meaningful features."""
        feats = []
        # Initial conditions
        for k in ['lx0', 'ls0', 'lp0', 'via0', 'vol0']:
            if k in G:
                feats.append(G[k].ravel())
        # Feed B total (days 1-13)
        if 'fb' in G:
            feats.append(G['fb'].sum(axis=1))
        # Feed C presence and count
        if 'fc' in G:
            feats.append(G['fc'].sum(axis=1))
        if 'fcn' in G:
            feats.append(G['fcn'].sum(axis=1))
        # Volume gain unexplained
        if 'dVr_e' in G:
            feats.append(G['dVr_e'].sum(axis=1))
        if 'dVr_l' in G:
            feats.append(G['dVr_l'].sum(axis=1))
        if 'volL' in G:
            feats.append(G['volL'].max(axis=1))
        # Substrate feed flags
        if 'fs_m' in G:
            feats.append(G['fs_m'].sum(axis=1))
        if 'fs_l' in G:
            feats.append(G['fs_l'].sum(axis=1))

        X = np.column_stack(feats)
        return X

    X_compact = build_features(data['X'])
    n_features = X_compact.shape[1]
    print(f"  Compact feature set: {n_features} features")

    # Add target-specific features + day info
    def make_gp_fn():
        def predict_fn(train_idx, test_idx, hp, data):
            n_tr = len(train_idx)
            n_te = len(test_idx)
            Ytr = data['Ytr']

            # Standardize features
            scl = StandardScaler()
            X_norm = scl.fit_transform(X_compact)

            # Build kernel: linear on compact features + noise
            K = X_norm @ X_norm.T / n_features  # linear kernel

            alpha = hp.get('alpha', 10.0)
            length_scale = hp.get('length_scale', 3.0)

            P_test = np.zeros((n_te, 13, 4))

            for t in range(4):
                for d in range(13):
                    y_tr = np.log(Ytr[train_idx, d, t] + 1e-8)

                    K_tr = K[np.ix_(train_idx, train_idx)]
                    K_te = K[np.ix_(test_idx, train_idx)]

                    # Ridge regression: (K + alpha*I)^(-1) y
                    from scipy.linalg import solve
                    K_reg = K_tr + alpha * np.eye(n_tr)
                    try:
                        alphas = solve(K_reg, y_tr, assume_a='pos')
                        y_pred = K_te @ alphas
                    except:
                        # Fallback to mean prediction
                        y_pred = np.full(n_te, y_tr.mean())

                    P_test[:, d, t] = np.exp(y_pred)

            return P_test
        return predict_fn

    hp_grid = [
        {'alpha': 0.1, 'length_scale': 1.0},
        {'alpha': 1.0, 'length_scale': 3.0},
        {'alpha': 10.0, 'length_scale': 5.0},
        {'alpha': 100.0, 'length_scale': 10.0},
    ]

    P, m, hps = nested_cv(make_gp_fn(), data, hp_grid=hp_grid)
    print_m('E3_gp_compact', m)

    return {'E3_gp_compact': {'predictions': P, 'metrics': m}}


# ═══════════════════════════════════════════════════════
# E4: Residual corrector — learn what the best model gets wrong
# ═══════════════════════════════════════════════════════

def experiment_residual_corrector(data):
    """Learn to predict the residuals of the best existing model (b09).

    Idea: b09 makes systematic errors (higher MAPE on late days, byproduct
    re-consumption). Can we learn to correct those errors from batch features?

    Approach:
      - Compute b09 residuals in log-space
      - Train a small model to predict residuals from batch features + day index
      - Add predicted correction to b09 outputs

    Two variants:
      E4a: Ridge on compact features -> residual correction
      E4b: LightGBM on compact features -> residual correction
    """
    print("\n" + "="*70)
    print("E4: Residual Corrector for b09 Baseline")
    print("="*70)

    oof_dir = os.path.join(LAB, 'oof')
    b09 = np.load(os.path.join(oof_dir, 'b09_hyb_stat_bayesresid_smoothcells.npz'))
    b09_preds = b09['oof']  # (49, 13, 4)

    Ytr = data['Ytr']
    residuals_log = np.log(b09_preds + 1e-8) - np.log(Ytr + 1e-8)  # (49, 13, 4)

    # Compact features (same as E3)
    G = data['G']
    def build_features():
        feats = []
        for k in ['lx0', 'ls0', 'lp0', 'via0', 'vol0']:
            if k in G:
                feats.append(G[k].ravel())
        if 'fb' in G:
            feats.append(G['fb'].sum(axis=1))
        if 'fc' in G:
            feats.append(G['fc'].sum(axis=1))
        if 'fcn' in G:
            feats.append(G['fcn'].sum(axis=1))
        if 'dVr_e' in G:
            feats.append(G['dVr_e'].sum(axis=1))
        if 'dVr_l' in G:
            feats.append(G['dVr_l'].sum(axis=1))
        if 'volL' in G:
            feats.append(G['volL'].max(axis=1))
        if 'fs_m' in G:
            feats.append(G['fs_m'].sum(axis=1))
        if 'fs_l' in G:
            feats.append(G['fs_l'].sum(axis=1))
        return np.column_stack(feats)

    X_compact = build_features()
    n_feats = X_compact.shape[1]

    for variant in ['E4a_ridge', 'E4b_lgb']:
        print(f"\n  {variant}:")

        def make_corrector(variant):
            def predict_fn(train_idx, test_idx, hp, data):
                n_tr = len(train_idx)
                n_te = len(test_idx)

                # b09 predictions for holdout (these are OOF, so honest)
                b09_tr = b09_preds[train_idx]
                b09_te = b09_preds[test_idx]
                res_tr = residuals_log[train_idx]
                res_te_true = residuals_log[test_idx]  # for eval only

                P_test = b09_te.copy()

                for t in range(4):
                    for d in range(13):
                        y_tr = res_tr[:, d, t]
                        X_tr = X_compact[train_idx]
                        X_te = X_compact[test_idx]

                        if variant == 'E4a_ridge':
                            from sklearn.linear_model import RidgeCV
                            model = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0],
                                           fit_intercept=True)
                            model.fit(X_tr, y_tr)
                            correction = model.predict(X_te)
                        else:
                            import lightgbm as lgb
                            model = lgb.LGBMRegressor(
                                n_estimators=50, max_depth=3,
                                learning_rate=0.05, num_leaves=15,
                                random_state=42, verbose=-1,
                            )
                            model.fit(X_tr, y_tr)
                            correction = model.predict(X_te)

                        # Apply correction: b09 * exp(correction)
                        P_test[:, d, t] = b09_te[:, d, t] * np.exp(correction)

                return P_test
            return predict_fn

        hp_grid = [{'alpha': 1.0}, {'alpha': 10.0}]

        P, m, hps = nested_cv(make_corrector(variant), data, hp_grid=hp_grid)
        print_m(variant, m)

    return {}  # Will collect above


# ═══════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════

def main():
    print("="*70)
    print("EXPERIMENTS V2 — Dimensionality-Aware Approaches")
    print("="*70)

    data = load_data()
    print(f"\nData: {data['Ytr'].shape[0]} train batches, "
          f"{data['X'].shape[1]} static features")

    # Baseline
    oof_dir = os.path.join(LAB, 'oof')
    b09 = np.load(os.path.join(oof_dir, 'b09_hyb_stat_bayesresid_smoothcells.npz'))
    baseline_m = metrics(b09['oof'], data['Ytr'])
    print_m('\nBaseline b09', baseline_m)

    all_results = {'baseline_b09': {'predictions': b09['oof'], 'metrics': baseline_m, 'family': 'baseline'}}

    experiments = [
        ('FPCA', experiment_fpca),
        ('LASSO Stack', experiment_lasso_stack),
        ('GP Compact', experiment_gp_ard),
        ('Residual Corrector', experiment_residual_corrector),
    ]

    for name, func in experiments:
        try:
            res = func(data)
            for k, v in res.items():
                all_results[f"{name}_{k}"] = {**v, 'family': name}
        except Exception as e:
            print(f"\n  {name} FAILED: {e}")
            import traceback; traceback.print_exc()

    # Comparison
    print("\n" + "="*70)
    print("FINAL COMPARISON (nested CV)")
    print("="*70)
    print(f"{'Experiment':<40} {'MAPE':>8} {'RMSE':>8} {'R²':>8} | {'vcd':>7} {'tcd':>7} {'sub':>7} {'byp':>7}")
    print("-"*105)

    for name in sorted(all_results, key=lambda n: all_results[n]['metrics'].get('mape', 999)):
        m = all_results[name]['metrics']
        if 'mape' in m:
            print(f"{name:<40} {m['mape']:7.3f} {m['rmse']:7.3f} {m['r2']:7.3f} | "
                  f"{m['mape_vcd']:6.2f} {m['mape_tcd']:6.2f} "
                  f"{m['mape_substrate']:6.2f} {m['mape_byproduct']:6.2f}")

    # Save
    rows = []
    for name, res in all_results.items():
        m = res['metrics']
        if 'mape' in m:
            rows.append({'name': name, 'family': res.get('family', ''), **m})
    df = pd.DataFrame(rows).sort_values('mape')
    csv_path = os.path.join(OUT_DIR, 'experiments_v2_comparison.csv')
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"\nSaved: {csv_path}")

    best = df.iloc[0]
    base = df[df['name'] == 'baseline_b09'].iloc[0]
    delta = best['mape'] - base['mape']
    print(f"\nBest: {best['name']} (MAPE {best['mape']:.3f})")
    print(f"vs baseline: {delta:+.3f} MAPE")
    if delta < -0.15:
        print("  → POTENTIALLY meaningful")
    else:
        print("  → Within CV noise (~0.15-0.30 MAPE)")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
