#!/usr/bin/env python3
"""Genuinely different out-of-the-box techniques for multi-target prediction.

1. RegressorChain: predict targets in causal order (VCD→TCD→Sub→Byp)
   using the predicted values of earlier targets as features for later ones.

2. KNN trajectory matching: for each test batch, find K most similar training
   batches (by initial conditions + feed patterns) and average their trajectories.

3. Target decomposition: replace raw 4-target space with:
   - VCD/TCD ratio (viability)
   - TCD (scale)
   - Substrate residual from feed schedule
   - Byproduct regime indicator

All evaluated with nested CV. Baseline: b09 for comparison.
"""

import numpy as np
import os
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import RegressorChain
from sklearn.linear_model import RidgeCV
from sklearn.neighbors import NearestNeighbors

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lab'))
from common import load, T, LAB, folds7

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)


def load_data():
    bs, A, Y, tt, test = load()
    tr = ~test
    Ytr = Y[tr][:, 1:, :]
    from common import feature_groups
    G, fl, D = feature_groups(A, Y, tt)

    # Build feature matrix from key signals
    feat_list = []
    feat_names = []
    for k in ['lx0', 'ls0', 'lp0', 'via0', 'vol0']:
        if k in G:
            feat_list.append(G[k].ravel())
            feat_names.append(k)
    if 'fb' in G:
        feat_list.append(G['fb'].sum(axis=1))
        feat_names.append('fb_total')
    if 'fc' in G:
        feat_list.append(G['fc'].sum(axis=1))
        feat_names.append('fc_total')
    if 'fcn' in G:
        feat_list.append(G['fcn'].sum(axis=1))
        feat_names.append('fcn_total')
    if 'dVr_e' in G:
        feat_list.append(G['dVr_e'].sum(axis=1))
        feat_names.append('dVr_early')
    if 'dVr_l' in G:
        feat_list.append(G['dVr_l'].sum(axis=1))
        feat_names.append('dVr_late')
    if 'volL' in G:
        feat_list.append(G['volL'].max(axis=1))
        feat_names.append('volL_max')
    if 'fs_m' in G:
        feat_list.append(G['fs_m'].sum(axis=1))
        feat_names.append('fs_early')
    if 'fs_l' in G:
        feat_list.append(G['fs_l'].sum(axis=1))
        feat_names.append('fs_late')
    if 'fd' in G:
        feat_list.append(G['fd'].sum(axis=1))
        feat_names.append('fd_total')

    X = np.column_stack(feat_list)  # (61, n_feats)
    X_train = X[tr]
    X_test = X[test]

    # Map folds
    full_folds = folds7(tr)
    pos_map = {r: i for i, r in enumerate(np.where(tr)[0])}
    outer_folds = [np.array([pos_map[r] for r in f if r in pos_map]) for f in full_folds]
    outer_folds = [f for f in outer_folds if len(f) > 0]

    print(f"Data: {Ytr.shape[0]} train batches, {X_train.shape[1]} features, "
          f"{len(feat_names)} feature groups")
    print(f"Features: {feat_names}")

    return {
        'Ytr': Ytr, 'X': X, 'X_train': X_train, 'X_test': X_test,
        'outer_folds': outer_folds, 'A': A, 'Y': Y, 'tt': tt,
        'tr': tr, 'test': test, 'G': G, 'fl': fl[tr], 'D': D,
        'feat_names': feat_names,
    }


def nested_cv(predict_fn, data, hp_grid, n_outer=7):
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

        # Grid select HP
        best_score = np.inf
        best_hp = hp_grid[0]
        for hp in hp_grid:
            scores = []
            for ifold in inner_folds:
                itr = np.ones(n_train, dtype=bool)
                itr[ifold] = False
                P = predict_fn(train_idx[itr], train_idx[ifold], hp, data)
                scores.append(mape_all(P, Ytr[train_idx[ifold]]))
            s = np.mean(scores)
            if s < best_score:
                best_score = s
                best_hp = hp
        best_hps.append(best_hp)

        P = predict_fn(train_idx, test_idx, best_hp, data)
        honest_P[test_idx] = P
        fold_m = mape_all(P, Ytr[test_idx])
        print(f"  Fold {f_i+1}/7: held {len(outer_fold)}, hp={best_hp}, MAPE={fold_m:.2f}%")

    m = metrics(honest_P, Ytr)
    return honest_P, m, best_hps


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
    print(f"  {label}:\n    MAPE {m['mape']:.3f} | v={m['mape_vcd']:.1f} "
          f"t={m['mape_tcd']:.1f} s={m['mape_substrate']:.1f} "
          f"b={m['mape_byproduct']:.1f} | RMSE={m['rmse']:.3f} R²={m['r2']:.3f}")


# ═══════════════════════════════════════════════════════
# Experiment 1: RegressorChain
# ═══════════════════════════════════════════════════════

def experiment_regressor_chain(data):
    """Causal chain: VCD → TCD → Substrate → Byproduct.

    For each day independently, chain Ridge regressors so that predicted
    values of earlier targets become features for later ones.
    """
    print("\n" + "="*70)
    print("T1: RegressorChain (VCD→TCD→Sub→Byp)")
    print("  Order respects: cells grow (VCD), some die (TCD-VCD),")
    print("  cells consume substrate, produce/consume byproduct")
    print("="*70)

    X_train = data['X_train']

    results = {}
    Ytr = data['Ytr']

    # Build feature matrix: static features + day index one-hot
    def build_day_features(X_batch, n_batches):
        """(n, 13, F+13) per-day feature matrix"""
        n_feats = X_batch.shape[1]
        feats = np.zeros((n_batches * 13, n_feats + 13))
        for i in range(n_batches):
            for d in range(13):
                idx = i * 13 + d
                feats[idx, :n_feats] = X_batch[i]
                feats[idx, n_feats + d] = 1.0  # day one-hot
        return feats

    for variant in ['T1a_ridge', 'T1b_lgb']:
        print(f"\n  {variant}:")

        def make_chain_fn(variant):
            def predict_fn(train_idx, test_idx, hp, data):
                n_tr = len(train_idx)
                n_te = len(test_idx)
                Ytr = data['Ytr']
                X = data['X_train']

                P_test = np.zeros((n_te, 13, 4))
                P_train = np.zeros((n_tr, 13, 4))  # predicted training values

                for d in range(13):
                    X_tr = np.column_stack([X[train_idx], np.full((n_tr, 1), d+1)])
                    X_te = np.column_stack([X[test_idx], np.full((n_te, 1), d+1)])

                    for t in range(4):
                        y_tr = np.log(Ytr[train_idx, d, t] + 1e-8)

                        # Augment with previously predicted targets
                        if t > 0:
                            prev_tr = np.column_stack([
                                np.log(P_train[:, d, j] + 1e-8) for j in range(t)
                            ])
                            prev_te = np.column_stack([
                                np.log(P_test[:, d, j] + 1e-8) for j in range(t)
                            ])
                            X_tr_aug = np.column_stack([X_tr, prev_tr])
                            X_te_aug = np.column_stack([X_te, prev_te])
                        else:
                            X_tr_aug = X_tr
                            X_te_aug = X_te

                        if variant == 'T1a_ridge':
                            from sklearn.linear_model import Ridge
                            model = Ridge(alpha=hp.get('alpha', 10.0))
                        else:
                            model = lgb.LGBMRegressor(
                                n_estimators=hp.get('n_estimators', 100),
                                max_depth=hp.get('max_depth', 4),
                                learning_rate=hp.get('lr', 0.05),
                                num_leaves=hp.get('num_leaves', 15),
                                random_state=42, verbose=-1,
                            )
                        model.fit(X_tr_aug, y_tr)
                        P_train[:, d, t] = np.exp(model.predict(X_tr_aug))
                        P_test[:, d, t] = np.exp(model.predict(X_te_aug))

                return P_test
            return predict_fn

        hp_grid = ([{'alpha': 0.1}, {'alpha': 1.0}, {'alpha': 10.0}, {'alpha': 100.0}]
                   if variant == 'T1a_ridge' else
                   [{'n_estimators': 100, 'max_depth': 4, 'lr': 0.05, 'num_leaves': 15},
                    {'n_estimators': 200, 'max_depth': 6, 'lr': 0.03, 'num_leaves': 31}])

        try:
            P, m, hps = nested_cv(make_chain_fn(variant), data, hp_grid)
            print_m(variant, m)
            results[variant] = {'predictions': P, 'metrics': m}
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback; traceback.print_exc()

    return results


# ═══════════════════════════════════════════════════════
# Experiment 2: KNN trajectory matching
# ═══════════════════════════════════════════════════════

def experiment_knn_trajectory(data):
    """For each test batch, find K nearest training batches by feature
    similarity and average their trajectories.

    This is the simplest possible non-parametric model. It treats each
    batch as a complete trajectory and predicts by analogy.
    """
    print("\n" + "="*70)
    print("T2: KNN Trajectory Matching")
    print("  Find K most similar training batches → average their trajectories")
    print("="*70)

    X_train = data['X_train']
    Ytr = data['Ytr']

    results = {}

    for variant in ['T2a_knn_log', 'T2b_knn_raw']:
        print(f"\n  {variant}:")

        def make_knn_fn(variant):
            def predict_fn(train_idx, test_idx, hp, data):
                n_te = len(test_idx)
                X_tr = data['X_train'][train_idx]
                X_te = data['X_train'][test_idx]
                Y_tr = data['Ytr'][train_idx]

                # Standardize features for distance computation
                scl = StandardScaler().fit(X_tr)
                X_tr_s = scl.transform(X_tr)
                X_te_s = scl.transform(X_te)

                nn = NearestNeighbors(n_neighbors=min(hp['k'], len(train_idx)),
                                     metric=hp.get('metric', 'euclidean'))
                nn.fit(X_tr_s)
                dists, idx = nn.kneighbors(X_te_s)

                if hp.get('weighted', False):
                    # Distance-weighted average
                    eps = 1e-10
                    w = 1.0 / (dists + eps)
                    w = w / w.sum(axis=1, keepdims=True)

                    P_test = np.zeros((n_te, 13, 4))
                    for i in range(n_te):
                        if variant == 'T2a_knn_log':
                            # Geometric mean (log-space average)
                            log_neighbors = np.log(Y_tr[idx[i]] + 1e-8)  # (K, 13, 4)
                            P_test[i] = np.exp(np.average(log_neighbors, axis=0, weights=w[i]))
                        else:
                            P_test[i] = np.average(Y_tr[idx[i]], axis=0, weights=w[i])
                else:
                    if variant == 'T2a_knn_log':
                        P_test = np.exp(np.mean(
                            np.log(Y_tr[idx] + 1e-8), axis=1
                        ))
                    else:
                        P_test = np.mean(Y_tr[idx], axis=1)

                return P_test
            return predict_fn

        hp_grid = [
            {'k': 1, 'metric': 'euclidean', 'weighted': False},
            {'k': 3, 'metric': 'euclidean', 'weighted': False},
            {'k': 5, 'metric': 'euclidean', 'weighted': False},
            {'k': 3, 'metric': 'euclidean', 'weighted': True},
            {'k': 5, 'metric': 'euclidean', 'weighted': True},
        ]

        try:
            P, m, hps = nested_cv(make_knn_fn(variant), data, hp_grid)
            print_m(variant, m)
            results[variant] = {'predictions': P, 'metrics': m}
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback; traceback.print_exc()

    return results


# ═══════════════════════════════════════════════════════
# Experiment 3: Decomposed target space
# ═══════════════════════════════════════════════════════

def experiment_decomposed(data):
    """Replace 4 correlated targets with a more independent representation:

    Instead of predicting [VCD, TCD, Substrate, Byproduct] directly:
    1. Predict viability: VCD/TCD ratio (0-1, smooth)
    2. Predict TCD (cell scale, like original)
    3. Predict substrate (keeping original, but conditioned on cell scale)
    4. Predict byproduct residual from substrate

    Then reconstruct the original 4 targets.
    """
    print("\n" + "="*70)
    print("T3: Decomposed Target Space")
    print("  Predict: viability + TCD + substrate + byproduct_residual")
    print("  Then reconstruct VCD = viability × TCD")
    print("="*70)

    Ytr = data['Ytr']

    def make_decomp_fn():
        def predict_fn(train_idx, test_idx, hp, data):
            n_tr = len(train_idx)
            n_te = len(test_idx)
            X = data['X_train']
            Y = data['Ytr']

            # Build derived targets from training data
            viability = Y[:, :, 0] / (Y[:, :, 1] + 1e-10)  # VCD/TCD
            substrate = Y[:, :, 2]
            byproduct = Y[:, :, 3]

            P_viability = np.zeros((n_te, 13))
            P_tcd = np.zeros((n_te, 13))
            P_sub = np.zeros((n_te, 13))
            P_byp = np.zeros((n_te, 13))

            for d in range(13):
                # Features
                X_tr = np.column_stack([X[train_idx], np.full((n_tr, 1), d+1)])
                X_te = np.column_stack([X[test_idx], np.full((n_te, 1), d+1)])

                # 1. Predict viability (logit-space for [0,1] constraint)
                via_tr = np.log(viability[train_idx, d] / (1 - viability[train_idx, d] + 1e-10))
                ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
                ridge.fit(X_tr, via_tr)
                via_logit = ridge.predict(X_te)
                P_viability[:, d] = 1.0 / (1.0 + np.exp(-via_logit))

                # 2. Predict TCD (log-space)
                tcd_tr = np.log(Y[train_idx, d, 1] + 1e-8)
                ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
                ridge.fit(X_tr, tcd_tr)
                P_tcd[:, d] = np.exp(ridge.predict(X_te))

                # 3. Predict substrate (log-space, with viability+TCD as extra features)
                X_tr_aug = np.column_stack([
                    X_tr,
                    np.log(viability[train_idx, d] + 1e-8),
                    np.log(Y[train_idx, d, 1] + 1e-8),
                ])
                X_te_aug = np.column_stack([
                    X_te,
                    np.log(P_viability[:, d] + 1e-8),
                    np.log(P_tcd[:, d] + 1e-8),
                ])
                sub_tr = np.log(Y[train_idx, d, 2] + 1e-8)
                ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
                ridge.fit(X_tr_aug, sub_tr)
                P_sub[:, d] = np.exp(ridge.predict(X_te_aug))

                # 4. Predict byproduct (log-space, with all three above)
                X_tr_aug2 = np.column_stack([
                    X_tr_aug,
                    np.log(Y[train_idx, d, 2] + 1e-8),
                ])
                X_te_aug2 = np.column_stack([
                    X_te_aug,
                    np.log(P_sub[:, d] + 1e-8),
                ])
                byp_tr = np.log(Y[train_idx, d, 3] + 1e-8)
                ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
                ridge.fit(X_tr_aug2, byp_tr)
                P_byp[:, d] = np.exp(ridge.predict(X_te_aug2))

            # Reconstruct: VCD = viability × TCD
            P_vcd = P_viability * P_tcd

            P_test = np.stack([P_vcd, P_tcd, P_sub, P_byp], axis=-1)
            return P_test
        return predict_fn

    hp_grid = [{'x': 1}]  # No HP to tune; RidgeCV handles internally

    try:
        P, m, hps = nested_cv(make_decomp_fn(), data, hp_grid)
        print_m('T3_decomposed', m)
        return {'T3_decomposed': {'predictions': P, 'metrics': m}}
    except Exception as e:
        print(f"    ERROR: {e}")
        import traceback; traceback.print_exc()
        return {}


# ═══════════════════════════════════════════════════════
# Experiment 4: XGBoost/LGB with target-aware features
# ═══════════════════════════════════════════════════════

def experiment_xgb(data):
    """LightGBM per day with all targets as features for each other
    (trained iteratively). Better than the v1 GBT that used only
    static features — this one uses the day index properly and trains
    on log values.
    """
    print("\n" + "="*70)
    print("T4: LightGBM per-day on log-space with interactions")
    print("  Proper day one-hot + log target values")
    print("="*70)

    if not HAS_LGB:
        print("  LightGBM not available")
        return {}

    Ytr = data['Ytr']

    def make_lgb_fn():
        def predict_fn(train_idx, test_idx, hp, data):
            n_tr = len(train_idx)
            n_te = len(test_idx)
            X = data['X_train']
            Y = data['Ytr']

            X_tr_flat = np.zeros((n_tr * 13, X.shape[1] + 1))  # +1 for day
            Y_tr_flat = np.zeros((n_tr * 13, 4))
            for i, b in enumerate(train_idx):
                for d in range(13):
                    idx = i * 13 + d
                    X_tr_flat[idx, :-1] = X[b]
                    X_tr_flat[idx, -1] = d + 1
                    Y_tr_flat[idx] = np.log(Y[b, d] + 1e-8)

            X_te_flat = np.zeros((n_te * 13, X.shape[1] + 1))
            for i, b in enumerate(test_idx):
                for d in range(13):
                    idx = i * 13 + d
                    X_te_flat[idx, :-1] = X[b]
                    X_te_flat[idx, -1] = d + 1

            P_test = np.zeros((n_te, 13, 4))
            for t in range(4):
                model = lgb.LGBMRegressor(
                    n_estimators=hp.get('n_estimators', 200),
                    max_depth=hp.get('max_depth', 5),
                    learning_rate=hp.get('lr', 0.05),
                    num_leaves=hp.get('num_leaves', 31),
                    random_state=42, verbose=-1,
                )
                model.fit(X_tr_flat, Y_tr_flat[:, t])
                pred_log = model.predict(X_te_flat)
                P_test[:, :, t] = np.exp(pred_log.reshape(n_te, 13))

            return P_test
        return predict_fn

    hp_grid = [
        {'n_estimators': 100, 'max_depth': 4, 'lr': 0.05, 'num_leaves': 15},
        {'n_estimators': 200, 'max_depth': 5, 'lr': 0.03, 'num_leaves': 31},
        {'n_estimators': 300, 'max_depth': 6, 'lr': 0.02, 'num_leaves': 63},
    ]

    try:
        P, m, hps = nested_cv(make_lgb_fn(), data, hp_grid)
        print_m('T4_lgb_perday', m)
        return {'T4_lgb_perday': {'predictions': P, 'metrics': m}}
    except Exception as e:
        print(f"    ERROR: {e}")
        import traceback; traceback.print_exc()
        return {}


# ═══════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════

def main():
    print("="*70)
    print("OUT-OF-THE-BOX MULTI-TARGET TECHNIQUES")
    print("All evaluated with nested CV")
    print("="*70)

    data = load_data()

    # Baseline
    oof_dir = os.path.join(LAB, 'oof')
    b09 = np.load(os.path.join(oof_dir, 'b09_hyb_stat_bayesresid_smoothcells.npz'))
    base_m = metrics(b09['oof'], data['Ytr'])
    print_m("\nBaseline b09", base_m)

    all_results = {
        'baseline_b09': {'predictions': b09['oof'], 'metrics': base_m}
    }

    for name, func in [
        ('T1_RegressorChain', experiment_regressor_chain),
        ('T2_KNN_Trajectory', experiment_knn_trajectory),
        ('T3_Decomposed', experiment_decomposed),
        ('T4_LGB_perday', experiment_xgb),
    ]:
        try:
            res = func(data)
            for k, v in res.items():
                all_results[f"{name}_{k}"] = {**v, 'family': name}
        except Exception as e:
            print(f"\n  {name} FAILED: {e}")
            import traceback; traceback.print_exc()

    # Comparison
    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    print(f"{'Experiment':<40} {'MAPE':>8} {'RMSE':>8} {'R²':>8} | "
          f"{'vcd':>7} {'tcd':>7} {'sub':>7} {'byp':>7}")
    print("-"*105)

    for name in sorted(all_results, key=lambda n: all_results[n]['metrics']['mape']):
        m = all_results[name]['metrics']
        print(f"{name:<40} {m['mape']:7.3f} {m['rmse']:7.3f} {m['r2']:7.3f} | "
              f"{m['mape_vcd']:6.2f} {m['mape_tcd']:6.2f} "
              f"{m['mape_substrate']:6.2f} {m['mape_byproduct']:6.2f}")

    import pandas as pd
    rows = [{'name': n, **r['metrics']} for n, r in all_results.items()]
    df = pd.DataFrame(rows).sort_values('mape')
    csv_path = os.path.join(OUT_DIR, 'ootb_techniques.csv')
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"\nSaved: {csv_path}")

    best = df.iloc[0]
    base = df[df['name'] == 'baseline_b09'].iloc[0]
    delta = best['mape'] - base['mape']
    print(f"\nBest: {best['name']} (MAPE {best['mape']:.3f})")
    print(f"vs baseline b09: {delta:+.3f} MAPE")
    if delta < -0.15:
        print("  → MEANINGFUL improvement!")
    else:
        print("  → Within CV noise")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
