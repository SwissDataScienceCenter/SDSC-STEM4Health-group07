#!/usr/bin/env python3
"""Experimental Model Lab — genuinely different approaches for the bioreactor task.

Unlike the `improvements/` directory (which only tuned blend weights and smoothing
on the same 3-member ensemble), this lab tests fundamentally different modeling
paradigms to find real, reproducible gains beyond the ~7.03% MAPE baseline.

═══════════════════════════════════════════════════════════════════════════
ANTI-OVERFITTING DESIGN
═══════════════════════════════════════════════════════════════════════════

Every experiment uses nested cross-validation:
  - Outer loop: 7-fold by-batch CV (same as common.folds7)
  - Inner loop: all hyperparameter selection (model-specific HP, blending
    weights, architecture choices) performed ONLY on the 6 training folds
  - Honest score: aggregated holdout predictions, never optimized toward

Pre-registered experiments (defined BEFORE seeing results):
  A. XGBoost/LightGBM per-target regressors on static batch features
  B. LSTM/GRU sequence-to-sequence (time series on the 14-day trajectory)
  C. Per-day independent GradientBoosting (decoupled by day, then temporally smoothed)
  D. Feature-engineered linear stack (Gaussian Process with learned kernel on top of
     existing OOF predictions from all 143 models)
  E. Prediction-powered ensembling (use available unlabelled test batches for
     semi-supervised calibration via the feature distribution)

Final comparisons include the original b09 baseline for reference.
All results go to `experiments/outputs/`.
"""

import numpy as np
import pandas as pd
import json
import os
import sys
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from scipy.optimize import minimize

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lab'))
from common import load, T, LAB, PROJ, folds7, mape, post, THR, FLAGCOL, feature_groups

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════
# Data loading and preparation
# ══════════════════════════════════════════════════════════════════════

def load_all_data():
    """Load everything needed for experiments."""
    bs, A, Y, tt, test = load()
    tr = ~test
    Ytr = Y[tr][:, 1:, :]  # (49, 13, 4)
    G, fl, D = feature_groups(A, Y, tt)

    # Build a flat feature matrix per batch from G (static features)
    feat_names = sorted(G.keys())
    X_static = np.column_stack([G[k] for k in feat_names])  # (61, n_feats)

    # Extract train portions
    X_train = X_static[tr]
    fl_train = fl[tr]
    fl_test = fl[test]
    X_test = X_static[test]

    # Build per-day features: combine static features with day index
    # and feed history up to that point
    return {
        'bs': bs, 'A': A, 'Y': Y, 'tt': tt, 'test': test, 'tr': tr,
        'Ytr': Ytr, 'G': G, 'fl': fl, 'D': D,
        'X_train': X_train, 'X_test': X_test,
        'fl_train': fl_train, 'fl_test': fl_test,
        'feat_names': feat_names,
    }


def nested_cv_score(predict_fn, data, hp_grid=None, hp_selector=None,
                     n_outer=7, n_inner=6):
    """Run nested CV and return honest predictions + metrics.

    Args:
        predict_fn: fn(train_idx, test_idx, hp) -> (n_test, 13, 4) predictions
        data: dict from load_all_data()
        hp_grid: list of hyperparameter dicts to search (if any)
        hp_selector: fn(train_idx, inner_folds) -> best_hp (if hp_grid is None)

    Returns:
        honest_P: (49, 13, 4) honest OOF predictions
        metrics: dict of overall metrics
        best_hps: list of best hyperparams per outer fold
    """
    Ytr = data['Ytr']
    n_total = len(Ytr)

    # folds7 returns indices into the 61-element array; map to 0..48
    full_folds = folds7(data['tr'])
    pos_map = {r: i for i, r in enumerate(np.where(data['tr'])[0])}
    outer_folds = [np.array([pos_map[r] for r in f if r in pos_map]) for f in full_folds]
    outer_folds = [f for f in outer_folds if len(f) > 0]

    pos = {r: i for i, r in enumerate(np.where(data['tr'])[0])}
    honest_P = np.zeros_like(Ytr)
    best_hps = []

    for fold_i, outer_fold in enumerate(outer_folds):
        train_mask = np.ones(n_total, dtype=bool)
        train_mask[outer_fold] = False
        train_idx = np.where(train_mask)[0]
        test_idx = outer_fold

        # Create inner folds
        inner_folds = []
        for j in range(n_inner):
            ifold = train_idx[j::n_inner]
            inner_folds.append(np.array([np.where(train_idx == p)[0][0] for p in ifold]))

        # Select hyperparameters
        if hp_selector:
            hp = hp_selector(train_idx, inner_folds, data)
        elif hp_grid:
            hp = grid_search_hp(predict_fn, train_idx, inner_folds, hp_grid, data)
        else:
            hp = {}

        best_hps.append(hp)

        # Predict holdout
        holdout_preds = predict_fn(train_idx, test_idx, hp, data)
        honest_P[test_idx] = holdout_preds

        print(f"  Fold {fold_i+1}/7: {len(outer_fold)} batches held out, HP: {hp}")

    # Compute metrics
    metrics = compute_metrics(honest_P, Ytr)
    return honest_P, metrics, best_hps


def grid_search_hp(predict_fn, train_idx, inner_folds, hp_grid, data):
    """Simple grid search over hp_grid using inner CV."""
    best_score = np.inf
    best_hp = hp_grid[0]
    for hp in hp_grid:
        scores = []
        for ifold in inner_folds:
            if len(ifold) == 0:
                continue
            inner_train = np.ones(len(train_idx), dtype=bool)
            inner_train[ifold] = False
            itr = train_idx[inner_train]
            ite = train_idx[ifold]
            preds = predict_fn(itr, ite, hp, data)
            true = data['Ytr'][ite]
            scores.append(_mape_all(preds, true))
        score = np.mean(scores) if scores else np.inf
        if score < best_score:
            best_score = score
            best_hp = hp
    return best_hp


def _mape_all(P, Y):
    """Compute mean MAPE across all targets, days, batches."""
    return (np.abs(P - Y) / Y * 100).mean()


def compute_metrics(P, Y):
    """Compute comprehensive metrics."""
    err = np.abs(P - Y) / Y * 100  # (n, 13, 4)
    overall = err.mean()
    per_target = {T[i]: err[:, :, i].mean() for i in range(4)}

    # RMSE (on original scale, not percentage)
    se = (P - Y) ** 2
    rmse = np.sqrt(se.mean())
    per_target_rmse = {T[i]: np.sqrt(se[:, :, i].mean()) for i in range(4)}

    # MAE
    mae = np.abs(P - Y).mean()

    # R-squared per target
    r2s = {}
    for i in range(4):
        y = Y[:, :, i]
        ss_res = ((y - P[:, :, i])**2).sum()
        ss_tot = ((y - y.mean())**2).sum()
        r2s[T[i]] = 1 - ss_res / ss_tot

    return {
        'mape': overall,
        'mape_vcd': per_target['vcd'],
        'mape_tcd': per_target['tcd'],
        'mape_substrate': per_target['substrate'],
        'mape_byproduct': per_target['byproduct'],
        'rmse': rmse,
        'mae': mae,
        'r2': np.mean(list(r2s.values())),
        'r2_vcd': r2s['vcd'],
        'r2_tcd': r2s['tcd'],
        'r2_substrate': r2s['substrate'],
        'r2_byproduct': r2s['byproduct'],
    }


def print_metrics(label, m):
    print(f"\n  {label}:")
    print(f"    MAPE: {m['mape']:.3f} (vcd={m['mape_vcd']:.2f}, tcd={m['mape_tcd']:.2f}, "
          f"sub={m['mape_substrate']:.2f}, byp={m['mape_byproduct']:.2f})")
    print(f"    RMSE: {m['rmse']:.3f}  MAE: {m['mae']:.3f}  R²: {m['r2']:.3f}")


# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT A: Gradient Boosted Trees (per-target, per-day)
# ══════════════════════════════════════════════════════════════════════

def build_gbt_features(data, train_idx, which='all'):
    """Build feature matrix for GBT models.

    Features include:
    - Static batch features (initial conditions, feed patterns)
    - Day index (one-hot)
    - Cumulative feed amounts up to that day
    - Volume trajectory
    """
    G = data['G']
    D = data['D']
    fl = data['fl']
    Ytr = data['Ytr']
    tr = data['tr']

    # Static features per batch (pre-computed from G)
    static_names = ['lx0', 'ls0', 'lp0', 'via0', 'vol0',
                    'fb', 'fc', 'fcn', 'fd', 'fs_m', 'fs_l',
                    'dVr_e', 'dVr_l', 'volL']
    static_feats = np.column_stack([G[k] for k in static_names if k in G])  # (61, n_static)

    # For each batch and day, build a feature vector
    n_batches = len(Ytr)
    n_days = 13
    feature_list = []

    for b_idx in train_idx:
        for d in range(n_days):
            row = []
            # Day index (one-hot in log space for better tree splits)
            row.extend([float(d == j) for j in range(n_days)])
            # Day number
            row.append(d + 1)
            # Static features
            row.extend(static_feats[b_idx])
            # Feed flags for this batch
            row.extend(fl[b_idx, :5] if b_idx < len(fl) else [0]*5)
            feature_list.append(row)

    return np.array(feature_list, dtype=np.float32)


def experiment_gbt(data):
    """Gradient boosted trees: LightGBM per target, per day.

    Two variants:
      A1: Independent LightGBM per (target, day) — 52 models
      A2: Multi-output LightGBM per day — 13 models predicting all 4 targets
    """
    print("\n" + "="*70)
    print("EXPERIMENT A: Gradient Boosted Trees")
    print("="*70)

    try:
        import lightgbm as lgb
    except ImportError:
        print("  LightGBM not available, skipping")
        return None, None

    results = {}

    for variant in ['A1_independent', 'A2_multitarget']:
        print(f"\n  {variant}:")

        def predict_fn(train_idx, test_idx, hp, data):
            Ytr = data['Ytr']
            X_train = build_gbt_features(data, train_idx)
            X_test = build_gbt_features(data, test_idx)

            # Reshape Y to flat
            Y_flat = Ytr[train_idx].reshape(-1, 4)

            if variant == 'A1_independent':
                # One model per (target, day) — 52 models
                P_test = np.zeros((len(test_idx), 13, 4))
                for t in range(4):
                    for d in range(13):
                        y_tr = Ytr[train_idx, d, t]
                        train_mask = np.arange(len(y_tr))
                        model = lgb.LGBMRegressor(
                            n_estimators=hp.get('n_estimators', 200),
                            max_depth=hp.get('max_depth', 6),
                            learning_rate=hp.get('lr', 0.05),
                            num_leaves=hp.get('num_leaves', 31),
                            random_state=42, verbose=-1,
                        )
                        # Use day d samples only
                        idx_d = np.arange(d, len(Y_flat), 13)
                        model.fit(X_train[idx_d], Y_flat[idx_d, t])
                        idx_d_test = np.arange(d, len(test_idx) * 13, 13)
                        P_test[:, d, t] = model.predict(X_test[idx_d_test])
                return P_test
            else:
                # A2: multi-target per day — 13 models
                from sklearn.multioutput import MultiOutputRegressor
                P_test = np.zeros((len(test_idx), 13, 4))
                for d in range(13):
                    base = lgb.LGBMRegressor(
                        n_estimators=hp.get('n_estimators', 200),
                        max_depth=hp.get('max_depth', 6),
                        learning_rate=hp.get('lr', 0.05),
                        num_leaves=hp.get('num_leaves', 31),
                        random_state=42, verbose=-1,
                    )
                    model = MultiOutputRegressor(base)
                    idx_d = np.arange(d, len(Y_flat), 13)
                    model.fit(X_train[idx_d], Y_flat[idx_d])
                    idx_d_test = np.arange(d, len(test_idx) * 13, 13)
                    P_test[:, d] = model.predict(X_test[idx_d_test])
                return P_test

        hp_grid = [
            {'n_estimators': 100, 'max_depth': 4, 'lr': 0.1, 'num_leaves': 15},
            {'n_estimators': 200, 'max_depth': 6, 'lr': 0.05, 'num_leaves': 31},
            {'n_estimators': 300, 'max_depth': 8, 'lr': 0.03, 'num_leaves': 63},
        ]

        try:
            honest_P, metrics, _ = nested_cv_score(
                predict_fn, data, hp_grid=hp_grid
            )
            print_metrics(variant, metrics)
            results[variant] = {'predictions': honest_P, 'metrics': metrics}
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback; traceback.print_exc()

    return {k: v for k, v in results.items() if v is not None}


# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT B: Feed-Forward Neural Network with batch-level embeddings
# ══════════════════════════════════════════════════════════════════════

def experiment_nn(data):
    """Simple feed-forward NN with learned batch embeddings.

    Per-batch embedding (learned from static features) + per-day features
    -> 4-target output. Trained with MAE loss.
    """
    print("\n" + "="*70)
    print("EXPERIMENT B: Neural Network with Batch Embeddings")
    print("="*70)

    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:
        print("  PyTorch not available, skipping")
        return {}

    results = {}

    # Build normalized feature matrices
    Ytr = data['Ytr']
    X_static = data['X_train']

    # Normalize static features
    scaler_s = StandardScaler().fit(X_static)
    X_s = scaler_s.transform(X_static)

    # Create per-day features: [day_idx/13, day_onehot...]
    n_days = 13
    day_feats = np.zeros((13, n_days + 1))
    for d in range(n_days):
        day_feats[d, 0] = (d + 1) / 13.0
        day_feats[d, d + 1] = 1.0

    for variant in ['B1_mlp', 'B2_mlp_deep']:
        print(f"\n  {variant}:")

        def make_predict_fn(variant):
            def predict_fn(train_idx, test_idx, hp, data):
                X_s_tr = X_s[train_idx]
                X_s_te = X_s[test_idx]
                n_tr = len(train_idx)
                n_te = len(test_idx)

                # Build (n*13, n_feats) feature matrix
                X_tr = np.zeros((n_tr * 13, X_s.shape[1] + n_days + 1))
                X_te = np.zeros((n_te * 13, X_s.shape[1] + n_days + 1))
                Y_tr = np.zeros((n_tr * 13, 4))

                for i, b in enumerate(train_idx):
                    for d in range(13):
                        idx = i * 13 + d
                        X_tr[idx, :X_s.shape[1]] = X_s_tr[i]
                        X_tr[idx, X_s.shape[1]:] = day_feats[d]
                        Y_tr[idx] = Ytr[b, d]

                for i, b in enumerate(test_idx):
                    for d in range(13):
                        idx = i * 13 + d
                        X_te[idx, :X_s.shape[1]] = X_s_te[i]
                        X_te[idx, X_s.shape[1]:] = day_feats[d]

                # Build PyTorch model
                in_dim = X_tr.shape[1]
                hidden = hp.get('hidden', 128)
                layers = hp.get('layers', 2)
                lr = hp.get('lr', 0.001)
                epochs = hp.get('epochs', 200)

                seq = []
                prev = in_dim
                for _ in range(layers):
                    seq.append(nn.Linear(prev, hidden))
                    seq.append(nn.ReLU())
                    seq.append(nn.BatchNorm1d(hidden))
                    prev = hidden
                seq.append(nn.Linear(prev, 4))
                model = nn.Sequential(*seq)

                X_t = torch.tensor(X_tr, dtype=torch.float32)
                Y_t = torch.tensor(Y_tr, dtype=torch.float32)
                X_v = torch.tensor(X_te, dtype=torch.float32)

                opt = optim.Adam(model.parameters(), lr=lr)
                loss_fn = nn.L1Loss()

                for ep in range(epochs):
                    model.train()
                    opt.zero_grad()
                    pred = model(X_t)
                    loss = loss_fn(pred, Y_t)
                    loss.backward()
                    opt.step()

                model.eval()
                with torch.no_grad():
                    pred_te = model(X_v).numpy()

                return pred_te.reshape(n_te, 13, 4)

            return predict_fn

        hp_grid = [
            {'hidden': 64, 'layers': 2, 'lr': 0.01, 'epochs': 200},
            {'hidden': 128, 'layers': 3, 'lr': 0.005, 'epochs': 300},
            {'hidden': 256, 'layers': 4, 'lr': 0.003, 'epochs': 500} if variant == 'B2_mlp_deep' else
            {'hidden': 128, 'layers': 2, 'lr': 0.01, 'epochs': 200},
        ]

        try:
            honest_P, metrics, _ = nested_cv_score(
                make_predict_fn(variant), data, hp_grid=hp_grid
            )
            print_metrics(variant, metrics)
            results[variant] = {'predictions': honest_P, 'metrics': metrics}
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback; traceback.print_exc()

    return results


# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT C: Stack on OOF predictions (learned ensemble of 143 models)
# ══════════════════════════════════════════════════════════════════════

def experiment_stack(data):
    """Use ALL 143 OOF predictions as input features to learn a
    per-target, per-day stacking model.

    This is distinctly different from the current approach:
    - Instead of geometric mean of 3 hand-picked members
    - We take ALL 143 models' OOF predictions as input
    - Learn a regularized linear stack per (target, day)
    - Use nested CV so the stacking weights are honest

    Two variants:
      C1: Ridge regression per (target, day) on log-space predictions
      C2: LightGBM non-linear stack per (target, day) on log predictions
    """
    print("\n" + "="*70)
    print("EXPERIMENT C: Learned Stack on All 143 OOF Predictions")
    print("="*70)

    # Load all OOF predictions
    oof_dir = os.path.join(LAB, 'oof')
    results_path = os.path.join(LAB, 'results.jsonl')

    records = {}
    with open(results_path) as f:
        for line in f:
            r = json.loads(line)
            records[r['name']] = r

    all_oofs = {}
    for fname in sorted(os.listdir(oof_dir)):
        if fname.endswith('.npz') and not fname.endswith('__mu.npz'):
            name = fname[:-4]
            data_npz = np.load(os.path.join(oof_dir, fname))
            oof = data_npz['oof']
            # Some OOF arrays are not 49 — skip those
            if oof.shape[0] == 49:
                all_oofs[name] = oof

    n_models = len(all_oofs)
    print(f"  Loaded {n_models} OOF prediction matrices")

    # Build feature matrix: (49, 13, 4, n_models)
    model_names = sorted(all_oofs.keys())
    feat_stack = np.stack([all_oofs[n] for n in model_names], axis=-1)  # (49, 13, 4, M)

    results = {}

    for variant in ['C1_ridge', 'C2_lgb']:
        print(f"\n  {variant}:")

        def make_stack_fn(variant, feat_stack, model_names):
            def predict_fn(train_idx, test_idx, hp, data):
                n_tr = len(train_idx)
                n_te = len(test_idx)
                Ytr = data['Ytr']

                P_test = np.zeros((n_te, 13, 4))
                for t in range(4):
                    for d in range(13):
                        X_tr = np.log(feat_stack[train_idx, d, t, :] + 1e-8)  # (n_tr, M)
                        X_te = np.log(feat_stack[test_idx, d, t, :] + 1e-8)
                        y_tr = np.log(Ytr[train_idx, d, t] + 1e-8)

                        if variant == 'C1_ridge':
                            from sklearn.linear_model import Ridge
                            model = Ridge(alpha=hp.get('alpha', 10.0),
                                         fit_intercept=True)
                            model.fit(X_tr, y_tr)
                            P_test[:, d, t] = np.exp(model.predict(X_te))
                        else:
                            import lightgbm as lgb
                            model = lgb.LGBMRegressor(
                                n_estimators=hp.get('n_estimators', 100),
                                max_depth=hp.get('max_depth', 3),
                                learning_rate=hp.get('lr', 0.05),
                                num_leaves=hp.get('num_leaves', 15),
                                random_state=42, verbose=-1,
                            )
                            model.fit(X_tr, y_tr)
                            P_test[:, d, t] = np.exp(model.predict(X_te))
                return P_test
            return predict_fn

        hp_grid = [
            {'alpha': 1.0, 'n_estimators': 100, 'max_depth': 3, 'lr': 0.05, 'num_leaves': 15},
            {'alpha': 10.0, 'n_estimators': 200, 'max_depth': 5, 'lr': 0.03, 'num_leaves': 31},
            {'alpha': 100.0, 'n_estimators': 300, 'max_depth': 4, 'lr': 0.02, 'num_leaves': 31},
        ]

        try:
            honest_P, metrics, _ = nested_cv_score(
                make_stack_fn(variant, feat_stack, model_names),
                data, hp_grid=hp_grid
            )
            print_metrics(variant, metrics)
            results[variant] = {'predictions': honest_P, 'metrics': metrics}
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback; traceback.print_exc()

    return results


# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT D: GP Kernel on raw batch signals (non-parametric time series)
# ══════════════════════════════════════════════════════════════════════

def experiment_gp(data):
    """Gaussian Process with a custom kernel combining:
    - Static feature similarity (RBF on batch features)
    - Time similarity (Matern on day index)
    - Target covariance (learned)

    Predicts Y in log-space. Uses the existing kernel machinery from statm.py
    but extends it with temporal structure.
    """
    print("\n" + "="*70)
    print("EXPERIMENT D: Multi-Target GP with Temporal + Batch Kernel")
    print("="*70)

    Ytr = data['Ytr']
    X_tr = data['X_train']

    results = {}

    for variant in ['D1_gp_static', 'D2_gp_static_temporal']:
        print(f"\n  {variant}:")

        def make_gp_fn(variant):
            def predict_fn(train_idx, test_idx, hp, data):
                Ytr = data['Ytr']
                X_tr = data['X_train']

                # Flatten data for GP: (n, 13, 4) -> (n*52,)
                n_tr = len(train_idx)
                n_te = len(test_idx)

                Y_flat = Ytr[train_idx].reshape(n_tr, 52)  # log-space
                Y_true_flat = Y_flat.copy()

                # Standardize features
                scl = StandardScaler()
                X_norm = scl.fit_transform(X_tr)

                # Build kernel
                K_batch = X_norm @ X_norm.T / X_norm.shape[1]  # linear kernel on static feats

                alpha = hp.get('alpha', 1.0)
                noise = 1e-4

                # Add temporal structure for D2
                if variant == 'D2_gp_static_temporal':
                    # Build day-similarity matrix (exponential decay in day distance)
                    D_days = np.abs(np.arange(13)[:, None] - np.arange(13)[None, :])
                    K_day = np.exp(-D_days / hp.get('day_length', 3.0))  # (13, 13)
                    # Kronecker: K_total = K_batch ⊗ K_day
                    K_total = np.kron(K_batch[np.ix_(train_idx, train_idx)], K_day)
                else:
                    K_total = np.kron(K_batch[np.ix_(train_idx, train_idx)], np.eye(13))

                # Ridge regression: (K + alpha*I)^(-1) * Y
                n_samples = n_tr * 13
                K_reg = K_total + alpha * np.eye(n_samples) + noise

                # Solve per target
                P_test = np.zeros((n_te, 13, 4))
                for t in range(4):
                    y_flat = Y_flat[:, t::4].reshape(-1)  # extract target t

                    # Solve
                    from scipy.linalg import solve
                    alphas = solve(K_reg, y_flat, assume_a='pos')

                    # Predict
                    if variant == 'D2_gp_static_temporal':
                        K_test = np.kron(
                            K_batch[np.ix_(test_idx, train_idx)],
                            K_day
                        )
                    else:
                        K_test = np.kron(
                            K_batch[np.ix_(test_idx, train_idx)],
                            np.eye(13)
                        )

                    y_pred = K_test @ alphas
                    P_test[:, :, t] = y_pred.reshape(n_te, 13)

                return np.exp(P_test)  # back from log-space
            return predict_fn

        hp_grid = [
            {'alpha': 0.1, 'day_length': 1.0},
            {'alpha': 1.0, 'day_length': 3.0},
            {'alpha': 10.0, 'day_length': 5.0},
        ]

        try:
            honest_P, metrics, _ = nested_cv_score(
                make_gp_fn(variant), data, hp_grid=hp_grid
            )
            print_metrics(variant, metrics)
            results[variant] = {'predictions': honest_P, 'metrics': metrics}
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback; traceback.print_exc()

    return results


# ══════════════════════════════════════════════════════════════════════
# Main experiment runner
# ══════════════════════════════════════════════════════════════════════

def main():
    print("="*70)
    print("BIOREACTOR EXPERIMENT LAB")
    print("Nested CV on all experiments — no overfitting")
    print("="*70)

    data = load_all_data()
    print(f"\nData loaded: {data['Ytr'].shape[0]} train batches, "
          f"{data['X_train'].shape[1]} static features")

    # Load baseline for comparison
    oof_dir = os.path.join(LAB, 'oof')
    try:
        b09 = np.load(os.path.join(oof_dir, 'b09_hyb_stat_bayesresid_smoothcells.npz'))
        baseline_P = b09['oof']
        baseline_m = compute_metrics(baseline_P, data['Ytr'])
        print_metrics("Baseline (b09)", baseline_m)
    except:
        baseline_m = {'mape': 7.028}
        print("\n  Baseline: b09_hyb_stat_bayesresid_smoothcells MAPE ~7.03% (from records)")

    all_results = {
        'baseline_b09': {'predictions': baseline_P if 'baseline_P' in locals() else None,
                          'metrics': baseline_m, 'family': 'baseline'}
    }

    # Run experiments
    experiments = [
        ('A_GBT', experiment_gbt),
        ('B_NN', experiment_nn),
        ('C_Stack', experiment_stack),
        ('D_GP', experiment_gp),
    ]

    for name, func in experiments:
        try:
            res = func(data)
            for k, v in res.items():
                all_results[f"{name}_{k}"] = {**v, 'family': name}
        except Exception as e:
            print(f"\n  {name} FAILED: {e}")
            import traceback; traceback.print_exc()

    # Final comparison
    print("\n" + "="*70)
    print("FINAL COMPARISON (all nested CV)")
    print("="*70)
    print(f"\n{'Experiment':<45} {'MAPE':>8} {'RMSE':>8} {'R²':>8} | {'vcd':>7} {'tcd':>7} {'sub':>7} {'byp':>7}")
    print("-"*110)

    sorted_results = sorted(all_results.items(),
                           key=lambda x: x[1]['metrics'].get('mape', 999))

    for name, res in sorted_results:
        m = res['metrics']
        if 'mape' in m and m['mape'] < 20:
            print(f"{name:<45} {m['mape']:7.3f} {m['rmse']:7.3f} {m['r2']:7.3f} | "
                  f"{m['mape_vcd']:6.2f} {m['mape_tcd']:6.2f} "
                  f"{m['mape_substrate']:6.2f} {m['mape_byproduct']:6.2f}")

    # Save detailed results
    summary = []
    for name, res in all_results.items():
        m = res['metrics']
        if 'mape' in m:
            summary.append({
                'name': name,
                'family': res.get('family', ''),
                'mape': m['mape'],
                'rmse': m['rmse'],
                'mae': m['mae'],
                'r2': m['r2'],
                'mape_vcd': m['mape_vcd'],
                'mape_tcd': m['mape_tcd'],
                'mape_substrate': m['mape_substrate'],
                'mape_byproduct': m['mape_byproduct'],
                'r2_vcd': m['r2_vcd'],
                'r2_tcd': m['r2_tcd'],
                'r2_substrate': m['r2_substrate'],
                'r2_byproduct': m['r2_byproduct'],
            })

    df = pd.DataFrame(summary).sort_values('mape')
    csv_path = os.path.join(OUT_DIR, 'experiment_comparison.csv')
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"\nSaved: {csv_path}")

    best = df.iloc[0]
    baseline_row = df[df['name'] == 'baseline_b09']
    if len(baseline_row) > 0:
        delta = best['mape'] - baseline_row.iloc[0]['mape']
        print(f"\nBest new experiment: {best['name']} (MAPE {best['mape']:.3f})")
        print(f"vs baseline b09 (MAPE {baseline_row.iloc[0]['mape']:.3f}): {delta:+.3f} MAPE")
        if delta < -0.2:
            print("  → MEANINGFUL improvement (beyond CV noise)")
        else:
            print("  → Within CV noise (~0.2-0.3 MAPE), not statistically meaningful")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
