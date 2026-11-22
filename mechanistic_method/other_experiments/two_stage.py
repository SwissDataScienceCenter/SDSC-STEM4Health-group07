"""Two-stage substrate prediction.

The bottleneck for substrate MAPE is days 11-13, where values drop below
~1 g/L and the bolus feed rule creates a discontinuity. This module splits
substrate prediction into two stages:

Stage 1: Classify whether a bolus will happen on days 11-12
Stage 2: Use separate regression models for bolus vs. no-bolus regimes

The approach uses the ensemble members' substrate predictions as features
and the known feed flags (derived from feed logs) as labels.
"""

import numpy as np


def extract_substrate_features(members_oof, Y_true, fl):
    """Extract features and labels for the two-stage substrate model.

    Features: each member's substrate prediction for days 11, 12, 13
              + the day 10 substrate prediction (latest before the bolus region)
    Labels: binary bolus indicator for each batch (from feed flags)

    Args:
        members_oof: list of (n,13,4) OOF arrays
        Y_true: (n,13,4) true values
        fl: (n, 5) feed flags [low7-9 x3, low11, low12]

    Returns:
        X: (n, n_features) feature matrix
        y: (n,) binary labels (1 = bolus on day 11 or 12)
    """
    n_batches = len(Y_true)
    features = []

    # Day 10 substrate prediction from each member (index 9, target 2)
    for oof in members_oof:
        features.append(oof[:, 9, 2:3])  # (n, 1)

    # Day 11, 12, 13 substrate predictions from each member
    for oof in members_oof:
        features.append(oof[:, 10:13, 2])  # (n, 3)

    # Day 10 true substrate
    features.append(Y_true[:, 9, 2:3])

    # Day 10-11 substrate delta from each member
    for oof in members_oof:
        delta = oof[:, 9, 2] - oof[:, 10, 2]
        features.append(delta[:, None])

    X = np.column_stack(features)

    # Label: low11 or low12 flag (bolus happened)
    # fl[:, 3] = low11, fl[:, 4] = low12
    y = (fl[:, 3] > 0).astype(int)  # use low11 as indicator

    return X, y


def fit_two_stage_predictor(members_oof, Y_true, fl):
    """Fit a two-stage substrate predictor.

    Stage 1: Simple threshold-based classifier on the ensemble's day-10
             substrate prediction.
    Stage 2: Weighted blend specific to each regime.

    Returns a function predict(X_features) -> (n, 3) substrate predictions
    for days 11-13.
    """
    # Stage 1: classify bolus regime
    # Use the winner member's day-10 substrate prediction
    day10_pred = np.mean([o[:, 9, 2] for o in members_oof], axis=0)
    y = (fl[:, 3] > 0).astype(int)

    # Find best split threshold
    best_acc = 0
    best_thresh = 1.5
    for thresh in np.linspace(0.5, 3.0, 50):
        pred = (day10_pred < thresh).astype(int)
        acc = (pred == y).mean()
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh

    # Stage 2: per-regime blend weights (use equal weights for simplicity)
    n_days = 3  # days 11, 12, 13
    n_members = len(members_oof)

    # Fit separate weights for each regime
    weights_bolus = np.full((n_days, n_members), 1.0 / n_members)
    weights_nobolus = np.full((n_days, n_members), 1.0 / n_members)

    # Simple grid search for per-regime weights
    mask_bolus = y == 1
    mask_nobolus = y == 0

    for regime_mask, regime_name in [(mask_bolus, 'bolus'), (mask_nobolus, 'nobolus')]:
        if regime_mask.sum() < 3:
            continue
        weights = weights_bolus if regime_name == 'bolus' else weights_nobolus

        for day_idx in range(n_days):
            best_w = np.full(n_members, 1.0 / n_members)
            best_err = np.inf
            for w1 in np.linspace(0, 1, 6):
                if n_members >= 2:
                    for w2 in np.linspace(0, 1 - w1, max(1, int(6 * (1 - w1)))):
                        w = np.array([w1, w2, 1 - w1 - w2][:n_members])
                        w = np.maximum(w, 1e-10)
                        w = w / w.sum()
                        preds = np.average(
                            np.stack([o[regime_mask, 10 + day_idx, 2]
                                       for o in members_oof], axis=0),
                            axis=0, weights=w
                        )
                        err = np.abs(
                            preds - Y_true[regime_mask, 10 + day_idx, 2]
                        ).mean()
                        if err < best_err:
                            best_err = err
                            best_w = w.copy()
                else:
                    best_w = np.array([1.0])
            weights[day_idx] = best_w

    def predict(day10_ensemble):
        """Predict days 11-13 substrate given day-10 ensemble substrate."""
        regime = (day10_ensemble < best_thresh).astype(int)
        result = np.zeros((len(day10_ensemble), 3))
        for i, r in enumerate(regime):
            w = weights_bolus if r == 1 else weights_nobolus
            for d in range(3):
                preds = [o[i, 10 + d, 2] for o in members_oof]
                result[i, d] = np.average(preds, weights=w[d])
        return result

    return predict, best_thresh, best_acc


def apply_two_stage(members_oof, Y_true, fl):
    """Apply two-stage substrate correction and return improved predictions.

    Returns a copy of the leading member's predictions with days 11-13
    substrate replaced by the two-stage predictions.
    """
    predict_fn, threshold, acc = fit_two_stage_predictor(members_oof, Y_true, fl)

    # Use blended day-10 substrate as input
    day10_ensemble = np.mean([o[:, 9, 2] for o in members_oof], axis=0)

    # Get two-stage predictions for days 11-13
    improved_sub = predict_fn(day10_ensemble)

    # Build improved predictions (use equal blend as base)
    improved = np.exp(np.mean([np.log(o) for o in members_oof], axis=0))

    # Replace days 11-13 substrate (indices 10-12)
    improved[:, 10:13, 2] = improved_sub

    # Apply feed-rule truncation (same as original post())
    # THR = {7:1.95, 8:1.95, 9:1.95, 11:0.967, 12:0.967}
    # FLAGCOL = {7:0, 8:1, 9:2, 11:3, 12:4}
    for i in range(len(fl)):
        for k, thr in [(11, 0.967), (12, 0.967)]:
            col = {11: 3, 12: 4}[k]
            if fl[i, col] > 0:
                improved[i, k - 1, 2] = min(improved[i, k - 1, 2], thr * 0.99)
            else:
                improved[i, k - 1, 2] = max(improved[i, k - 1, 2], thr * 1.01)

    return improved, {'threshold': threshold, 'accuracy': acc}
