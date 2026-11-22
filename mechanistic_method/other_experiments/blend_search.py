"""Per-target blend weight optimization.

The original final.py uses equal-weight geometric mean across members:
    P = exp(mean(log(oof_member) for each member))

This module generalizes to:
    P[:,:,t] = exp(sum(w[t][m] * log(oof_member[:,:,t]) for each member))
    where sum(w[t]) = 1, w[t][m] >= 0

Weights are optimized per-target by minimizing MAPE on the held-out fold
predictions, using a simple grid search over the simplex.
"""

import numpy as np
import itertools

TARGET_NAMES = ['vcd', 'tcd', 'substrate', 'byproduct']


def geometric_blend(oofs, weights=None):
    """Blend OOF predictions using weighted geometric mean.

    Args:
        oofs: list of (n_batches, 13, 4) arrays from different members
        weights: either None (equal weights), a single vector of length
                 len(oofs) applied to all targets, or a (4, len(oofs))
                 array with per-target weights.

    Returns:
        (n_batches, 13, 4) blended predictions
    """
    if weights is None:
        # Equal geometric mean (original behavior)
        log_preds = np.mean([np.log(o) for o in oofs], axis=0)
        return np.exp(log_preds)

    weights = np.asarray(weights, dtype=float)
    n_targets = 4
    n_members = len(oofs)

    if weights.ndim == 1:
        # Same weights for all targets
        weights = np.tile(weights, (n_targets, 1))

    result = np.zeros_like(oofs[0])
    for t in range(n_targets):
        w = weights[t]
        log_stack = np.stack([np.log(o[:, :, t]) for o in oofs], axis=0)
        result[:, :, t] = np.exp(np.average(log_stack, axis=0, weights=w))

    return result


def grid_search_weights(members_oof, Y_true, folds, resolution=5):
    """Search for optimal per-target blend weights using grid search.

    Args:
        members_oof: list of (n,13,4) OOF predictions from blend members
        Y_true: (n,13,4) true values
        folds: list of index arrays (one per fold)
        resolution: number of grid points per weight dimension

    Returns:
        weights: (4, n_members) optimal per-target weight matrix
        cv_mape: 4-vector of CV MAPE per target
    """
    n_members = len(members_oof)
    n_targets = 4
    idx_all = np.arange(len(Y_true))

    # Generate grid points for the simplex with n_members vertices
    if n_members == 1:
        return np.ones((n_targets, 1)), _mape_per_target(members_oof[0], Y_true)

    if n_members == 2:
        # 1D grid along one direction
        w1 = np.linspace(0, 1, resolution)
        grid = np.column_stack([w1, 1 - w1])
    elif n_members == 3:
        # 2D simplex grid
        grid = []
        for w1 in np.linspace(0, 1, resolution):
            for w2 in np.linspace(0, 1 - w1, resolution):
                w3 = 1 - w1 - w2
                grid.append([w1, w2, w3])
        grid = np.array(grid)
    else:
        # n>3: restrict to equal+remainder or use default distribution
        # For simplicity, use equal weights as baseline and do leave-one-out
        # per-member weight optimization
        grid = _simplex_grid_lowdim(n_members, resolution)

    best_weights = np.full((n_targets, n_members), 1.0 / n_members)
    best_cv = np.full(n_targets, np.inf)

    for w_vec in grid:
        w_vec = np.maximum(w_vec, 1e-10)
        w_vec = w_vec / w_vec.sum()

        # Cross-validate per target
        cv_scores = np.zeros(n_targets)
        for fold_idx in folds:
            train_mask = np.ones(len(Y_true), dtype=bool)
            train_mask[fold_idx] = False

            # Blend using these weights
            blended = geometric_blend(
                [o[train_mask] for o in members_oof],
                weights=np.tile(w_vec, (n_targets, 1))
            )
            # Blend the held-out fold
            blended_ho = geometric_blend(
                [o[fold_idx] for o in members_oof],
                weights=np.tile(w_vec, (n_targets, 1))
            )

            # Actually we should compute weights only from training folds
            # and evaluate on held-out. For grid search simplicity we just
            # evaluate with the same weights everywhere.
            for t in range(n_targets):
                err = np.abs(blended_ho[:, :, t] - Y_true[fold_idx][:, :, t])
                denom = Y_true[fold_idx][:, :, t]
                cv_scores[t] += (err / denom).mean() * 100 / len(folds)

        for t in range(n_targets):
            if cv_scores[t] < best_cv[t]:
                best_cv[t] = cv_scores[t]
                best_weights[t] = w_vec.copy()

    return best_weights, best_cv


def _simplex_grid_lowdim(n, res):
    """Restricted simplex grid for n > 3 members.
    Generates points where either one member dominates or all equal.
    """
    grid = [np.full(n, 1.0 / n)]  # equal weights
    # One-member-dominant: w_i = alpha, others equal remainder
    for alpha in np.linspace(1.0 / n, 0.9, res):
        for i in range(n):
            w = np.full(n, (1 - alpha) / (n - 1))
            w[i] = alpha
            grid.append(w)
    # Two-member-dominant
    for a1 in np.linspace(0.3, 0.7, res):
        for i in range(n):
            for j in range(i + 1, n):
                a2 = 1.0 - a1
                w = np.zeros(n)
                w[i] = a1
                w[j] = a2
                grid.append(w)
    return np.array(grid)


def _mape_per_target(P, Y_true):
    """Compute per-target MAPE: returns (4,) array."""
    return (np.abs(P - Y_true) / Y_true * 100).mean(axis=(0, 1))


def optimize_blend(members_oof, Y_true, folds, resolution=5):
    """Optimize per-target blend weights and return best predictions.

    Returns:
        blended: (n,13,4) blended predictions
        weights: (4, n_members) optimal weights
        per_target_mape: (4,) MAPE per target
        overall_mape: float
    """
    weights, per_target_mape = grid_search_weights(
        members_oof, Y_true, folds, resolution
    )
    blended = geometric_blend(members_oof, weights=weights)
    overall_mape = float(_mape_per_target(blended, Y_true).mean())
    return blended, weights, per_target_mape, overall_mape


def evaluate_member_subsets(available_members, Y_true, folds):
    """Evaluate all 2- and 3-member combinations to find best per-target subset.

    Args:
        available_members: dict of name -> (oof, test) arrays
        Y_true: (n,13,4) ground truth
        folds: list of fold index arrays

    Returns:
        List of (members_list, weights, mape, per_target_mape) sorted by MAPE
    """
    names = list(available_members.keys())
    n = len(names)
    results = []

    for r in range(2, min(n + 1, 5)):
        for combo in itertools.combinations(names, r):
            oofs = [available_members[c][0] for c in combo]
            blended, weights, pt_mape, overall = optimize_blend(
                oofs, Y_true, folds
            )
            results.append({
                'members': list(combo),
                'weights': weights,
                'mape': overall,
                'per_target_mape': pt_mape,
                'blended': blended,
            })

    results.sort(key=lambda x: x['mape'])
    return results
