"""Extended temporal smoothing for all four targets.

The original method smooths only cell densities (VCD, TCD) using
a 3-pass 1/3-weight lowpass filter. This module extends smoothing to
substrate and byproduct, and explores different filter configurations.
"""

import numpy as np


def lowpass(d, w, passes):
    """1D lowpass filter along the time axis (axis=1).

    d: (n, T) array
    w: weight for neighbors (0 < w < 0.5)
    passes: number of passes
    """
    for _ in range(passes):
        e = np.concatenate([d[:, :1], d, d[:, -1:]], axis=1)
        d = w * e[:, :-2] + (1 - 2 * w) * e[:, 1:-1] + w * e[:, 2:]
    return d


def gp_smooth(ell, snr, n=13):
    """GP-based smoother matrix (n x n).

    Returns a smoothing matrix S such that smoothed = d @ S.T
    """
    k = np.arange(n)
    K = snr * np.exp(-0.5 * ((k[:, None] - k[None, :]) / ell) ** 2) + snr * 0.5
    return K @ np.linalg.inv(K + np.eye(n))


def smooth_predictions(P, ref, cfg=None):
    """Apply temporal smoothing to predictions.

    Args:
        P: (n, 13, 4) predictions
        ref: (13, 4) reference means in log space
        cfg: dict mapping target_index to smoothing config.
             None defaults to original behavior (lp on cells only).
             Config tuples:
               ('lp', w, passes) — lowpass filter
               ('gp', ell, snr) — GP smoother
               ('mean', shrink) — shrink toward batch mean
               None — no smoothing for this target

    Returns:
        Smoothed predictions, same shape as P
    """
    if cfg is None:
        # Original behavior: lp(1/3, 3) on VCD and TCD only
        cfg = {0: ('lp', 1 / 3, 3), 1: ('lp', 1 / 3, 3)}

    L = np.log(P).copy()
    out = L.copy()

    for t, c in cfg.items():
        if c is None:
            continue
        d = L[:, :, t] - ref[:, t]
        if c[0] == 'lp':
            d = lowpass(d, c[1], c[2])
        elif c[0] == 'gp':
            d = d @ gp_smooth(c[1], c[2]).T
        elif c[0] == 'mean':
            d = (1 - c[1]) * d + c[1] * d.mean(axis=1, keepdims=True)
        out[:, :, t] = ref[:, t] + d

    return np.exp(out)


def search_smoothing(P, Y_true, folds, idx_map, ref):
    """Search for best smoothing config per target.

    Args:
        P: (n, 13, 4) raw (unsmoothed) predictions
        Y_true: (n, 13, 4) ground truth
        folds: list of fold index arrays
        idx_map: dict mapping global index -> position in P
        ref: (13, 4) reference log-means

    Returns:
        best_cfg: dict of best config per target
        best_mape: 4-vector of MAPE with best config
    """
    # Candidate configs per target
    candidates = {
        'none': None,
        'lp_w20_p3': ('lp', 0.20, 3),
        'lp_w25_p3': ('lp', 0.25, 3),
        'lp_w33_p3': ('lp', 1 / 3, 3),
        'lp_w33_p6': ('lp', 1 / 3, 6),
        'lp_w40_p3': ('lp', 0.40, 3),
        'gp_e2_s10': ('gp', 2, 10),
        'gp_e3_s10': ('gp', 3, 10),
        'gp_e5_s10': ('gp', 5, 10),
        'mean_s50': ('mean', 0.5),
        'mean_s80': ('mean', 0.8),
    }

    best_cfg = {}
    best_mape = np.zeros(4)

    for t in range(4):
        best_score = np.inf
        best_name = 'lp_w33_p3'
        for name, cfg in candidates.items():
            score = 0.0
            for fold_idx in folds:
                # Smooth held-out with all data's ref (following original pattern)
                P_sm = smooth_predictions(P, ref, {t: cfg})
                err = np.abs(
                    P_sm[fold_idx, :, t] - Y_true[fold_idx][:, :, t]
                )
                denom = Y_true[fold_idx][:, :, t]
                score += (err / denom).mean() * 100 / len(folds)
            if score < best_score:
                best_score = score
                best_name = name
        best_cfg[t] = candidates[best_name]
        best_mape[t] = best_score

    return best_cfg, best_mape
