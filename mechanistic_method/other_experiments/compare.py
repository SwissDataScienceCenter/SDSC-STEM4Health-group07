"""Comprehensive multi-metric model comparison.

Loads OOF predictions for all approaches and computes MAPE, RMSE, MAE, R²
per target. Also compares across strategies (original blends, improved blends,
two-stage, etc.).
"""

import numpy as np
import pandas as pd
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lab'))
from common import load, T

LAB = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lab')


def compute_all_metrics(P_oof, Y_true):
    """Compute all metrics per target and overall.

    Returns dict with keys like 'mape', 'mape_vcd', 'rmse', 'rmse_vcd', etc.
    """
    Yk4 = Y_true  # (n, 13, 4) — expects already without day 0

    mape_arr = (np.abs(P_oof - Yk4) / Yk4 * 100).mean(axis=0)  # (13, 4)
    rmse_arr = np.sqrt(((P_oof - Yk4) ** 2).mean(axis=0))
    mae_arr = np.abs(P_oof - Yk4).mean(axis=0)
    ss_res = ((Yk4 - P_oof) ** 2).sum(axis=0)
    ss_tot = ((Yk4 - Yk4.mean(axis=0)) ** 2).sum(axis=0)
    r2_arr = np.where(ss_tot > 1e-12, 1 - ss_res / ss_tot, 0)

    result = {}
    for name, arr in [('mape', mape_arr), ('rmse', rmse_arr),
                       ('mae', mae_arr), ('r2', r2_arr)]:
        result[name] = float(arr.mean())
        for i, t in enumerate(T):
            result[f'{name}_{t}'] = float(arr[:, i].mean())

    return result


def compare_all(extra_models=None):
    """Compare all models from results.jsonl plus optional extra models.

    Args:
        extra_models: dict of name -> {predictions: (n,13,4), family: str,
                       desc: str} for additional models.

    Returns:
        pd.DataFrame sorted by MAPE
    """
    bs, A, Y, tt, test = load()
    Ytr = Y[~test][:, 1:, :]  # (47, 13, 4)

    # Original models from results.jsonl
    results_path = os.path.join(LAB, 'results.jsonl')
    with open(results_path) as f:
        records = [json.loads(l) for l in f]

    rows = []
    for r in records:
        name = r['name']
        oof_path = os.path.join(LAB, 'oof', f'{name}.npz')
        if not os.path.exists(oof_path):
            continue
        P = np.load(oof_path)['oof']
        m = compute_all_metrics(P, Ytr)
        m['name'] = name
        m['family'] = r['family']
        m['desc'] = r.get('desc', '')
        rows.append(m)

    # Extra models
    if extra_models:
        for name, info in extra_models.items():
            P = info['predictions']
            m = compute_all_metrics(P, Ytr)
            m['name'] = name
            m['family'] = info.get('family', 'improved')
            m['desc'] = info.get('desc', '')
            rows.append(m)

    df = pd.DataFrame(rows)
    df = df.sort_values('mape').reset_index(drop=True)
    return df


def print_comparison(df, top_n=20):
    """Pretty-print comparison table."""
    print("=" * 130)
    print(f"TOP {min(top_n, len(df))} MODELS (ranked by MAPE)")
    print("=" * 130)
    header = (
        f"{'#':>3}  {'approach':<42} {'family':<18} "
        f"{'MAPE':>7} {'RMSE':>7} {'MAE':>7} {'R²':>7} | "
        f"{'MAPE_vcd':>8} {'MAPE_tcd':>8} {'MAPE_sub':>8} {'MAPE_byp':>8}"
    )
    print(header)
    print("-" * 130)
    for idx, (_, r) in enumerate(df.head(top_n).iterrows()):
        print(
            f"{idx + 1:>3}  {r['name']:<42} {r['family']:<18} "
            f"{r['mape']:7.3f} {r['rmse']:7.3f} {r['mae']:7.3f} {r['r2']:7.3f} | "
            f"{r['mape_vcd']:8.2f} {r['mape_tcd']:8.2f} "
            f"{r['mape_substrate']:8.2f} {r['mape_byproduct']:8.2f}"
        )

    # Best per family
    print("\n" + "=" * 130)
    print("BEST PER FAMILY")
    print("=" * 130)
    best_per_family = df.loc[df.groupby('family')['mape'].idxmin()].sort_values('mape')
    print(
        f"{'family':<22} {'approach':<44} "
        f"{'MAPE':>7} {'RMSE':>7} {'MAE':>7} {'R²':>7}"
    )
    print("-" * 130)
    for _, r in best_per_family.iterrows():
        print(
            f"{r['family']:<22} {r['name']:<44} "
            f"{r['mape']:7.3f} {r['rmse']:7.3f} {r['mae']:7.3f} {r['r2']:7.3f}"
        )

    # Family summary
    print("\n" + "=" * 130)
    print("FAMILY SUMMARY")
    print("=" * 130)
    summary = df.groupby('family').agg(
        count=('mape', 'count'),
        best_mape=('mape', 'min'),
        best_rmse=('rmse', 'min'),
        best_r2=('r2', 'max'),
    ).round(3)
    print(summary.to_string())

    # Per-target deep dive for top models
    print("\n" + "=" * 130)
    print("PER-TARGET BREAKDOWN (top 5 plus relevant comparisons)")
    print("=" * 130)
    print(
        f"{'approach':<44} "
        f"{'MAPE_vcd':>9} {'MAPE_tcd':>9} {'MAPE_sub':>9} {'MAPE_byp':>9} | "
        f"{'R²_vcd':>7} {'R²_tcd':>7} {'R²_sub':>7} {'R²_byp':>7}"
    )
    print("-" * 130)
    for _, r in df.head(top_n).iterrows():
        print(
            f"{r['name']:<44} "
            f"{r['mape_vcd']:9.2f} {r['mape_tcd']:9.2f} "
            f"{r['mape_substrate']:9.2f} {r['mape_byproduct']:9.2f} | "
            f"{r['r2_vcd']:7.3f} {r['r2_tcd']:7.3f} "
            f"{r['r2_substrate']:7.3f} {r['r2_byproduct']:7.3f}"
        )
