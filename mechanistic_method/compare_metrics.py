#!/usr/bin/env python3
"""Compare model approaches using multiple evaluation metrics.

Loads existing OOF predictions and results.jsonl, computes MAPE, RMSE, MAE, R²
per target and as averages, then prints comparison tables.
"""

import numpy as np
import pandas as pd
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lab'))
from common import load, T, LAB, PROJ


def compute_metrics(P_oof, Y_true):
    """P_oof: (n,13,4) predictions, Y_true: (n,14,4) with first column being day 0.
    Returns dict with per-target and mean values."""
    Yk4 = Y_true[:, 1:, :]  # skip day 0

    mape = (np.abs(P_oof - Yk4) / Yk4 * 100).mean(axis=0)  # (13,4)
    rmse = np.sqrt(((P_oof - Yk4) ** 2).mean(axis=0))
    mae = np.abs(P_oof - Yk4).mean(axis=0)
    ss_res = ((Yk4 - P_oof) ** 2).sum(axis=0)
    ss_tot = ((Yk4 - Yk4.mean(axis=0)) ** 2).sum(axis=0)
    r2 = np.where(ss_tot > 1e-12, 1 - ss_res / ss_tot, 0)

    result = {}
    for name, arr in [('mape', mape), ('rmse', rmse), ('mae', mae), ('r2', r2)]:
        result[name] = float(arr.mean())
        for i, t in enumerate(T):
            result[f'{name}_{t}'] = float(arr[:, i].mean())
    return result


def main():
    bs, A, Y, tt, test = load()
    Ytr = Y[~test]  # labelled batches only

    # Load results
    results_path = os.path.join(LAB, 'results.jsonl')
    records = [json.loads(l) for l in open(results_path)]
    df = pd.DataFrame(records).drop_duplicates('name', keep='last')
    df = df.sort_values('mean').reset_index(drop=True)

    # Compute all metrics from OOF predictions
    rows = []
    for _, r in df.iterrows():
        name = r['name']
        oof_path = os.path.join(LAB, 'oof', f'{name}.npz')
        if not os.path.exists(oof_path):
            continue
        oof = np.load(oof_path)['oof']
        m = compute_metrics(oof, Ytr)
        m['name'] = name
        m['family'] = r['family']
        m['desc'] = r.get('desc', '')
        rows.append(m)

    comp = pd.DataFrame(rows)
    comp = comp.sort_values('mape').reset_index(drop=True)

    # ── Print top 20 approaches (ranked by MAPE) ──
    print("=" * 120)
    print("TOP 20 MODELS (ranked by MAPE)")
    print("=" * 120)
    header = (
        f"{'#':>3}  {'approach':<40} {'family':<18} "
        f"{'MAPE':>7} {'RMSE':>7} {'MAE':>7} {'R²':>7} | "
        f"{'MAPE_vcd':>8} {'MAPE_tcd':>8} {'MAPE_sub':>8} {'MAPE_byp':>8}"
    )
    print(header)
    print("-" * 120)
    for i, r in comp.head(20).iterrows():
        print(
            f"{i+1:>3}  {r['name']:<40} {r['family']:<18} "
            f"{r['mape']:7.3f} {r['rmse']:7.3f} {r['mae']:7.3f} {r['r2']:7.3f} | "
            f"{r['mape_vcd']:8.2f} {r['mape_tcd']:8.2f} "
            f"{r['mape_substrate']:8.2f} {r['mape_byproduct']:8.2f}"
        )

    # ── Best per family ──
    print("\n" + "=" * 120)
    print("BEST PER FAMILY")
    print("=" * 120)
    best_per_family = comp.loc[comp.groupby('family')['mape'].idxmin()].sort_values('mape')
    print(
        f"{'family':<20} {'approach':<42} "
        f"{'MAPE':>7} {'RMSE':>7} {'MAE':>7} {'R²':>7} | "
        f"{'MAPE_vcd':>8} {'MAPE_tcd':>8} {'MAPE_sub':>8} {'MAPE_byp':>8}"
    )
    print("-" * 120)
    for _, r in best_per_family.iterrows():
        print(
            f"{r['family']:<20} {r['name']:<42} "
            f"{r['mape']:7.3f} {r['rmse']:7.3f} {r['mae']:7.3f} {r['r2']:7.3f} | "
            f"{r['mape_vcd']:8.2f} {r['mape_tcd']:8.2f} "
            f"{r['mape_substrate']:8.2f} {r['mape_byproduct']:8.2f}"
        )

    # ── Family summary ──
    print("\n" + "=" * 120)
    print("FAMILY SUMMARY (count, best MAPE, best RMSE, best MAE, best R²)")
    print("=" * 120)
    summary = comp.groupby('family').agg(
        count=('mape', 'count'),
        best_mape=('mape', 'min'),
        best_rmse=('rmse', 'min'),
        best_mae=('mae', 'min'),
        best_r2=('r2', 'max'),
    ).round(3)
    print(summary.to_string())

    # ── Final blends (b01-b11) detail ──
    blends = comp[comp['name'].str.match(r'^b\d{2}_')].sort_values('mape')
    if len(blends) > 0:
        print("\n" + "=" * 120)
        print("BLEND DETAIL (b01-b11)")
        print("=" * 120)
        print(
            f"{'#':>3}  {'approach':<42} "
            f"{'MAPE':>7} {'RMSE':>7} {'MAE':>7} {'R²':>7} | "
            f"{'MAPE_vcd':>8} {'MAPE_tcd':>8} {'MAPE_sub':>8} {'MAPE_byp':>8}"
        )
        print("-" * 120)
        for i, (_, r) in enumerate(blends.iterrows()):
            print(
                f"{i+1:>3}  {r['name']:<42} "
                f"{r['mape']:7.3f} {r['rmse']:7.3f} {r['mae']:7.3f} {r['r2']:7.3f} | "
                f"{r['mape_vcd']:8.2f} {r['mape_tcd']:8.2f} "
                f"{r['mape_substrate']:8.2f} {r['mape_byproduct']:8.2f}"
            )

    # ── Save CSV ──
    out_path = os.path.join(PROJ, 'model_comparison.csv')
    comp.to_csv(out_path, index=False, float_format='%.4f')
    print(f"\nFull comparison saved to: {out_path}")


if __name__ == '__main__':
    main()
