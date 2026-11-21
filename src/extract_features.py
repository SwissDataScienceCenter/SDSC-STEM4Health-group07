"""
extract_features.py - Feature extraction from Project 1 DTB bioreaction data.

This script loads the two CSV files from the input-data folder and extracts
general summary features to help understand the data structure and meaning.

Usage:
    python src/extract_features.py
"""

import pandas as pd
import numpy as np
import os

# ── Paths ────────────────────────────────────────────────────────────────────
INPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "input-data",
                         "Project 1 - DTB")
REMAINING_PATH = os.path.join(INPUT_DIR, "bioreaction_remaining.csv")
TEMPLATE_PATH = os.path.join(INPUT_DIR, "bioreaction_template.csv")


def load_data():
    """Load both CSV files."""
    df_rem = pd.read_csv(REMAINING_PATH)
    df_tpl = pd.read_csv(TEMPLATE_PATH)
    return df_rem, df_tpl


def basic_info(df_rem, df_tpl):
    """Print basic shapes and column information."""
    print("=" * 70)
    print("1. BASIC INFORMATION")
    print("=" * 70)

    print("\n── bioreaction_remaining.csv ──")
    print(f"  Shape:        {df_rem.shape[0]} rows × {df_rem.shape[1]} columns")
    print(f"  Columns:      {list(df_rem.columns)}")
    print(f"  Memory usage: {df_rem.memory_usage(deep=True).sum() / 1024:.1f} KB")

    print("\n── bioreaction_template.csv ──")
    print(f"  Shape:        {df_tpl.shape[0]} rows × {df_tpl.shape[1]} columns")
    print(f"  Columns:      {list(df_tpl.columns)}")
    print(f"  Memory usage: {df_tpl.memory_usage(deep=True).sum() / 1024:.1f} KB")


def column_semantics(df_rem, df_tpl):
    """Explain the meaning and statistics of each column."""
    print("\n" + "=" * 70)
    print("2. COLUMN DESCRIPTION & STATISTICS")
    print("=" * 70)

    # ── batch_id ──
    n_batches_rem = df_rem["batch_id"].nunique()
    n_batches_tpl = df_tpl["batch_id"].nunique()
    print(f"\n  batch_id  ─  Batch identifier (string)")
    print(f"    Unique batches in remaining:   {n_batches_rem}")
    print(f"    Unique batches in template:    {n_batches_tpl}")
    print(f"    Template batches (subset):     {sorted(df_tpl['batch_id'].unique())}")

    # ── elapsed_time ──
    print(f"\n  elapsed_time  ─  Time since start of batch [hours]")
    print(f"    Range (remaining):  {df_rem['elapsed_time'].min():.4f} – {df_rem['elapsed_time'].max():.4f}")
    print(f"    Range (template):   {df_tpl['elapsed_time'].min():.4f} – {df_tpl['elapsed_time'].max():.4f}")
    print(f"    Typical step ~1 h; each batch ends around 13 h total.")

    # ── vcd ──
    print(f"\n  vcd  ─  Viable Cell Density  [×10⁶ cells/mL ?]")
    print(f"    Range (non-null):   {df_rem['vcd'].min():.4f} – {df_rem['vcd'].max():.4f}")
    print(f"    Null count:         {df_rem['vcd'].isna().sum()} / {len(df_rem)}")

    # ── tcd ──
    print(f"\n  tcd  ─  Total Cell Density  [×10⁶ cells/mL ?]")
    print(f"    Range (non-null):   {df_rem['tcd'].min():.4f} – {df_rem['tcd'].max():.4f}")
    print(f"    Null count:         {df_rem['tcd'].isna().sum()} / {len(df_rem)}")
    print(f"    Note: tcd ≥ vcd always (tcd includes dead cells).")

    # ── substrate ──
    print(f"\n  substrate  ─  Substrate concentration (e.g. glucose) [g/L ?]")
    print(f"    Range (non-null):   {df_rem['substrate'].min():.4f} – {df_rem['substrate'].max():.4f}")
    print(f"    Null count:         {df_rem['substrate'].isna().sum()} / {len(df_rem)}")
    print(f"    Typical trend: starts ~4.5-4.9, declines toward ~0.3-1.6 by end.")

    # ── byproduct ──
    print(f"\n  byproduct  ─  Byproduct concentration (e.g. lactate) [g/L ?]")
    print(f"    Range (non-null):   {df_rem['byproduct'].min():.4f} – {df_rem['byproduct'].max():.4f}")
    print(f"    Null count:         {df_rem['byproduct'].isna().sum()} / {len(df_rem)}")

    # ── feed columns ──
    for col in ["feed_substrate", "feed_b", "feed_c", "feed_d"]:
        nulls = df_rem[col].isna().sum()
        present = len(df_rem) - nulls
        print(f"\n  {col}  ─  Feed rate / amount of {col.split('_')[-1].upper()} component")
        print(f"    Non-null values:    {present} / {len(df_rem)}")
        if present > 0:
            print(f"    Range:              {df_rem[col].min():.6f} – {df_rem[col].max():.6f}")
        print(f"    Note: Only recorded at specific time points (feeding events).")

    # ── volume ──
    print(f"\n  volume  ─  Reactor working volume [L]")
    print(f"    Range (non-null):   {df_rem['volume'].min():.4f} – {df_rem['volume'].max():.4f}")
    print(f"    Null count:         {df_rem['volume'].isna().sum()} / {len(df_rem)}")
    print(f"    Trend: increases over time due to feeding.")

    # ── to_predict ──
    print(f"\n  to_predict  ─  Boolean flag: should this row be predicted?")
    counts = df_rem["to_predict"].value_counts()
    for val, cnt in counts.items():
        print(f"    {val}:  {cnt}")
    print(f"    Note: TRUE rows = template rows where vcd/tcd/substrate/byproduct are 0.")
    print(f"    These are the rows that must be predicted for the leaderboard.")


def rows_per_batch(df_rem):
    """Summarise how many rows each batch has and what is recorded."""
    print("\n" + "=" * 70)
    print("3. ROWS PER BATCH")
    print("=" * 70)

    batch_counts = df_rem.groupby("batch_id").size()
    print(f"  Total batches:              {len(batch_counts)}")
    print(f"  Min rows in a batch:        {batch_counts.min()}")
    print(f"  Max rows in a batch:        {batch_counts.max()}")
    print(f"  Mean rows per batch:        {batch_counts.mean():.1f}")
    print(f"  Median rows per batch:      {batch_counts.median():.0f}")


def data_pattern(df_rem, df_tpl):
    """Explain the data recording pattern."""
    print("\n" + "=" * 70)
    print("4. DATA RECORDING PATTERN")
    print("=" * 70)

    print("""
  Each row represents ONE measurement at a specific time. NOT every quantity is measured at every time point!

  ┌──────────────────────────────────────────────────────────────────┐
  │  ~hourly (0, 1, 2, 3, ... 13):   vcd, tcd, substrate, byproduct │
  │  at feeding times:                feed_substrate, feed_b,        │
  │                                   feed_c, feed_d                 │
  │  at other times:                  volume                         │
  └──────────────────────────────────────────────────────────────────┘

  The CSV is SPARSE: most cells are NaN because only a subset of
  measurements is taken at each time point. This is typical of
  industrial bio-process data.

  Example from Batch1 (first few rows):
""")
    batch1 = df_rem[df_rem["batch_id"] == "Batch1"].head(12)
    # Compact column display
    cols = ["elapsed_time", "vcd", "tcd", "substrate", "byproduct",
            "feed_substrate", "feed_b", "feed_c", "feed_d", "volume"]
    sub = batch1[cols].copy()
    sub.loc[:, "elapsed_time"] = sub["elapsed_time"].round(2)
    print(sub.to_string(index=False))
    print()


def prediction_task(df_rem, df_tpl):
    """Explain the prediction task."""
    print("=" * 70)
    print("5. PREDICTION TASK")
    print("=" * 70)

    pred_rows = df_rem[df_rem["to_predict"] == True]
    print(f"""
  The template file (bioreaction_template.csv) contains {len(df_tpl)} rows
  for specific (batch_id, elapsed_time) combinations.

  In the remaining file, these exact rows have:
    - vcd, tcd, substrate, byproduct  = NaN  (to be predicted)
    - feed_substrate, feed_b, feed_c, feed_d, volume = measured (given)

  The task is to predict the FOUR target variables:
    - vcd       (Viable Cell Density)
    - tcd       (Total Cell Density)
    - substrate (Substrate concentration)
    - byproduct (Byproduct concentration)

  Evaluation: Mean Absolute Percentage Error (MAPE) per target,
  then averaged into a single "mean_mape" score.

  Example template rows (what must be predicted):
""")

    # Show a couple of template rows merged with remaining context
    tpl_show = df_tpl.head(6).copy()
    print(tpl_show.to_string(index=False))

    print(f"""
  Template batches (those with to_predict=True rows):
    {sorted(df_tpl['batch_id'].unique())}

  Remaining batches (complete measurement data for training):
    All other batches not in the template set.
""")


def feature_missingness(df_rem, df_tpl):
    """Show missingness pattern as a summary."""
    print("=" * 70)
    print("6. MISSING DATA SUMMARY")
    print("=" * 70)

    # Compute missingness for remaining
    total = len(df_rem)
    print(f"\n  {'Column':<20s} {'# Present':>10s}   {'% Present':>10s}  {'Role':>20s}")
    print("  " + "-" * 70)

    roles = {
        "batch_id": "key", "elapsed_time": "key",
        "vcd": "target", "tcd": "target",
        "substrate": "target", "byproduct": "target",
        "feed_substrate": "predictor", "feed_b": "predictor",
        "feed_c": "predictor", "feed_d": "predictor",
        "volume": "predictor", "to_predict": "flag",
    }
    for col in df_rem.columns:
        present = df_rem[col].notna().sum()
        pct = present / total * 100
        role = roles.get(col, "?")
        print(f"  {col:<20s} {present:>10d}   {pct:>9.1f}%  {role:>20s}")


def main():
    print("=" * 70)
    print("  EXTRACTION OF GENERAL FEATURES FROM BIOREACTION DATA")
    print("  Project 1 – DTB (Digital Twin of a Bioreactor)")
    print("=" * 70)

    df_rem, df_tpl = load_data()

    basic_info(df_rem, df_tpl)
    column_semantics(df_rem, df_tpl)
    rows_per_batch(df_rem)
    data_pattern(df_rem, df_tpl)
    prediction_task(df_rem, df_tpl)
    feature_missingness(df_rem, df_tpl)


if __name__ == "__main__":
    main()
