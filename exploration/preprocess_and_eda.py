#!/usr/bin/env python3
"""
Preprocessing and Exploratory Data Analysis for bioreaction_remaining.csv
from Project 1 - DTB.

Key findings from initial inspection:
- 2858 rows × 12 columns, 61 unique batches
- The dataset has 3 types of rows interleaved:
  1. MEASUREMENT rows: contain vcd, tcd, substrate, byproduct, volume (698 rows)
  2. FEED rows: contain feed_substrate, feed_b, feed_c, feed_d, volume (some also) (2004 rows)
  3. PREDICT-ONLY rows (to_predict=True): only elapsed_time & volume, no measurements (156 rows)
- These three types NEVER overlap (0 rows with both measurements and feed data)
- The 156 to_predict rows are the held-out test set (12 batches × ~13 timepoints each)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({
    'figure.dpi': 120,
    'savefig.dpi': 120,
    'figure.figsize': (12, 6),
})

OUTPUT_DIR = Path("output_eda")
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("=" * 70)
print("1. LOADING DATA")
print("=" * 70)

df_raw = pd.read_csv("input-data/Project 1 - DTB/bioreaction_remaining.csv")
print(f"Raw shape: {df_raw.shape}")
print(f"Columns: {list(df_raw.columns)}")

# ============================================================================
# 2. UNDERSTAND ROW TYPES AND SPLIT
# ============================================================================
print("\n" + "=" * 70)
print("2. ROW TYPE CLASSIFICATION")
print("=" * 70)

meas_cols = ['vcd', 'tcd', 'substrate', 'byproduct']
feed_cols = ['feed_substrate', 'feed_b', 'feed_c', 'feed_d']

df_raw['row_type'] = 'unknown'
df_raw.loc[df_raw[meas_cols].notna().any(axis=1), 'row_type'] = 'measurement'
df_raw.loc[df_raw[feed_cols].notna().any(axis=1), 'row_type'] = 'feed'
df_raw.loc[df_raw['to_predict'] == True, 'row_type'] = 'predict_target'

print("Row type distribution:")
print(df_raw['row_type'].value_counts())
print(f"\n(measurement + feed overlap check: {((df_raw['row_type']=='measurement') & (df_raw['row_type']=='feed')).sum()})")

# ============================================================================
# 3. SPLIT INTO THREE SEPARATE DATAFRAMES
# ============================================================================
print("\n" + "=" * 70)
print("3. SPLITTING DATA")
print("=" * 70)

# Measurement table: one row per (batch, time) with state variables
df_meas = df_raw[df_raw['row_type'] == 'measurement'].copy()
df_meas = df_meas[['batch_id', 'elapsed_time', 'vcd', 'tcd', 'substrate', 'byproduct', 'volume']]

# Feed table: one row per (batch, time) with feed additions
df_feed = df_raw[df_raw['row_type'] == 'feed'].copy()
df_feed = df_feed[['batch_id', 'elapsed_time', 'feed_substrate', 'feed_b', 'feed_c', 'feed_d', 'volume']]

# Predict targets (the held-out batch × timepoints to predict)
df_predict = df_raw[df_raw['row_type'] == 'predict_target'].copy()
df_predict = df_predict[['batch_id', 'elapsed_time', 'volume']]

print(f"Measurement rows: {len(df_meas)} (batches: {df_meas['batch_id'].nunique()})")
print(f"Feed rows:       {len(df_feed)} (batches: {df_feed['batch_id'].nunique()})")
print(f"Predict targets: {len(df_predict)} (batches: {df_predict['batch_id'].nunique()})")

# Identify which batches are in predict set vs training set
predict_batches = set(df_predict['batch_id'].unique())
train_batches = set(df_meas['batch_id'].unique()) - predict_batches
print(f"\nTraining batches (with measurements):     {len(train_batches)}")
print(f"Prediction-only batches (to_predict):      {len(predict_batches)}")
print(f"Prediction batches: {sorted(predict_batches, key=lambda x: int(x.replace('Batch','')))}")

# ============================================================================
# 4. MERGE MEASUREMENTS AND FEEDS INTO A COMBINED TIMELINE
# ============================================================================
print("\n" + "=" * 70)
print("4. MERGING MEASUREMENTS + FEEDS")
print("=" * 70)

# Combine measurement and feed into a unified timeline (outer join by batch_id, elapsed_time)
# They don't overlap so a concat + sort works well
df_combined = pd.concat([df_meas, df_feed], ignore_index=True)
df_combined = df_combined.sort_values(['batch_id', 'elapsed_time']).reset_index(drop=True)

# Forward-fill measurements across feed rows within each batch
# This lets us see what the state was at the time of each feed addition
state_cols = ['vcd', 'tcd', 'substrate', 'byproduct', 'volume']
for col in state_cols:
    df_combined[col] = df_combined.groupby('batch_id')[col].ffill()

# Forward-fill feed columns (they are mostly NaN except at feed events)
feed_cols_all = ['feed_substrate', 'feed_b', 'feed_c', 'feed_d']
# Don't forward fill feeds -- they are discrete events

print(f"Combined timeline: {len(df_combined)} rows")
print(f"Rows with state data after merge: {df_combined['vcd'].notna().sum()}")

# ============================================================================
# 5. EDA: BATCH-LEVEL OVERVIEW
# ============================================================================
print("\n" + "=" * 70)
print("5. EDA: BATCH-LEVEL OVERVIEW")
print("=" * 70)

# Per-batch summary of measurements
batch_summary = df_meas.groupby('batch_id').agg(
    n_measurements=('elapsed_time', 'count'),
    t_start=('elapsed_time', 'min'),
    t_end=('elapsed_time', 'max'),
    vcd_max=('vcd', 'max'),
    tcd_max=('tcd', 'max'),
    substrate_start=('substrate', 'first'),
    substrate_end=('substrate', 'last'),
    byproduct_max=('byproduct', 'max'),
    volume_start=('volume', 'first'),
    volume_end=('volume', 'last'),
).reset_index()

batch_summary['is_predict'] = batch_summary['batch_id'].isin(predict_batches)
batch_summary['duration'] = batch_summary['t_end'] - batch_summary['t_start']

print("\nAll training batches summary:")
print(batch_summary[~batch_summary['is_predict']].describe().round(4))

print(f"\nAverage measurements per training batch: {batch_summary[~batch_summary['is_predict']]['n_measurements'].mean():.1f}")

# ============================================================================
# 6. EDA: FEED ANALYSIS
# ============================================================================
print("\n" + "=" * 70)
print("6. EDA: FEED ANALYSIS")
print("=" * 70)

# Which feed columns are populated?
for col in feed_cols:
    n = df_feed[col].notna().sum()
    print(f"  {col}: {n} non-null rows ({n/len(df_feed)*100:.1f}%)")

print(f"\nFeed events per batch (training):")
feed_counts = df_feed[~df_feed['batch_id'].isin(predict_batches)].groupby('batch_id').size()
print(feed_counts.describe().round(2))

# When do feed events happen?
feed_times = df_feed['elapsed_time'].dropna()
print(f"\nFeed timing stats (all batches):")
print(f"  Mean: {feed_times.mean():.2f}, Median: {feed_times.median():.2f}")
print(f"  Min: {feed_times.min():.2f}, Max: {feed_times.max():.2f}")

# Per-feed-type distributions
print("\nPer-feed-type value distributions:")
for col in feed_cols:
    non_null = df_feed[col].dropna()
    if len(non_null) > 0:
        print(f"  {col}: mean={non_null.mean():.6f}, median={non_null.median():.6f}, "
              f"min={non_null.min():.6f}, max={non_null.max():.6f}, n={len(non_null)}")

# ============================================================================
# 7. EDA: MEASUREMENT TRENDS
# ============================================================================
print("\n" + "=" * 70)
print("7. EDA: MEASUREMENT STATS")
print("=" * 70)

train_meas = df_meas[~df_meas['batch_id'].isin(predict_batches)]
for col in meas_cols + ['volume']:
    print(f"\n  {col}:")
    print(f"    n={train_meas[col].notna().sum()}, mean={train_meas[col].mean():.4f}, "
          f"std={train_meas[col].std():.4f}, min={train_meas[col].min():.4f}, max={train_meas[col].max():.4f}")

# ============================================================================
# 8. CORRELATION ANALYSIS
# ============================================================================
print("\n" + "=" * 70)
print("8. CORRELATION MATRIX (measurements only)")
print("=" * 70)

corr_cols = ['elapsed_time', 'vcd', 'tcd', 'substrate', 'byproduct', 'volume']
corr = train_meas[corr_cols].corr()
print(corr.round(4))

# ============================================================================
# 9. PREDICTION TARGET ANALYSIS
# ============================================================================
print("\n" + "=" * 70)
print("9. PREDICTION TARGET (to_predict) ANALYSIS")
print("=" * 70)

for batch in sorted(predict_batches, key=lambda x: int(x.replace('Batch', ''))):
    subset = df_predict[df_predict['batch_id'] == batch]
    print(f"  {batch}: {len(subset)} timepoints, t=[{subset['elapsed_time'].min():.3f}..{subset['elapsed_time'].max():.3f}], "
          f"volume=[{subset['volume'].min():.3f}..{subset['volume'].max():.3f}]")

# ============================================================================
# 10. EXPORT PROCESSED DATASETS
# ============================================================================
print("\n" + "=" * 70)
print("10. EXPORTING PROCESSED DATA")
print("=" * 70)

# Add a train/predict flag to the measurement dataframe
df_meas['dataset'] = df_meas['batch_id'].apply(
    lambda b: 'predict' if b in predict_batches else 'train'
)

# Export key tables
df_meas.to_csv(OUTPUT_DIR / "measurements_clean.csv", index=False)
df_feed.to_csv(OUTPUT_DIR / "feeds_clean.csv", index=False)
df_combined.to_csv(OUTPUT_DIR / "combined_timeline.csv", index=False)
df_predict.to_csv(OUTPUT_DIR / "predict_targets.csv", index=False)

# Export a pivot: wide-format per batch for ML
# Pivot measurements to wide format (time-aligned might need interpolation, 
# but here each batch has its own timepoints)
df_meas_pivot = df_meas.pivot_table(
    index='batch_id',
    columns='elapsed_time',
    values=['vcd', 'tcd', 'substrate', 'byproduct', 'volume'],
    aggfunc='first'
)
df_meas_pivot.to_csv(OUTPUT_DIR / "measurements_wide.csv")

# Summary per batch
batch_summary.to_csv(OUTPUT_DIR / "batch_summary.csv", index=False)

print(f"Files written to {OUTPUT_DIR}/:")
for f in sorted(OUTPUT_DIR.iterdir()):
    print(f"  {f.name} ({f.stat().st_size:,} bytes)")

# ============================================================================
# 11. VISUALIZATION
# ============================================================================
print("\n" + "=" * 70)
print("11. GENERATING PLOTS")
print("=" * 70)

# --- Fig 1: Time-series of state variables for training batches ---
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
metrics = [('vcd', 'VCD (Viable Cell Density)'), 
           ('tcd', 'TCD (Total Cell Density)'),
           ('substrate', 'Substrate'),
           ('byproduct', 'Byproduct')]

for (col, title), ax in zip(metrics, axes.flat):
    for batch in sorted(train_batches, key=lambda x: int(x.replace('Batch', '')))[:15]:  # first 15 batches
        subset = df_meas[df_meas['batch_id'] == batch].dropna(subset=[col])
        ax.plot(subset['elapsed_time'], subset[col], marker='.', alpha=0.6, linewidth=1, markersize=3)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Elapsed Time (h)')
    ax.set_ylabel(col)

fig.suptitle('Training Batches (first 15) — State Variable Trajectories', fontsize=15, fontweight='bold', y=1.01)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "trajectories_training.png", bbox_inches='tight', dpi=150)
plt.close(fig)
print("  → trajectories_training.png")

# --- Fig 2: Volume evolution ---
fig, ax = plt.subplots(figsize=(14, 5))
for batch in sorted(train_batches, key=lambda x: int(x.replace('Batch', '')))[:20]:
    subset = df_meas[df_meas['batch_id'] == batch].dropna(subset=['volume'])
    ax.plot(subset['elapsed_time'], subset['volume'], marker='.', alpha=0.5, linewidth=1, markersize=3)
ax.set_title('Volume Evolution — Training Batches (first 20)', fontsize=14, fontweight='bold')
ax.set_xlabel('Elapsed Time (h)')
ax.set_ylabel('Volume')
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "volume_evolution.png", bbox_inches='tight', dpi=150)
plt.close(fig)
print("  → volume_evolution.png")

# --- Fig 3: Correlation heatmap ---
fig, ax = plt.subplots(figsize=(8, 6))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, mask=mask, annot=True, fmt='.3f', cmap='RdBu_r',
            center=0, square=True, linewidths=0.5, cbar_kws={'shrink': 0.8}, ax=ax)
ax.set_title('Correlation Matrix — Measurement Variables', fontsize=14, fontweight='bold')
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "correlation_heatmap.png", bbox_inches='tight', dpi=150)
plt.close(fig)
print("  → correlation_heatmap.png")

# --- Fig 4: VCD vs TCD scatter (they should be highly correlated) ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for batch in sorted(train_batches, key=lambda x: int(x.replace('Batch', '')))[:10]:
    subset = df_meas[df_meas['batch_id'] == batch].dropna(subset=['vcd', 'tcd'])
    axes[0].plot(subset['vcd'], subset['tcd'], marker='.', alpha=0.5, markersize=3)

axes[0].set_xlabel('VCD')
axes[0].set_ylabel('TCD')
axes[0].set_title('VCD vs TCD (per batch)', fontweight='bold')

# All training data pooled
pooled = df_meas[~df_meas['batch_id'].isin(predict_batches)].dropna(subset=['vcd', 'tcd'])
axes[1].scatter(pooled['vcd'], pooled['tcd'], alpha=0.3, s=10)
axes[1].set_xlabel('VCD')
axes[1].set_ylabel('TCD')
axes[1].set_title('VCD vs TCD (all training pooled)', fontweight='bold')

# Add y=x line
lims = [min(axes[1].get_xlim()[0], axes[1].get_ylim()[0]),
        max(axes[1].get_xlim()[1], axes[1].get_ylim()[1])]
axes[1].plot(lims, lims, 'k--', alpha=0.3, label='y=x (VCD=TCD)')
axes[1].legend()

fig.suptitle('VCD vs TCD Relationship', fontsize=14, fontweight='bold')
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "vcd_vs_tcd.png", bbox_inches='tight', dpi=150)
plt.close(fig)
print("  → vcd_vs_tcd.png")

# --- Fig 5: Feed events histogram ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Feed times histogram
axes[0].hist(df_feed['elapsed_time'].dropna(), bins=50, edgecolor='white', alpha=0.8)
axes[0].set_xlabel('Elapsed Time (h)')
axes[0].set_ylabel('Count')
axes[0].set_title('Distribution of Feed Event Times', fontweight='bold')

# Feed types per batch
feed_type_counts = {
    'feed_substrate': df_feed['feed_substrate'].notna().sum(),
    'feed_b': df_feed['feed_b'].notna().sum(),
    'feed_c': df_feed['feed_c'].notna().sum(),
    'feed_d': df_feed['feed_d'].notna().sum(),
}
axes[1].bar(feed_type_counts.keys(), feed_type_counts.values(), edgecolor='white', alpha=0.8)
axes[1].set_title('Number of Feed Events by Type', fontweight='bold')
axes[1].set_ylabel('Count')
for i, (k, v) in enumerate(feed_type_counts.items()):
    axes[1].text(i, v + 5, str(v), ha='center', fontsize=10)

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "feed_analysis.png", bbox_inches='tight', dpi=150)
plt.close(fig)
print("  → feed_analysis.png")

# --- Fig 6: Substrate consumption vs Byproduct accumulation ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for batch in sorted(train_batches, key=lambda x: int(x.replace('Batch', '')))[:12]:
    subset = df_meas[df_meas['batch_id'] == batch].dropna(subset=['substrate', 'byproduct'])
    axes[0].plot(subset['elapsed_time'], subset['substrate'], alpha=0.5, linewidth=1, marker='.', markersize=3)
    axes[1].plot(subset['elapsed_time'], subset['byproduct'], alpha=0.5, linewidth=1, marker='.', markersize=3)

axes[0].set_title('Substrate Over Time', fontweight='bold')
axes[0].set_xlabel('Elapsed Time (h)')
axes[0].set_ylabel('Substrate')
axes[1].set_title('Byproduct Over Time', fontweight='bold')
axes[1].set_xlabel('Elapsed Time (h)')
axes[1].set_ylabel('Byproduct')

fig.suptitle('Substrate Consumption & Byproduct Formation', fontsize=14, fontweight='bold')
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "substrate_byproduct.png", bbox_inches='tight', dpi=150)
plt.close(fig)
print("  → substrate_byproduct.png")

# --- Fig 7: Predict batches volume trajectories ---
fig, ax = plt.subplots(figsize=(14, 5))
for batch in sorted(predict_batches, key=lambda x: int(x.replace('Batch', ''))):
    subset = df_predict[df_predict['batch_id'] == batch]
    ax.plot(subset['elapsed_time'], subset['volume'], marker='o', linewidth=2, markersize=4, label=batch)
ax.set_title('Volume Trajectories — PREDICTION Batches (to_predict=True)', fontsize=14, fontweight='bold')
ax.set_xlabel('Elapsed Time (h)')
ax.set_ylabel('Volume')
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "predict_batches_volume.png", bbox_inches='tight', dpi=150)
plt.close(fig)
print("  → predict_batches_volume.png")

# --- Fig 8: Measurements per batch ---
fig, ax = plt.subplots(figsize=(16, 5))
batch_order = sorted(
    batch_summary['batch_id'].unique(),
    key=lambda x: int(x.replace('Batch', ''))
)
colors = ['#2196F3' if b not in predict_batches else '#FF5722' for b in batch_order]
bar_data = batch_summary.set_index('batch_id').loc[batch_order]
ax.bar(range(len(batch_order)), bar_data['n_measurements'], color=colors, edgecolor='white')
ax.set_xticks(range(len(batch_order)))
ax.set_xticklabels(batch_order, rotation=90, fontsize=7)
ax.set_ylabel('Number of Measurements')
ax.set_title('Measurements per Batch (blue=train, red=predict)', fontweight='bold')
ax.axhline(y=bar_data['n_measurements'].mean(), color='gray', linestyle='--', alpha=0.5, label=f'Mean ({bar_data["n_measurements"].mean():.1f})')
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "measurements_per_batch.png", bbox_inches='tight', dpi=150)
plt.close(fig)
print("  → measurements_per_batch.png")

print("\n" + "=" * 70)
print("DONE! All outputs in:", OUTPUT_DIR.resolve())
print("=" * 70)
