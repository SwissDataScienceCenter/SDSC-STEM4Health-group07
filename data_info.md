# Project 1 – DTB (Digital Twin of a Bioreactor): Data Explanation

## Overview

The data describes a **fed-batch mammalian cell culture bioreactor** used for biopharmaceutical production. The goal is to build a **digital twin** — i.e. a model that can predict key process variables at given future time points given the feeding schedule and reactor conditions.

There are two CSV files:

| File | Purpose |
|---|---|
| `bioreaction_remaining.csv` | Full historical data from ~60 bioreactor batches with **measured** process variables. Used for **training / validation**. |
| `bioreaction_template.csv` | A small subset of time points from ~12 batches where the key variables are **set to 0**. These must be **predicted** and submitted to the leaderboard. |

---

## Column-by-Column Explanation

| Column | Type | Description | Unit (likely) |
|---|---|---|---|
| `batch_id` | string | Unique identifier for each bioreactor batch. e.g. `Batch1`, `Batch24`, etc. | — |
| `elapsed_time` | float | Time elapsed since the start of the batch. Spans 0 to ~13 hours. | hours |
| `vcd` | float | **Viable Cell Density**. Number of living cells per unit volume. Increases over the batch as cells grow, then may plateau or decline. | ×10⁶ cells/mL |
| `tcd` | float | **Total Cell Density**. Total number of cells (viable + dead) per unit volume. Always ≥ vcd; the difference (tcd − vcd) is dead cell density. | ×10⁶ cells/mL |
| `substrate` | float | **Substrate concentration**. This is the main nutrient consumed by the cells (typically glucose). Starts at ~4.5–4.9 g/L and declines toward ~0.3–1.6 g/L as cells consume it. | g/L |
| `byproduct` | float | **Byproduct concentration**. Metabolic waste product produced by the cells (typically lactate). Starts near ~0.2–0.3 g/L, increases as cells metabolise glucose, then may plateau or decrease if cells switch metabolism. | g/L |
| `feed_substrate` | float | Amount of **substrate feed** added at specific feeding times. Also called a bolus feed. Sparse — only recorded at discrete feeding events. | L or kg |
| `feed_b` | float | Amount of **Feed B** added at specific times. An additional nutrient/component feed. | L or kg |
| `feed_c` | float | Amount of **Feed C** added at specific times. Another supplement feed. | L or kg |
| `feed_d` | float | Amount of **Feed D** added at specific times. Another supplement feed. | L or kg |
| `volume` | float | **Working volume** of the reactor. Increases over time as feeds are added. Starts around 10.3–10.6 L and ends around 15.3–17.2 L. | L |
| `to_predict` | bool | Flag that is `True` for rows that come from the **template**. These rows have `vcd`, `tcd`, `substrate`, and `byproduct` set to missing — they are the **prediction targets**. | — |

---

## Data Structure & Sparsity

**Not every measurement is taken at every time point!** The data is **sparse** by design:

- **~hourly** (at times 0, 1, 2, 3, …, 13 h): `vcd`, `tcd`, `substrate`, `byproduct`, `volume`
- **At feeding events** (irregular times): `feed_substrate`, `feed_b`, `feed_c`, `feed_d`
- **Between hourly readings** (finer-granularity times): sometimes only `feed_*` or `volume`

This results in a CSV where **most cells are NaN**. Here is an example from Batch 1:

| elapsed_time | vcd | tcd | substrate | byproduct | feed_substrate | feed_b | feed_c | feed_d | volume |
|---|---|---|---|---|---|---|---|---|---|
| 0.00 | 0.98 | 0.99 | 4.51 | 0.24 | — | — | — | — | 10.50 |
| 1.00 | 1.91 | 1.93 | 3.83 | 0.94 | — | — | — | — | 10.49 |
| 1.01 | — | — | — | — | 0.062 | — | — | — | — |
| 1.98 | 4.57 | 4.61 | 2.83 | 1.86 | — | — | — | — | 10.54 |
| 2.04 | — | — | — | — | 0.136 | — | — | — | — |
| 2.72 | — | — | — | — | — | 0.00085 | — | — | — |
| … | … | … | … | … | … | … | … | … | … |

Where `—` means NaN (no measurement at that time).

---

## Data Sizes

| File | Rows | Columns | Batches |
|---|---|---|---|
| `bioreaction_remaining.csv` | 2,860 | 13 | ~60 (varies by slice) |
| `bioreaction_template.csv` | ~140–160 | 6 | ~12 |

Template batches: `Batch5, Batch6, Batch12, Batch24, Batch32, Batch35, Batch40, Batch48, Batch54, Batch56, Batch57, Batch60`

---

## The Prediction Task

The objective is to predict **four target variables** at specific time points:

1. **vcd** — Viable Cell Density
2. **tcd** — Total Cell Density
3. **substrate** — Substrate concentration
4. **byproduct** — Byproduct concentration

The prediction time points are defined by `bioreaction_template.csv`: each row gives a `(batch_id, elapsed_time)` pair where these four quantities must be predicted. At those same time points, the feed amounts (`feed_substrate`, `feed_b`, `feed_c`, `feed_d`) and `volume` are given as inputs.

### Evaluation Metric

The leaderboard uses **Mean Absolute Percentage Error (MAPE)** on each target, then averages the four MAPEs to produce a single **mean_mape** score. Lower is better.

---

## Training Strategy Suggestions

Because the data is sparse and multi-variate:

1. **Interpolation / Imputation**: Hourly process variables (`vcd`, `tcd`, `substrate`, `byproduct`, `volume`) could be interpolated to a regular time grid, or missing values could be filled forward.

2. **Feeding integration**: Feed amounts are given as discrete additions (boluses). They can be converted to cumulative feed totals, or treated as impulses.

3. **Cross-batch modelling**: Each batch is an independent time series of 0–13 hours. Models can be:
   - Per-time-point regression models (predict each target at each prediction time independently)
   - Sequence models (LSTM, GRU, Transformer) that use the full history to predict all future points
   - Gaussian Processes or ODE-informed models (since this is a bioprocess that follows known kinetics)

4. **Train/test split**: Use batches that have complete data (i.e., not in the template list) for training and validation. The template batches are the final test set (leaderboard evaluation).

---

## Running the Extraction Script

```bash
python src/extract_features.py
```

The script loads both CSVs, prints summary statistics for every column, explains the data recording pattern, the prediction task, and the missing-data landscape.
