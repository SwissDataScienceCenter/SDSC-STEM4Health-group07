# Submission files

`submission_v9_best.csv` is the final submission.

| File | What it is | Leaderboard (vcd / tcd / substrate / byproduct → mean) |
|---|---|---|
| `submission_v9_best.csv` | **final**: best model per measurement | 6.627 / 6.569 / 6.326 / 7.314 → **6.709** |
| `submission_pa.csv` | ODE-only cells, sugar simulator days 5–8 × 1.02, statistical byproduct × 1.02 | 6.649 / 6.569 / 6.326 / 7.582 → 6.782 |
| `submission_pb.csv` | particle-simulator cells, sugar simulator weight 0.5, hybrid byproduct | 6.876 / 7.120 / 6.592 / 8.242 → 7.208 |
| `submission_pc.csv` | kernel ridge VCD, particle-simulator TCD, grid-search sugar, byproduct × 1.02 (× 1.04 days 5–8) | 6.627 / 7.081 / 6.357 / 7.314 → 6.845 |
| `submission_v8_mix.csv` | v1 cells + v4 sugar + v7 byproduct (not submitted) | known in advance: 6.850 |
| `submission_v7_single_shot.csv` | new small cell model + v4 sugar + v1 byproduct × 1.02 | 7.145 / 7.054 / 6.573 / 7.432 → 7.051 |
| `submission_v6_mix.csv` | v1 cells and byproduct + v4 sugar | 6.745 / 6.649 / 6.573 / 7.446 → 6.853 |
| `submission_v5.csv` | v4 with new cell signals | not known |
| `submission_v4.csv` | sugar simulators + small statistical models for cells and byproduct | 7.112 / 7.016 / 6.573 / 7.652 → 7.088 |
| `submission_v3_byproduct_phases.csv` | v2 with byproduct blended in three phases | not submitted |
| `submission_v2.csv` | first sugar simulator + small statistical models | not known |
| `submission_v1_leaderboard_7.018.csv` | old technique, version 1 | 6.745 / 6.649 / 7.234 / 7.446 → 7.018 |
| `submission_best.csv` | same content as `submission_v2.csv` (despite the name) | |
| `submission_nosearch.csv`, `submission_mechanistic_blend.csv`, `submission_mechanistic.csv`, `submission.csv` | early variants of the old technique | |

`bioreaction_template.csv` is the required format; `Leaderboard_template.ipynb` is the organisers' submission notebook.
