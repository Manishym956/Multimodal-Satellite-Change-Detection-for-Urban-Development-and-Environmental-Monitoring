# Results Audit

Read-only audit of the technical evidence package.
Generated after all figures, tables, qualitative examples, and error maps were produced.

---

## Audit checklist

| # | Claim | Evidence | Status |
|---|---|---|---|
| 1 | No model was trained during results generation | `generate_results.py` calls only `torch.load` and `model.eval()`. No `optimizer.step()`, no `loss.backward()`, no training loop. | PASS |
| 2 | No model architecture was changed | Models loaded from frozen checkpoints. `DualStreamUNetClean` and `TransformerCrossAttention` instantiated with the same hyperparameters recorded in each `metrics.json`. | PASS |
| 3 | No dataset files were modified | Rasterio and OpenCV used in read-only mode (`rasterio.open`, `cv2.resize`). No write calls on any GeoTIFF. | PASS |
| 4 | Train/validation/test split unchanged | Split defined by `split_train_cities(seed=42)` in `train_sar_baseline.py`. Qualitative examples and error maps use only the 10 official test cities. | PASS |
| 5 | All quantitative figures use recorded test metrics | Bar charts and scatter plots read from the locked `MODELS` dict populated from `metrics.json` values. No computation of new metrics from raw predictions. | PASS |
| 6 | All qualitative examples come from official test cities | Cities used: Montpellier, Dubai, Las Vegas, Rio, Norcia — all in `test_cities` list from every `metrics.json`. | PASS |
| 7 | Predictions use the correct checkpoints | CNN: `checkpoints/asynchronous_dual_stream_clean/best.pt` (epoch 26, val F1 0.0992). Cross-Attn: `checkpoints/transformer_cross_attention/best.pt` (epoch 33, val F1 0.1428). Both confirmed to exist. | PASS |
| 8 | Clean Async CNN uses SAR T1/T2 + historical S2 T1 only | `prepare_city()` in `generate_results.py` loads `imgs_1_rect` only. No `imgs_2_rect` reference. Input is `(1,15,H,W)` assembled as `cat(s1_t1, s1_t2, s2_t1)`. | PASS |
| 9 | Transformer Cross-Attention uses SAR as Query, optical as Key/Value | Confirmed in `transformer_cross_attention.py`: `CrossAttentionBlock.forward(x_sar, x_opt)` sets `q = norm_q(x_sar)`, `kv = norm_kv(x_opt)`. Unchanged model file. | PASS |
| 10 | S2 T2 never used in asynchronous inference | No `imgs_2_rect` path opened in `prepare_city()`. `optical_t2_handling` field in both transformer `metrics.json` files: "S2 T2 never loaded or used". | PASS |
| 11 | No fabricated values | All metrics sourced from `experiments/*/metrics.json` and `experiments/*/per_city_metrics.csv`, both written by the training scripts at evaluation time. | PASS |
| 12 | Undefined per-city F1 preserved as undefined in tables | `per_city_comparison.csv` preserves empty strings from CSV source. Figures note that undefined values are plotted as 0 with a text annotation on the figure. | PASS WITH LIMITATION — see note below |
| 13 | Training history not fabricated | Three training history CSVs confirmed present and loaded: `asynchronous_dual_stream_clean`, `transformer_self_attention`, `transformer_cross_attention` (50 epochs each). No histories invented. | PASS |
| 14 | SAR-only, optical-only, and synchronous CNN metrics not re-evaluated | Those numbers are taken directly from their locked `metrics.json` files and not recomputed. | PASS |
| 15 | Validation/test limitations not hidden | FIGURE_INDEX.md documents that (a) validation F1 is computed on 4 validation cities, not test cities; (b) undefined F1 values exist for Self-Attn on 4 cities; (c) CNN/transformer loss functions differ (pos_weight). | PASS |

---

## Limitation notes

**L1 — Undefined F1 plotted as 0 in grouped bar charts**
Transformer Self-Attention has undefined F1 (TP=0, FP=0) for brasilia, norcia, valencia, milano. The grouped bar chart `per_city_f1_comparison.png` renders these as 0 for visual completeness. This is noted on the figure and in FIGURE_INDEX.md. In the paper, these must be labelled "undefined" in any table, not "0.000".

**L2 — Training loss curves are not directly comparable across loss functions**
The Clean Async CNN uses `BCEWithLogitsLoss` (no pos_weight). Both Transformer models use `BCEWithLogitsLoss(pos_weight=41.0)`, which inflates the loss magnitude roughly 41× for positive pixels. The `training_loss.png` figure shows all three on the same axes. The different scales are expected and explained in the figure notes, but the y-axis starts at 0, so the CNN loss (range ~0.1–0.5) and transformer loss (range ~0.8–1.4) appear on separate visual bands. This is not a fabrication — it is the actual recorded loss — but it should be acknowledged in the paper.

**L3 — Qualitative predictions are single-orbit snapshots**
Each city is evaluated on the first listed orbit (e.g. Norcia orbit 117). For cities with multiple orbits (e.g. Valencia: 4 orbits, Norcia: 4 orbits) the visual appearance would differ per orbit. The choice of first orbit is consistent with the evaluation protocol used in all training scripts.

**L4 — No training histories for SAR-only, optical-only, fusion, dual-stream, or naive-async CNN**
Those five experiments predate the training-history CSV logging that was added for the clean async and transformer runs. Their training histories are not available on disk. Only test metrics were recorded. This is documented in `experiment_summary.json` under `training_histories_missing`.

**L5 — Validation F1 curves for transformers show high noise**
The Transformer Self-Attention validation F1 oscillates between ~0.004 and ~0.133 across 50 epochs. This is genuine behaviour from the saved `training_history.csv` and is not smoothed or filtered. The curve is plotted as-is.

---

## File inventory — all generated artifacts

### results/figures/ (PNG + PDF pairs)
- `training_loss.png` / `.pdf`
- `validation_f1.png` / `.pdf`
- `f1_comparison.png` / `.pdf`
- `iou_comparison.png` / `.pdf`
- `far_comparison.png` / `.pdf`
- `precision_recall_comparison.png` / `.pdf`
- `parameter_comparison.png` / `.pdf`
- `per_city_f1_comparison.png` / `.pdf`
- `per_city_far_comparison.png` / `.pdf`
- `per_city_precision_recall.png` / `.pdf`
- `confusion_matrix_comparison.png` / `.pdf`

### results/qualitative/
- `example_01/` — Montpellier (Type A: clear urban change)
- `example_02/` — Dubai (Type B: high transformer FAR)
- `example_03/` — Las Vegas (Type C: cross-attention outperforms CNN)
- `example_04/` — Rio (Type D: high false-alarm scene)
- `example_05/` — Norcia (Type E: subtle small-area change)

Each example contains: `{city}_panel.png`, `{city}_sar_t1.png`, `{city}_sar_t2.png`, `{city}_s2_rgb.png`, `{city}_ground_truth.png`, `{city}_pred_cnn.png`, `{city}_pred_cross.png`

### results/error_analysis/
- `montpellier/` — `{cnn,cross_attn}_error_analysis.png`, `_false_positives.png`, `_false_negatives.png`
- `dubai/` — same
- `lasvegas/` — same
- `rio/` — same

### results/tables/
- `final_model_comparison.csv` — all 8 models, pooled test metrics
- `per_city_comparison.csv` — 10 test cities × 3 asynchronous models
- `experiment_summary.json` — machine-readable summary

### results/
- `FIGURE_INDEX.md` — per-figure documentation
- `RESULTS_AUDIT.md` — this file

---

## Overall audit result

**PASS WITH LIMITATIONS**

All 15 checklist items pass. Four limitations are documented (L1–L5) and none involve fabricated data, changed checkpoints, or protocol violations. They are presentation and comparability issues that should be addressed in paper text.

---

## Scientific issues to address before writing the paper

1. **Clarify loss-function difference in any model comparison table or figure.** The CNN baselines and the transformer baselines used different loss functions (unweighted vs pos_weight=41). State this explicitly. Do not present the two sets as trained under identical conditions without this caveat.

2. **Do not report Transformer Self-Attention per-city F1 as 0.000 for brasilia, norcia, valencia, milano.** These are undefined (no positive predictions). Report them as "—" or "N/A" in any table.

3. **Validation F1 is not an unbiased estimate of test performance.** It is computed on 4 held-out training cities, not the official test set. The best checkpoint is selected on this validation signal, which is appropriate for model selection, but the validation F1 values themselves should not be compared directly to test F1 values in the paper.

4. **The temporal gap question in the research question cannot be answered from this data.** Sentinel-1 acquisition dates are unknown (the files are 3-month Earth Engine means without stored scene timestamps). The paper should explicitly state that the temporal-gap experiment is infeasible with the current data, as documented in `experiments/asynchronous_data_feasibility.md`.

5. **Single-seed results.** All experiments used seed 42 only. Variance across seeds is unknown. This should be noted as a limitation.

6. **OSCD test set size.** 10 test cities with varying sizes. Per-city standard deviations are large (e.g., F1 std of 0.12–0.18 across cities for the best models), indicating high scene-to-scene variability. Pooled metrics summarise a dataset of modest size. This should be acknowledged.
