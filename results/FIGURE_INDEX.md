# Figure Index

All figures are in `results/figures/` unless noted.  DPI = 300.  PDF copies accompany every PNG.

---

## Training curves

### training_loss.png
- **Shows:** Epoch-by-epoch training loss for Clean Async CNN, Transformer Self-Attention, Transformer Cross-Attention.
- **Source data:** `experiments/asynchronous_dual_stream_clean/training_history.csv`, `experiments/transformer_self_attention/training_history.csv`, `experiments/transformer_cross_attention/training_history.csv`
- **Paper section:** Methods → Training details / Appendix
- **Suggested caption:** "Training loss per epoch for the three asynchronous models. Loss is mean batch BCE over the 10 training cities."
- **Notes:** CNN uses unweighted BCE; Transformers use pos_weight=41. Scales are therefore not directly comparable.

### validation_f1.png
- **Shows:** Validation F1 per epoch for the same three models. None values (no positive predictions) shown as gaps.
- **Source data:** same training history CSVs
- **Paper section:** Methods → Training details / Appendix
- **Suggested caption:** "Validation F1 per epoch. Gaps indicate epochs where no positive predictions were produced (F1 undefined). Best validation epoch is used for checkpoint selection."
- **Caveats:** Transformer Self-Attention produces F1 values but they are noisy due to instability of predictions on the full validation city images.

---

## Model comparison

### f1_comparison.png
- **Shows:** Test F1 bar chart, all 8 models.
- **Paper section:** Results → Overall model comparison
- **Suggested caption:** "Pooled test F1 scores across all evaluated models. Metrics computed on the 10 official OSCD test cities."

### iou_comparison.png
- **Shows:** Test IoU bar chart, all 8 models.
- **Paper section:** Results → Overall model comparison
- **Suggested caption:** "Pooled test IoU scores across all evaluated models."

### far_comparison.png
- **Shows:** Test FAR bar chart, all 8 models. FAR = FP / (FP + TN).
- **Paper section:** Results → Overall model comparison
- **Suggested caption:** "False alarm rate across all models. Lower is better. Transformer Self-Attention has the highest FAR (0.239)."
- **Caveats:** FAR axis is capped at 0.30 for readability; Transformer Self-Attention FAR (0.239) is visible; Self-Attention FAR on individual cities reaches up to 0.994.

### precision_recall_comparison.png
- **Shows:** Precision vs Recall scatter plot, all 8 models, with iso-F1 contours.
- **Paper section:** Results → Precision/recall trade-off
- **Suggested caption:** "Precision vs recall for all models on the pooled OSCD test set. Dashed curves show iso-F1 contours at F1 = 0.2, 0.3, 0.4."

### parameter_comparison.png
- **Shows:** Parameter count (log scale), all 8 models.
- **Paper section:** Results → Efficiency
- **Suggested caption:** "Model parameter counts. Transformer models have approximately 20× fewer parameters than the CNN baselines."

---

## Per-city analysis

### per_city_f1_comparison.png
- **Shows:** Per-city F1 grouped bar chart (10 test cities × 3 models). Undefined values plotted as 0.
- **Source data:** per_city_metrics.csv for each model
- **Paper section:** Results → Per-city analysis
- **Suggested caption:** "Per-city F1 scores on the 10 OSCD test cities. Undefined values (zero TP and zero FP) are plotted as 0 and labelled in the text."

### per_city_far_comparison.png
- **Shows:** Per-city FAR grouped bar chart. Note: Self-Attn Dubai/Rio FAR exceeds 0.9, clipping the axis.
- **Paper section:** Results → Per-city analysis
- **Suggested caption:** "Per-city false alarm rate. Note the extreme FAR for Transformer Self-Attention on Dubai (0.994) and Rio (0.894), indicating near-total false positive flooding on those cities."
- **Caveats:** Y-axis capped at 1.05; Self-Attention Dubai/Rio values are at the cap.

### per_city_precision_recall.png
- **Shows:** Per-city precision vs recall scatter (3 subplots, one per model).
- **Paper section:** Results → Per-city analysis / Appendix

---

## Qualitative examples

All panels are in `results/qualitative/example_0X/`.

### example_01 / montpellier_panel.png
- **Type A:** Clear successful urban change detection
- **Shows:** SAR T1, SAR T2, historical S2 T1 (RGB), ground truth, CNN prediction, Cross-Attn prediction.
- **City:** Montpellier (test)
- **Notes:** Both models detect the main change regions. CNN F1=0.271, Cross-Attn F1=0.393.

### example_02 / dubai_panel.png
- **Type B:** CNN performs better; Transformer produces high false positives
- **City:** Dubai (test)
- **Notes:** CNN F1=0.479, Cross-Attn F1=0.381. Cross-Attn FAR=0.234 vs CNN FAR=0.057.

### example_03 / lasvegas_panel.png
- **Type C:** Cross-Attention outperforms CNN
- **City:** Las Vegas (test)
- **Notes:** Cross-Attn F1=0.529, CNN F1=0.452. Both precision and recall improve.

### example_04 / rio_panel.png
- **Type D:** Difficult scene / high false-alarm case
- **City:** Rio (test)
- **Notes:** Both models have elevated FAR (CNN 0.025, Cross-Attn 0.317). Scene with complex land cover.

### example_05 / norcia_panel.png
- **Type E:** Subtle/small urban change
- **City:** Norcia (test)
- **Notes:** CNN F1=0.260, Cross-Attn F1=0.319. Small change area relative to scene size.

---

## Error analysis

All error maps in `results/error_analysis/{city}/`. Green=TP, Red=FP, Blue=FN, White=TN.

### {city}_{model}_error_analysis.png (montpellier, dubai, lasvegas, rio)
- **Shows:** Ground truth, binary prediction, colour-coded TP/FP/FN/TN map, and probability heatmap.
- **Paper section:** Results → Error analysis
- **Suggested caption:** "Error map for {city}. Green: true positive. Red: false positive. Blue: false negative."
- **Caveats:** Error types are labelled by pixel outcome only. Causal attribution (e.g., speckle, shadow) requires manual inspection and is not claimed here.

---

## Confusion matrix

### confusion_matrix_comparison.png
- **Shows:** Normalised 2×2 confusion matrices for Clean Async CNN, Cross-Attn, Self-Attn (pooled test set).
- **Paper section:** Results → Precision/recall trade-off
- **Suggested caption:** "Normalised confusion matrices (percentage of all pixels) for the three asynchronous models on the pooled OSCD test set."

---

## Tables (in results/tables/)

### final_model_comparison.csv
All 8 models, test F1/IoU/FAR/Precision/Recall/params.

### per_city_comparison.csv
F1, FAR, Precision, Recall for each of the 10 test cities × 3 asynchronous models.

### experiment_summary.json
Machine-readable summary of all metrics, pixel counts, history availability, and experimental notes.
