# Section 3: Study Design and Methodology (v2)

> **Version note:** This is v2, incorporating corrections from the scientific consistency
> audit (October 2026).  Changes from v1 are indicated inline with `[CORRECTED]` markers
> where relevant.  See `03_evidence_corrections_v2.md` for the full audit log.

---

## 3.1 Research Questions and Problem Formulation

Urban change detection from satellite Earth observation is critical for municipal tracking,
infrastructure monitoring, and land-use assessment. Bitemporal optical satellite imagery
(e.g., Sentinel-2) provides high spatial and spectral resolution, but operational monitoring
is frequently disrupted by persistent cloud cover, haze, and adverse illumination. Synthetic
Aperture Radar (SAR, e.g., Sentinel-1) provides all-weather, day-and-night active microwave
imaging, but suffers from speckle noise, geometric distortions (layover, foreshortening,
and shadow), and sensitivity to surface roughness and soil moisture fluctuations that do not
represent structural change.

This study investigates **asynchronous multimodal satellite change detection**, wherein
bitemporal SAR observations ($T_1$ and $T_2$) are combined with historical, pre-change
optical imagery ($T_1$) to detect urban surface changes under the realistic operational
constraint that contemporaneous post-change optical imagery ($T_2$) is unavailable.

We investigate five research questions (RQs):

- **RQ1 (Historical Guidance Utility):** Can historical Sentinel-2 optical imagery improve
  Sentinel-1 SAR-based change detection when contemporaneous post-change optical imagery
  is unavailable?
- **RQ2 (Temporal Gap Sensitivity):** How does the temporal separation between historical
  optical imagery and the target SAR observation affect change detection performance?
  **[UNTESTED: no multi-gap experiment was conducted; RQ2 is stated as a motivation
  but is not answered by the available results.]**
- **RQ3 (SAR Polarization Contribution):** Can additional SAR polarization information
  (VV vs. VH) improve discrimination between structural changes and environmental
  pseudo-changes?
  **[UNTESTED: no single-pol vs. dual-pol ablation was conducted; RQ3 is stated as a
  motivation but is not answered by the available results.]**
- **RQ4 (Comparative Architecture Hierarchy):** How does the asynchronous multimodal
  architecture compare against unimodal (SAR-only, optical-only), early fusion, and
  synchronous dual-stream baselines?
- **RQ5 (Operating Conditions and Error Modes):** Under what spatial and observational
  conditions does historical optical guidance provide performance benefits, and where
  does it fail?

---

## 3.2 Dataset Descriptions and Label Construction

Experiments are conducted on two distinct benchmark scenarios. Cross-dataset performance
figures are not directly comparable due to differences in spatial resolution, sensor
preprocessing, geographic coverage, label definitions, and evaluation conditions.
**[CORRECTED: explicit cross-dataset disclaimer added; see §3.2 note below.]**

### 3.2.1 Scenario 1: Onera Satellite Change Detection (OSCD)

The OSCD benchmark comprises Sentinel-2 optical imagery and coregistered Sentinel-1 SAR
composites across 24 global cities.

- **Optical Imagery:** Level-1C top-of-atmosphere reflectance Sentinel-2 imagery with 13
  spectral bands ($B_1$ to $B_{12}$, including $B_{8A}$) resampled to 10 m spatial
  resolution.
- **SAR Imagery:** Sentinel-1 C-band SAR Level-1 Ground Range Detected (GRD) intensity in
  Interferometric Wide (IW) swath mode, single polarization (VV). The SAR data are
  multi-temporal composites generated across orbital cycles. Exact acquisition timestamps
  for the composite are not available from the dataset metadata; therefore no precise
  temporal-gap characterisation is reported.
- **Labels:** Binary ground-truth change masks ($y \in \{0, 1\}$) annotated manually by
  remote sensing experts, indicating true urban changes (e.g., new building construction,
  earthworks, and road expansion). Natural seasonal variations in vegetation and water
  bodies are annotated as unchanged.

### 3.2.2 Scenario 2: SEN12-MT Urban Mapping (SpaceNet 7 Sites)

Scenario 2 evaluates a multimodal extension of the SEN12-MT Urban Mapping dataset,
covering 28 geographically distinct SpaceNet 7 regions of interest.

- **Temporal Pair:** A fixed bitemporal pair spanning **March 2018 ($T_1$)** to **December
  2019 ($T_2$)**, representing a uniform **21-month temporal separation** across all
  evaluated sites.
- **Optical Imagery:** Sentinel-2 Level-2A bottom-of-atmosphere surface reflectance,
  containing 10 spectral bands ($B_2, B_3, B_4, B_5, B_6, B_7, B_8, B_{8A}, B_{11},
  B_{12}$) resampled to 10 m spatial resolution. In asynchronous configurations, $T_2$
  optical data are strictly withheld.
- **SAR Imagery:** Sentinel-1 IW GRD bitemporal observations in dual polarization:
  Vertical transmit / Vertical receive (VV) and Vertical transmit / Horizontal receive
  (VH) for both dates.
- **Binary Change Proxy Derivation:** Ground-truth change annotations are not directly
  surveyed in this dataset. Instead, binary change proxy labels are derived by applying an
  XOR operation between bitemporal building-probability maps generated by deep building
  footprint segmentation models:
  $$\hat{y}_{i,j} = \left(\text{Building}_{T2}(i,j) \ge 0.50\right) \oplus \left(\text{Building}_{T1}(i,j) \ge 0.50\right)$$
  where $\hat{y}_{i,j} = 1$ indicates a transition between non-building and building
  status (or vice versa). **These labels serve as proxy indicators of structural building
  development, not independently validated environmental, agricultural, or general
  land-cover change annotations. Errors in the upstream building segmentation models
  propagate into the proxy labels.**

### Cross-dataset comparability note

The OSCD and Scenario 2 benchmarks differ in sensor preprocessing, label construction
method, number of training and test sites, class imbalance, SAR polarisation mode, and
available validation strategies. Their absolute F1 scores should not be treated as a
controlled comparison of task difficulty. Lower absolute F1 on Scenario 2 relative to
OSCD is consistent with these differences but does not, on its own, establish that
Scenario 2 is an inherently harder task. **[CORRECTED — v1 made an unsupported
cross-dataset difficulty claim; this is replaced by an explicit non-comparability
statement.]**

---

## 3.3 Geographic Splits and Data Leakage Prevention

To prevent spatial autocorrelation and geographic data leakage, all splits are partitioned
strictly by non-overlapping geographic sites:

### 3.3.1 OSCD Geographic Partition

The 24 cities are partitioned into three mutually disjoint subsets:

- **Training Subset (10 cities):** `bordeaux`, `nantes`, `rennes`, `saclay_e`, `abudhabi`,
  `cupertino`, `pisa`, `hongkong`, `beirut`, `mumbai`.
- **Validation Subset (4 cities):** `beihai`, `bercy`, `aguasclaras`, `paris`.
- **Test Subset (10 official test cities):** `brasilia`, `montpellier`, `norcia`, `rio`,
  `saclay_w`, `valencia`, `dubai`, `lasvegas`, `milano`, `chongqing`.

*Note on terminology:* This is a $10 / 4 / 10$ city split (24 cities total). Validation
cities are isolated during training and used exclusively for model checkpoint selection.

### 3.3.2 Scenario 2 Locked Geographic Split

The 28 SpaceNet 7 sites are split according to Strategy B, locked and verified in
`experiments/scenario_2/configs/split_definition.json`
(`SHA256: 5d0841adf0c651166cb18ecea310c6ef8cce69172065fc2d699a29083c272e27`):

- **Training Set (20 sites):** `L15-0331E-1257N`, `L15-0357E-1223N`, `L15-0358E-1220N`,
  `L15-0368E-1245N`, `L15-0434E-1218N`, `L15-0506E-1204N`, `L15-0544E-1228N`,
  `L15-0632E-0892N`, `L15-1138E-1216N`, `L15-1204E-1204N`, `L15-1209E-1113N`,
  `L15-1276E-1107N`, `L15-1296E-1198N`, `L15-1335E-1166N`, `L15-1389E-1284N`,
  `L15-1439E-1134N`, `L15-1617E-1207N`, `L15-1669E-1153N`, `L15-1672E-1207N`,
  `L15-1703E-1219N`.
- **Validation Set (4 sites):** `L15-0457E-1135N`, `L15-0487E-1246N`,
  `L15-0977E-1187N`, `L15-1716E-1211N`.
- **Test Set (4 sites):** `L15-0924E-1108N`, `L15-1289E-1169N`,
  `L15-1298E-1322N`, `L15-1438E-1134N`.

---

## 3.4 Preprocessing, NaN Handling, and Input Channels

### 3.4.1 Preprocessing and Normalization

- Optical reflectances are pre-normalised into $[0.0, 1.0]$ in the dataset files; no
  further scaling is applied.
- SAR backscatter intensities are pre-normalised into $[0.0, 1.0]$ in the dataset files.
  In Scenario 2, raw SAR tiles contain invalid nodata pixels represented as NaNs resulting
  from sensor orbit boundaries and swath edges. The preprocessing pipeline reads each SAR
  raster, records pixel validity based on the original NaN indicators, and then fills NaNs
  with zero:

  $$V_t(i,j) = \mathbf{1}\!\left[\text{neither VV nor VH is NaN at time } t\right] \quad \text{(recorded before filling)}$$
  $$\text{SAR}_t[i,j] \leftarrow \begin{cases} \text{SAR}_t[i,j] & V_t(i,j)=1 \\ 0.0 & V_t(i,j)=0 \end{cases}$$

- A joint pixel validity mask $M(i,j)$ is then constructed as:
  $$M(i,j) = V_{T1}(i,j) \land V_{T2}(i,j)$$
  where $V_t(i,j) = 1$ if and only if neither the VV nor the VH channel contains NaN at
  observation time $t$.

  **[CORRECTED: v1 described validity as based on zero-valued SAR after filling.
  The implementation records validity from NaN indicators in the original raster
  *before* NaN-to-zero replacement. A SAR pixel that is genuinely zero-valued (not
  NaN) would be considered valid. Source:
  `src/datasets/scenario2_dataset.py`, `_read_s1_with_validity`, line 113.]**

### 3.4.2 Input Channel Summary

| Scenario & Model | SAR Channels ($T_1, T_2$) | Optical Channels ($T_1, T_2$) | Total |
|---|:---:|:---:|:---:|
| **OSCD SAR-only U-Net** | 2 (VV $T_1$, VV $T_2$) | 0 | 2 |
| **OSCD Optical-only U-Net** | 0 | 26 (13 bands $T_1$, 13 bands $T_2$) | 26 |
| **OSCD Sync Early Fusion U-Net** | 2 (VV $T_1$, VV $T_2$) | 26 (13 bands $T_1$, 13 bands $T_2$) | 28 |
| **OSCD Sync Dual-Stream CNN** | 2 (VV $T_1$, VV $T_2$) | 26 (13 bands $T_1$, 13 bands $T_2$) | 28 |
| **OSCD Naive Async CNN** | 2 (VV $T_1$, VV $T_2$) | 26 (13 bands $T_1$, 13 zero-filled $T_2$) | 28 |
| **OSCD Clean Async CNN** | 2 (VV $T_1$, VV $T_2$) | 13 (13 bands $T_1$; $T_2$ withheld) | 15 |
| **OSCD Transformer Self-Attn** | 2 (VV $T_1$, VV $T_2$) | 13 (13 bands $T_1$; $T_2$ withheld) | 15 |
| **OSCD Transformer Cross-Attn** | 2 (VV $T_1$, VV $T_2$) | 13 (13 bands $T_1$; $T_2$ withheld) | 15 |
| **Scenario 2 Async Dual-Stream CNN** | 4 (VV, VH $T_1$; VV, VH $T_2$) | 10 (10 bands $T_1$; $T_2$ withheld) | 14 |
| **Scenario 2 Sync Dual-Stream CNN** | 4 (VV, VH $T_1$; VV, VH $T_2$) | 20 (10 bands $T_1$; 10 bands $T_2$) | 24 |
| **Scenario 2 Cross-Attention Transformer** | 4 (VV, VH $T_1$; VV, VH $T_2$) | 10 (10 bands $T_1$; $T_2$ withheld) | 14 |

---

## 3.5 Model Architectures and Parameter Counts

All deep neural architectures were verified against saved model definitions and checkpoint
state dictionaries.

1. **Dual-Stream U-Net (CNN):**
   - Comprises two parallel encoder streams: a SAR encoder and an Optical encoder. Each
     encoder employs four double-convolution stages with batch normalization and ReLU
     activations, and max-pooling ($64, 128, 256, 512$ feature channels).
   - In the bottleneck, deep spatial representations from both modalities are concatenated.
   - A single shared decoder reconstructs the spatial resolution using transposed
     convolutions, skip connections from both modality encoders, and a final $1 \times 1$
     convolution generating logit predictions.
   - **Naive Asynchronous CNN:** The optical encoder receives 13 optical $T_1$ bands
     concatenated with 13 zero-filled placeholder channels (representing the withheld
     $T_2$ optical data), totalling 28 input channels. The zero-filling is done at the
     input tensor level before training.
   - **Clean Asynchronous CNN:** The optical encoder receives exclusively the 13 historical
     $T_1$ bands (15 total input channels: 2 SAR + 13 optical), with no zero-filled
     placeholders. This results in a different first-layer weight tensor and a smaller
     total parameter count. **[CORRECTED: v1 stated both CNNs have the same parameter
     count. They differ due to the different optical encoder input width: 27,284,097
     (Naive, 28-channel input) vs. 27,276,609 (Clean, 15-channel input), a difference
     of 7,488 parameters in the first encoder conv layer. Source:
     `src/models/dual_stream_unet_clean.py`, docstring line 23.]**

2. **Transformer Cross-Attention:**
   - Patch embedding layer with patch stride 4 maps SAR and optical inputs to an embedding
     dimension $D=128$.
   - Two self-attention blocks process SAR and optical tokens independently.
   - A cross-attention module queries the historical optical tokens: SAR patch tokens act
     as **Query ($Q$)**, while historical optical tokens act as **Key ($K$)** and
     **Value ($V$)**:
     $$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
   - A lightweight convolutional decoder upsamples token representations to full spatial
     resolution.

### Verified Parameter Counts

| Scenario | Model Architecture | Trainable Parameters |
|---|---|---:|
| **OSCD** | SAR-only U-Net | 31,037,057 |
| **OSCD** | Optical-only U-Net | 31,050,881 |
| **OSCD** | Sync Early Fusion U-Net | 31,052,033 |
| **OSCD** | Sync Dual-Stream CNN | 27,284,097 |
| **OSCD** | Naive Async Dual-Stream CNN | 27,284,097 |
| **OSCD** | Clean Async Dual-Stream CNN | 27,276,609 |
| **OSCD** | Transformer Self-Attention | 1,358,081 |
| **OSCD** | Transformer Cross-Attention | 1,156,993 |
| **Scenario 2** | Async Dual-Stream CNN | 27,276,033 |
| **Scenario 2** | Sync Dual-Stream CNN | 27,281,793 |
| **Scenario 2** | Cross-Attention Transformer | 1,154,945 |

*Note: Scenario 2 and OSCD models of the same architecture type have different parameter*
*counts because Scenario 2 uses 4-channel dual-pol SAR and 10-band optical, whereas OSCD*
*uses 2-channel single-pol SAR and 13-band optical. The change in first-layer input width*
*accounts for the parameter differences.*

---

## 3.6 Loss Functions and Class Imbalance Handling

Change detection in satellite imagery suffers from extreme class imbalance, where unchanged
background pixels vastly outnumber change pixels.

### 3.6.1 Standard Weighted Binary Cross-Entropy

For pixel logit $z_{i,j}$ and binary target $y_{i,j} \in \{0, 1\}$, the unreduced weighted
BCE loss is:
$$\ell(z_{i,j}, y_{i,j}) = - \left[ w \cdot y_{i,j} \cdot \log \sigma(z_{i,j}) + (1 - y_{i,j}) \cdot \log (1 - \sigma(z_{i,j})) \right]$$
where $\sigma(z) = \frac{1}{1 + e^{-z}}$ and $w$ is the positive-class weighting factor
(`pos_weight`).

**OSCD Loss Configurations:**

- All 6 CNN baselines (SAR-only, Optical-only, Sync Fusion, Sync Dual-Stream, Naive
  Async, Clean Async) were trained using **standard BCE with $w = 1.0$** (no
  `pos_weight` key present in any CNN config file). In PyTorch,
  `BCEWithLogitsLoss` with no `pos_weight` argument uses $w = 1$.
- Both Transformer baselines (Self-Attention, Cross-Attention) were trained using
  **weighted BCE with $w = 41.0$** (derived from the training set pixel ratio:
  126,312 change / 5,146,327 no-change $\approx 1:40.7$).

**[CLARIFICATION: "standard unweighted BCE" in v1 was shorthand for $w=1$, i.e., no
positive-class weighting. The distinction from "weighted BCE" is correct. The shorthand
is retained where unambiguous but the full description is given here.]**

**Scenario 2 Loss Configuration:**

Both the standard and masked loss conditions in Scenario 2 use **weighted BCE with
$w = 94.0$** (derived from training pixel ratio: 50,387 change / 4,718,348 no-change
$= 93.64 \to 94.0$). **[CORRECTED: v1 described the standard condition as "unweighted."
Both conditions use `pos_weight=94.0`. Source: `train_scenario2_dual_stream.py` line 345;
`train_scenario2_sync_multiseed.py` lines 317–322.]**

### 3.6.2 Masked Weighted BCE Formulation (`MaskedBCEWithLogitsLoss`)

In Scenario 2, invalid nodata pixels in SAR rasters are filled with zeros after recording
their validity status. Under standard weighted BCE, these zero-filled pixels still receive
gradient updates (their logit predictions and targets are included in the loss sum and
divided by the batch pixel count). Under Masked Weighted BCE, gradients are restricted to
valid pixels:

$$\mathcal{L}_{\text{masked}} = \frac{\displaystyle\sum_{i,j} M(i,j) \cdot \ell(z_{i,j}, y_{i,j})}{\displaystyle\sum_{i,j} M(i,j)}$$

where $M(i,j) \in \{0, 1\}$ is the SAR validity mask (constructed from pre-fill NaN
indicators; see §3.4.1) and $\ell(\cdot)$ is the per-pixel weighted BCE with
$w = 94.0$.

The **effect on gradients** is: invalid pixels (where $M=0$) contribute zero to the
numerator and are excluded from the denominator. The effective loss is therefore a
mean over valid pixels only. Valid pixels receive the same gradient as under standard
weighted BCE, scaled by the ratio of all pixels to valid pixels.

The **effect on the denominator** under standard weighted BCE is division by all pixels
(batch_size × H × W). Under masked BCE the denominator is $\sum M$, which varies
per batch depending on how many valid SAR pixels fall within the sampled crops.

If a batch contains no valid pixels ($\sum M = 0$), the loss safely reduces to
$0 \cdot \sum z$ to preserve gradient connectivity without computing NaN. Source:
`src/models/masked_loss.py`.

---

## 3.7 Training Pipeline and Deterministic Sampling Controls

- **Optimization:** Adam optimizer ($\beta_1=0.9, \beta_2=0.999$, weight decay $0$),
  initial learning rate $\alpha = 10^{-4}$.
- **Patch Sampling:** 64 crops per site per epoch, crop size $32 \times 32$ pixels.
  Batch size 64. Total budget: 50 epochs.
- **Scenario 2 Matched Multi-Seed Protocol:**
  To eliminate spatial sampling and initialization confounds across seeds 43 and 44,
  a deterministic crop coordinate schedule was pre-generated using a separate
  `numpy.random.RandomState(seed)` RNG (independent of PyTorch's global RNG), based on
  the verified spatial dimensions of all 20 training sites:
  $$\text{top}_{i,k} \sim \mathcal{U}(0, H_i - 32), \quad \text{left}_{i,k} \sim \mathcal{U}(0, W_i - 32)$$
  For each seed, both the standard-weighted-BCE and masked-weighted-BCE models were
  initialized with identical parameter weights ($\Delta \theta = 0.0$, confirmed by
  preflight state_dict comparison) and trained on identical sequences of spatial crops
  in identical order. Seeds 43 and 44 produce different initializations and different
  crop schedules from each other and from seed 42.

- **Checkpoint Selection Criteria:**
  - **OSCD:** Highest validation F1 score evaluated on full scenes of the 4 validation
    cities (all-pixel protocol).
  - **Scenario 2 Baselines (Seed 42):** Highest validation F1 on full scenes of the 4
    validation sites under the All-Pixel Protocol (`val_f1_all`) for the standard loss
    runs; under the Valid-Pixel Protocol (`val_f1_valid`) for the masked loss runs.
    This difference in selection criterion is a confound in the seed-42 comparison.
  - **Scenario 2 Multi-Seed (Seeds 43 & 44):** Predeclared as highest validation F1 on
    the 4 validation sites under the **Valid-Pixel Protocol** (`val_f1_valid`) for all
    four matched runs. Both `val_f1_valid` and `val_f1_all` were recorded at every epoch.

---

## 3.8 Evaluation Metrics and Protocol Definitions

Predictions are generated by applying a threshold $\tau$ to predicted probabilities
$p_{i,j} = \sigma(z_{i,j})$:
$$\hat{Y}_{i,j} = \mathbb{I}(p_{i,j} \ge \tau)$$
Evaluations are computed from pooled confusion counts across all test sites: True Positives
($TP$), False Positives ($FP$), False Negatives ($FN$), and True Negatives ($TN$).

The performance metrics are defined as:
$$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}$$
$$\text{F1-score} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$
$$\text{Intersection over Union (IoU)} = \frac{TP}{TP + FP + FN}$$
$$\text{False Alarm Rate (FAR)} = \frac{FP}{FP + TN}$$

### Dual-Protocol Evaluation in Scenario 2

1. **All-Pixel Protocol:** Evaluates all pixels across full scene rasters, including
   pixels where SAR nodata was filled with zeros.
2. **Valid-Pixel Protocol:** Restricts evaluation to pixels where $M(i,j) = 1$, i.e.,
   where neither VV nor VH contained NaN at either time step in the original rasters.
   Pixels with filled-zero SAR that were NaN-in-origin are excluded.

   **[CORRECTED: v1 defined valid-pixel as "SAR data is non-zero in both time steps."
   The actual validity criterion is based on the original NaN indicators recorded before
   zero-filling. Genuinely zero-valued (non-NaN) SAR pixels are considered valid.
   Source: `src/datasets/scenario2_dataset.py`, `_read_s1_with_validity`, line 113.]**

### Thresholding Protocols

- **Primary Threshold:** Fixed default threshold **$\tau = 0.50$**.
- **Secondary Validation-Calibrated Operating Point ($\tau^*$):** A sweep over
  $\tau \in [0.01, 0.99]$ in increments of $0.01$ is performed exclusively on the pooled
  validation scenes to identify $\tau^*$ maximizing validation F1 (valid-pixel). The
  resulting $\tau^*$ is then applied to the test scenes as an empirical operating-point
  diagnostic. Test labels are never accessed during calibration. **Results at $\tau^*$
  are secondary and descriptive; the differences between models at their respective
  $\tau^*$ values are not a controlled comparison because each model may select a
  different $\tau^*$.**
