# Does historical Sentinel-2 imagery help Sentinel-1 urban change detection without a post-change optical image? A controlled multi-seed re-examination on OSCD

**Manish Y. M.**\
*Polaris School of Technology, Medhavi Skills University*\
manishym946@gmail.com

---

## Abstract

When the post-change Sentinel-2 image is missing or cloudy, a SAR–optical change detector must rely on Sentinel-1 and,
at best, an older optical image. Saha et al. introduced this setting on the multi-sensor OSCD benchmark and reported that
the pre-change optical image raised F1 above SAR-only baselines. We re-examine the contribution of that historical image
under a controlled protocol: the official dual-stream U-Net (DS-UNet) and U-Net baselines, 10 paired training seeds per
model, validation-only model and threshold selection, and a repository pre-registration before each test evaluation. In
a DS-UNet trained with modality dropout, the historical image made no detectable F1 difference (ΔF1 +0.007, 95% CI
[−0.004, +0.018]). In one early-fusion U-Net trained the same way it lowered F1 (−0.091; −0.116 on 10 new seeds), mainly
through false alarms. The same early-fusion model given no optical input scored above a U-Net trained on SAR alone
(+0.072 on new seeds), but the gain is concentrated in one city (+0.028 without Las Vegas; 5 of 10 cities positive).
Over acquisition timings of 1–18 months, the registered test found no linear trend (t-test p 0.105; Wilcoxon p 0.049),
and timing is confounded with how much of the labelled change the image already shows. On real cloudy scenes in 7
cities, a model trained only on full input fell below SAR alone (−0.275); the dropout model with cloud pixels masked
recovered most of the loss (+0.208) and scored slightly above SAR alone (+0.020), but discarding the cloudy image was
slightly better than masking it (by 0.007). Training seeds replicate training, not geography: every result concerns the
same 10 OSCD test cities.

**Keywords:** change detection, SAR–optical fusion, Sentinel-1, Sentinel-2, missing modality, modality dropout,
historical optical imagery, cloud cover, OSCD, pre-registration

---

## 1. Introduction

Optical satellite images carry the spectral detail that makes urban change easy to recognise, but on the date a change
must be assessed they are often unusable because clouds cover part or all of the scene. Synthetic aperture radar (SAR)
is unaffected by cloud, which motivates SAR–optical fusion for change detection [1], [8]. Most fusion models are trained
and evaluated with SAR and optical images at both dates. When the post-change optical image is missing, an older,
cloud-free optical image is often available.

This setting has been studied before. Saha et al. [29] proposed change detection from pre-change optical and SAR images
and post-change SAR only, on the multi-sensor OSCD dataset [8], and compared it with SAR-only baselines: in their Table I
an early-fusion network scored F1 22.88% with SAR at both dates and 27.72% with the pre-change optical image added, and
their proposed three-branch network 32.14%; removing the pre-change optical input at test time lowered their model's F1
by 7.04 points. Song et al. [2] used a historical cloud-free Sentinel-2 image as a semantic prior for a SAR-based detector
of construction-ready bare land and reported lower F1 as the image aged (83.99% at 1 month, 78.31% at 12 months; single
runs). Methods for a missing optical modality reconstruct or hallucinate the missing features [12], [31], learn
sensor-invariant representations [30], or train with the modality removed from some samples [11], [12].

What has not been examined together is how large and how stable the historical image's contribution is when the
comparison is controlled: paired over many training seeds, with pre-specified tests, across fusion designs and training
regimes, across image timings, and on real cloudy post-change scenes. We address five questions:

- **RQ1:** Does historical Sentinel-2 imagery improve Sentinel-1 change detection when the post-change optical image is
  unavailable?
- **RQ2:** How sensitive is the result to the acquisition timing of the historical image (1–18 months before the
  post-change date)?
- **RQ3:** Does a second polarisation (VH with VV) improve SAR change detection or reduce false alarms?
- **RQ4:** How do the models that work without the post-change optical image compare with SAR-only, optical-only,
  input-fusion and dual-stream models?
- **RQ5:** Under which timing and land-cover conditions does the historical image help most?

This is a measurement study on OSCD [3] enriched with Sentinel-1, using the official DS-UNet code [1]; we propose no new
architecture. None of its components is new on its own: the setting and its SAR-only comparison come from [29], modality
dropout from [11], training with samples that lack optical data from [12], a timing experiment from [2], and evaluation
on cloud-affected inputs from [12], [33]. What the study adds is their combination under one shared, pre-registered,
multi-seed protocol, and the following findings for that protocol:

- **(i) The historical image added no detectable F1 in the late-fusion DS-UNet and lowered F1 in the one early-fusion
  U-Net we trained**, through false alarms; the loss replicated on new seeds. This differs in direction from [29]
  (§6.2).
- **(ii) Training with optical data that is randomly removed gave a better SAR-only detector** than training on SAR alone
  (+0.072, replicated on new seeds), but the gain is concentrated in one test city.
- **(iii) Missing and cloudy are different failures.** A dropout-trained model handles a missing post-change image but
  reads unmasked cloud as change; zero-filling detected cloud recovers most of the loss, and on real scenes discarding the
  cloudy image was slightly better than masking it. A model trained only with full input falls far below SAR alone on
  real clouds.
- **(iv) Timing and polarisation.** No detectable linear trend over 1–18 months under the registered test, with a
  confounded 1-month low; for SAR alone, adding VH to VV lowered F1.

Because OSCD is small and seed variance alone can reverse benchmark rankings [17], every comparison of the main study
uses 10 paired training seeds and Holm-corrected tests [4]; every checkpoint and threshold is selected on validation
cities only; and a pre-registration with code hashes was written before each test evaluation (committed to version
control from Step 4B on). Analyses chosen after results were seen are labelled exploratory. The 10 seeds replicate
training on the same 10 test cities; they are not geographic replicates, and §4.7 shows how much of each pooled result
depends on single cities.

## 2. Related Work

**SAR–optical change detection with both modalities.** Siamese fully convolutional networks established bitemporal change
detection on optical images [5], with U-Net [6] variants as common backbones; reviews [7] cover the field. Hafner et al. [1]
fuse Sentinel-1 VV and Sentinel-2 in a dual-stream U-Net with decision-level fusion and report that it beats unimodal
U-Nets and input-level fusion on OSCD enriched with Sentinel-1. Ebel et al. [8] curated Sentinel-1 data for OSCD and fuse
both sensors in a Siamese network; DSNUNet [10] combines Sentinel-1 and Sentinel-2 for forest change; Hafner et al. [32]
add a cross-modal consistency loss for semi-supervised urban change detection. These methods assume both modalities at
both dates. Heterogeneous change detection instead compares an optical image at one date with a SAR image at the other
[9], [25], [26], and multimodal optical–SAR datasets now support all-weather damage assessment [27]. In our setting SAR is
available at both dates and the optical image only before the change.

**Post-change optical image absent.** Saha et al. [29] introduced this setting on the multi-sensor OSCD dataset of [8]:
a Siamese encoder shared by the pre- and post-change SAR images, a separate encoder for the pre-change optical image, and
a common decoder. Their Table I compares early fusion and Siamese networks with SAR at both dates (F1 22.88% and 23.08%)
against early fusion (27.72%) and their architecture (32.14%) with the pre-change optical image added, and a test-time
ablation in which removing the pre-change optical input costs 7.04 F1 points and removing the pre-change SAR input 4.81.
They use a 12-city training, 2-city validation and 10-city test split (the same 10 test cities as ours), and average
their proposed method over three random seeds (author's reading of the full text). Their released code reads two SAR
bands per date from the Sentinel-1 data of [8] (ascending orbit, exported from Google Earth Engine) and clips them to
[−25, 0] dB, and trains for 50 epochs with Adam at learning rate 1e-4 by default. We could not verify which polarisation
the second band holds, whether each SAR image is a single acquisition or a temporal composite, or whether the Table I
baselines are averaged over seeds; these are stated where they matter (§6.2). Song et al. [2] monitor construction-ready
bare land in Guangzhou with Sentinel-1 VH change screening and a historical cloud-free Sentinel-2 image as a semantic
prior; varying its age (1, 3, 6 and 12 months before the target date) lowered F1 from 83.99% to 78.31%, the gap table
reports single runs, and the evaluation classifies candidate patches rather than pixels. In their framework the image
informs land cover at the target date; in a bitemporal model it fills the pre-change slot, where an image that already
shows the change hides it, so the expected sign of a timing effect differs.

**Missing modalities.** Modality dropout was introduced for gesture recognition [11]. Kampffmeyer et al. [31] train a
hallucination network that replaces a modality missing at test time in urban land-cover classification, and Zheng et al.
[30] learn a sensor-invariant representation for SAR-only mapping when optical data are missing. Hafner and Ban [12]
train an urban mapping model on mini-batches that mix multimodal samples and samples without optical data, and
reconstruct optical features from SAR; on samples with missing optical data their model beats zero-fill (F1 0.327
against 0.233) but falls slightly short of a SAR-only model (0.348).

**Clouds.** SAR-guided cloud removal reconstructs cloud-free Sentinel-2 images [13], and SEN12MS-CR-TS provides real cloudy
Sentinel-1/2 time series [28]. Zitzlsberger and Podhoranyi [33] monitor urban change from Sentinel-1/2 time series in which
cloud-masked observations are skipped, and report greater reliance on optical than SAR input. Real cloud masks are
available in the Sentinel-2 Cloud Mask Catalogue [14]; operational detectors such as s2cloudless [15] are imperfect, and an
intercomparison of cloud-masking algorithms documents large differences between methods [16]. Masking detected cloud and
treating it as missing data is standard practice; we measure how its errors interact with a dropout-trained change
detector.

**Evaluation practice.** Variance from data sampling, initialisation and hyperparameters can change benchmark
conclusions [17]; Appendix A shows a ranking reversal on OSCD under a weaker protocol.

Table 1 summarises the studies closest to this one.

**Table 1.** Studies closest to this one. "Historical optical" means an optical image acquired before the change used
when no post-change optical image is available. Entries for [29] combine the author's reading of the full text and its
released code.

| study | data | modalities | fusion | missing-modality handling | historical optical | cloud evaluation | limitation relevant here |
|---|---|---|---|---|---|---|---|
| Hafner et al. [1] | OSCD + S1 | S1 VV + S2, both dates | late (decision-level) dual stream | none | no | no | assumes synchronous optical |
| Ebel et al. [8] | OSCD + S1 | S1 + S2, both dates | Siamese fusion | none | no | no | assumes both modalities at both dates |
| Saha et al. [29] | multi-sensor OSCD [8] | pre: S2 + S1; post: S1 | SAR Siamese + optical encoder, common decoder; early-fusion baseline | post-change optical absent by design; test-time ablation | yes; compared with SAR-only baselines | no | one training regime; proposed method over 3 seeds |
| Song et al. [2] | Guangzhou (own) | S1 VH + S2 | two-stage screening and refinement | post-change optical absent by design | yes, as semantic prior; 1–12 months | motivation only | one region; patch-level evaluation; single-run gap results |
| Hafner and Ban [12] | SpaceNet 7 + S1/S2 | S1 + S2 time series | dual stream + SAR→optical reconstruction | mixed training batches; reconstruction | no | missing = cloud-affected timestamps | urban mapping, not change detection |
| Zheng et al. [30] | VHR optical/SAR mapping set | optical + SAR | meta-sensory prototype network | SAR only at test | historical multisensor data in training | no | segmentation; very high resolution |
| Zitzlsberger and Podhoranyi [33] | Mariupol (own) | S1 + S2 time series | deep-temporal network | masked observations skipped; loss-of-observation ablation | time-series context | operational cloud masks | one site |
| this study | OSCD + S1 | S1 VV (VH for RQ3) + S2 | late (DS-UNet) and early (U-Net) | training-time modality dropout; zero-fill; cloud masks | yes, pre-change slot; 1–18 months | synthetic, composite and real (one scene per city) | 10 fixed test cities; one benchmark |

## 3. Data and Methods

Figure 1 summarises the workflow and Figure 2 the networks and input conditions.

![Figure 1. Study workflow (schematic; no data). Each arm is trained per seed, selected on the 4 validation cities, pre-registered with hashes of code, inputs and checkpoints, and then evaluated once on the 10 fixed test cities; contrasts are paired over seeds.](figures_v2/fig01_workflow.png)

![Figure 2. Networks and input conditions (schematic drawn from the training code and the official DS-UNet network file; no data). (a) DS-UNet: separate SAR and optical U-Nets whose final 64-channel features are concatenated and fused by a 1×1 convolution, the only point where the streams meet. (b) Single-stream U-Net baselines trained on full input. (c) Early-fusion model E and the input conditions; hatched slots are set to zero. In `no_o2` the date-1 optical slot holds the historical image.](figures_v2/fig02_architectures.png)

### 3.1 Data and split

**OSCD.** OSCD [3] provides 24 bitemporal Sentinel-2 L1C image pairs with pixel-level urban change labels. We use the 13
rectified bands as supplied by OSCD (`imgs_1_rect`, `imgs_2_rect`), which the dataset's documentation describes as the
original bands resampled to 10 m and cropped to the city's bounding box [3]; the interpolation used to upsample the
20 m and 60 m bands is not documented in the dataset README, the dataset paper or its cropping toolbox, and remains
unknown. Within each pair OSCD registered the newer image onto the older one with GeFolki [3]. The `_rect` bands and the
labels carry no georeferencing and share the pixel grid of the date-1 10 m bands. Bands are divided by 10000 and
clipped to [0, 1]; the label is the `cm` raster minus 1 (0 no change,
1 change). Ten of the 14 official training cities are used for training and the other 4 (aguasclaras, beihai, bercy,
paris), drawn once with a fixed random generator, for validation; the 10 official test cities (brasilia, montpellier,
norcia, rio, saclay_w, valencia, dubai, lasvegas, milano, chongqing) are used for testing. Test cities never enter training
or validation, which is asserted in code. Change prevalence on the test cities is 0.052; labelling every pixel as change
gives F1 0.098.

**Sentinel-1.** VV rasters released with DS-UNet [1]. As documented in [1] and its download notebook, Sentinel-1 GRD
scenes were processed in Google Earth Engine (thermal noise removal, radiometric calibration, terrain correction,
conversion to dB), values below −25 dB were masked, the scenes were separated by relative orbit, and each orbit's scenes
were averaged pixel-wise over the three months before the date-1 Sentinel-2 acquisition and over the three months after
the date-2 acquisition (Mumbai, a training city, has a longer date-1 window); VV was rescaled from [−25, 5] dB to [0, 1]
and exported at 10 m in geographic coordinates over the OSCD city bounding box. We use the rasters without further
scaling and resize them to the label grid by cubic interpolation, as in [1]; training samples an orbit at random and
evaluation uses the first listed orbit. Neither [1] nor our code performs an image-based co-registration of Sentinel-1
to Sentinel-2: the SAR raster and the date-1 optical crop cover the same bounding box, and in the 10 test cities every
SAR raster edge lies within 0.94 Sentinel-2 pixels of it (a footprint check on raster metadata). SAR–optical alignment
in image content was not evaluated, by [1] or by us. For RQ3 we exported VV and VH from Google
Earth Engine [18] with DS-UNet's recipe: `COPERNICUS/S1_GRD`, IW mode, one relative orbit, the mean over the window,
linearly rescaled from [−25, 5] dB (VV) or [−30, 0] dB (VH, fixed before download) to [0, 1] and clipped; windows without
scenes become 0. The exported VV equals the released raster pixel for pixel in all 102 files. Las Vegas has no VH at
date 1 on its evaluated orbit (an all-zero band), so the RQ3 analysis uses 9 test cities. Date-1 windows hold fewer scenes
than date-2 windows in 22 of 23 test-city orbits (median 6 against 14 VV scenes), so date-1 means are noisier.

**Historical Sentinel-2 images (RQ2).** For each city, one Sentinel-2 L1C scene at each target date 2 − g, g ∈ {1, 3, 6,
12, 18} months, and one at OSCD's own date-1 day as a control of the processing chain. Rule, fixed before any model saw a
scene: within ±45 days of the target, full coverage required, the day nearest the target with city-box cloud fraction
≤ 0.02, else the clearest day. The primary set is the 9 test cities whose scene has cloud fraction ≤ 0.10 at every gap
(Chongqing excluded). Alignment against OSCD's own date-1 bands gives median band correlation 0.940 and median shift 0.27
px. At 1–12 months no primary city's image precedes OSCD date 1; at 18 months, 4 of the 9 do.

**Real cloudy scenes (Step 11).** For each test city, the full-coverage Sentinel-2 L1C day within ±30 days of OSCD date 2
whose city-box cloud fraction (s2cloudless probability ≥ 40) lies in [0.05, 0.70], nearest to date 2; Montpellier, Rio and
Las Vegas had none and contribute their nearest scene with cloud ≥ 0.90 ("overcast"). Each city also has OSCD's own date-2
day as a control. The rule was applied to stored cloud fractions before download; the scenes and their cloud fractions are
listed in the Step 11 pre-registration. Scenes are resampled bilinearly to the label grid and scaled as OSCD.

**Land cover.** ESA WorldCover 2020 v100 [19] on the label grid (nearest neighbour). It post-dates OSCD date 2 and is used
as a stratification, not as a label; OSCD has no labels separating urban from agricultural or seasonal change.

### 3.2 Models and training recipe

The network and loss are the official DS-UNet files (`networks.DualStreamUNet`, topology [64, 128, 256, 512];
`jaccard_like_loss`) loaded at run time from commit 8bfa4be [1]. Each date contributes its SAR channel(s) followed by the
13 optical bands. In DS-UNet the SAR stream (VV at both dates, 2 channels) and the optical stream (26 channels) are
separate U-Nets whose 64-channel outputs are concatenated and mapped to the change logit by a 1×1 convolution, so the
optical stream never sees SAR (Figure 2a). The U-Net baselines are DS-UNet's single-stream U-Net on the SAR (2), optical
(26) or stacked (28) channels, with the same recipe; E uses the 28-channel stacked input (Figure 2c).

Training: Adam, learning rate 5e-5, weight decay 5e-4, batch 64, 2500 steps, validation every 25 steps. Each batch draws
64 training cities with replacement, a random orbit for each, and one 32×32 crop per sample by DS-UNet's importance rule
(20 random windows, one chosen with probability proportional to its number of changed pixels plus 5); the crop is flipped
horizontally and vertically, each with probability 0.5, and rotated by 90°, 180° or 270°. The step budget was chosen on
validation score only (390 steps: 0.354; 1000: 0.382; 2500: 0.405). Python, NumPy and PyTorch are seeded with the run's
seed and cuDNN runs deterministically (PyTorch 2.5.1+cu124, one RTX 4060 laptop GPU); one generator per run draws cities,
crops, augmentations and input conditions, so arms with the same seed share these draws and contrasts between arms are
paired by seed. As a reference, the official 390-step protocol on all 14 training cities gives test F1 0.557 ± 0.035,
below the published 0.600 [1]; our conclusions concern differences between arms under one shared protocol.

### 3.3 Arms

A missing optical image is represented by zeros in its channels; zero is the fill value used here, not an intrinsic
missing-data token. Conditions: `full`; `no_o2` (post-change optical zero, pre-change optical image present: the
historical-image setting); `no_opt` (both optical dates zero: SAR only); `o2_copy` (the pre-change optical image copied
into the post-change slot). During training each sample's condition is drawn with the arm's probabilities.

Table 2 lists the arms.

**Table 2.** Training arms. Probabilities are per training sample.

| arm | network | training conditions | step |
|---|---|---|---|
| A | DS-UNet | full input | Phase 2 |
| B | DS-UNet | full 0.50; `no_o2` 0.25; `no_opt` 0.25 (modality dropout) | Phase 2 |
| C | DS-UNet | full 0.50; post-change optical zero under a cloud-shaped mask 0.25; `no_o2` 0.125; `no_opt` 0.125 | Step 4B |
| async | DS-UNet | `no_o2` only | Step 6 |
| SAR-only, optical-only, input fusion | U-Net | full input | Step 6 |
| E | U-Net, VV and optical stacked at the input | as B (modality dropout) | Step 9 (seeds 1–10), Step 10 (seeds 11–20) |
| VH, VV+VH | U-Net and DS-UNet | full input | Step 7 |

Each arm is trained with seeds 1–10, and E and the SAR-only U-Net again with seeds 11–20 (Step 10). In `no_o2` the
historical image is OSCD's own date-1 image or, for RQ2, one of the historical scenes in the pre-change slot. E is the
early-fusion counterpart of B: in E the historical image enters the same encoder as SAR, so it can condition the SAR
features. The async arm is the closest analogue of the setting of [29]: trained and tested with the pre-change optical
image and no post-change optical image.

**Selection (validation cities only).** Validation counts are pooled over the 4 cities at 101 thresholds {0, 0.01, …, 1},
using the first listed orbit. Checkpoint: the step with the highest mean best-threshold validation F1 over the conditions
the arm trains on. Threshold: the validation-F1 argmax for each input condition, frozen and applied unchanged on test.
Where a model is evaluated on inputs it has no threshold for (clouds, §5), its frozen full-input threshold is used.

### 3.4 Statistics and integrity

**Metrics.** F1, precision, recall and false-alarm rate (FP / (FP + TN)) are computed from pixel counts pooled over the
analysed test cities, not averaged over cities; every table and figure uses this pooling unless it says otherwise.

**Unit of replication.** A training seed is one replicate of training (initialisation, sampled crops, augmentations,
input conditions) on fixed data; all seeds are evaluated on the same test cities. Confidence intervals and p-values
therefore describe variation across training runs on these 10 cities, not across places. City-level resampling and
leave-one-city-out analyses (§4.7) are exploratory.

**Tests.** Each step has pre-specified families of paired comparisons over seeds (two-sided paired t-test, Holm
correction [4]), reported with the mean difference, its 95% CI, d_z and the number of seeds with a positive difference.
The t-test is the registered decision rule; a Wilcoxon signed-rank test is computed alongside and reported where it
disagrees at α = 0.05. Every contrast is written "first − second", so a positive Δ favours the first term. Labels such as
"higher" require Holm p < 0.05; otherwise "no detectable difference", which is not evidence of equivalence: no equivalence
margin was pre-registered. With 10 pairs the minimum detectable effect at 80% power is about 1.00–1.22 SD of the paired
differences, depending on the family. A detectable difference is not necessarily a practically important one; we give
effect sizes in F1 and leave the practical threshold to the application.

**Pre-specification.** Each step's pre-registration (hypotheses, primary family, decision rule, SHA-256 hashes of code,
inputs and checkpoints) was written before its first test evaluation; for Steps 4B–11 it was committed to version
control first, and from Step 9 on a gate script refused to run unless the committed file and every hash were unchanged.
These are repository files, not public registrations. "Step 4B" is the spatial-dropout arm C of the robustness study;
"Stage 4B" is a later, separate evaluation of B with and without the historical image, registered after partial
unblinding (§4.2). Evaluations enforce anchors, for example that re-evaluated test counts equal stored ones exactly.
Audit scripts written with the same AI assistant recompute the statistics of Phases 2–8 (Appendix D). Analyses added after
results were seen are labelled exploratory; one of them (§4.3) was then tested on new seeds as a registered hypothesis.

Figure 3 summarises the registered primary contrasts of the study.

![Figure 3. Registered primary paired contrasts: mean difference in test F1 (first − second) with 95% CI over 10 training seeds on fixed test cities; filled marker: Holm p < 0.05 (Stage 4B is a single test, p < 0.05). Group labels give the city set (10, 9 or 7 cities) and the threshold rule; F1 is pooled over the cities of each group.](figures_v2/fig03_forest.png)

## 4. Results

### 4.1 RQ4: comparison with SAR-only, optical-only, conventional fusion and dual-stream approaches

Table 3 gives the full-input test metrics and Table 4 the registered contrasts.

**Table 3.** Full input: test metrics at the validation-selected threshold, 10 test cities, mean ± SD over 10 seeds
(Step 6).

| arm | F1 | precision | recall | IoU | false-alarm rate |
|---|---|---|---|---|---|
| DS-UNet A | 0.556 ± 0.029 | 0.568 ± 0.071 | 0.562 ± 0.090 | 0.385 ± 0.028 | 0.025 ± 0.013 |
| DS-UNet B (modality dropout) | 0.553 ± 0.030 | 0.466 ± 0.041 | 0.682 ± 0.032 | 0.382 ± 0.028 | 0.043 ± 0.007 |
| SAR-only U-Net | 0.418 ± 0.043 | 0.312 ± 0.051 | 0.648 ± 0.027 | 0.265 ± 0.035 | 0.081 ± 0.020 |
| optical-only U-Net | 0.498 ± 0.056 | 0.588 ± 0.080 | 0.451 ± 0.101 | 0.333 ± 0.048 | 0.019 ± 0.012 |
| input-fusion U-Net | 0.495 ± 0.026 | 0.556 ± 0.063 | 0.452 ± 0.047 | 0.329 ± 0.023 | 0.020 ± 0.006 |

**Table 4.** Registered primary family of Step 6 for RQ4: A − each baseline, paired test F1 differences over 10 seeds
(Holm over 3).

| contrast | mean Δ | 95% CI | d_z | Δ > 0 | Holm p | label |
|---|---|---|---|---|---|---|
| A − SAR-only | +0.138 | [+0.094, +0.181] | +2.26 | 10/10 | 1.6e-04 | DS-UNet better |
| A − optical-only | +0.058 | [+0.004, +0.112] | +0.77 | 8/10 | 0.0370 | DS-UNet better |
| A − input fusion | +0.061 | [+0.033, +0.089] | +1.54 | 9/10 | 0.0017 | DS-UNet better |

Across training seeds, the dual-stream design scores above all three U-Net baselines, in line with Hafner et al. [1].
Input fusion is not distinguishable from optical alone (Δ −0.003, 95% CI [−0.045, +0.039]). The recipe and step budget
were chosen on arm A, the baselines were not tuned separately, and the selected thresholds of the optical-only and
input-fusion U-Nets reach 0.99 or above in several seeds, which suggests poorly scaled outputs. In the exploratory city
resampling (§4.7, Table 14), A's lead over SAR-only is concentrated in Las Vegas and not established beyond these
cities.

**Without the post-change image.** B and async without the post-change image are not detectably different from the
SAR-only U-Net (+0.013, −0.016; registered, §4.2) and, in an exploratory comparison, score below the optical-only,
input-fusion and DS-UNet models that have that image.

### 4.2 RQ1: the historical image, with late and with early fusion

**Against SAR alone (Step 6).** Table 5 compares both models without the post-change optical image with the
SAR-only U-Net.

**Table 5.** Registered primary family of Step 6 for RQ1: setting `no_o2` against the SAR-only U-Net, test F1 at each
model's validation-selected threshold, 10 test cities (Holm over 2).

| contrast | F1 | mean Δ | 95% CI | d_z | Δ > 0 | Holm p | label |
|---|---|---|---|---|---|---|---|
| B@no_o2 − SAR-only | 0.431 ± 0.034 | +0.013 | [−0.015, +0.040] | +0.33 | 6/10 | 0.6576 | no detectable difference |
| async@no_o2 − SAR-only | 0.402 ± 0.059 | −0.016 | [−0.059, +0.026] | −0.27 | 5/10 | 0.6576 | no detectable difference |

The F1 column is the first term; the SAR-only U-Net scores 0.418 ± 0.043. Neither model is detectably better than SAR
alone. The companion study (Appendix A) had reported a margin for its missing-optical model under a weaker protocol; under
the shared protocol it does not survive.

**Within the late-fusion model (Stage 4B).** At B's frozen full-input threshold, the threshold used for every corrupted
input in §5, B without the post-change image scores above the SAR-only U-Net (+0.060). Stage 4B decomposes this margin
by evaluating B with the historical image present and absent (both optical slots zero; Table 6):

**Table 6.** B at its frozen full-input threshold, 10 test cities (Stage 4B; primary: present − absent, single test;
other rows descriptive).

| contrast | mean Δ | 95% CI | d_z | Δ > 0 | p | label |
|---|---|---|---|---|---|---|
| B: image present − absent | +0.007 | [−0.004, +0.018] | +0.44 | 7/10 | 0.198 | no detectable difference |
| B, image present − SAR-only | +0.060 | [+0.026, +0.094] | +1.27 | 8/10 | 0.00303 | (descriptive) |
| B, image absent − SAR-only | +0.053 | [+0.022, +0.085] | +1.20 | 9/10 | 0.00415 | (descriptive) |

The historical image contributes no detectable F1 (it raises recall by 0.018, Holm p 0.030, with no detectable change in
precision). B's margin over the SAR-only U-Net comes from B itself: with no optical input at all it already scores 0.053
above it. At B's own `no_o2` threshold the within-B contrast is also not detectable (+0.002). Stage 4B was registered
after the within-B contrast had been seen at two other thresholds, which its pre-registration states.

**With early fusion (Step 9).** DS-UNet's late fusion lets the historical image add only a SAR-independent term to each
pixel's score. To test whether that explains the null, we trained E, a U-Net with VV and the 13 optical bands stacked at
the input, with B's dropout mixture (validation-selected thresholds per condition, as Step 6):

**Table 7.** Step 9, E against B and the baselines, 10 test cities (primary family Holm over 2; secondary families Holm
over 2 each).

| contrast | mean Δ | 95% CI | d_z | Δ > 0 | Holm p | label |
|---|---|---|---|---|---|---|
| E: image present − absent | −0.091 | [−0.124, −0.058] | −1.96 | 0/10 | 0.0003 | lower with the image |
| (E present − absent) − (B present − absent) | −0.081 | [−0.122, −0.040] | −1.41 | 2/10 | 0.0016 | early fusion lower |
| E@no_o2 − SAR-only | −0.019 | [−0.072, +0.034] | −0.26 | 5/10 | 0.4383 | no detectable difference |
| E@no_o2 − B@no_o2 | −0.032 | [−0.068, +0.005] | −0.63 | 2/10 | 0.1590 | no detectable difference |
| E@full − input fusion | +0.040 | [+0.006, +0.074] | +0.85 | 9/10 | 0.0496 | E higher |
| E@full − A | −0.021 | [−0.055, +0.013] | −0.44 | 3/10 | 0.2012 | no detectable difference |

In E the historical image lowers F1 by 0.091 (0.399 ± 0.039 with it, 0.490 ± 0.010 without it), in every seed, and the
difference from its effect in B is detectable. It does so through false alarms (exploratory): precision falls by 0.133
(95% CI [−0.195, −0.071]) and the false-alarm rate rises by 0.041, with no detectable change in recall. This pattern is consistent with the network reading an optical
image at date 1 next to zeros at date 2 as change, although dropout training showed it this input in a quarter of its
samples; we did not test the mechanism directly. On 10 new seeds (Step 10, Table 8) the loss
replicates (−0.116). On full input E is not detectably different from A and scores above the full-input-only
input-fusion U-Net (Holm p 0.0496).

**Answer to RQ1.** On OSCD, historical Sentinel-2 imagery did not detectably improve Sentinel-1 VV change detection
without the post-change optical image. In the late-fusion DS-UNet it made no detectable difference (95% CI [−0.004, +0.018] at the
full-input threshold); in an early-fusion U-Net, where it can condition the SAR features, it lowered F1 by raising false
alarms. Late fusion alone therefore does not explain the DS-UNet null in this comparison, although only one early-fusion
design was tested. The setting-matched async arm was not detectably different from SAR alone either, which differs in
direction from the gain reported by Saha et al. [29] under a different protocol (§6.2). Effects below about 1.22 SD of
the paired differences can be missed, and the Sentinel-1 inputs are 3-month means.

### 4.3 A dropout-trained network as a SAR-only detector

Table 7's model E, evaluated with both optical slots zero, scored above the SAR-only U-Net on seeds 1–10 (+0.072, 95% CI
[+0.045, +0.099], 10/10; exploratory, chosen after the results were seen). We registered this as the primary hypothesis
of Step 10 and retrained E and the SAR-only U-Net on seeds 11–20, which no arm had used:

**Table 8.** Step 10, seeds 11–20, 10 test cities (primary: single test; secondary: Holm over 2).

| contrast | mean Δ | 95% CI | d_z | Δ > 0 | Holm p | label |
|---|---|---|---|---|---|---|
| E@no_opt − SAR-only | +0.072 | [+0.038, +0.107] | +1.48 | 10/10 | 0.0011 | E without optical higher |
| E: image present − absent | −0.116 | [−0.145, −0.086] | −2.76 | 0/10 | 2.2e-05 | lower with the image |
| E@no_opt − SAR-only, threshold 0.5 | +0.109 | [+0.070, +0.148] | +2.00 | 10/10 | 1.4e-04 | E without optical higher |

The gain replicates over training seeds (E 0.495 ± 0.012, SAR-only 0.423 ± 0.045) and holds at a fixed threshold, so it
is not only a threshold effect. B, trained with the same mixture but late fusion, shows a smaller gain without optical
input (+0.022 at its `no_opt` threshold, 95% CI [−0.003, +0.048], exploratory). The new seeds replicate training, not
cities: in the exploratory city analysis the gain is concentrated in Las Vegas (+0.157 in that city); without it the
pooled gain is +0.028, it stays positive with any one city left out (+0.028 to +0.082), and only 5 of 10 cities share its
sign, two of them by +0.005 or less (Table 14). The result is a replication over training seeds on the fixed OSCD test
cities, not evidence that the gain generalises to other places. The mechanism, for example regularisation by the optical inputs during training, was not tested.

### 4.4 RQ2: sensitivity to the acquisition timing of the historical image

Under OSCD's label definition (change between two fixed dates), varying the image's timing varies two things at once: its
age and how much of the labelled change it may already show, because every image at 1–12 months post-dates OSCD date 1.
An image acquired after part of the change has happened shows that change in the pre-change slot, where it hides the
change from a bitemporal model; an older image does not. The experiment therefore measures sensitivity to timing, not
decay with age. Table 9 and Figure 4 give F1 by timing.

**Table 9.** Test F1 (mean ± SD over 10 seeds) by the timing of the image in the pre-change slot, setting `no_o2`, frozen
`no_o2` threshold, 9 primary cities (Step 8; descriptive).

| arm | OSCD date 1 | Earth Engine date 1 | 1 month | 3 months | 6 months | 12 months | 18 months |
|---|---|---|---|---|---|---|---|
| B dropout | 0.420 ± 0.039 | 0.420 ± 0.039 | 0.389 ± 0.044 | 0.422 ± 0.038 | 0.409 ± 0.035 | 0.417 ± 0.037 | 0.412 ± 0.033 |
| async | 0.405 ± 0.067 | 0.407 ± 0.066 | 0.349 ± 0.059 | 0.416 ± 0.061 | 0.370 ± 0.072 | 0.422 ± 0.057 | 0.383 ± 0.066 |

![Figure 4. Test F1 (pooled over the 9 primary cities) by acquisition timing of the historical image, setting `no_o2`, frozen `no_o2` thresholds; mean and 95% CI over 10 training seeds. Dashed: SAR-only U-Net on the same cities. Timing is confounded with how much labelled change the image already shows; each timing has one scene per city.](figures_v2/fig04_timing.png)

**Primary test:** the per-seed least-squares slope of B's F1 against the gap has mean +0.0083 F1 per year (95% CI
[−0.0021, +0.0187]), t-test p 0.105: **no detectable linear trend** under the registered test. The exact Wilcoxon test on
the same slopes gives p 0.049, with 8 of 10 slopes positive, so the two tests disagree at α = 0.05: the sign (older
images not worse) is suggestive, not established. Two related slopes are also not detectable: B on all 10 cities
(sensitivity; +0.0042 per year, 95% CI [−0.0002, +0.0086], p 0.057) and async (secondary; +0.0161 per year, 95% CI
[−0.0001, +0.0322], p 0.051). Without its largest slope (seed 7) B's mean slope is +0.0042 (exploratory). Against SAR alone (0.406 ± 0.048 on these cities),
B shows no detectable difference at any timing. **Exploratory:** the timings do differ (Friedman p 0.00062 for B), with the
1-month image lowest for both models; B's 1-month deficit relative to OSCD's date-1 image comes mostly from Saclay and is
not established when cities are resampled (95% CI [−0.116, +0.003]). Each timing has one scene per city, so scene
properties are not replicated by seeds. In Stage 4B's descriptive comparison, no timing scored detectably above B without any
optical input (largest difference +0.004), and the 1-month image scored below it (−0.029, uncorrected p 0.041).

**Answer to RQ2.** No detectable linear trend of F1 with acquisition timing over 1–18 months under the registered test
(with a Wilcoxon result at p 0.049 that points, if anything, to older images doing no worse), and no timing at which the
image beats SAR alone. The low 1-month score is consistent with the image already showing part of the change, but this
was not measured, so the design does not isolate image age. Separating age from the labelled change requires labels dated
at the historical image. The decline reported by Song et al. [2] concerns a different role of the image and is not tested
here.

### 4.5 RQ3: VV and VH polarisation

Table 10 gives the test metrics and Table 11 the registered contrasts.

**Table 10.** Full input: test metrics at the frozen threshold, 9 test cities with VH at both dates, mean ± SD over 10
seeds (Step 7).

| arm | F1 | precision | recall | IoU | false-alarm rate |
|---|---|---|---|---|---|
| SAR-only U-Net, VV | 0.452 ± 0.011 | 0.367 ± 0.017 | 0.590 ± 0.029 | 0.292 ± 0.009 | 0.049 ± 0.005 |
| SAR-only U-Net, VH | 0.383 ± 0.029 | 0.346 ± 0.048 | 0.442 ± 0.069 | 0.237 ± 0.022 | 0.042 ± 0.016 |
| SAR-only U-Net, VV+VH | 0.408 ± 0.031 | 0.405 ± 0.040 | 0.424 ± 0.081 | 0.257 ± 0.024 | 0.031 ± 0.010 |
| DS-UNet, VV | 0.501 ± 0.024 | 0.523 ± 0.067 | 0.500 ± 0.089 | 0.335 ± 0.021 | 0.024 ± 0.013 |
| DS-UNet, VH | 0.484 ± 0.043 | 0.511 ± 0.030 | 0.462 ± 0.061 | 0.320 ± 0.036 | 0.021 ± 0.002 |
| DS-UNet, VV+VH | 0.511 ± 0.019 | 0.571 ± 0.029 | 0.465 ± 0.036 | 0.344 ± 0.017 | 0.017 ± 0.003 |

**Table 11.** Polarisation contrasts, test F1, 9 cities (Step 7; primary family: first two rows, Holm over 2;
secondary: last two rows, Holm over 2).

| contrast | mean Δ | 95% CI | d_z | Δ > 0 | Holm p | label |
|---|---|---|---|---|---|---|
| SAR-only: VV+VH − VV | −0.044 | [−0.065, −0.023] | −1.49 | 1/10 | 0.0011 | VV+VH lower |
| SAR-only: VH − VV | −0.070 | [−0.091, −0.048] | −2.32 | 0/10 | 8.8e-05 | VH lower |
| DS-UNet: VV+VH − VV | +0.010 | [−0.016, +0.036] | +0.27 | 7/10 | 0.4125 | no detectable difference |
| DS-UNet: VH − VV | −0.017 | [−0.039, +0.004] | −0.59 | 3/10 | 0.1925 | no detectable difference |

The first two rows are the primary family, the last two a secondary family. By land-cover class, the SAR-only VV+VH model
has lower false-alarm rates than VV in four classes but lower recall in every class (0.424 against 0.590 overall): it
flags less change everywhere, a shift of operating point, not evidence that VH separates urban development from
pseudo-change. Including Las Vegas, whose VH band at date 1 is empty, the VV+VH SAR-only model drops to F1 0.224 ± 0.032.

**Answer to RQ3.** For the SAR-only U-Net, adding VH to VV lowered F1 and VH alone lowered it more; with optical input,
polarisation made no detectable difference. These are results for 3-month mean backscatter whose date-1 means average fewer
scenes, and OSCD has no labels for the cause of a false alarm.

### 4.6 RQ5: conditions in which historical optical information helps

Table 12 compares false-alarm rates by land-cover class.

**Table 12.** False-alarm rate by WorldCover class, B with OSCD's date-1 image minus the SAR-only U-Net, 9 primary
cities (Step 8; Holm over the 7 classes with ≥ 1% of the pixels).

| class | pixel share | mean Δ | 95% CI | Δ > 0 | Holm p | label |
|---|---|---|---|---|---|---|
| tree cover | 15.6% | −0.001 | [−0.003, −0.000] | 3/10 | 0.0973 | no detectable difference |
| shrubland | 1.3% | −0.006 | [−0.010, −0.003] | 1/10 | 0.0127 | B lower |
| grassland | 7.7% | +0.002 | [−0.002, +0.006] | 5/10 | 0.9703 | no detectable difference |
| cropland | 8.0% | −0.015 | [−0.034, +0.004] | 2/10 | 0.5354 | no detectable difference |
| built-up | 40.6% | −0.004 | [−0.015, +0.008] | 4/10 | 0.9804 | no detectable difference |
| bare | 24.6% | −0.025 | [−0.077, +0.027] | 4/10 | 0.9703 | no detectable difference |
| water | 1.7% | +0.001 | [−0.008, +0.011] | 4/10 | 0.9804 | no detectable difference |

The one detectable difference, in shrubland (1.3% of pixels), is between two networks. Within B (Stage 4B), the historical
image does not detectably change shrubland false alarms (Holm p 0.28), so that difference cannot be attributed to the
image. Within B the image lowers the tree-cover false-alarm rate by 0.002 (Holm p 0.005) and raises recall in tree cover
and built-up areas (+0.020 each); these effects are small.

**Answer to RQ5.** Within the tested conditions (1–18 month timings, seven land-cover classes, OSCD urban change), no
timing gave a detectable F1 gain over SAR alone and no land-cover class showed a detectable false-alarm reduction that
can be attributed to the image; the detectable within-B effects (tree-cover false alarms −0.002, recall +0.020 in tree
cover and built-up areas) are small. No practical threshold was pre-specified, so "small" is descriptive.

Figure 5 illustrates the error patterns of three single cities.

![Figure 5. Error maps for the best (Las Vegas), median (Rio) and worst (Valencia) OSCD test city by DS-UNet A's mean per-city F1 (selection rule committed before drawing); seed 1, frozen thresholds; panel F1 values are that city and seed only. Blue: true positive; orange: false positive; green: false negative. Illustrative of single cities, not of pooled results. Contains modified Copernicus Sentinel data 2017–2018.](figures_v2/fig05_error_maps.png)

### 4.7 Test cities and training seeds (exploratory)

Table 13 compares seed and city variability for five contrasts; Table 14 gives the per-city differences.

**Table 13.** Seed and city variability of five contrasts (exploratory). Per-city counts at the same frozen thresholds;
the first interval resamples seeds, the second resamples test cities and then seeds (two-stage percentile bootstrap,
10000 replicates). Dubai, Las Vegas and Chongqing hold 77% of the change pixels.

| contrast | cities | mean Δ | 95% CI, seeds | SE, seeds | SE, cities | cities with same sign | 95% CI, cities and seeds | Δ, one city left out |
|---|---|---|---|---|---|---|---|---|
| A (full input) − SAR-only | 10 | +0.138 | [+0.094, +0.181] | 0.019 | 0.087 | 6/10 | [−0.022, +0.246] | +0.049 to +0.156 |
| B@no_o2, own threshold − SAR-only | 10 | +0.013 | [−0.015, +0.040] | 0.012 | 0.004 | 6/10 | [−0.019, +0.043] | +0.009 to +0.014 |
| async@no_o2 − SAR-only | 10 | −0.016 | [−0.059, +0.026] | 0.019 | 0.039 | 7/10 | [−0.095, +0.047] | −0.055 to −0.001 |
| B@no_o2, clean threshold − SAR-only | 10 | +0.060 | [+0.026, +0.094] | 0.015 | 0.037 | 6/10 | [+0.007, +0.117] | +0.024 to +0.073 |
| B: 1-month − OSCD date-1 image | 9 | −0.031 | [−0.056, −0.005] | 0.011 | 0.026 | 5/9 | [−0.116, +0.003] | −0.043 to −0.008 |

For four of the five contrasts, resampling cities adds more uncertainty than resampling seeds. DS-UNet's lead over the
SAR-only U-Net is concentrated in Las Vegas (without it +0.049) and its city-and-seed interval includes zero; this shows
that ten cities do not establish the lead beyond them, not that it is absent. Las Vegas also carries most of the Step 10
gain (§4.3): with 10 cities of very unequal size, a single large desert city can drive pooled differences.

**Table 14.** Per-city test F1 difference (mean over seeds of each city's own F1 difference; exploratory) and each city's
share of the test change pixels. The pooled rows are pixel-pooled over cities, as in the rest of the paper. Seeds 1–10
except the last column (Step 10, seeds 11–20); validation-selected thresholds.

| city | change-pixel share | A − SAR-only | B@no_o2 − SAR-only | async@no_o2 − SAR-only | E@no_opt − SAR-only (Step 10) |
|---|---|---|---|---|---|
| brasilia | 3% | +0.102 | +0.013 | −0.076 | −0.011 |
| montpellier | 8% | +0.064 | +0.020 | −0.023 | −0.019 |
| norcia | 1% | −0.002 | −0.016 | −0.107 | −0.008 |
| rio | 5% | −0.033 | −0.022 | −0.040 | −0.006 |
| saclay_w | 3% | −0.053 | +0.077 | +0.047 | +0.044 |
| valencia | 1% | +0.037 | +0.008 | +0.000 | −0.003 |
| dubai | 31% | +0.113 | +0.014 | −0.025 | +0.053 |
| lasvegas | 28% | +0.304 | +0.010 | +0.043 | +0.157 |
| milano | 2% | −0.162 | −0.017 | −0.050 | +0.005 |
| chongqing | 18% | +0.010 | −0.002 | −0.105 | +0.002 |
| pooled, 10 cities | | +0.138 | +0.013 | −0.016 | +0.072 |
| pooled, without Las Vegas | | +0.049 | +0.013 | −0.055 | +0.028 |

Pooled F1 counts every changed pixel equally, so cities with many changed pixels dominate it: Las Vegas holds 28% of them
and has the largest per-city differences for A and for E, so removing it reduces A's lead from +0.138 to +0.049 and E's
gain from +0.072 to +0.028. The contrasts without a detectable pooled effect (B, async) do not depend on Las Vegas in this
way.

## 5. Missing, synthetic-cloud and real-cloud post-change images

RQ1 assumes that the post-change optical image is unavailable; in practice it is often present but cloudy. All corruptions
below act on the post-change Sentinel-2 image only, and models use their frozen full-input threshold (a deployed model
does not know the cloud level); §5.1 is the exception and uses per-condition thresholds. Zero is the fill value used
here, not an intrinsic missing-data token. Five post-change inputs are distinguished: (1) **missing** — the optical
channels zero-filled (`no_o2`; "image discarded" in §5.5); (2) **unmasked cloudy** — the cloudy image passed through;
(3) **cloud-masked** — the cloudy image with detected (or, synthetically, true) cloud pixels zero-filled and the rest
kept; (4) **copy-forward** — the pre-change image in the post-change slot (§5.1); and (5) **SAR-only inference** — the
separately trained SAR-only U-Net, which ignores optical input. A mask that covers most of the image makes (3) close
to (1).

### 5.1 Whole-modality dropout handles a missing image at no detectable full-input cost (Phase 2)

Table 15 compares A and B with the post-change image present, missing, replaced by the pre-change image, or with
both optical images absent.

**Table 15.** Missing post-change image (Phase 2): test F1 at the validation-selected threshold for each condition, mean
± SD over 10 seeds, 10 test cities (primary family B − A, Holm over 4).

| condition | A full-only | B dropout | Δ (B − A) | 95% CI | B > A | Holm p |
|---|---|---|---|---|---|---|
| full | 0.556 ± 0.029 | 0.553 ± 0.030 | −0.003 | [−0.037, +0.031] | 5/10 | 0.8449 |
| post optical missing (`no_o2`) | 0.190 ± 0.068 | 0.431 ± 0.034 | +0.241 | [+0.176, +0.305] | 10/10 | 5.8e-05 |
| copy-forward (`o2_copy`) | 0.314 ± 0.034 | 0.438 ± 0.038 | +0.124 | [+0.087, +0.160] | 10/10 | 9.5e-05 |
| no optical (`no_opt`) | 0.397 ± 0.106 | 0.440 ± 0.027 | +0.043 | [−0.036, +0.123] | 5/10 | 0.4964 |

B is better with the post-change image missing or replaced by the pre-change image. On full input there is no detectable
cost, although losses up to 0.037 F1 are compatible with the data; at a fixed threshold of 0.5, B is lower (−0.086), so
the dropout model needs its threshold chosen on validation data.

### 5.2 Partial synthetic cloud is a different failure, and dropout does not fix it (Phase 3)

Clouds are real catalogue masks [14] scaled to 10 / 30 / 60% of each city, filled with an opaque, uniform cloud spectrum.

Table 16 gives F1 at three cloud levels and with the image missing.

**Table 16.** Synthetic cloud (Phase 3): test F1 at the frozen clean threshold, mean ± SD over 10 seeds, 10 test cities
(B − A, Holm over 4).

| condition | A full-only | B dropout | Δ (B − A) | Holm p | B > A |
|---|---|---|---|---|---|
| clean | 0.556 ± 0.029 | 0.553 ± 0.030 | | | |
| c10 | 0.285 ± 0.015 | 0.329 ± 0.008 | +0.044 | 9.6e-05 | 10/10 |
| c30 | 0.165 ± 0.007 | 0.191 ± 0.012 | +0.026 | 0.0010 | 10/10 |
| c60 | 0.123 ± 0.002 | 0.134 ± 0.011 | +0.011 | 0.0144 | 10/10 |
| missing | 0.142 ± 0.050 | 0.478 ± 0.030 | +0.336 | 9.5e-08 | 10/10 |

For B, a partly cloudy post-change image is far worse than none (0.329 at 10% cover against 0.478 missing): almost every
pixel under a cloud is predicted as change (false-alarm rate inside the mask 0.96–1.00), and false alarms spill into
nearby clear pixels. Dropout taught B to ignore zero-filled optical input, not cloud-valued input.

### 5.3 Zero-filling under the cloud recovers most of the loss, mainly for the dropout model (Step 4A)

Table 17 shows how much of the cloud-induced loss zero-filling the true mask recovers.

**Table 17.** Zero-filling the true cloud mask (Step 4A): mean test F1 over 10 seeds at the frozen clean threshold;
recovery = share of the cloud-induced F1 loss regained, mean ± SD over seeds (descriptive).

| arm | level | F1 cloud | F1 zero-fill | Δ | recovery of the cloud loss |
|---|---|---|---|---|---|
| A | 10% | 0.285 | 0.393 | +0.108 | 0.403 ± 0.189 |
| A | 30% | 0.165 | 0.261 | +0.097 | 0.249 ± 0.168 |
| A | 60% | 0.123 | 0.192 | +0.069 | 0.161 ± 0.132 |
| B | 10% | 0.329 | 0.539 | +0.210 | 0.939 ± 0.016 |
| B | 30% | 0.191 | 0.516 | +0.325 | 0.899 ± 0.033 |
| B | 60% | 0.134 | 0.499 | +0.365 | 0.872 ± 0.041 |

With the true mask, zero-fill returns B's in-mask false-alarm rate to its clean value; A reads the zero-filled region as
change too. The remedy works well only together with dropout training.

### 5.4 Mask errors, spatial dropout and a real detector on composites (Steps 4A-bis–5)

Four further steps, detailed in Appendix E, qualify §5.3. With synthetic mask errors at 30% cover (Table E1), over-masking
is almost free for B (masking 21% of the clear pixels costs 0.010 F1) while under-masking is expensive (missing a quarter of
the cloud costs 0.168). Training with cloud-shaped zero masks (arm C, Table E2) gave no detectable gain over B in any of 7
primary comparisons. Dilating an under-masking mask repairs missed cloud edges but not missed whole clouds (Table E3). With
real catalogue cloud pixels pasted into the post-change images and masked by s2cloudless at operating points chosen on
validation composites (Table E4), the recall-biased point rb95 beat the most accurate point at 30% and 60% cover, but it
does so by masking 0.596–0.673 of the clear pixels, so its gain is consistent with B falling back on SAR and the pre-change
image rather than with good cloud localisation.

### 5.5 Real cloudy scenes (Step 11)

Step 11 evaluates the frozen A, B and SAR-only models on the real scenes of §3.1, masked by s2cloudless at Step 5's frozen
operating points. Before any change model saw a scene, the detector outputs showed that rb99 masks essentially the whole
image and that rb95 masks 0.10–0.98 of the clear control scenes, so the registered analysis compares masking with
discarding the image.

Table 18 gives the registered contrasts, Table 19 the F1 of each post-change input, and Figure 6 plots them.

**Table 18.** Step 11, 7 partly cloudy cities (city-box cloud 0.08–0.68; scenes 3–15 days from OSCD date 2), F1 at the
frozen full-input thresholds (primary family Holm over 3; secondary Holm over 4).

| contrast | mean Δ | 95% CI | d_z | Δ > 0 | Holm p | label |
|---|---|---|---|---|---|---|
| B: rb95 mask − no mask | +0.208 | [+0.179, +0.236] | +5.22 | 10/10 | 1.5e-07 | higher with the mask |
| B with rb95 mask − SAR-only | +0.020 | [+0.003, +0.036] | +0.86 | 9/10 | 0.0242 | B higher |
| B − A, no mask | +0.087 | [+0.061, +0.113] | +2.36 | 10/10 | 7.6e-05 | B higher |
| B: bal mask − no mask | +0.209 | [+0.182, +0.236] | +5.54 | 10/10 | 8.7e-08 | higher with the mask |
| B: rb95 mask − image discarded | −0.007 | [−0.012, −0.002] | −0.94 | 2/10 | 0.0200 | lower with the mask |
| A, no mask − SAR-only | −0.275 | [−0.291, −0.259] | −12.55 | 0/10 | 8.2e-11 | A lower |
| B: Earth Engine control − OSCD image | −0.002 | [−0.003, −0.001] | −1.03 | 1/10 | 0.0200 | control lower |

**Table 19.** Step 11, test F1 on the 7 partly cloudy cities (mean ± SD over 10 seeds; descriptive).

| post-change input | A full-only | B dropout |
|---|---|---|
| OSCD image (clear) | 0.488 ± 0.030 | 0.527 ± 0.021 |
| Earth Engine control (clear, same day) | 0.481 ± 0.028 | 0.525 ± 0.021 |
| real cloudy scene, no mask | 0.162 ± 0.015 | 0.248 ± 0.041 |
| real cloudy scene, bal mask | 0.164 ± 0.053 | 0.457 ± 0.033 |
| real cloudy scene, rb95 mask | 0.133 ± 0.045 | 0.456 ± 0.016 |
| image discarded | 0.133 ± 0.056 | 0.463 ± 0.011 |

The SAR-only U-Net, which ignores optical input, scores 0.436 ± 0.011 on these cities. Real cloud reproduces the synthetic
result: passed through unmasked, it drops A far below SAR alone and B to 0.248. With cloud pixels zero-filled, B recovers
most of the loss and scores slightly above the SAR-only fallback, a margin that stays positive with any one city left out
(+0.010 to +0.022). Discarding the image was slightly but detectably better than masking it (rb95 mask − image discarded
−0.007, 95% CI [−0.012, −0.002], Holm p 0.0200, 2/10 seeds positive), and B with the image discarded also scored above
SAR-only (+0.026, exploratory). Figure 7 shows why: at the frozen rb95 operating point the detector masks large parts of
each scene, including most of the Chongqing scene whose city-box cloud fraction is 0.08, so the masked input is close
to the missing-image input. The experiment therefore does not show that masking recovers reliable optical information.
On the 3 overcast cities and on all 10, the three primary contrasts have the same sign and are detectable (Table 20). The processing control differs from
OSCD's own image by −0.002. Each city has one real scene, cloud cover is not controlled, and the Las Vegas "overcast"
scene is probably bright desert flagged by the detector rather than opaque cloud, and the cloud fractions that define
the city groups come from the same detector, not from cloud ground truth.

**Table 20.** Step 11 primary contrasts on the 3 overcast cities (Montpellier, Rio, Las Vegas) and on all 10 cities,
F1 at the frozen full-input thresholds (Holm over 3 within each city set).

| city set | contrast | mean Δ | 95% CI | Δ > 0 | Holm p |
|---|---|---|---|---|---|
| 3 overcast | B: rb95 mask − no mask | +0.260 | [+0.217, +0.304] | 10/10 | 7.9e-07 |
| 3 overcast | B with rb95 mask − SAR-only | +0.091 | [+0.028, +0.155] | 8/10 | 0.0120 |
| 3 overcast | B − A, no mask | +0.020 | [+0.007, +0.033] | 9/10 | 0.0120 |
| all 10 | B: rb95 mask − no mask | +0.232 | [+0.200, +0.264] | 10/10 | 1.6e-07 |
| all 10 | B with rb95 mask − SAR-only | +0.056 | [+0.022, +0.091] | 8/10 | 0.0051 |
| all 10 | B − A, no mask | +0.053 | [+0.041, +0.065] | 10/10 | 8.3e-06 |

![Figure 6. Step 11: test F1 pooled over the 7 partly cloudy cities for each post-change input, A (trained on full input only) and B (modality dropout), mean and 95% CI over 10 training seeds at the frozen full-input thresholds. Dashed line and band: SAR-only U-Net (mean and 95% CI), which ignores optical input. "Image discarded" is the missing-image input `no_o2`. One real scene per city.](figures_v2/fig06_cloud_conditions.png)

![Figure 7. Real post-change scenes and their s2cloudless masks at the frozen bal and rb95 operating points (Step 5) for the partly cloudy cities with the lowest (Chongqing), median (Brasilia) and highest (Saclay) city-box cloud fraction; rule fixed before drawing. White: masked, zero-filled before the change model. Panel titles give the masked fraction. Cloud fractions are detector-based, not ground truth. Contains modified Copernicus Sentinel data 2017–2018.](figures_v2/fig07_cloud_masks.png)

## 6. Discussion

### 6.1 What the historical image contributes

Without the post-change optical image, all post-change information comes from SAR. Under our protocol the pre-change
optical image added no detectable change evidence in the late-fusion DS-UNet (B, +0.007 F1 at the full-input threshold,
+0.002 at its own `no_o2` threshold), and a DS-UNet trained only on that input (async) was not detectably different from
the SAR-only U-Net (−0.016). Within B the image raised recall slightly (+0.018) without a detectable change in precision;
this is a small, detectable effect, not an F1 gain. A null is not evidence of no effect: differences up to the upper CI
limits (+0.018 for B) remain compatible with the data.

### 6.2 Comparison with Saha et al. [29]

Saha et al. established that change can be detected on multi-sensor OSCD from pre-change optical and SAR and post-change
SAR, and that, in their models, adding the pre-change optical image raised F1 over SAR-only baselines (early fusion
22.88% → 27.72%; their architecture 32.14%) and removing it at test time lowered F1. Two parts of our results are
consistent with theirs: models without a post-change optical image remain usable, and a model trained with optical input
degrades when that input is removed at test time (A at `no_o2`, 0.190, §5.1), loosely parallel to their test-time
ablation. The direction of the main comparison differs: in our setting-matched arm (async) and in B, the pre-change
optical image gave no detectable F1 gain over SAR alone, and in E it lowered F1.

The two studies differ in many respects, and we did not test which of them, if any, produces the disagreement. They use a
12/2/10 split and we a 10/4/10 split, with the same 10 test cities; their SAR input is two bands per date from the
Sentinel-1 data of [8], clipped to [−25, 0] dB (single acquisition or composite not verified), ours 3-month mean VV;
their architectures (a three-branch Siamese network and an early-fusion network) and training (50 epochs, Adam 1e-4 in
the released code) differ from DS-UNet's late fusion and our recipe; their threshold rule and the number of runs behind
each Table I baseline were not verified; and our SAR-only baseline is trained and selected with the same recipe as the
fusion models. F1 values are not comparable across the two protocols, and we draw no conclusion from their absolute
levels. Resolving the disagreement needs both pipelines run under one protocol.

### 6.3 Why fusion design matters

DS-UNet's late fusion lets the optical stream add only a SAR-independent term to each pixel's score, which could in
principle explain a null. The early-fusion model E, in which the historical image enters the same encoder as SAR, argues
against that explanation for this comparison: given the chance to condition the SAR features, the image lowered F1, and
the difference from its effect in B is detectable (−0.081). The rise in false alarms is consistent with the network
reading an optical image at date 1 next to zeros at date 2 as change, a reading that a quarter of dropout samples did not
remove; we did not test this mechanism. Both results concern one network of each kind and one recipe.

### 6.4 What modality dropout contributes

Dropout training made two models usable without the post-change image and protected B against a missing image
(+0.241 over A at `no_o2`) at no detectable full-input cost (−0.003, with losses up to 0.037 compatible with the data).
With no optical input at all, E scored detectably above the U-Net trained on SAR alone, and the gain replicated on seeds
11–20 and at a fixed threshold of 0.5; B scored above it too, but not detectably (+0.022). Training with optical inputs
that are sometimes removed may act as a regulariser or as cross-modal supervision of the SAR pathway; we did not test the
mechanism.

### 6.5 Why the early-fusion gain needs geographic caution

The replication on new seeds shows that the Step 10 gain is not an accident of training randomness on these cities; it
does not show that it holds elsewhere. In the city analysis the gain is +0.157 within Las Vegas and +0.028 pooled over the
other nine cities, and only 5 of 10 cities share its sign, two of them by +0.005 or less. Pooled pixel metrics weight
cities by their changed pixels: Dubai, Las Vegas and Chongqing hold 77% of them, so one large desert city with a large
per-city difference can dominate a pooled contrast. The same applies to DS-UNet A's lead over the SAR-only U-Net (§4.7).

### 6.6 What the timing experiment can and cannot establish

Under OSCD's label definition the timing of the historical image changes both its age and how much of the labelled change
it may already show, because every image at 1–12 months post-dates OSCD date 1. The experiment therefore measures the
sensitivity of the result to acquisition timing, not the decay of the image's value with age. The registered t-test
found no linear trend (+0.0083 F1 per year, p 0.105); the Wilcoxon test on the same slopes gave p 0.049 with 8 of 10
slopes positive, so a positive slope (older images not worse) is suggestive, not established; and the timings differ in
an exploratory Friedman test, driven by a low 1-month score concentrated in Saclay. An isolated test of image age would
need change labels dated at each historical acquisition, or historical images that all precede the pre-change date and
vary only in age. The decline reported by Song et al. [2] concerns a different role of the image and is not tested here.

### 6.7 Clouds and operational value

A model trained only with full input reads cloud, or zero-filled cloud, as change; on real scenes it fell 0.275 below
SAR alone. A dropout-trained model turns the cloud problem back into the missing-data problem once cloud pixels are
zero-filled, and the outcome then depends on the direction of the mask error: over-masking is cheap, under-masking is
expensive (Appendix E). On real scenes the recall-biased mask removed much of each image (Figure 7), and discarding the
image was slightly better than masking it (by 0.007). The cloud experiments therefore do not show that masking recovers
reliable optical information from a cloudy scene. For this configuration the practical rule is: train with modality
dropout, and when cloud is detected over the scene, drop the post-change optical image rather than pass it through or
mask it heavily; in our data this gave a small, detectable gain over the SAR-only U-Net (+0.026, exploratory) that is
itself partly a threshold effect (§4.2).

### 6.8 Implications for monitoring

For SAR-based urban change monitoring with Sentinel-1 VV, our results do not support adding an older Sentinel-2 image to
a bitemporal model as a routine substitute for a missing post-change image; they do support training with modality
dropout so that a model degrades gracefully when the post-change image is missing or cloudy, selecting thresholds on
held-out data from the target region, and checking per-region results before trusting a pooled score. They do not show
that historical imagery is useless in other roles, such as the semantic prior of [2].

## 7. Limitations

- **One benchmark, fixed test geography.** Results use 9 or 10 OSCD test cities and 4 validation cities. Seeds measure
  training variability on fixed cities; several effects (A's lead over SAR-only, the Step 10 gain) are concentrated in
  Las Vegas, and the city-resampled interval of A's lead includes zero.
- **Nulls and power.** Effects below about 1.00–1.22 SD of the paired differences can be missed; no equivalence margin
  was pre-registered, so nulls are not equivalence.
- **Timing confound.** Image age is confounded with how much labelled change the image shows; one scene per city and
  timing.
- **Unexplained disagreement with [29].** We did not run their pipeline; several of its details could not be verified.
- **Architectures and recipe.** Two fusion designs (late-fusion DS-UNet, one early-fusion U-Net), one recipe tuned on
  DS-UNet A, untuned baselines whose selected thresholds sometimes reach the top of the grid; reconstruction- or
  attention-based models [2], [12], [29] may behave differently.
- **Inputs.** 3-month mean VV backscatter, with date-1 means averaging fewer scenes; Sentinel-1 is placed on the
  Sentinel-2 grid by matching footprints, not by image-based co-registration, so residual SAR–optical misregistration is
  unquantified; OSCD's interpolation of the 20 m and 60 m bands is undocumented.
- **Clouds.** Synthetic and composite clouds are opaque and shadowless. The real-cloud test has one scene per city,
  3–15 days from the label date, uncontrolled cover, cloud fractions defined by the same detector that makes the masks,
  no cloud ground truth, and a Las Vegas "overcast" scene that is probably bright desert.
- **No pseudo-change labels.** RQ3 and RQ5 use false-alarm rates by land-cover class, a stratification, not a label of
  the cause.
- **Internal pre-registration.** Pre-registrations are repository files; Stage 4B followed partial unblinding, the Step 10
  hypothesis came from Step 9's results, and the same AI assistant wrote analysis code, audits and simulated reviews.

**Future work**, in order of what it would resolve: (1) the same paired protocol on an independent multi-sensor change
dataset with more cities, to separate geographic from training variation; (2) running the setting-matched arm and the
architecture of [29] under one protocol, to locate the disagreement; (3) change labels dated at the historical
acquisitions, to isolate image age; (4) real cloudy scenes with cloud ground truth, to separate mask error from model
error; (5) land-cover change labels, to attribute false alarms.

## 8. Conclusion

Under one multi-seed, pre-registered protocol on the 10 OSCD test cities, a historical Sentinel-2 image did not detectably
improve Sentinel-1 VV change detection without the post-change optical image: it made no detectable F1 difference in a
late-fusion DS-UNet and lowered F1 in one early-fusion U-Net, unlike the gain reported by Saha et al. [29] under a
different protocol. Its acquisition time over 1–18 months showed no detectable linear trend under the registered test,
and timing is confounded with change exposure. For SAR alone, adding VH to VV lowered F1. With full input, DS-UNet scored
highest across seeds, and modality dropout gave a SAR-only detector better than SAR-only training; both leads are
concentrated in one city. On real clouds, a dropout-trained model with cloud masked scored slightly above SAR alone,
discarding the cloudy image was slightly better still, and a model trained only on full input fell far below SAR alone.
Whether these results hold on other cities, with dated labels, or with other architectures remains open.

## Declarations

**Data availability.** OSCD [3] and the Cloud Mask Catalogue [14] are public; the Sentinel-1 VV rasters are from the public
DS-UNet repository [1]; Sentinel-1, Sentinel-2 (historical and real-cloud scenes) and ESA WorldCover [19] were exported from
Google Earth Engine [18] with the scripts in the project repository. OSCD change maps are released by their authors under the CC BY-NC-SA licence [34]. Code, pre-registrations, run files, analysis tables,
audits and reports are public at
https://github.com/Manishym956/Multimodal-Satellite-Change-Detection-for-Urban-Development-and-Environmental-Monitoring
(branch `main`; code under the GNU GPL, version 2 or later). Checkpoints, downloaded scenes and raw count arrays are not in
the repository because of their size, and neither are the real-cloud detector outputs. They are identified by SHA-256
hashes in the pre-registrations, in run files such as `experiments/robust/realcloud/detector_outputs/detector_info.json`,
and in `experiments/robust/ARTIFACT_SHA256_*.txt`, and are available from the author on request; they are needed to rerun
training and evaluation and to redraw Figure 7. The real-cloud scene manifest (`datasets/oscd_s2_realcloud/manifest.json`)
is in the repository, so `paper/check_final_submission_v2_numbers.py` and `paper/check_final_submission_v2_tables.py` need
only files in the repository.

**Ethics.** The study uses public satellite imagery and labels only; no human participants or personal data are involved.

**Author contributions (CRediT).** [author: to be agreed with the mentor. Draft: Manish Y. M.: conceptualization,
methodology, software, investigation, formal analysis, data curation, visualization, writing (original draft and editing);
Soumen Mukherjee: supervision, writing (review).]

**Conflict of interest.** [author: confirm.] The author declares no conflict of interest.

**Funding.** [author: confirm.] No external funding.

**Use of AI tools.** An AI assistant (Claude, Anthropic) helped write analysis and evaluation code, audit scripts, report
builders and figures, and drafted parts of the text. All results come from the code and outputs in the repository; the
author checked and is responsible for the content. The same assistant ran the pre-submission integrity checks and simulated
peer review, so those checks are not independent of the drafting.

## References

[1] Hafner, S., Nascetti, A., Azizpour, H., & Ban, Y. (2022). Sentinel-1 and Sentinel-2 data fusion for urban change detection using a dual stream U-Net. *IEEE Geoscience and Remote Sensing Letters, 19*, 1–5. https://doi.org/10.1109/LGRS.2021.3119856

[2] Song, W., Lv, Q., Ding, Z., Hong, S., & Qi, Z. (2026). A dual-modal framework integrating SAR-based change screening and optical-scene-informed identification for high-frequency monitoring of construction-ready bare land. *Remote Sensing, 18*(8), 1103. https://doi.org/10.3390/rs18081103

[3] Caye Daudt, R., Le Saux, B., Boulch, A., & Gousseau, Y. (2018). Urban change detection for multispectral earth observation using convolutional neural networks. In *Proc. IEEE IGARSS*, 2115–2118. https://doi.org/10.1109/IGARSS.2018.8518015

[4] Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics, 6*(2), 65–70. https://www.jstor.org/stable/4615733

[5] Caye Daudt, R., Le Saux, B., & Boulch, A. (2018). Fully convolutional siamese networks for change detection. In *Proc. IEEE ICIP*, 4063–4067. https://doi.org/10.1109/ICIP.2018.8451652

[6] Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. In *MICCAI*, LNCS 9351, 234–241. https://doi.org/10.1007/978-3-319-24574-4_28

[7] Shafique, A., Cao, G., Khan, Z., Asad, M., & Aslam, M. (2022). Deep learning-based change detection in remote sensing images: A review. *Remote Sensing, 14*(4), 871. https://doi.org/10.3390/rs14040871

[8] Ebel, P., Saha, S., & Zhu, X. X. (2021). Fusing multi-modal data for supervised change detection. *International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences, XLIII-B3-2021*, 243–249. https://doi.org/10.5194/isprs-archives-XLIII-B3-2021-243-2021

[9] Saha, S., Ebel, P., & Zhu, X. X. (2022). Self-supervised multisensor change detection. *IEEE Transactions on Geoscience and Remote Sensing, 60*, 4405710. https://doi.org/10.1109/TGRS.2021.3109957

[10] Jiang, J., Xing, Y., Wei, W., Yan, E., Xiang, J., & Mo, D. (2022). DSNUNet: An improved forest change detection network by combining Sentinel-1 and Sentinel-2 images. *Remote Sensing, 14*(19), 5046. https://doi.org/10.3390/rs14195046

[11] Neverova, N., Wolf, C., Taylor, G. W., & Nebout, F. (2016). ModDrop: Adaptive multi-modal gesture recognition. *IEEE Transactions on Pattern Analysis and Machine Intelligence, 38*(8), 1692–1706. https://doi.org/10.1109/TPAMI.2015.2461544

[12] Hafner, S., & Ban, Y. (2023). Multi-modal deep learning for multi-temporal urban mapping with a partly missing optical modality. In *Proc. IEEE IGARSS*, 6843–6846. https://doi.org/10.1109/IGARSS52108.2023.10281626

[13] Ebel, P., Meraner, A., Schmitt, M., & Zhu, X. X. (2021). Multisensor data fusion for cloud removal in global and all-season Sentinel-2 imagery. *IEEE Transactions on Geoscience and Remote Sensing, 59*(7), 5866–5878. https://doi.org/10.1109/TGRS.2020.3024744

[14] Francis, A., Mrziglod, J., Sidiropoulos, P., & Muller, J.-P. (2020). Sentinel-2 Cloud Mask Catalogue (Version 1). Zenodo. https://doi.org/10.5281/zenodo.4172871

[15] Sinergise EO research team (2024). s2cloudless: Sentinel Hub's cloud detector for Sentinel-2 imagery (Version 1.7.3) [Software]. https://github.com/sentinel-hub/sentinel2-cloud-detector

[16] Skakun, S., Wevers, J., Brockmann, C., Doxani, G., Aleksandrov, M., Batič, M., Frantz, D., Gascon, F., Gómez-Chova, L., Hagolle, O., López-Puigdollers, D., Louis, J., Lubej, M., Mateo-García, G., Osman, J., Peressutti, D., Pflug, B., Puc, J., Richter, R., Roger, J.-C., Scaramuzza, P., Vermote, E., Vesel, N., Zupanc, A., & Žust, L. (2022). Cloud Mask Intercomparison eXercise (CMIX): An evaluation of cloud masking algorithms for Landsat 8 and Sentinel-2. *Remote Sensing of Environment, 274*, 112990. https://doi.org/10.1016/j.rse.2022.112990

[17] Bouthillier, X., Delaunay, P., Bronzi, M., Trofimov, A., Nichyporuk, B., Szeto, J., et al. (2021). Accounting for variance in machine learning benchmarks. In *Proceedings of Machine Learning and Systems (MLSys), 3*.

[18] Gorelick, N., Hancher, M., Dixon, M., Ilyushchenko, S., Thau, D., & Moore, R. (2017). Google Earth Engine: Planetary-scale geospatial analysis for everyone. *Remote Sensing of Environment, 202*, 18–27. https://doi.org/10.1016/j.rse.2017.06.031

[19] Zanaga, D., et al. (2021). ESA WorldCover 10 m 2020 v100. Zenodo. https://doi.org/10.5281/zenodo.5571936

[20] Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., et al. (2021). An image is worth 16×16 words: Transformers for image recognition at scale. In *International Conference on Learning Representations (ICLR)*.

[21] Bandara, W. G. C., & Patel, V. M. (2022). A transformer-based Siamese network for change detection. In *Proc. IEEE IGARSS*, 207–210. https://doi.org/10.1109/IGARSS46834.2022.9883686

[22] Chen, H., Qi, Z., & Shi, Z. (2022). Remote sensing image change detection with transformers. *IEEE Transactions on Geoscience and Remote Sensing, 60*, 1–14. https://doi.org/10.1109/TGRS.2021.3095166

[23] Van Etten, A., Hogan, D., Martinez Manso, J., Shermeyer, J., Weir, N., & Lewis, R. (2021). The multi-temporal urban development SpaceNet dataset. In *Proc. IEEE/CVF CVPR*, 6394–6403. https://doi.org/10.1109/CVPR46437.2021.00633

[24] Schmitt, M., Hughes, L. H., Qiu, C., & Zhu, X. X. (2019). SEN12MS – A curated dataset of georeferenced multi-spectral Sentinel-1/2 imagery for deep learning and data fusion. *ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences, IV-2/W7*, 153–160. https://doi.org/10.5194/isprs-annals-IV-2-W7-153-2019

[25] Liu, J., Gong, M., Qin, K., & Zhang, P. (2018). A deep convolutional coupling network for change detection based on heterogeneous optical and radar images. *IEEE Transactions on Neural Networks and Learning Systems, 29*(3), 545–559. https://doi.org/10.1109/TNNLS.2016.2636227

[26] Luppino, L. T., Kampffmeyer, M., Bianchi, F. M., Moser, G., Serpico, S. B., Jenssen, R., & Anfinsen, S. N. (2022). Deep image translation with an affinity-based change prior for unsupervised multimodal change detection. *IEEE Transactions on Geoscience and Remote Sensing, 60*, 1–22. https://doi.org/10.1109/TGRS.2021.3056196

[27] Chen, H., Song, J., Dietrich, O., Broni-Bediako, C., Xuan, W., Wang, J., et al. (2025). BRIGHT: A globally distributed multimodal building damage assessment dataset with very-high-resolution for all-weather disaster response. *Earth System Science Data, 17*(11), 6217–6253. https://doi.org/10.5194/essd-17-6217-2025

[28] Ebel, P., Xu, Y., Schmitt, M., & Zhu, X. X. (2022). SEN12MS-CR-TS: A remote-sensing data set for multimodal multitemporal cloud removal. *IEEE Transactions on Geoscience and Remote Sensing, 60*, 1–14. https://doi.org/10.1109/TGRS.2022.3146246

[29] Saha, S., Shahzad, M., Ebel, P., & Zhu, X. X. (2022). Supervised change detection using prechange optical-SAR and postchange SAR data. *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 15*, 8170–8178. https://doi.org/10.1109/JSTARS.2022.3206898

[30] Zheng, Z., Ma, A., Zhang, L., & Zhong, Y. (2021). Deep multisensor learning for missing-modality all-weather mapping. *ISPRS Journal of Photogrammetry and Remote Sensing, 174*, 254–264. https://doi.org/10.1016/j.isprsjprs.2020.12.009

[31] Kampffmeyer, M., Salberg, A.-B., & Jenssen, R. (2018). Urban land cover classification with missing data modalities using deep convolutional neural networks. *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 11*(6), 1758–1768. https://doi.org/10.1109/JSTARS.2018.2834961

[32] Hafner, S., Ban, Y., & Nascetti, A. (2023). Semi-supervised urban change detection using multi-modal Sentinel-1 SAR and Sentinel-2 MSI data. *Remote Sensing, 15*(21), 5135. https://doi.org/10.3390/rs15215135

[33] Zitzlsberger, G., & Podhoranyi, M. (2024). Monitoring of urban changes with multimodal Sentinel 1 and 2 data in Mariupol, Ukraine, in 2022/23. *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 17*, 5245–5265. https://doi.org/10.1109/JSTARS.2024.3362688

[34] Caye Daudt, R., Le Saux, B., Boulch, A., & Gousseau, Y. (n.d.). Onera Satellite Change Detection dataset: README.txt, "Copyright" section [Dataset documentation]. IEEE DataPort. https://ieee-dataport.org/open-access/oscd-onera-satellite-change-detection

---

## Appendix A: Preliminary study under a simpler protocol

Before the shared protocol of §3, a preliminary study trained re-implemented models on the same split with a simpler
recipe (BCE loss, Adam 1e-4, 50 epochs, no augmentation or importance sampling, threshold 0.5) and 3–5 seeds per model
(Table A1). Its dual-stream model is a re-implementation, not the official DS-UNet; its missing-optical models are
dual-stream networks trained without the post-change optical image (dropping those channels, "clean", or zero-filling
them, "naive"); and it includes two lightweight Transformers [20] (1.16–1.36M parameters), motivated by Transformer change
detectors [21], [22]. Its results are superseded by §4 and matter here for one reason: seed-to-seed SDs of 0.06–0.12 F1
were common, and the first seed alone ranked the clean missing-optical model (0.412) above the synchronous model (0.387),
the opposite of the five-seed mean, which motivated the 10-seed paired design. Under this weaker recipe every baseline
scored far lower than under the shared protocol (SAR-only 0.241 against 0.418 in Step 6), so its missing-optical margin
(0.309 against 0.241, not significant) is not evidence, and its Transformer results (best 0.385, one seed, at a
validation-selected threshold) concern these small models under this recipe only. Pairwise tests and threshold-selected
Transformer results are in the repository (`experiments/paper_statistics.json`,
`experiments/transformer_tuning_2026-10-15/`).

**Table A1: OSCD test results of the preliminary study (10 test cities, τ = 0.5; mean ± sample SD over n seeds)**

| Model | Loss | n | F1 | Precision | Recall | IoU | FAR |
|---|---|---|---|---|---|---|---|
| SAR-only U-Net | pw=1 | 5 | 0.241 ± 0.101 | 0.221 ± 0.152 | 0.342 ± 0.074 | 0.140 ± 0.065 | 0.112 ± 0.106 |
| Optical-only U-Net | pw=1 | 3 | 0.204 ± 0.055 | 0.393 ± 0.176 | 0.167 ± 0.101 | 0.114 ± 0.035 | 0.021 ± 0.022 |
| Early Fusion U-Net | pw=1 | 3 | 0.376 ± 0.022 | 0.343 ± 0.081 | 0.439 ± 0.070 | 0.232 ± 0.017 | 0.050 ± 0.021 |
| Sync Dual-Stream CNN | pw=1 | 5 | 0.459 ± 0.061 | 0.420 ± 0.079 | 0.518 ± 0.074 | 0.299 ± 0.051 | 0.041 ± 0.017 |
| Naive Missing-Optical CNN | pw=1 | 3 | 0.297 ± 0.089 | 0.307 ± 0.132 | 0.324 ± 0.141 | 0.177 ± 0.061 | 0.047 ± 0.035 |
| Clean Missing-Optical CNN | pw=1 | 5 | 0.309 ± 0.095 | 0.428 ± 0.154 | 0.254 ± 0.075 | 0.186 ± 0.066 | 0.023 ± 0.019 |
| Transformer Self-Attn (original) | pw=41 | 3 | 0.199 ± 0.048 | 0.117 ± 0.029 | 0.679 ± 0.178 | 0.111 ± 0.030 | 0.283 ± 0.070 |
| Transformer Cross-Attn (original) | pw=41 | 3 | 0.275 ± 0.124 | 0.190 ± 0.103 | 0.632 ± 0.116 | 0.163 ± 0.080 | 0.240 ± 0.255 |
| Transformer Cross-Attn (pw=1) | pw=1 | 3 | 0.088 ± 0.012 | 0.053 ± 0.001 | 0.408 ± 0.348 | 0.046 ± 0.007 | 0.399 ± 0.345 |
| Transformer Self-Attn (tuned, wd=1e-4) | pw=41 | 3 | 0.212 ± 0.024 | 0.125 ± 0.017 | 0.695 ± 0.018 | 0.118 ± 0.015 | 0.269 ± 0.047 |
| Transformer Cross-Attn (tuned, wd=1e-4) | pw=41 | 3 | 0.231 ± 0.065 | 0.142 ± 0.048 | 0.723 ± 0.168 | 0.131 ± 0.041 | 0.285 ± 0.201 |

*pw = BCE pos_weight. "Original" Transformers use weight decay 0; "tuned" use weight decay 1e-4 selected on validation F1.*

## Appendix B: SEN12-MT note

The preliminary study also trained models on one 21-month SAR–optical interval of a pre-packaged Sentinel-1/2 archive
over SpaceNet 7 [23] sites (distinct from SEN12MS [24]), with proxy labels from building footprints and 4 test sites; all
models scored F1 0.08–0.11 with one seed. The archive's provenance and metadata could not be verified, so these results are
not used. RQ2 is answered by Step 8 on OSCD (§4.4).

## Appendix C: Deviations and gaps

1. **Phase 2 step budget.** A 1000-step budget was first proposed after the test F1 of a 3-seed step sweep had been viewed. That
   choice was discarded and the budget re-chosen on validation score (2500 steps). Seeds 1–3 of A share the configuration of a
   sweep run whose test F1 had been seen; they were retrained from scratch, reproduced identically, and no Phase 2 decision
   used them.
2. **Phase 3 masks.** The first mask build used an absolute cloud-fraction tolerance, which could nearly double a small cloud;
   the masks were rebuilt with a relative tolerance before any model saw a mask. A missing `model.eval()` in the Phase 3
   evaluation script was fixed after a validation-only smoke test, before any test evaluation.
3. **Steps 4A and 4A-bis.** Both pre-registrations cite a validation-only smoke test for which no log exists. In Step 4A-bis,
   the evaluation printed per-seed test F1 while running; the analysis crashed on its final console print and was re-run
   unchanged; two source files hashed in every run file were missing from the pre-registration's hash table (unchanged since
   Phase 3). Inside/outside-mask metrics and IoU were added as a labelled exploratory addendum.
4. **Phase 3 report wording.** `PHASE3_REPORT.md` calls the all-change F1 the "change prevalence"; the prevalence is 0.052. The
   correction is in `experiments/robust/PHASE3_REPORT_ERRATA.md`.
5. **Step 4B.** Test F1 was printed at the end of each training run, as pre-specified. The pre-registration says "33"
   evaluation conditions but lists 34; all 34 were evaluated.
6. **Step 5S.** The validation smoke test's script was not committed, so the smoke test cannot be re-run from the repository.
7. **Step 5.** The scene pools were redesigned before any composite was built (arm C reported as flagged); the composite build
   was interrupted twice and resumed (89 files verified, not byte for byte); a float32/float64 comparison in an export check
   was corrected. The rb99 operating point uses the pre-specified fallback.
8. **Step 6.** No deviation. The machine slept once during training (about 21:04–21:29 IST); the runs were suspended, not
   restarted, and all 40 finished. We did not check separately that the pause left the weights unchanged.
9. **Steps 7 and 8 (one deviation, `0e49c87`).** The Step 7 and Step 8 evaluators lacked `model.eval()` and cuDNN determinism.
   This was found when Step 8's anchor failed on its first checkpoint, diagnosed on validation cities only, and fixed with the
   author's approval before any Step 7 or Step 8 result was saved or read; nothing else changed.
10. **Steps 7 and 8, interruption.** Both were stopped by the author on 2026-10-23 and resumed the next day. Step 8 was re-run
    from scratch and reproduced the 16 files of the interrupted run byte for byte; in Step 7, 5 completed runs were kept and the
    interrupted run was retrained from scratch. That retrain was checked only against the interrupted log's last line: at
    step 1925 its loss (−0.2677) and validation score (0.1623) match to the four decimals the log keeps. Weights were not
    compared, so this is not a bit-identity check.
11. **Error maps.** The selection rule was committed (`afa354e`) before the script first ran; no deviation.
12. **Revision analyses (2026-11-02).** After a simulated peer review of the manuscript, the exploratory analyses of
    §3.4 were added. They were chosen after all results were known, read only stored counts and are not pre-registered;
    no registered analysis or result was changed.
13. **Stage 4B (post-review evaluation).** Registered after the within-B contrast had been seen at B's own `no_o2`
    threshold and at 0.5; only the clean-threshold, per-city and land-cover counts were unseen. The report generator
    reuses the column header "Δ (present − absent)" for its descriptive rows, which are other contrasts; the paper's
    Table 4 labels them correctly.
14. **Line endings (found before Step 9).** Git (`core.autocrlf=true`) had rewritten some committed files with CRLF line
    endings, so their on-disk hashes no longer matched earlier pre-registrations although their content did.
    `train_rrd.py`, `verify_prereg_hashes.py` and `smoke_phase6.py` were restored from the commit before Step 9; three
    files that Steps 9–11 do not use were left with CRLF endings.
15. **Step 9.** E was added after the Stage 3 review named late fusion as an alternative explanation of the RQ1 null; the
    comparison of E without optical input against the SAR-only U-Net was chosen after Step 9's results were seen and was
    then tested on new seeds as Step 10's registered primary hypothesis.
16. **Step 10.** The first smoke-test run failed on a JSON type error in the analysis (a numpy integer), fixed before the
    logged run; the smoke test used only already-reported seed 1–10 runs and printed no metric.
17. **Step 11.** The cloud detector was run on the real scenes before the pre-registration was written, as in Step 5; its
    masked fractions are stated there and led to rb99 being reported descriptively. The smoke test evaluated seed 1 into a
    deleted temporary folder without reading any metric. The Las Vegas scene chosen by the "overcast" rule is probably
    bright desert flagged by the detector; it was kept as the fixed rule selected it.
18. **Steps 9–11, report labels.** The generated reports print labels such as "higher/lower higher"; the paper's labels
    are derived from the same sign and Holm p-value.
19. **Manuscript revision (2026-11-12).** Text revision after an audit of claims against repository outputs
    (`paper/review/PHASE1_CLAIM_AUDIT_2026-11-12.md`) and a literature check that found Saha et al. [29]
    (`paper/review/PHASE2_LITERATURE_2026-11-12.md`). No model was trained or evaluated. Added from existing outputs:
    the Wilcoxon and sensitivity slopes of Step 8 (§4.4), the per-city table (Table 14, from the Stage 4 exploratory
    report and the Step 10 report), the Step 11 overcast and all-city families (Table 20), and figures drawn from stored
    summaries and detector outputs (`paper/figures_v2/`). The Step 11 mask figure's city rule (lowest, median and highest
    city-box cloud) was fixed before drawing. A targeted methods audit (`paper/review/METHODS_AUDIT_2026-11-12.md`)
    then replaced two author-confirmation notes in §3.1 with documented sources and a raster-metadata footprint check
    (`paper/review/methods_audit_footprints.py`); details it could not establish are stated as unknown.

## Appendix D: Reproducibility

Each step's pre-registration, report and audit are in `experiments/robust/`: Phases 2–5 (commits `eead015`, `25fe13b`,
`5dfaa17`, `cde5434`, `3c5d85a`, `f56ba71`, `d9e29ff`, `5f5be9c`, `3ed2fb7`); Step 6 (pre-registration `5476ffe`, results
`2c74a73`); Step 7 (`1ff4251`, `6374275`); Step 8 (`7989248`, `f50d73b`); evaluator deviation `0e49c87`; error maps
(`afa354e`, `b49bd87`); Stage 4B (`260c9b2`, `experiments/robust/stage4b/`); Step 9 (`2a4cf09`, `step9/`); Step 10
(`4c721d8`, `step10/`); Step 11 (`50d3d3d`, `realcloud/`). Each pre-registration's hash table can be re-verified with
`verify_prereg_hashes.py`. Repeat training was checked twice and gave bit-identical weights. Audit scripts recompute every
reported statistic from raw counts: Phase 2, 579 checks; Phase 3, 167; Step 4A-bis, 1338; Step 4B, 432; Step 5S, 826;
Step 5, 731; Step 6, 1046; Step 7, 994; Step 8, 235; all with 0 failures (Step 4A has no audit of its own). Stage 4B and
Steps 9–11 have no separate audit script: their evaluations enforce anchors (for example, Step 11 reproduces Step 5's
stored clean test counts exactly at all 101 thresholds, and Step 10's per-city counts sum to the counts stored at
training), and their analyses recompute every rate from stored counts. The exploratory re-analysis is in
`experiments/robust/stage4/`, the exploratory contrasts quoted in §4.2–5.5 in `experiments/robust/revision/`, and every
number and table cell of this paper is checked against these files by `paper/check_final_paper_numbers.py` and
`paper/check_final_paper_tables.py`. For this revision, `paper/check_final_submission_v2_numbers.py` and
`paper/check_final_submission_v2_tables.py` check `paper/final_submission_v2.md` in the same way, and
`paper/figures_v2/make_figures_v2.py` redraws Figures 1–4 and 6 from files in the repository. Figure 5 is copied unchanged
from the preliminary study's error-map figure (`figures/paper/fig13_error_maps.png`), and Figure 7 also needs the
real-cloud scenes and detector outputs, which are not in the repository (see Data availability).

## Appendix E: Synthetic and composite cloud details (Steps 4A-bis–5)

**Table E1. Step 4A-bis: imperfect masks.** For B at 30% cover, each imperfect mask against the oracle mask (Holm over 9);
every error type costs F1 (Holm p < 0.05, worse than the oracle in 10/10 seeds):

| mask | missed cloud | over-masked clear | B F1 | Δ vs oracle | 95% CI | retained gain |
|---|---|---|---|---|---|---|
| oracle (z30) | 0 | 0 | 0.516 | | | |
| dilate2 | 0.000 | 0.051 | 0.513 | −0.003 | [−0.003, −0.002] | 0.99 |
| dilate4 | 0.000 | 0.107 | 0.511 | −0.005 | [−0.006, −0.005] | 0.98 |
| dilate8 | 0.000 | 0.214 | 0.506 | −0.010 | [−0.012, −0.009] | 0.97 |
| erode2 | 0.116 | 0.000 | 0.416 | −0.100 | [−0.141, −0.060] | 0.70 |
| erode4 | 0.232 | 0.000 | 0.329 | −0.187 | [−0.220, −0.153] | 0.43 |
| erode8 | 0.431 | 0.000 | 0.273 | −0.243 | [−0.272, −0.213] | 0.26 |
| miss10 | 0.100 | 0.000 | 0.426 | −0.090 | [−0.103, −0.077] | 0.73 |
| miss25 | 0.250 | 0.000 | 0.348 | −0.168 | [−0.188, −0.149] | 0.49 |
| miss50 | 0.500 | 0.000 | 0.266 | −0.250 | [−0.276, −0.225] | 0.23 |

Retained gain = (F1_mask − F1_cloud) / (F1_oracle − F1_cloud), computed per seed and averaged, with F1_cloud = 0.191. Over-masking is almost free for B:
masking 21% of the clear pixels costs 0.010 F1. Under-masking is expensive whatever its shape; on missed-cloud pixels, B's
false-alarm rate is 0.909–0.962 for every missed-cloud and erode4/8 variant. Over-masking does hurt the full-input model A:
its false-alarm rate on over-masked clear pixels rises from ≈ 0.024 to 0.385–0.497. An exploratory partition by the true
mask shows that a thin rim of unmasked cloud brings back the Phase 3 spill-over: for B with erode4, the false-alarm rate on
clear pixels outside the cloud is 0.073, the same as with no fill (0.075), against 0.045 with the oracle mask.

**Table E2. Step 4B: spatial dropout training.** C − B test F1 at the frozen clean threshold (Holm over 7):

| condition | B | C | Δ (C − B) | 95% CI | C > B | Holm p |
|---|---|---|---|---|---|---|
| clean | 0.553 ± 0.030 | 0.564 ± 0.034 | +0.011 | [−0.024, +0.046] | 7/10 | 1.0 |
| z10 | 0.539 ± 0.028 | 0.553 ± 0.033 | +0.014 | [−0.019, +0.048] | 7/10 | 1.0 |
| z30 | 0.516 ± 0.028 | 0.529 ± 0.032 | +0.013 | [−0.020, +0.047] | 6/10 | 1.0 |
| z60 | 0.499 ± 0.030 | 0.508 ± 0.028 | +0.009 | [−0.024, +0.041] | 6/10 | 1.0 |
| c30, erode4 | 0.329 ± 0.029 | 0.312 ± 0.034 | −0.018 | [−0.050, +0.015] | 2/10 | 1.0 |
| c30, dilate4 | 0.511 ± 0.028 | 0.522 ± 0.030 | +0.011 | [−0.021, +0.043] | 6/10 | 1.0 |
| c30, miss25 | 0.348 ± 0.007 | 0.341 ± 0.006 | −0.007 | [−0.015, +0.001] | 3/10 | 0.55 |

All 7 primary comparisons are "no detectable difference", and C shows no clean regression. C halves B's whole-image dropout,
so the contrast mixes spatial masks with less whole-image dropout; descriptively, C is lower than B on `no_opt` (−0.028) and
`o2_copy` (−0.039), and C still calls almost every unmasked cloud pixel change (false-alarm rate under the cloud with no fill
0.994, against 0.961 for B).

**Table E3. Step 5S: dilated masks.** B at 30% cover, test F1 with each under-masking mask P and with P dilated by r* = 16 px (Holm over 6), and the remaining difference from the oracle mask after dilation (secondary):

| mask | B F1, P | B F1, P dilated by r* | Δ | 95% CI | Δ > 0 | Holm p | Δ vs oracle at r* |
|---|---|---|---|---|---|---|---|
| erode2 | 0.416 ± 0.050 | 0.500 ± 0.027 | +0.085 | [+0.044, +0.125] | 10/10 | 0.0041 | −0.016 |
| erode4 | 0.329 ± 0.029 | 0.500 ± 0.027 | +0.171 | [+0.138, +0.204] | 10/10 | 4.3e-06 | −0.016 |
| erode8 | 0.273 ± 0.018 | 0.468 ± 0.020 | +0.194 | [+0.172, +0.216] | 10/10 | 6.0e-08 | −0.048 |
| miss10 | 0.426 ± 0.013 | 0.424 ± 0.015 | −0.002 | [−0.005, +0.001] | 4/10 | 0.2382 | −0.092 |
| miss25 | 0.348 ± 0.007 | 0.348 ± 0.008 | +0.000 | [−0.001, +0.002] | 5/10 | 0.7405 | −0.168 |
| miss50 | 0.266 ± 0.010 | 0.267 ± 0.011 | +0.002 | [+0.001, +0.003] | 9/10 | 0.0261 | −0.249 |

Dilation is better for all three eroded masks, with every seed improving, and afterwards they are within 0.048 F1 of the
oracle. For missed whole clouds it makes no practical difference: miss50 meets the label "dilation better" with an effect of
0.002 F1 that the Holm-corrected Wilcoxon test does not support (p 0.059). Dilation can only reach a missed cloud that lies
near a detected one. The validation score was still rising at r* = 16 px, the edge of the grid.

**Table E4. Step 5: composite clouds and a real detector.** These composites are not real cloudy acquisitions: thin cloud carries the foreign surface from its source scene, there are
no matching shadows, and the atmosphere outside the cloud is unchanged. Primary family: B, test F1 at the frozen clean
threshold, Holm over 9.

| contrast | Δ | 95% CI | Δ > 0 | Holm p | label |
|---|---|---|---|---|---|
| c10: rb95 − bal | −0.009 | [−0.019, +0.001] | 3/10 | 0.0672 | no detectable difference |
| c10: rb95 − oracle | −0.055 | [−0.065, −0.045] | 0/10 | 4.4e-06 | rb95 worse |
| c10: rb95 − cloud | +0.036 | [+0.017, +0.055] | 10/10 | 0.0056 | rb95 better |
| c30: rb95 − bal | +0.023 | [+0.008, +0.037] | 9/10 | 0.0119 | rb95 better |
| c30: rb95 − oracle | −0.053 | [−0.067, −0.039] | 0/10 | 6.7e-05 | rb95 worse |
| c30: rb95 − cloud | +0.150 | [+0.125, +0.175] | 10/10 | 2.1e-06 | rb95 better |
| c60: rb95 − bal | +0.057 | [+0.037, +0.078] | 10/10 | 5.6e-04 | rb95 better |
| c60: rb95 − oracle | −0.020 | [−0.024, −0.016] | 0/10 | 6.8e-06 | rb95 worse |
| c60: rb95 − cloud | +0.257 | [+0.223, +0.291] | 10/10 | 3.4e-07 | rb95 better |

**Detector masks on the test composites and B's mean test F1 (descriptive).** Cloud recall and the fraction of clear pixels masked
are computed against the true mask with no change model; rb99 is the flagged fallback mask.

| level | rb95 cloud recall | rb95 clear masked | bal cloud recall | bal clear masked | B cloud | B bal | B rb95 | B rb99 | B oracle | rb95 retained gain |
|---|---|---|---|---|---|---|---|---|---|---|
| c10 | 0.892 | 0.596 | 0.532 | 0.069 | 0.447 | 0.492 | 0.483 | 0.478 | 0.538 | 0.35 ± 0.21 |
| c30 | 0.837 | 0.615 | 0.555 | 0.087 | 0.315 | 0.442 | 0.465 | 0.478 | 0.518 | 0.74 ± 0.06 |
| c60 | 0.963 | 0.673 | 0.747 | 0.094 | 0.217 | 0.417 | 0.474 | 0.478 | 0.494 | 0.93 ± 0.02 |

The recall-biased mask beats the most accurate one at 30% and 60% cover and beats no fill at every level, but stays below
the oracle mask. It retains 0.740 of the oracle gain at 30% cover, just below the pre-specified 0.75, so no pre-specified
decision branch applies exactly. Three results qualify the gain: the detector misses more cloud on test than on validation
(11–16% of test cloud pixels reach the model at 10% and 30% cover); rb95 reaches its recall by masking 0.596–0.673 of the
clear pixels and costs B 0.054 F1 on clean images; and rb99, which masks almost the whole post-change image, gives 0.478 at
every level, above rb95 at 30% cover. The gain over bal is therefore consistent with B falling back on SAR and the
pre-change image, not evidence that the detector locates cloud well. For A, rb95 is worse than bal at every level (−0.148,
−0.083, −0.032; Holm p < 0.05).

**Table E5. SAR-only fallback rule on composite clouds (exploratory).** A simpler operational rule is to use A when the post-change optical image is available and clear, and the SAR-only U-Net
otherwise. The SAR-only score (0.418 ± 0.043) does not depend on the optical image, so it is the same at every cloud
level. Mean difference B − SAR-only in test F1 with its 95% CI over seeds, B at its frozen clean threshold on Step 5's
composite clouds, by cloud mask:

| cloud cover | no mask | bal | rb95 | rb99 | oracle |
|---|---|---|---|---|---|
| c10 | +0.029 [−0.004, +0.062] | +0.074 [+0.041, +0.108] | +0.065 [+0.032, +0.098] | +0.060 [+0.026, +0.094] | +0.120 [+0.086, +0.154] |
| c30 | −0.103 [−0.137, −0.070] | +0.024 [−0.010, +0.058] | +0.047 [+0.014, +0.079] | +0.059 [+0.026, +0.093] | +0.099 [+0.067, +0.132] |
| c60 | −0.202 [−0.236, −0.167] | −0.001 [−0.036, +0.033] | +0.056 [+0.023, +0.088] | +0.060 [+0.026, +0.094] | +0.076 [+0.043, +0.109] |

When the post-change image is available and clear, the rule uses A: B is not detectably different in F1 (−0.003) but has
lower precision (−0.101, 95% CI [−0.161, −0.041]), higher recall (+0.120) and a higher false-alarm rate (+0.018). When
the image is missing, the rule uses SAR-only, which B does not detectably beat at its registered threshold (+0.013). On
composite clouds without a mask, B is below the SAR-only fallback at 30% and 60% cover; with the recall-biased detector
mask rb95 it is above it at every level, and with rb99, which discards almost the whole post-change image, by +0.059 to
+0.060. Because rb99 nearly removes the image, much of B's advantage here is the clean-threshold margin of §4.2 rather
than information from the masked image. These comparisons use composites, not real cloudy acquisitions, and were made
after the results were known.
