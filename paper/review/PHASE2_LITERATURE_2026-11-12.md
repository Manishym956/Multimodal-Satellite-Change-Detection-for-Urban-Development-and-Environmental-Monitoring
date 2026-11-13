# Phase 2 — literature verification, comparison table and novelty statement (2026-11-12)

Methods: Crossref metadata lookup for every DOI; abstracts from Semantic Scholar / OpenAlex; full text where obtainable
(local `sources/` PDFs, arXiv). Only titles/DOIs were sent to external services. "Full text" means I read the relevant
passage; "abstract" means only the abstract was available to me.

## 1. Verification of the existing reference list

| Ref | Metadata (Crossref) | Claim made in manuscript | Evidence level | Verdict |
|---|---|---|---|---|
| [1] Hafner et al. 2022 LGRS | matches | DS-UNet, VV + S2, decision-level fusion, beats unimodal and input fusion on OSCD+S1 | full text (`sources/Researchpapers.pdf`) + code | OK |
| [2] Song et al. 2026 RS 18:1103 | matches | VH screening + historical cloud-free S2 as semantic prior, Guangzhou; gaps 1/3/6/12 months; F1 83.99% → 78.31%; gap results without repeats | full text (`sources/remotesensing-18-01103.pdf`, §Table 4) | OK. Add: the gap table is a single run (main Table 2 is 5 seeds); their evaluation classifies **candidate patches**, not pixels; trend is not monotonic in IoU (59.09, 61.95, 55.46, 47.75) |
| [3] Caye Daudt et al. 2018 IGARSS | matches | OSCD | metadata | OK |
| [4] Holm 1979 | JSTOR stable URL | — | standard | OK |
| [5] Caye Daudt et al. 2018 ICIP | matches | FC Siamese CD | metadata | OK |
| [6] Ronneberger 2015 | matches | U-Net | metadata | OK |
| [7] Shafique et al. 2022 | matches | review | metadata | OK |
| [8] Ebel, Saha, Zhu 2021 | matches | fuse S1+S2 in Siamese networks; curated OSCD+S1 dataset | abstract | OK; **also the source of the multi-sensor OSCD used by [29]** |
| [9] Saha, Ebel, Zhu 2022 TGRS | matches (article 4405710) | self-supervised CD across sensors | abstract | OK (uses 4 bitemporal scenes, unlabeled) |
| [10] Jiang et al. 2022 DSNUNet | matches | S1+S2 forest change | abstract | OK |
| [11] Neverova et al. 2016 ModDrop | matches | modality dropout, gesture recognition | metadata | OK |
| [12] Hafner & Ban 2023 IGARSS | matches | on missing-optical samples: proposed 0.327, zero-fill DS-UNet 0.233, SAR U-Net 0.348 | full text (arXiv 2306.00640, Table 1, 5 runs, SpaceNet 7) | OK |
| [13] Ebel et al. 2021 TGRS cloud removal | matches | SAR-guided cloud removal | metadata | OK |
| [14] Francis et al. 2020 Cloud Mask Catalogue | Zenodo DOI resolves | real cloud masks | DOI | OK |
| [15] s2cloudless v1.7.3 | GitHub software | detector | — | OK as software citation; year/version should match the installed package (to confirm from `.venv`) |
| [16] Skakun et al. 2022 CMIX | matches | masking algorithms differ | metadata | OK |
| [17] Bouthillier et al. 2021 MLSys | no DOI (proceedings) | variance changes benchmark conclusions | known venue | OK |
| [18] Gorelick et al. 2017 | matches | GEE | metadata | OK |
| [19] Zanaga et al. 2021 WorldCover | Zenodo DOI resolves | land cover | DOI | OK |
| [20]–[22] | match | Transformers (Appendix A only) | metadata | OK |
| [23] Van Etten et al. 2021 CVPR | **pages 6394–6403** | — | Crossref | **Fix pages (manuscript has 6398–6407)** |
| [24]–[28] | match | — | metadata | OK |

No reference in the list is fabricated. One page-range error ([23]).

## 2. Uncited work that changes the novelty claim

**[29] Saha, S., Shahzad, M., Ebel, P., & Zhu, X. X. (2022). Supervised change detection using prechange optical-SAR
and postchange SAR data. *IEEE JSTARS, 15*, 8170–8178. https://doi.org/10.1109/JSTARS.2022.3206898**

- Abstract: proposes "a novel change detection data setting which uses both optical and SAR images pre-change, yet only
  SAR imagery post-change", motivated by clouds; a Siamese SAR encoder plus a separate pre-change optical encoder, fused
  and decoded by a common decoder.
- Dataset: the authors' code README (github.com/sudipansaha/optSarSarCdJstars2022) instructs downloading the
  **OSCD multi-sensor dataset of Ebel et al. [8]**, i.e. the same benchmark family as ours.
- **Full text checked by the author (2026-11-12; I could not access it, HTTP 403, so these are author-reported):**
  - Split: multi-sensor OSCD, 24 cities, **12 train / 2 validation / 10 test** (ours: 10 / 4 / 10).
  - Table I, F1 (%): early fusion, pre-SAR/post-SAR 22.88; Siamese, pre-SAR/post-SAR 23.08; early fusion,
    pre-optical+SAR/post-SAR 27.72; proposed, pre-optical+SAR/post-SAR 32.14 (all F1 in %).
  - The proposed method is averaged over 3 random seeds (whether the baselines are also averaged: not stated in the
    author's summary; to confirm).
  - Test-time modality-dropout ablation (Table III, §IV-D): dropping pre-change optical −7.04 F1 points; dropping
    pre-change SAR −4.81.

Consequences:
1. The manuscript sentence "Whether a historical optical image adds change evidence to bitemporal SAR … ha[s] not been
   measured on a public change detection benchmark" (§1) is **false** and must be removed. Saha et al. proposed the
   setting **and** compared it with SAR-only baselines on the same benchmark family.
2. **Their direction differs from ours.** They report the pre-change optical input raising F1 over SAR-only (+4.84
   points with early fusion; +9.06 with their architecture). Our closest analogue is the `async` arm (DS-UNet trained only
   on the `no_o2` input, i.e. without dropout): async@no_o2 − SAR-only = −0.016, 95% CI [−0.059, +0.026], no detectable
   difference; B and E show no detectable gain or a loss. This disagreement must be stated in the paper, not hidden.
   Candidate reasons (none tested; to be listed as such): different splits and validation sets; different SAR inputs
   (DS-UNet 3-month VV means here); different architectures and losses; a much stronger SAR-only baseline here (absolute
   F1 values are **not comparable** across protocols and must not be compared as evidence of superiority); number of runs
   per baseline; threshold selection.
3. Test-time dropout ablation is already in [29]; our addition is **training-time** dropout and its comparison with a
   full-input model (A) and a no_o2-only model (async).

## 3. Other verified additions

| Ref | Citation | Why it matters | Evidence |
|---|---|---|---|
| [30] | Zheng, Z., Ma, A., Zhang, L., & Zhong, Y. (2021). Deep multisensor learning for missing-modality all-weather mapping. *ISPRS J. Photogramm. Remote Sens., 174*, 254–264. https://doi.org/10.1016/j.isprsjprs.2020.12.009 | Missing-optical (SAR-only at test) mapping that leverages **historical multisensor data** during training; segmentation, not CD | abstract |
| [31] | Kampffmeyer, M., Salberg, A.-B., & Jenssen, R. (2018). Urban land cover classification with missing data modalities using deep convolutional neural networks. *IEEE JSTARS, 11*(6), 1758–1768. https://doi.org/10.1109/JSTARS.2018.2834961 | Hallucination network for a modality missing at test time (optical + DSM) | abstract |
| [32] | Hafner, S., Ban, Y., & Nascetti, A. (2023). Semi-supervised urban change detection using multi-modal Sentinel-1 SAR and Sentinel-2 MSI data. *Remote Sensing, 15*(21), 5135. https://doi.org/10.3390/rs15215135 | S1+S2 urban CD on SpaceNet 7 with cross-modal consistency; assumes both modalities | abstract |
| [33] | Zitzlsberger, G., & Podhoranyi, M. (2024). Monitoring of urban changes with multimodal Sentinel 1 and 2 data in Mariupol, Ukraine, in 2022/23. *IEEE JSTARS, 17*, 5245–5265. https://doi.org/10.1109/JSTARS.2024.3362688 | S1+S2 time series with cloud-masked, irregular observations; ablation on loss of observations; found stronger dependence on optical than SAR | full text (`sources/1939-1404-2024v17p5245.pdf`) |
| [34] | Wang, Y., Makrogiannis, S., & Kambhamettu, C. (2026). FAF-CD: Frequency-aware fusion for change detection under imperfect multimodal remote sensing. arXiv:2606.03114 (preprint) | Asynchronous, cross-sensor pre/post pairs (EO pre-event, SAR post-event, BRIGHT) | full text (local); **preprint, not peer-reviewed** — cite only if needed |

Considered but **not added** (could not verify the specific content needed): Garnot et al. 2022 (ISPRS, multimodal
temporal attention; abstract unavailable to me), Ma et al. 2022 CVPR (missing modality in Transformers; outside RS —
optional), MCD-Mamba 2026 (IJRS, multimodal OSCD; not about missing/asynchronous optical).

## 4. Literature comparison table (for the manuscript)

| Study | Dataset | Modalities | Fusion | Missing-modality handling | Historical / asynchronous optical | Cloud evaluation | Limitation relevant here |
|---|---|---|---|---|---|---|---|
| Caye Daudt et al. [5] | OSCD | S2 bitemporal | early fusion; Siamese | none | none | none (cloud-free pairs) | optical only |
| Hafner et al. [1] | OSCD + S1 | S1 VV + S2, both dates | late (decision-level) dual-stream | none | none | none | assumes synchronous optical; single-protocol comparison |
| Ebel et al. [8] | OSCD + S1 (curated) | S1 + S2, both dates | Siamese fusion | none | none | none | assumes both modalities at both dates |
| Saha et al. [29] | OSCD multi-sensor [8]; 12/2/10 split | pre: S2 + S1; post: S1 | separate optical encoder + shared SAR Siamese, common decoder; early-fusion baseline | post-change optical absent by design; **test-time** dropout ablation | **yes: pre-change optical; compared with SAR-only baselines (gain reported)** | none (author-reported) | proposed method averaged over 3 seeds; no timing, cloud or city-level analysis (author-reported) |
| Hafner & Ban [12] | SpaceNet 7 + S1/S2 | S1 + S2 monthly | dual-stream + SAR→optical feature reconstruction | partly missing optical at inference (~12% of test timestamps) | no | missing = cloud-affected timestamps, not masked pixels | urban mapping, not CD; reconstruction below SAR-only on missing samples |
| Zheng et al. [30] | high-res optical/SAR all-weather mapping set | optical + SAR (0.5 m) | meta-sensory prototype network | SAR-only at test | historical multisensor data used in training | not evaluated per cloud | segmentation; VHR, not Sentinel |
| Kampffmeyer et al. [31] | urban land cover (optical + DSM) | optical + DSM | hallucination network | DSM missing at test | no | no | not SAR; not CD |
| Neverova et al. [11] | gesture recognition | video/audio/pose | multi-scale fusion | ModDrop training | no | no | outside RS |
| Song et al. [2] | Guangzhou (own) | S1 VH + S2 (RGB+NIR) | two-stage: SAR screening, optical semantic refinement | post-change optical absent by design | **yes: historical S2 as semantic prior; gaps 1–12 months** | motivation only | single region; patch-level evaluation; gap results single-run |
| Zitzlsberger & Podhoranyi [33] | Mariupol (own) | S1 + S2 time series | deep-temporal network | cloud/out-of-swath observations masked; loss-of-observation ablation | time-series context | operational cloud masks | one site; relies more on optical |
| Ebel et al. [13], [28] | SEN12MS-CR(-TS) | S1 + S2 | SAR-guided reconstruction | reconstruct cloudy optical | multitemporal inputs | real clouds | cloud removal, not CD |
| **This study** | OSCD + S1 (DS-UNet rasters) | S1 VV (VH for RQ3) + S2 | late (DS-UNet) and early (U-Net) fusion | whole-modality dropout; zero-fill; cloud masking | historical S2 in the pre-change slot; 1–18-month timings | synthetic, composite and real cloudy scenes (one per city) | 10 test cities; one benchmark; seeds ≠ cities |

## 5. Proposed novelty statement

> Saha et al. [29] introduced the setting in which only SAR is available after the change and pre-change optical and SAR
> images are available before it, and reported on the multi-sensor OSCD benchmark that adding the pre-change optical
> image raised F1 over SAR-only baselines. Song et al. [2] used a historical optical image as a semantic prior for
> SAR-based screening and reported lower F1 for older images. We re-examine the contribution of the historical optical
> image in this setting under a more controlled protocol: paired comparisons over 10 training seeds with pre-specified
> tests and a fresh-seed replication; training with modality dropout, compared with full-input and no-post-optical
> training; a late-fusion (DS-UNet) and an early-fusion design; the acquisition timing of the historical image; real
> cloudy post-change scenes, including whether masking cloud beats discarding the image; and the dependence of pooled
> results on individual test cities. We propose no new architecture.

What is **not** claimed as new: the setting and the SAR-only comparison ([29]); test-time modality ablation ([29]);
training with samples lacking optical data ([12], mixed multimodal/missing-optical mini-batches; [11] for modality
dropout in general); a temporal-gap experiment ([2]); evaluation on real cloud-affected inputs ([12], [33]). What our
experiments add is the combination of these factors under one shared, pre-registered, multi-seed protocol, and the
finding that, under that protocol, the historical image gave no detectable F1 gain (late fusion) or lowered F1 (one
early-fusion design), which **differs in direction from [29]**.

## 6. Open items for the author

1. ~~Obtain Saha et al. [29]~~ done by the author (2026-11-12). Still to confirm: whether Table I baselines are
   single runs or seed averages, which S1 product/polarisations they use, and their threshold rule (these determine how
   the disagreement in direction can be discussed).
2. Confirm the installed s2cloudless version for [15].
3. Decide whether to cite the preprint [34]; recommendation: no, unless a reviewer asks about heterogeneous EO→SAR CD.
