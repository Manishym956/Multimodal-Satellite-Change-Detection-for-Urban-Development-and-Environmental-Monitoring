# Literature review and positioning (Phase 7 of the final plan)

2026-10-23. Bibliographic data checked against Crossref, DBLP or the publisher page. **Verification level:** "abstract"
means the claim below was checked against the abstract or publisher summary only; such claims must be re-checked against
the full text before submission. Nothing here is taken from the RRD's summary table alone.

## 1. Papers

| # | Reference (verified metadata) | Problem / inputs / data | Contribution | Limitation relevant here | Verified |
|---|---|---|---|---|---|
| H22 | Hafner, Nascetti, Azizpour, Ban (2022). Sentinel-1 and Sentinel-2 data fusion for urban change detection using a dual stream U-Net. *IEEE GRSL* 19, 4019907. doi:10.1109/LGRS.2021.3119856 | Bitemporal urban change; S1 VV + S2 at both dates; OSCD enriched with S1 | Dual-stream U-Net with decision-level fusion beats unimodal U-Nets and input-level fusion | Assumes all four images; VV only | Code and data (repo), abstract |
| S26 | Song, Lv, Ding, Hong, Qi (2026). A dual-modal framework integrating SAR-based change screening and optical-scene-informed identification for high-frequency monitoring of construction-ready bare land. *Remote Sensing* 18(8), 1103. doi:10.3390/rs18081103 | Construction-ready bare land, Guangzhou; Sentinel-1 **VH** change screening (annual-mean T1 baseline) + **historical** cloud-free S2, no synchronous optical | Two-stage SCS-OI with ASG-Net; SAR-only FR 22.4% vs lower with the optical prior. **§5.4 varies the gap: 1, 3, 6, 12 months before T2 (closest cloud-free acquisition in a window); F1 83.99 → 83.24 → 81.51 → 78.31%, IoU 59.09 → 47.75%, recall stable (87.86–91.29%)**. §5.1 argues VH over VV physically (depolarization loss on leveled bare soil; VH less sensitive to soil moisture) | One region, one change type; gap results appear to be single runs (no seeds or intervals reported in the text read); VV vs VH argued from backscatter analysis, not a model ablation | Full text (§2, §5.1, §5.4, Table 4) |
| HB23 | Hafner, Ban (2023). Multi-modal deep learning for multi-temporal urban mapping with a partly missing optical modality. *IGARSS 2023*, 6843–6846. doi:10.1109/IGARSS52108.2023.10281626 | Multi-temporal urban **mapping** (not change detection); optical partly missing due to clouds | Reconstructs optical features from SAR when optical is missing; beats zero-fill and SAR-only | Mapping task; reconstruction not tested under imperfect cloud masks | Abstract |
| E21 | Ebel, Saha, Zhu (2021). Fusing multi-modal data for supervised change detection. *ISPRS Archives* XLIII-B3-2021, 243–249. doi:10.5194/isprs-archives-XLIII-B3-2021-243-2021 | OSCD + co-registered S1; bitemporal, both modalities at both dates | Siamese SAR–optical fusion; new multi-modal CD dataset | Synchronous inputs; no missing/cloudy optical | Abstract |
| SEZ22 | Saha, Ebel, Zhu (2022). Self-supervised multisensor change detection. *IEEE TGRS* 60, 4405710. doi:10.1109/TGRS.2021.3109957 | Pre/post images from **different** sensors (optical vs SAR); unlabeled | Self-supervised (deep clustering, contrastive) multisensor CD | Four scenes; different task (cross-sensor, unsupervised) | Abstract |
| J22 | Jiang et al. (2022). DSNUNet: an improved forest change detection network by combining Sentinel-1 and Sentinel-2 images. *Remote Sensing* 14(19), 5046. doi:10.3390/rs14195046 | Forest change; S1 + S2 | Dual-stream network for forest change | Forest only | Metadata, abstract |
| D18 | Daudt, Le Saux, Boulch (2018). Fully convolutional siamese networks for change detection. *ICIP*, 4063–4067. doi:10.1109/ICIP.2018.8451652 | Optical bitemporal CD | Siamese FC networks | Optical only | Metadata |
| N16 | Neverova, Wolf, Taylor, Nebout (2016). ModDrop: adaptive multi-modal gesture recognition. *IEEE TPAMI* 38(8), 1692–1706 | Gesture recognition | Modality dropout for robustness to missing modalities | Not remote sensing | DBLP (volume, issue, pages) |
| E20 | Ebel, Meraner, Schmitt, Zhu (2021). Multi-sensor data fusion for cloud removal in global and all-season Sentinel-2 imagery. *IEEE TGRS* 59(7), 5866–5878. doi:10.1109/TGRS.2020.3024744 | SAR-guided cloud removal; SEN12MS-CR dataset | Reconstructs cloud-free S2 from cloudy S2 + S1 | Reconstruction, not change detection | DLR record |
| SK22 | Skakun, Wevers, Brockmann, et al. (2022). Cloud Mask Intercomparison eXercise (CMIX). *Remote Sens. Environ.* 274, 112990. doi:10.1016/j.rse.2022.112990 | 10 cloud-masking algorithms (9 teams) vs several reference datasets, Landsat 8 and S2 | Performance depends strongly on the reference dataset (cloud definitions differ) | — | Abstract (author list to complete) |
| F20 | Francis, Mrziglod, Sidiropoulos, Muller (2020). Sentinel-2 Cloud Mask Catalogue. Zenodo. doi:10.5281/zenodo.4172871 | Labelled S2 cloud masks | Source of cloud shapes and pixels in Phases 3–5 | — | Zenodo |
| ZS | Sentinel Hub (Zupanc). Improving cloud detection with machine learning (s2cloudless). https://medium.com/sentinel-hub/improving-cloud-detection-with-machine-learning-c09dc5d7cf13 | Pixel-based S2 cloud detector on 10 bands | Detector used in Step 5 | Blog post: date given as 2017 by one source and 2020 by Crossref; cite with the s2cloudless package version (1.7.3) | URL |

## 2. Answers to the review questions

1. **Historical optical guidance (RQ1).** S26 reports that historical cloud-free optical imagery used as semantic
   context reduces SAR false alarms for one change type in one region. HB23 handles missing optical by reconstructing it
   from SAR, for mapping. Neither measures, on a public change benchmark with multiple training seeds, whether historical
   optical beats SAR alone for urban change. That is what Step 6 tests.
2. **Temporal separation (RQ2).** S26 (§5.4) already varied the gap (1, 3, 6, 12 months) for construction-ready bare
   land in one region and found F1 falling from 83.99% to 78.31%. Step 8 is therefore a **test of that finding** in a
   different setting (urban change, 10 public test cities, 10 training seeds, 1–18 months, with the confound that a
   recent image may already show the change measured explicitly), not a first study of the gap.
3. **VV/VH (RQ3).** S26 chose VH over VV from backscatter physics for bare-land formation; no model ablation of VV, VH
   and VV+VH for urban change detection was found. Related work on floods reports opposite preferences (VH better for
   deep-learning flood segmentation; VV more sensitive to urban flooding). H22 used VV only because VH was not available
   for early acquisitions. Step 7 is a controlled ablation on OSCD.
4. **Difference from asynchronous methods.** S26 proposes a method; this study measures. It isolates the missing and
   the cloudy optical cases, tests zero-fill under real and synthetic mask errors, and evaluates gap, polarization and
   baselines under one audited, multi-seed protocol on a public benchmark.
5. **New vs replication.** Replication or extension: dual-stream ≥ simpler fusion (H22) under a different protocol;
   modality dropout (N16) applied to SAR–optical CD; the gap effect (S26) and the VH preference (S26) tested on a public
   urban benchmark with seeds. New (subject to the remaining checks): the missing-vs-cloudy separation, the asymmetric
   cost of mask errors, the real-detector test on composite clouds, and the measured confound between gap and visible
   change.

## 3. Still to do before submission

- S26: confirm from Table S1 / §4 whether the gap results are single runs, and the baselines' protocol. HB23: full-text
  check of the zero-fill baseline details.
- Complete the SK22 author list from the publisher page.
- Targeted search for SAR–optical change detection with missing or cloudy optical input published 2023–2026.
- Claim-to-evidence matrix: filled in when Steps 6–8 are reported.

Sources: [Song et al. 2026](https://mdpi-res.com/d_attachment/remotesensing/remotesensing-18-01103/article_deploy/remotesensing-18-01103.pdf);
[Hafner & Ban 2023, arXiv 2306.00640](https://export.arxiv.org/abs/2306.00640); [Ebel et al. 2021, ISPRS](https://isprs-archives.copernicus.org/articles/XLIII-B3-2021/243/2021/);
[Saha et al. 2022, arXiv 2103.05102](https://arxiv.org/abs/2103.05102v3); [DSNUNet](https://www.mdpi.com/2072-4292/14/19/5046);
[ModDrop, arXiv 1501.00102](https://ar5iv.arxiv.org/html/1501.00102); [CMIX, NASA NTRS](https://ntrs.nasa.gov/citations/20220006382);
[Ebel et al. 2020, arXiv 2009.07683](https://arxiv.org/pdf/2009.07683); [s2cloudless PyPI](https://pypi.python.org/pypi/s2cloudless);
[VH vs VV flood study](https://isprs-annals.copernicus.org/articles/X-4-W8-2025/501/2026/isprs-annals-X-4-W8-2025-501-2026.pdf);
[urban flooding VV/VH](https://www.mdpi.com/2072-4292/13/22/4511); Crossref API for DOIs.
