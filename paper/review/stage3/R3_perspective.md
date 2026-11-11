# Peer Review Report

## Manuscript Information
- **Title**: Multimodal Satellite Change Detection for Urban Development and Environmental Monitoring
- **Manuscript ID**: FINAL_RESEARCH_PAPER.md (SHA-256 7220d00b9c4e2ec08803faceeacc5437d4876243edf89ea91565c9514aa1bc8f, verified before review)
- **Review Date**: 2026-11-02
- **Review Round**: Round 1 (Stage 3, full mode)

---

## Reviewer Information

### Reviewer Role *
Peer Reviewer 3 (Perspective)

### Reviewer Identity *
Operational Earth-observation practitioner who builds change-monitoring services for urban planning and environmental agencies, with a background in Sentinel-2 cloud screening. I review as an outsider to the deep-learning benchmarking tradition: I am asking whether a service operator could act on these results. Statistical detail is outside my remit and I defer on it to Reviewer 1.

### Review Focus *
(1) Whether the simulated/composite-cloud robustness results (§5) and the closing practical recommendation (modality-dropout DS-UNet with a validation-selected threshold) would transfer to real cloudy acquisitions and to an operational pipeline. (2) Whether the title's "environmental monitoring" claim is supported. (3) Whether the released code and data are usable by practitioners, and which ideas from adjacent fields (missing-data methods, test-time adaptation, operational accuracy assessment) would strengthen the work.

---

## Overall Assessment *

### Recommendation *
- [ ] **Accept**
- [ ] **Minor Revision**
- [x] **Major Revision**
- [ ] **Reject**

### Confidence Score *
4 — operational Sentinel-2 cloud screening and change-service design are core expertise; deep-learning training detail is adjacent.

Confidence is an uncertainty/scope disclosure only; it never changes consensus counts, severity, decision bearing, or arbitration.

### Calibration Status
`NOT_CALIBRATED`

### Summary Assessment *
The paper is a measurement study of the official DS-UNet and U-Net baselines on OSCD with Sentinel-1, answering five pre-set research questions with 10 paired seeds, plus a seven-step robustness study of missing and cloudy post-change Sentinel-2 input. Its headline answers are mostly nulls: historical optical imagery gives no detectable gain over bitemporal SAR, timing has no detectable effect, and VH does not help. The robustness study adds that a partly cloudy image is worse than a missing one and that over-masking is cheap while under-masking is expensive.

From an operational view, the reporting is unusually honest. The paper states its limits on clouds, labels and power, and the mask-error asymmetry (§5.4) is a directly usable result for anyone tuning a cloud-mask threshold. Three things keep it from being operationally actionable as written. First, the title promises "environmental monitoring", but only urban change labels were tested, and §4.4/§7 concede that environmental pseudo-change cannot be assessed. Second, no result uses a real cloudy acquisition, although the authors' own Earth Engine pipeline (Step 8) already retrieves real, partly cloudy scenes. Third, the closing recommendation is never compared with the simplest operational policy: switching to the SAR-only model whenever cloud is detected. That policy is cloud-immune at 0.418 F1 and competitive with every masked-B result below 30% cover. All three are repairable through retitling, one added experiment on real scenes, and one added comparator, so I recommend Major Revision.

---

## Strengths *

### S1: Mask-error asymmetry is a directly actionable operational finding
§5.4 shows that, for the dropout model, over-masking 21% of clear pixels costs about 0.010 F1, while missing a quarter of the clouds costs about 0.168. This gives a cloud-screening operator a concrete design rule: tune the detector toward recall. §5.7 then checks the rule with a real detector, s2cloudless, rather than leaving it at synthetic masks. Few change-detection papers connect model behaviour to cloud-mask operating points this explicitly.
**Evidence Anchor**: table: §5.4 table — dilate8 Δ vs oracle −0.010 against miss25 −0.168

### S2: Honest separation of "missing" from "partly cloudy" input
§5.2 shows that robustness to a zero-filled image does not carry over to cloud-valued pixels (B: 0.329 at 10% cloud against 0.478 with the image missing). It also explains the mechanism with inside-mask and outside-mask false-alarm rates. This corrects a common implicit assumption in missing-modality work, namely that modality dropout makes a model robust to clouds.
**Evidence Anchor**: text: §5.2 "Dropout taught B to ignore zero-filled optical input, not cloud-valued optical input."

### S3: Operationally relevant failure exposed for a missing SAR band
The Las Vegas case shows that a VV+VH model collapses (F1 0.224) when one polarisation is absent at one date. This is a real operational risk: polarisation and orbit coverage gaps occur, and the Sentinel-1B failure in December 2021 caused long coverage gaps. The paper reports this case rather than hiding it behind the city exclusion.
**Evidence Anchor**: text: §4.4 "A model trained with a second polarisation depends on that band being present."

### S4: Limitations are stated where a practitioner would look for them
§6 and §7 explicitly limit the cloud conclusions to opaque, shadowless and composite clouds. §4.3 names the timing/label confound instead of over-reading RQ2. Threshold dependence is surfaced (§4.1, §5.1, §6). This reduces the risk of practitioners misusing the results.
**Evidence Anchor**: text: §6 "Whether this holds for real cloudy acquisitions, thin cloud, haze or cloud shadow was not tested."

---

## Weaknesses *

### W1: Title (and RRD framing) claims environmental monitoring that was not tested
**Problem**: The title names "Urban Development and Environmental Monitoring". Every labelled result, however, is OSCD urban change. OSCD carries no environmental-change labels, and the manuscript concedes that environmental or agricultural pseudo-change cannot be assessed (§3.1, §4.4, §7). WorldCover strata only show *where* false alarms fall; they do not show environmental change detection. An agency reading the title would expect evidence on deforestation, flooding, wetland or agricultural change, and none is provided.
**Evidence Anchor**: text: Title "Multimodal Satellite Change Detection for Urban Development and Environmental Monitoring"
**Why it matters**: The gap between title and evidence is the first thing an operational reader meets, and it overstates the scope of a careful study.
**Suggestion**: The minimal fix is to retitle to the tested scope, for example "Historical optical imagery, polarisation and cloud robustness in SAR–optical urban change detection: a multi-seed study on OSCD", and to state in the abstract that environmental change was not evaluated. A stronger option is to add one environmental change test with Sentinel-1/2 labels, e.g. forest disturbance using the open RADD alerts (Reiche et al., 2021) as reference over a test region, kept clearly separate from the OSCD results.
**Severity**: Major
**Confidence**: 5 — core expertise: environmental change-monitoring products

### W2: No test on real cloudy acquisitions, although the pipeline to obtain them already exists
**Problem**: All §5 conclusions rest on opaque, uniform, shadowless synthetic clouds (Phases 3–5S) or on catalogue cloud pixels pasted into clear scenes (Step 5). Real Sentinel-2 cloud brings thin cirrus, haze, adjacency brightening and cloud shadows that s2cloudless commonly misses, plus scene-level atmospheric changes. These are exactly the under-masking cases that §5.4 shows to be most damaging. The composite results already show detector recall falling from validation to test (§5.7). The Step 8 Earth Engine pipeline already selects real scenes by cloud probability and found real cloudy ones (Chongqing, cloud fraction 0.977), so a real-cloud test is within reach.
**Evidence Anchor**: text: §7 "None of these results measures performance on real cloudy acquisitions."
**Why it matters**: The paper's practical message (mask, zero-fill, fall back on SAR) depends on the mask quality reached on real clouds. Under the paper's own §5.4 asymmetry, real-world under-masking from thin cloud and shadow could erase most of the reported recovery.
**Suggestion**: Add a small real-cloud evaluation. For each test city, use the Step 8 rule to export the Sentinel-2 L1C scene nearest OSCD date 2 (within a few days, so the labels remain valid) with partial cloud, mask it with s2cloudless and with Cloud Score+ (Pasquarella et al., 2023, available in Earth Engine as GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED), and report B, A and SAR-only F1 with and without masking. Report the date offset as a caveat. If label validity is a concern, SEN12MS-CR-TS (Ebel et al., 2022) provides co-registered real cloudy S2/S1 time series for checking mask quality and shadow behaviour, though it has no change labels. If none of this is done, the §8 recommendation should be stated conditionally on synthetic clouds in the conclusion itself as well as in §6.
**Severity**: Major
**Confidence**: 5 — core expertise: Sentinel-2 cloud screening

### W3: The practical recommendation is not compared with the simplest operational policy (switch to SAR-only when cloud is present)
**Problem**: The conclusion recommends a single modality-dropout DS-UNet that "falls back on SAR when it is missing or masked". Under missing input, however, B's fallback (0.431) is not detectably better than a separate SAR-only U-Net (0.418, §4.2). In the cloud experiments (§5.2–5.7), the SAR-only model, which is unaffected by optical clouds, is never shown as a reference line. A two-model policy is common in operational practice: run A on clear scenes and switch to SAR-only on any detected cloud. Using the paper's own numbers, that policy gives 0.418 on any cloudy scene. This exceeds B with no mask at every cloud level (0.329/0.191/0.134) and is close to B with rb95 on composites (0.465–0.483). The paper's own rb99 result (masking almost the whole image gives 0.478 at every level) points the same way: the gain comes from discarding the image.
**Evidence Anchor**: text: §8 "the most practical is a modality-dropout DS-UNet with a threshold selected on validation data"
**Why it matters**: Without this comparator, a practitioner cannot tell whether the dropout model brings any operational benefit beyond the convenience of a single model. The recommendation is the paper's most decision-relevant output.
**Suggestion**: Add the SAR-only U-Net, evaluated on the same cities, as a reference row in the §5.2, §5.3 and §5.7 tables and in Figure 2. Also report a "switch" policy (A when the detector flags no cloud above a set cover, else SAR-only), using the frozen thresholds already available. No retraining is needed. Then rephrase §8 according to the outcome. A single model is only preferable if it beats or matches the switch, or if model-management cost is stated as the reason.
**Severity**: Major
**Confidence**: 4 — core expertise: operational processing chains; the numbers come from the paper's own tables, which come from slightly different city sets and thresholds

### W4: Threshold transfer to new regions is unaddressed, yet the recommendation depends on it
**Problem**: The recommendation requires a validation-selected threshold. B loses 0.086 F1 at the default 0.5, and baseline thresholds range from 0.58 to 0.99. The thresholds were selected on four validation cities. §6 advises selecting thresholds "on held-out data from the target region", but agencies rarely have pixel-level change labels in a new region. How sensitive F1 is to using another region's threshold is not reported. The per-city spread is visible (Valencia 0.04, Las Vegas 0.66), which suggests strong threshold-region interaction.
**Evidence Anchor**: text: §6 "Deployments should select the threshold on held-out data from the target region."
**Why it matters**: An operator cannot apply the recommended configuration in a region without labels, which is the normal case.
**Suggestion**: Report leave-one-validation-city-out threshold variation, or the F1 loss when the threshold selected on validation is applied per test city against each city's own oracle threshold. These can be computed from stored counts. Also discuss label-free options: threshold selection on a small, probability-sampled reference set, as recommended by Olofsson et al. (2014), or test-time adaptation of batch-norm statistics or entropy (Tent; Wang et al., 2021) as a direction for the documented probability shift of the dropout model.
**Severity**: Minor
**Confidence**: 4 — core expertise: deployment of change products to new regions

### W5: Zero-fill is overloaded as both a "missing" token and a valid reflectance value
**Problem**: Missing optical data and masked cloud are both encoded as zeros in the same channels that carry reflectance. Dark surfaces (deep water, terrain and cloud shadow, fresh asphalt in some bands) lie near zero in TOA reflectance scaled to [0, 1]. The paper notes that zero is "not an intrinsic missing-data token" but does not test whether real dark pixels are confused with missingness. It also does not consider the standard alternative from the missing-data literature: an explicit binary mask (missing-indicator) channel.
**Evidence Anchor**: text: §5 "Zero is the fill value used here, not an intrinsic missing-data token."
**Why it matters**: Under real clouds, shadows next to clouds are dark and unmasked by many detectors. A model that has learned "near-zero means ignore optical" could under-detect change in shadowed or water-adjacent areas. This can only be checked with real scenes (W2).
**Suggestion**: Discuss this risk in §6/§7. As future work, or as a cheap ablation, add a mask channel to arm C's spatial-dropout training. Cite missing-modality designs that learn shared and modality-specific features, e.g. ShaSpec (Wang et al., 2023), as an alternative to zero-filling, alongside the already-cited reconstruction approach [12].
**Severity**: Minor
**Confidence**: 3 — adjacent field: missing-data methods; the effect on dark pixels is a hypothesis, not observed

### W6: Pixel-pooled F1 does not map to the accuracy and area statistics agencies report
**Problem**: Results are pixel-pooled F1/precision/recall/false-alarm rate over 9–10 cities. Planning and environmental agencies typically need per-site (object) detection rates, minimum mapping unit behaviour and design-based area estimates with confidence intervals. The paper itself shows that small, thin changes are missed almost entirely (Valencia, F1 0.01–0.04), and §7 lists small-change analysis as not studied.
**Evidence Anchor**: absence: §3.4 and §7 — expected object-level or minimum-mapping-unit accuracy, or a note on area estimation for operational use; checked §3.4, §4.1–4.5, §5, §6, §7
**Why it matters**: A practitioner cannot translate "+0.013 F1" or "−0.044 F1" into a missed-site rate or an area error. This limits the paper's usefulness to the planning audience its title addresses.
**Suggestion**: Add a short paragraph in §6/§7 noting this, and if possible report per-city F1 and object-level recall by change-object size class for A, B and SAR-only, from stored predictions. Cite Olofsson et al. (2014) for the operational accuracy-assessment standard.
**Severity**: Minor
**Confidence**: 4 — core expertise: operational accuracy assessment

### W7: Released artefacts are hard for a practitioner to reuse
**Problem**: Checkpoints are "available from the author on request", so the recommended model cannot be applied without retraining. The repository I inspected has no LICENSE file and no inference script for a new area of interest (only per-step evaluation scripts on OSCD). The run scripts `cd` into the original machine's absolute path (README acknowledges this). DS-UNet must be cloned at a fixed relative path. `requirements.txt` omits s2cloudless and earthengine-api, which are used in Steps 5 and 7–8 and kept in separate environments (`experiments/robust/phase5/detector_env_freeze.txt`, `.venv_gee`).
**Evidence Anchor**: text: Declarations "they are listed with sha256 hashes in `experiments/robust/ARTIFACT_SHA256_*.txt` and are available from the author on request"
**Why it matters**: The integrity work (hashes, audits, pre-registrations) is excellent for verification but does not make the result usable. Without a licence, agencies legally cannot reuse the code. Checkpoints on request do not last beyond the author's availability.
**Suggestion**: Deposit the A, B and SAR-only checkpoints (seed 1, or all 10) and the frozen thresholds on Zenodo with a DOI. Add a LICENSE compatible with DS-UNet's licence. Add one `predict.py` that takes S1 VV and S2 GeoTIFFs plus an optional cloud mask and writes a georeferenced change map. Make paths relative or configurable, and list every dependency, or the separate environment files, in the README.
**Severity**: Minor
**Confidence**: 5 — direct inspection of README.md, requirements.txt and the repository layout

### W8: Submission-readiness placeholders remain in Declarations
**Problem**: The author-contribution, conflict-of-interest and funding statements still contain bracketed author-to-do placeholders.
**Evidence Anchor**: text: Declarations "[author: to be agreed with the mentor."
**Why it matters**: The statements are not final, and the co-author's role is undetermined.
**Suggestion**: Resolve all bracketed placeholders and confirm authorship before submission.
**Severity**: Minor
**Confidence**: 5 — direct reading

### W9: Data vintage and processing level differ from current operational inputs
**Problem**: OSCD uses Sentinel-2 L1C (TOA) pairs from roughly 2015–2018. Operational services today mostly use L2A surface reflectance, harmonised processing baselines (post-2022 offset) and, after 2022, single-satellite Sentinel-1 revisit. The paper does not discuss whether models trained on L1C scaled by 10000 would transfer to L2A or post-baseline-04.00 inputs.
**Evidence Anchor**: text: §3.1 "OSCD [3] provides 24 bitemporal Sentinel-2 L1C image pairs with pixel-level urban change labels."
**Why it matters**: A practitioner adopting the recommended model on current data would face an input-distribution shift that is not mentioned.
**Suggestion**: Add one sentence to §7 on processing level and baseline harmonisation (the paper already uses COPERNICUS/S2_HARMONIZED in Step 8, which is the right choice), and on reduced Sentinel-1 revisit after 2021 for the 3-month mean recipe.
**Severity**: Minor
**Confidence**: 4 — core expertise: Sentinel-2 processing baselines

---

## Detailed Comments *

### Assumption Audit
- **Explicit assumptions**: The RRD premise that the post-change optical image is "often unavailable" is reframed correctly in §5: in practice it is more often present but partly cloudy. That reframing is the paper's most useful operational insight.
- **Implicit assumptions**: (a) Agencies have labelled validation data in the target region for threshold selection (W4). (b) Zero is a safe missing-data token for reflectance (W5). (c) Pixel F1 is the relevant operational metric (W6). (d) A single model is preferable to a model switch (W3). None of these is stated, and each affects deployment.
- **Paradigmatic assumptions**: The benchmark paradigm of a fixed test set, training-seed variance and pooled pixel F1 is well executed. Operational EO, however, treats uncertainty as map-level accuracy from a probability sample of reference sites. The two views are complementary, and the paper would gain from naming the gap (W6).

### Cross-Disciplinary Connections
- **Parallel research**: Operational SAR-based alerting (RADD; Reiche et al., 2021) already runs on Sentinel-1 alone with optical confirmation only when available. This is the "SAR-first, optical-when-clear" design that W3 asks the authors to benchmark.
- **Borrowing opportunities**: Missing-indicator channels and shared/specific feature learning (ShaSpec), test-time adaptation for probability shift (Tent), and quality scores rather than binary masks (Cloud Score+), which would allow a soft or recall-biased masking policy consistent with §5.4.
- **Methodological borrowing**: Design-based accuracy assessment and area estimation (Olofsson et al., 2014) for any claim aimed at planners.

### Practical Impact
- **Real-world application**: The two most transferable messages are "mask cloud aggressively" and "partly cloudy is worse than missing". Both are useful today. The model recommendation itself needs W2 and W3 before an operator should adopt it.
- **Implementation feasibility**: Training on one laptop GPU is a plus for small agencies. Checkpoints on request and the missing licence are the main barriers (W7).
- **Stakeholders**: Planners need site-level detection and small-change behaviour (Valencia-type thin changes are missed). Environmental agencies are not served by the current evidence (W1).

### Broader Implications
- **Ethical dimensions**: No concerns for public imagery. The AI-use disclosure is candid that the same assistant drafted text and ran integrity checks. I report this as transparent rather than as a defect. I found no text in the manuscript addressed to reviewers.
- **Social impact**: The test cities are mostly large, data-rich cities. Smaller or Global South settlements with informal growth and persistent cloud (e.g. humid tropics) are where SAR fallback matters most, and they are underrepresented. Chongqing is excluded from Step 8 for cloud.
- **Future directions**: A real-cloud test set built near OSCD date 2. A dated-label design that breaks the RQ2 confound. An environmental-change companion test.

### Title & Abstract
The abstract is careful and hedged. The title is not (W1).

### Discussion / Conclusion
§6 states the synthetic-cloud limit correctly. §8's recommendation sentence should carry the same qualifier and the SAR-only comparator (W3).

---

## Cross-Disciplinary Reading Recommendations
All were verified to exist by web search during this review.
- Olofsson, P., Foody, G. M., Herold, M., Stehman, S. V., Woodcock, C. E., & Wulder, M. A. (2014). Good practices for estimating area and assessing accuracy of land change. *Remote Sensing of Environment, 148*, 42–57. https://doi.org/10.1016/j.rse.2014.02.015 — the operational accuracy standard (W4, W6).
- Pasquarella, V. J., Brown, C. F., Czerwinski, W., & Rucklidge, W. J. (2023). Comprehensive quality assessment of optical satellite imagery using weakly supervised video learning. *CVPR Workshops (EarthVision)* — Cloud Score+, a graded alternative to s2cloudless that is available in Earth Engine (W2).
- Ebel, P., Xu, Y., Schmitt, M., & Zhu, X. X. (2022). SEN12MS-CR-TS: A remote-sensing data set for multimodal multitemporal cloud removal. *IEEE TGRS*. https://doi.org/10.1109/TGRS.2022.3146246 — real cloudy S1/S2 time series (W2).
- Reiche, J., et al. (2021). Forest disturbance alerts for the Congo Basin using Sentinel-1. *Environmental Research Letters, 16*, 024005. https://doi.org/10.1088/1748-9326/abd0a8 — operational SAR-first environmental change monitoring (W1, W3).
- Wang, D., Shelhamer, E., Liu, S., Olshausen, B., & Darrell, T. (2021). Tent: Fully test-time adaptation by entropy minimization. *ICLR 2021* — for the probability and threshold shift (W4).
- Wang, H., Chen, Y., Ma, C., Avery, J., Hull, L., & Carneiro, G. (2023). Multi-modal learning with missing modality via shared-specific feature modelling. *CVPR 2023*, 15878–15887 — alternative to zero-fill (W5).

---

## Questions for Authors *
1. With the SAR-only U-Net as a reference line in §5.2–5.7, is there any cloud level, mask or composite condition where B beats a policy of "switch to SAR-only on detected cloud"? If not, what is the operational argument for the single dropout model?
2. Can you retrieve, for each test city, a real partly cloudy Sentinel-2 scene within a few days of OSCD date 2 with your Step 8 pipeline, and report masked and unmasked results on it?
3. How much does test F1 drop when the threshold is chosen on a different set of cities (e.g. leave-one-validation-city-out), as a proxy for deployment without local labels?
4. Will you retitle to the tested urban scope, or add an environmental-change test?

---

## Minor Issues

### Language / Grammar
- Abstract: "Following a research requirement document with five questions" is project-internal framing. External readers do not know the RRD, so state the questions' motivation instead.

### Figures and Tables
- Figure 2 and the §5 tables would gain a horizontal SAR-only reference line (see W3).
- §5.7 tables: report the per-city spread in detector cloud recall, since under-masking drives the losses.

### Layout
- Figure paths mix `../figures/paper/` and `../experiments/robust/...`. Consolidate them for the submission build.

---

## Criterion-Bound Judgements *

Calibration status: `NOT_CALIBRATED`

| Dimension | Criterion source | Judgement | Evidence anchor(s) | Rationale | Uncertainty / scope limit | Decision bearing? |
|---|---|---|---|---|---|---|
| Originality | quality_rubrics.md; Config Card #4 | MEETS | text: §5.2 "Dropout taught B to ignore zero-filled optical input, not cloud-valued optical input." | Missing-vs-cloudy separation and mask-error asymmetry are new in the CD setting | Novelty audit is R2's remit | no — not my core remit |
| Methodological Rigor | quality_rubrics.md | NOT_ASSESSED | — | R1's remit | statistical detail is my stated blind spot | no |
| Evidence Sufficiency | Config Card #4 focus (1) | PARTLY_MEETS | text: §7 "None of these results measures performance on real cloudy acquisitions." | Practical recommendation lacks real-cloud and SAR-only-switch evidence (W2, W3) | repairable with one experiment and one comparator | yes — the recommendation depends on it |
| Argument Coherence | quality_rubrics.md | NOT_ASSESSED | — | DA's remit | — | no |
| Writing Quality | quality_rubrics.md | MEETS | text: §6 "Whether this holds for real cloudy acquisitions, thin cloud, haze or cloud shadow was not tested." | Clear and hedged; placeholders remain (W8) | — | no |
| Literature Integration | quality_rubrics.md | PARTLY_MEETS | absence: §2 — expected operational SAR-first monitoring and accuracy-assessment literature; checked §2, §6, §7 | Adjacent operational literature missing; ML side covered | R2 owns completeness | no |
| Significance & Impact | Config Card #4 focus (2) | PARTLY_MEETS | text: Title "Multimodal Satellite Change Detection for Urban Development and Environmental Monitoring" | Urban significance is real; environmental claim is unsupported (W1); limited reuse (W7) | — | yes — title/scope mismatch |

**Recommendation rationale**: Major Revision. The unresolved decision-bearing items are the title/scope mismatch (W1, repairable by retitling), the absence of any real-cloud evaluation (W2, repairable with the existing Earth Engine pipeline), and the missing SAR-only/switch comparator for the closing recommendation (W3, repairable from stored results without retraining). None is fatal. The study's core measurement results stand regardless, but the operational recommendation should not be published in its current form without W2 and W3.
