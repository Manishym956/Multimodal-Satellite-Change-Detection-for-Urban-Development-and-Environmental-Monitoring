# Peer Review Report

## Manuscript Information
- **Title**: Multimodal Satellite Change Detection for Urban Development and Environmental Monitoring
- **Manuscript ID**: FINAL_RESEARCH_PAPER.md (SHA-256 7220d00b9c4e2ec08803faceeacc5437d4876243edf89ea91565c9514aa1bc8f, verified before review)
- **Review Date**: 2026-11-02
- **Review Round**: Round 1 (ARS Stage 3, full mode)

---

## Reviewer Information

### Reviewer Role *
Peer Reviewer 2 (Domain)

### Reviewer Identity *
SAR–optical remote-sensing researcher working on Sentinel-1/Sentinel-2 urban change detection and data fusion (Reviewer Configuration Card #3).

### Review Focus *
Whether OSCD's label definition and the inherited Sentinel-1 preprocessing (3-month mean backscatter, VV/VH, orbit handling, co-registration) bias the answers to RQ1–RQ3 and RQ5. Whether the literature covers SAR–optical and heterogeneous change detection, missing-modality learning, cloud masking and temporal-gap work. Whether the comparisons with Song et al. [2] and Hafner et al. [1, 12] are fair and accurate, and whether the conclusions on historical optical imagery and polarisation are physically plausible and properly scoped. Statistics are outside my remit and I do not assess them.

**Verification performed.** I read the manuscript in full. I checked the Sentinel-1 export code (`download_gee.py`) and the scene inventory (`experiments/robust/feasibility/inventory.json`). I also read the full text of Song et al. [2] (MDPI PDF, §2.2–2.3, §3.1.1, Table 2, §5.1, §5.4/Table 4) and the arXiv HTML of Hafner & Ban [12] (results table), and I looked up every reference I suggest below. I found no text in the manuscript that addresses reviewers or tries to steer the review. The Declarations still contain unresolved author placeholders (listed under Minor Issues).

---

## Overall Assessment *

### Recommendation *
- [ ] **Accept**
- [ ] **Minor Revision**
- [x] **Major Revision** — substantial revisions needed, re-review required after revision
- [ ] **Reject**

### Confidence Score *
4. Mostly within my expertise (Sentinel-1/2 preprocessing, SAR–optical change detection, OSCD). I did not assess the statistical machinery.

Confidence is an uncertainty/scope disclosure only; it never changes consensus counts, severity, decision bearing, or arbitration.

### Summary Assessment *
The manuscript is a multi-seed measurement study on OSCD enriched with Sentinel-1. It uses the official DS-UNet and its U-Net to answer five questions: whether historical Sentinel-2 imagery helps SAR change detection without a post-change optical image (RQ1), the effect of the image's acquisition timing (RQ2), the value of VH (RQ3), the comparison of dual-stream fusion with simpler baselines (RQ4), and the conditions in which the historical image helps (RQ5). A cloud-robustness study is added. On data provenance the work is careful: the Earth Engine VV export matches the released rasters pixel for pixel, there is a same-date control for the historical Sentinel-2 chain, and the OSCD label-timing confound is stated openly. Most RQ answers are already scoped as nulls.

From a domain standpoint, four problems weaken the core claims. First, §4.3 frames the comparison with Song et al. [2] as a replication. In fact the historical image does opposite jobs in the two designs: in Song et al. it is a land-cover prior for the target date, while here it fills the pre-change slot of a bitemporal model. Second, the paper overlooks the heterogeneous (optical-to-SAR) change detection literature, which is the established framing of "post-change optical missing". Third, the paper does not disclose a preprocessing asymmetry: date-1 Sentinel-1 composites average about half as many scenes as date-2 composites, and for some arid cities date-1 VH comes from one scene or none. This gives a physical alternative explanation for the RQ3 VH result. Fourth, the RQ5 land-cover effect is attributed to the historical image, but it is measured across two different architectures.

All four can be repaired without new training. Hence Major Revision.

---

## Strengths *

### S1: Careful provenance control of the inherited and new satellite inputs
The new Earth Engine Sentinel-1 export reproduces the released DS-UNet VV rasters exactly ("in all 102 files"). The historical Sentinel-2 chain has a same-date control (`gee_t1`, Δ +0.000 F1) and a geometric check (median shift 0.27 px, every city below 1 px). This rules out the most common confound in studies that rebuild satellite inputs: a different processing chain rather than different information. Few OSCD+S1 papers do this.
**Evidence Anchor**: text: §3.1 "The exported VV equals the released DS-UNet VV raster pixel for pixel in all 102 files"

### S2: The OSCD label-timing confound in RQ2 is identified and stated
The paper explains that, under OSCD's fixed date-1/date-2 labels, a historical image acquired after date 1 can already show part of the labelled change. It therefore answers RQ2 as "acquisition timing", not image age. This is correct reasoning about label semantics, and many temporal-gap studies omit it.
**Evidence Anchor**: text: §4.3 "An image taken 1–6 months before date 2 lies after date 1 and may already show part of the labelled change"

### S3: Prior work is quoted accurately where I could check it
I checked the cited numbers against the sources. Song et al. [2] §5.4/Table 4 does report F1 83.99% (1 month) falling to 78.31% (12 months) with no repeated runs, and five seeds with paired t-tests for the main comparison (their Table 2). Hafner & Ban [12] does report 0.327 (proposed) against 0.233 (zero-fill DS U-Net) and 0.348 (SAR-only) on missing-optical samples.
**Evidence Anchor**: text: §2 "on samples with missing optical data their model beats zero-fill (F1 0.327 against 0.233) but falls slightly short of a SAR-only model (0.348)"

### S4: The missing and partly cloudy cases are separated in a physically meaningful way
§5.2–5.4 show that a dropout-trained model reads bright cloud-valued pixels as change, and that zero-filling under a mask turns cloud back into the missing-data case. The paper also measures the asymmetric cost of mask errors (over-masking almost free, under-masking expensive). This is a useful, physically interpretable result for operational Sentinel-2 cloud handling. Its limits (opaque, shadowless clouds and composites) are stated plainly in §5.7 and §7.
**Evidence Anchor**: text: §5.2 "Dropout taught B to ignore zero-filled optical input, not cloud-valued optical input."

### S5: RQ3 and RQ5 are scoped correctly with respect to missing pseudo-change labels
The paper states that OSCD has no labels for agricultural or seasonal change and that WorldCover 2020 post-dates OSCD date 2. It then treats land cover as a stratification, not a cause, and calls the VV+VH false-alarm reduction a shift in operating point rather than discrimination. This is the right reading of the evidence.
**Evidence Anchor**: text: §4.4 "This is a shift of operating point, not evidence that VH separates urban development from environmental or agricultural pseudo-change."

---

## Weaknesses *

### W1: The comparison with Song et al. [2] is framed as a replication, but the historical image plays the opposite role
**Problem**: In Song et al. [2] the historical Sentinel-2 image is a semantic prior on land-cover state. Their §2.3 defines it as "a cloud-free Sentinel-2 observation acquired prior to T2 … describing pre-change land-cover conditions". In the main experiment it is a 2019 annual L2A composite. Their second stage uses it, through asymmetric cross-attention, to decide whether a SAR-screened candidate is bare land. In such a design, an image closer to T2 is expected to help, because it shows a land-cover state closer to the target.

In this manuscript the image goes into the pre-change slot of a bitemporal change model. There, an image closer to date 2 is expected to hurt, because it hides change (as §4.3 itself argues). The two designs therefore predict opposite signs for the 1-month effect. "Not reproduced" is the wrong frame for that.

Song et al.'s gap experiment has the same timing structure (gaps measured from T2, so a 1-month image probably post-dates their T1), and the manuscript does not say so. §4.3 also contradicts §2, which says a different outcome "does not contradict their result". Finally, §2 says the authors read only "§2, §5.1, §5.4 and Table 4" of Song et al. A full reading is needed before a published comparison. For example, the claim that VH was chosen "from backscatter physics rather than from a model ablation" leaves out the class-wise VV/VH backscatter distribution analysis in their §3.1.1/Fig. 4.
**Evidence Anchor**: text: §4.3 "the decline with age reported by Song et al. [2]" … "is not reproduced"
**Why it matters**: The RQ2 contribution (ii) is presented as a test of Song et al.'s temporal-gap finding. As framed, readers will take it as evidence against that finding, although the two experiments measure different mechanisms.
**Suggestion**: Read [2] in full and remove the "sections we read" sentence. In §2 and §4.3, say explicitly that the historical image is a target-state semantic prior in [2] and a pre-change baseline here. Explain that the expected sign of the timing effect differs, and that [2]'s gap images are also referenced to T2. Replace "is not reproduced" with "was not observed in a design where the image plays a different role". Mention [2]'s VV/VH distribution analysis (§3.1.1) when describing their polarisation choice.
**Severity**: Major
**Confidence**: 4 — core expertise in SAR–optical change detection; I read the source paper

### W2: The heterogeneous (optical-to-SAR) change detection literature is missing, and §2's summary of the field is inaccurate
**Problem**: The RQ1 setting is pre-change optical plus post-change SAR, without post-change optical. This is close to the established multimodal or heterogeneous change detection problem, where the pre-event image is optical and the post-event image is SAR because clouds or urgency rule out optical. That literature dates back at least to coupled and translation networks, and it now has a global benchmark that uses exactly this pre-optical/post-SAR design. §2 states that the prior fusion work assumes both modalities at both dates. That holds for the papers cited, but as a description of the field it is wrong, and it overstates the gap the paper fills.

Verified references:
- J. Liu, M. Gong, A. K. Qin, P. Zhang, "A deep convolutional coupling network for change detection based on heterogeneous optical and radar images," IEEE TNNLS 29(3):545–559, 2018, https://doi.org/10.1109/TNNLS.2016.2636227.
- L. T. Luppino et al., "Deep image translation with an affinity-based change prior for unsupervised multimodal change detection," IEEE TGRS, 2021/2022, https://doi.org/10.1109/TGRS.2021.3056196.
- H. Chen et al., "BRIGHT: a globally distributed multimodal building damage assessment dataset with very-high-resolution for all-weather disaster response," Earth Syst. Sci. Data 17:6217–6253, 2025, https://doi.org/10.5194/essd-17-6217-2025 (pre-event optical, post-event SAR).
**Evidence Anchor**: text: §2 "All of these assume both modalities at the dates compared, or address a different task."
**Why it matters**: Reviewers in this field will see the missing-post-optical framing as a known problem. The paper's contribution is better stated as a measurement of whether pre-change optical imagery adds to bitemporal SAR in a fusion network. That is still useful, but narrower.
**Suggestion**: Add a "Heterogeneous / multimodal change detection" paragraph to §2 with the references above. State how the RQ1 setting differs: SAR is available at both dates, so the optical image is auxiliary rather than the only pre-change source. Rephrase the gap claim in §1 and §2 accordingly.
**Severity**: Major
**Confidence**: 4 — core expertise; all references verified online

### W3: Undisclosed asymmetry in the Sentinel-1 composites between dates, with consequences for RQ3 (and RQ5)
**Problem**: The 3-month mean composites are not equally well averaged at both dates. Most OSCD date-1 acquisitions fall in 2015–2016, before Sentinel-1B was operational. In the project's own inventory (`experiments/robust/feasibility/inventory.json`), date-1 composites typically average about half as many scenes as date-2 composites: Montpellier 7 vs 14 and 6 vs 15, Norcia 8 vs 15 and 5 vs 14, Valencia 7 vs 13–15, Saclay_w 6 vs 14.

For VH it is worse. Dubai's evaluated orbit 130 has date-1 VH from a single scene against VV from 3; Abu Dhabi has 1 VH scene; Chongqing has 3–5. Las Vegas has none, as the paper notes. Residual speckle in a temporal mean falls roughly with the number of scenes, so the date-1 image is systematically noisier than the date-2 image. Bitemporal comparisons read that difference as change. The effect is largest in VH, which is already lower in backscatter and closer to the noise floor, and in arid, low-backscatter scenes, where Fig. 1 shows SAR false alarms concentrating.

This is a physical alternative explanation for the RQ3 result (VH and VV+VH lowering SAR-only F1). The paper does not discuss it. Instead, §6 attributes the result to VH's "lower backscatter and range".
**Evidence Anchor**: absence: §3.1 (Sentinel-1 VV and VH) and §7 (SAR input) — expected the number of scenes per composite per date and polarisation, and a discussion of unequal speckle between dates; checked §3.1, §4.4, §6 Polarisation, §7, Appendices C–D
**Why it matters**: RQ3 is answered as a property of adding VH under "3-month mean backscatter". The scope statement does not cover the fact that VH quality differs between dates and across test cities. The SAR-only false-alarm patterns used for RQ5 and Fig. 1 may partly reflect this asymmetry rather than land-cover physics.
**Suggestion**: Report scene counts per city, orbit, date and polarisation (the inventory already has them). Add a per-city breakdown of the VV+VH − VV difference against date-1 VH scene count. If possible, add a matched control: subsample date-2 scenes to the date-1 count, or drop test cities with fewer than k date-1 VH scenes. All of these are evaluation-side analyses or re-exports, not retraining. Tone down the mechanism in §6 ("with its lower backscatter and range, makes the model more conservative") to a hypothesis next to this alternative.
**Severity**: Major
**Confidence**: 4 — core expertise in Sentinel-1 preprocessing; counts read from the project's own inventory

### W4: The RQ5 land-cover effect is attributed to the historical image but is measured across architectures
**Problem**: The RQ5 table compares B (dual-stream, modality-dropout, its own threshold) given OSCD's date-1 image against the SAR-only U-Net (single-stream, full-input training, a different threshold). Any difference mixes the historical image with architecture, training mixture and operating point. Yet the abstract, §4.5 and §8 attribute the shrubland reduction to "the historical optical image". The paper already uses the clean within-model contrast for RQ1 (B with the image present minus absent, Δ −0.010). The same contrast is needed here.
**Evidence Anchor**: text: §4.5 "False-alarm rate by WorldCover class, B with OSCD's date-1 image minus the SAR-only U-Net"
**Why it matters**: RQ5 is a headline answer (abstract and conclusion). Its only positive finding may come from the architecture rather than the optical image.
**Suggestion**: Repeat the stratified false-alarm analysis for B@`no_o2` − B@`no_opt` (and async with vs. without the image). These are evaluations of existing checkpoints. Report it as the primary RQ5 contrast, or reword the claim as "B without post-change optical vs. a SAR-only U-Net".
**Severity**: Major
**Confidence**: 4 — the design is fully described in §3.3 and §4.5

### W5: RQ1/RQ5 conclusions are not scoped to temporally averaged SAR whose windows extend beyond the labelled interval
**Problem**: The SAR inputs are 3-month means over [date 1 − 3 months, date 1] and [date 2, date 2 + 3 months] (confirmed in `download_gee.py`). Temporal averaging suppresses the short-lived soil-moisture, rainfall and harvest backscatter changes that motivate historical optical guidance in [2]. Song et al. use a single filtered target-date image against an annual-mean baseline (their §2.3, §5.1). With averaging, less of that pseudo-change is left for the optical image to remove, which biases RQ1 and RQ5 toward a null. In addition, the date-2 SAR window extends up to 3 months past the labelled date 2, so construction completed in that period appears in SAR but not in the labels. That penalises SAR-based arms (RQ4: SAR-only vs. optical-only) through unlabelled true change. §4.4 and §6 scope RQ3 to 3-month means, but the abstract and the RQ1/RQ5 answers do not.
**Evidence Anchor**: text: §3.1 "the mean of the 3 months before OSCD date 1 and after date 2"
**Why it matters**: Without this scoping, readers may generalise the RQ1 null to single-date SAR monitoring, which is the operational use case the Introduction motivates.
**Suggestion**: Add "with 3-month mean Sentinel-1 composites" to the RQ1 and RQ5 answers, the abstract and §8. In §7, note that the windows extend outside the labelled interval and what that implies for SAR false alarms. A single-scene SAR variant would be the stronger test, but scoping is sufficient here.
**Severity**: Minor
**Confidence**: 4 — core expertise; preprocessing confirmed in code

### W6: The RQ2 timing analysis ignores seasonal phase and train/test image shift
**Problem**: The primary RQ2 test is a linear slope of F1 against gap. Gaps of 6 and 18 months put the historical image in the opposite season from date 2, while 12 months is the same season. Phenology and illumination (sun elevation, snow, crop state) therefore vary non-monotonically with gap. The descriptive pattern (B: 0.389, 0.422, 0.409, 0.417, 0.412) is consistent with this, though not established. Also, every model was trained only with OSCD's date-1 image in that slot, so every non-date-1 image is a test-time input shift. The paper notes this only in §7.
**Evidence Anchor**: table: §4.3 Step 8 table — B dropout 0.389 (1 month), 0.422 (3), 0.409 (6), 0.417 (12), 0.412 (18)
**Why it matters**: "No detectable effect of acquisition timing" may hide a seasonal effect that a linear slope cannot detect.
**Suggestion**: Report each image's day-of-year offset from OSCD date 1 and date 2. Add a descriptive same-season vs. opposite-season contrast, or state explicitly that the slope test is blind to seasonal phase.
**Severity**: Minor
**Confidence**: 3 — domain reasoning; effect size not testable from the paper

### W7: Orbit handling and SAR–optical geometry are not examined
**Problem**: Training samples a relative orbit at random. Evaluation uses "the first listed orbit", which mixes ascending and descending geometries across cities. In urban and hilly test cities (Rio, Norcia), layover, foreshortening and building orientation relative to the look direction change the backscatter strongly, so results depend on an arbitrary orbit choice. SAR–optical co-registration is inherited from [1] without a check. The paper checks optical–optical alignment only, and SAR is resampled by cubic interpolation of dB values.
**Evidence Anchor**: text: §3.1 "training samples an orbit at random and evaluation uses the first listed orbit"
**Why it matters**: It adds unquantified geometry-dependent variance to every SAR arm, and orbit direction is a known driver of urban backscatter.
**Suggestion**: State the pass direction of each evaluated orbit. Add a sensitivity check that evaluates all available orbits per test city (or averages their predictions). Report a SAR–optical offset estimate for a few cities, for example by cross-correlating the edge or gradient maps of VV and Sentinel-2 B8.
**Severity**: Minor
**Confidence**: 4 — core expertise

### W8: Composite construction details that affect low-backscatter areas are undisclosed
**Problem**: Pixels below the range floor are masked before the temporal mean, so the mean is censored. It is also taken in dB, which is a geometric mean in linear power. This biases means upward in dark targets (water, smooth bare soil, desert sand in VH), which are exactly the classes and cities where SAR false alarms concentrate. The export script also uses a non-standard date-1 window for Mumbai (−12 to +5 months), so its "pre-change" SAR overlaps the change interval of a training city. The paper does not mention this.
**Evidence Anchor**: text: §3.1 "low values masked, unit-scaled and clamped"
**Why it matters**: These choices are inherited and probably small, but they are not reproducible from the text, and they bear on the arid-city false alarms the paper discusses.
**Suggestion**: State in §3.1 that the masking is applied per scene before averaging, that averaging is in dB, and that Mumbai has the non-standard window (inherited from [1]).
**Severity**: Minor
**Confidence**: 4 — confirmed in `download_gee.py`

### W9: Missing references on SAR urban change, multimodal under-use of optical data, missing-sensor EO learning and real cloudy benchmarks
**Problem**: Several directly relevant and verified works are not cited:
(a) S. Hafner, Y. Ban, A. Nascetti, "Semi-supervised urban change detection using multi-modal Sentinel-1 SAR and Sentinel-2 MSI data," Remote Sensing 15(21):5135, 2023, https://doi.org/10.3390/rs15215135. This is the same group's S1/S2 urban change detection follow-up.
(b) S. Hafner, Y. Ban, A. Nascetti, "Investigating imbalances between SAR and optical utilization for multi-modal urban mapping," JURSE 2023, arXiv:2304.05080, https://arxiv.org/abs/2304.05080. It reports that SAR–optical networks under-use the optical branch, which directly supports the second explanation in §6.
(c) Y. Ban, O. A. Yousif, "Multitemporal spaceborne SAR data for urban change detection in China," IEEE JSTARS 5(4):1087–1094, 2012, https://doi.org/10.1109/JSTARS.2012.2201135. A foundation for SAR-only urban change detection.
(d) F. Mena, D. Arenas, A. Dengel, "Increasing the robustness of model predictions to missing sensors in Earth observation," MACLEAN@ECML-PKDD 2024, arXiv:2407.15512, https://arxiv.org/abs/2407.15512. Sensor dropout in EO, which is closer than gesture recognition [11].
(e) C. Aybar et al., "CloudSEN12, a global dataset for semantic understanding of cloud and cloud shadow in Sentinel-2," Scientific Data 9:782, 2022, https://doi.org/10.1038/s41597-022-01878-2, and P. Ebel et al., "SEN12MS-CR-TS: a remote-sensing data set for multimodal multitemporal cloud removal," IEEE TGRS 60:5222414, 2022, https://doi.org/10.1109/TGRS.2022.3146246. Both offer real cloudy and cloud-shadow Sentinel-2 data, with co-located Sentinel-1, that could close the "real cloudy acquisitions" gap stated in §7 and §8.
**Evidence Anchor**: absence: §2 Related Work and §6 Discussion — expected citations to S1/S2 urban change follow-ups, optical under-utilisation, EO sensor dropout and real-cloud S1/S2 datasets; checked §1, §2, §6, §7, §8, References
**Why it matters**: These works would sharpen the positioning (b, d), give the null in §6 a published mechanism (b), and point to the obvious next real-cloud test (e).
**Suggestion**: Cite (a)–(e) in §2 and §6, and use (e) in the future-work sentence of §8.
**Severity**: Minor
**Confidence**: 4 — all references verified online

### W10: The title and RRD framing overstate the scope or read as project-internal
**Problem**: The title promises "Environmental Monitoring", but OSCD labels only urban change, and §4.4 and §7 state that environmental or agricultural change cannot be evaluated. The abstract and §1 frame the study around a "research requirement document" whose questions are reproduced "word for word". That is project-management language, not a scientific motivation.
**Evidence Anchor**: text: Title "Multimodal Satellite Change Detection for Urban Development and Environmental Monitoring"
**Why it matters**: The title misleads about scope, and the framing distracts from the scientific question.
**Suggestion**: Retitle to reflect the measurement study, for example "Does historical Sentinel-2 imagery help Sentinel-1 urban change detection when post-change optical data are missing? A multi-seed study on OSCD". Introduce the RQs as scientific questions without referring to the RRD.
**Severity**: Minor
**Confidence**: 4 — domain and venue conventions

---

## Detailed Comments *

### Title & Abstract
- The abstract is accurate and appropriately hedged, but see W10 (title) and W5 (add "3-month mean SAR" to the RQ1 sentence). The shrubland claim in the abstract depends on W4.

### Introduction
- The Introduction cites [2] as showing that historical images "can stand in for the missing post-change image". In [2] the image is a land-cover prior for stage-2 refinement, not a stand-in for the post-change image in a bitemporal comparison (W1).

### Literature Review / Theoretical Framework
- **Coverage**: The main gaps are heterogeneous change detection (W2) and the items in W9.
- **Integration**: The comparisons with [2] and [12] go beyond listing and are critical, and the cited numbers are correct. The problem is the framing of [2] (W1).
- **Gap argument**: It is overstated until heterogeneous change detection is acknowledged.
- **Framework**: The implicit framework is "optical as semantic context for SAR change". The paper does not state which role the optical image plays (pre-change baseline vs. target-state prior), and that role decides the expected direction of the RQ2 effect.

### Methodology / Research Design (domain aspects only)
- The Sentinel-1 composite asymmetry (W3), censored dB means and the Mumbai window (W8), orbit choice (W7), and SAR windows outside the labelled interval (W5).
- The Sentinel-2 L1C / S2_HARMONIZED choice matches OSCD's L1C and suits 2015–2018 scenes, because the post-2022 baseline offset does not apply to them. The same-date control in S1 confirms this.

### Results / Findings
- RQ4 agrees with [1] and is physically plausible: decision-level fusion stops VV, with its lower SNR, from swamping optical features when the two are stacked at the input.
- The RQ1 null is physically plausible for 3-month-mean VV, where new buildings give a strong double-bounce increase. It is also consistent with [12], where zero-filled DS U-Net fell below SAR-only on missing-optical samples (0.233 vs 0.348), and with (b) in W9. The paper should make that link explicit.
- RQ3: W3. RQ5: W4.

### Discussion
- §6's two explanations for RQ1 are reasonable. The second (fusion does not use optical context without a post-change image) has published support (W9b).
- The cloud discussion is properly limited to synthetic and composite clouds.

### Conclusion
- The "most practical configuration" statement is suitably scoped to OSCD and the configurations tested. Add the SAR-compositing qualifier (W5).

### References
- The references are accurate for [2] and [12] as checked. [17] lacks a DOI or URL. [20]–[24] are cited only in appendices.

---

## Questions for Authors *

1. How many Sentinel-1 scenes enter each composite, by test city, evaluated orbit, date and polarisation? Does the SAR-only VV+VH − VV difference track the number of date-1 VH scenes (e.g., Dubai, Chongqing vs. Montpellier, Milano)?
2. If the RQ5 stratified false-alarm analysis is redone as B@`no_o2` − B@`no_opt` (same model, image present vs. absent), does the shrubland effect remain?
3. For each primary city, what are the acquisition dates of the Song-style gap images relative to OSCD date 1, date 2 and the SAR windows? Does F1 vary by season (same vs. opposite season to date 2) rather than linearly with gap?
4. Are the evaluated "first listed" orbits ascending or descending, and how much do test F1 values change if the other available orbit is used?

---

## Minor Issues

### Language / Grammar
- §2: "The sections we read (§2, §5.1, §5.4 and Table 4)" should be removed after a full reading (W1).
- §6: "with its lower backscatter and range, makes the model more conservative" is phrased as an established mechanism; it should be a hypothesis (W3).

### Citation Format
- [17] Bouthillier et al. has no DOI or URL. [15] is software: give the version DOI or commit if available.

### Figures and Tables
- Fig. 1 would benefit from a SAR VV date-1/date-2 panel for Las Vegas, to show whether the desert false alarms follow backscatter noise.

### Layout
- Declarations still contain author placeholders ("[author: to be agreed with the mentor", "[author: confirm.]" twice). These must be resolved before submission.

---

## Criterion-Bound Judgements *

Calibration status: `NOT_CALIBRATED`

| Dimension | Criterion source | Judgement | Evidence anchor(s) | Rationale | Uncertainty / scope limit | Decision bearing? |
|---|---|---|---|---|---|---|
| Originality | Card #3 domain remit | PARTLY_MEETS | text: §2 "All of these assume both modalities at the dates compared" | A controlled measurement on OSCD+S1 is new. The gap claim ignores heterogeneous change detection (W2). | Novelty judged against literature I could verify | yes — positioning must be fixed |
| Methodological Rigor | Card #3 (domain aspects only) | PARTLY_MEETS | absence: §3.1 scene counts (W3) | Strong provenance controls (S1), but undisclosed SAR composite asymmetry and orbit handling (W3, W7, W8) | Statistics not assessed | yes — W3 |
| Evidence Sufficiency | Card #3 | PARTLY_MEETS | text: §4.5 RQ5 contrast (W4) | RQ5 attribution is cross-architecture; RQ3 has a plausible alternative cause | Fixes are evaluation-only | yes — W4, W3 |
| Argument Coherence | Card #3 | PARTLY_MEETS | text: §4.3 "is not reproduced" (W1) | §2 and §4.3 contradict each other on what the result means for [2] | none identified | yes — W1 |
| Writing Quality | Card #3 | MEETS | text: Title (W10) | Clear and hedged; title and RRD framing are minor issues | none identified | no |
| Literature Integration | Card #3 focus (2) | PARTLY_MEETS | absence: §2 (W9) | Accurate where cited; heterogeneous change detection and several directly relevant works missing | none identified | yes — W2 |
| Significance & Impact | Card #3 | MEETS | text: §5.2 (S4) | Useful nulls and an operationally relevant missing-vs-cloudy distinction for Sentinel-1/2 users | Single benchmark | no |

**Recommendation rationale.** Major Revision. Four decision-bearing issues remain: W1 (framing of the Song et al. comparison), W2 (heterogeneous change detection positioning), W3 (Sentinel-1 composite asymmetry as an alternative explanation for RQ3) and W4 (RQ5 attribution). Each can be repaired with rewriting and evaluation-only analyses on existing checkpoints and exports. None invalidates the core measurements, and the provenance controls and scoping elsewhere are strong.

Sources consulted: Song et al. full text (https://mdpi-res.com/d_attachment/remotesensing/remotesensing-18-01103/article_deploy/remotesensing-18-01103.pdf); Hafner & Ban 2023 (https://arxiv.org/html/2306.00640v1); https://arxiv.org/abs/2304.05080; https://arxiv.org/abs/2407.15512; https://doi.org/10.1109/TNNLS.2016.2636227 (dblp https://dblp.uni-trier.de/rec/journals/tnn/LiuGQZ18.html); https://arxiv.org/abs/2001.04271; https://essd.copernicus.org/preprints/essd-2025-269; https://doaj.org/article/c757d75f184c482699d4cc0e66ae5d78; https://arxiv.org/abs/2201.09613; CloudSEN12 (https://doi.org/10.1038/s41597-022-01878-2); Ban & Yousif 2012 (https://doi.org/10.1109/JSTARS.2012.2201135).
