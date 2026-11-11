# Peer Review Report

## Manuscript Information
- **Title**: Multimodal Satellite Change Detection for Urban Development and Environmental Monitoring
- **Manuscript ID**: FINAL_RESEARCH_PAPER.md (SHA-256 7220d00b9c4e2ec08803faceeacc5437d4876243edf89ea91565c9514aa1bc8f, verified)
- **Review Date**: 2026-11-02
- **Review Round**: Round 1 (Stage 3, full mode)

---

## Reviewer Information

### Reviewer Role *
Peer Reviewer 1 (Methodology)

### Reviewer Identity *
Statistician specialising in the evaluation of machine-learning models: seed variance, paired designs and multiple-comparison control. Remote-sensing specifics (SAR preprocessing, cloud physics, co-registration) are outside my core competence and are assessed only where they touch inference.

### Review Focus *
(1) What 10 training seeds evaluated on one fixed set of 9–10 test cities can and cannot support; (2) power, minimum detectable effects, reporting of nulls, Holm family definitions, t-test vs Wilcoxon; (3) validation-only selection, frozen thresholds, leakage, reproducibility, and the audit/pre-registration claims. Throughout: does any conclusion exceed what the design licenses?

Verification performed: read the full manuscript; read `experiments/robust/PHASE6/7/8_PREREGISTRATION.md`, `phase8/summary.json` and `src/training/train_rrd.py`; recomputed every primary-family p-value in §4 from the reported d_z (n = 10, df = 9) and the minimum-detectable-effect (MDE) multipliers by simulation; ran exploratory omnibus and pairwise tests on the Step 8 per-seed F1 values stored in the repository.

---

## Overall Assessment *

### Recommendation *
- [ ] **Accept**
- [ ] **Minor Revision**
- [x] **Major Revision**
- [ ] **Reject**

### Confidence Score *
4 — the statistical design and inference are my core area; remote-sensing preprocessing is adjacent.

Confidence is an uncertainty/scope disclosure only; it never changes consensus counts, severity, decision bearing, or arbitration.

### Summary Assessment *
The paper is a measurement study on OSCD. It uses the official DS-UNet and U-Net baselines with 10 training seeds per arm, selects checkpoints and thresholds on 4 validation cities, and runs paired t-tests with Holm correction inside pre-specified families. It answers five RQs and adds a seven-step cloud-robustness study. For an ML benchmark paper the statistical hygiene is unusually good. Every primary p-value I recomputed matches. Thresholds are frozen from validation. Deviations are disclosed in detail, and the text repeatedly says that a null is not equivalence and that seeds are not cities.

Three problems remain where the conclusions go further than the design allows. First, the headline RQ2 answer ("no detectable effect of acquisition timing") rests on a test of a linear trend. The stored per-seed data show a strong non-linear timing effect: in an exploratory Friedman test, p = 6e-4 for B and 5e-6 for async. Second, every interval in the paper describes training-seed variability on a fixed set of 9–10 cities, and per-city performance ranges from about 0.04 to 0.66 F1. Even so, the abstract and conclusion use unconditional verbs ("beats", "outperforms", "most practical"), and no analysis that resamples cities is offered. Third, the step budget was tuned only for DS-UNet arm A, and the 0.035 F1 bound quoted for the B-vs-A "no cost" claim is below the actual MDE of that family. These problems can be fixed with reanalysis of data the authors already hold, with no new training, so I recommend Major Revision rather than rejection.

---

## Strengths *

### S1: Strict validation-only selection with frozen thresholds and test-count anchors
Each run selects its checkpoint by validation F1 every 25 steps. A threshold is chosen per input condition on the 4 validation cities, stored with the checkpoint, and applied unchanged on test. The code (`train_rrd.py` l.173–188) evaluates test exactly once, on the saved checkpoint. From Phase 3 onward, corrupted conditions use the clean-validation threshold, which is the realistic deployment choice. Re-evaluations must reproduce the stored test counts exactly (anchors), and in Step 8 this check caught a missing `model.eval()` before any result was read.
**Evidence Anchor**: text: §3.3 "it is frozen and applied unchanged on test"

### S2: Correct, complete paired reporting with nulls labelled as nulls
Every primary contrast reports the mean Δ, 95% CI, d_z, a sign count, Holm-adjusted p and a Wilcoxon test. Non-significant results are reported in full, not dropped. The labelling rule ("no detectable difference"; "a null is not evidence of equivalence") is applied consistently. All nine primary-family p-values in §4.1, §4.2 and §4.4 that I recomputed from d_z agree with the printed values to rounding (see "Arithmetic recompute" below).
**Evidence Anchor**: text: §3.4 "Labels are \"X better/lower\" if Holm p < 0.05 with that sign, otherwise \"no detectable difference\"; a null is not evidence of equivalence"

### S3: Honest handling of t-test/Wilcoxon disagreement and near-miss decision thresholds
When the registered t-test and the Wilcoxon test disagree (RQ2 slope: t p = 0.105, Wilcoxon p = 0.049; Step 5S miss50), the registered test governs and the other is reported as context. The retained gain of 0.740 against a pre-specified 0.75 is reported as "no pre-specified decision branch applies exactly" instead of being rounded up. This is the behaviour a pre-registration is meant to produce.
**Evidence Anchor**: text: §4.3 "the registered test is the t-test, so the slope's sign is suggestive, not established"

### S4: Pre-specified analysis sets and disclosed deviations
The Las Vegas exclusion in Step 7 and the 9-city primary set in Step 8 are fixed in the pre-registrations from data-only criteria before any test evaluation (PHASE7_PREREGISTRATION §2; PHASE8_PREREGISTRATION §2). Appendix C lists 11 deviations, including ones that weaken the paper: a test-informed step-budget proposal, missing smoke logs, and a retrain checked only against a log line.
**Evidence Anchor**: text: Appendix C item 1 "A 1000-step budget was first proposed after the test F1 of a 3-seed step sweep had been viewed"

### S5: Deterministic training verified, so seed variance is the only stochastic source
cuDNN determinism plus two bit-identical retrains mean that the seed-to-seed spread is genuine training variance and not run-to-run noise. In `train_rrd.py`, one NumPy generator seeded per run drives city, crop, augmentation and condition sampling. This gives arms trained with the same seed common random data streams, which partly justifies the pairing (see W6).
**Evidence Anchor**: text: §3.2 "Repeat training was checked twice and gave bit-identical weights"

---

## Weaknesses *

### W1: RQ2 "no detectable effect of acquisition timing" generalises a linear-trend null to all timing effects; the stored data show a strong non-linear effect
**Problem**: The registered primary test is the per-seed least-squares slope of F1 against gap months. A slope of zero excludes only a monotone linear trend. The pattern in the table is non-monotone: 1 month is lowest, 3 months highest, 6 and 18 months intermediate. Yet the abstract, §4.3, §4.5, §6 and §8 all say that timing has "no detectable effect". Using `phase8/summary.json`, I ran exploratory, unregistered, unadjusted tests. For B, the Friedman test across the five gaps gives p = 6.2e-4 and repeated-measures ANOVA gives F(4,36) = 5.7, p = 0.001. B@1 month is below B@3 months (Δ −0.033, 9/10 seeds, Wilcoxon p = 0.004), below B@12 months (10/10 seeds) and below B@OSCD date 1 (10/10 seeds). For async, Friedman p = 5e-6 and 1 vs 12 months gives Δ −0.074, d_z −1.97. The secondary family in the paper already finds async@1 month detectably below SAR alone. The t-test on slopes is also sensitive to one outlying seed: seed 7's slope is +0.045 F1/year against ≤ 0.011 for the others. This is why t gives p 0.105 and Wilcoxon gives p 0.049.
**Evidence Anchor**: text: Abstract "shows no detectable effect of acquisition timing (slope +0.0083 F1 per year, p = 0.105)"
**Why it matters**: RQ2 asks how the gap "affects" performance. The paper's answer denies an effect that the paper's own data detect, as long as the test is not restricted to a linear form. The comparison with Song et al. ("decline … is not reproduced") then rests on the wrong summary.
**Suggestion**: Limit the claim to what was tested: "no detectable linear trend with gap". Report an omnibus within-checkpoint test (Friedman or RM-ANOVA) and the 1-month vs other-gap contrasts as clearly labelled exploratory analyses with their own Holm family. Rewrite the abstract and conclusion along the lines of "timing matters (1-month images score lowest), but there is no monotone age trend". Name the seed-7 outlier explicitly when explaining the t/Wilcoxon split.
**Severity**: Major
**Confidence**: 5 — core expertise: repeated-measures and trend testing; recomputed from repository data

### W2: In RQ2 the unit of replication is the scene, which seeds do not replicate
**Problem**: Each city has exactly one scene per gap (45 scenes in the primary set). The 10 seeds re-run the same frozen models on the same 45 scenes, so any gap effect, or its absence, is confounded with scene-specific properties: season, residual haze, illumination, and how much of the labelled change the scene shows. The paper names the change-exposure confound but not this one. Seasonal mismatch with OSCD date 1 is a plausible alternative cause of the 1-month drop. The confound statement also understates its own reach. The manuscript says "At 1–6 months no primary city's image precedes OSCD date 1". In fact, according to `summary.json` → `image_before_oscd_t1`, no primary city's 12-month image precedes date 1 either. Four of the five gaps are therefore entirely post-date-1, and only the 18-month gap varies precedence.
**Evidence Anchor**: text: §3.1 "At 1–6 months no primary city's image precedes OSCD date 1; at 18 months, 4 of the 9 do"
**Why it matters**: The p-values for RQ2 are conditional on 45 specific scenes. Inference about "acquisition timing" as a factor would need several scenes per city per gap, or at least a scene-level covariate analysis.
**Suggestion**: State that 1–12 months are all post-date-1 for the primary cities. Add a scene-level descriptive table (season, day-of-year offset from date 1, cloud fraction) per gap. If feasible, add a second scene per gap from the ±45-day window as a sensitivity analysis; this is evaluation only, with no retraining. Otherwise frame RQ2 explicitly as conditional on the selected scenes.
**Severity**: Major
**Confidence**: 4 — core expertise on replication units; domain detail (seasonality) adjacent

### W3: No analysis accounts for test-city sampling, yet headline claims are stated unconditionally
**Problem**: §3.4 and §7 correctly say that intervals cover training seeds only. However, the abstract ("DS-UNet beats …"), §4.1 ("outperforms … which agrees with Hafner et al."), §8 ("the most practical is a modality-dropout DS-UNet") and the RQ5 answer read as properties of the methods, not of these 9–10 cities. City heterogeneity is large: A's per-city F1 ranges from 0.04 (Valencia) to 0.66 (Las Vegas) (§4.5). Pooled pixel F1 weights cities by size and change area. Las Vegas alone halves the VV+VH F1 (§4.4), which shows how much a single city can drive a pooled metric. None of the reported uncertainty reflects this.
**Evidence Anchor**: text: §3.4 "they do not include the sampling variability of the cities themselves"
**Why it matters**: The 10/10 sign counts and Holm p as small as 1e-4 give an impression of strength that applies only to repeated training on these cities. A reader will reasonably carry that strength over to new cities, which the design does not license.
**Suggestion**: The per-city confusion counts are already saved (e.g. `phase8/runs/*.npz`). Report a two-stage (city × seed) bootstrap CI, or a leave-one-city-out sign analysis, for each primary contrast, together with per-city Δ for at least RQ4 and RQ1. Add "on these test cities" to the abstract and conclusion verbs. If a primary contrast loses significance under city resampling, say so.
**Severity**: Major
**Confidence**: 5 — core expertise: hierarchical uncertainty in ML evaluation

### W4: Training budget tuned for arm A only; baselines inherit it, which biases RQ4 toward DS-UNet
**Problem**: The 2500-step budget was chosen on the validation score of DS-UNet arm A (390 / 1000 / 2500 steps). 2500 is the edge of that grid and the score was still rising. The same budget, learning rate and loss are then applied to all U-Net baselines, which "are not tuned separately" (§7). Selected U-Net thresholds reach 0.99, the edge of the threshold grid, which is a sign of a miscalibrated, possibly under- or over-trained model. The grid for the re-choice was also set after test F1 of the sweep had been seen (Appendix C item 1). This is a forking-paths risk, although the final choice used validation data.
**Evidence Anchor**: text: §3.2 "The step budget was chosen on validation score only (390 steps: 0.354; 1000: 0.382; 2500: 0.405; mean of 3 seeds"
**Why it matters**: RQ4's margins over the optical-only (+0.058, Holm p 0.037) and input-fusion (+0.061) U-Nets are moderate. A tuning asymmetry of the size typical in benchmark studies could account for them. The issue affects RQ1 less, because there the untuned SAR-only U-Net is the comparator and still wins.
**Suggestion**: Re-select the step budget (and preferably the learning rate) for each U-Net baseline on validation cities with the same grid, or show validation curves demonstrating that every baseline has plateaued by 2500 steps. Report how often a selected threshold or checkpoint falls at a grid edge. At minimum, qualify RQ4 as "under a recipe tuned for DS-UNet".
**Severity**: Major
**Confidence**: 4 — core expertise: tuning asymmetry in model comparisons

### W5: The "no full-input cost" claim for B quotes an excluded-difference bound smaller than the CI and the MDE
**Problem**: §5.1 says "differences below about 0.035 F1 cannot be excluded". The B − A CI is [−0.037, +0.031], so losses up to 0.037 are compatible with the data. The SD of the paired differences implied by that CI is 0.047. At the first Holm step of a 4-test family (α = 0.0125), the 80%-power MDE is about 1.26 SD ≈ 0.06 F1. §7 gives yet another range ("roughly 0.035–0.07 F1"). The conclusion's practical recommendation ("not detectably different from the full-input model") is in effect a non-inferiority claim made without a margin.
**Evidence Anchor**: text: §5.1 "differences below about 0.035 F1 cannot be excluded"
**Why it matters**: The recommendation of B as "the most practical" configuration depends on this equivalence-like reading.
**Suggestion**: Report the CI bound (losses up to 0.037 F1 not excluded) and the family-correct MDE. If an equivalence claim is wanted, pre-state a margin and run TOST. Otherwise phrase it as "no detectable cost; losses up to ≈ 0.04 F1 are compatible with the data". Make the MDE statements in §3.4 (1.00/1.14 SD), the Step 8 pre-registration (1.33 SD) and §7 (1.00–1.46 SD) consistent, and say which family each refers to.
**Severity**: Minor
**Confidence**: 5 — core expertise; recomputed MDE by simulation (1.00 SD at α 0.05, 1.14 at 0.025, 1.33 at 0.01, matching the paper's multipliers)

### W6: The basis for pairing across different architectures is not stated
**Problem**: Contrasts between independently trained models (A vs SAR-only U-Net, VV vs VV+VH, B vs SAR-only by land cover) are paired by seed index. Within one checkpoint (gap tags, mask variants, Step 5), pairing is natural. Across architectures, it is valid only as a common-random-numbers design. The code suggests that this exists for `train_rrd.py` arms: one seeded NumPy generator drives cities, crops, augmentation and condition draws, but weight initialisation differs by architecture. The paper never says this, and Phase 2 arms were trained with a different script. "Paired training seeds" (Abstract) is therefore undefined, and d_z, sign counts and the Wilcoxon test all depend on it.
**Evidence Anchor**: text: Abstract "10 paired training seeds"
**Why it matters**: A paired t-test on arbitrarily matched independent runs remains valid but is not more informative than Welch. Sign counts like "10/10" read as stronger evidence than they are if the pairing carries no shared variance.
**Suggestion**: State what a shared seed fixes in each script. Report the between-arm correlation of per-seed F1 for the primary contrasts, and give Welch (or a permutation test over all pairings) as a sensitivity analysis for cross-architecture contrasts.
**Severity**: Minor
**Confidence**: 4 — core expertise; code read for one of two training scripts

### W7: RQ5 false-alarm differences are read without the operating-point check applied to RQ3
**Problem**: In §4.4 the paper correctly argues that VV+VH's lower false-alarm rates (FAR) reflect a shift of operating point, given its lower recall. RQ5 compares B@no_o2 with the SAR-only U-Net at different frozen thresholds, but the per-class recall needed for the same check is not shown. Per-class recall is stored in `summary.json` (`worldcover` → recall). The shrubland "B lower" result (1.3% of pixels) is also one significant test out of 7 in a secondary family. The paper as a whole has many secondary families, each Holm-corrected only internally. Yet this result appears in the abstract and conclusion as RQ5's answer.
**Evidence Anchor**: table: §4.5 RQ5 table — shrubland mean Δ −0.006, Holm p 0.0127, label "B lower"
**Why it matters**: An operating-point shift or a chance finding would turn the RQ5 answer from "one small class" into "none", which is a small change. The inconsistency of method between RQ3 and RQ5 is still a defect.
**Suggestion**: Add per-class recall for B and SAR-only next to the FAR table and apply the §4.4 reasoning. State in §3.4 how many tests were run across all secondary families, and that family-wise control is per family. Remove the shrubland result from the abstract or mark it as secondary.
**Severity**: Minor
**Confidence**: 4 — core expertise on multiplicity; class-level interpretation adjacent

### W8: Abstract says "an independent audit for every step"; the body contradicts both "independent" and "every"
**Problem**: §5 states "Step 4A has no audit of its own". Declarations state that the same AI assistant wrote the analysis code, the audit scripts and the text. An audit script by the same author or agent catches arithmetic and transcription slips, not shared design errors, so it is not independent in the usual sense. The phrase "pre-registration" is also used for internal files that, for Phases 2–4A-bis, are dated only by file timestamps. The Step 8 pre-registration was also written after Step 6's results were read (PHASE8_PREREGISTRATION §"Timing"). The paper discloses all of this in the body, but the abstract does not reflect it.
**Evidence Anchor**: text: Abstract "an internal pre-registration and an independent audit for every step"
**Why it matters**: Statements about procedure in the abstract set the reader's level of trust. Here they are stronger than the body supports.
**Suggestion**: Reword to "an internal pre-registration and a recomputation audit script for each step except Step 4A". Drop "independent", or define it (e.g. "a separate script recomputing statistics from raw counts").
**Severity**: Minor
**Confidence**: 5 — direct textual contradiction

### W9: Raw counts behind the audits are not public, so the audits cannot be re-run externally
**Problem**: The paper says that "an independent audit script recomputes every reported statistic from raw counts". However, it also says raw count arrays are "not in the repository because of their size" and are available on request. Locally, the Step 8 count files total about 10 MB and Step 7 about 5 MB. These are not large.
**Evidence Anchor**: text: Declarations "Checkpoints and raw count arrays are not in the repository because of their size"
**Why it matters**: The paper's central credibility claims are reproducibility and auditability. Without the counts, a reader can verify only summaries.
**Suggestion**: Deposit the count arrays (and ideally the checkpoints) in Zenodo or a similar archive with a DOI, and cite it.
**Severity**: Minor
**Confidence**: 4 — file sizes checked on the local copy; repository contents not checked

### W10: Inconsistent reporting of the declared statistics set across tables
**Problem**: §3.4 says that every primary family is reported with mean Δ, 95% CI, d_z, sign count and Holm-Wilcoxon. The Phase 3 table (§5.2) has no CI. §5.3 reports only "Holm p < 0.05 for all 6 tests". §4.5 and §5.7 omit d_z. Holm-Wilcoxon values are mostly absent. §3.4 also says "Each step has one primary family", but Step 6 has two (primary_rq4 and primary_rq1; PHASE6_PREREGISTRATION ll.48–49). The 95% CIs are unadjusted while the decisions use Holm-adjusted p, which explains small mismatches, e.g. a tree-cover CI excluding 0 alongside Holm p 0.097.
**Evidence Anchor**: table: §5.2 Phase 3 table — columns Δ, Holm p, B > A; no 95% CI column
**Why it matters**: Clarity and checkability; the core claims are unaffected.
**Suggestion**: Use one complete column set for all primary tables (a supplementary table is acceptable). State that CIs are unadjusted, or give Holm-compatible simultaneous CIs. Correct "one primary family" for Step 6.
**Severity**: Minor
**Confidence**: 5 — direct reading

---

## Detailed Comments *

### Title & Abstract
- The abstract is dense and largely accurate, but it inherits W1 (timing), W3 (unconditional verbs) and W8 (audit/pre-registration claims). Its final sentence ("seeds are training replicates, and an undetected difference is not evidence of no effect") is good practice and should be kept.

### Introduction
- The RQs are clear and are quoted verbatim from the RRD. Contribution (i), "a controlled, multi-seed answer", should be qualified as conditional on these test cities (W3).

### Methodology / Research Design
- **Design.** There are 9 sequential steps, each with its own pre-registration and primary family. This is reasonable for a measurement study but creates a paper-level multiplicity that is never quantified (W7).
- **Replication.** Seeds are the replicate, and cities and scenes are fixed (W2, W3). With n = 10 the MDE is about 1 SD of the paired differences. In F1 units this is about 0.04 for RQ1-B and 0.07 for RQ1-async. The RQ1 answer correctly says a small benefit cannot be excluded. The CI upper bounds (+0.040 and +0.026) are more informative than the MDE and should be the headline bound, i.e. "a benefit above 0.04 F1 is not supported".
- **Selection.** Checkpoint and threshold selection on 4 validation cities is noisy, but it is honest because test is touched once. The different checkpoint rules per arm ("by design", §7) are fine. The tuning asymmetry is not (W4).
- **Leakage.** I found none: city separation is asserted in code, test is evaluated once after selection, and anchors are in place. The Step 4A-bis and 4B printing of test F1 during runs is a mild exposure that is disclosed and that was not used for decisions.
- **Test choice.** The t-test as primary with Wilcoxon alongside is defensible at n = 10, given that the minimum attainable Wilcoxon p is 0.002 (correctly stated). Outliers (the collapsed A seed in `no_opt`, seed 7 in the RQ2 slopes) are where the two tests diverge. The paper handles the first well and should name the second (W1).

### Results / Findings
- RQ4 and RQ3: the arithmetic is consistent, and the interpretation is appropriately arm-specific. RQ4 is subject to W3 and W4.
- RQ1: correctly framed as a null with a stated MDE.
- RQ2: see W1 and W2. This is the most over-reaching section.
- RQ5: see W7.
- §5: the robustness results are large effects (often 10/10 seeds, Δ of 0.1–0.3 F1) and are caveated as synthetic or composite clouds. The inference is adequate for what is claimed. In §5.7 the reading of rb95's gain as "discarding the image" is well argued from the rb99 control.

### Discussion
- §6 correctly keeps both explanations for the RQ1 null open. The "possible reason for the low 1-month score" paragraph should add the scene-specific alternative (W2), and should note that the low 1-month score is itself statistically detectable in a within-checkpoint comparison (W1).

### Conclusion
- "(RQ2) its acquisition timing between 1 and 18 months has no detectable effect" should become "no detectable linear trend" (W1). The "most practical" recommendation needs the CI-bound phrasing (W5) and conditioning on the test cities (W3).

### References
- Not assessed (outside remit).

---

## Arithmetic recompute (standard-mode record)

All values below come from manuscript-reported d_z with n = 10 and df = 9, two-sided, with t = d_z·√10. Holm is applied as printed family order × (k − rank + 1). These are auditable recomputations, not adjudicated truths.

| contrast | reported | recomputed raw p → Holm | status |
|---|---|---|---|
| A − SAR-only (d_z 2.26, k=3, rank 1) | 1.6e-04 | 5.39e-05 → 1.62e-04 | consistent |
| A − optical (0.77, rank 3) | 0.0370 | 0.0377 → 0.0377 | consistent (d_z rounding) |
| A − fusion (1.54, rank 2) | 0.0017 | 8.84e-04 → 0.0018 | consistent (rounding) |
| B@no_o2 − SAR (0.33, k=2, rank 1) | 0.6576 | 0.324 → 0.648 | consistent (d_z rounding) |
| async@no_o2 − SAR (−0.27, rank 2) | 0.6576 | 0.415 → 0.648 after monotonicity | consistent |
| SAR VV+VH − VV (−1.49) | 0.0011 | 0.0011 | consistent |
| SAR VH − VV (−2.32) | 8.8e-05 | 4.39e-05 × 2 = 8.78e-05 | consistent |
| DS VV+VH − VV (0.27) | 0.4125 | 0.415 | consistent |
| DS VH − VV (−0.59) | 0.1925 | 0.0949 × 2 = 0.190 | consistent |
| RQ2 slope (mean 0.0083/yr, CI [−0.0021, 0.0187]) | p 0.105 | t = 1.81, p = 0.104 | consistent |
| Wilcoxon minimum p, n = 10 | 0.0020 | 2/1024 = 0.00195 | consistent |
| MDE multipliers (80% power) | 1.00 / 1.14 / 1.33 SD | simulated 0.99 / 1.14 / 1.33 | consistent |

No arithmetic mismatch was found. The W5 problem concerns which MDE is quoted, not arithmetic.

---

## Questions for Authors *

1. Under a two-stage city × seed bootstrap of the saved per-city counts, do the RQ4 contrasts (especially A − optical-only, Holm p 0.037) and the RQ3 SAR-only contrast keep their signs and significance?
2. Do you accept that the Step 8 data show a detectable non-linear timing effect (1-month images lower in 9–10 of 10 seeds for both models)? Would you report an omnibus within-checkpoint test as exploratory and revise the RQ2 answer?
3. Were the U-Net baselines' validation curves still rising at 2500 steps? How many selected thresholds or checkpoints across all arms lie at a grid edge?
4. In `train_robust.py` (Phase 2 arms), does a shared seed produce the same crop, augmentation and condition stream as in `train_rrd.py`? If not, what does "paired" mean for the A/B vs Step 6 baseline contrasts?

---

## Minor Issues

### Language / Grammar
- §3.4: "1.14 SD at α = 0.05/2" should name the family it applies to (Step 6 RQ1). Step 8 uses 1.33 at α = 0.05/5.
- §4.3 "Answer": "the positive slope is not established (registered t-test p 0.105)" duplicates the preceding paragraph.

### Figures and Tables
- Add the SAR-only reference (0.406 ± 0.048) as a column or row in the Step 8 timing table, rather than in a separate sentence.
- Figure 2–4 captions: state the threshold regime (frozen clean-validation) and that error bars, if any, are seed SDs.

### Layout
- Appendix C items 9–10 could carry the affected sections (§4.3, §4.4) inline for traceability.

### Manuscript-integrity note
- The manuscript contains no text addressed to reviewers and no instruction-like content. Bracketed author placeholders remain in the Declarations ("[author: to be agreed with the mentor …]", "[author: confirm.]") and must be resolved before submission.

---

## Criterion-Bound Judgements *

Calibration status: `NOT_CALIBRATED`

| Dimension | Criterion source | Judgement | Evidence anchor(s) | Rationale | Uncertainty / scope limit | Decision bearing? |
|---|---|---|---|---|---|---|
| Originality | quality_rubrics | NOT_ASSESSED | — | Outside methodology remit | — | no |
| Methodological Rigor | quality_rubrics; Config Card #2 focus 1–3 | PARTLY_MEETS | text: §3.3 "frozen and applied unchanged on test"; text: §3.2 step-budget quote | Selection, leakage control and test reporting are strong; tuning asymmetry (W4) and the replication unit (W2, W3) are unaddressed | Baseline tuning effect size unknown until rerun | yes — W3, W4 repairable by reanalysis or limited reruns |
| Evidence Sufficiency | quality_rubrics | PARTLY_MEETS | text: Abstract timing quote; table: §4.5 RQ5 | RQ1, RQ3 and §5 are adequately supported for their stated scope; RQ2 and RQ4 are stated more strongly than the evidence allows | City-bootstrap result unknown | yes — W1, W3 |
| Argument Coherence | quality_rubrics | PARTLY_MEETS | text: §5.1 0.035 bound; text: §3.4 one primary family | Inconsistent MDE statements and RQ3/RQ5 operating-point logic | — | no (Minor-level) |
| Writing Quality | quality_rubrics | NOT_ASSESSED | — | Outside remit | — | no |
| Literature Integration | quality_rubrics | NOT_ASSESSED | — | Reviewer 2's remit | — | no |
| Significance & Impact | quality_rubrics | NOT_ASSESSED | — | Reviewer 3's remit | — | no |

**Recommendation rationale.** Major Revision. The unresolved criteria that bear on the decision are Evidence Sufficiency (W1: the RQ2 conclusion is contradicted by the paper's own stored data when the test is not restricted to a linear trend; W3: no uncertainty from test-city sampling) and Methodological Rigor (W4: tuning asymmetry in RQ4). W2 limits how far RQ2 can be generalised. All are repairable with existing per-city and per-scene counts plus, for W4, a validation-only step-budget selection for the three U-Net baselines. No finding is singly Critical: the paper's procedural integrity is high, and its main nulls (RQ1) and large effects (§5) survive the scrutiny I applied.
