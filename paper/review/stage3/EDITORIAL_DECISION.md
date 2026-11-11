# Stage 3 Editorial Decision and Revision Roadmap

- **Date:** 2026-11-02
- **Manuscript:** `paper/FINAL_RESEARCH_PAPER.md`, SHA-256 `7220d00b9c4e2ec08803faceeacc5437d4876243edf89ea91565c9514aa1bc8f`. All five reviewers checked this hash before reading.
- **Target:** generic IEEE remote-sensing journal. `criteria_binding_unavailable`: no venue-specific fit is claimed.
- **Reports:** `R_EIC_journal_fit.md`, `R1_methodology.md`, `R2_domain.md`, `R3_perspective.md`, `DA_devils_advocate.md` (all in this folder).

## Decision: MAJOR REVISION

All five seats independently recommend Major Revision. No seat recommends Reject. The panel's agreed view:
- **Process is strong.** Validation-only selection, frozen thresholds, pre-registered families and disclosed deviations are praised by R1, EIC, DA and R2.
- **Arithmetic is correct.** R1 recomputed every primary p-value and found no mismatch.
- **Some conclusions are worded more strongly than the design allows.** RQ2 is the clearest case.
- **The framing doesn't serve outside readers.** This covers the title, the use of the internal requirements document, the contribution statement, length and the literature gap.

Most issues can be fixed by rewriting and by re-analysing results that are already stored, with no new training.

## Panel provenance (six axes)

| Axis | Value |
|---|---|
| Role separation | Five seats, each given its own role card and instructions |
| Invocation context | Each seat ran in a fresh subagent session with no prior conversation |
| Peer-output visibility | None before submission: each seat was told not to read the others' reports or the Stage 2.5 report |
| Model family | All five: Claude (Opus 5.5) |
| Provider | Anthropic, all five |
| Accountable human | None per seat. The author reads and decides |

**Correlated-error disclosure:** all seats share one model family, so they can share blind spots. Agreement between them is not independent confirmation. Two seats (R1, DA) independently found the same RQ2 problem in the stored data, which raises confidence in that finding but doesn't rule out a shared error.

**Protocol deviation:** each seat wrote its report in one pass. The plugin's two-call sprint contract (commit criteria without seeing the paper, then review) and its JSON conformance checkers were not run. Execution provenance was recorded by hand in this table, not built with `review_panel_provenance.py`.

## Devil's Advocate CRITICAL adjudication

**C1 (RQ2 "no detectable effect of timing"): VALIDATED. It blocks Accept until fixed.**
- **R1 W1 corroborates it independently:** on the stored per-seed data, a Friedman test gives p = 6.2e-4 for B. The 1-month image is below the 3-month image in 9 of 10 seeds.
- **EIC W4 raises the same scope problem** from the journal side.
- **My check:** the registered test is a straight-line slope (§4.3), and the Step 8 table is not monotone. "No detectable linear trend" is supported; "no detectable effect" is not.
- **Resolution, by text and re-analysis only:**
  - Reword every instance in the Abstract, §1, §4.3, §6 and §8.
  - Report an omnibus test and the 1-month contrasts as a separately labelled exploratory family.
  - Name seed 7 as the reason the t-test and Wilcoxon disagree.

## Consensus findings (Major)

| # | Issue | Raised by | Summary |
|---|---|---|---|
| 1 | RQ2 overreach and Song comparison | DA C1, M3; R1 W1, W2; EIC W4; R2 W1, W6 | Linear-trend null stated as no effect. Every 1–12 month primary image post-dates OSCD date 1; the paper says 1–6 (R1, DA, both from `phase8/summary.json`). One scene per city per gap, so seeds don't replicate scenes. In Song et al. the image is a target-date prior, here a pre-change baseline, so "not reproduced" is the wrong frame |
| 2 | No uncertainty from the choice of test cities | R1 W3; DA M8; EIC W6 | All intervals cover seeds on 9–10 fixed cities. Asks for a city × seed bootstrap or leave-one-city-out from stored per-city counts, and "on these test cities" in the headline wording |
| 3 | RQ1 depends on the threshold rule and the architecture | DA M1, M2; R2 W5 | At the frozen clean threshold used in §5, B without the post-change image scores 0.478 against SAR-only's 0.418, which DA tests as p 0.003. The paper's 0.431 uses per-condition thresholds. Promote the within-B present-minus-absent contrast to primary. Scope RQ1 to late-fusion DS-UNet with 3-month mean SAR |
| 4 | RQ5 attribution | R2 W4; DA M6; R1 W7 | The shrubland effect compares B with a different network. Asks for B with the historical image minus B without it, by class, or a reworded claim, plus per-class recall |
| 5 | RQ4 framing and tuning asymmetry | DA M4, M5; R1 W4 | RQ4 asks about the asynchronous approach but is answered with the synchronous model A. The step budget was tuned only on A; baselines are untuned, with thresholds up to 0.99 |
| 6 | Practical recommendation not compared with simple rules | R3 W3; DA M7, M10; R1 W5 | No comparison with "A when the image is present, SAR-only when missing or cloudy", or with discarding the image when cloud is detected. "Most practical" reads a null as equivalence; losses up to 0.037 F1 are not excluded |
| 7 | Title and framing | EIC W1–W3, W8; R3 W1; R2 W10; DA m4 | Title claims environmental monitoring that wasn't tested. Research questions are justified from the internal requirements document. Contributions don't say why the nulls matter. Abstract about 305 words |
| 8 | Literature | R2 W2, W9; EIC W7 | Heterogeneous (optical-to-SAR) change detection and 2023–2026 missing-modality work are missing. §2's claim "All of these assume both modalities" overstates the gap |
| 9 | Sentinel-1 scene asymmetry between dates | R2 W3 | Date-1 composites average about half the date-2 scene count (pre-Sentinel-1B); Dubai's date-1 VH comes from a single scene. An alternative explanation for the RQ3 VH result. Must be confirmed against the export logs |
| 10 | Length and structure | EIC W5, W12 | Main text about 9,450 words with 16 tables; the cloud study reads as a second paper in project order. Needs a forest plot and a timing plot |
| 11 | No real cloudy images | R3 W2 | All cloud results are synthetic or composite. A real near-date-2 scene per city is suggested |

**Single-seat Majors are carried forward:** R2 W2 (literature) and R3 W2 (real clouds) are included above.

## Minor findings (carried to revision)

- **Audit wording.** "An independent audit for every step" is wrong on both counts: Step 4A has none, and the same AI wrote the code and the audits (R1 W8, DA m1).
- **MDE statements and the 0.035 bound** are inconsistent (R1 W5).
- **Pairing across architectures** is not explained (R1 W6).
- **The full statistics set** is missing in some tables (R1 W10).
- **Raw counts** are not deposited publicly, though they are only about 15 MB (R1 W9, R3 W7).
- **Repository reuse:** no LICENSE file, no prediction script, a hard-coded path, and missing requirements (R3 W7).
- **Threshold transfer** to new regions (R3 W4); zero-fill doubling as the missing token (R3 W5); pixel F1 against agency accuracy statistics (R3 W6); L1C against L2A (R3 W9).
- **SAR details:** orbit and geometry (R2 W7); composite details and the Mumbai date-1 window (R2 W8).
- **Contribution (iv)** is already known [17] (EIC W11). The dilation radius sits at the edge of its grid (DA m5).
- **Placeholders and authorship:** Declarations still have placeholders, and the mentor is in CRediT but not on the byline (EIC W9, R3 W8, DA m3, R1).
- **Song reading:** the paper says it read only some sections (EIC W10, R2 W1). The full text was read in Stage 2.5, so the sentence can now go.

## Disagreements

- **Splitting the paper.** EIC offers splitting out the cloud study or cutting the main text to about 7,000 words. Nobody else asks for a split, and R3 and R1 treat §5 as the most useful part. **Arbitration:** keep one paper, but restructure it (see the roadmap below).
- **Cloud-study novelty.** DA M10 says it is close to standard mask-then-treat-as-missing practice. EIC S3 and R3 S1 call the mask-error asymmetry practically useful. **Arbitration:** keep the result, but frame it against standard practice and add the image-level discard comparison.

## Revision Roadmap

Tier 1 needs only rewriting and analysis of results already on disk. Tier 2 adds new evaluations of existing checkpoints. Under the project rule, a pre-registration must be committed before each new test evaluation. Tier 3 needs new training.

**Tier 1: required**
1. RQ2:
   - Reword to "no detectable linear trend".
   - Add an exploratory omnibus test and 1-month contrasts.
   - Correct "1–6 months" to "1–12 months post-date-1".
   - Frame the Song comparison as a different role for the image.
   - Note the one-scene-per-gap limit and seed 7.
2. City-level uncertainty:
   - Add a city × seed bootstrap and leave-one-city-out for every primary contrast, from stored per-city counts.
   - Write a new script; label it exploratory.
   - Condition the headline wording on these cities.
3. RQ1:
   - Report both threshold rules.
   - Promote the within-B contrast.
   - Scope the claim to late-fusion DS-UNet with 3-month mean SAR.
4. RQ4:
   - Answer it for the asynchronous arms as well.
   - Qualify that the baselines use a recipe tuned on DS-UNet.
5. Practical recommendation:
   - Compare the switch rule and image-discard rule from stored results (as far as the stored counts allow).
   - Replace "most practical" and the equivalence wording.
6. Framing:
   - New title; research questions motivated from the literature, without the requirements document.
   - Contributions as scoped findings; drop contribution (iv).
   - Abstract of 250 words or fewer; correct the audit wording.
7. Literature: add a heterogeneous change-detection paragraph and recent missing-modality work, using the references the reviewers verified.
8. Disclose the Sentinel-1 scene counts per date, after confirming them against the export logs.
9. RQ5:
   - Add per-class recall.
   - Reword the attribution, unless Tier 2 item 12 runs.
10. Restructure:
    - Move the Phase/Step chronology and commit hashes to appendices.
    - Add a forest plot of primary contrasts and an RQ2 timing plot.
    - Make the statistics columns consistent; fix the MDE statements.
11. Declarations: authorship (author's decision), conflict of interest, funding.

**Tier 2: recommended if time allows (existing checkpoints, pre-registration first)**

12. RQ5 within-B contrast: B with the historical image minus B without it, by land-cover class.
13. Real-cloud check: one partly cloudy scene near OSCD date 2 per test city, masked with s2cloudless, comparing A, B and SAR-only.
14. Sentinel-1 scene-count control for RQ3 (subsample date-2 scenes, or drop cities with too few date-1 scenes).

**Tier 3: optional (new training; probably not feasible by 2026-11-15)**

15. Per-baseline step-budget tuning on validation cities (R1 W4).
16. An early-fusion arm in which the historical image can interact with SAR (DA M2).

If Tier 3 is skipped, the paper must state these as limitations, which items 3 and 4 already do.

**Repository (outside the manuscript):** add a LICENSE compatible with DS-UNet's, complete `requirements.txt`, and deposit the count arrays on Zenodo (R1 W9, R3 W7).
