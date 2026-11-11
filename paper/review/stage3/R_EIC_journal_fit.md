# Peer Review Report

## Manuscript Information
- **Title**: Multimodal Satellite Change Detection for Urban Development and Environmental Monitoring
- **Manuscript ID**: FINAL_RESEARCH_PAPER.md (SHA-256 7220d00b9c4e2ec08803faceeacc5437d4876243edf89ea91565c9514aa1bc8f, verified before review)
- **Review Date**: 2026-11-02
- **Review Round**: Round 1 (ARS Stage 3, full mode)

---

## Reviewer Information

### Reviewer Role *
Journal-Fit Reviewer (EIC seat, Reviewer Configuration Card #1)

### Reviewer Identity *
Associate editor at an IEEE remote-sensing journal, field-general. No target journal has been chosen.

criteria_binding_unavailable

No claim is made about fit to any named venue. Where I mention IEEE norms (letter vs. regular paper length, abstract length), I am referring to general IEEE GRSS practice, not to one journal's criteria.

### Review Focus *
(1) Whether a pre-registered measurement study on OSCD whose results are mostly null is original and significant enough for IEEE remote-sensing readers. (2) Whether the length (about 11,950 words including appendices; about 9,450 in the main text; 16 main-text tables) and the structure fit IEEE journal or letter norms. (3) How clear the title, the abstract and the contribution statement are. I do not re-derive the statistics.

---

## Overall Assessment *

### Recommendation *
- [ ] Accept
- [ ] Minor Revision
- [x] **Major Revision**
- [ ] Reject

### Confidence Score *
4. Journal fit, framing and structure are within my remit. My confidence in statistical details is lower (see per-finding confidence).

Confidence is an uncertainty/scope disclosure only; it never changes consensus counts, severity, decision bearing, or arbitration.

### Calibration Status
`NOT_CALIBRATED`

### Summary Assessment *
The paper evaluates the official DS-UNet and U-Net baselines on OSCD (Sentinel-1 VV/VH plus Sentinel-2) under one shared protocol: 10 paired seeds, validation-only selection, internal pre-registration and scripted audits. It addresses five project research questions and adds a seven-step study of robustness to clouds.

- **Main findings.** Dual-stream fusion beats U-Net baselines with full input (RQ4). The other four questions give null or near-null answers: no detectable gain from historical optical imagery (RQ1), no detectable timing effect (RQ2), VH hurts only the SAR-only model (RQ3), and a detectable false-alarm reduction only in shrubland, 1.3% of pixels (RQ5).
- **Rigour and reporting.** The procedure and the honesty of reporting are well above the norm for this field. The prior missing-optical margin disappears under a matched baseline (0.418 vs 0.241), a corrective that readers should see.
- **Missing vs. cloudy result.** The most transferable result is in §5: a partly cloudy post-change image is worse than a missing one, and errors in the cloud mask cost very different amounts depending on their direction.
- **Weaknesses as a journal submission.**
  - The paper is framed around an internal "Research Requirement Document" rather than a question from the field.
  - The title promises more than the paper delivers.
  - The main contribution is a set of nulls on 9–10 test cities, with uncertainty that covers training seeds only.
  - RQ2's design cannot isolate the variable it names.
  - Two papers' worth of material is packed into one manuscript that is ordered by process.

None of this is fatal. The message survives the nulls only if the paper is reframed around what a reader can use: a matched-protocol null, a corrected earlier claim, and the missing-versus-cloudy distinction. Major Revision.

---

## Strengths *

### S1: Unusually transparent, pre-specified and audited evaluation protocol
Every step has a pre-registration with hashed code and inputs, a frozen decision rule and an independent recomputation of the reported statistics. The paper also lists its deviations openly (Appendix C). IEEE remote-sensing papers rarely do this, and it makes the null results credible as nulls rather than as failures of tuning.
**Evidence Anchor**: `text: §3.4 "an independent audit script recomputes every reported statistic from raw counts (Step 6: 1046 checks; Step 7: 994; Step 8: 235"`

### S2: A matched-protocol correction of a previous positive claim
The paper shows that the earlier missing-optical margin (0.309 vs 0.241) was an artefact of a weak SAR-only baseline. Under the shared protocol the SAR-only U-Net reaches 0.418 and the margin disappears. This is a useful corrective for the community, and the field would benefit from more of them.
**Evidence Anchor**: `text: §4.2 "under the shared protocol the SAR-only U-Net reaches 0.418, and the margin does not survive."`

### S3: The missing-versus-cloudy distinction and the asymmetric cost of mask errors are practically useful
§5.2–5.4 show three things a practitioner can act on:
- A dropout-trained model handles a zero-filled image but not a cloud-valued one.
- Zero-fill under the mask recovers most of the loss.
- Over-masking costs almost nothing, while under-masking is expensive.

This is the most transferable result in the manuscript, with caveats (synthetic clouds).
**Evidence Anchor**: `table: §5.4 table — dilate8 Δ −0.010 vs miss25 Δ −0.168 against the oracle mask`

### S4: Limitations and the scope of each claim are stated carefully and consistently
Each answer to a research question states its scope. Null results are explicitly not presented as evidence of equivalence. The Limitations section covers the main threats.
**Evidence Anchor**: `text: §4.2 "This is not evidence that it never helps: the minimum detectable effect is about 1.14 SD of the paired differences"`

### S5: Public code, pre-registrations and audits, with hashed checkpoints
**Evidence Anchor**: `text: Declarations "Code, pre-registrations, run files, analysis tables, audits and reports are public at"`

---

## Weaknesses *

### W1: Title does not describe the paper
**Problem**: The title promises a general multimodal method for urban and environmental monitoring. The paper is a measurement study of one existing architecture on one urban benchmark, and it studies no environmental change: OSCD has no environmental labels, and §4.4 says pseudo-change discrimination cannot be shown. "Asynchronous/historical optical", "missing modality", "cloud" and "OSCD" do not appear in the title at all.
**Evidence Anchor**: `text: Title "Multimodal Satellite Change Detection for Urban Development and Environmental Monitoring"`
**Why it matters**: Readers and editors judge scope and fit from the title. A generic title that suggests a method will draw method-paper expectations, and a measurement study then disappoints those expectations at desk review.
**Suggestion**: Retitle to describe the study, e.g. "Does historical optical imagery help SAR change detection when the post-change optical image is missing or cloudy? A pre-registered multi-seed study on OSCD". Drop "Environmental Monitoring".
**Severity**: Major
**Confidence**: 5 — core editorial competence

### W2: The paper is framed around an internal project document rather than a question from the field
**Problem**: The Introduction presents the research questions as those of "the project's Research Requirement Document (RRD), word for word", and the Abstract opens its method with "Following a research requirement document". §5 is motivated by "The RRD's premise", and §7 cites features "listed in the RRD". Journal readers have no access to this document, and it is not a justification. The questions need to be motivated from the literature: the gap in Song et al. [2], Hafner and Ban [12], and missing-modality CD. Some RQ wording ("the proposed asynchronous multimodal approach") also implies a proposed method that the paper does not have.
**Evidence Anchor**: `text: §1 "This paper answers the five research questions of the project's Research Requirement Document (RRD), word for word"`
**Why it matters**: The study reads as a project report rather than a contribution to the field, which is a common desk-reject signal.
**Suggestion**:
- Remove every reference to the RRD.
- Motivate the questions from the gap in prior work, in one paragraph.
- Reword RQ4 so it does not imply a proposed method.
- Consider merging RQ5 into RQ1/RQ2. RQ5 is essentially a stratified secondary analysis.
**Severity**: Major
**Confidence**: 5 — core editorial competence

### W3: The contribution statement does not say why a set of nulls on one small benchmark matters to readers
**Problem**:
- Contribution (i) is "an answer to RQ1–RQ5 … including null results". This describes an activity, not a finding.
- The strongest takeaways are buried in §4.2 and §6 rather than stated as contributions:
  1. A matched SAR-only baseline removes the reported benefit of historical optical imagery.
  2. A dropout-trained DS-UNet costs nothing detectable on full input and falls back on SAR.
  3. Partial cloud is a worse problem than missing data, and masks should err toward over-masking.
- RQ3 and RQ5 add little to the main message. A reader finishing the Abstract cannot tell what to do differently.

**Evidence Anchor**: `text: §1 Contributions "(i) A controlled, multi-seed answer to RQ1–RQ5 on a public urban benchmark under one audited protocol, including null results."`
**Why it matters**: A null-result paper on OSCD (24 cities) is significant to IEEE remote-sensing readers only if it changes a belief or a practice. As written, that case is implied but never made.
**Suggestion**:
- Rewrite the contributions as 3 claim-shaped findings with their scope, e.g.: "(1) Under a matched protocol, historical Sentinel-2 adds no detectable F1 over bitemporal Sentinel-1 VV on OSCD (minimum detectable effect ≈ 0.04 F1), overturning an earlier single-protocol margin…"
- Put methodological transparency as a separate, secondary contribution.
- Say explicitly what the community should change: matched baselines, multi-seed reporting, and treating cloudy and missing inputs separately.
**Severity**: Major
**Confidence**: 4 — editorial judgement of significance framing

### W4: The RQ2 experiment cannot isolate the variable the question names
**Problem**: RQ2 asks how the temporal gap affects performance. The authors acknowledge that in the Step 8 design, image age is confounded with how much labelled change the image already shows. At 1–6 months every image post-dates OSCD date 1. In addition, frozen models trained on the date-1 image were evaluated on older images and none was retrained. RQ2's headline answer ("no detectable effect", and Song et al. "not reproduced") therefore rests on an experiment that, by the authors' own account, cannot test age.
**Evidence Anchor**: `text: §4.3 "This experiment therefore measures the effect of the historical image's acquisition timing under that label definition; it is not a clean causal test of image age"`
**Why it matters**: One of the five stated questions gets an answer that is not interpretable for the question asked. The "not reproduced" framing against Song et al. over-reaches, given how much the two settings differ and the confound.
**Suggestion**: Either:
- (a) demote RQ2 to a clearly labelled secondary, descriptive analysis of "acquisition-timing sensitivity of a frozen model" and remove "not reproduced" from the Answer, Abstract and Conclusion; or
- (b) add a confound-free subset: only images that precede date 1, e.g. the 12/18-month images for cities where they do, or images older than date 1 at all gaps. Ideally also include a model trained with aged images.

At minimum, report RQ2 as "timing", not "gap", throughout.
**Severity**: Major
**Confidence**: 4 — design logic, not statistics

### W5: Length and process-ordered structure: two studies in one manuscript, reported in project chronology
**Problem**:
- **Length.** The main text is about 9,450 words with 16 tables, plus about 1,800 words of appendices. That is far beyond a letter format (IEEE GRSL-type letters are about 5 pages), and at the upper end of a regular paper once tables and figures are typeset.
- **Two studies.** The robustness study (§5, seven sub-steps) is a self-contained second paper with its own arms (C), conditions, detector and decision rules. It is only loosely tied to RQ1–RQ5.
- **Process ordering.** The text is organised by project chronology and labels rather than by question: "Phase 2", "Phase 3", "Step 4A-bis", "Step 5S", "Step 6–8", commit hashes and internal tokens (`no_o2`, `p2_dropout_seed1`, `gee_t1`).

**Evidence Anchor**: `text: §5 "This section reports the seven-step robustness study with the arms A, B and C (§3.3)."`
**Why it matters**: Readers lose the main message among process detail, and the length raises reviewing and page-charge costs for a paper with mostly null outcomes.
**Suggestion**: Choose one of two options.
- **(a) Split.** Paper 1 covers RQ1/RQ4 plus a short RQ3/RQ2 analysis (letter or short regular paper). Paper 2 covers missing vs. cloudy post-change optical input (§5), which is the more novel and practically useful study.
- **(b) Keep one regular paper.** Cut it to about 7,000 words:
  - Move the §5.5–5.6 and Step 4A detail and the per-class tables to supplementary material.
  - Replace step and phase names with descriptive names.
  - Move commit hashes and audit counts to a single reproducibility appendix.
  - Delete Appendix B (results not used).
**Severity**: Major
**Confidence**: 4 — general IEEE GRSS length and format norms; no venue chosen

### W6: The uncertainty behind the null claims covers training seeds only, not cities, so the scope of the nulls is narrower than readers will assume
**Problem**: The paper states clearly that its CIs and p-values cover training-seed variation on fixed test cities only. The headline claims, however, are phrased as properties of the benchmark ("On OSCD … gives no detectable improvement"), with 9–10 test cities and pooled pixel counts. Readers of a null-result paper will want to know whether the null holds across cities, or whether a few large cities dominate the pooled F1. Las Vegas alone nearly halves the VV+VH F1 when included.
**Evidence Anchor**: `text: §3.4 "they do not include the sampling variability of the cities themselves."`
**Why it matters**: For a paper whose message is a null, the relevant population is the cities. The current analysis cannot distinguish "no effect anywhere" from "positive in some cities, negative in others".
**Suggestion**:
- Add a per-city table or forest plot for the RQ1 primary contrasts. Show the per-city Δ, averaged over seeds.
- Add a city-level (or city × seed) bootstrap interval as a secondary analysis.
- Make the Abstract's scope sentence explicit: "the intervals reflect training variability on 9–10 fixed test cities".
**Severity**: Major
**Confidence**: 3 — the design point is clear to me; the best resampling scheme is outside my remit

### W7: Literature positioning is thin for a 2026 IEEE remote-sensing audience
**Problem**: Related Work cites 24 references. Most of the substantive ones are from 2018–2022 (OSCD, DS-UNet, FC-Siam, a 2022 review, DSNUNet, ModDrop). It does not engage with:
- the broader missing-modality and incomplete-multimodal learning literature in remote sensing and computer vision after 2022;
- cloud-robust or SAR-guided change detection beyond one cloud-removal paper;
- the remote-sensing foundation-model and pretrained-encoder work that readers will ask about when a 2022 U-Net is the only architecture.

Transformer CD papers [20–22] are cited only in an appendix.
**Evidence Anchor**: `absence: §2 Related Work — expected engagement with post-2022 missing-modality, cloud-robust change detection and pretrained/foundation-model SAR–optical work; checked §2, §6 Discussion, §7 Limitations, reference list [1]–[24]`
**Why it matters**: Without that positioning, a reader cannot judge whether the null is specific to DS-UNet or likely to be general. "One architecture" is listed as a limitation but not discussed against the alternatives.
**Suggestion**: Add a paragraph each on missing-modality learning in remote sensing and on cloud-robust or SAR-guided change detection (2023–2026). In §6, discuss why a pretrained or attention-based fusion model might use the historical optical image differently.
**Severity**: Major
**Confidence**: 3 — field-general editor; no systematic search done for this review

### W8: The Abstract is too long and too dense to carry the message
**Problem**: The Abstract is about 305 words, above the typical 150–250-word IEEE limit. It packs in more than 15 numbers, five research-question answers, a confound caveat and the robustness results. It opens with the RRD instead of the problem, and it ends on caveats rather than a take-home message.
**Evidence Anchor**: `text: Abstract "Following a research requirement document with five questions, we evaluate the official dual-stream U-Net (DS-UNet) and U-Net baselines"`
**Why it matters**: An abstract that reads as a results dump obscures the one or two findings that would interest readers. Focus (3) of this review.
**Suggestion**: Cut to 200–250 words with this structure:
- problem;
- design (one sentence);
- 3 key findings with one number each (RQ1 null with its minimum detectable effect; dual-stream > baselines; missing ≠ cloudy and the mask-error asymmetry);
- scope (one sentence);
- implication (one sentence).

Move the RQ2/RQ3/RQ5 details to the body.
**Severity**: Minor
**Confidence**: 5 — core editorial competence

### W9: Declarations are unfinished and the authorship is inconsistent
**Problem**: The byline lists one author. The CRediT statement names a second contributor (Soumen Mukherjee) and still contains editorial placeholders. The conflict-of-interest and funding statements carry "[author: confirm.]".
**Evidence Anchor**: `text: Declarations "[author: to be agreed with the mentor. Draft: Manish Y. M.: conceptualization,"`
**Why it matters**: A manuscript submitted in this state would be returned unreviewed.
**Suggestion**: Resolve the authorship (add the co-author to the byline or move them to Acknowledgements), finalise CRediT, COI and funding, and remove all bracketed author notes.
**Severity**: Minor
**Confidence**: 5 — editorial process

### W10: The paper admits reading only parts of the central comparator paper
**Problem**: RQ2 and RQ3 are positioned as tests of Song et al. [2], yet the text says only some sections of that paper were read.
**Evidence Anchor**: `text: §2 "The sections we read (§2, §5.1, §5.4 and Table 4) report five seeds and paired t-tests"`
**Why it matters**: Reviewers will question whether the comparison is characterised fairly. It also signals an incomplete literature review.
**Suggestion**: Read the full paper. Then either state its gap-experiment protocol as fact or say it is not reported. Remove "the sections we read".
**Severity**: Minor
**Confidence**: 5 — editorial

### W11: Contribution (iv) is not new
**Problem**: The finding that single-seed rankings can reverse is well established [17]. Showing it again in a superseded preliminary study with a different, re-implemented model is context, not a contribution.
**Evidence Anchor**: `text: §1 "(iv) A preliminary single-protocol study (Appendix A) showing that single-seed rankings on OSCD can reverse."`
**Why it matters**: It dilutes the contribution list and invites a "not novel" reaction.
**Suggestion**: Drop it as a contribution. Keep one sentence in §3 to motivate the 10-seed design, and move Appendix A to supplementary material.
**Severity**: Minor
**Confidence**: 4 — editorial

### W12: Too few figures for the claims; the null-result and timing results have no visual summary
**Problem**: There are 4 figures against 16 main-text tables. RQ1 (the headline null) and RQ2 (timing) are shown only as tables. Figures 2–4 point to experiment folders (`../experiments/robust/...`) rather than to paper figures.
**Evidence Anchor**: `absence: §4.2–§4.3 — expected a figure of per-seed or per-city Δ for RQ1 and F1 against acquisition timing for RQ2; checked §4, §5 figure list, Figures 1–4`
**Why it matters**: A forest-style plot of the contrasts would communicate "no detectable difference with this interval" faster than tables, and would make the paper shorter.
**Suggestion**: Add one forest plot for all primary contrasts across the RQs and one timing plot for RQ2. Convert several tables to supplementary material.
**Severity**: Minor
**Confidence**: 4 — editorial/presentation

---

## Detailed Comments *

### Journal Fit
- **Scope.** The topic (SAR–optical fusion, missing or cloudy optical input, Sentinel-1/2) is squarely within the scope of IEEE remote-sensing journals. No venue is bound (criteria_binding_unavailable), so I make no claim about a specific journal.
- **Format.** In its current form the manuscript is too long and too detailed for a letter. As a regular paper, it needs the restructuring in W5. If the authors split it, the §5 robustness study fits a regular-paper or JSTARS-style applied venue on its own. A condensed RQ1/RQ4 matched-protocol result could work as a letter.
- **Style.** Experiment-log terms (Phase/Step labels, commit hashes, file names) do not fit any IEEE journal's style. They belong in a reproducibility appendix or the repository.
- **Reviewer-directed text.** I found no text in the manuscript aimed at reviewers and no attempt at instruction injection. The AI-use statement, which says the same assistant ran the integrity checks and simulated peer review, is an appropriate disclosure, not a finding.

### Originality
- **New elements.**
  - A matched-protocol, multi-seed, pre-registered test of whether historical optical imagery helps SAR change detection on a public urban benchmark.
  - The separation of the missing and cloudy post-change cases, with mask-error asymmetry, in a change-detection setting.
- **Not new.**
  - RQ4 confirms Hafner et al. [1].
  - Modality dropout [11] and zero-fill are existing techniques.
  - Seed-variance ranking reversals are known [17].
- **Overall.** Originality lies in the evaluation design and the corrective result rather than in a method. That is legitimate, but the paper has to claim it explicitly (W3).

### Significance
- **The question I was asked to judge: does the message survive being mostly nulls on one small benchmark?** Partly.
  - **What survives.** The RQ1 null is informative because it overturns a previous single-protocol margin and comes with a stated minimum detectable effect. The §5 result is actionable.
  - **What does not survive.** RQ2 is uninterpretable for age (W4). RQ3 is a negative result specific to the SAR-only U-Net, with no mechanism. RQ5's single shrubland effect (1.3% of pixels) is too small to carry weight.
  - **The scope.** Generality is limited by the 9–10 test cities and by uncertainty that covers seeds only (W6).
- **What would raise significance.**
  - A per-city view.
  - A second benchmark or region, even small, for RQ1.
  - Or one pretrained or attention-based fusion model.

  These are recommendations, not requirements, if the paper is reframed honestly.

### Structural Coherence
- **Title.** It does not match the content (W1).
- **Introduction.** It answers all five RQs in advance, which is efficient but turns §4 into a confirmation.
- **Conclusion.** It addresses every RQ consistently with the results. It is careful and does not over-claim, except that the Abstract and §4.3 say "not reproduced" about Song et al. (W4).
- **§5.** It is coherent internally but structurally separate from the RQs (W5).

### Title & Abstract
- See W1 and W8. The keywords are good and could anchor a better title.

### Conclusion
- It is faithful to the results and explicit about nulls. The final practical recommendation (a dropout DS-UNet with a validation-selected threshold) is the paper's most useful sentence. It should appear in the Abstract and the contributions.

---

## Questions for Authors *
1. **RQ1 by city.** In how many of the 10 test cities is the per-city B@no_o2 − SAR-only difference positive (averaged over seeds)? Is the pooled null driven by one or two large cities?
2. **RQ2 with clean images.** Did any historical images strictly precede OSCD date 1 at all gaps for a subset of cities? If so, does the timing slope on that confound-free subset differ?
3. **Target format.** Do you intend a letter, a regular paper, or two papers? The required restructuring differs a lot between them (W5).
4. **Pretrained encoders.** Would a pretrained or attention-based fusion encoder, which might use optical context without a post-change image, change the RQ1 answer? If you cannot test this, can you argue for or against it from the Hafner and Ban [12] results?

---

## Minor Issues

### Language / Grammar
- §1: "word for word" and "RRD" should be removed (see W2).
- §4.3/§6: "acquisition timing" and "temporal gap" are used interchangeably. Use one term, "timing", given the confound.

### Citation Format
- References [20]–[24] are cited only in the appendices. If the appendices move to supplementary material, check the numbering.
- [17] has no DOI or page range.
- [15] is credited to "Sinergise EO research team (2024)". Check the preferred citation for s2cloudless.

### Figures and Tables
- Figure paths point into `experiments/robust/`. Render final paper figures with consistent styling and axis labels.
- The §4.1 performance table and the §5.1 table repeat the A/B full-input numbers. Keep one.
- The RQ2 table header "OSCD date 1 / Earth Engine date 1" needs a footnote for non-expert readers.

### Layout
- Commit hashes in Appendix D and in the text (`0e49c87`, `afa354e`): consolidate them in a single reproducibility appendix, or link them from the repository README.
- Appendix B reports results that "are not used". Delete it.

---

## Criterion-Bound Judgements *

Calibration status: `NOT_CALIBRATED`

criteria_binding_unavailable. Criterion source below is the ARS quality rubric dimensions as applied under Card #1, not any venue's criteria.

| Dimension | Criterion source | Judgement | Evidence anchor(s) | Rationale | Uncertainty / scope limit | Decision bearing? |
|---|---|---|---|---|---|---|
| Originality | ARS rubric; Card #1 focus 1 | PARTLY_MEETS | §1 Contributions (i)–(iv); S2; S3 | The matched-protocol null and the missing≠cloudy distinction are new for change detection. RQ4 and contribution (iv) are confirmatory or known. | Field-general; no systematic novelty search | yes: the reframing (W3) is needed for the originality to be visible |
| Methodological Rigor | ARS rubric | MEETS | §3.4; Appendix C; S1 | Pre-registered, audited, validation-only selection. RQ2's confound is acknowledged but limits that question. | Statistical depth outside my remit; deferred to the methodology seat | partly: W4, W6 |
| Evidence Sufficiency | ARS rubric | PARTLY_MEETS | §3.4 "sampling variability of the cities"; §4.3 confound | 9–10 cities, seed-only uncertainty; RQ2 not interpretable for age. | Moderate | yes: W4, W6 |
| Argument Coherence | ARS rubric; Card #1 focus 3 | PARTLY_MEETS | Title; §1 RRD framing; §5 | Conclusions follow the results. The title and the RRD framing do not fit the content; §5 is loosely tied to the RQs. | Low | yes: W1, W2 |
| Writing Quality | ARS rubric; Card #1 focus 2–3 | PARTLY_MEETS | Abstract (~305 words); Phase/Step labels | Precise and careful, but dense, process-ordered, too long, with internal jargon. | Low | yes: W5; W8 is minor |
| Literature Integration | ARS rubric | PARTLY_MEETS | §2; references [1]–[24] | Core works cited. Post-2022 missing-modality, cloud-robust and foundation-model work is absent. | No systematic search done | yes, moderately: W7 |
| Significance & Impact | ARS rubric; Card #1 focus 1 | PARTLY_MEETS | §4.2; §5.2–5.4; §8 | The corrective null and the cloud/mask results are useful. RQ2/RQ3/RQ5 add little; generality is limited to one small benchmark. | Editorial judgement | yes |

**Recommendation rationale.** The unresolved criteria that bear on the decision are framing and coherence (W1, W2, W3), structure and length (W5), RQ2 interpretability (W4), and the scope of the nulls across cities (W6). All are repairable without new training runs, except optionally W6, which needs only re-analysis of the stored counts, and W4(b). Methodological transparency does not offset these. The problem is not rigour but that the paper does not yet tell IEEE readers why its nulls matter.

---

## Recommendation: Major Revision

As a journal-fit editor, I see a rigorous and unusually honest study that is not yet a journal paper.

- **What a reader can use is already in the results.** Under a matched, pre-registered protocol, historical Sentinel-2 gives no detectable gain over bitemporal SAR on OSCD, which overturns an earlier single-protocol margin. A dropout-trained DS-UNet gives up nothing detectable on full input. A partly cloudy post-change image is worse than a missing one, and cloud masks should err toward over-masking.
- **What hides it:**
  - the paper is presented as answers to an internal requirements document;
  - the title is generic and over-broad;
  - the abstract is about 305 words of results;
  - two studies are reported in project chronology across about 9,500 main-text words.
- **Where the nulls fall short.** The null results can carry a paper, but only with their scope explicit. Uncertainty covers seeds only, not cities. RQ2 cannot speak to image age, and the claim that Song et al. is "not reproduced" should be withdrawn.
- **Why not Reject.** None of these defects invalidates the core findings, and most can be fixed by reframing, cutting or splitting, and re-analysing stored counts per city.
- **Why not Minor.** The required changes to title, framing, contribution statement, structure and length are substantial and need re-review.
