# Stage 2.5 Integrity Report (pre-review)

- **Date:** 2026-11-02
- **Draft:** `paper/FINAL_RESEARCH_PAPER.md`, SHA-256 `4108bd34120e524cbb13a3de80500d5306818fb7174c0de6e2c3ff8cf6f14f45`, on top of commit `dfeba6c` with uncommitted edits
- **Mode:** Initial verification, run in the main session by one model. This is not an independent check.

## Verdict: FAIL

No reference is fabricated, and every number traced back to its source. The failure comes from one cited paper being misdescribed and from several sentences claiming more than the data shows. Two checklist modes (2 and 5) block until the author answers.

## Summary

| Check | Population | Result |
|---|---|---|
| Reference exists | 24 of 24 | 24 found: 18 by DOI in Crossref or Zenodo, 6 by Crossref or web search |
| Bibliographic details | 24 | 2 to fix or confirm, 1 title variant, several missing DOIs |
| Ghost citations | Body and appendices | 0 dangling, 0 orphan; [20]–[24] are cited only in the appendices |
| Citation context | [1], [2], [12], [16], [17] checked against sources | 1 MAJOR_DISTORTION, 1 MINOR_DISTORTION, 1 VERIFIED, 2 sources with full text not accessible |
| Number and table scripts | 440 numbers, 573 table cells | 0 unsourced, 0 mismatches |
| Prose numbers traced in context | About 60 high-risk values | All found in the expected report or pre-registration |
| Audit counts in the paper | 9 of 9 | All match the reports |
| Commit hashes in Appendix D | 19 of 19 | All exist, and their messages match the stated step |
| Internal consistency | Whole paper | 4 wording problems (M1–M4) |
| Originality | 2 phrase searches | **Far below the required 30%. Not a real originality check.** |

## Issues

### SERIOUS

**IL-SERIOUS-1: [12] is misdescribed (§2, line 106; MAJOR_DISTORTION)**

- **Paper says:** Hafner and Ban "report gains over zero-fill and SAR-only models."
- **Source says:** on missing-modality samples the model beats zero-fill (F1 0.327 vs 0.233) but falls short of SAR-only (0.327 vs 0.348). It beats SAR-only on multi-modal samples (0.435 vs 0.364) and across all samples (0.423 vs 0.362).
- **Why it matters:** this is the closest prior work, and the missing-optical case is this paper's topic. The source supports your RQ1 result, so the correct version helps you. The literature log shows only the abstract was read.
- **Proposed text:** "...by reconstructing optical features from SAR; on samples with missing optical data their model beats zero-fill (F1 0.327 against 0.233) but falls slightly short of a SAR-only U-Net (0.348)."
- **Source:** https://ar5iv.labs.arxiv.org/html/2306.00640

### MEDIUM

**M1: A null result is stated as a positive finding (§6, lines 572–574).** "It shows that it does not transfer to OSCD" contradicts §3.4, where a null is not evidence of absence.
Proposed: "we did not detect it on OSCD urban change under this protocol."

**M2: The cloud headline leaves out that the mask was perfect (Abstract, lines 33–34; §8, lines 651–652).** "Recovers most of the loss" holds only for the oracle mask (0.87–0.94). With the real detector on composites, retained gain is 0.35, 0.74 and 0.93 at 10%, 30% and 60% cover, and §5.7 says this comes partly from discarding the image.
Proposed abstract text: "zero-filling cloud pixels under a perfect mask recovers most of the loss for the dropout model, and a real detector on composite clouds recovers less at low cover."

**M3: "Recent images are the least useful" is not what the table shows (§1 lines 68–69; §4.3 lines 296–297; §6 lines 576–578).**
- Only the 1-month image scores low. The 3-month image scores highest of all for B (0.422, above the date-1 control at 0.420), although every 1–6 month image also post-dates date 1.
- B's F1 by gap is 0.389, 0.422, 0.409, 0.417, 0.412. Async zigzags: 0.349, 0.416, 0.370, 0.422, 0.383.
- The explanation (the image already shows the change) was not measured, as line 290 admits, and it doesn't readily explain the 3-month result. The §6 heading "Why the most recent image scores lowest" and the sentence "tells the model that the change was already there" present it as a mechanism.
- Other possible causes are not ruled out, such as the choice of individual scenes or a mismatch between frozen thresholds and new scenes.

Proposed: change "recent images" to "the 1-month image", note the 3-month result, and retitle the §6 paragraph "A possible reason for the low 1-month score", stated as an untested hypothesis. It would also be worth checking per-city scene dates and cloud fractions for the 1-month images; `PHASE8_REPORT.md` already lists them.

**M4: Equivalence wording for undetected differences (§4.1 lines 228–229; §6 line 566; §8 line 653).** "Matches A", "scores the same", "adds nothing measurable" and "only the dual-stream design gains from SAR" are stronger than the intervals allow (for example, input fusion minus optical has CI [−0.045, +0.039]).
Proposed: "not detectably different from", with the interval where one is reported.

**M5: Two claims about [1] that I could not confirm, one of them overstated (§2 line 90; §6 lines 587–588).**
- "They used VV only because VH was not available for early acquisitions": the abstract and the repository README don't mention polarisation, and the literature log lists only those two as read.
- "Agree with Hafner et al. [1]: SAR helps only when it is encoded separately": the README table has no optical-only U-Net row to compare against. "Only" also overstates your own result, whatever [1] says.
- The IEEE full text is needed. **Author action:** check it through college access. If the VV statement isn't in the paper, delete it or cite the repository.

**M6: Reference details.**
- [1]: article no. 4019907 is not confirmed. Crossref gives pp. 1–5.
- [9]: article no. 4405710 is not confirmed. Crossref gives pp. 1–10.
- [13]: published title is "Multisensor", not "Multi-sensor".

Check [1] and [9] on IEEE Xplore.

### MINOR

| ID | Location | Issue | Fix |
|---|---|---|---|
| IL-MINOR-1 | §2 line 112, [17] | "Can reverse model rankings on small benchmarks": the abstract covers variance from data sampling, initialisation and hyperparameters, and does not mention rankings or small benchmarks (MINOR_DISTORTION; abstract only) | "Variance from data sampling, initialisation and hyperparameters can change benchmark conclusions [17]; Appendix A shows a ranking reversal on OSCD." |
| IL-MINOR-2 | §5.4 line 478 | The formula applied to the table's means gives 0.25 for erode8 and 0.72 for miss10, but the table says 0.26 and 0.73. Probably a mean of per-seed ratios (not checked) | Say which one it is |
| IL-MINOR-3 | [4], [11], [23] | DOIs missing: 10.2307/4615733, 10.1109/TPAMI.2015.2461544, 10.1109/CVPR46437.2021.00633 | Add |
| IL-MINOR-4 | [15] | The author label "Sinergise EO research team" was not confirmed. Version 1.7.3 and its 2024 date were | Check the repository's citation details |
| IL-MINOR-5 | Lines 7–14 | Internal-draft note ("not for circulation") is still in the paper | Delete before submission |
| IL-MINOR-6 | References | APA-like format, but IEEE was requested. Items with more than six authors can stay as "et al." | Convert at Stage 5 |

## Notes (do not block)

- **[2] Song et al.:** the reference exists (Crossref), but MDPI refused access today (HTTP 403). The 83.99%→78.31% figures and §5.1/§5.4 details rest on the 2026-10 literature log. Open the article and confirm them, since RQ2 is framed around this result.
- **Abstract-only reads:** the literature log lists [8], [9] and [10] as read from the abstract, the same as [12]. Their sentences here are general and consistent with those abstracts, but the [12] error shows the risk.
- **Baseline number:** the repository confirms 0.600 for DS-UNet and 0.555 for input fusion (P 0.687, R 0.532). Your reproduced gain (0.061) is close to their 0.045.
- **Figures:** no figure trace exists for Figures 1–4 (trace-unavailable). All four image files exist.

## AI research failure-mode checklist

| Mode | Outcome | Evidence | Blocks? |
|---|---|---|---|
| 1 Implementation bug | CLEAR | 9 of 9 audit counts match, with 0 failures reported; evaluator bugs are disclosed in Appendix C. I read the recorded results and did not re-run the audits | No |
| 2 Hallucinated citation | **SUSPECTED** | All 24 references exist, but [12] is credited with a result it does not report (IL-SERIOUS-1) | **Yes** |
| 3 Hallucinated result | CLEAR | Scripts report no mismatches; run directories show 10 seeds per arm; prose numbers traced | No |
| 4 Shortcut reliance | INSUFFICIENT EVIDENCE | No ablation targets shortcuts directly. Flag only at this stage; reviewers will look at it | No |
| 5 Bug or artefact told as insight | **SUSPECTED** | The 1-month dip is given a mechanism that was never measured, and the 3-month score doesn't readily fit it (M3). No bug is shown | **Yes** |
| 6 Methodology fabrication | CLEAR on the sample | `p2_dropout_seed1` settings match §3.2 (2500 steps, lr 5e-5, wd 5e-4, batch 64, crop 32, validation every 25, torch 2.5.1+cu124); RQ1–RQ5 match the RRD word for word; test and validation cities were not re-checked here | No |
| 7 Frame-lock | INSUFFICIENT EVIDENCE | The RRD assumes a "proposed asynchronous approach", and the paper became a measurement study, openly. The author must judge this | No |

## Deviations from the plugin's protocol

- **C4 experiment provenance: FAIL by contract.** This run started mid-way, so there is no Material Passport with an experiment declaration or provenance block. `experiments/robust/` holds the equivalent evidence, but in a different form. I did not build a formal block.
- **Evidence rows (E3.1) were not built.** Sources were read through search and fetch summaries, so exact text wasn't held to bind rows to. The findings cite URLs instead.
- **Originality (Phase D) is under-sampled.** Only 2 phrase searches were run, against the required 30% of paragraphs, and web search is a weak test. Use iThenticate or Turnitin through your college before submitting.
- **Self-plagiarism (D2) was not run.** The memory notes show no earlier publication of this work, but I did not search for one.
- **Claim registry:** `stage2_5_claim_registry.json` holds only the 48 sentences the plugin's detector recognises, because its grammar misses plain decimals like 0.556. Its tier labels are defaults set by the script, not sentence-by-sentence judgements. Replay passes. The 24 unresolved candidates are all reference-list entries, not claims. Numeric sentences were covered by the checker scripts and my read-through, and semantic completeness is unknown.
- **E4 scope advisory skipped:** there is no RQ Brief. **E5 novelty:** I saw no "first to" claims in my read-through. **E6:** skipped, first pass.

## Verification trail

- **Crossref by DOI:** [1], [2], [3], [5], [6], [7], [8], [9], [10], [12], [13], [16], [18], [21], [22], [24]. **Crossref by title:** [11].
- **Zenodo API:** [14] record 4172871, [19] record 5571936.
- **Web search:** [4] (https://bibbase.org/network/publication/holm-asimplesequentiallyrejectivemultipletestprocedure-1979), [17] (https://proceedings.mlsys.org/paper_files/paper/2021/hash/0184b0cd3cfb185989f858a1d9f5c1eb-Abstract.html), [23] (https://openaccess.thecvf.com/content/CVPR2021/html/Van_Etten_The_Multi-Temporal_Urban_Development_SpaceNet_Dataset_CVPR_2021_paper.html), [15] version and date (https://pypi.org/project/s2cloudless).
- **[20]:** confirmed by web search as ICLR 2021 with the listed authors (https://arxiv.org/abs/2010.11929, https://iclr.cc/virtual/2021/oral/3458).
- **Source text:** [12] https://ar5iv.labs.arxiv.org/html/2306.00640; [1] https://github.com/SebastianHafner/DS_UNet; [17] https://arxiv.org/abs/2103.03098; [16] https://elib.dlr.de/187708.

> This originality check is not professional plagiarism detection, and coverage is limited to a very small sample of publicly searchable text.

## Round 2: re-verification after fixes (2026-11-02)

**Author decisions, in their words:** "1 yes, 2 yes-with disclosure, 3 i have no idea". Decision 1 approved the fixes. Decision 2 accepted the missing provenance block, with disclosure. Decision 3 left the source checks to the run.

**Draft now:** `paper/FINAL_RESEARCH_PAPER.md` SHA-256 `918cd92be24ab4e322b822c7441865059e658feb4ed4b10e59e593922aee4292`; PDF `7b5961f9a4cbd21a5ac3360b561d5024f03b7d206b65080efea0302d2015dcd9` (22 pages). A copy of the pre-fix text is not kept in the repository.

**Verdict: PASS WITH NOTES, on the corrected passages only.** Modes 2 and 5 are cleared. This round re-checked the changes and the sources below. It did not repeat the whole first-round check.

| Finding | Outcome | Evidence |
|---|---|---|
| IL-SERIOUS-1 [12] | Fixed in §2 and in `docs/final/LITERATURE_REVIEW.md` | Values from the arXiv text listed above |
| M1, M2, M3, M4 | Fixed in the Abstract, §1, §4.1, §4.3, §6 and §8 | Wording search finds none of the flagged phrases in these uses |
| M5 VV claim | Reason removed; now says the released rasters and notebook use VV | `sentinel1_download.ipynb` calls `orbit_mean(..., 'VV')`; the IEEE full text was not read |
| M5 "only" claim | Reworded as this study's result, with its interval | §4.1 values |
| M6 [1] | Article number replaced by pp. 1–5 | Crossref and OpenAlex |
| M6 [9] | Article no. 4405710 kept, now confirmed | https://elib.dlr.de/145752 |
| M6 [13] | Title corrected to "Multisensor" | Crossref |
| IL-MINOR-1, 2, 3, 5 | Fixed. The retained gain is per seed, as `PHASE4ABIS_PREREGISTRATION.md` states | Paper text |
| IL-MINOR-4 [15] | No change needed | PyPI lists "Sinergise EO research team" as the author |
| IL-MINOR-6 | Open until Stage 5 (IEEE reference format) | |
| [2] Song et al. | Now read from the publisher PDF. 83.99% to 78.31% and Table 4 are confirmed, and §5.1 supports the VH statement | https://mdpi-res.com/d_attachment/remotesensing/remotesensing-18-01103/article_deploy/remotesensing-18-01103.pdf |

**Added while reading [2]:** §2 said the sections read report no repeated runs. The method comparison does use five seeds and paired t-tests, while the gap results in Table 4 show no repeats or intervals. §2 now says this.

**Checkers after the fixes:** 442 of 442 numbers found in sources; 573 table cells, 0 mismatches. `LITERATURE_REVIEW.md` now holds the [12] numbers, which is why they count as sourced; the checker does not confirm them independently.

**Still open, not blocking Stage 3:**
- C4 provenance block: not produced. The author accepted this with disclosure, and the paper's "Use of AI tools" statement now says so.
- Originality: still only 2 phrase searches. A similarity check through the college is needed before submission.
- Self-plagiarism check (D2) and evidence rows (E3.1): not run.
- [1], [8] and [10] are still read from abstracts or the repository only.
- [Institution], CRediT, conflict of interest and funding: author decisions.
- This check is by the same model that made the fixes, so it is not independent.
