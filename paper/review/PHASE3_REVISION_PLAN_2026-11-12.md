# Phase 3 — prioritised revision plan (2026-11-12)

Target file for Phase 4: `paper/final_submission_v2.md` (new). `FINAL_RESEARCH_PAPER.md` is not modified.
Inputs: `PHASE1_CLAIM_AUDIT_2026-11-12.md`, `PHASE2_LITERATURE_2026-11-12.md` (updated with the author's reading of
Saha et al. [29]). Every new number will come from an existing repository file named below; no new training or test
evaluation is planned. Declarations, CRediT, funding and similar sections are copied unchanged (out of scope).

Priority: **P1** = needed for scientific correctness; **P2** = needed for completeness/reproducibility;
**P3** = presentation.

## 1. Critical scientific corrections (P1)

| # | Change | Where | Evidence | Audit/lit ref |
|---|---|---|---|---|
| C1 | Remove "have not been measured on a public change detection benchmark". Introduce Saha et al. [29] as the origin of the setting **and** of the optical-vs-SAR-only comparison on multi-sensor OSCD | §1, §2 | author's reading of [29] | P2 §2 |
| C2 | New paragraph (Discussion) on the **difference in direction from [29]**: they report the pre-change optical input raising F1 over SAR-only baselines; our closest analogue (async, trained only on `no_o2`) shows no detectable difference (−0.016, CI [−0.059, +0.026]), B none, E a loss. List candidate reasons as untested: split (12/2/10 vs 10/4/10), SAR product and temporal averaging, architectures and losses, seeds per baseline, thresholds, a stronger SAR-only baseline here. **No comparison of absolute F1 across studies** | §6 | PHASE6_REPORT, Table 4.2 | P2 §2 |
| C3 | Report the RQ2 Wilcoxon result next to the registered t-test: t p 0.105 vs exact Wilcoxon p 0.049, 8/10 seeds positive; all-cities sensitivity +0.0042/yr p 0.057; async slope +0.0161/yr p 0.051. Keep the t-test as the decision rule; say the sign is suggestive (positive, i.e. *older images not worse*), not established; note the largest-slope seed sensitivity (stage4 E1) | §4.4 | `PHASE8_REPORT.md:36-52,81-116`; `stage4/STAGE4_EXPLORATORY_REPORT.md:9,18` | A4 |
| C4 | "Discarding did at least as well as masking" → "discarding was slightly but detectably better (rb95 − discarded −0.007, 95% CI [−0.012, −0.002], Holm p 0.020)" | Abstract, §1, §5.5, §6, §8 | `STEP11_REPORT.md:17` | A14 |
| C5 | Remove generalisations from one early-fusion design: "where it can interact with SAR it hurts" → "in the one early-fusion U-Net we trained"; "the null in DS-UNet is therefore not an artefact of late fusion" → "late fusion alone does not explain the null in this comparison"; "rules out" → "argues against" | §1(i), §4.2, §6 | Step 9 | A15, A16 |
| C6 | Replace "material benefit" with what was and was not detectable (no detectable F1 gain at any timing or land-cover class; small detectable within-B effects: recall +0.018, tree-cover FAR −0.002) | §4.6, §8 | `STAGE4B_REPORT.md` | A17 |
| C7 | Reframe RQ2 as **acquisition-timing sensitivity**, not temporal decay: explain the age × change-exposure confound (at 1–12 months every image post-dates OSCD date 1, so it may already show part of the labelled change); state what an isolated design needs (change labels dated at each historical acquisition, or images all preceding date 1 with varying age) | §1 RQ2 wording, §4.4, §7 | §3.1, PHASE8 prereg | Task C |
| C8 | Make the seed-vs-geography distinction explicit wherever "replicated" appears: Step 10 replicates **training**, not cities; Las Vegas's within-city Δ is +0.157 and the pooled Δ without it is +0.028; 5/10 cities positive, two of them ≤ +0.005 | Abstract, §1(ii), §4.3, §6, §8 | `step10/STEP10_REPORT.md:22-33` | A11, Task B |
| C9 | Cloud section: separate the five input regimes explicitly (missing optical/zero-filled; unmasked cloudy; cloud-masked = zero-filled under mask; image discarded = `no_o2`; SAR-only model). State that rb95 masks 0.10–0.98 of clear control scenes, so "masked" partly *is* "discarded"; the experiment does not show that masking recovers reliable optical information | §5.5, §6 | `STEP11_REPORT.md`, `STEP11_PREREGISTRATION.md` | Task D |

## 2. Methodological completeness (P2) — new §3 text, code-traced

| # | Add | Evidence |
|---|---|---|
| M1 | OSCD: `imgs_1_rect`/`imgs_2_rect` bands used as supplied by OSCD; 13 bands /10000, clip [0,1]; labels = `cm` raster − 1 (binary); label grid = S2 10 m grid | `src/datasets/oscd_dataset.py:6-8,93-109` |
| M2 | Validation cities drawn once with a private `Random(42)`, checked in code | `train_sar_baseline.py:98-101` |
| M3 | S1 (released): DS-UNet rasters, no further scaling; cubic resize to label grid; **no CRS transform or co-registration step in our code** (stated as a limitation, alignment inherited from the release) | `oscd_dataset.py:8-13,116-119` |
| M4 | S1 (GEE, RQ3): `COPERNICUS/S1_GRD`, IW, one relative orbit, mean over window, `unitScale` VV [−25, 5] dB, VH [−30, 0] dB, clamp, `unmask()` → empty windows become 0 (explains Las Vegas VH) | `download_gee.py:5-8,47,122-128` |
| M5 | Input tensors: per date [SAR channels, 13 optical]; DS-UNet IN_CHANNELS 28 (VV) ; U-Net baselines channel subsets; E stacks VV + 13 bands per date at the input | `train_robust.py:65-72`; `train_rrd.py:79-89` |
| M6 | Missing-modality implementation: zeroing optical channels (`no_o2`, `no_opt`), `o2_copy` copies date-1 optical into date 2; sampled per training sample with the arm's probabilities | `train_robust.py:75-96` |
| M7 | Importance crop: 20 random 32×32 windows, one drawn with probability ∝ (changed pixels + 5) | `train_robust.py:99-111` |
| M8 | Augmentation: independent horizontal/vertical flips (p 0.5 each), then rot90 by k ∈ {1, 2, 3} (always applied, as in DS-UNet's RandomRotate) | `train_robust.py:114-123` |
| M9 | Loss `jaccard_like_loss` (DS-UNet); Adam 5e-5, wd 5e-4; batch 64 (cities drawn with replacement, random orbit per sample); 2500 steps; validation every 25 | `train_rrd.py:122-125`; `train_robust.py:201-221` |
| M10 | Checkpoint/threshold selection: pooled validation counts on 101 thresholds, first listed orbit per city; checkpoint = max mean best-threshold F1 over trained conditions; per-condition threshold frozen | `train_robust.py:126-163,234-242`; `train_rrd.py:173-182` |
| M11 | Seeding: `random`, `numpy`, `torch`, CUDA seeded; cuDNN deterministic, benchmark off; one `numpy` Generator per run drives city, crop, augmentation and condition draws | `train_sar_baseline.py:89-95`; `train_robust.py:186` |
| M12 | Metrics: pooled pixel counts over analysed cities (not city-averaged); state this once and in every caption | §3.4 (exists) |
| M13 | Items **requiring author confirmation** (marked `[author: confirm]` in the text, not guessed): exact 3-month windows and GRD preprocessing of the released S1 rasters; OSCD's resampling of `_rect` bands; any S1–S2 alignment check; run time | P1 audit |

## 3. Literature (P2)

| # | Change |
|---|---|
| L1 | Rewrite §2 into four paragraphs: SAR–optical CD with synchronous optical ([1],[5],[8],[10],[32]); **post-change optical absent / historical optical** ([29] first, then [2]); missing modalities and reconstruction ([11],[12],[30],[31]); clouds ([13],[14]–[16],[28],[33]) |
| L2 | Insert the literature comparison table (P2 §4, updated) as Table 1 |
| L3 | New novelty paragraph at the end of §1 (P2 §5): what [29], [2], [12] already established vs what this study adds; "no new architecture" |
| L4 | Song et al. caveats: single-run gap table; patch-level evaluation; non-monotonic IoU |
| L5 | Fix [23] pages (6394–6403); add [29]–[33]; do not cite the preprint [34] |

## 4. Statistical clarifications (P2)

| # | Change |
|---|---|
| S1 | One paragraph in §3.4 defining the unit of replication: seeds are paired training replicates on fixed cities; CIs describe training variability, not geographic generalisation; city resampling in §4.7 is exploratory |
| S2 | Make "no detectable difference ≠ equivalence" explicit at each null used in an argument (Stage 4B, B@no_o2, DS-UNet polarisation); report MDE once (1.00–1.22 SD of paired differences) |
| S3 | Label every result's status in tables/captions: registered primary / registered secondary / descriptive / exploratory; mark partial unblinding for Stage 4B and Step 10's hypothesis origin |
| S4 | Contrast sign convention stated once ("Δ = first − second; positive favours the first") and checked in every table (Table 7 rows and Fig. 1 already consistent; to be re-checked) |
| S5 | Practical vs statistical significance: e.g. Step 11 B-rb95 − SAR-only +0.020 is detectable but small; within-B recall +0.018 is small |
| S6 | Distinguish the threshold rules that change conclusions (B@no_o2: +0.013 at its own threshold vs +0.060 at the full-input threshold) in one place |

## 5. Presentation: figures and tables (P3) — all from existing files

| # | Item | Source | Status |
|---|---|---|---|
| F1 | **Workflow diagram** (data → arms → selection on validation → frozen test evaluation → pre-registered analysis), schematic, labelled as such | text of §3 | new (drawn, no data) |
| F2 | **Architecture diagram**: DS-UNet (two U-Nets, 1×1 fusion of final features), SAR-only/optical-only U-Net, early-fusion U-Net (E), with the dropout conditions `full` / `no_o2` / `no_opt`. Replaces the obsolete `fig10` (not used) | `train_robust.py`, `train_rrd.py`, DS-UNet code | new (drawn, no data) |
| T1 | Literature comparison table | P2 §4 | new |
| T2 | **Per-city table**: change-pixel share and per-city Δ for A − SAR-only, B@no_o2 − SAR-only, Step 10 E@no_opt − SAR-only, Step 11 B-rb95 − SAR-only; LOCO pooled Δ | `stage4/STAGE4_EXPLORATORY_REPORT.md` E2; `step10/STEP10_REPORT.md`; `realcloud/STEP11_REPORT.md` per-city | new |
| T3 | Step 11 all three city sets (7 cloudy, 3 overcast, all 10) in one table | `STEP11_REPORT.md:5-36` | new (replaces passing mentions) |
| F3 | Forest plot: add city subset (10/9/7) and threshold rule to row labels; caption states registered-primary status | `fig14_forest.png` script | revise |
| F4 | Timing plot: keep; caption gains t-test and Wilcoxon results and the confound | `fig15_timing.png` | caption only |
| F5 | **Cloud-condition plot**: F1 for A, B under clear / unmasked / bal / rb95 / discarded, SAR-only reference line, 7 cloudy cities, mean ± 95% CI over seeds | `realcloud/summary.json` | new |
| F6 | **Cloud-mask examples**: 2–3 cities, true-colour post-change scene, rb95 and bal masks; selection rule stated in caption (e.g. lowest, median, highest cloud fraction) | `realcloud/detector_inputs`, `detector_outputs` | new |
| F7 | Error maps: add city labels/legend inside the figure if missing (to check) | `fig13_error_maps.png` | check |

## 6. Text restructuring (P3)

- Title: current title is broader than the study (see decision D1).
- Abstract (≤ 250 words): setting credited to [29]; three main findings with CIs; geographic caveat; real-cloud result with
  "discarding slightly better".
- §1: shorten findings list; RQ2 renamed "acquisition timing"; novelty paragraph.
- §4–5: no number changes except additions listed above.
- §6 Discussion: answer the seven questions of Task H in order, including the [29] disagreement (C2).
- §7 Limitations: add — alignment inherited, not verified (M3/M13); disagreement with [29] unexplained; detector-defined
  cloud fractions; no cloud ground truth on real scenes; seeds ≠ cities; recipe tuned on A.
- Future work (short, limitation-linked): independent geography (another multi-sensor CD dataset); change labels dated at
  historical acquisitions; real cloud ground truth; land-cover-specific false-alarm labels.
- Appendices A, C, D, E: kept (see decision D2).

## 7. Cannot be fixed without new experiments or evidence

1. Geographic generalisation (needs other cities/datasets).
2. Isolating image age from change exposure (needs dated labels or a different image-selection design).
3. Explaining the direction disagreement with [29] (would need running both pipelines under one protocol).
4. Mechanism of the Step 10 gain (regularisation vs cross-modal supervision) — untested.
5. Cloud ground truth for the real scenes.
6. Author-only facts listed in M13.

## 8. Decisions for the author before Phase 4

- **D1 Title.** Keep "Multimodal Satellite Change Detection for Urban Development and Environmental Monitoring", or use a
  precise title, e.g. *"Does historical Sentinel-2 imagery help Sentinel-1 urban change detection without a post-change
  optical image? A controlled multi-seed re-examination on OSCD"*. Recommendation: precise title.
- **D2 Appendix A** (preliminary study, Transformers). Keep as is, shorten to one paragraph + Table A1, or drop.
  Recommendation: shorten; it supports only the seed-variance motivation.
- **D3 New figures F1/F2/F5/F6** are generated by new scripts in a new folder (`paper/figures_v2/`), leaving existing
  figures untouched. Recommendation: yes.
