# Stage 4 Response Matrix and Gap Analysis

- **Date:** 2026-11-02
- **Scope:** Option A, as the author instructed: rewriting, plus re-analysis of stored results. No training, no new evaluation, and no change to any existing result, log, pre-registration, checkpoint or audit file.
- **Manuscript:** before `7220d00b…1bc8f`; after `f5c5bcfe…db8ef` (`paper/FINAL_RESEARCH_PAPER.md`). PDF after `4ce0ba93…c617`, 26 pages.
- **New analysis:** `experiments/robust/stage4/analyze_stage4.py`, which writes `STAGE4_EXPLORATORY_REPORT.md` and `stage4_summary.json`.
  - All of it is exploratory and not pre-registered.
  - Before computing anything new, the script asserts that it reproduces the audited values for Step 8 timing F1, Step 5 mask F1, the Step 6 SAR-only mean and the B@no_o2 − SAR-only contrast.
- **Checks:** numbers 459 of 459 sourced; tables 628 cells, 0 mismatches. The new §4.6 and §5.8 tables are checked against `stage4_summary.json`.

**Status key:**
- **Resolved:** fixed in text or by stored-data analysis.
- **Partly:** fixed as far as stored data allow, and the rest is stated as a limitation.
- **Open:** needs Option B, Option C or an author decision.

## Preregistered results kept unchanged

- The registered primary tests, labels and values are unchanged in every section:
  - RQ2 slope: p 0.105
  - RQ1: +0.013 and −0.016
  - RQ4: A versus baselines
  - RQ3 and RQ5 tables
  - all §5 tables
- Exploratory additions are marked "exploratory" or "unplanned" in the text. They are listed in §3.4 and in Appendix C item 12, and their limits are given in §7.
- **One registered secondary contrast is now given more weight:** B with versus without the historical image at each input's own threshold (Step 6 secondary family). It was already pre-registered and is not a new test.

## Journal-Fit Reviewer (EIC)

| ID | Criticism | Change made | Evidence | Status |
|---|---|---|---|---|
| W1 | Title over-broad ("Environmental Monitoring") | Title replaced with the one the author gave; §7 adds a "Scope" item saying environmental change was not evaluated | Title; §7 | Resolved |
| W2 | Framed around the internal requirements document | Every reference to it removed. The questions are introduced as "fixed before the experiments", and §2 now places the setting against heterogeneous change detection | §1; §2; §5 intro; §7 | Resolved |
| W3 | Contributions don't say why the nulls matter | Contributions rewritten as four scoped findings, including the within-model null, the timing confound and the city dependence | §1 | Resolved |
| W4 | RQ2 can't isolate image age, yet says "not reproduced" | "Not reproduced" removed. RQ2 is stated as acquisition timing, with no detectable linear trend and an exploratory 1-month low. The Song comparison is reframed: the image has a different role, so this is not a test of their finding | §2; §4.3; §8 | Resolved |
| W5 | Length and project-history structure | Repeat-training checks and audit counts moved to Appendix D, and Phase/Step labels are explained as pre-registered steps (§3.4). **Main text grew from about 9,450 to about 11,000 words because of the new analyses** | Word count | **Partly** |
| W6 | Uncertainty covers seeds only | New §4.6: city-then-seed bootstrap, leave-one-city-out, per-city Δ, and SD across cities against SD across seeds | §4.6; STAGE4 report E2 | Resolved, except optical-only and fusion, which have no stored per-city counts |
| W7 | Thin recent literature | Added heterogeneous change detection [25], [26], the BRIGHT dataset [27] and SEN12MS-CR-TS [28], all checked in Crossref. **Recent missing-modality learning work was not added** | §2; References | Partly |
| W8 | Abstract too long (about 305 words) | Rewritten at 248 words, with narrower claims | Abstract | Resolved |
| W9 | Declarations unfinished; author mismatch | Institution added. **CRediT, conflict of interest, funding and the mentor's byline position are still author decisions** | Declarations | **Open (author)** |
| W10 | Only some sections of Song read | Full text read in Stage 2.5. "The sections we read" removed, and §2 now states their seed and t-test practice | §2 | Resolved |
| W11 | Contribution (iv) not new | Dropped as a contribution; [17] is now cited as motivation in §1 | §1 | Resolved |
| W12 | Too few figures | **No forest plot or timing plot added yet.** Two new tables were added instead | — | **Open** |

## Reviewer 1 (Methodology)

| ID | Criticism | Change made | Evidence | Status |
|---|---|---|---|---|
| W1 | RQ2 "no effect" rests on a linear test | Registered slope kept and relabelled "no detectable linear trend". Added an exploratory Friedman test, repeated-measures ANOVA and Holm-corrected 1-month contrasts, and named seed 7 | §4.3; STAGE4 E1 (B Friedman p 0.00062; async 5.3e-06) | Resolved |
| W2 | Scenes are the replication unit; 12-month images also post-date date 1 | "1–6" corrected to "1–12" throughout. One scene per city and timing is stated, and seasonal offsets are reported | §3.1; §4.3; §7; STAGE4 E1 scene table | Resolved. Extra scenes per timing would need Option B |
| W3 | No uncertainty from test cities; claims stated unconditionally | §4.6 added. Abstract, RQ4 and conclusion now condition on cities ("lead over SAR-only … uncertain", 95% CI [−0.022, +0.246]) | §4.6; Abstract; §8 | Resolved for 5 contrasts (optical-only and fusion have no per-city counts) |
| W4 | Step budget tuned only on A | Stated in §4.1 and §7. Threshold-edge counts reported (optical 8/30, fusion 6/30, A 6/30) | STAGE4 E7 | **Partly**: per-baseline re-tuning is Option C |
| W5 | The 0.035 bound is wrong and MDE statements are inconsistent | §5.1 now says losses up to 0.037 are compatible. §3.4 lists 1.00, 1.14 and 1.33 SD with their α levels | §3.4; §5.1 | Resolved |
| W6 | Basis for pairing across architectures not stated | §3.2 states that the shared seed drives the sampling generator in both training scripts, while different architectures get different initial weights | `train_robust.py:186`, `train_rrd.py:133` (`default_rng(args.seed)`) | Resolved (described). The suggested Welch sensitivity check was not added |
| W7 | RQ5 lacks an operating-point check; multiplicity | Shrubland recall added (B 0.146 against SAR-only 0.151), and the attribution is limited to "two networks" | §4.5; STAGE4 E6 | Partly: total test count across families not tabulated |
| W8 | Abstract says "an independent audit for every step" | Removed from the abstract. §3.4 now says "a separate audit script, written with the same AI assistant … every step except Step 4A" | Abstract; §3.4 | Resolved |
| W9 | Raw counts not public | Not changed (repository action) | — | **Open** (Zenodo deposit) |
| W10 | Inconsistent statistics columns; "one primary family" | "One primary family" corrected (Step 6 has two). **Column completeness in older tables not changed** | §3.4 | Partly |

## Reviewer 2 (Domain)

| ID | Criticism | Change made | Evidence | Status |
|---|---|---|---|---|
| W1 | Song comparison framed as a replication when the image role differs | §2 explains the prior-for-target-date role against the pre-change-slot role and the opposite expected sign. "Not reproduced" removed | §2; §4.3 | Resolved |
| W2 | Heterogeneous change detection literature missing; "all assume both modalities" | New sentences with [25]–[27] and the explicit difference of our setting | §2 | Resolved |
| W3 | Sentinel-1 scene asymmetry between dates undisclosed | Disclosed from the feasibility inventory over the same windows: date 1 has fewer scenes in 22 of 23 windows, median 6 against 14, and Dubai orbit 130 has one VH scene. The RQ3 alternative explanation is stated as untested | §3.1; §4.4; §6; §7; STAGE4 E8 | **Partly**: the export logs record no counts, and the matched control is Option B |
| W4 | RQ5 attributed to the image across architectures | Reworded as "B against the SAR-only U-Net … cannot attribute to the image itself"; recall added | §4.5; §7 | **Partly**: the within-B by-class contrast is Option B |
| W5 | Conclusions not scoped to 3-month mean SAR | Added to the RQ1 answer, RQ3 answer, §8 and §6 | §4.2; §4.4; §8 | Resolved |
| W6 | Seasonality ignored in RQ2 | Mean seasonal offsets per timing reported; no clear relation to F1 | §4.3; STAGE4 E1 | Resolved (descriptive) |
| W7 | Orbit handling and geometry | Not changed | — | Open (Option B) |
| W8 | Composite details (dB averaging, Mumbai window) | Not changed | — | Open (needs a check of the export code) |
| W9 | Missing references | [28] added. Others (urban SAR change detection, CloudSEN12) not added | References | Partly |
| W10 | Title and requirements-document framing | As EIC W1 and W2 | — | Resolved |

## Reviewer 3 (Perspective)

| ID | Criticism | Change made | Evidence | Status |
|---|---|---|---|---|
| W1 | Title claims environmental monitoring | New title; "Scope" limitation added | Title; §7 | Resolved |
| W2 | No real cloudy acquisitions | Stated as a limitation, unchanged | §7 | **Open (Option B)** |
| W3 | No comparison with a SAR-only fallback rule | New §5.8 with exploratory per-seed contrasts. Without a mask, B is below the fallback at 30% and 60% cover; with rb95, B is above it at every level. Much of the margin comes from the threshold rule, as stated | §5.8; STAGE4 E5 | Resolved (composite clouds only) |
| W4 | Threshold transfer to new regions | Threshold-rule sensitivity added in §4.2 and §6. Per-city oracle thresholds not computed | §4.2; §6 | Partly |
| W5 | Zero-fill used as the missing-data token | Already stated in §5 ("not an intrinsic missing-data token"); no change | §5 | Partly |
| W6 | Pixel F1 against agency statistics | Not changed | — | Open |
| W7 | Repository reuse (LICENSE, `predict.py`, requirements) | Not changed (repository action) | — | **Open** |
| W8 | Placeholders | As EIC W9 | — | Open (author) |
| W9 | L1C against L2A | Not changed | — | Open (one-sentence fix possible) |

## Devil's Advocate

| ID | Criticism | Change made | Evidence | Status |
|---|---|---|---|---|
| **C1** | RQ2 "no detectable effect" contradicted by the data | As R1 W1. Every "no detectable effect" for timing is reworded in the Abstract, §1, §4.3, §6 and §8. A text search finds no remaining instance | §4.3; STAGE4 E1 | **Resolved** |
| M1 | RQ1 depends on the threshold rule | §4.2 "Threshold rule (exploratory)": +0.013 registered, +0.060 at the clean threshold (city bootstrap [+0.007, +0.117]), +0.018 at 0.5. The within-B contrast is now the main RQ1 evidence | §4.2; §4.6; STAGE4 E2, E3 | **Partly**: within-B at the clean threshold needs a new evaluation (Option B) |
| M2 | Late-fusion architecture makes the null close to built in | Architecture confirmed in code (`networks.py`: streams concatenated, then `OutConv`). Stated in §3.2, the RQ1 answer, §6 and §7; the RQ1 claim is scoped to this architecture | `repos/2/DS_UNet/utils/networks.py` | **Partly**: an early-fusion arm is Option C |
| M3 | Gap images are mid-interval, so "not reproduced" doesn't follow | As R1 W2 and R2 W1 | §2; §3.1; §4.3 | Resolved |
| M4 | RQ4 answered with the synchronous A | "The asynchronous approach (exploratory)": B and async against all baselines, Holm over 8. The RQ4 answer now leads with the asynchronous result | §4.1; STAGE4 E4 | Resolved |
| M5 | Capacity and recipe confound in dual-stream superiority | Recipe asymmetry stated (§4.1, §7). Not tested | — | Partly (Option C) |
| M6 | RQ5 attribution | As R2 W4 | — | Partly |
| M7 | "Most practical" reads a null as equivalence; no switching policy | "Most practical" removed. B's lower precision and higher false-alarm rate on full input reported; switching rule in §5.8 | §4.1; §5.8; §8 | Resolved |
| M8 | Seed-only inference | As R1 W3 | §4.6 | Resolved (5 contrasts) |
| M9 | VH result may be an operating-point effect | RQ3 answer now says "at validation-selected thresholds"; the scene-count explanation added. No threshold-free metric | §4.4 | Partly |
| M10 | Cloud baseline is a straw man; image-level discard | §5.8 compares rb99 (near-discard) with the SAR-only fallback and attributes most of the margin to the threshold rule | §5.8 | Partly: an image-level discard policy driven by detected cloud fraction was not computed |
| m1 | Audit wording | As R1 W8 | — | Resolved |
| m2 | Frozen-threshold shift for gap images | Mentioned as a possible cause (§6). Not tested | — | Open |
| m3 | Placeholders | Author decision | — | Open (author) |
| m4 | Requirements-document framing | Removed | — | Resolved |
| m5 | Dilation radius at grid edge | Already disclosed; unchanged | — | Unchanged |
| m6 | Novelty of contribution (iii) | Contribution (iii) restated with the fallback comparison. Not positioned against standard practice in detail | §1 | Partly |

## Gap analysis: what Option A could not resolve

| Gap | Why stored data can't answer it | What would resolve it | Option |
|---|---|---|---|
| B with versus without the historical image at the frozen clean threshold (DA M1) | `no_opt` counts exist only at B's own validation threshold and at 0.5 | Re-evaluate B checkpoints on `no_opt`, saving all 101 thresholds | B |
| RQ5 attributed to the image itself (R2 W4, DA M6) | No per-class counts for B without the historical image | Re-evaluate B on `no_opt` with WorldCover class counts | B |
| Real cloudy acquisitions (R3 W2) | None evaluated | Export a near-date-2 partly cloudy scene per city; evaluate A, B and SAR-only with and without an s2cloudless mask | B |
| Sentinel-1 scene-count control (R2 W3) | Export logs hold no counts; no matched-count data | Re-export with matched scene counts, or drop low-count cities, then evaluate | B |
| Second scene per timing (R1 W2) | One scene per city per timing | Export and evaluate a second scene within ±45 days | B |
| Optical-only and fusion city resampling (R1 W3) | Only pooled counts stored | Re-evaluate these checkpoints per city | B |
| Baseline tuning (R1 W4, DA M5) | Requires training | Per-baseline step budget on validation cities | C |
| Early-fusion historical-image arm (DA M2) | Requires training | Train a stacked-input U-Net with SAR at both dates and the historical image | C |
| Figures (EIC W12) | None needed from new data | Forest plot of primary contrasts; timing plot | A (not done yet) |
| Length (EIC W5) | Editorial | Move §5.4–5.6 detail and Appendix A to supplementary material | A (author choice) |
| Repository: LICENSE, `predict.py`, requirements, Zenodo counts (R1 W9, R3 W7) | Outside the manuscript | Repository changes and a Zenodo deposit | Author |
| Declarations and authorship (EIC W9) | Author decision | — | Author |

Under the project rule, each Option B item needs a committed pre-registration before its test evaluation.

## Post-check corrections (2026-11-02, after an independent check of the city bootstrap)

A separate session of the same AI assistant (fresh context, same model family, so not independent in error process)
re-derived Stage 4 E2 with its own code (`scratchpad/bootcheck/check.py`). All numbers reproduced within Monte Carlo error.
Corrections made:

| finding | change | where |
|---|---|---|
| "SD across cities" (unweighted per-city Δ) and "SD across seeds" (pooled Δ) are not comparable | Replaced by SEs of the same pooled Δ under seed and city (jackknife) resampling; "cities ≥ seeds" claim corrected (false for B at its own threshold) | §4.6 table and text; analyze_stage4.py (new fields, old values unchanged) |
| LOCO p-values hold the other cities fixed and say nothing about city sampling | LOCO p column replaced by the range of the pooled Δ with one city left out; Las Vegas drives the A lead (+0.138 → +0.049) | §4.6 |
| "comes mainly from Las Vegas, Dubai and Brasilia" wrong for Dubai and Brasilia | "concentrated in Las Vegas" | §4.6, §7, §8 |
| Clean-threshold B margin percentile CI [+0.007, +0.117] narrowly excludes 0; jackknife-t [−0.030, +0.150] does not | Both reported; "not established across cities" | §4.2, §4.6 |
| Overstated framing of contribution (iv), abstract, §4.1, §3.4 | Labelled exploratory; method details (percentile, paired indices, pixel pooling, unweighted signs) added | Abstract, §1, §3.4, §4.1 |
| (own re-read) "within-model contrast attributes none of this to the image" — the within-B contrast was never measured at the clean threshold | Reworded: unmeasured at that threshold; this is what Stage 4B tests | §4.2 (two places) |
