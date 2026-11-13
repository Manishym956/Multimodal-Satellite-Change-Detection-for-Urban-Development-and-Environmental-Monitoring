# Phase 6 — final report of the scientific revision (2026-11-12)

## 1. Deliverables

| item | path |
|---|---|
| Revised manuscript | `paper/final_submission_v2.md`, `paper/final_submission_v2.pdf` (33 pages, A4) |
| Figures | `paper/figures_v2/fig01–fig07*.png`, script `paper/figures_v2/make_figures_v2.py` |
| Claim-to-evidence audit | `paper/review/PHASE1_CLAIM_AUDIT_2026-11-12.md` |
| Literature verification and comparison table | `paper/review/PHASE2_LITERATURE_2026-11-12.md`; Table 1 of the manuscript |
| Revision plan | `paper/review/PHASE3_REVISION_PLAN_2026-11-12.md` |
| Revision log | `paper/review/REVISION_LOG_v2_2026-11-12.md` |
| Checkers | `paper/check_final_submission_v2_numbers.py`, `paper/check_final_submission_v2_tables.py` |

Untouched: `paper/FINAL_RESEARCH_PAPER.md` / `.pdf`, `figures/paper/*`, the original checkers, all experiment files.

## 2. Quality-assurance results

| check | result |
|---|---|
| Every decimal number in the body found in a repository source (existence check) | 490/490 |
| Every results-table cell against summary JSONs (incl. new Tables 14, 20) | 778 cells, 0 mismatches |
| New prose numbers traced by hand (RQ2 Wilcoxon/sensitivity slopes; Step 11 −0.007 row; Step 10 per-city; Saha et al. figures) | traced to PHASE8_REPORT, stage4 E1, STEP11_REPORT, step10 summary, author's reading of [29] |
| References: DOI → Crossref title/authors/year | 29 DOIs resolve and match (2 Zenodo DOIs resolve via doi.org); [4] JSTOR URL, [15] software URL (version 1.7.3 = installed, per `realcloud/detector_outputs/detector_info.json`), [17] and [20] proceedings without DOI; [23] pages corrected |
| Cited claims vs source content | [1], [2], [12], [33] full text; [8], [9], [10], [29], [30], [31], [32] abstract (+ code for [29]); [29] tables per the author's reading |
| Fabricated references | none |
| Abstract ↔ results ↔ discussion ↔ conclusion | consistent on: +0.007 null, −0.091/−0.116, +0.072/+0.028/5 of 10, t p 0.105 vs Wilcoxon p 0.049, −0.275, +0.208, +0.020, discard better by 0.007 |
| Figures/tables cited in text | Figures 1–7 and Tables 1–20 each referenced outside their caption |
| Exploratory labels | Tables 13, 14, Figure-5 maps, per-city and LOCO statements, +0.026 and +0.022 marked exploratory |
| Overclaim scan ("material", "rules out", "at least as well", "not measured", "artefact", "novel", "generalis*") | none remaining in an unsupported sense |
| Rendered PDF | 7 figures with captions, no markdown residue; pages 1, 3, 15, 20 inspected visually |

## 3. Claims weakened or removed

1. "Whether a historical optical image adds change evidence … ha[s] not been measured on a public change detection
   benchmark" — **removed** (contradicted by Saha et al. [29]).
2. "Where it can interact with SAR it hurts" / "the null in DS-UNet is therefore not an artefact of late fusion" /
   "rules out the most obvious explanation" — narrowed to the one early-fusion design tested; "argues against".
3. "Discarding the cloudy image did at least as well as masking it" — corrected to "slightly but detectably better".
4. "No condition in which the historical image gives a material benefit" — replaced by what was and was not detectable.
5. RQ2 "no detectable linear trend" — kept as the registered result but now reported with the disagreeing Wilcoxon
   p 0.049 and two near-threshold slopes; reframed as timing sensitivity, not decay.
6. "Did not improve" (RQ1 answer, conclusion) → "did not detectably improve".
7. Step 10 "replicated" — qualified everywhere as training-seed replication on fixed cities, with the Las Vegas dependence.
8. §5 intro's "models use their frozen full-input threshold" — corrected (§5.1 uses per-condition thresholds).

## 4. New figures and tables

Figures 1 (workflow, schematic), 2 (architectures, schematic from code), 6 (real-cloud conditions), 7 (real scenes and
masks); Figure 3 (forest) revised labels; Figures 4–5 carried over with expanded captions. Tables 1 (literature), 14
(per-city differences), 20 (Step 11 overcast and all-city families); all other tables given numbered captions.

## 5. Methodological details still requiring author confirmation

Items 1–3 below were resolved as far as sources allow by the targeted methods audit (`METHODS_AUDIT_2026-11-12.md`);
the `[author: confirm]` markers in §3.1 were replaced by sourced statements and explicit "unknown" statements.

1. OSCD 20 m/60 m bands: documented as resampled to 10 m and cropped ([3], dataset README); **interpolation method
   unknown** (undocumented) — stated in §3.1 and §7.
2. Sentinel-1/Sentinel-2 co-registration: **none image-based** in [1], its code or ours; footprints agree (edges within
   0.94 S2 px, test cities); **content misregistration unquantified** — stated in §3.1 and §7.
3. Sentinel-1 windows: **verified** — 3 months before date 1 and after date 2 ([1] §II-A, DS-UNet notebook,
   bit-identical re-export).
4. Wall-clock training time per run (not reported).
5. For [29]: which polarisation the second SAR band holds, single acquisition vs composite, threshold rule, and whether
   Table I baselines are seed averages (stated as unverified in §2 and §6.2).

The declarations (`[author: …]` in CRediT, COI, funding) are out of scope and unchanged.

## 6. Summary of scientific improvements

- Novelty repositioned from "first measurement" to a controlled re-examination of [29]'s question, with an explicit list
  of what is not new and a dedicated discussion of the opposite-direction result.
- Statistical reporting made complete (no selective test reporting for RQ2) and its unit of replication explicit.
- Geographic dependence made visible (per-city table, pooling explanation, caveats at every "replicated").
- Cloud results separated into five input regimes, with evidence (Figure 7) that heavy masking approaches discarding.
- Methods made reproducible from the code, with unknowns flagged rather than inferred.
- Every table and figure captioned with comparison, metric, city set, uncertainty and caveats.

## 7. Remaining research limitations (not fixable by revision)

Fixed 10-city test geography; age × change-exposure confound in RQ2; unexplained disagreement with [29]; one scene per
city and no cloud ground truth in the real-cloud test; one recipe tuned on DS-UNet A and two fusion designs; no
land-cover change labels for false alarms; internal (repository) pre-registrations and same-assistant audits.
