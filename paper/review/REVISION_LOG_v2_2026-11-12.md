# Revision log — `paper/final_submission_v2.md` (2026-11-12)

Base: `paper/FINAL_RESEARCH_PAPER.md` (2026-11-11 15:20, unchanged). No model was trained or evaluated for this revision.
Audit trail: `PHASE1_CLAIM_AUDIT_2026-11-12.md` → `PHASE2_LITERATURE_2026-11-12.md` → `PHASE3_REVISION_PLAN_2026-11-12.md`
→ this log → `PHASE6_FINAL_REPORT_2026-11-12.md`. Plan item IDs (C*, M*, L*, S*, F*, T*) refer to the Phase 3 plan.

| section | change | plan IDs | evidence |
|---|---|---|---|
| Title | new title (author-approved D1) | D1 | — |
| Abstract | rewritten: setting credited to [29]; "one early-fusion U-Net"; Las Vegas dependence; RQ2 t-test and Wilcoxon; "discarding slightly better than masking"; seeds ≠ geography | C1, C4, C5, C8, C3 | Stage 4B, Step 9–11, PHASE8 reports |
| §1 | novelty claim "not measured on a public benchmark" removed; [29] and [2] results stated; components explicitly not claimed as new; RQ2 reworded as timing sensitivity; findings (i)–(iv) narrowed | C1, C5, C7, L3 | Phase 2 report |
| §2 | restructured into four paragraphs; [29] paragraph with code-verified details and stated unverifiable items; Song et al. caveats; [30]–[33] added; Table 1 (literature comparison) | L1, L2, L4, L5 | Crossref, abstracts, Saha code repo, author's reading |
| §3 | Figures 1–2 (schematics); methods details traced to code (bands, labels, validation draw, S1 handling and absent co-registration, GEE recipe, real-scene resampling, channel counts, importance crops, augmentation, batch sampling, seeding, selection grid); `[author: confirm]` markers; Table 2 caption; async named as the setting-matched arm; §3.4 adds unit of replication, sign convention, Wilcoxon role, equivalence and practical-significance statements | M1–M13, S1–S5 | `oscd_dataset.py`, `train_robust.py`, `train_rrd.py`, `train_sar_baseline.py`, `download_gee.py`, `export_real_cloud_inputs.py`, DS-UNet `networks.py` |
| §4 | all tables captioned and numbered (3–14); "therefore not explained by late fusion" → "late fusion alone does not explain … one early-fusion design"; RQ1 answer notes the difference from [29]; Step 10 caveat (two of five positive cities ≤ +0.005; seed replication ≠ geography); §4.4 retitled, confound explained, Wilcoxon p 0.049, all-city and async slopes, largest-slope sensitivity; "material benefit" replaced; new Table 14 (per-city) and pooling explanation; in-text references to every table and figure | C3, C5, C6, C7, C8, S3, T2 | PHASE8_REPORT, stage4 E1/E2, STEP10_REPORT |
| §5 | five input regimes defined; §5.1 threshold exception stated (was inconsistent); Tables 15–20 captioned; "masking did not beat discarding" → "discarding slightly but detectably better (−0.007, Holm p 0.0200)"; mask-coverage explanation; new Table 20 (overcast, all 10); Figures 6–7; detector-defined cloud fractions caveat | C4, C9, T3, F5, F6 | STEP11_REPORT, realcloud summary.json, detector outputs, manifest |
| §6 | rewritten as 8 subsections answering Task H; dedicated §6.2 comparison with [29] (agreement, disagreement, differences listed without causal attribution, no cross-study F1 comparison) | C2 | Phase 2 report |
| §7 | specific limitations incl. unexplained disagreement with [29], inherited alignment, detector-defined cloud groups; five prioritised future-work items | — | — |
| §8 | concise conclusion; "did not detectably improve" | C6 | — |
| References | [23] pages fixed (6394–6403); [29]–[33] added | L5 | Crossref |
| Appendix A | reduced to one paragraph + Table A1 (author-approved D2); Tables A2–A3 removed (pointers to repository files kept) | D2 | `experiments/paper_statistics.json` |
| Appendix B | cross-reference corrected (§4.3 → §4.4) | — | — |
| Appendix C | item 19 records this revision | — | — |
| Appendix D | v2 checkers and figure script listed | — | — |
| Declarations | unchanged (out of scope) | — | — |

**Addendum — targeted methods audit (2026-11-12, after Phase 4; `METHODS_AUDIT_2026-11-12.md`).**

| section | change | evidence |
|---|---|---|
| §3.1 OSCD | `[author: confirm]` on band resampling replaced: bands documented as resampled to 10 m and cropped; interpolation stated as undocumented/unknown; OSCD's GeFolki optical–optical registration and the ungeoreferenced shared pixel grid stated | OSCD `images/README.txt`; [3] §2, §4; Medusa `sentinel_crop.py`; `methods_audit_footprints.json` |
| §3.1 Sentinel-1 | 3-month windows, GRD chain, −25 dB mask, orbit split, scaling and export grid stated from the source; `[author: confirm]` on alignment replaced by: no image-based co-registration in [1] or here; footprint edges within 0.94 S2 px; content alignment not evaluated | [1] §II-A (local full text); `sentinel1_download.ipynb`; `preprocessing.py`; `download_gee.py:110-111`; `PHASE7_REPORT.md`; footprint check |
| §7 Inputs | limitation rewritten: footprint matching, not co-registration; misregistration unquantified; OSCD interpolation undocumented | as above |
| Appendix C item 19 | audit recorded | — |
| checker | `methods_audit_footprints.json` added as a number source | — |

Figures (`paper/figures_v2/`, generated by `make_figures_v2.py`; originals in `figures/paper/` untouched):
fig01 workflow (schematic), fig02 architectures (schematic, from code), fig03 forest (row groups now state city set and
threshold rule), fig04 timing (regenerated, same data), fig05 error maps (copy of fig13), fig06 cloud conditions (new),
fig07 cloud masks (new; city rule lowest/median/highest cloud fixed before drawing). Obsolete `fig10_architecture_schematic.png`
(preliminary re-implementation) is not used.

Checks: `check_final_submission_v2_numbers.py` 490/490 numbers sourced; `check_final_submission_v2_tables.py` 778 cells,
0 mismatches (adds Tables 14 and 20; drops removed A2/A3). Original checkers still pass on the unchanged original.
