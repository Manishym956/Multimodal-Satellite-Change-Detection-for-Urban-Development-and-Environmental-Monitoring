# Phase 1 claim-to-evidence audit — FINAL_RESEARCH_PAPER.md (2026-11-12)

Scope: the manuscript `paper/FINAL_RESEARCH_PAPER.md` (the PDF is built from it; md 15:20, PDF 15:21 on 2026-11-11).
Automated checks run: `check_final_paper_numbers.py` (491/491 numbers found in sources) and
`check_final_paper_tables.py` (752 cells, 0 mismatches). The number checker only checks that each number occurs in a
source file, so the key claims below were also traced by hand.

Status codes: **V** verified against the named file; **V\*** verified, but the manuscript or the brief needs a change;
**U** unverified (needs literature or author evidence); **X** contradicted.

| # | Claim | Manuscript location | Evidence source | Status | Required action |
|---|---|---|---|---|---|
| 1 | Full-input F1: A 0.556±0.029, B 0.553±0.030, SAR 0.418±0.043, opt 0.498±0.056, fusion 0.495±0.026; A−baselines +0.138/+0.058/+0.061 | Abstract, §4.1 | `PHASE6_REPORT.md`; table checker | V | none |
| 2 | A's lead over SAR-only concentrated in Las Vegas (+0.049 without it); city+seed CI [−0.022, +0.246] | §4.1, §4.7 | `experiments/robust/stage4/`; table checker | V | Add a per-city table (Task B) |
| 3 | Polarisation (SAR-only VV 0.452, VH 0.383, VV+VH 0.408; DS-UNet n.s.; 9 cities) | §4.5 | `PHASE7_REPORT.md`; table checker | V | none |
| 4 | RQ2 slope +0.0083/yr [−0.0021, +0.0187], t p 0.105 | §4.4 | `PHASE8_REPORT.md:36-38,81` | **V\*** | **Selective reporting.** The exact Wilcoxon test on the same slopes gives p 0.049, with 8/10 slopes positive. The sensitivity analysis on all 10 cities gives p 0.057 and the async slope p 0.051; neither is in the paper. All must be reported, with the t-test kept as the registered decision |
| 5 | Stage 4B: B present − absent +0.007 [−0.004, +0.018], p 0.198 | Abstract, §4.2 Table 4 | `stage4b/STAGE4B_REPORT.md:9` | V | none |
| 6 | Stage 4B recall +0.018 [+0.006, +0.031] | §4.2 | `STAGE4B_REPORT.md:17` | V | The manuscript's "Holm p 0.030" is correct. **The brief's p 0.00977 is the Wilcoxon p, not the t-test or Holm p.** Keep the manuscript value |
| 7 | B margin +0.060 = +0.007 (image) + 0.053 (B without optical) | §4.2 Table 4 | Table 4 rows | V | Keep it described as descriptive arithmetic, not a causal attribution (already does) |
| 8 | Step 9: E present − absent −0.091 [−0.124, −0.058], 0/10, Holm 0.0003 | Abstract, §4.2 Table 5 | `step9/STEP9_REPORT.md`; table checker | V | none |
| 9 | Step 9 exploratory E@no_opt − SAR +0.072 [+0.045, +0.099] | §4.3 | number checker only | V (weak) | Trace to `step9/summary.json` in Phase 5 |
| 10 | Step 10: +0.072 [+0.038, +0.107] Holm 0.0011; thr 0.5 +0.109, Holm 1.4e-04; −0.116, Holm 2.2e-05 | §4.3 Table 6 | `step10/summary.json` (holm_p 0.001135, 1.369e-04, 2.18e-05) | V | none |
| 11 | Las Vegas "+0.157"; without LV +0.028; 5/10 cities same sign; LOCO +0.028 to +0.082 | §4.3 | `step10/STEP10_REPORT.md:22-33` | V\* | +0.157 is LV's **own within-city Δ**, not its "contribution" to the pooled value (the brief's wording is wrong; the manuscript is right). Also show that two of the five positive cities are tiny (Milano +0.005, Chongqing +0.002) |
| 12 | Step 11 partly cloudy (7 cities): +0.208, +0.020 (Holm 0.0242), +0.087; rb95 − discarded −0.007 (Holm 0.020); A − SAR −0.275 | Abstract, §5.5 Table 7 | `realcloud/STEP11_REPORT.md:7-21` | V | none |
| 13 | Step 11 all 10 cities: +0.232, +0.056, +0.053; overcast: +0.260, +0.091, +0.020 | §5.5 (only +0.056 and +0.091 quoted) | `STEP11_REPORT.md:23-36` | V\* | Report the all-10 and overcast families in a table, not in passing |
| 14 | "Discarding the cloudy image did at least as well as masking it" | Abstract, §1(iii), §6, §8 | Table 7 row 5 | **V\*** | Understated: discarding was detectably better (+0.007, Holm 0.020, 8/10 seeds). Reword to "slightly but detectably better" |
| 15 | "Where it can interact with SAR it hurts"; "the null in DS-UNet is therefore not an artefact of late fusion" | §1(i) | Step 9 (one early-fusion U-Net) | X (overgeneralised) | Limit to the one early-fusion design tested; replace "therefore not an artefact" with "late fusion alone does not explain the null in this comparison" |
| 16 | "The early-fusion experiment rules out the most obvious explanation" | §6 | as 15 | X (overclaim) | "argues against" |
| 17 | "No condition in which the historical image gives a material benefit" | §4.6, §8 | — | X (untested threshold) | No practical margin was pre-registered. State what was and was not detectable |
| 18 | "Have not been measured on a public change detection benchmark" | §1 | — | U | Literature check (Phase 2) |
| 19 | Song et al. [2]: 1/3/6/12-month gaps, F1 83.99% → 78.31%, no repeated runs, VH screening | §1, §2, §4.4 | — | U | Verify against the paper (Phase 2) |
| 20 | Hafner & Ban [12] F1 0.327 vs 0.233 vs 0.348 | §2 | — | U | Verify (Phase 2) |
| 21 | All other references [1]–[28] | References | — | U | Verify title, authors, year, venue, DOI (Phase 2) |
| 22 | Preprocessing: S2 13 bands /10000, clip [0,1]; S1 cubic resize to label grid | §3.1 | `src/datasets/oscd_dataset.py:6-8,93-118` | V | Add: OSCD `imgs_*_rect` bands used as supplied; labels = `cm` raster − 1; **no CRS transform and no co-registration step for S1** (resize only) |
| 23 | S1 GEE recipe: IW, one relative orbit, mean, VV [−25, 5] dB, VH [−30, 0] dB, unitScale, clamp, unmask | §3.1 (VH range only) | `download_gee.py:5-8,47,122-128` | V | Document fully. Note that `unmask()` turns empty windows into 0, which is why the Las Vegas VH band is empty |
| 24 | Recipe: Adam 5e-5, wd 5e-4, batch 64, 32 px crops, 2500 steps, eval every 25 | §3.2 | `src/training/train_rrd.py:122-125`, `train_robust.py:171-177` | V | Add: importance crop picks 1 of 20 random windows with probability ∝ (changed pixels + 5); augmentation = independent H/V flips (p 0.5 each), then **always** rot90 by k ∈ {1,2,3}; threshold grid 101 points; validation split = private `Random(42)` draw |
| 25 | Seeds: `random`, `numpy`, `torch` seeded; cuDNN deterministic | §3.2 | `train_sar_baseline.py:89-95` | V | none |
| 26 | `fig10_architecture_schematic.png` | not cited | figure file | X (if reused) | Shows the **preliminary re-implementation** (bottleneck concatenation, shared decoder, 2-ch SAR), **not** the official DS-UNet (separate U-Nets fused by a final 1×1 conv). It must not be used. A new diagram is needed |
| 27 | Fig. 1 forest plot | §3.4 | `fig14_forest.png` | V\* | Rows mix city subsets (10 / 9 / 7 cities) and threshold rules; put these in the row labels and the caption |
| 28 | Missing figures/tables | — | — | — | Workflow diagram, architecture diagram, per-city table, cloud-condition plot, cloud-mask examples |

## Items requiring author confirmation (not established by code)

1. Processing of the DS-UNet-released S1 VV rasters: exact 3-month windows, and the GRD preprocessing applied by Earth
   Engine (the code only shows `COPERNICUS/S1_GRD`, mean, unitScale).
2. How OSCD produced its `imgs_*_rect` bands (resampling to 10 m). The code consumes them as supplied.
3. Whether the S1 and S2 grids were ever checked for geometric alignment, beyond the resize to the label shape (the code
   shows a resize only).
4. Wall-clock training time per run (not recorded in the manuscript).
