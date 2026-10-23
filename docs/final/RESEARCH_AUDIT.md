# Research audit and experiment matrix (Phase 0)

State on 2026-10-23, branch `robust-phase4` at `f56ba71` (local only; `origin` has `main` only). Goal: one paper titled
"Multimodal Satellite Change Detection for Urban Development and Environmental Monitoring" answering RRD RQ1–RQ5.
This file records what exists; it changes no result. Numbers are quoted from the named source files.

## 1. Sources

| Source | Content | Status |
|---|---|---|
| `Research Requirement Document.pdf` | RQ1–RQ5, objectives, baselines, ablations, metrics (11 pages) | Governing document |
| `paper/RESEARCH_PAPER_FINAL.md` (companion) | Missing post-change optical cost; baselines; Transformers; SEN12-MT Scenario 2 | Draft; unpublished |
| `paper/ROBUSTNESS_PAPER_DRAFT.md` | Seven-step robustness study (Phases 2–5) | Draft, wording-revised 2026-10-23, uncommitted |
| `experiments/robust/*_REPORT.md`, `*_PREREGISTRATION.md` | Step reports and internal pre-registrations | Committed (`eead015` … `f56ba71`) |
| `artifacts/final_release_2026-10-15/` | RQ status, RQ2 decision memo, provenance, experiment registry, reference verification | Untracked |
| `datasets/DATASET_INVENTORY.md` | Data on disk; VH absent on the urban set | Untracked |

## 2. Shared data and split

- OSCD, 24 cities, Sentinel-2 L1C 13 bands (/10000, clipped to [0, 1]); dates per city in `dates.txt` (date 1: 2015-07 to
  2017-04; date 2: 2017-06 to 2018-04).
- SAR: Sentinel-1 **VV only**, the rasters released with DS-UNet (`datasets/oscd_sentinel1`), built by Hafner's GEE notebook
  (`repos/2/DS_UNet/sentinel1_download.ipynb`: S1_GRD IW, per relative orbit, 3-month means, [−25, 5] dB → [0, 1]).
  Georeferenced (EPSG:4326); OSCD optical rasters are not, so SAR–optical alignment is inherited from Hafner.
- Split (identical in both papers): train 10 (bordeaux, nantes, rennes, saclay_e, abudhabi, cupertino, pisa, hongkong,
  beirut, mumbai), validation 4 (aguasclaras, beihai, bercy, paris), test = the 10 official OSCD test cities.

## 3. Experiment matrix

### 3.1 Companion paper (`experiments/<name>[_seedNN]/`, `experiments/*_multiseed_summary.json`)

Common protocol: project re-implementations (not the official DS-UNet code), BCEWithLogitsLoss, Adam lr 1e-4, 50 epochs ×
64 crops per city, crop 32, **no augmentation**, checkpoint = best validation F1, **threshold fixed at 0.5**, seeds 42–46.
No pre-registration; integrity file `experiments/evidence_audit/multiseed_audit_integrity.json`. Result files are
modified or untracked in git (§5).

| Experiment | Model / input | Seeds | Test F1 (mean ± SD) | RRD use |
|---|---|---|---|---|
| sar_baseline | U-Net, S1 VV t1+t2 | 5 | 0.241 ± 0.101 | RQ4 SAR-only |
| optical_baseline | U-Net, S2 t1+t2 | 3 | 0.204 ± 0.055 | RQ4 optical-only |
| fusion_baseline | U-Net, input concatenation | 3 | 0.376 ± 0.022 | RQ4 conventional fusion |
| dual_stream_baseline | dual-stream re-implementation, full input | 5 | 0.459 ± 0.061 | RQ4 dual stream |
| asynchronous_dual_stream_clean | dual stream, no post-change S2 (trained so) | 5 | 0.309 ± 0.095 | RQ1/RQ4 async |
| asynchronous_dual_stream | as above, zero-filled variant | 3 | 0.297 ± 0.089 | RQ1 |
| transformer_cross/self_attention | 1.2–1.4M-parameter Transformers, pos_weight 41 | 3 | 0.275 ± 0.124 / 0.199 ± 0.048 | Exploratory (RRD §7.3 optional) |
| transformer_cross_attention_pw1 | as above, matched loss | 3 | 0.088 ± 0.012 | Exploratory |
| SEN12-MT Scenario 2 | one 21-month interval, 4 test sites, proxy labels | 1 | 0.08–0.11 | RQ2 feasibility only; provenance unresolved |

### 3.2 Robustness study (`experiments/robust/`; each step: pre-registration, anchors, analysis, independent audit, report)

Common protocol: official DS-UNet code (`repos/2/DS_UNet@8bfa4be`), Jaccard-like loss, Adam lr 5e-5, 2500 steps, batch 64,
importance crops, flips/rotations, validation every 25 steps, checkpoint and threshold chosen on validation only,
**10 paired seeds**; arms A (full-only), B (modality dropout), C (spatial dropout).

| Step | Question | Primary family | Audit | Commit |
|---|---|---|---|---|
| Phase 2 | Dropout vs full-only under missing optical | B − A, Holm over 4 | 579 / 0 failed | `eead015` |
| Phase 3 | Simulated opaque cloud | B − A, Holm over 4 | 167 / 0 | `eead015` |
| Step 4A | Oracle zero-fill | zero − cloud | none of its own (per-seed F1 re-checked by 4A-bis audit) | `eead015` |
| Step 4A-bis | Synthetic mask errors | imperfect − oracle, Holm over 9 | 1338 / 0 | `eead015` |
| Step 4B | Spatial-dropout training (C) | C − B, Holm over 7 | 432 / 0 | `25fe13b`, `5dfaa17` |
| Step 5S | Dilation of under-masking masks | Holm over 6 | 826 / 0 | `cde5434`, `3c5d85a`, `f56ba71` |
| Step 5 | s2cloudless on composite clouds | Holm over 9 | 731 / 0 | `d9e29ff`, `5f5be9c`, `3ed2fb7` |

Known gaps (disclosed in the draft): no smoke-test logs for 4A/4A-bis; 5S smoke script not committed; Phase 2 step-budget
sweep test F1 viewed before the budget was re-chosen on validation; 89 resumed Step 5 composites not byte-verified;
pre-registration timing for Phases 2–4A-bis rests on file times.

## 4. Can the companion results be reused in the combined paper?

**Not as primary evidence.** They are reproducible as recorded runs, but they are not comparable with the robustness
study: different model code, loss, learning rate, budget, augmentation, threshold rule (0.5 vs validation-selected) and
seed count, with no pre-registration. Mixing them in one comparison table would confound protocol with model. Under the
plan's rule (retrain only when the comparison is methodologically inconsistent), the RQ4 baselines must be re-run under
the robustness protocol. The companion numbers stay in the paper as a clearly labelled preliminary study (appendix),
together with the Transformer results, which have no counterpart and are exploratory. SEN12-MT Scenario 2 is reported
only as a feasibility note because its archive provenance is unresolved (`RQ2_DECISION_MEMO.md`).

The single-seed reversal recorded in the companion paper (seed 42 put async above sync; five seeds reversed it) is
kept as a methodological finding.

## 5. Git and environment state

- 116 changed or untracked paths. Companion experiment files (`experiments/<name>/…`) are modified relative to their
  last commit (`63305d7`) and the multi-seed folders are untracked; they need committing before they are cited.
- The robustness paper edits and checker extensions (2026-10-23) are uncommitted.
- Environments: `.venv` (training/evaluation, PyTorch 2.5.1+cu124, pinned `requirements.txt`), `.venv_cloud`
  (s2cloudless 1.7.3), `.venv_gee` (earthengine-api 1.4.3; created 2026-10-23, not yet authenticated).
- Hardware: one RTX 4060 laptop GPU; one robustness training run takes about 6 minutes.
- In progress (started 2026-10-23 18:41): `logs/robust/p6_parity.sh`, a code-equivalence check of the new
  `src/training/train_rrd.py` against `p2_dropout_seed1`. It trains on the training cities and compares weights; it
  produces no new test result. No Step 6 training has started.

## 6. Research-question status

| RQ | Current evidence | Status |
|---|---|---|
| RQ1 historical optical helps SAR without post-change optical | Companion: async 0.309 vs SAR-only 0.241 (n = 5, not significant after Holm). Robustness Phase 2: B with historical optical (`no_o2`) 0.431 vs B with no optical (`no_opt`) 0.440, descriptive only; **no evidence of a historical-optical benefit within B**; no SAR-only arm under the robustness protocol | Open: needs a matched SAR-only baseline (Step 6) |
| RQ2 temporal gap | One 21-month interval (SEN12-MT, preliminary) | Open: needs controlled multi-gap data |
| RQ3 VV/VH | None; VH not on disk | Open: needs Sentinel-1 VH |
| RQ4 baselines | Companion (different protocol); robustness A/B only | Partial: needs matched baselines |
| RQ5 conditions / pseudo-change | Cloud conditions only (robustness study) | Open: needs gap and land-cover stratification |
