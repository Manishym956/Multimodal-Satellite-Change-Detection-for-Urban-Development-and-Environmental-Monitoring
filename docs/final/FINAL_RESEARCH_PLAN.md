# Final research plan: hypotheses, comparisons and decision rules

Title: **Multimodal Satellite Change Detection for Urban Development and Environmental Monitoring**. Written 2026-10-23,
before any test evaluation of Steps 6–8. Each step also gets its own pre-registration (hashes, smoke test, anchors,
independent audit), committed before its test evaluation, as in Steps 2–5. Inputs: `RESEARCH_AUDIT.md`,
`DATA_FEASIBILITY_REPORT.md`.

**Common protocol (all new steps):** official DS-UNet code, the robustness recipe (Jaccard loss, Adam 5e-5, 2500 steps,
augmentation), split train 10 / validation 4 / test 10, checkpoint and threshold chosen on validation only, 10 paired
seeds, test F1 at the validation-selected threshold, Holm within each family, paired t-test primary with Wilcoxon
alongside; Precision, Recall, F1, IoU, false-alarm rate reported. Labels: "X better" / "X worse" if Holm p < 0.05 with
that sign, otherwise "no detectable difference". A null is not equivalence; minimum detectable effects are registered.
Seeds are training replicates on one fixed set of test cities, not geographic replication.

**Existing evidence is not re-interpreted.** In Phase 2, B with the historical optical image (`no_o2`, 0.431) and B with
no optical image (`no_opt`, 0.440) were not compared by a registered test; this does **not** show that historical optical
helps B, and the paper will not say it does. RQ1 is decided by Step 6 and Step 8.

## RQ4: comparison with baselines (Step 6)

- **Arms:** SAR-only U-Net, optical-only U-Net, input-fusion U-Net (Hafner's "fusion" baseline), all trained on full
  input; DS-UNet A (existing); asynchronous DS-UNet trained only with SAR + historical optical (`no_o2`); B (existing).
- **Primary (Holm over 3):** A − each U-Net baseline, full input.
- **Support:** "DS-UNet better" vs a baseline. **Failure:** "no detectable difference" or "DS-UNet worse"; reported as is.
- **Companion results** (different protocol) go to an appendix, labelled as a preliminary study.

## RQ1: historical optical without post-change optical (Steps 6 and 8)

- **Primary (Step 6, Holm over 2):** B@no_o2 − SAR-only U-Net; async@no_o2 − SAR-only U-Net.
- **Support:** "SAR + historical optical better" for at least one. **Failure:** no detectable difference or worse,
  reported as "historical optical gives no measurable benefit over SAR alone under this protocol".
- Secondary: async@no_o2 − B@no_o2; B@no_o2 − B@no_opt (historical optical removed within B).
- Step 8 repeats the comparison at each temporal gap (RQ2/RQ5).

## RQ2: temporal gap (Step 8, evaluation only)

- **Inputs:** historical S2 L1C at g ∈ {1, 3, 6, 12, 18} months before OSCD date 2, plus OSCD date 1; same cities,
  labels, SAR and frozen checkpoints/thresholds (B, async, A); `no_o2` setting.
- **Primary:** per seed, the least-squares slope of B's test F1 against g (5 gaps); one-sample t-test of the 10 slopes
  against 0, two-sided.
- **Secondary (Holm over 5):** B at each g − SAR-only U-Net (does historical optical still help at that gap?); the same
  for async; A descriptive.
- **Decision:** a negative slope with p < 0.05 supports "older historical optical is less useful"; a positive slope means
  recent images hurt (the confound in the feasibility report: they may already show the change); otherwise "no
  detectable effect of the gap". The confound is reported with every RQ2 result.
- **Analysis set:** test cities with a clear scene at every gap (expected 9); 10-city sensitivity.

## RQ3: polarization (Step 7)

- **Data:** new GEE export of VV and VH with Hafner's recipe on Hafner's grid; VV pipeline check against Hafner's file.
- **Arms:** SAR-only U-Net and DS-UNet (full-only recipe), each with VV, VH and VV+VH from the new export; 10 seeds each.
- **Primary (Holm over 2):** SAR-only U-Net, VV+VH − VV and VH − VV, on the 9 test cities with VH at both dates.
- **Secondary:** the same for DS-UNet; false-alarm rate by land cover (RQ5).
- **Decision:** "VV+VH better" supports RQ3; otherwise reported as no detectable benefit of VH in this setting.

## RQ5: conditions where historical optical helps most (Step 8, descriptive with one tested family)

- **By gap:** from RQ2.
- **By land cover:** false-alarm rate and F1 per WorldCover class for SAR-only, B@no_o2 (native and each gap), B full.
- **Tested family (Holm over the classes with ≥ 1% of test pixels):** false-alarm rate, B@no_o2 − SAR-only, per class.
- **Restriction:** no labels separate pseudo-change from real change; claims are about where false alarms occur, not why.
- Qualitative error maps by a fixed selection rule.

## Cloud and missing-modality robustness (completed)

Phases 2–5 are preserved unchanged and become one section. Zero-fill is the tested strategy, not the best available;
conclusions are bounded to DS-UNet, OSCD and the tested corruption pipeline; all pre-registration exceptions stay
disclosed. A further robustness experiment (reconstruction baseline, validity channel or cloud-valued training) is chosen
only after the literature review, and at most one.

## Order of work

1. Step 6 (no new data): pre-registration → training (≈ 4 h) → analysis → audit → report.
2. GEE downloads (needs authentication) → Step 7 and Step 8 pre-registrations → runs → reports.
3. Error maps and figures; literature review and claim-to-evidence matrix (in parallel).
4. Final manuscript `paper/FINAL_RESEARCH_PAPER.md` and PDF; checkers extended to every new table.

## Definition of done

Every RQ has either a registered answer (support, failure or null) or a documented, evidence-based infeasibility; the
two papers are merged without changing completed results; every new number is generated from audited outputs.
