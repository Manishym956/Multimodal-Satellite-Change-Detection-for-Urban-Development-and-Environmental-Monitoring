# Errata for `PHASE3_REPORT.md`

`PHASE3_REPORT.md` is an audited, generated report and is left unchanged. Corrections are recorded here instead.

## 1. "Change prevalence" (found 2026-10-22, during the manuscript audit)

- **Where:** §2, second bullet: "whose F1 is ≈ 0.098 (the change prevalence)".
- **Correction:** 0.098 is the **F1 of a detector that labels every pixel as change**. The change prevalence on the test cities
  is **0.052**. That is the precision of the all-change detector: in the generated table (§8), A at `c100` has recall 1.000 and
  precision 0.052. The two values are consistent: F1 = 2p / (1 + p) = 2 × 0.052 / 1.052 ≈ 0.099.
- **Effect:** wording only. No number, test or conclusion in the report changes. The comparison it makes ("both arms are close to an
  everything-is-change detector") refers to F1, and the F1 value is correct.
- The same wording was not carried into `paper/ROBUSTNESS_PAPER_DRAFT.md`, which uses the corrected terms and lists this erratum in Appendix A.
