# Manuscript audit: `ROBUSTNESS_PAPER_DRAFT.md` (2026-10-22)

A scientific review of the draft, done after its first commit (`a26adc1`), before any submission or sharing.
It asks whether each claim is supported by the audited results, and whether the paper states its evidence honestly.

## 1. Mechanical checks

| check | script | result |
|---|---|---|
| every decimal in the body occurs in an audited report or pre-registration | `check_robustness_paper_numbers.py` | 186 / 186 found |
| every cell of the five results tables equals its source value (right row, right column) | `check_robustness_paper_tables.py` | 167 cells, 0 mismatches |
| the table checker detects errors | mutation test: one Δ and one seed count altered in a copy | both caught; file restored |

Limits: the table checker covers §4 tables only. Numbers quoted in running text are covered by the occurrence check plus the
manual review in §2, not by a placement check.

## 2. Claims against hypotheses and results

| claim in the draft | source | verdict |
|---|---|---|
| Dropout (B) has no full-input cost | REPORT.md §6: Δ −0.003 at the validation-selected threshold; −0.086 at 0.5 | **revised**: qualified by the threshold rule in the abstract |
| At 30% cloud B is "close to" the all-change F1 | PHASE3_REPORT §2: B 0.191 vs 0.098 | **revised**: 0.191 is about twice 0.098; now "against 0.098" |
| Oracle zero-fill recovers 87–94% for B, 16–40% for A | PHASE4A_REPORT §7 recovery column | supported |
| Over-masking ≤ 0.010 F1, under-masking 0.090–0.250 | PHASE4ABIS_REPORT §2 | supported (B, 30% cloud, primary) |
| Step 4B: no detectable difference on all 7 primary conditions | PHASE4B_REPORT §2; pre-registration H1–H3 | supported; stated as not equivalence, with the MDE |
| "The cloud mask, not the fusion architecture, is the limiting factor" | one architecture tested | **revised**: limited to this architecture and simulated corruption |
| "Training-time augmentation has reached its limit" | only zero-fill augmentation tested | **revised**: "zero-fill augmentation did not add robustness"; cloud-valued training not tested |
| "Five pre-registered steps" | Step 4B pre-registration committed before training (`25fe13b`); earlier steps rest on file times | **revised**: internal pre-registration; timing evidence stated per step (§3.5) |
| Novelty | Hafner & Ban (IGARSS 2023) address partly missing optical data due to clouds and report that reconstruction beats zero-fill | **revised**: contributions stated as measurements; related work now describes [19] accurately |

## 3. Items added to limitations and deviations

- The arms use different checkpoint-selection rules by design (A: full; B and C: full + no_o2 + no_opt).
- The pre-registrations are internal repository files.
- Only OSCD and only zero-fill were studied.
- The Step 4A pre-registration also cites a smoke test with no log, not only Step 4A-bis.
- The Phase 3 report's "change prevalence" wording: erratum in `experiments/robust/PHASE3_REPORT_ERRATA.md`.

## 4. References

| ref | status |
|---|---|
| [19] Hafner & Ban 2023 | title, IGARSS 2023, pp. 6843–6846 and arXiv:2306.00640 confirmed (arXiv, DBLP); DOI still to add |
| [18] ModDrop | authors, title, TPAMI 2016 and arXiv:1501.00102 confirmed (HAL, arXiv); volume 38(8), pp. 1692–1706 not confirmed online, check on IEEE Xplore |
| [22] s2cloudless | the citable source is the 2017 Sentinel Hub blog post "Improving Cloud Detection with Machine Learning"; URL to add |
| [23] CMIX, Skakun et al. 2022 | RSE, DOI 10.1016/j.rse.2022.112990 confirmed via the s2cloudless package documentation; full author list, title and volume to check |
| [17] companion manuscript | unpublished; cite as a preprint or remove before submission |
| affiliation | to be supplied by the author |

## 5. Open items for the author

1. Supply the affiliation; finish the [verify] / [add …] reference items.
2. Decide how to cite the companion manuscript [17].
3. Decide whether the title should keep "Pre-Specified Analyses" or drop the pre-registration framing entirely.
4. A placement check for numbers quoted in running text would require tagging each one with its source. This was not done.
