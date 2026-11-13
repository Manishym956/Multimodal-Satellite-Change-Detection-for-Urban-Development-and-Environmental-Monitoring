# Submission checklist

Audit of 2026-11-11. Manuscript `paper/FINAL_RESEARCH_PAPER.md`, SHA-256 `608c273a…2d82`; PDF
`paper/FINAL_RESEARCH_PAPER.pdf`, 25 pages, SHA-256 `8478d3bf…57e4`. Any later edit changes these hashes.

## Done

| item | evidence |
|---|---|
| Manuscript read in full and corrected | 17 wording and claim fixes after the rewrite (sign of the Stage 4B effect, overclaims in RQ4, Discussion and Conclusion, the mechanism of the early-fusion loss stated as a hypothesis, "Step 4B" vs "Stage 4B") |
| Every number traced to a stored result file | `paper/check_final_paper_numbers.py`: 491 of 491 |
| Every table cell checked against the summaries, labels against sign and p | `paper/check_final_paper_tables.py`: 752 cells, 0 mismatches |
| References | all 28 listed are cited, every citation is listed; 24 DOIs resolved and matched (title, year, first author); Holm (1979) has no DOI, now the JSTOR stable link; s2cloudless 1.7.3 checked on PyPI; Bouthillier et al. checked in the MLSys 2021 proceedings; Dosovitskiy et al. (ICLR 2021) has no DOI |
| Results committed | Stage 4B, Steps 9–11 outputs in commit `26da11f` (= `origin/main`) |
| PDF inspected | all pages rendered; no markup artefacts; figures and tables legible |
| Licence | `LICENSE` (GPL-2.0 text) and a GPL-2.0-or-later notice in `README.md`; stated in Data availability |
| Length | abstract 221 words; abstract to conclusion 5,587 words without table rows |

## Open (decisions)

| item | who | where to change |
|---|---|---|
| Mentor on byline or in acknowledgments | author and mentor | `DECLARATIONS_VARIANTS.md` → manuscript Declarations and byline |
| Conflict of interest, funding | author | same |
| Target journal | mentor | then apply the journal's template, reference style and limits, and fill in `COVER_LETTER_DRAFT.md` |

## Open (checks for the author)

- Click https://www.jstor.org/stable/4615733 once: the page blocks automated reading, so its identity with Holm (1979) is
  not machine-confirmed.
- Read the final PDF once yourself, especially the abstract, §4.2–4.3 and §5.5.
- If the journal limits figures or tables, Appendix E and Appendix A can move to supplementary material without changing
  the main text.
- Optional: deposit the count arrays and the real-cloud scene list on Zenodo and cite the DOI in Data availability.

## Files to upload (journal-independent)

- Manuscript: `paper/FINAL_RESEARCH_PAPER.md` (convert to the journal's LaTeX or Word template) and the PDF.
- Figures: `figures/paper/fig14_forest.png` (Fig. 1), `figures/paper/fig15_timing.png` (Fig. 2),
  `figures/paper/fig13_error_maps.png` (Fig. 3); all 300 dpi PNG.
- Cover letter: `paper/submission/COVER_LETTER_DRAFT.md`.

## Commit before submitting

Not yet committed: the revised manuscript and PDF, both checkers, `experiments/robust/revision/`, the two new figures,
`LICENSE`, the README licence section and `paper/submission/`.
