# Multimodal Satellite Change Detection for Urban Development and Environmental Monitoring

Urban change detection from Sentinel-1 SAR and Sentinel-2 optical imagery on the Onera Satellite Change Detection
(OSCD) benchmark, with a focus on the common case where the post-change optical image is missing or cloudy.

The manuscript is [`paper/FINAL_RESEARCH_PAPER.md`](paper/FINAL_RESEARCH_PAPER.md)
([PDF](paper/FINAL_RESEARCH_PAPER.pdf)). It answers the five research questions of the Research Requirement Document
(`Research Requirement Document.pdf`) and reports a robustness study on missing and cloudy optical input.

## Research questions and findings

All results: official DS-UNet [Hafner et al. 2022] and U-Net baselines, 10 paired training seeds, thresholds chosen on
validation cities only, paired t-tests with Holm correction. "No detectable difference" means the test did not detect
one, not that there is none.

| RQ | Question | Finding |
| --- | --- | --- |
| RQ1 | Does historical optical imagery improve SAR-based detection when the post-change optical image is missing? | No detectable gain over a SAR-only U-Net (+0.013 modality-dropout DS-UNet, −0.016 DS-UNet trained on SAR + historical optical) |
| RQ2 | How does the temporal gap of the historical image affect performance? | No detectable effect over 1–18 months (slope +0.0083 F1/yr, p = 0.105); images after OSCD date 1 may already show the change, so this is not a clean test of image age |
| RQ3 | Do VV and VH polarisations differ? | SAR-only U-Net: adding VH to VV lowers F1 (−0.044); with optical input, no detectable difference |
| RQ4 | How does DS-UNet compare with SAR-only, optical-only and input-fusion U-Nets? | DS-UNet is best with full input (test F1 0.556 vs 0.418, 0.498, 0.495) |
| RQ5 | Where does historical optical information help most? | Lower false-alarm rate detectably only in shrubland (1.3% of pixels) |

Robustness (paper §5): modality dropout handles a missing post-change image but not a partly cloudy one; zero-filling
cloud pixels recovers most of the loss for the dropout model. Clouds were simulated or composited from other scenes, not
real cloudy acquisitions.

## How the study was run

Each step (Phases 2–5, Steps 6–8 in `experiments/robust/`) follows the same procedure:

1. A pre-registration (`PHASE*_PREREGISTRATION.md`) fixes hypotheses, analysis and a sha256 table of the code, committed
   before any test evaluation. `verify_prereg_hashes.py` re-checks the table.
2. Training and evaluation; anchors check that re-evaluation reproduces stored counts exactly.
3. An independent audit (`audit_p*.py` → `p*_audit.json`).
4. A report (`PHASE*_REPORT.md`) built from a template by `build_phase*_report.py`, which refuses to build if a stated
   claim is not supported by the results.

Deviations are listed in the paper's Appendix C. Two checkers tie the paper to the results:
`paper/check_final_paper_numbers.py` (every number traced to a source) and `paper/check_final_paper_tables.py` (every
table cell against the summary files).

## Data

- **OSCD**: 24 cities, bitemporal Sentinel-2 (13 bands), pixel-level change masks; 14 training / 10 test cities. Our
  validation split holds out 4 training cities (beihai, bercy, aguasclaras, paris).
- **Sentinel-1 VV**: the rasters released with DS-UNet.
- **Sentinel-1 VH, historical Sentinel-2, ESA WorldCover**: exported from Google Earth Engine with `download_gee.py`.
- **Sentinel-2 Cloud Mask Catalogue** (Francis et al. 2020): source of the cloud shapes used in the robustness study
  (paper §5).

Datasets, checkpoints and raw count arrays (`*.npz`) are not in Git. Their sha256 hashes are in
`experiments/robust/ARTIFACT_SHA256_*.txt`; checkpoints are available on request.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/datasets/`, `src/models/`, `src/training/`, `src/evaluation/` | Data loading, models, training (`train_robust.py`, `train_rrd.py`) and metrics |
| `eval_phase*.py`, `analyze_phase*.py`, `audit_p*.py`, `build_phase*_report.py` | Per-step evaluation, analysis, audit and report builders |
| `smoke_phase*.py` | Validation-only smoke tests run before each step |
| `make_error_maps.py` | Qualitative error maps (best, median and worst test city) |
| `experiments/robust/` | Pre-registrations, reports, audits, per-run results, summaries |
| `experiments/` (other folders) | Preliminary study (paper Appendix A): baselines and Transformer runs |
| `logs/robust/` | Run scripts (`p*_run.sh`) and their logs |
| `paper/` | Manuscript, PDF and the paper checkers |
| `figures/paper/` | Figures used by the manuscript |
| `docs/final/` | Research plan, data feasibility report and literature review |

## Setup

- Python 3.11, PyTorch 2.5.1+cu124, one NVIDIA RTX 4060 laptop GPU; `pip install -r requirements.txt` into `.venv`.
- The DS-UNet network and loss are loaded at run time from a clone of
  [SebastianHafner/DS_UNet](https://github.com/SebastianHafner/DS_UNet) at commit `8bfa4be`, expected at
  `../../repos/2/DS_UNet` relative to this folder.
- Earth Engine exports use a separate environment `.venv_gee`:
  `.venv_gee/Scripts/python.exe download_gee.py s1|s2|wc --project <gcp-project>`.

## Reproduction

Each step has a run script in `logs/robust/` that checks the smoke test and the pre-registered hashes, then runs
evaluation, analysis and audit, for example (Step 8):

```bash
bash logs/robust/p8_run.sh > logs/robust/p8_run.log 2>&1
.venv/Scripts/python.exe build_phase8_report.py
```

The Phase 1–2 commands are listed in `experiments/robust/REPORT.md` §8. The run scripts `cd` into this project's
absolute path on the original machine; edit that line before running elsewhere. Training is run with cuDNN determinism;
repeat training gave bit-identical weights where checked (paper §3.2).

To check the manuscript against the results:

```bash
.venv/Scripts/python.exe paper/check_final_paper_numbers.py
.venv/Scripts/python.exe paper/check_final_paper_tables.py
```

## Citation

Paper in preparation. DS-UNet: Hafner, S., Nascetti, A., Azizpour, H., & Ban, Y. (2022). Sentinel-1 and Sentinel-2 data
fusion for urban change detection using a dual stream U-Net. *IEEE GRSL, 19*, 4019907.
OSCD: Daudt, R. C., Le Saux, B., Boulch, A., & Gousseau, Y. (2018). *Proc. IEEE IGARSS*, 2115–2118.
