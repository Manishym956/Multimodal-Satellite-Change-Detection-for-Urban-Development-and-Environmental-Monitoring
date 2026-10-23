# Mentor brief: status and decisions needed

**Project:** Multimodal Satellite Change Detection for Urban Development and Environmental Monitoring
**Student:** Manish Y M · **Mentor:** Soumen Mukherjee · **Date:** 2026-10-23 · *Internal, not for circulation*

## 1. Where the project stands

Two drafts exist. A **companion paper** measured what losing the post-change optical image costs (baselines,
dual-stream, Transformers; 3–5 seeds). A **robustness study** (seven pre-registered, audited steps, 10 seeds each)
studied missing and cloud-covered optical images with the official DS-UNet. The plan is to merge both into one paper
under the title above, organised by the five research questions of the Research Requirement Document (RRD).

## 2. Research-question status

| RQ | What exists | What is being done | Status |
|---|---|---|---|
| RQ1 historical optical vs SAR alone | Companion: async 0.309 vs SAR-only 0.241 (not significant). Robustness: no SAR-only arm; no evidence that historical optical helps | Step 6 (running): SAR-only baseline under the shared protocol; async vs SAR-only, registered test | In progress |
| RQ2 temporal gap | One 21-month interval on an archive with unresolved provenance | Step 8: historical Sentinel-2 at 1, 3, 6, 12, 18 months before the target date, same OSCD cities and labels at every gap | Data feasible; needs Earth Engine access |
| RQ3 VV / VH / VV+VH | VV only | Step 7: new Sentinel-1 VV+VH export with Hafner's recipe; 9 test cities (Las Vegas has no VH at date 1) | Data feasible; needs Earth Engine access |
| RQ4 baselines | Companion (different protocol); robustness: DS-UNet only | Step 6 (running): SAR-only, optical-only, input-fusion U-Nets, async DS-UNet, 10 seeds | In progress |
| RQ5 conditions / pseudo-change | Cloud conditions only | Step 8: false alarms by land cover (ESA WorldCover) and by gap; error maps | Feasible in restricted form |

## 3. Decisions needed from you

1. **Must every RQ be answered experimentally, or may an RQ be closed by a documented infeasibility?** The data
   inventory suggests all five can be addressed, but RQ5 only in restricted form (below).
2. **RQ5:** OSCD has no labels separating urban change from agricultural or seasonal change. The feasible analysis is
   false-alarm rates by land-cover class (cropland, vegetation, built-up…). Is that acceptable, or should RQ5 be narrowed?
3. **RQ2 definition:** with OSCD labels fixed between date 1 and date 2, a historical image taken 1–6 months before the
   target date lies *between* the two dates and may already show part of the change. This follows the RRD's definition
   (historical image before the target SAR date), but it mixes "older context" with "change already visible". Accept
   with this caveat, or restrict to images before date 1 (possible only for about half the cities)?
4. **Companion results:** they used a different protocol, so the merged paper re-runs the baselines and reports the
   companion numbers in an appendix as a preliminary study. Agreed?
5. **Authorship, affiliation and publication of code and checkpoints** (currently local only).

## 4. Integrity safeguards (unchanged across the project)

Every step has an internal pre-registration committed before its test evaluation, validation-only selection, anchors
that reproduce stored results exactly, an independent audit script and a generated report. Completed results are not
changed. Simulated and composite clouds are never presented as real cloudy acquisitions.

Attached: the robustness draft (PDF), the research audit, the data feasibility report and the final research plan.
