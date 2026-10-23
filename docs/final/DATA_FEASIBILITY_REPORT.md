# Data feasibility report (Phase 2 of the final plan)

2026-10-23. Evidence: `inventory_feasibility.py` → `experiments/robust/feasibility/inventory.json` (metadata only, public
Planetary Computer STAC catalogue; no imagery downloaded, no model run). Final data come from Google Earth Engine
(`download_gee.py`), whose own manifest re-checks every scene. Decisions below are fixed before any new test evaluation.

## Summary

| RQ | Data needed | Finding | Decision |
|---|---|---|---|
| RQ2 temporal gap | Historical S2 L1C at several gaps before the target SAR date, same sites and labels | Clear scenes for 22–24/24 cities at every gap; labels are OSCD's at every gap (same sites, same pixels) | **GO**, common-site design |
| RQ3 VV/VH | S1 VV and VH from the same acquisitions, same windows as the existing VV | VV+VH for 23/24 cities at both dates (first orbit); Las Vegas has none at date 1 | **GO**, polarization comparison on 9 test cities |
| RQ5 conditions / pseudo-change | Labels separating urban change from environmental change | None exist for OSCD; land cover (ESA WorldCover 2020) available | **GO, restricted**: false alarms stratified by land cover and by gap; no claim about the cause of a change |

## RQ2: temporal gap

**Why the earlier memo said no.** `RQ2_DECISION_MEMO.md` concerned the SEN12-MT archive: unverified provenance and only 7
sites shared by all intervals. That archive is not used here.

**Design that is feasible.** Keep OSCD's 24 cities, labels, SAR (t1, t2) and split. Replace the historical optical
image (the S2 t1 slot, with S2 t2 absent, i.e. the `no_o2` setting) by a Sentinel-2 L1C acquisition taken g months
before OSCD date 2, g ∈ {1, 3, 6, 12, 18}, plus OSCD's own date 1 (native gap) and a GEE re-export of date 1 (a
same-date control for the source). Every gap uses the same cities and the same label pixels, which is the common-site
design the plan prefers.

**Inventory** (scenes within ±45 days of the target; "clear" = a day with scene cloud cover ≤ 10%):

| target | cities with a clear day | without | target before OSCD date 1 |
|---|---|---|---|
| date 2 − 1 month | 24/24 | — | 0/24 |
| date 2 − 3 months | 23/24 | chongqing (test) | 0/24 |
| date 2 − 6 months | 24/24 | — | 0/24 |
| date 2 − 12 months | 24/24 | — | 3/24 |
| date 2 − 18 months | 22/24 | beihai (validation), chongqing (test) | 11/24 |
| OSCD date 1 | 24/24 | — | 24/24 |

**Decisions.**
1. Scene rule (fixed in `download_gee.py`): the day with the lowest cloud fraction over the city box within ±45 days,
   s2cloudless probability ≥ 40, full coverage required.
2. Primary analysis set: test cities whose selected scene has cloud fraction ≤ 0.10 over the box at **every** gap.
   From the inventory this is expected to be 9 test cities (chongqing excluded); the GEE manifest decides. The 10-city
   analysis with whatever scene is clearest is reported as a sensitivity check.
3. **Known confound, stated before results.** OSCD labels mark change between date 1 and date 2. For gaps of 1–6
   months every historical image lies between the two dates and may already show part of the labelled change; at 18
   months 11/24 targets precede date 1. The gap therefore changes both how old the optical context is and how much of the
   change it already contains. Results are reported against the gap and, descriptively, by whether the image precedes
   date 1. The OSCD date-1 image is the only clean "before the change" reference.
4. The models are evaluated, not retrained, at each gap (frozen B, async and A checkpoints; frozen thresholds), so the
   gap is the only thing that changes. Training at each gap is not done: it would multiply the cost by five and confound
   the gap with selection.

**No-go conditions.** If fewer than 8 test cities have a clear scene at every gap, or the same-date control differs
from OSCD's own image by more than an alignment check allows (registered in the Step 8 pre-registration), RQ2 is
reported on the cities that pass, with the limitation, or as infeasible.

## RQ3: polarization

**Comparability.** New SAR is exported with Hafner's own recipe (S1_GRD IW, same relative orbits, same 3-month windows,
same masking and scaling), on the exact pixel grid of Hafner's VV rasters, for VV and VH from the same scenes. The
re-exported VV is compared with Hafner's released VV as a pipeline check. VH uses a [−30, 0] dB range instead of
[−25, 5] (VH is 5–10 dB lower), fixed before export.

**Inventory** (first-listed orbit, the one used at evaluation; scenes with both VV and VH, date 1 / date 2):
23/24 cities have ≥ 1 dual-polarization scene at both dates. **Las Vegas has none at date 1** (both orbits are VV-only
then); Dubai's second orbit also lacks VH at date 1 (its first orbit has one). Minimum counts: abudhabi 1 at date 1,
dubai 1 at date 1, cupertino 1 at date 2.

**Decisions.**
1. The polarization comparison (VV-only, VH-only, VV+VH, all from the new export) uses the 9 test cities other than Las
   Vegas, identical for every configuration. Las Vegas is reported for VV only.
2. Training uses all training cities; orbits whose window has no VH scene are excluded from training draws for the VH
   and VV+VH configurations only if the export shows an empty VH band (registered in Step 7; detection rule: VH band
   all zero).
3. VV from the new export (not Hafner's file) is the VV arm, so all three arms share one processing chain.

## RQ5: environmental conditions and pseudo-change

OSCD labels mark urban change only; nothing distinguishes agricultural, vegetation or soil-moisture change from
"no change". So the cause of a false alarm cannot be labelled. What can be measured:
1. **False-alarm rate by land cover**: ESA WorldCover 2020 (10 m) on the same grid; false alarms per class (cropland,
   tree cover, grassland, built-up, bare, water) for SAR-only, SAR + historical optical and full-input models. Caveat:
   WorldCover is from 2020, after OSCD date 2, so "built-up" includes new construction; it is a stratification, not a
   label.
2. **By gap and season** (from RQ2): where historical optical helps or hurts.
3. **Qualitative error maps** chosen by a fixed rule (median-F1 city, best and worst city), not by appearance.
Claims stay at "false alarms occur more on class X"; no claim that they are agricultural or seasonal pseudo-changes.

## Open items for the mentor

- Whether RQ5 restricted to land-cover-stratified false alarms satisfies the RRD, since true pseudo-change labels do not
  exist for OSCD.
- Whether the RQ2 confound (historical image may post-date the start of the change) is acceptable as the RRD's own
  definition of the gap ("historical optical selected before the target SAR observation").
