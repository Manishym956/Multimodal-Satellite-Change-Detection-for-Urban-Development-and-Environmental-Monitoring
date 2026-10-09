"""Temporal pairing feasibility analysis for SEN12-MT Urban Mapping.

Evaluates three T1/T2 selection strategies:
  A. Fixed 2018-01 → 2020-01
  B. Fixed near-24-month pairs (best-coverage alternatives)
  C. Earliest/latest valid per site (fallback)

For each strategy reports per-site eligibility across:
  - buildings label valid at T1 and T2
  - S1 usable at T1 and T2 (metadata s1=True AND not in bad_data S1)
  - S2 usable at T1 (metadata s2=True AND not in bad_data S2)
  - S2 usable at T2 (for synchronous model)
  - Full async eligibility: bld+S1+S2 at T1, bld+S1 at T2
  - Full sync eligibility: bld+S1+S2 at both T1 and T2
  - Temporal gap in months

Also performs raster-level label validity checks (change fraction, threshold
sensitivity) on a representative sample without loading all 18 GB.

Outputs:
  docs/SCENARIO_2_TEMPORAL_PAIRING_ANALYSIS.md
  docs/SCENARIO_2_SITE_ELIGIBILITY.csv

Run from project root:
  .venv\\Scripts\\python.exe src\\training\\temporal_pairing_analysis.py
"""

import csv
import json
from pathlib import Path

import numpy as np
import rasterio

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "datasets" / "sen12_mt_urbanmapping" / "multimodal_cd_dataset"
DOCS_DIR = PROJECT_ROOT / "docs"


# ── load metadata ─────────────────────────────────────────────────────────────

meta = json.loads((DATASET_ROOT / "metadata.json").read_text())
bad  = json.loads((DATASET_ROOT / "bad_data.json").read_text())

# Build per-site lookup: (year, month) → record + bad flags
site_records = {}
for site_id, records in meta.items():
    bad_s1_idx = set(bad.get(site_id, {}).get("S1", []))
    bad_s2_idx = set(bad.get(site_id, {}).get("S2", []))
    ts_map = {}
    for idx, r in enumerate(records):
        key = (r["year"], r["month"])
        ts_map[key] = {
            "year": r["year"],
            "month": r["month"],
            "s1_meta": r["s1"],
            "s2_meta": r["s2"],
            "bld_meta": r["buildings"],
            "masked": r["masked"],
            "s1_bad": idx in bad_s1_idx,
            "s2_bad": idx in bad_s2_idx,
        }
    site_records[site_id] = ts_map


def is_s1_usable(site_id, year, month):
    """S1 usable = metadata s1=True and not in bad_data and not masked."""
    ts = site_records[site_id].get((year, month))
    if ts is None:
        return False
    return ts["s1_meta"] and not ts["s1_bad"] and not ts["masked"]


def is_s2_usable(site_id, year, month):
    """S2 usable = metadata s2=True and not in bad_data and not masked."""
    ts = site_records[site_id].get((year, month))
    if ts is None:
        return False
    return ts["s2_meta"] and not ts["s2_bad"] and not ts["masked"]


def is_bld_usable(site_id, year, month):
    """Buildings usable = metadata buildings=True and not masked."""
    ts = site_records[site_id].get((year, month))
    if ts is None:
        return False
    return ts["bld_meta"] and not ts["masked"]


def month_gap(y1, m1, y2, m2):
    return (y2 - y1) * 12 + (m2 - m1)


def bld_path(site_id, year, month):
    return DATASET_ROOT / site_id / "buildings" / f"buildings_{site_id}_{year}_{month:02d}.tif"


# ── strategy definitions ──────────────────────────────────────────────────────

# Strategy A: fixed 2018-01 → 2020-01
T1_A = (2018, 1)
T2_A = (2020, 1)

# Strategy B: evaluate several fixed pairs near 24-month gap
# Candidates: 2018-01→2020-01, 2018-02→2020-01, 2018-03→2019-12, 2018-01→2019-12
B_CANDIDATES = [
    ((2018, 1), (2020, 1), "2018-01→2020-01"),
    ((2018, 2), (2020, 1), "2018-02→2020-01"),
    ((2018, 3), (2019, 12), "2018-03→2019-12"),
    ((2018, 1), (2019, 12), "2018-01→2019-12"),
    ((2018, 2), (2019, 12), "2018-02→2019-12"),
]


def count_eligible(t1, t2):
    """Return counts of sites eligible for async and sync under fixed pair (t1, t2)."""
    y1, m1 = t1
    y2, m2 = t2
    async_ok = 0
    sync_ok  = 0
    bld_ok   = 0
    s1_ok    = 0
    s2t1_ok  = 0
    total = len(site_records)
    for site_id in site_records:
        b1 = is_bld_usable(site_id, y1, m1)
        b2 = is_bld_usable(site_id, y2, m2)
        s1t1 = is_s1_usable(site_id, y1, m1)
        s1t2 = is_s1_usable(site_id, y2, m2)
        s2t1 = is_s2_usable(site_id, y1, m1)
        s2t2 = is_s2_usable(site_id, y2, m2)
        if b1 and b2:
            bld_ok += 1
        if s1t1 and s1t2:
            s1_ok += 1
        if s2t1:
            s2t1_ok += 1
        # async: bld+S1+S2 at T1, bld+S1 at T2 (S2 T2 withheld)
        if b1 and b2 and s1t1 and s1t2 and s2t1:
            async_ok += 1
        # sync: bld+S1+S2 at both endpoints
        if b1 and b2 and s1t1 and s1t2 and s2t1 and s2t2:
            sync_ok += 1
    return dict(total=total, bld=bld_ok, s1=s1_ok, s2t1=s2t1_ok,
                async_ok=async_ok, sync_ok=sync_ok)


# ── Strategy A ────────────────────────────────────────────────────────────────
print("=" * 60)
print("Strategy A: Fixed 2018-01 → 2020-01")
ca = count_eligible(T1_A, T2_A)
print(f"  Total sites:              {ca['total']}")
print(f"  Buildings valid both:     {ca['bld']}")
print(f"  S1 usable both:           {ca['s1']}")
print(f"  S2 usable at T1:          {ca['s2t1']}")
print(f"  Full ASYNC eligible:      {ca['async_ok']}")
print(f"  Full SYNC eligible:       {ca['sync_ok']}")

# ── Strategy B candidates ──────────────────────────────────────────────────────
print()
print("Strategy B: Fixed pair candidates")
print(f"  {'Pair':<25} {'Bld':>5} {'S1':>5} {'S2T1':>6} {'Async':>7} {'Sync':>6}")
best_b = None; best_b_label = ""; best_b_counts = None
for t1, t2, label in B_CANDIDATES:
    c = count_eligible(t1, t2)
    mark = " ← best async" if best_b is None or c['async_ok'] > best_b['async_ok'] else ""
    if best_b is None or c['async_ok'] > best_b['async_ok']:
        best_b = c; best_b_label = label; best_b_t1 = t1; best_b_t2 = t2
    print(f"  {label:<25} {c['bld']:>5} {c['s1']:>5} {c['s2t1']:>6} {c['async_ok']:>7} {c['sync_ok']:>6}{mark}")

# ── Strategy C: earliest/latest per site ──────────────────────────────────────
print()
print("Strategy C: Earliest/latest valid unmasked per site")

def get_earliest_latest(site_id):
    """Return (T1, T2) where T1 = earliest unmasked bld+S1+S2 timestep,
    T2 = latest unmasked bld+S1 timestep (S2 not required at T2 for async)."""
    timestamps = sorted(site_records[site_id].keys())
    t1_candidates = [(y, m) for (y, m) in timestamps
                     if is_bld_usable(site_id, y, m)
                     and is_s1_usable(site_id, y, m)
                     and is_s2_usable(site_id, y, m)]
    # T2 must have bld + S1 (S2 optional for async, required for sync)
    t2_candidates = [(y, m) for (y, m) in timestamps
                     if is_bld_usable(site_id, y, m)
                     and is_s1_usable(site_id, y, m)]
    if not t1_candidates or not t2_candidates:
        return None, None
    t1 = t1_candidates[0]
    # T2 must be strictly later than T1
    t2_later = [(y, m) for (y, m) in t2_candidates
                if (y, m) > t1]
    if not t2_later:
        return None, None
    t2 = t2_later[-1]
    return t1, t2

async_c = 0; sync_c = 0; total_c = len(site_records)
gaps_c = []
site_c_pairs = {}
for site_id in site_records:
    t1, t2 = get_earliest_latest(site_id)
    if t1 is None:
        site_c_pairs[site_id] = None
        continue
    y1, m1 = t1; y2, m2 = t2
    gap = month_gap(y1, m1, y2, m2)
    s2t2 = is_s2_usable(site_id, y2, m2)
    async_ok = True  # by construction (T1 has bld+S1+S2, T2 has bld+S1)
    sync_ok  = s2t2
    async_c += async_ok
    sync_c  += sync_ok
    gaps_c.append(gap)
    site_c_pairs[site_id] = (t1, t2, gap, sync_ok)

print(f"  Total sites:              {total_c}")
print(f"  Full ASYNC eligible:      {async_c}")
print(f"  Full SYNC eligible:       {sync_c}")
if gaps_c:
    print(f"  Temporal gap (months): min={min(gaps_c)} max={max(gaps_c)} mean={sum(gaps_c)/len(gaps_c):.1f} median={sorted(gaps_c)[len(gaps_c)//2]}")

# ── per-site detail table ──────────────────────────────────────────────────────
# Build the full per-site eligibility for Strategy A (used in CSV)
rows = []
for site_id in sorted(site_records.keys()):
    # Strategy A
    y1a, m1a = T1_A; y2a, m2a = T2_A
    b1a = is_bld_usable(site_id, y1a, m1a)
    b2a = is_bld_usable(site_id, y2a, m2a)
    s1t1a = is_s1_usable(site_id, y1a, m1a)
    s1t2a = is_s1_usable(site_id, y2a, m2a)
    s2t1a = is_s2_usable(site_id, y1a, m1a)
    s2t2a = is_s2_usable(site_id, y2a, m2a)
    async_a = b1a and b2a and s1t1a and s1t2a and s2t1a
    sync_a  = async_a and s2t2a
    excl_a_reason = ""
    if not async_a:
        reasons = []
        if not b1a: reasons.append("bld_T1_invalid")
        if not b2a: reasons.append("bld_T2_invalid")
        if not s1t1a: reasons.append("S1_T1_bad")
        if not s1t2a: reasons.append("S1_T2_bad")
        if not s2t1a: reasons.append("S2_T1_bad")
        excl_a_reason = "|".join(reasons)

    # Strategy C
    t1c, t2c = get_earliest_latest(site_id)
    if t1c:
        y1c, m1c = t1c; y2c, m2c = t2c
        gap_c = month_gap(y1c, m1c, y2c, m2c)
        s2t2c = is_s2_usable(site_id, y2c, m2c)
        t1c_str = f"{y1c}-{m1c:02d}"
        t2c_str = f"{y2c}-{m2c:02d}"
    else:
        t1c_str = ""; t2c_str = ""; gap_c = None; s2t2c = False

    rows.append({
        "site_id": site_id,
        "strat_A_t1": f"{y1a}-{m1a:02d}",
        "strat_A_t2": f"{y2a}-{m2a:02d}",
        "strat_A_bld_T1": b1a, "strat_A_bld_T2": b2a,
        "strat_A_S1_T1": s1t1a, "strat_A_S1_T2": s1t2a,
        "strat_A_S2_T1": s2t1a, "strat_A_S2_T2": s2t2a,
        "strat_A_async_eligible": async_a,
        "strat_A_sync_eligible": sync_a,
        "strat_A_exclusion_reason": excl_a_reason,
        "strat_A_gap_months": month_gap(y1a, m1a, y2a, m2a),
        "strat_C_t1": t1c_str,
        "strat_C_t2": t2c_str,
        "strat_C_gap_months": gap_c,
        "strat_C_async_eligible": t1c is not None,
        "strat_C_sync_eligible": t1c is not None and s2t2c,
        "strat_C_exclusion_reason": "" if t1c else "no_valid_pair",
    })

# ── label validity: raster checks on all async-eligible Strategy A sites ──────
print()
print("=" * 60)
print("Label validity check (Strategy A eligible sites)")

eligible_sites_a = [r["site_id"] for r in rows if r["strat_A_async_eligible"]]
change_fracs = {}
zero_positive = []
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]
thresh_positive_counts = {t: 0 for t in THRESHOLDS}

print(f"  Checking {len(eligible_sites_a)} sites...")
for site_id in eligible_sites_a:
    p1 = bld_path(site_id, *T1_A)
    p2 = bld_path(site_id, *T2_A)
    if not p1.exists() or not p2.exists():
        zero_positive.append(site_id)
        continue
    with rasterio.open(p1) as d1, rasterio.open(p2) as d2:
        a1 = d1.read(1)
        a2 = d2.read(1)

    for thresh in THRESHOLDS:
        changed = (a1 >= thresh) ^ (a2 >= thresh)
        thresh_positive_counts[thresh] += changed.sum()

    # Primary threshold 0.5
    b1 = a1 >= 0.5
    b2 = a2 >= 0.5
    changed = b1 ^ b2
    frac = changed.mean()
    change_fracs[site_id] = float(frac)
    if changed.sum() == 0:
        zero_positive.append(site_id)

total_px = sum(1 for _ in eligible_sites_a) * (489 * 488)  # approximate
print(f"  Change fraction stats (thresh=0.5) across {len(change_fracs)} sites:")
fracs = list(change_fracs.values())
if fracs:
    print(f"    min:    {min(fracs)*100:.3f}%")
    print(f"    max:    {max(fracs)*100:.3f}%")
    print(f"    mean:   {sum(fracs)/len(fracs)*100:.3f}%")
    print(f"    median: {sorted(fracs)[len(fracs)//2]*100:.3f}%")

print(f"  Sites with zero positive pixels (thresh=0.5): {len(zero_positive)}")
if zero_positive:
    for s in zero_positive:
        print(f"    {s}")

print()
print("  Threshold sensitivity (total changed pixels across eligible sites):")
for t in THRESHOLDS:
    print(f"    thresh={t:.1f}: {thresh_positive_counts[t]:,} positive pixels")

# pos_weight estimate using 0.5 threshold on these sites
total_changed_px = sum(int(f * 489 * 488) for f in change_fracs.values())
total_all_px     = len(change_fracs) * 489 * 488
total_unchanged  = total_all_px - total_changed_px
if total_changed_px > 0:
    pw = total_unchanged / total_changed_px
    print(f"\n  Estimated pos_weight (thresh=0.5, Strategy A sites): {pw:.1f}")
    print(f"  (no_change={total_unchanged:,}, change={total_changed_px:,})")

# ── write CSV ─────────────────────────────────────────────────────────────────
csv_path = DOCS_DIR / "SCENARIO_2_SITE_ELIGIBILITY.csv"
fieldnames = list(rows[0].keys())
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
print(f"\nCSV written: {csv_path.name}  ({len(rows)} sites)")

# ── store results for report ──────────────────────────────────────────────────
RESULTS = {
    "strategy_A": ca,
    "strategy_A_label_fracs": change_fracs,
    "strategy_A_zero_positive": zero_positive,
    "strategy_A_thresh_sensitivity": thresh_positive_counts,
    "strategy_B_best": {"label": best_b_label, "t1": best_b_t1, "t2": best_b_t2, **best_b},
    "strategy_C": {"async_ok": async_c, "sync_ok": sync_c,
                   "gap_min": min(gaps_c) if gaps_c else None,
                   "gap_max": max(gaps_c) if gaps_c else None,
                   "gap_mean": round(sum(gaps_c)/len(gaps_c), 1) if gaps_c else None,
                   "gap_median": sorted(gaps_c)[len(gaps_c)//2] if gaps_c else None},
}

print("\nDone. Results stored for report generation.")

# Export RESULTS for the report script to import
import pickle
(PROJECT_ROOT / "docs" / "_pairing_results.pkl").write_bytes(pickle.dumps(RESULTS))
