"""Split feasibility check for Strategy B: 2018-03 → 2019-12.

Determines the common eligible site set for fair three-model comparison,
proposes reproducible train/val/test split, validates all labels and
observations, computes pos_weight from training sites, and compares
Strategy B vs Strategy C.

Run from project root:
    .venv\\Scripts\\python.exe src\\training\\split_feasibility.py
"""

import json
import random
from pathlib import Path

import numpy as np
import rasterio

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "datasets" / "sen12_mt_urbanmapping" / "multimodal_cd_dataset"

T1 = (2018, 3)   # March 2018
T2 = (2019, 12)  # December 2019
THRESH = 0.5
SEED = 42

# ── load metadata ─────────────────────────────────────────────────────────────
meta = json.loads((DATASET_ROOT / "metadata.json").read_text())
bad  = json.loads((DATASET_ROOT / "bad_data.json").read_text())

def get_record(site_id, year, month):
    records = meta[site_id]
    bad_s1 = set(bad.get(site_id, {}).get("S1", []))
    bad_s2 = set(bad.get(site_id, {}).get("S2", []))
    for idx, r in enumerate(records):
        if r["year"] == year and r["month"] == month:
            return r, idx in bad_s1, idx in bad_s2
    return None, False, False

def is_s1_ok(site_id, year, month):
    r, bad_s1, _ = get_record(site_id, year, month)
    return r is not None and r["s1"] and not bad_s1 and not r["masked"]

def is_s2_ok(site_id, year, month):
    r, _, bad_s2 = get_record(site_id, year, month)
    return r is not None and r["s2"] and not bad_s2 and not r["masked"]

def is_bld_ok(site_id, year, month):
    r, _, _ = get_record(site_id, year, month)
    return r is not None and r["buildings"] and not r["masked"]

def bld_file(site_id, year, month):
    return DATASET_ROOT / site_id / "buildings" / f"buildings_{site_id}_{year}_{month:02d}.tif"

# ── classify each site ────────────────────────────────────────────────────────
y1, m1 = T1
y2, m2 = T2

site_classes = {}
for site_id in sorted(meta.keys()):
    b1  = is_bld_ok(site_id, y1, m1)
    b2  = is_bld_ok(site_id, y2, m2)
    s1t1 = is_s1_ok(site_id, y1, m1)
    s1t2 = is_s1_ok(site_id, y2, m2)
    s2t1 = is_s2_ok(site_id, y1, m1)
    s2t2 = is_s2_ok(site_id, y2, m2)

    # async: bld+S1+S2 at T1, bld+S1 at T2 (S2 T2 withheld intentionally)
    async_ok = b1 and b2 and s1t1 and s1t2 and s2t1
    # sync: additionally needs S2 at T2
    sync_ok  = async_ok and s2t2
    # common: eligible for ALL three models
    common   = sync_ok   # sync is the most restrictive

    excl_reasons = []
    if not b1:   excl_reasons.append("bld_T1_unavail")
    if not b2:   excl_reasons.append("bld_T2_unavail")
    if not s1t1: excl_reasons.append("S1_T1_bad/unavail")
    if not s1t2: excl_reasons.append("S1_T2_bad/unavail")
    if not s2t1: excl_reasons.append("S2_T1_bad/unavail")
    if not s2t2: excl_reasons.append("S2_T2_unavail")

    site_classes[site_id] = {
        "bld_T1": b1, "bld_T2": b2,
        "S1_T1": s1t1, "S1_T2": s1t2,
        "S2_T1": s2t1, "S2_T2": s2t2,
        "async_ok": async_ok,
        "sync_ok": sync_ok,
        "common": common,
        "exclusion_reasons": "|".join(excl_reasons) if excl_reasons else "",
    }

async_sites  = [s for s, v in site_classes.items() if v["async_ok"]]
sync_sites   = [s for s, v in site_classes.items() if v["sync_ok"]]
common_sites = [s for s, v in site_classes.items() if v["common"]]  # = sync_sites

print("=" * 65)
print(f"Strategy B: 2018-03 → 2019-12  (threshold={THRESH})")
print("=" * 65)
print(f"Total sites:                    {len(meta)}")
print(f"Async eligible (bld+S1+S2@T1, bld+S1@T2): {len(async_sites)}")
print(f"Sync eligible  (adds S2@T2):               {len(sync_sites)}")
print(f"Common set (all 3 models):                 {len(common_sites)}")
print()
print(f"Sites async but NOT sync ({len(async_sites)-len(sync_sites)} sites — only S2@T2 missing):")
async_only = [s for s in async_sites if s not in sync_sites]
for s in async_only:
    print(f"  {s}")

# ── label validation on ALL common sites ──────────────────────────────────────
print()
print("─" * 65)
print("Label validation on all common sites")
print("─" * 65)

change_fracs = {}
zero_px_sites = []
label_warnings = []

for site_id in common_sites:
    p1 = bld_file(site_id, y1, m1)
    p2 = bld_file(site_id, y2, m2)
    if not p1.exists():
        label_warnings.append(f"{site_id}: bld_T1 file missing on disk")
        zero_px_sites.append(site_id)
        continue
    if not p2.exists():
        label_warnings.append(f"{site_id}: bld_T2 file missing on disk")
        zero_px_sites.append(site_id)
        continue
    with rasterio.open(p1) as d1, rasterio.open(p2) as d2:
        a1 = d1.read(1)
        a2 = d2.read(1)
    b1_bin = a1 >= THRESH
    b2_bin = a2 >= THRESH
    changed = b1_bin ^ b2_bin
    frac = changed.mean()
    total_px = a1.size
    change_px = int(changed.sum())
    change_fracs[site_id] = {
        "frac": float(frac),
        "change_px": change_px,
        "total_px": total_px,
        "shape": a1.shape,
    }
    if change_px == 0:
        zero_px_sites.append(site_id)

fracs = [v["frac"] for v in change_fracs.values()]
print(f"Sites checked: {len(change_fracs)}")
if label_warnings:
    for w in label_warnings:
        print(f"  WARNING: {w}")
print(f"Zero-positive sites (thresh={THRESH}): {len(zero_px_sites)}")
if zero_px_sites:
    for s in zero_px_sites:
        print(f"  {s}")
print(f"Change fraction stats (thresh={THRESH}):")
if fracs:
    print(f"  min:    {min(fracs)*100:.3f}%")
    print(f"  max:    {max(fracs)*100:.3f}%")
    print(f"  mean:   {sum(fracs)/len(fracs)*100:.3f}%")
    sorted_fracs = sorted(fracs)
    print(f"  median: {sorted_fracs[len(sorted_fracs)//2]*100:.3f}%")

# ── reproducible train/val/test split ────────────────────────────────────────
# Use common_sites (sync-eligible) so all three models use identical sites.
# Exclude zero-positive-pixel sites from the split.
usable_sites = [s for s in common_sites if s not in zero_px_sites]
print()
print("─" * 65)
print("Reproducible split (seed=42, common usable sites)")
print("─" * 65)
print(f"Usable sites (common + non-zero positive): {len(usable_sites)}")

rng = random.Random(SEED)
shuffled = list(usable_sites)
rng.shuffle(shuffled)

n = len(shuffled)
n_test = max(3, round(n * 0.15))
n_val  = max(3, round(n * 0.15))
n_train = n - n_test - n_val

test_sites  = sorted(shuffled[:n_test])
val_sites   = sorted(shuffled[n_test:n_test + n_val])
train_sites = sorted(shuffled[n_test + n_val:])

print(f"  Train: {n_train} sites")
print(f"  Val:   {n_val} sites")
print(f"  Test:  {n_test} sites")
print(f"  Total: {n_train + n_val + n_test}")
print()
print("Train sites:")
for s in train_sites: print(f"  {s}")
print("Validation sites:")
for s in val_sites:   print(f"  {s}")
print("Test sites:")
for s in test_sites:  print(f"  {s}")

# Verify no overlap
assert not set(train_sites) & set(val_sites), "Train/val overlap!"
assert not set(train_sites) & set(test_sites), "Train/test overlap!"
assert not set(val_sites)   & set(test_sites), "Val/test overlap!"
print("\nSplit separation check: PASS (no overlaps)")

# ── pos_weight from training sites only ───────────────────────────────────────
print()
print("─" * 65)
print("pos_weight from training sites only")
print("─" * 65)

train_change_px  = sum(change_fracs[s]["change_px"] for s in train_sites)
train_total_px   = sum(change_fracs[s]["total_px"]  for s in train_sites)
train_nochange_px = train_total_px - train_change_px

if train_change_px > 0:
    pw = train_nochange_px / train_change_px
    print(f"  Training sites:   {len(train_sites)}")
    print(f"  Total pixels:     {train_total_px:,}")
    print(f"  Change pixels:    {train_change_px:,}  ({100*train_change_px/train_total_px:.3f}%)")
    print(f"  No-change pixels: {train_nochange_px:,}")
    print(f"  pos_weight:       {pw:.1f}")
else:
    print("  WARNING: zero change pixels in training sites")
    pw = None

# ── Strategy B vs Strategy C comparison ───────────────────────────────────────
print()
print("=" * 65)
print("Strategy B vs Strategy C comparison")
print("=" * 65)
print(f"{'Metric':<45} {'Strat B':>10} {'Strat C':>10}")
print(f"  {'Fixed pair':<43} {'2018-03→2019-12':>10} {'Var per site':>10}")
print(f"  {'Temporal gap':<43} {'21 mo (uniform)':>10} {'12–26 mo':>10}")
print(f"  {'Async eligible sites':<43} {len(async_sites):>10} {'59':>10}")
print(f"  {'Sync eligible sites':<43} {len(sync_sites):>10} {'51':>10}")
print(f"  {'Common eligible sites (all 3 models)':<43} {len(common_sites):>10} {'51':>10}")
print(f"  {'Zero-positive excluded':<43} {len(zero_px_sites):>10} {'TBD':>10}")
print(f"  {'Usable sites for split':<43} {len(usable_sites):>10} {'TBD':>10}")
print(f"  {'Train sites':<43} {n_train:>10} {'~36':>10}")
print(f"  {'Val sites':<43} {n_val:>10} {'~8':>10}")
print(f"  {'Test sites':<43} {n_test:>10} {'~8':>10}")
print(f"  {'pos_weight (from train)':<43} {f'{pw:.1f}' if pw else 'N/A':>10} {'TBD':>10}")

# ── evaluation adequacy assessment ───────────────────────────────────────────
print()
print("─" * 65)
print("Evaluation adequacy assessment")
print("─" * 65)
total_test_px   = sum(change_fracs[s]["total_px"]  for s in test_sites)
test_change_px  = sum(change_fracs[s]["change_px"] for s in test_sites)
total_val_px    = sum(change_fracs[s]["total_px"]  for s in val_sites)
val_change_px   = sum(change_fracs[s]["change_px"] for s in val_sites)

print(f"  Test sites:       {n_test}")
print(f"  Test pixels:      {total_test_px:,}")
print(f"  Test change px:   {test_change_px:,}  ({100*test_change_px/total_test_px:.3f}%)")
print(f"  Val sites:        {n_val}")
print(f"  Val pixels:       {total_val_px:,}")
print(f"  Val change px:    {val_change_px:,}  ({100*val_change_px/total_val_px:.3f}%)")

# Comparison to OSCD: ~158K total pixels in test set
oscd_test_px = 3_078_936  # TP+TN+FP+FN from CNN clean async metrics
print(f"\n  OSCD test set pixels (reference): {oscd_test_px:,}")
print(f"  SEN12-MT test set pixels:          {total_test_px:,}")
if total_test_px >= oscd_test_px * 0.5:
    print(f"  Assessment: SUFFICIENT (≥50% of OSCD test volume)")
else:
    print(f"  Assessment: SMALL ({100*total_test_px/oscd_test_px:.0f}% of OSCD test volume)")

# ── save split to JSON ─────────────────────────────────────────────────────────
import csv as csv_module

split_doc = {
    "strategy": "B",
    "T1": {"year": y1, "month": m1},
    "T2": {"year": y2, "month": m2},
    "interval_months": 21,
    "label_threshold": THRESH,
    "seed": SEED,
    "pos_weight": round(pw, 1) if pw else None,
    "train": train_sites,
    "validation": val_sites,
    "test": test_sites,
    "excluded_from_split": {
        "zero_positive_pixels": zero_px_sites,
        "async_ineligible": [s for s, v in site_classes.items() if not v["async_ok"]],
        "sync_ineligible_only": async_only,
    },
}

split_path = (PROJECT_ROOT / "experiments" / "scenario_2" / "configs" / "split_definition.json")
split_path.parent.mkdir(parents=True, exist_ok=True)
import json as _json
split_path.write_text(_json.dumps(split_doc, indent=2))
print(f"\nSplit written to: {split_path.relative_to(PROJECT_ROOT)}")

# Append per-site change fraction to site eligibility CSV
elig_csv = PROJECT_ROOT / "docs" / "SCENARIO_2_SITE_ELIGIBILITY.csv"
rows_new = []
if elig_csv.exists():
    with elig_csv.open(encoding="utf-8") as f:
        reader = csv_module.DictReader(f)
        existing = list(reader)
    for row in existing:
        sid = row["site_id"]
        cf = change_fracs.get(sid)
        row["common_B"] = site_classes[sid]["common"]
        row["strat_B_change_frac"] = f"{cf['frac']:.6f}" if cf else ""
        row["strat_B_change_px"] = cf["change_px"] if cf else ""
        row["split_assignment"] = (
            "train" if sid in train_sites else
            "val"   if sid in val_sites   else
            "test"  if sid in test_sites  else
            "excluded"
        )
        rows_new.append(row)
    new_fields = list(rows_new[0].keys())
    with elig_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv_module.DictWriter(f, fieldnames=new_fields)
        writer.writeheader()
        writer.writerows(rows_new)
    print(f"Updated: {elig_csv.name} (added common_B, change_frac, split columns)")

print("\nDone.")
