"""Lightweight inspection of the SEN12-MT Urban Mapping dataset.
Run from project root:
    .venv\\Scripts\\python.exe src\\training\\inspect_sen12mt.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import rasterio

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "datasets" / "sen12_mt_urbanmapping" / "multimodal_cd_dataset"

# ── overview from metadata ────────────────────────────────────────────────────
meta = json.loads((DATASET_ROOT / "metadata.json").read_text())
bad  = json.loads((DATASET_ROOT / "bad_data.json").read_text())

sites = list(meta.keys())
lengths = [len(v) for v in meta.values()]
all_years = sorted({r["year"] for records in meta.values() for r in records})
s1_avail = sum(1 for records in meta.values() for r in records if r["s1"])
s2_avail = sum(1 for records in meta.values() for r in records if r["s2"])
total_ts  = sum(lengths)
bad_s1 = sum(len(v["S1"]) for v in bad.values())
bad_s2 = sum(len(v["S2"]) for v in bad.values())

# file counts
s1_files  = list(DATASET_ROOT.rglob("s1/*.tif"))
s2_files  = list(DATASET_ROOT.rglob("s2/*.tif"))
bld_files = list(DATASET_ROOT.rglob("buildings/*.tif"))
total_bytes = sum(f.stat().st_size for f in DATASET_ROOT.rglob("*.tif"))

print("=" * 60)
print("SEN12-MT Urban Mapping — Dataset Inspection")
print("=" * 60)
print(f"Dataset root:        {DATASET_ROOT}")
print(f"Sites:               {len(sites)}")
print(f"Timestamps per site: min={min(lengths)} max={max(lengths)} mean={sum(lengths)/len(lengths):.1f}")
print(f"Year range:          {all_years}")
print(f"Total timesteps:     {total_ts}")
print(f"S1 available:        {s1_avail} / {total_ts}")
print(f"S2 available:        {s2_avail} / {total_ts}")
print(f"Bad S1 timesteps:    {bad_s1}")
print(f"Bad S2 timesteps:    {bad_s2}")
print(f"S1 GeoTIFFs on disk: {len(s1_files)}")
print(f"S2 GeoTIFFs on disk: {len(s2_files)}")
print(f"Buildings GeoTIFFs:  {len(bld_files)}")
print(f"Total size on disk:  {total_bytes/1e9:.2f} GB")

# ── raster metadata for one well-known good site ─────────────────────────────
site = "L15-0331E-1257N_1327_3160_13"
print(f"\n--- Raster metadata (site: {site}) ---")

# find first non-all-NaN S1
for ts in ["2018_02", "2018_06", "2019_01"]:
    s1_path = DATASET_ROOT / site / "s1" / f"s1_{site}_{ts}.tif"
    with rasterio.open(s1_path) as ds:
        arr = ds.read()
        valid = arr[~np.isnan(arr)]
        if len(valid) > 100:
            print(f"\nS1 ({ts}):")
            print(f"  Bands:         {ds.count}  {ds.descriptions}")
            print(f"  Dtype:         {ds.dtypes[0]}")
            print(f"  Shape:         {ds.height} x {ds.width}")
            print(f"  CRS:           {ds.crs}")
            print(f"  Resolution:    {ds.res} m")
            print(f"  NaN fraction:  {np.isnan(arr).mean():.4f}")
            print(f"  Value range:   [{valid.min():.4f}, {valid.max():.4f}]")
            break

s2_path = DATASET_ROOT / site / "s2" / f"s2_{site}_2018_06.tif"
with rasterio.open(s2_path) as ds:
    arr = ds.read().astype(float)
    valid = arr[~np.isnan(arr)]
    print(f"\nS2 (2018_06):")
    print(f"  Bands:         {ds.count}  {ds.descriptions}")
    print(f"  Dtype:         {ds.dtypes[0]}")
    print(f"  Shape:         {ds.height} x {ds.width}")
    print(f"  CRS:           {ds.crs}")
    print(f"  Resolution:    {ds.res} m")
    print(f"  Value range:   [{valid.min():.4f}, {valid.max():.4f}]  (already normalised to [0,1])")

bld_path = DATASET_ROOT / site / "buildings" / f"buildings_{site}_2018_01.tif"
with rasterio.open(bld_path) as ds:
    arr = ds.read(1)
    print(f"\nBuildings label (2018_01):")
    print(f"  Bands:         {ds.count}")
    print(f"  Dtype:         {ds.dtypes[0]}")
    print(f"  Shape:         {ds.height} x {ds.width}")
    print(f"  Value range:   [{arr.min():.2f}, {arr.max():.2f}]")
    print(f"  Unique values: continuous float in [0,1] (building probability)")
    print(f"  At thresh 0.5: {(arr>=0.5).sum()} building pixels ({100*(arr>=0.5).mean():.2f}%)")

# ── change derivation sample (earliest vs latest unmasked) ───────────────────
print("\n--- Change derivation sample (thresh=0.5, T1=earliest vs T2=latest unmasked) ---")
print(f"{'Site':<35} {'T1':>10} {'T2':>10} {'changed%':>10} {'new_build%':>12}")

for site_id in sites[:8]:
    records = sorted(
        [r for r in meta[site_id] if not r["masked"] and r["buildings"]],
        key=lambda r: (r["year"], r["month"])
    )
    if len(records) < 2:
        continue
    r1, r2 = records[0], records[-1]
    t1 = f"{r1['year']}_{r1['month']:02d}"
    t2 = f"{r2['year']}_{r2['month']:02d}"
    b1p = DATASET_ROOT / site_id / "buildings" / f"buildings_{site_id}_{t1}.tif"
    b2p = DATASET_ROOT / site_id / "buildings" / f"buildings_{site_id}_{t2}.tif"
    if b1p.exists() and b2p.exists():
        with rasterio.open(b1p) as d1, rasterio.open(b2p) as d2:
            a1 = d1.read(1) >= 0.5
            a2 = d2.read(1) >= 0.5
        chg  = (a1 ^ a2).mean() * 100
        built = ((~a1) & a2).mean() * 100
        print(f"  {site_id:<35} {t1:>10} {t2:>10} {chg:>9.2f}% {built:>11.2f}%")

# ── key question: change label availability ───────────────────────────────────
print("\n--- Key findings ---")
print("1. Labels: 'buildings' folder contains CONTINUOUS building probability maps per timestep.")
print("   These are NOT pre-computed binary change masks.")
print("   Binary change mask must be DERIVED: (buildings_T2 >= 0.5) XOR (buildings_T1 >= 0.5)")
print("   OR: (buildings_T2 - buildings_T1) > threshold for soft change.")
print()
print("2. S1: 2-band (VV + VH), float32, normalised [0,1], 10m, Pseudo-Mercator.")
print("   S2: 10-band (B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12), float32, already normalised [0,1], 10m.")
print("   Note: S2 is already in [0,1] — do NOT divide by 10000.")
print()
print("3. No official train/val/test split file detected in the dataset directory.")
print("   Split must be defined manually at site level.")
print()
print("4. NaN pixels present in S1. Need to be handled (e.g. fill with 0 or mask out).")
print("   bad_data.json lists known bad timestep indices per site.")
