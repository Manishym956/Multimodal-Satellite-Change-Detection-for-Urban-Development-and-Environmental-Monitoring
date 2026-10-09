"""Generate docs/SCENARIO_2_DATASET_MANIFEST.csv from metadata.json.
Run from project root:
    .venv\\Scripts\\python.exe src\\training\\generate_manifest.py
"""
import csv, json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "datasets" / "sen12_mt_urbanmapping" / "multimodal_cd_dataset"
OUT_PATH     = PROJECT_ROOT / "docs" / "SCENARIO_2_DATASET_MANIFEST.csv"

meta = json.loads((DATASET_ROOT / "metadata.json").read_text())
bad  = json.loads((DATASET_ROOT / "bad_data.json").read_text())

rows = []
for site_id, records in meta.items():
    bad_s1 = set(bad.get(site_id, {}).get("S1", []))
    bad_s2 = set(bad.get(site_id, {}).get("S2", []))
    for idx, r in enumerate(records):
        ts = f"{r['year']}_{r['month']:02d}"
        s1_path = DATASET_ROOT / site_id / "s1" / f"s1_{site_id}_{ts}.tif"
        s2_path = DATASET_ROOT / site_id / "s2" / f"s2_{site_id}_{ts}.tif"
        bld_path = DATASET_ROOT / site_id / "buildings" / f"buildings_{site_id}_{ts}.tif"
        rows.append({
            "aoi_id": site_id,
            "year": r["year"],
            "month": r["month"],
            "timestamp": ts,
            "s1_available": r["s1"],
            "s2_available": r["s2"],
            "buildings_available": r["buildings"],
            "masked": r["masked"],
            "s1_bad": idx in bad_s1,
            "s2_bad": idx in bad_s2,
            "s1_file_exists": s1_path.exists(),
            "s2_file_exists": s2_path.exists(),
            "bld_file_exists": bld_path.exists(),
            "s1_path": str(s1_path.relative_to(PROJECT_ROOT)) if s1_path.exists() else "",
            "s2_path": str(s2_path.relative_to(PROJECT_ROOT)) if s2_path.exists() else "",
            "bld_path": str(bld_path.relative_to(PROJECT_ROOT)) if bld_path.exists() else "",
        })

fieldnames = list(rows[0].keys())
with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Manifest written: {OUT_PATH}  ({len(rows)} rows)")
