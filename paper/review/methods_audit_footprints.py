"""Metadata check for the 2026-11-12 methods audit (no model, no test metric).

For each OSCD test city: does the raw date-1 Sentinel-2 10 m band cover exactly the city's bounding box, do the `_rect`
bands and the label share its pixel grid, and how far are the released Sentinel-1 raster edges from that box, in
Sentinel-2 10 m pixels. This compares footprints only; it does not measure image-content alignment.

Usage (repository root):  .venv/Scripts/python.exe paper/review/methods_audit_footprints.py
"""
import glob
import json
import sys
from pathlib import Path

import rasterio

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.datasets.oscd_dataset import ORBITS  # noqa: E402

CITIES = ["brasilia", "montpellier", "norcia", "rio", "saclay_w", "valencia", "dubai", "lasvegas", "milano", "chongqing"]
D = ROOT / "datasets"


def edge_offset(b, bb, px):
    return max(abs(b[0] - bb[0]) / px[0], abs(b[2] - bb[2]) / px[0], abs(b[1] - bb[1]) / px[1], abs(b[3] - bb[3]) / px[1])


rows = []
for c in CITIES:
    g = json.loads((D / "oscd" / "images" / c / f"{c}.geojson").read_text())["features"][0]["geometry"]["coordinates"][0]
    bb = (min(p[0] for p in g), min(p[1] for p in g), max(p[0] for p in g), max(p[1] for p in g))
    with rasterio.open(glob.glob(str(D / "oscd" / "images" / c / "imgs_1" / "*B02.tif"))[0]) as s:
        raw_b, px, raw_shape, raw_crs = s.bounds, s.res, s.shape, s.crs.to_string()
    with rasterio.open(D / "oscd" / "images" / c / "imgs_1_rect" / "B02.tif") as s:
        rect_shape, rect_crs = s.shape, s.crs
    with rasterio.open(D / "oscd" / "test_labels" / c / "cm" / f"{c}-cm.tif") as s:
        label_shape = s.shape
    s1 = []
    for o in ORBITS[c]:
        for t in (1, 2):
            with rasterio.open(D / "oscd_sentinel1" / f"sentinel1_{c}_{o}_t{t}.tif") as s:
                s1.append(edge_offset(s.bounds, bb, px))
    rows.append({"city": c, "s2_raw_edge_offset_px": round(edge_offset(raw_b, bb, px), 3), "raw_crs_geographic_wgs84": "WGS 84" in raw_crs,
                 "rect_has_crs": rect_crs is not None, "rect_shape_equals_raw_10m": rect_shape == raw_shape,
                 "label_shape_equals_rect": label_shape == rect_shape, "s1_files": len(s1), "s1_max_edge_offset_px": round(max(s1), 2)})
out = {"rows": rows, "s1_max_edge_offset_px_all": max(r["s1_max_edge_offset_px"] for r in rows)}
(Path(__file__).parent / "methods_audit_footprints.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
for r in rows:
    print(r)
print("worst S1 edge offset (S2 px):", out["s1_max_edge_offset_px_all"])
