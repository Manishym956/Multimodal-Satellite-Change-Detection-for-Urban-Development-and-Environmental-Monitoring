"""Download the real-cloud Sentinel-2 L1C scenes for the OSCD test cities (no model sees them here).

Scene rule, applied mechanically to the city-box cloud fractions already measured in
experiments/robust/feasibility/real_cloud_gee.json (COPERNICUS/S2_CLOUD_PROBABILITY >= 40, full coverage, +-30 days of
OSCD date 2), fixed before any download:
    cloudy    the full-coverage day with city-box cloud in [0.05, 0.70] nearest to OSCD date 2 (tie: earlier day)
    overcast  only for a city without a cloudy day: the full-coverage day with cloud >= 0.90 nearest to date 2 (tie: earlier)
    control   OSCD's own date-2 day (same-date check of the Earth Engine processing chain, as Step 8's gee_t1)
Each scene: COPERNICUS/S2_HARMONIZED, 13 bands as uint16 DN, same-day tiles mosaicked, on the grid of the city's first
listed Sentinel-1 orbit raster, exactly as download_gee.py s2. Output datasets/oscd_s2_realcloud/{city}_{tag}.tif and
manifest.json. Refuses to overwrite.

Usage (repository root):
    .venv_gee/Scripts/python.exe experiments/robust/realcloud/download_real_cloud.py --project <gcp-project> [--select-only]
"""

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import ee  # noqa: E402

from download_gee import ORBITS, S2_BANDS, box, download, grid  # noqa: E402

FEAS = ROOT / "experiments" / "robust" / "feasibility" / "real_cloud_gee.json"
OUT = ROOT / "datasets" / "oscd_s2_realcloud"
DESCR = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]
LO, HI, OVERCAST = 0.05, 0.70, 0.90


def select(cities: dict) -> list[dict]:
    rows = []
    for city, v in cities.items():
        full = [r for r in v["days"] if r["valid"] is not None and r["valid"] >= 0.99 and r["cloud"] is not None]
        near = lambda rs: min(rs, key=lambda r: (abs(r["days_from_date2"]), r["days_from_date2"])) if rs else None
        cloudy = near([r for r in full if LO <= r["cloud"] <= HI])
        pick, tag = (cloudy, "cloudy") if cloudy else (near([r for r in full if r["cloud"] >= OVERCAST]), "overcast")
        control = next(r for r in full if r["days_from_date2"] == 0)
        for t, r in ((tag, pick), ("control", control)):
            assert r is not None, f"{city}: no {t} scene"
            rows.append({"city": city, "tag": t, "day": r["day"], "days_from_date2": r["days_from_date2"],
                         "cloud": r["cloud"], "oscd_date2": v["oscd_date2"]})
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True)
    p.add_argument("--select-only", action="store_true")
    a = p.parse_args()
    rows = select(json.loads(FEAS.read_text(encoding="utf-8"))["cities"])
    for r in rows:
        print(f"{r['city']:12s} {r['tag']:9s} {r['day']} ({r['days_from_date2']:+d} d) cloud {r['cloud']:.3f}")
    if a.select_only:
        return
    ee.Initialize(project=a.project)
    OUT.mkdir(parents=True, exist_ok=True)
    for r in rows:
        g, profile = grid(r["city"], ORBITS[r["city"]][0])
        nxt = str(dt.date.fromisoformat(r["day"]) + dt.timedelta(days=1))
        col = ee.ImageCollection("COPERNICUS/S2_HARMONIZED").filterBounds(box(r["city"])).filterDate(r["day"], nxt)
        download(col.mosaic().select(S2_BANDS).toUint16(), g, profile, OUT / f"{r['city']}_{r['tag']}.tif", DESCR, bands=S2_BANDS)
        print(r["city"], r["tag"], "downloaded", flush=True)
    (OUT / "manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("REAL-CLOUD DOWNLOAD DONE")


if __name__ == "__main__":
    main()
