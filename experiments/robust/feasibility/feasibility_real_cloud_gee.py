"""Real-cloud feasibility, step 2 (numbers only, no imagery downloaded): city-box cloud fraction near OSCD date 2.

For each OSCD test city, every Sentinel-2 L1C acquisition day (COPERNICUS/S2_HARMONIZED) within +-WINDOW days of OSCD
date 2 that has a cloud-probability image, with Step 8's measure: mean over the city box of
COPERNICUS/S2_CLOUD_PROBABILITY >= 40 ("cloud") and of the B2 mask ("valid"), at 20 m. Same-day tiles are mosaicked,
as in download_gee.py. Only reduceRegion scalars are fetched. Writes real_cloud_gee.json next to this file.

Usage (repository root):  .venv_gee/Scripts/python.exe experiments/robust/feasibility/feasibility_real_cloud_gee.py [--project P]
"""

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import ee

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from download_gee import CLOUD_PROB, box, oscd_dates  # noqa: E402

TEST = ("brasilia", "montpellier", "norcia", "rio", "saclay_w", "valencia", "dubai", "lasvegas", "milano", "chongqing")
WINDOW = 30


def city_days(city: str) -> list[dict]:
    d2 = oscd_dates(city)[1]
    roi = box(city)
    lo, hi = str(d2 - dt.timedelta(days=WINDOW)), str(d2 + dt.timedelta(days=WINDOW + 1))
    col = ee.ImageCollection("COPERNICUS/S2_HARMONIZED").filterBounds(roi).filterDate(lo, hi)
    prob = ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY").filterBounds(roi).filterDate(lo, hi)
    day_list = lambda c: set(c.aggregate_array("system:time_start").map(lambda t: ee.Date(t).format("YYYY-MM-dd")).getInfo())
    rows = []
    for day in sorted(day_list(col) & day_list(prob)):
        nxt = str(dt.date.fromisoformat(day) + dt.timedelta(days=1))
        img, cp = col.filterDate(day, nxt).mosaic(), prob.filterDate(day, nxt).mosaic()
        s = ee.Image.cat(cp.gte(CLOUD_PROB).rename("cloud"), img.select("B2").mask().rename("valid")).reduceRegion(
            ee.Reducer.mean(), roi, scale=20, maxPixels=1e9).getInfo()
        rows.append({"day": day, "days_from_date2": (dt.date.fromisoformat(day) - d2).days,
                     "cloud": s.get("cloud"), "valid": s.get("valid")})
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project")
    a = p.parse_args()
    ee.Initialize(project=a.project) if a.project else ee.Initialize()
    out = {"window_days": WINDOW, "cloud_prob": CLOUD_PROB, "cities": {}}
    for city in TEST:
        rows = city_days(city)
        out["cities"][city] = {"oscd_date2": str(oscd_dates(city)[1]), "days": rows}
        full = [r for r in rows if r["valid"] is not None and r["valid"] >= 0.99 and r["cloud"] is not None]
        part = [r for r in full if 0.05 <= r["cloud"] <= 0.70]
        print(f"{city}: {len(rows)} days, {len(full)} full coverage, {len(part)} with city-box cloud 5-70%: "
              + ", ".join(f"{r['days_from_date2']:+d}d {r['cloud']:.2f}" for r in part), flush=True)
    path = Path(__file__).with_name("real_cloud_gee.json")
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {path.name}")


if __name__ == "__main__":
    main()
