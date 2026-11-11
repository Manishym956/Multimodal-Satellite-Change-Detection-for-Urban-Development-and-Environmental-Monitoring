"""Real-cloud feasibility (metadata only, no imagery): partly cloudy Sentinel-2 acquisitions near OSCD date 2 per test city.

For each OSCD test city, Sentinel-2 L2A items from the public Planetary Computer STAC catalogue (no login) within
+-WINDOW days of OSCD date 2 whose city box they cover, with the tile-level eo:cloud_cover. A candidate is a day with
tile cloud cover in [LO, HI] %: likely partly cloudy, but tile cover is for a ~110 km tile, not the city box, so the
city-box cloud fraction still has to be measured before any scene is chosen. L2A stands in for L1C (same acquisitions).
Writes real_cloud_inventory.json next to this file and prints one line per city.

Usage (repository root):  .venv_gee/Scripts/python.exe experiments/robust/feasibility/feasibility_real_cloud.py
"""

import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from download_gee import box_coords, oscd_dates  # noqa: E402
from inventory_feasibility import search  # noqa: E402

TEST = ("brasilia", "montpellier", "norcia", "rio", "saclay_w", "valencia", "dubai", "lasvegas", "milano", "chongqing")
WINDOW, LO, HI = 30, 5, 90


def covers(item, bbox) -> bool:
    x0, y0, x1, y1 = item["bbox"]
    return x0 <= bbox[0] and y0 <= bbox[1] and x1 >= bbox[2] and y1 >= bbox[3]


def main() -> None:
    out = []
    for city in TEST:
        d2 = oscd_dates(city)[1]
        (x0, y0), (x1, y1) = box_coords(city)
        bbox = [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
        items = [i for i in search("sentinel-2-l2a", bbox, d2 - dt.timedelta(days=WINDOW), d2 + dt.timedelta(days=WINDOW))
                 if covers(i, bbox) and i["properties"].get("eo:cloud_cover") is not None]
        days = {}
        for i in items:  # one entry per day: the lowest tile cover among items covering the box
            day, cc = i["properties"]["datetime"][:10], i["properties"]["eo:cloud_cover"]
            days[day] = min(cc, days.get(day, 101))
        cand = sorted((d, round(c, 1)) for d, c in days.items() if LO <= c <= HI)
        out.append({"city": city, "oscd_date2": str(d2), "n_days": len(days), "n_candidates": len(cand),
                    "candidates": [{"date": d, "tile_cloud": c, "days_from_date2": (dt.date.fromisoformat(d) - d2).days}
                                   for d, c in cand]})
        near = min(cand, key=lambda x: abs((dt.date.fromisoformat(x[0]) - d2).days)) if cand else None
        print(f"{city}: date2 {d2}, {len(days)} days covered, {len(cand)} candidates; nearest {near}", flush=True)
    path = Path(__file__).with_name("real_cloud_inventory.json")
    path.write_text(json.dumps({"window_days": WINDOW, "tile_cloud_range": [LO, HI], "cities": out}, indent=2), encoding="utf-8")
    print(f"wrote {path.name}")


if __name__ == "__main__":
    main()
