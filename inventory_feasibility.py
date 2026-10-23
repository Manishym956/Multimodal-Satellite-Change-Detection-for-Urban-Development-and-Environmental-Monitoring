"""Data-feasibility inventory (metadata only, no imagery): Sentinel-1 VV/VH and Sentinel-2 scene availability per OSCD city,
from the public Planetary Computer STAC catalogue (no login). Preliminary to the Earth Engine downloads; the final check is
download_gee.py's own manifest.

S1 (RQ3): for each city and relative orbit in DS-UNet's list, IW GRD scenes in Hafner's windows (3 months before date 1,
3 months after date 2; mumbai -12/+5 months at t1) and how many carry both VV and VH.
S2 (RQ2): for each city and target (OSCD date 1; date 2 minus 1, 3, 6, 12, 18 months), L2A scenes within +-45 days and
the lowest scene-level cloud cover (L2A availability stands in for L1C; same acquisitions).
Writes experiments/robust/feasibility/inventory.json and prints a summary.

Usage:
    .venv_gee/Scripts/python.exe inventory_feasibility.py
"""

import datetime as dt
import json
from pathlib import Path

import requests

from download_gee import GAPS, ORBITS, WINDOW, add_months, box_coords, oscd_dates

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
OUT = Path(__file__).resolve().parent / "experiments" / "robust" / "feasibility"


def search(collection, bbox, start, end, query=None):
    body = {"collections": [collection], "bbox": bbox, "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z", "limit": 1000}
    if query:
        body["query"] = query
    items, url = [], STAC
    while url:
        r = requests.post(url, json=body, timeout=120)
        r.raise_for_status()
        d = r.json()
        items += d["features"]
        nxt = [l for l in d.get("links", []) if l.get("rel") == "next"]
        url, body = (nxt[0]["href"], nxt[0].get("body", body)) if nxt else (None, None)
    return items


def main():
    inv = {"s1": [], "s2": []}
    for city, orbits in ORBITS.items():
        d1, d2 = oscd_dates(city)
        (x0, y0), (x1, y1) = box_coords(city)
        bbox = [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
        w1 = (add_months(d1, -12), add_months(d1, 5)) if city == "mumbai" else (add_months(d1, -3), d1)
        w2 = (d2, add_months(d2, 3))
        for orbit in orbits:
            for t, (a, b) in ((1, w1), (2, w2)):
                items = [i for i in search("sentinel-1-grd", bbox, a, b, {"sar:instrument_mode": {"eq": "IW"}})
                         if i["properties"].get("sat:relative_orbit") == orbit]
                pols = [set(i["properties"].get("sar:polarizations", [])) for i in items]
                inv["s1"].append({"city": city, "orbit": orbit, "t": t, "n": len(items), "n_vv": sum("VV" in p for p in pols),
                                  "n_vh": sum("VH" in p for p in pols), "n_vv_vh": sum({"VV", "VH"} <= p for p in pols)})
        for tag, target in {"oscd_t1": d1, **{f"gap{m:02d}": add_months(d2, -m) for m in GAPS}}.items():
            a, b = target - dt.timedelta(days=WINDOW), target + dt.timedelta(days=WINDOW)
            items = search("sentinel-2-l2a", bbox, a, b)
            cc = [i["properties"].get("eo:cloud_cover") for i in items if i["properties"].get("eo:cloud_cover") is not None]
            days = {i["properties"]["datetime"][:10] for i in items}
            inv["s2"].append({"city": city, "tag": tag, "target": str(target), "n_scenes": len(items), "n_days": len(days),
                              "min_cloud": min(cc) if cc else None, "n_days_cloud_le10": len({i["properties"]["datetime"][:10]
                              for i in items if (i["properties"].get("eo:cloud_cover") or 100) <= 10}),
                              "before_oscd_t1": target <= d1})
        print(city, "done", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "inventory.json").write_text(json.dumps(inv, indent=2), encoding="utf-8")
    print("inventory written")


if __name__ == "__main__":
    main()
