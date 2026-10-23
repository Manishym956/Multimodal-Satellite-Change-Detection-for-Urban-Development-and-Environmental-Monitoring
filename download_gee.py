"""Download Sentinel-1 VV+VH (Step 7, RQ3), historical Sentinel-2 L1C (Step 8, RQ1/RQ2) and ESA WorldCover (Step 8, RQ5)
from Google Earth Engine, each on the exact pixel grid (CRS, transform, size) of the Sentinel-1 VV raster that DS-UNet
released for that city and orbit (datasets/oscd_sentinel1), so every product is aligned exactly as the VV the models use.

s1   Hafner's recipe (repos/2/DS_UNet/sentinel1_download.ipynb): COPERNICUS/S1_GRD, IW, one relative orbit, mean of the
     3 months before OSCD date 1 (t1) and the 3 months after date 2 (t2) (mumbai: t1 window -12 to +5 months), pixels
     below the floor masked, unitScale to [0, 1], clamp, unmask. VV: [-25, 5] dB as Hafner; VH: [-30, 0] dB (VH is
     5-10 dB lower than VV; registered in PHASE7_PREREGISTRATION.md). Output datasets/oscd_s1_gee/sentinel1_{city}_{orbit}_t{1,2}.tif
     with band descriptions VV, VH.
s2   COPERNICUS/S2_HARMONIZED (L1C, 13 bands, DN = reflectance x 10000) for each target date (OSCD date 1, and date 2
     minus each gap in GAPS months). Scene rule (fixed before any model saw a downloaded scene): among acquisition days
     within +-WINDOW days of the target with full coverage and a cloud-probability image, the day nearest to the target
     whose cloud fraction over the city box (COPERNICUS/S2_CLOUD_PROBABILITY >= 40) is <= CLEAR; if none, the day with the
     lowest cloud fraction, ties nearest. Same-day tiles are mosaicked. For OSCD date 1 (the same-date control) OSCD's own
     acquisition day is used when it has full coverage. Output datasets/oscd_s2_gee/{city}_{tag}.tif (13 bands,
     B01..B12 order as OSCD) plus a manifest row (date, cloud fraction, valid fraction). Grid: the first listed orbit's raster.
wc   ESA/WorldCover/v100 (2020), nearest neighbour, datasets/oscd_wc_gee/{city}_worldcover.tif.
Refuses to overwrite any file.

Usage (Earth Engine venv):
    .venv_gee/Scripts/python.exe download_gee.py s1|s2|wc --project <gcp-project>
"""

import argparse
import datetime as dt
import io
import json
import time
import zipfile
from pathlib import Path

import ee
import numpy as np
import rasterio
import requests

ROOT = Path(__file__).resolve().parent
OSCD_IMAGES = ROOT / "datasets" / "Onera Satellite Change Detection dataset - Images"
HAFNER_S1 = ROOT / "datasets" / "oscd_sentinel1"
OUT = {"s1": ROOT / "datasets" / "oscd_s1_gee", "s2": ROOT / "datasets" / "oscd_s2_gee", "wc": ROOT / "datasets" / "oscd_wc_gee"}
ORBITS = {
    "aguasclaras": [24], "bercy": [59, 8, 110], "bordeaux": [30, 8, 81], "nantes": [30, 81], "paris": [59, 8, 110],
    "rennes": [30, 81], "saclay_e": [59, 8], "abudhabi": [130], "cupertino": [35, 115, 42], "pisa": [15, 168],
    "beihai": [157], "hongkong": [11, 113], "beirut": [14, 87], "mumbai": [34], "brasilia": [24], "montpellier": [59, 37],
    "norcia": [117, 44, 22, 95], "rio": [155], "saclay_w": [59, 8, 110], "valencia": [30, 103, 8, 110], "dubai": [130, 166],
    "lasvegas": [166, 173], "milano": [66, 168], "chongqing": [55, 164]}  # as DS-UNet's notebook and oscd_dataset.ORBITS
S1_RANGE = {"VV": (-25, 5), "VH": (-30, 0)}
S2_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]
GAPS = (1, 3, 6, 12, 18)  # months before OSCD date 2
WINDOW = 45  # days either side of the target date
CLOUD_PROB = 40
CLEAR = 0.02  # cloud fraction over the city box below which a day counts as clear


def oscd_dates(city):
    lines = (OSCD_IMAGES / city / "dates.txt").read_text().splitlines()
    return [dt.date(int(x[8:12]), int(x[12:14]), int(x[14:16])) for x in lines if x.strip()]


def box_coords(city):
    """Lower-left and upper-right corners of the OSCD city box, as DS-UNet's notebook reads them."""
    coords = json.loads((OSCD_IMAGES / city / f"{city}.geojson").read_text())["features"][0]["geometry"]["coordinates"][0]
    return coords[0], coords[2]


def box(city):
    return ee.Geometry.Rectangle(list(box_coords(city)))


def grid(city, orbit):
    with rasterio.open(HAFNER_S1 / f"sentinel1_{city}_{orbit}_t1.tif") as src:
        t = src.transform
        return {"crs": "EPSG:4326", "crs_transform": [t.a, t.b, t.c, t.d, t.e, t.f], "dimensions": f"{src.width}x{src.height}"}, src.profile


def fetch(image, g, profile):
    for attempt in range(6):  # Earth Engine returns transient 429/5xx under load
        url = image.getDownloadURL({**g, "format": "GEO_TIFF"})
        r = requests.get(url, timeout=600)
        if r.status_code < 500 and r.status_code != 429 or attempt == 5:
            break
        print("retry after HTTP", r.status_code, flush=True)
        time.sleep(30 * (attempt + 1))
    r.raise_for_status()
    data = r.content
    if data[:2] == b"PK":  # some responses are zipped
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            data = z.read(z.namelist()[0])
    with rasterio.MemoryFile(data) as mem, mem.open() as src:
        assert (src.width, src.height) == (profile["width"], profile["height"]) and src.transform.almost_equals(profile["transform"])
        return src.read()


def download(image, g, profile, path, descriptions, bands=None):
    """Fetch image (or its bands in groups, to stay under Earth Engine's 48 MB request limit) and write a GeoTIFF."""
    if path.exists():
        raise SystemExit(f"{path} exists; refusing to overwrite")
    groups = [bands[i:i + 4] for i in range(0, len(bands), 4)] if bands else [None]
    arr = np.concatenate([fetch(image if grp is None else image.select(grp), g, profile) for grp in groups])
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", width=profile["width"], height=profile["height"], count=arr.shape[0],
                       dtype=arr.dtype, crs=profile["crs"], transform=profile["transform"], compress="deflate") as dst:
        dst.write(arr)
        dst.descriptions = tuple(descriptions)
    return arr


def s1(city):
    d1, d2 = (ee.Date(str(d)) for d in oscd_dates(city))
    r1 = ee.DateRange(d1.advance(-1, "year"), d1.advance(5, "month")) if city == "mumbai" else ee.DateRange(d1.advance(-3, "month"), d1)
    r2 = ee.DateRange(d2, d2.advance(3, "month"))
    for orbit in ORBITS[city]:
        g, profile = grid(city, orbit)
        for t, rng in ((1, r1), (2, r2)):
            path = OUT["s1"] / f"sentinel1_{city}_{orbit}_t{t}.tif"
            if path.exists():  # resume: a file is written only after its download completed
                print(city, orbit, f"t{t}", "exists, skipped", flush=True)
                continue
            bands = []
            for pol in ("VV", "VH"):
                lo, hi = S1_RANGE[pol]
                col = (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(box(city)).filterDate(rng.start(), rng.end())
                       .filterMetadata("instrumentMode", "equals", "IW").filterMetadata("relativeOrbitNumber_start", "equals", orbit)
                       .filter(ee.Filter.listContains("transmitterReceiverPolarisation", pol)).select(pol))
                n = col.size().getInfo()
                col = col.map(lambda img, lo=lo: img.updateMask(img.gte(lo)).copyProperties(img))
                # no scene in the window: all zero, which is what unmask() gives a fully masked mean
                band = col.mean().unitScale(lo, hi).clamp(0, 1).unmask() if n else ee.Image.constant(0)
                bands.append(band.float().rename(pol))
                print(city, orbit, f"t{t}", pol, "images", n, flush=True)
            download(ee.Image.cat(bands), g, profile, path, ["VV", "VH"])


def add_months(d, months):
    m = d.month - 1 + months
    y, m = d.year + m // 12, m % 12 + 1
    return dt.date(y, m, min(d.day, 28))


def s2(city, manifest):
    d1, d2 = oscd_dates(city)
    g, profile = grid(city, ORBITS[city][0])
    roi = box(city)
    targets = {"oscd_t1": d1, **{f"gap{m:02d}": add_months(d2, -m) for m in GAPS}}
    for tag, target in targets.items():
        if any(r["city"] == city and r["tag"] == tag for r in manifest):  # resume: already decided
            continue
        lo, hi = target - dt.timedelta(days=WINDOW), target + dt.timedelta(days=WINDOW)
        col = ee.ImageCollection("COPERNICUS/S2_HARMONIZED").filterBounds(roi).filterDate(str(lo), str(hi + dt.timedelta(days=1)))
        prob = ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY").filterBounds(roi).filterDate(str(lo), str(hi + dt.timedelta(days=1)))
        day_list = lambda c: set(c.aggregate_array("system:time_start").map(lambda t: ee.Date(t).format("YYYY-MM-dd")).getInfo())
        # a day without a cloud-probability image cannot be judged clear, so it is not a candidate
        # days nearest the target first (ties: earlier day; for oscd_t1, OSCD's own day first). The rule below picks the
        # nearest full-coverage clear day, so the scan can stop at the first one; otherwise every day is evaluated.
        dist_day = lambda d: abs((dt.date.fromisoformat(d) - target).days)
        days = sorted(day_list(col) & day_list(prob), key=lambda d: (not (tag == "oscd_t1" and d == str(d1)), dist_day(d), d))
        rows = []
        for day in days:
            nxt = str(dt.date.fromisoformat(day) + dt.timedelta(days=1))
            img, cp = col.filterDate(day, nxt).mosaic(), prob.filterDate(day, nxt).mosaic()
            stats = ee.Image.cat(cp.gte(CLOUD_PROB).rename("cloud"), img.select("B2").mask().rename("valid")).reduceRegion(
                ee.Reducer.mean(), roi, scale=20, maxPixels=1e9).getInfo()
            r = {"day": day, "cloud": stats.get("cloud"), "valid": stats.get("valid")}
            rows.append(r)
            full = r["cloud"] is not None and r["valid"] is not None and r["valid"] >= 0.99
            if full and (r["cloud"] <= CLEAR or (tag == "oscd_t1" and day == str(d1))):
                break
        ok = [r for r in rows if r["cloud"] is not None and r["valid"] is not None and r["valid"] >= 0.99]
        if tag == "oscd_t1":  # same-date control: OSCD's own acquisition day when available
            ok = [r for r in ok if r["day"] == str(d1)] or ok
        if not ok:
            manifest.append({"city": city, "tag": tag, "target": str(target), "day": None, "reason": "no full-coverage scene"})
            save_manifest(manifest)
            print(city, tag, "NO SCENE", flush=True)
            continue
        dist = lambda r: abs((dt.date.fromisoformat(r["day"]) - target).days)
        clear = [r for r in ok if r["cloud"] <= CLEAR]
        # nearest essentially clear day, so the realised gap stays close to the nominal one; else the clearest day
        best = min(clear, key=dist) if clear else min(ok, key=lambda r: (r["cloud"], dist(r)))
        nxt = str(dt.date.fromisoformat(best["day"]) + dt.timedelta(days=1))
        # L1C DN (reflectance x 10000) are integers: uint16 is exact and keeps large cities under the 48 MB request limit
        image = col.filterDate(best["day"], nxt).mosaic().select(S2_BANDS).toUint16()
        stale = OUT["s2"] / f"{city}_{tag}.tif"
        if stale.exists():  # no manifest row yet, so this file is from an interrupted download
            stale.unlink()
        download(image, g, profile, OUT["s2"] / f"{city}_{tag}.tif", ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A",
                                                                     "B09", "B10", "B11", "B12"], bands=S2_BANDS)
        manifest.append({"city": city, "tag": tag, "target": str(target), "day": best["day"], "cloud": best["cloud"],
                         "valid": best["valid"], "candidates": len(rows), "oscd_t1": str(d1), "oscd_t2": str(d2)})
        save_manifest(manifest)
        print(city, tag, target, "->", best["day"], f"cloud {best['cloud']:.3f}", flush=True)


def save_manifest(manifest):
    OUT["s2"].mkdir(parents=True, exist_ok=True)
    (OUT["s2"] / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def wc(city):
    g, profile = grid(city, ORBITS[city][0])
    download(ee.ImageCollection("ESA/WorldCover/v100").first().select("Map").toUint8(), g, profile,
             OUT["wc"] / f"{city}_worldcover.tif", ["Map"])
    print(city, "worldcover", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=("s1", "s2", "wc"))
    p.add_argument("--project", required=True)
    p.add_argument("--cities", nargs="*", default=list(ORBITS))
    a = p.parse_args()
    ee.Initialize(project=a.project)
    f = OUT["s2"] / "manifest.json"
    manifest = json.loads(f.read_text(encoding="utf-8")) if f.exists() else []  # s2 resumes from the saved manifest
    for c in a.cities:
        {"s1": s1, "s2": lambda c: s2(c, manifest), "wc": wc}[a.what](c)
