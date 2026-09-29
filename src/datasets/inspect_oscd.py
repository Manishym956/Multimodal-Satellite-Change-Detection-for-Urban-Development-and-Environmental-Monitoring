"""Read-only validation of one OSCD city before training.

Run from the project root:

    python src/datasets/inspect_oscd.py
"""

import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OSCD_ROOT = PROJECT_ROOT / "datasets" / "oscd"
S1_ROOT = PROJECT_ROOT / "datasets" / "oscd_sentinel1"

CITY = "abudhabi"
BANDS = [
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B10",
    "B11",
    "B12",
]


def raster_path(date_folder: str, band: str) -> Path:
    return OSCD_ROOT / "images" / CITY / date_folder / f"{band}.tif"


def nan_count(array: np.ndarray) -> int:
    if np.issubdtype(array.dtype, np.floating):
        return int(np.isnan(array).sum())
    return 0


def describe_raster(path: Path) -> dict:
    with rasterio.open(path) as dataset:
        array = dataset.read()
        crs = dataset.crs
        transform = dataset.transform
        resolution = dataset.res
        stored_georef = crs is not None and transform != Affine.identity()
        return {
            "path": path,
            "shape": array.shape,
            "height": dataset.height,
            "width": dataset.width,
            "bands": dataset.count,
            "dtype": str(array.dtype),
            "min": float(np.nanmin(array)),
            "max": float(np.nanmax(array)),
            "mean": float(np.nanmean(array)),
            "nan": nan_count(array),
            "crs": crs,
            "transform": transform,
            "resolution": resolution if stored_georef else None,
            "raw_resolution": resolution,
            "spatial": (dataset.height, dataset.width),
            "array": array,
        }


def print_raster(info: dict) -> None:
    print(f"path: {info['path']}")
    print(f"shape: {info['shape']}")
    print(f"bands: {info['bands']}")
    print(f"dtype: {info['dtype']}")
    print(f"min: {info['min']}")
    print(f"max: {info['max']}")
    print(f"mean: {info['mean']}")
    print(f"nan: {info['nan']}")
    if info["crs"] is None:
        print("crs: none stored in file")
    else:
        print(f"crs: {info['crs']}")
    print(f"transform: {info['transform']}")
    if info["resolution"] is None:
        print("resolution: none stored in file")
    else:
        print(f"resolution: {info['resolution']}")
    print()


def load_date(date_folder: str) -> list[dict]:
    loaded = []
    for band in BANDS:
        path = raster_path(date_folder, band)
        if not path.is_file():
            raise FileNotFoundError(path)
        loaded.append(describe_raster(path))
    return loaded


def same_dimensions(rasters: list[dict]) -> bool:
    sizes = {item["spatial"] for item in rasters}
    return len(sizes) == 1


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    checks: list[tuple[str, bool]] = []

    dates_path = OSCD_ROOT / "images" / CITY / "dates.txt"
    print("dates.txt")
    print(f"path: {dates_path}")
    dates_text = dates_path.read_text(encoding="utf-8")
    print(dates_text.rstrip())
    print()
    checks.append(("dates.txt loaded", dates_path.is_file() and bool(dates_text.strip())))

    print("Sentinel-2 T1 rectified bands")
    t1 = load_date("imgs_1_rect")
    for info in t1:
        print_raster(info)
    checks.append(("Sentinel-2 T1 13 bands loaded", len(t1) == len(BANDS)))

    print("Sentinel-2 T2 rectified bands")
    t2 = load_date("imgs_2_rect")
    for info in t2:
        print_raster(info)
    checks.append(("Sentinel-2 T2 13 bands loaded", len(t2) == len(BANDS)))

    label_path = OSCD_ROOT / "train_labels" / CITY / "cm" / f"{CITY}-cm.tif"
    print("change mask")
    label = describe_raster(label_path)
    print_raster(label)
    values, counts = np.unique(label["array"], return_counts=True)
    print("unique values and pixel counts:")
    for value, count in zip(values.tolist(), counts.tolist()):
        print(f"  {value}: {count}")
    print()
    checks.append(("change mask loaded", label_path.is_file()))

    print("Sentinel-1 files containing 'abudhabi'")
    matches = sorted(
        path for path in S1_ROOT.iterdir() if path.is_file() and "abudhabi" in path.name
    )
    if not matches:
        print("no matching files")
    for path in matches:
        print(path.name)
    print()
    s1 = []
    if len(matches) == 2:
        print("Sentinel-1 rasters")
        for path in matches:
            info = describe_raster(path)
            s1.append(info)
            print_raster(info)
    checks.append(("exactly two Sentinel-1 files found and opened", len(s1) == 2))

    t1_same = same_dimensions(t1)
    t2_same = same_dimensions(t2)
    t1_t2_same = t1[0]["spatial"] == t2[0]["spatial"] if t1_same and t2_same else False
    mask_same = label["spatial"] == t1[0]["spatial"] if t1_same else False
    checks.append(("Sentinel-2 T1 bands have identical dimensions", t1_same))
    checks.append(("Sentinel-2 T2 bands have identical dimensions", t2_same))
    checks.append(("Sentinel-2 T1 and T2 dimensions match", t1_t2_same))
    checks.append(("change mask dimensions match rectified Sentinel-2", mask_same))

    print("CRS metadata")
    groups = [("Sentinel-2 T1", t1), ("Sentinel-2 T2", t2), ("change mask", [label])]
    if s1:
        groups.append(("Sentinel-1", s1))
    for name, rasters in groups:
        present = sum(item["crs"] is not None for item in rasters)
        print(f"{name}: {present}/{len(rasters)} files store a CRS")
    print()

    print("validation summary")
    for name, passed in checks:
        print(f"{'PASS' if passed else 'FAIL'}  {name}")


if __name__ == "__main__":
    main()
