"""Build the fixed Phase 3 corruption assets (no model, no OSCD label is read).

1. Cloud spectrum: K_SCENES catalogue scenes drawn with BASE_SEED from those tagged thick=1, thin=0,
   snow/ice=0, difficulty<=2, cloud_percent>=30, dataset MAIN. From each, SAMPLES_PER_SCENE CLOUD pixels
   are drawn uniformly; the spectrum is the per-band median of the pooled pixels, clipped to [0, 1]
   (OSCD inputs are clipped to [0, 1]). Scenes are read by HTTP range from the Zenodo zip and not stored.
2. Masks: one per (city, level) for the 4 validation and 10 test cities, see src/phase3/corruption.py.

Writes experiments/robust/phase3/{cloud_spectrum.json, masks/<city>_<level>.npy, mask_manifest.json}.
Refuses to overwrite existing assets.

Usage:
    .venv/Scripts/python.exe build_phase3_corruptions.py
"""

import csv
import hashlib
import json

import numpy as np

from src.datasets.oscd_dataset import OSCDDataset
from src.phase3.corruption import BASE_SEED, CATALOGUE, LEVELS, PHASE3, catalogue_mask_files, make_mask
from src.phase3.remote_zip import open_remote_zip
from src.training.train_sar_baseline import split_train_cities

SUBSCENES_URL = "https://zenodo.org/api/records/4172871/files/subscenes.zip/content"
K_SCENES = 8
SAMPLES_PER_SCENE = 50000


def build_spectrum() -> None:
    with open(CATALOGUE / "classification_tags.csv", newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["thick"] == "1" and r["thin"] == "0" and r["snow/ice"] == "0"
                and int(r["difficulty"]) <= 2 and float(r["cloud_percent"]) >= 30 and r["dataset"] == "MAIN"]
    rng = np.random.default_rng(BASE_SEED)
    chosen = sorted(rng.choice([r["scene"] for r in rows], K_SCENES, replace=False))
    z = open_remote_zip(SUBSCENES_URL)
    pooled, per_scene = [], []
    for scene in chosen:
        info = z.getinfo(f"subscenes/{scene}.npy")
        image = np.load(z.open(info), allow_pickle=False)
        cloud = np.load(CATALOGUE / "masks_raw" / "masks" / f"{scene}.npy", allow_pickle=False)[..., 1]
        pixels = image[cloud]
        pixels = pixels[rng.choice(len(pixels), min(SAMPLES_PER_SCENE, len(pixels)), replace=False)]
        pooled.append(pixels)
        per_scene.append({"scene": scene, "zip_crc32": info.CRC, "n_cloud_pixels": int(cloud.sum()),
                          "median": np.median(pixels, 0).round(5).tolist()})
        print(scene, np.median(pixels, 0).round(3), flush=True)
    spectrum = np.clip(np.median(np.concatenate(pooled), 0), 0, 1)
    doc = {"bands": ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"],
           "spectrum": spectrum.round(6).tolist(), "eligible_scenes": len(rows), "scenes": per_scene,
           "rule": __doc__.split("2. Masks")[0].strip(), "source": "Sentinel-2 Cloud Mask Catalogue, Zenodo 4172871, CC-BY-4.0"}
    (PHASE3 / "cloud_spectrum.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")


def build_masks() -> None:
    files = catalogue_mask_files()
    train_ds, test_ds = OSCDDataset(split="train", mode="fusion"), OSCDDataset(split="test", mode="fusion")
    _, val_cities = split_train_cities(train_ds.cities)
    (PHASE3 / "masks").mkdir(parents=True)
    manifest = []
    for split, ds, cities in (("val", train_ds, val_cities), ("test", test_ds, list(test_ds.cities))):
        for city in cities:
            shape = tuple(ds[ds.cities.index(city)]["label"].shape[1:])
            for level in LEVELS:
                mask, info = make_mask(city, level, shape, files)
                path = PHASE3 / "masks" / f"{city}_{level}.npy"
                np.save(path, mask)
                manifest.append({"split": split, **info, "file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
                print(split, city, level, info["scene"][:40], f"raw {info['raw_fraction']:.3f} -> {info['fraction']:.4f}", flush=True)
    (PHASE3 / "mask_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    PHASE3.mkdir(parents=True, exist_ok=True)
    # each asset is built only if absent; delete it to rebuild
    if not (PHASE3 / "masks").exists():
        build_masks()
    if not (PHASE3 / "cloud_spectrum.json").exists():
        build_spectrum()
