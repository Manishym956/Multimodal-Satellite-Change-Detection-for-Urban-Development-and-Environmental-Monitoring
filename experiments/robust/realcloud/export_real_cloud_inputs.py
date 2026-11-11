"""Step 11: write the cloud detector's inputs for the real Sentinel-2 scenes (no model is run).

Each scene in datasets/oscd_s2_realcloud (download_real_cloud.py) is resized from the Sentinel-1 grid to the city's label
grid with bilinear interpolation, divided by 10000 and clipped to [0, 1], exactly as eval_phase8.historical. Output:
experiments/robust/realcloud/detector_inputs/{city}_{tag}.npz with x (H, W, 10) float32 in Step 5's DETECTOR_BANDS order.
Refuses to overwrite.

Usage (repository root):  .venv/Scripts/python.exe experiments/robust/realcloud/export_real_cloud_inputs.py
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import rasterio
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from src.datasets.oscd_dataset import ORBITS  # noqa: E402
from src.phase5.composite import DETECTOR_BANDS  # noqa: E402
from src.training import train_rrd as rrd  # noqa: E402
from src.training.train_sar_baseline import sample_at_orbit  # noqa: E402

SCENES = ROOT / "datasets" / "oscd_s2_realcloud"
INPUTS = HERE / "detector_inputs"


def scene(city: str, tag: str, shape: tuple[int, int]) -> torch.Tensor:
    """(13, H, W) tensor of one downloaded scene in OSCD's scaling on the label grid."""
    with rasterio.open(SCENES / f"{city}_{tag}.tif") as src:
        bands = [cv2.resize(src.read(i).astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
                 for i in range(1, 14)]
    return torch.from_numpy(np.clip(np.stack(bands) / 10000.0, 0, 1).astype(np.float32))


def manifest() -> list[dict]:
    return json.loads((SCENES / "manifest.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    INPUTS.mkdir(parents=True, exist_ok=True)
    ds = rrd.RRDDataset("test", "hafner_vv")
    for r in manifest():
        f = INPUTS / f"{r['city']}_{r['tag']}.npz"
        if f.exists():
            raise SystemExit(f"{f} exists; refusing to overwrite")
        shape = tuple(sample_at_orbit(ds, ds.cities.index(r["city"]), int(ORBITS[r["city"]][0]))["label"].shape[1:])
        x = scene(r["city"], r["tag"], shape)[list(DETECTOR_BANDS)].permute(1, 2, 0).contiguous().numpy()
        np.savez_compressed(f, x=x.astype(np.float32))
        print(f.name, x.shape, flush=True)
