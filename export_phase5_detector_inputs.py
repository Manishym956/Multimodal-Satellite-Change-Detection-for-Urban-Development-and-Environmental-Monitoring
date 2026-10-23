"""Step 5: write the cloud detector's inputs, i.e. the 10 s2cloudless bands of every post-change optical image the detector sees.

For each city of the split: the clean OSCD post-change image ({city}_clean) and every composite ({city}_{level}_k{k},
src/phase5/composite.py). Output: experiments/robust/phase5/detector_inputs/{split}/{name}.npz with x (H, W, 10) float32,
bands in DETECTOR_BANDS order, plus the true mask M for composites. No model is run. Refuses to overwrite a file.

Usage:
    .venv/Scripts/python.exe export_phase5_detector_inputs.py val|test
"""

import json
import sys

import numpy as np
import torch

from src.datasets.oscd_dataset import ORBITS, OSCDDataset
from src.phase5.composite import DETECTOR_BANDS, LEVELS, N_S1, PHASE5, REPLICATES, composite_post, upsample
from src.training.train_sar_baseline import sample_at_orbit, split_train_cities


def city_inputs(ds, city):
    """(post-change tensor (1, 14, H, W), {name: (composite tensor, M)}) for one city."""
    t2 = sample_at_orbit(ds, ds.cities.index(city), int(ORBITS[city][0]))["t2_img"][None]
    shape = tuple(t2.shape[-2:])
    out = {}
    for level in LEVELS:
        for k in range(REPLICATES):
            z = np.load(PHASE5 / "composites" / f"{city}_{level}_k{k}.npz")
            m = torch.from_numpy(upsample(z["cloud20"], shape))
            bands = torch.from_numpy(np.ascontiguousarray(upsample(z["bands20"], shape).transpose(2, 0, 1)))
            out[f"{city}_{level}_k{k}"] = (composite_post(t2, m, bands), m)
    return t2, out


def detector_input(t2):
    return t2[0, N_S1:][list(DETECTOR_BANDS)].permute(1, 2, 0).contiguous().numpy().astype(np.float32)


if __name__ == "__main__":
    split = sys.argv[1]
    assert split in ("val", "test")
    ds = OSCDDataset(split="train" if split == "val" else "test", mode="fusion")
    cities = split_train_cities(ds.cities)[1] if split == "val" else list(ds.cities)
    out = PHASE5 / "detector_inputs" / split
    out.mkdir(parents=True, exist_ok=True)
    manifest = {e["file"][:-4]: e for e in json.loads((PHASE5 / "composite_manifest.json").read_text(encoding="utf-8"))["entries"]}
    for city in cities:
        t2, comps = city_inputs(ds, city)
        for name, x, m in [(f"{city}_clean", t2, None)] + [(n, x, m) for n, (x, m) in comps.items()]:
            f = out / f"{name}.npz"
            if f.exists():
                raise SystemExit(f"{f} exists; refusing to overwrite")
            if m is not None:
                assert manifest[name]["split"] == split and float(m.numpy().mean()) == manifest[name]["fraction"]  # float64, as in the builder
                assert torch.equal(x[:, :N_S1], t2[:, :N_S1]) and torch.equal(x[0, N_S1:, ~m], t2[0, N_S1:, ~m])
            np.savez_compressed(f, x=detector_input(x), **({} if m is None else {"m": m.numpy()}))
        print(split, city, len(comps) + 1, "inputs", flush=True)
