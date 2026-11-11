"""Step 11: run s2cloudless on the real-scene detector inputs, exactly as detect_phase5.py (runs in the detector venv).

Input experiments/robust/realcloud/detector_inputs/{name}.npz; output detector_outputs/{name}.npz with prob (H, W) float32
(get_cloud_probability_maps, no smoothing) and default (the library's default mask), plus detector_info.json. Refuses to
overwrite.

Usage (repository root):  .venv_cloud/Scripts/python.exe experiments/robust/realcloud/detect_real_cloud.py
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import s2cloudless
from lightgbm import Booster
from s2cloudless import S2PixelCloudDetector
from s2cloudless.pixel_classifier import PixelClassifier

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    src, out = HERE / "detector_inputs", HERE / "detector_outputs"
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    out.mkdir(parents=True)
    det = S2PixelCloudDetector()
    model = Path(det.model_filename)  # read by Python: LightGBM's loader cannot open this project's non-ASCII path
    det._classifier = PixelClassifier(Booster(model_str=model.read_text()))
    info = {"s2cloudless": s2cloudless.__version__, "numpy": np.__version__, "threshold": det.threshold,
            "average_over": det.average_over, "dilation_size": det.dilation_size, "all_bands": det.all_bands,
            "model_file": model.name, "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(), "files": {}}
    for f in sorted(src.glob("*.npz")):
        prob = det.get_cloud_probability_maps(np.load(f)["x"][None])[0].astype(np.float32)
        default = det.get_mask_from_prob(prob[None])[0].astype(bool)
        g = out / f.name
        np.savez_compressed(g, prob=prob, default=default)
        info["files"][f.name] = {"input_sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
                                 "output_sha256": hashlib.sha256(g.read_bytes()).hexdigest()}
        print(f.stem, flush=True)
    (out / "detector_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(len(info["files"]), "files")
