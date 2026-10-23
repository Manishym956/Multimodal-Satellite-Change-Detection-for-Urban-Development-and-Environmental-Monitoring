"""Step 5: run the s2cloudless cloud detector on the exported detector inputs. Runs in the separate detector venv.

Input: experiments/robust/phase5/detector_inputs/{split}/{name}.npz (x: (H, W, 10) TOA reflectance).
Output: experiments/robust/phase5/detector_outputs/{split}/{name}.npz with
    prob     (H, W) float32, S2PixelCloudDetector().get_cloud_probability_maps (no smoothing)
    default  (H, W) bool, the library's default mask: S2PixelCloudDetector() defaults (threshold 0.4, average_over 1,
             dilation_size 1) applied with get_mask_from_prob
plus detector_outputs/{split}/detector_info.json (library version, defaults, model sha256, per-file sha256). Refuses to overwrite.

Usage:
    .venv_cloud/Scripts/python.exe detect_phase5.py val|test
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import s2cloudless
from lightgbm import Booster
from s2cloudless import S2PixelCloudDetector
from s2cloudless.pixel_classifier import PixelClassifier

PHASE5 = Path(__file__).resolve().parent / "experiments" / "robust" / "phase5"

if __name__ == "__main__":
    split = sys.argv[1]
    assert split in ("val", "test")
    src, out = PHASE5 / "detector_inputs" / split, PHASE5 / "detector_outputs" / split
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    out.mkdir(parents=True)
    det = S2PixelCloudDetector()
    # LightGBM's C loader cannot open a path containing the non-ASCII hyphen in this project's path, so the packaged
    # model file is read by Python and passed as a string; the classifier is otherwise exactly the library's.
    model = Path(det.model_filename)
    det._classifier = PixelClassifier(Booster(model_str=model.read_text()))
    info = {"s2cloudless": s2cloudless.__version__, "numpy": np.__version__, "threshold": det.threshold,
            "average_over": det.average_over, "dilation_size": det.dilation_size, "all_bands": det.all_bands,
            "model_file": model.name, "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(), "files": {}}
    for f in sorted(src.glob("*.npz")):
        x = np.load(f)["x"]
        prob = det.get_cloud_probability_maps(x[None])[0].astype(np.float32)
        default = det.get_mask_from_prob(prob[None])[0].astype(bool)
        g = out / f.name
        np.savez_compressed(g, prob=prob, default=default)
        info["files"][f.name] = {"input_sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
                                 "output_sha256": hashlib.sha256(g.read_bytes()).hexdigest()}
        print(split, f.stem, f"mean prob {prob.mean():.3f}", flush=True)
    (out / "detector_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(len(info["files"]), "files")
