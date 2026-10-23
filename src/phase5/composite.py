"""Step 5: composite clouds (real catalogue cloud pixels pasted into OSCD post-change images) and detector masks.

Composite for one (split, city, level, replicate k):
    rng = default_rng(composite_seed(city, level, k))
    repeat: pick a scene uniformly from the split's scene pool; take its CLOUD channel (20 m); pick a uniform 20 m window
            covering the city at 10 m after x2 nearest-neighbour upsampling; accept if the cloud fraction of the
            upsampled, cropped window is within ±ACCEPT_REL (relative) of the level.
    The true mask M is that upsampled CLOUD window (no smoothing, no exact-fraction step: pixel values and labels stay paired).
    Composite post-change optical image: inside M, the catalogue subscene's 13 TOA bands (x2 upsampled, clipped to [0, 1]);
    outside M, the OSCD post-change image unchanged. SAR and the pre-change image are never changed.
Scene pools: all catalogue scenes except difficulty 5 ("possibly large errors"), split into a validation pool and a test
pool by the parity of sha256("p5:" + scene), so validation and test composites never share a scene. Scenes behind a Phase 3
mask, in the Step 4B mask bank or in the Phase 3 cloud spectrum are NOT excluded (excluding them leaves no partly cloudy
scene: the bank and Phase 3 together used 142 of the catalogue's 142 scenes with 5-95% cloud); they are flagged per composite.
Only arm C trained on catalogue data (binary 32x32 cloud shapes from the bank scenes), so only C's results are affected.
Detector: s2cloudless cloud probability on the composite post-change image (10 bands, 10 m grid). A detector mask at
operating point (t, r) is (prob >= t) dilated by a disk of radius r (10 m pixels; r = 0 is no dilation).
"""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

from src.phase3.corruption import CATALOGUE, PHASE3, cloud_channel
from src.phase3.mask_errors import disk

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE5 = PROJECT_ROOT / "experiments" / "robust" / "phase5"
LEVELS = {"c10": 0.10, "c30": 0.30, "c60": 0.60}
REPLICATES = 3
ACCEPT_REL = 0.1
BASE_SEED = 20261023
N_S1 = 1
# s2cloudless all_bands=False input order B01, B02, B04, B05, B08, B8A, B09, B10, B11, B12 -> OSCD band indices
DETECTOR_BANDS = (0, 1, 3, 4, 7, 8, 9, 10, 11, 12)
THRESHOLDS = tuple(round(0.05 * i, 2) for i in range(1, 19))  # 0.05 ... 0.90
RADII = (0, 1, 2, 3, 4, 6, 8, 12, 16)


def composite_seed(city: str, level: str, k: int) -> int:
    return BASE_SEED + int(hashlib.sha256(f"p5:{city}:{level}:{k}".encode()).hexdigest()[:8], 16)


def scene_pools() -> tuple[list[str], list[str], dict]:
    """(validation pool, test pool, exclusion and flag record), each pool sorted."""
    p3 = {e["scene"] for m in (PHASE3 / "mask_manifest.json", PHASE3 / "masks_superseded_tol005" / "mask_manifest.json")
          for e in json.loads(m.read_text(encoding="utf-8"))}
    bank = set(json.loads((PHASE3.parent / "phase4b" / "spatial_mask_bank_sources.json").read_text(encoding="utf-8"))["scenes"])
    spectrum = {s["scene"] for s in json.loads((PHASE3 / "cloud_spectrum.json").read_text(encoding="utf-8"))["scenes"]}
    tags = list(csv.DictReader(open(CATALOGUE / "classification_tags.csv", encoding="utf-8")))
    hard = {t["scene"] for t in tags if t["difficulty"] == "5"}
    have = {f.stem for f in (CATALOGUE / "masks_raw" / "masks").glob("*.npy")}
    rest = sorted(have - hard)
    val = [s for s in rest if int(hashlib.sha256(f"p5:{s}".encode()).hexdigest(), 16) % 2 == 0]
    test = [s for s in rest if s not in set(val)]
    return val, test, {"excluded_difficulty5": sorted(hard), "flag_phase3": sorted(p3), "flag_bank": sorted(bank),
                       "flag_spectrum": sorted(spectrum)}


def draw_window(city: str, level: str, k: int, shape: tuple[int, int], pool: list[str]) -> dict:
    """Pick the scene and 20 m window for one composite (masks only; no imagery is read)."""
    target, (h, w) = LEVELS[level], shape
    h20, w20 = (h + 1) // 2, (w + 1) // 2
    rng = np.random.default_rng(composite_seed(city, level, k))
    for attempt in range(1, 1000001):
        scene = pool[rng.integers(len(pool))]
        cloud = cloud_channel(CATALOGUE / "masks_raw" / "masks" / f"{scene}.npy")
        y, x = int(rng.integers(0, cloud.shape[0] - h20 + 1)), int(rng.integers(0, cloud.shape[1] - w20 + 1))
        m = upsample(cloud[y:y + h20, x:x + w20], shape)
        if abs(m.mean() - target) <= ACCEPT_REL * target:
            return {"city": city, "level": level, "k": k, "seed": composite_seed(city, level, k), "scene": scene,
                    "window_yx_20m": [y, x], "window_hw_20m": [h20, w20], "shape": [h, w], "attempts": attempt,
                    "fraction": float(m.mean())}
    raise RuntimeError(f"no window for {city} {level} {k}")


def upsample(a: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """x2 nearest-neighbour (20 m -> 10 m), cropped to the city's 10 m shape. Works on (h, w) and (h, w, c)."""
    return a.repeat(2, 0).repeat(2, 1)[:shape[0], :shape[1]]


def composite_post(t2, mask, cloud_bands):
    """t2: (B, 14, H, W) torch tensor (VV + 13 S2). mask: (H, W) bool tensor. cloud_bands: (13, H, W) tensor in [0, 1]."""
    t2 = t2.clone()
    t2[:, N_S1:] = t2[:, N_S1:].where(~mask[None, None], cloud_bands[None])
    return t2


def detector_mask(prob: np.ndarray, t: float, r: int) -> np.ndarray:
    m = prob >= t
    return ndimage.binary_dilation(m, disk(r)) if r > 0 else m


def mask_counts(m: np.ndarray, p: np.ndarray) -> np.ndarray:
    """(tp, fp, fn, tn) of detector mask p against true mask m."""
    return np.array([(m & p).sum(), (~m & p).sum(), (m & ~p).sum(), (~m & ~p).sum()], dtype=np.int64)
