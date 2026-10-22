"""Phase 3 optical corruption: real cloud shapes, simulated cloud reflectance.

Masks come from the Sentinel-2 Cloud Mask Catalogue (Francis, Mrziglod, Sidiropoulos, Muller 2020,
Zenodo 10.5281/zenodo.4172871, CC-BY-4.0): 1022x1022 one-hot masks at 20 m, classes CLEAR, CLOUD, CLOUD_SHADOW.

Mask for one (city, level):
    rng = default_rng(mask_seed(city, level))
    repeat: pick a catalogue mask uniformly, upsample CLOUD x2 (nearest, 20 m -> 10 m OSCD grid),
            pick a uniform window of the city's size; accept if |cloud fraction - level| <= ACCEPT_REL * level
    then make the fraction exact: blur the window (Gaussian, sigma BLUR_SIGMA px), add rng tie-break
    noise of 1e-6, and mark the k = round(level * H * W) highest pixels (k pixels exactly).
Reflectance (applied to S2 post-change bands only, after OSCD preprocessing /10000 + clip):
    x' = clip((1 - alpha) * x + alpha * c_b, 0, 1)
    c_b = per-band cloud spectrum (median TOA reflectance of CLOUD pixels in fixed catalogue scenes)
    alpha = 1 on mask pixels (opaque), 0 elsewhere; for haze alpha = HAZE_ALPHA everywhere.
"""

import functools
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CATALOGUE = PROJECT_ROOT / "datasets" / "s2_cloud_mask_catalogue"
PHASE3 = PROJECT_ROOT / "experiments" / "robust" / "phase3"
LEVELS = {"c10": 0.10, "c30": 0.30, "c60": 0.60}
ACCEPT_REL = 0.1  # accept a real window within ±10 % (relative) of the target, so the exact-fraction step changes few pixels
BLUR_SIGMA = 2.0
HAZE_ALPHA = 0.3
BASE_SEED = 20261021
N_S1 = 1


def mask_seed(city: str, level: str) -> int:
    """Deterministic seed from the names, independent of city ordering."""
    return BASE_SEED + int(hashlib.sha256(f"{city}:{level}".encode()).hexdigest()[:8], 16)


@functools.lru_cache(maxsize=None)
def cloud_channel(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=False)[..., 1]


def catalogue_mask_files() -> list[Path]:
    return sorted((CATALOGUE / "masks_raw" / "masks").glob("*.npy"))


def make_mask(city: str, level: str, shape: tuple[int, int], files: list[Path]) -> tuple[np.ndarray, dict]:
    target = LEVELS[level]
    rng = np.random.default_rng(mask_seed(city, level))
    h, w = shape
    for attempt in range(1, 1000001):
        f = files[rng.integers(len(files))]
        cloud = cloud_channel(f)
        cloud = cloud.repeat(2, 0).repeat(2, 1)
        y, x = rng.integers(0, cloud.shape[0] - h + 1), rng.integers(0, cloud.shape[1] - w + 1)
        window = cloud[y:y + h, x:x + w]
        raw = float(window.mean())
        if abs(raw - target) <= ACCEPT_REL * target:
            break
    else:
        raise RuntimeError(f"no window found for {city} {level}")
    field = gaussian_filter(window.astype(np.float64), BLUR_SIGMA) + rng.random(window.shape) * 1e-6
    k = int(np.round(target * h * w))
    mask = np.zeros(h * w, dtype=bool)
    mask[np.argsort(field, axis=None)[::-1][:k]] = True
    mask = mask.reshape(h, w)
    info = {"city": city, "level": level, "seed": mask_seed(city, level), "scene": f.stem, "window_yx_10m": [int(y), int(x)],
            "shape": [h, w], "attempts": attempt, "raw_fraction": raw, "k": k, "fraction": float(mask.mean()),
            "agreement_with_raw": float((mask == window).mean())}
    return mask, info


def load_spectrum() -> np.ndarray:
    doc = json.loads((PHASE3 / "cloud_spectrum.json").read_text(encoding="utf-8"))
    return np.asarray(doc["spectrum"], dtype=np.float32)


def corrupt_post(t2, mask, spectrum, alpha: float = 1.0):
    """t2: (B, 14, H, W) torch tensor (VV + 13 S2). mask: (H, W) bool tensor or None (= whole image)."""
    t2 = t2.clone()
    c = spectrum.view(1, -1, 1, 1)
    opt = t2[:, N_S1:]
    blended = ((1 - alpha) * opt + alpha * c).clamp(0, 1)
    t2[:, N_S1:] = blended if mask is None else opt.where(~mask[None, None], blended)
    return t2


if __name__ == "__main__":
    # self-check: exact fraction, determinism, corruption formula
    import torch
    files = catalogue_mask_files()
    m1, i1 = make_mask("selfcheck", "c30", (101, 77), files)
    m2, _ = make_mask("selfcheck", "c30", (101, 77), files)
    assert (m1 == m2).all() and m1.sum() == round(0.3 * 101 * 77), i1
    t2 = torch.rand(1, 14, 101, 77)
    spec = torch.full((13,), 0.5)
    out = corrupt_post(t2, torch.from_numpy(m1), spec)
    assert torch.equal(out[:, 0], t2[:, 0]) and torch.all(out[0, 1:, torch.from_numpy(m1)] == 0.5)
    assert torch.equal(out[0, 1:, ~torch.from_numpy(m1)], t2[0, 1:, ~torch.from_numpy(m1)])
    haze = corrupt_post(t2, None, spec, HAZE_ALPHA)
    assert torch.allclose(haze[:, 1:], 0.7 * t2[:, 1:] + 0.3 * 0.5)
    print("corruption self-check ok", i1)
