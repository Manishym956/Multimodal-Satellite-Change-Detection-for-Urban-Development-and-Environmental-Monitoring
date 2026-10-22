"""Step 4A-bis: controlled errors applied to an oracle cloud mask M, giving the detector mask P.

The image is always corrupted with the true mask M (Phase 3 clouds); zero-fill is applied under P. So pixels in M \\ P
keep cloud values (missed cloud), and pixels in P \\ M are clear pixels zero-filled (over-masking).

Variants (radius in 10 m pixels):
    erodeR   P = M eroded by a disk of radius R      (thin cloud edges and small clouds missed)
    dilateR  P = M dilated by a disk of radius R     (clear pixels around clouds masked)
    missQ    P = M minus exactly round(Q*|M|) cloud pixels, removed as whole connected clouds (8-connectivity) in a
             seeded random order, skipping any cloud that would overshoot; any remaining deficit is taken as the
             pixels of the largest remaining cloud nearest to a seeded random point inside it (one contiguous region).
"""

import hashlib

import numpy as np
from scipy import ndimage

from src.phase3.corruption import BASE_SEED

ERODE = (2, 4, 8)
DILATE = (2, 4, 8)
MISS = (0.10, 0.25, 0.50)
VARIANTS = tuple([f"erode{r}" for r in ERODE] + [f"dilate{r}" for r in DILATE] + [f"miss{int(q * 100)}" for q in MISS])


def disk(r: int) -> np.ndarray:
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def variant_seed(city: str, level: str, variant: str) -> int:
    return BASE_SEED + int(hashlib.sha256(f"{city}:{level}:{variant}".encode()).hexdigest()[:8], 16)


def miss_regions(m: np.ndarray, q: float, rng: np.random.Generator) -> np.ndarray:
    target = int(np.round(q * m.sum()))
    labels, n = ndimage.label(m, structure=np.ones((3, 3), dtype=bool))
    sizes = np.bincount(labels.ravel(), minlength=n + 1)
    drop = np.zeros_like(m)
    removed = 0
    for comp in rng.permutation(np.arange(1, n + 1)):
        if removed + sizes[comp] <= target:
            drop |= labels == comp
            removed += int(sizes[comp])
    deficit = target - removed
    if deficit > 0:
        remaining = np.where(drop, 0, labels)
        biggest = int(np.argmax(np.bincount(remaining.ravel(), minlength=n + 1)[1:]) + 1)
        ys, xs = np.nonzero(remaining == biggest)
        i = rng.integers(len(ys))
        order = np.argsort((ys - ys[i]) ** 2 + (xs - xs[i]) ** 2, kind="stable")[:deficit]
        drop[ys[order], xs[order]] = True
    assert int(drop.sum()) == target
    return m & ~drop


def make_detector_mask(m: np.ndarray, city: str, level: str, variant: str) -> np.ndarray:
    if variant.startswith("erode"):
        return ndimage.binary_erosion(m, disk(int(variant[5:])), border_value=1)
    if variant.startswith("dilate"):
        return ndimage.binary_dilation(m, disk(int(variant[6:])))
    return miss_regions(m, int(variant[4:]) / 100, np.random.default_rng(variant_seed(city, level, variant)))


def error_stats(m: np.ndarray, p: np.ndarray) -> dict:
    return {"missed_cloud_frac": float((m & ~p).sum() / m.sum()), "overmasked_clear_frac": float((p & ~m).sum() / (~m).sum()),
            "detector_fraction": float(p.mean())}


if __name__ == "__main__":
    m = np.zeros((60, 60), dtype=bool)
    m[5:25, 5:25] = True
    m[40:44, 40:44] = True
    m[30:50, 10:30] = True
    for v in VARIANTS:
        p = make_detector_mask(m, "selfcheck", "c30", v)
        s = error_stats(m, p)
        if v.startswith("erode"):
            assert not (p & ~m).any() and s["missed_cloud_frac"] > 0
        if v.startswith("dilate"):
            assert not (m & ~p).any() and s["overmasked_clear_frac"] > 0
        if v.startswith("miss"):
            assert not (p & ~m).any() and (m & ~p).sum() == round(int(v[4:]) / 100 * m.sum())
            assert (p == make_detector_mask(m, "selfcheck", "c30", v)).all()
        print(v, {k: round(x, 3) for k, x in s.items()})
    print("mask_errors self-check ok")
