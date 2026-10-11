"""SEN12-MT Urban Mapping dataset loader for Scenario 2.

Dataset: SEN12-MT Urban Mapping (SpaceNet 7 sites)
Root:    datasets/sen12_mt_urbanmapping/multimodal_cd_dataset/

Temporal pairing (Strategy B, locked in split_definition.json):
  T1 = March 2018    (2018-03)
  T2 = December 2019 (2019-12)
  Interval: 21 months, uniform across all 28 common sites.

Label derivation (threshold 0.5):
  change_mask = (buildings_T2 >= 0.5) XOR (buildings_T1 >= 0.5)
  Positive class = pixel changed from non-building to building, or vice versa.
  This is a derived proxy label, not a directly annotated change mask.

Modalities:
  SAR:     2 bands per date (VV, VH), 2 dates → 4 channels total
           Already normalised to [0, 1]. NaN pixels filled with 0.0.
  Optical: 10 bands at T1 only (B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12)
           Already normalised to [0, 1]. Do NOT divide by 10000.
  S2 T2:   NEVER loaded or used in asynchronous models.

Modes:
  "async"  — 4 SAR + 10 optical T1 = 14 channels
  "sync"   — 4 SAR + 10 optical T1 + 10 optical T2 = 24 channels

Split (seed 42, common to all three models):
  20 train, 4 validation, 4 test  (from split_definition.json)

Preprocessing invariants:
  - S1 NaN → 0.0 (fill before tensor construction)
  - S1 already in [0, 1]: no further scaling
  - S2 already in [0, 1]: no further scaling, no /10000
  - Buildings threshold: 0.5
  - All three modalities co-registered, 489×488 @ 10 m per site (approx.)
"""

import json
import warnings
from pathlib import Path

import numpy as np
import torch
from rasterio.errors import NotGeoreferencedWarning
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

try:
    import rasterio
except ImportError as e:
    raise ImportError("rasterio is required. Install from requirements.txt.") from e

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "datasets" / "sen12_mt_urbanmapping" / "multimodal_cd_dataset"
SPLIT_FILE   = PROJECT_ROOT / "experiments" / "scenario_2" / "configs" / "split_definition.json"

# ── fixed temporal pair ───────────────────────────────────────────────────────
T1_YEAR, T1_MONTH = 2018, 3    # March 2018
T2_YEAR, T2_MONTH = 2019, 12   # December 2019
LABEL_THRESHOLD    = 0.5

# ── channel constants ─────────────────────────────────────────────────────────
SAR_BANDS     = ("VV", "VH")       # 2 bands per date
SAR_CHANNELS  = 4                  # 2 bands × 2 dates
OPT_BANDS     = ("B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12")
OPT_CHANNELS  = 10                 # bands at T1 only (S2 T2 never loaded)
ASYNC_CHANNELS = SAR_CHANNELS + OPT_CHANNELS          # 14
SYNC_CHANNELS  = SAR_CHANNELS + OPT_CHANNELS * 2      # 24  (adds S2 T2)


# ── split loading ─────────────────────────────────────────────────────────────

def load_split() -> dict[str, list[str]]:
    """Read the locked split from split_definition.json."""
    if not SPLIT_FILE.exists():
        raise FileNotFoundError(
            f"Split file not found: {SPLIT_FILE}\n"
            "Run src/training/split_feasibility.py to generate it."
        )
    doc = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
    return {
        "train":      list(doc["train"]),
        "validation": list(doc["validation"]),
        "test":       list(doc["test"]),
    }


# ── file path helpers ─────────────────────────────────────────────────────────

def _s1_path(site_id: str, year: int, month: int) -> Path:
    return DATASET_ROOT / site_id / "s1" / f"s1_{site_id}_{year}_{month:02d}.tif"

def _s2_path(site_id: str, year: int, month: int) -> Path:
    return DATASET_ROOT / site_id / "s2" / f"s2_{site_id}_{year}_{month:02d}.tif"

def _bld_path(site_id: str, year: int, month: int) -> Path:
    return DATASET_ROOT / site_id / "buildings" / f"buildings_{site_id}_{year}_{month:02d}.tif"


# ── raster reading ────────────────────────────────────────────────────────────

def _read_s1_with_validity(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a 2-band S1 file as float32 (H, W, 2).
    Returns:
        (arr, valid_mask):
            arr: (H, W, 2) float32 with NaNs filled to 0.0
            valid_mask: (H, W) bool where True means neither VV nor VH is NaN
    """
    with rasterio.open(path) as ds:
        arr = ds.read().astype(np.float32)   # (2, H, W)
    valid_mask = ~np.any(np.isnan(arr), axis=0)  # (H, W)
    arr = np.nan_to_num(arr, nan=0.0)
    return arr.transpose(1, 2, 0), valid_mask


def _read_s1(path: Path) -> np.ndarray:
    """Read a 2-band S1 file as float32 (H, W, 2). NaN → 0.0."""
    arr, _ = _read_s1_with_validity(path)
    return arr


def _read_s2(path: Path) -> np.ndarray:
    """Read a 10-band S2 file as float32 (H, W, 10). Already in [0, 1]."""
    with rasterio.open(path) as ds:
        arr = ds.read().astype(np.float32)   # (10, H, W)
    return arr.transpose(1, 2, 0)            # (H, W, 10)


def _read_bld(path: Path) -> np.ndarray:
    """Read a building probability map as float32 (H, W)."""
    with rasterio.open(path) as ds:
        arr = ds.read(1).astype(np.float32)  # (H, W)
    return arr


# ── preprocessing ─────────────────────────────────────────────────────────────

def preprocess_site(site_id: str) -> dict[str, np.ndarray]:
    """Load and preprocess one site for the locked T1/T2 pair.

    Returns a dict with keys:
        s1_t1       (H, W, 2) float32 VV+VH at T1 in [0, 1], NaN→0
        s1_t2       (H, W, 2) float32 VV+VH at T2 in [0, 1], NaN→0
        s2_t1       (H, W, 10) float32 optical at T1 in [0, 1]
        s2_t2       (H, W, 10) float32 optical at T2 in [0, 1]  ← sync only
        label       (H, W) uint8, 0=no-change, 1=change
        valid_mask  (H, W, 1) float32 in {0.0, 1.0}, valid in both S1 T1 and T2
    """
    s1_t1, v1 = _read_s1_with_validity(_s1_path(site_id, T1_YEAR, T1_MONTH))
    s1_t2, v2 = _read_s1_with_validity(_s1_path(site_id, T2_YEAR, T2_MONTH))
    s2_t1     = _read_s2(_s2_path(site_id, T1_YEAR, T1_MONTH))
    s2_t2     = _read_s2(_s2_path(site_id, T2_YEAR, T2_MONTH))
    bld_t1    = _read_bld(_bld_path(site_id, T1_YEAR, T1_MONTH))
    bld_t2    = _read_bld(_bld_path(site_id, T2_YEAR, T2_MONTH))

    # Derive binary change mask
    label = ((bld_t1 >= LABEL_THRESHOLD) ^ (bld_t2 >= LABEL_THRESHOLD)).astype(np.uint8)

    # Pixel is valid if neither band is NaN at T1 AND neither band is NaN at T2
    sar_valid = (v1 & v2).astype(np.float32)[:, :, np.newaxis]  # (H, W, 1)

    return {
        "s1_t1":      s1_t1.astype(np.float32),
        "s1_t2":      s1_t2.astype(np.float32),
        "s2_t1":      s2_t1.astype(np.float32),
        "s2_t2":      s2_t2.astype(np.float32),
        "label":      label,
        "valid_mask": sar_valid,
    }


# ── dataset class ─────────────────────────────────────────────────────────────

class Scenario2Dataset(Dataset):
    """SEN12-MT Urban Mapping dataset for Scenario 2.

    Args:
        split:  "train", "validation", or "test".
        mode:   "async" (14 ch) or "sync" (24 ch).
                async = SAR T1+T2 + S2 T1 only (S2 T2 withheld).
                sync  = SAR T1+T2 + S2 T1 + S2 T2.

    Returns per __getitem__:
        image   (C, H, W) float32 tensor where C = 14 (async) or 24 (sync)
        label   (1, H, W) float32 tensor in {0.0, 1.0}
        site    str — site ID
    """

    CHANNEL_ORDER_ASYNC = (
        ["s1_t1_vv", "s1_t1_vh", "s1_t2_vv", "s1_t2_vh"]
        + [f"s2_t1_{b}" for b in OPT_BANDS]
    )
    CHANNEL_ORDER_SYNC = CHANNEL_ORDER_ASYNC + [f"s2_t2_{b}" for b in OPT_BANDS]

    def __init__(self, split: str = "train", mode: str = "async", return_mask: bool = False) -> None:
        if split not in ("train", "validation", "test"):
            raise ValueError("split must be 'train', 'validation', or 'test'")
        if mode not in ("async", "sync"):
            raise ValueError("mode must be 'async' or 'sync'")

        self.split = split
        self.mode  = mode
        self.return_mask = return_mask
        all_splits = load_split()
        self.sites = all_splits[split]
        self._cache: dict[str, dict[str, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.sites)

    def _arrays(self, site_id: str) -> dict[str, np.ndarray]:
        if site_id not in self._cache:
            self._cache[site_id] = preprocess_site(site_id)
        return self._cache[site_id]

    @property
    def input_channels(self) -> int:
        return ASYNC_CHANNELS if self.mode == "async" else SYNC_CHANNELS

    def __getitem__(self, index: int) -> dict:
        site_id = self.sites[index]
        arrays  = self._arrays(site_id)

        # S1: stack VV+VH for T1, then VV+VH for T2 → (H, W, 4)
        # Shape: (H, W, 2) cat along axis 2 → (H, W, 4)
        sar = np.concatenate([arrays["s1_t1"], arrays["s1_t2"]], axis=2)  # (H,W,4)

        # S2 T1: (H, W, 10)
        s2_t1 = arrays["s2_t1"]

        if self.mode == "async":
            # 14 channels: 4 SAR + 10 optical T1
            combined = np.concatenate([sar, s2_t1], axis=2)   # (H,W,14)
        else:
            # 24 channels: 4 SAR + 10 optical T1 + 10 optical T2
            s2_t2 = arrays["s2_t2"]
            combined = np.concatenate([sar, s2_t1, s2_t2], axis=2)  # (H,W,24)

        # label (H, W) → (1, H, W) float32
        label_np = arrays["label"].astype(np.float32)[:, :, np.newaxis]

        image = TF.to_tensor(combined)    # (C, H, W) float32
        label = TF.to_tensor(label_np)   # (1, H, W) float32 in {0., 1.}

        assert image.shape[0] == self.input_channels, (
            f"{site_id}: expected {self.input_channels} channels, got {image.shape[0]}"
        )

        out = {"image": image, "label": label, "site": site_id}
        if self.return_mask:
            out["valid_mask"] = TF.to_tensor(arrays["valid_mask"])  # (1, H, W) float32 in {0., 1.}
        return out


# ── random crop dataset ───────────────────────────────────────────────────────

class RandomCropScenario2(Dataset):
    """Draw aligned random crops from full-site Scenario 2 samples."""

    def __init__(self, base: Scenario2Dataset, crop_size: int, crops_per_site: int) -> None:
        self.base = base
        self.crop_size = crop_size
        self.crops_per_site = crops_per_site

    def __len__(self) -> int:
        return len(self.base) * self.crops_per_site

    def __getitem__(self, index: int) -> dict:
        sample = self.base[index % len(self.base)]
        image, label = sample["image"], sample["label"]
        _, h, w = image.shape
        cs = self.crop_size
        if h < cs or w < cs:
            raise ValueError(
                f"{sample['site']}: image ({h}×{w}) is smaller than crop size {cs}"
            )
        top  = int(torch.randint(0, h - cs + 1, (1,)).item())
        left = int(torch.randint(0, w - cs + 1, (1,)).item())
        out = {
            "image": image[:, top:top+cs, left:left+cs],
            "label": label[:, top:top+cs, left:left+cs],
            "site":  sample["site"],
        }
        if "valid_mask" in sample:
            out["valid_mask"] = sample["valid_mask"][:, top:top+cs, left:left+cs]
        return out
