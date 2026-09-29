"""OSCD arrays produced with the same steps as DS-UNet preprocessing.

This module does not modify repos/2/DS_UNet. The numeric steps follow
preprocessing.py in that repository:

- combine_bands: 13 rectified Sentinel-2 bands, divide by 10000, clip to [0, 1]
- process_city: change mask minus 1
- add_sentinel1: resize VV to the change-mask width and height with cubic interpolation

The GeoTIFFs are LZW-compressed. They are read with rasterio into a 2-D
(height, width) array, which is the layout tifffile.imread uses for these
single-band files. No scaling is applied to Sentinel-1 here. No CRS
transform is applied. Nothing is written unless save_preprocessed is called.
"""

import warnings
from pathlib import Path

import cv2
import numpy as np
import rasterio
import torch
from rasterio.errors import NotGeoreferencedWarning
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OSCD_ROOT = PROJECT_ROOT / "datasets" / "oscd"
S1_ROOT = PROJECT_ROOT / "datasets" / "oscd_sentinel1"

SENTINEL1_BANDS = ["VV"]
SENTINEL2_BANDS = [
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

ORBITS = {
    "aguasclaras": [24],
    "bercy": [59, 8, 110],
    "bordeaux": [30, 8, 81],
    "nantes": [30, 81],
    "paris": [59, 8, 110],
    "rennes": [30, 81],
    "saclay_e": [59, 8],
    "abudhabi": [130],
    "cupertino": [35, 115, 42],
    "pisa": [15, 168],
    "beihai": [157],
    "hongkong": [11, 113],
    "beirut": [14, 87],
    "mumbai": [34],
    "brasilia": [24],
    "montpellier": [59, 37],
    "norcia": [117, 44, 22, 95],
    "rio": [155],
    "saclay_w": [59, 8, 110],
    "valencia": [30, 103, 8, 110],
    "dubai": [130, 166],
    "lasvegas": [166, 173],
    "milano": [66, 168],
    "chongqing": [55, 164],
}


def load_cities(split: str) -> list[str]:
    """Read images/train.txt or images/test.txt the same way as DS-UNet load_cities."""
    cities_file = OSCD_ROOT / "images" / f"{split}.txt"
    text = cities_file.read_text(encoding="utf-8")
    return text[:-1].split(",")


def _read_band(path: Path) -> np.ndarray:
    with rasterio.open(path) as dataset:
        array = dataset.read(1)
    if array.ndim != 2:
        raise ValueError(f"expected a single-band raster, got shape {array.shape} for {path}")
    return array


def combine_bands(folder: Path) -> np.ndarray:
    blue = _read_band(folder / "B02.tif")
    image = np.ndarray((*blue.shape, len(SENTINEL2_BANDS)), dtype=np.float32)

    for index, band in enumerate(SENTINEL2_BANDS):
        array = _read_band(folder / f"{band}.tif")
        array = np.clip(array / 10000, a_min=0, a_max=1)
        image[:, :, index] = array

    return image


def read_change_mask(city: str) -> np.ndarray:
    split = "test" if city in load_cities("test") else "train"
    label_file = OSCD_ROOT / f"{split}_labels" / city / "cm" / f"{city}-cm.tif"
    label = _read_band(label_file)
    return label - 1


def sentinel1_path(city: str, orbit: int, date: int) -> Path:
    return S1_ROOT / f"sentinel1_{city}_{orbit}_t{date}.tif"


def resize_sentinel1(city: str, orbit: int, date: int, height: int, width: int) -> np.ndarray:
    image = _read_band(sentinel1_path(city, orbit, date))
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
    return image[:, :, None]


def preprocess_city(city: str, orbit: int | None = None) -> dict[str, np.ndarray]:
    """Return the arrays DS-UNet would save for one city and one orbit."""
    if orbit is None:
        orbit = int(ORBITS[city][0])

    sentinel2_t1 = combine_bands(OSCD_ROOT / "images" / city / "imgs_1_rect")
    sentinel2_t2 = combine_bands(OSCD_ROOT / "images" / city / "imgs_2_rect")
    label = read_change_mask(city)
    height, width = label.shape
    sentinel1_t1 = resize_sentinel1(city, orbit, 1, height, width)
    sentinel1_t2 = resize_sentinel1(city, orbit, 2, height, width)

    return {
        "sentinel2_t1": sentinel2_t1,
        "sentinel2_t2": sentinel2_t2,
        "sentinel1_t1": sentinel1_t1.astype(np.float32, copy=False),
        "sentinel1_t2": sentinel1_t2.astype(np.float32, copy=False),
        "label": label,
        "orbit": np.int64(orbit),
    }


def save_preprocessed(city: str, output_root: Path, orbit: int | None = None) -> dict[str, np.ndarray]:
    """Write the same .npy layout as preprocessing.py. Does not touch the source TIFFs."""
    arrays = preprocess_city(city, orbit)
    orbit_value = int(arrays["orbit"])
    city_dir = Path(output_root) / city

    sentinel2_dir = city_dir / "sentinel2"
    sentinel2_dir.mkdir(parents=True, exist_ok=True)
    np.save(sentinel2_dir / f"sentinel2_{city}_t1.npy", arrays["sentinel2_t1"])
    np.save(sentinel2_dir / f"sentinel2_{city}_t2.npy", arrays["sentinel2_t2"])

    label_dir = city_dir / "label"
    label_dir.mkdir(parents=True, exist_ok=True)
    np.save(label_dir / f"urbanchange_{city}.npy", arrays["label"])

    sentinel1_dir = city_dir / "sentinel1"
    sentinel1_dir.mkdir(parents=True, exist_ok=True)
    np.save(sentinel1_dir / f"sentinel1_{city}_{orbit_value}_t1.npy", arrays["sentinel1_t1"])
    np.save(sentinel1_dir / f"sentinel1_{city}_{orbit_value}_t2.npy", arrays["sentinel1_t2"])
    return arrays


def _band_selection(available: list[str], selected: list[str]) -> list[bool]:
    selection = [False for _ in available]
    for band in selected:
        selection[available.index(band)] = True
    return selection


class OSCDDataset(Dataset):
    """Full-city samples after DS-UNet preprocessing, before random cropping.

    mode is 'optical', 'sar', or 'fusion'. Fusion stacks VV in front of the
    13 Sentinel-2 bands, matching OSCDDataset.__getitem__ in DS-UNet.
    """

    def __init__(
        self,
        split: str = "train",
        mode: str = "optical",
        sentinel1_bands: list[str] | None = None,
        sentinel2_bands: list[str] | None = None,
        orbit: int | None = None,
    ):
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")
        if mode not in {"optical", "sar", "fusion"}:
            raise ValueError("mode must be 'optical', 'sar', or 'fusion'")

        self.split = split
        self.mode = mode
        self.cities = load_cities(split)
        self.fixed_orbit = orbit
        self.s1_band_selection = _band_selection(
            SENTINEL1_BANDS, sentinel1_bands or list(SENTINEL1_BANDS)
        )
        self.s2_band_selection = _band_selection(
            SENTINEL2_BANDS, sentinel2_bands or list(SENTINEL2_BANDS)
        )
        self._cache: dict[tuple[str, int], dict[str, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.cities)

    def _arrays(self, city: str, orbit: int) -> dict[str, np.ndarray]:
        key = (city, orbit)
        if key not in self._cache:
            self._cache[key] = preprocess_city(city, orbit)
        return self._cache[key]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        city = self.cities[index]
        if self.fixed_orbit is None:
            orbit = int(np.random.choice(ORBITS[city]))
        else:
            orbit = self.fixed_orbit

        arrays = self._arrays(city, orbit)
        sentinel1_t1 = arrays["sentinel1_t1"][:, :, self.s1_band_selection].astype(np.float32)
        sentinel1_t2 = arrays["sentinel1_t2"][:, :, self.s1_band_selection].astype(np.float32)
        sentinel2_t1 = arrays["sentinel2_t1"][:, :, self.s2_band_selection].astype(np.float32)
        sentinel2_t2 = arrays["sentinel2_t2"][:, :, self.s2_band_selection].astype(np.float32)
        label = np.asarray(arrays["label"], dtype=np.float32)[:, :, np.newaxis]

        if self.mode == "optical":
            t1_image, t2_image = sentinel2_t1, sentinel2_t2
        elif self.mode == "sar":
            t1_image, t2_image = sentinel1_t1, sentinel1_t2
        else:
            t1_image = np.concatenate((sentinel1_t1, sentinel2_t1), axis=2)
            t2_image = np.concatenate((sentinel1_t2, sentinel2_t2), axis=2)

        return {
            "t1_img": TF.to_tensor(t1_image),
            "t2_img": TF.to_tensor(t2_image),
            "label": TF.to_tensor(label),
            "city": city,
            "orbit": orbit,
        }
