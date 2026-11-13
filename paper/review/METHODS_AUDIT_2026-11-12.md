# Targeted methods audit (2026-11-12, after Phase 4)

Scope: three preprocessing details flagged in `final_submission_v2.md`. Sources were read; nothing was inferred from
convention. The footprint check (`methods_audit_footprints.py` → `methods_audit_footprints.json`) reads raster metadata
only; no model was run and no test metric was computed.

## 1. Sentinel-2 20 m / 60 m band resampling in OSCD

**Evidence.**
- OSCD `images/README.txt` (shipped with the dataset, `datasets/oscd/images/README.txt`): "folders imgs_1_rect and
  imgs_2_rect contain the same images as imgs_1 and imgs_2 resampled at 10m resolution and cropped accordingly for ease
  of use"; "For downloading and cropping the images, the Medusa toolbox was used"; "For precise registration of the
  images, the GeFolki toolbox was used".
- Caye Daudt et al., IGARSS 2018 [3] (arXiv 1810.08468), §2: bands cropped to the chosen coordinates with the Medusa
  toolbox; §4: "the channels with resolutions lower than 10m were upsampled to the resolution of 10m so that all
  channels could be concatenated with aligned pixels".
- Medusa toolbox (github.com/aboulch/medusa_tb, `medusa_tb/sentinel_crop.py`): crops with `gdalwarp -t_srs EPSG:4326
  -te <bbox>` and no resampling option. Nothing in the toolbox or the README says that this script produced the `_rect`
  bands.
- Metadata (`methods_audit_footprints.json`): raw `imgs_1` bands are in WGS 84 geographic coordinates and the 10 m band
  covers exactly the city bounding box; `imgs_1_rect` bands and labels carry **no** georeferencing and have the pixel
  shape of the raw date-1 10 m band in all 10 test cities.

**Can be stated.** OSCD supplies the 20 m and 60 m bands upsampled to the 10 m grid of the date-1 crop; the labels share
that grid.

**Unknown.** The interpolation method used to upsample (not documented in the README, the dataset paper or the toolbox).

## 2. Sentinel-1 / Sentinel-2 co-registration

**Evidence.**
- OSCD README and [3] §2: the two Sentinel-2 images of each pair were registered with GeFolki, the newer image
  transformed onto the older one. This is optical–optical registration only.
- Hafner et al. [1], §II-A (local full text, `sources/Researchpapers.pdf`): Sentinel-1 GRD processed in Earth Engine
  (thermal noise removal, radiometric calibration, terrain correction, dB), temporally averaged, then "resampled …
  using cubic convolution interpolation to the resolution of the corresponding urban change labels". No co-registration
  to Sentinel-2 is described.
- DS-UNet code (`repos/2/DS_UNet/sentinel1_download.ipynb`, `preprocessing.py`): export at 10 m scale in EPSG:4326 over
  the OSCD city `.geojson` bounding box; `cv2.resize(..., INTER_CUBIC)` to the label shape. No registration step.
- Our code (`src/datasets/oscd_dataset.py`): the same cubic resize; no reprojection or registration.
- Footprint check: in the 10 test cities, every released Sentinel-1 raster (46 files) has its edges within 0.94
  Sentinel-2 pixels of the city bounding box covered by the raw date-1 Sentinel-2 bands.
- Our historical Sentinel-2 scenes (Step 8) were checked against OSCD's own date-1 bands (median shift 0.27 px); that
  check is optical–optical and does not cover Sentinel-1.

**Can be stated.** Sentinel-1 is placed on the Sentinel-2 label grid by matching geographic footprints (same bounding box,
edges within 0.94 px) and cubic resizing; no image-based SAR–optical co-registration was performed or evaluated by
OSCD, DS-UNet or us.

**Unknown.** The residual SAR–optical misregistration in image content.

## 3. Sentinel-1 three-month windows

**Evidence.**
- Hafner et al. [1], §II-A: "The first subset includes images acquired between three months prior to the acquisition
  date of the first Sentinel-2 image and its acquisition date, and the second subsets includes images acquired between
  the acquisition date of the second Sentinel-2 image and the following three months. For Mumbai, the first time series
  was extended due to the lack of data." It also states backscatter below −25 dB was masked and collections were split by
  orbit number.
- DS-UNet notebook: `TS_LENGTH = 3`; `date_range_t1 = DateRange(t1 − 3 months, t1)`, `date_range_t2 = DateRange(t2,
  t2 + 3 months)`; Mumbai: `DateRange(t1 − 1 year, t1 + 5 months)`; `unitScale(-25, 5)`, `clamp(0, 1)`, `unmask()`.
- Reproduction: our exporter (`download_gee.py:110-111`) uses the same windows, and the Step 7 report records the
  re-exported VV as bit-identical to the released rasters (`PHASE7_REPORT.md`; 102 files per the manuscript).

**Can be stated.** Date-1 mean over the 3 months before the date-1 Sentinel-2 acquisition, date-2 mean over the 3 months
after the date-2 acquisition (Mumbai, a training city, extended), per orbit, after masking values below −25 dB.

**Unknown.** Nothing material for the test cities.

## Manuscript changes

§3.1 OSCD and Sentinel-1 paragraphs rewritten with these sources; the two `[author: confirm]` markers replaced by
explicit statements of what is undocumented; §7 "Inputs" limitation updated; Appendix C item 19 extended.
