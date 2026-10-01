# Asynchronous historical-optical data feasibility

This note records what the local files can support for the question of whether a historical Sentinel-2 image can be paired with Sentinel-1, and whether temporal-gap groups can be formed. It does not assign Sentinel-1 acquisition dates.

# 1. Dataset Evidence

OSCD optical dates are stored for all 24 cities in `datasets/oscd/images/{city}/dates.txt`. Each file has `date_1` and `date_2` as `YYYYMMDD`. No other `dates.txt` files are present under `datasets/oscd/`. The official city lists are `datasets/oscd/images/train.txt` (14 cities) and `datasets/oscd/images/test.txt` (10 cities).

Sentinel-1 files are the 102 GeoTIFFs in `datasets/oscd_sentinel1/`. Names follow `sentinel1_{city}_{orbit}_t1.tif` and `sentinel1_{city}_{orbit}_t2.tif`. Every orbit listed below has both a t1 and a t2 file. Raster metadata was read for all 102 files. The only tag is `AREA_OR_POINT=Area`, and the band description is `VV`. No acquisition date, start time, or end time is stored.

`repos/2/DS_UNet/sentinel1_download.ipynb` is the local code that defines these products. It reads `dates.txt`, then exports an Earth Engine mean of `COPERNICUS/S1_GRD` IW VV for one relative orbit. `TS_LENGTH` is 3. For a city other than Mumbai, t1 is the mean from `date_1` minus 3 months to `date_1`, and t2 is the mean from `date_2` to `date_2` plus 3 months. Mumbai's t1 window in that notebook is `date_1` minus 1 year through `date_1` plus 5 months. The t2 window is still `date_2` to `date_2` plus 3 months. The saved notebook output prints only `mumbai`, so this copy does not contain an execution log for the other 23 cities. Their filenames match the notebook's `fileNamePrefix`.

The notebook mean is not a single acquisition date. The GeoTIFF does not retain the scene list inside the mean.

# 2. Per-City Acquisition Dates

`S2 date_2 − date_1` is the interval between the two Sentinel-2 dates in `dates.txt`. It is not a Sentinel-1 gap. Every `date_1` is earlier than `date_2`: True. Orbit numbers are the relative-orbit tokens in the Sentinel-1 filenames, checked against the orbit lists in the download notebook. Filename orbits match that notebook for every city: yes. Both t1 and t2 exist for each of those orbits: yes.

| City | S2 date_1 | S2 date_2 | S2 date_2 − date_1 (days) | S1 orbits with both t1 and t2 | S1 t1 date | S1 t2 date |
| --- | --- | --- | ---: | --- | --- | --- |
| aguasclaras | 2015-09-16 | 2017-10-15 | 760 | 24 | UNKNOWN | UNKNOWN |
| bercy | 2016-11-30 | 2017-08-29 | 272 | 59, 8, 110 | UNKNOWN | UNKNOWN |
| bordeaux | 2016-05-04 | 2017-10-26 | 540 | 30, 8, 81 | UNKNOWN | UNKNOWN |
| nantes | 2015-08-21 | 2017-10-14 | 785 | 30, 81 | UNKNOWN | UNKNOWN |
| paris | 2016-11-30 | 2017-11-07 | 342 | 59, 8, 110 | UNKNOWN | UNKNOWN |
| rennes | 2015-08-21 | 2017-06-21 | 670 | 30, 81 | UNKNOWN | UNKNOWN |
| saclay_e | 2016-03-15 | 2017-08-29 | 532 | 59, 8 | UNKNOWN | UNKNOWN |
| abudhabi | 2016-01-20 | 2018-03-28 | 798 | 130 | UNKNOWN | UNKNOWN |
| cupertino | 2015-09-18 | 2018-03-26 | 920 | 35, 115, 42 | UNKNOWN | UNKNOWN |
| pisa | 2015-07-04 | 2018-02-11 | 953 | 15, 168 | UNKNOWN | UNKNOWN |
| beihai | 2016-12-09 | 2018-03-09 | 455 | 157 | UNKNOWN | UNKNOWN |
| hongkong | 2016-09-27 | 2018-03-23 | 542 | 11, 113 | UNKNOWN | UNKNOWN |
| beirut | 2015-08-20 | 2017-10-03 | 775 | 14, 87 | UNKNOWN | UNKNOWN |
| mumbai | 2015-11-30 | 2018-03-19 | 840 | 34 | UNKNOWN | UNKNOWN |
| brasilia | 2015-09-16 | 2017-10-17 | 762 | 24 | UNKNOWN | UNKNOWN |
| montpellier | 2015-08-12 | 2017-10-30 | 810 | 59, 37 | UNKNOWN | UNKNOWN |
| norcia | 2015-07-11 | 2017-10-18 | 830 | 117, 44, 22, 95 | UNKNOWN | UNKNOWN |
| rio | 2016-04-24 | 2017-10-11 | 535 | 155 | UNKNOWN | UNKNOWN |
| saclay_w | 2016-03-15 | 2017-08-29 | 532 | 59, 8, 110 | UNKNOWN | UNKNOWN |
| valencia | 2016-07-30 | 2017-11-07 | 465 | 30, 103, 8, 110 | UNKNOWN | UNKNOWN |
| dubai | 2015-12-11 | 2018-03-30 | 840 | 130, 166 | UNKNOWN | UNKNOWN |
| lasvegas | 2015-08-20 | 2018-02-05 | 900 | 166, 173 | UNKNOWN | UNKNOWN |
| milano | 2016-12-28 | 2018-01-22 | 390 | 66, 168 | UNKNOWN | UNKNOWN |
| chongqing | 2017-04-14 | 2018-04-02 | 353 | 55, 164 | UNKNOWN | UNKNOWN |

# 3. Sentinel-1 Metadata Availability

| Item | Local evidence | Result |
| --- | --- | --- |
| t1 and t2 files | `datasets/oscd_sentinel1/` | Present for every city and listed orbit |
| Relative orbit | Filename token and notebook `ORBITS` | Present |
| Polarization | Band description `VV` | VV only |
| Acquisition date | Raster tags and notebook outputs | UNKNOWN |
| Scene identifiers inside the mean | Not stored in the GeoTIFF or notebook output | UNKNOWN |
| Mumbai t1 window | Notebook special case | Different from the 3-month rule |

`s1_t1_date` and `s1_t2_date` are `UNKNOWN` for every city.

# 4. Valid Historical-Optical Configurations

The requested stack is Sentinel-1 t1, Sentinel-1 t2, and one historical Sentinel-2 date T0 with T0 earlier than the target Sentinel-1 t2 observation.

The image files for that stack exist: t1, t2, and the Sentinel-2 image at `date_1`. `date_1` is the only earlier optical date supplied by OSCD. `date_2` is the later optical date.

The statement `date_1` is earlier than a Sentinel-1 t2 acquisition is not established by a SAR timestamp. The notebook instead defines t2 as a mean whose window begins at `date_2`. Because `date_1` is earlier than `date_2` for every city, `date_1` is earlier than that coded window start. That is a property of the download code, not a date read from the Sentinel-1 file. Individual scenes inside the three-month t2 window remain undated here.

Mumbai's t1 mean is defined to extend after `date_1`. Its coded end is still before Mumbai `date_2` (2016-04-30 versus 2018-03-19). That still does not provide a scene date.

# 5. Actual Temporal Gaps

No Sentinel-1 acquisition date is known, so `temporal_gap_days` is `UNKNOWN` for every city.

The Sentinel-2 intervals `date_2 − date_1` are known and are not all equal. The distinct values, in days, are: 272, 342, 353, 390, 455, 465, 532, 535, 540, 542, 670, 760, 762, 775, 785, 798, 810, 830, 840, 900, 920, 953.

Those numbers measure the time between the two optical dates. They also equal the coded offset from `date_1` to the start of the t2 mean. They do not measure the time from `date_1` to a Sentinel-1 scene. Using them as SAR gaps would treat `date_2` as the SAR time. This study does not do that.

# 6. Cities Usable for the Experiment

No city is usable for a temporal-gap experiment that requires a known Sentinel-1 acquisition date.

All 24 cities do have the files for one earlier optical date (`date_1`), one later optical date (`date_2`), and paired Sentinel-1 t1/t2 orbit means. That supports a single historical-versus-later optical comparison only if the target SAR time is explicitly defined as the composite product, with the limitation in sections 3 and 5.

# 7. Cities That Cannot Be Used

All 24 cities cannot be used for a gap analysis keyed to Sentinel-1 acquisition dates: aguasclaras, bercy, bordeaux, nantes, paris, rennes, saclay_e, abudhabi, cupertino, pisa, beihai, hongkong, beirut, mumbai, brasilia, montpellier, norcia, rio, saclay_w, valencia, dubai, lasvegas, milano, chongqing.

The reason is the same for each city. The SAR date is `UNKNOWN`.

# 8. Temporal-Gap Experiment Feasibility

Multiple gap groups tied to Sentinel-1 acquisition dates are not available. There is no retained scene time series, and each city has only two Sentinel-2 dates. Reusing `date_1` under several invented SAR dates is not supported.

Binning cities by `date_2 − date_1` would group optical-date intervals. It would not group measured SAR gaps.

Conclusion for the combined question: **B. Partially supported**.

The files can identify one earlier optical date and the existing SAR composites. They cannot support the temporal-gap half of the question without Sentinel-1 scene dates.

# 9. Scientific Risks

- Treating t1 or t2 as if it were acquired on `date_1` or `date_2` is not supported by the GeoTIFF.
- The three-month means mix many acquisitions. A single gap in days is not defined for a mean.
- Mumbai's t1 window is not the three-month window used for the other cities.
- The saved notebook execution prints only Mumbai, so the other cities are tied to this procedure by filename and shared code, not by a per-city export log.
- Orbit number is not an acquisition date. Cities with several orbits still have unknown scene times.
- The synchronous fusion baseline uses both optical dates. A historical run would have to withhold `date_2` on purpose. That design choice is separate from knowing a SAR timestamp.

# 10. Recommended Experimental Design

Do not implement an asynchronous temporal-gap model on these files until Sentinel-1 scene dates, or the scene list inside each mean, are obtained from a source that records them.

A later single-configuration experiment can still be specified without inventing dates: Sentinel-1 t1, Sentinel-1 t2, and Sentinel-2 `date_1` only, with `date_2` withheld. The write-up would have to say that t2 is the existing orbit-mean product and that its acquisition date is unknown. It should not report a day gap.

Do not create gap bins from `date_2 − date_1` and describe them as SAR gaps. If those optical intervals are reported, label them as the difference between the two Sentinel-2 dates.
