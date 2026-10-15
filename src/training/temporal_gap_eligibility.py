"""Eligibility inventory for historical optical dates on the locked Scenario 2 sites.

This script does not train, score, or edit the locked split. The trained
Scenario 2 interval remains March 2018 optical with December 2019 target SAR
(21 months). That interval is recorded as the existing experiment, not as a
temporal-gap sweep.

Optical, SAR, and building-label availability are checked separately. A SAR
raster at a month is not treated as evidence that a historical optical raster
exists.

Run from the project root:

    .venv\\Scripts\\python.exe src\\training\\temporal_gap_eligibility.py
    .venv\\Scripts\\python.exe src\\training\\temporal_gap_eligibility.py --verify
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.scenario2_dataset import (  # noqa: E402
    DATASET_ROOT,
    SPLIT_FILE,
    T1_MONTH,
    T1_YEAR,
    T2_MONTH,
    T2_YEAR,
    _bld_path,
    _s1_path,
    _s2_path,
    load_split,
)

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "scenario_2_temporal_gap"
CSV_PATH = OUTPUT_DIR / "eligibility_by_site_month.csv"
JSON_PATH = OUTPUT_DIR / "eligibility_report.json"
PAIR_CSV_PATH = OUTPUT_DIR / "pair_eligibility_by_site_month.csv"
PAIR_JSON_PATH = OUTPUT_DIR / "pair_eligibility_report.json"
DISAGREEMENT_CSV_PATH = OUTPUT_DIR / "optical_disagreement_files.csv"

# Published in docs/paper/02_results_v2.md. Used only as a read-only check.
EXPECTED_SPLIT_SHA256 = "5d0841adf0c651166cb18ecea310c6ef8cce69172065fc2d699a29083c272e27"

TARGET_SAR = (T2_YEAR, T2_MONTH)  # December 2019
EXISTING_OPTICAL = (T1_YEAR, T1_MONTH)  # March 2018
LABEL_ENDPOINTS = (EXISTING_OPTICAL, TARGET_SAR)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def month_index(year: int, month: int) -> int:
    return year * 12 + month


def gap_months(earlier: tuple[int, int], later: tuple[int, int]) -> int:
    return month_index(*later) - month_index(*earlier)


def date_text(year: int, month: int) -> str:
    return f"{year:04d}-{month:02d}"


def load_metadata() -> tuple[dict, dict]:
    meta_path = DATASET_ROOT / "metadata.json"
    bad_path = DATASET_ROOT / "bad_data.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    bad = json.loads(bad_path.read_text(encoding="utf-8"))
    return metadata, bad


def index_site_records(records: list[dict]) -> tuple[dict[tuple[int, int], dict], list[str]]:
    """Map (year, month) to one metadata record. Duplicates are flagged, not merged."""
    notes: list[str] = []
    indexed: dict[tuple[int, int], dict] = {}
    for index, record in enumerate(records):
        key = (int(record["year"]), int(record["month"]))
        payload = {
            "index": index,
            "s1": bool(record.get("s1")),
            "s2": bool(record.get("s2")),
            "buildings": bool(record.get("buildings")),
            "masked": bool(record.get("masked")),
        }
        if key in indexed:
            notes.append(f"duplicate metadata record for {date_text(*key)} at index {index}")
            continue
        indexed[key] = payload
    return indexed, notes


def file_status(path: Path) -> tuple[bool, int | None, str]:
    if not path.exists():
        return False, None, ""
    if not path.is_file():
        return False, None, "path exists but is not a file"
    size = path.stat().st_size
    if size <= 0:
        return False, size, "file exists but is empty"
    return True, size, ""


def unexpected_tifs(site_id: str, folder: str, prefix: str) -> list[str]:
    directory = DATASET_ROOT / site_id / folder
    if not directory.is_dir():
        return [f"missing directory {folder}"]
    unexpected = []
    for path in sorted(directory.glob("*.tif")):
        expected_prefix = f"{prefix}_{site_id}_"
        name = path.name
        if not name.startswith(expected_prefix) or not name.endswith(".tif"):
            unexpected.append(name)
            continue
        tail = name[len(expected_prefix):-4]
        year_text, sep, month_text = tail.partition("_")
        if sep != "_" or len(year_text) != 4 or len(month_text) != 2:
            unexpected.append(name)
            continue
        if not year_text.isdigit() or not month_text.isdigit():
            unexpected.append(name)
    return unexpected


def modality_flags(
    site_id: str,
    year: int,
    month: int,
    records: dict[tuple[int, int], dict],
    bad_indices: set[int],
    path_for,
) -> dict:
    path = path_for(site_id, year, month)
    exists, size, note = file_status(path)
    record = records.get((year, month))
    metadata_present = record is not None
    bad_index = bool(record and record["index"] in bad_indices)
    masked = bool(record and record["masked"])
    return {
        "file_exists": exists,
        "file_bytes": size,
        "metadata_record": metadata_present,
        "metadata_flag": None if record is None else record,
        "bad_index": bad_index,
        "masked": masked,
        "note": note,
    }


def optical_metadata_usable(flags: dict) -> bool:
    record = flags["metadata_flag"]
    return bool(
        flags["metadata_record"]
        and record is not None
        and record["s2"]
        and not flags["bad_index"]
        and not flags["masked"]
    )


def sar_metadata_usable(flags: dict) -> bool:
    record = flags["metadata_flag"]
    return bool(
        flags["metadata_record"]
        and record is not None
        and record["s1"]
        and not flags["bad_index"]
        and not flags["masked"]
    )


def label_metadata_usable(flags: dict) -> bool:
    record = flags["metadata_flag"]
    return bool(
        flags["metadata_record"]
        and record is not None
        and record["buildings"]
        and not flags["masked"]
    )


def locked_sites(split: dict[str, list[str]]) -> list[tuple[str, str]]:
    ordered = []
    for name in ("train", "validation", "test"):
        for site_id in split[name]:
            ordered.append((site_id, name))
    return ordered


def candidate_months(metadata: dict, site_ids: list[str]) -> tuple[list[tuple[int, int]], list[str]]:
    """Historical months strictly before the December 2019 target SAR date."""
    notes = []
    found: set[tuple[int, int]] = set()
    target_index = month_index(*TARGET_SAR)
    for site_id in site_ids:
        records, dup_notes = index_site_records(metadata.get(site_id, []))
        notes.extend(f"{site_id}: {note}" for note in dup_notes)
        for key in records:
            if month_index(*key) < target_index:
                found.add(key)
        s2_dir = DATASET_ROOT / site_id / "s2"
        if not s2_dir.is_dir():
            notes.append(f"{site_id}: s2 directory missing")
            continue
        for path in s2_dir.glob("s2_*.tif"):
            parsed = parse_suffix_date(site_id, path.name, "s2")
            if parsed is None:
                notes.append(f"{site_id}: unparsed optical filename {path.name}")
                continue
            if month_index(*parsed) < target_index and parsed not in records:
                found.add(parsed)
                notes.append(
                    f"{site_id}: optical file {date_text(*parsed)} has no metadata record"
                )
    return sorted(found), notes


def parse_suffix_date(site_id: str, filename: str, prefix: str) -> tuple[int, int] | None:
    expected = f"{prefix}_{site_id}_"
    if not filename.startswith(expected) or not filename.endswith(".tif"):
        return None
    tail = filename[len(expected):-4]
    year_text, sep, month_text = tail.partition("_")
    if sep != "_" or len(year_text) != 4 or len(month_text) != 2:
        return None
    if not year_text.isdigit() or not month_text.isdigit():
        return None
    month = int(month_text)
    if month < 1 or month > 12:
        return None
    return int(year_text), month


def build_rows(split: dict[str, list[str]], metadata: dict, bad: dict) -> tuple[list[dict], dict]:
    sites = locked_sites(split)
    site_ids = [site_id for site_id, _ in sites]
    months, month_notes = candidate_months(metadata, site_ids)
    ambiguities = list(month_notes)
    rows: list[dict] = []
    fixed_checks = []

    for site_id, split_name in sites:
        records, dup_notes = index_site_records(metadata.get(site_id, []))
        ambiguities.extend(f"{site_id}: {note}" for note in dup_notes)
        if site_id not in metadata:
            ambiguities.append(f"{site_id}: absent from metadata.json")
        site_bad = bad.get(site_id, {})
        extra_bad_keys = sorted(set(site_bad) - {"S1", "S2"})
        if extra_bad_keys:
            ambiguities.append(f"{site_id}: uninterpreted bad_data keys {extra_bad_keys}")
        bad_s1 = set(site_bad.get("S1", []))
        bad_s2 = set(site_bad.get("S2", []))
        n_records = len(metadata.get(site_id, []))
        for label, indices in (("S1", bad_s1), ("S2", bad_s2)):
            out_of_range = sorted(index for index in indices if index < 0 or index >= n_records)
            if out_of_range:
                ambiguities.append(
                    f"{site_id}: {label} bad index out of range {out_of_range}"
                )
        for folder, prefix in (("s1", "s1"), ("s2", "s2"), ("buildings", "buildings")):
            for name in unexpected_tifs(site_id, folder, prefix):
                ambiguities.append(f"{site_id}: unexpected {folder} entry {name}")

        endpoint = {}
        for label, year, month, path_for, usable in (
            ("sar_2018_03", *EXISTING_OPTICAL, _s1_path, sar_metadata_usable),
            ("sar_2019_12", *TARGET_SAR, _s1_path, sar_metadata_usable),
            ("optical_2018_03", *EXISTING_OPTICAL, _s2_path, optical_metadata_usable),
            ("label_2018_03", *EXISTING_OPTICAL, _bld_path, label_metadata_usable),
            ("label_2019_12", *TARGET_SAR, _bld_path, label_metadata_usable),
        ):
            if label.startswith("sar"):
                bad_indices = bad_s1
            elif label.startswith("optical"):
                bad_indices = bad_s2
            else:
                bad_indices = set()
            flags = modality_flags(site_id, year, month, records, bad_indices, path_for)
            endpoint[f"{label}_file"] = flags["file_exists"]
            endpoint[f"{label}_metadata_usable"] = usable(flags)
        fixed_checks.append({"site_id": site_id, "split": split_name, **endpoint})

        for year, month in months:
            optical = modality_flags(site_id, year, month, records, bad_s2, _s2_path)
            sar = modality_flags(site_id, year, month, records, bad_s1, _s1_path)
            labels = modality_flags(site_id, year, month, records, set(), _bld_path)
            target_sar = modality_flags(site_id, *TARGET_SAR, records, bad_s1, _s1_path)
            optical_usable = optical_metadata_usable(optical)
            sar_usable = sar_metadata_usable(sar)
            label_usable = label_metadata_usable(labels)
            file_vs_metadata = optical["file_exists"] != optical_usable
            sar_without_optical = sar["file_exists"] and not optical["file_exists"]
            role = (
                "existing_trained_interval"
                if (year, month) == EXISTING_OPTICAL
                else "candidate_untrained"
            )
            notes = []
            if optical["note"]:
                notes.append(f"optical: {optical['note']}")
            if file_vs_metadata:
                notes.append("optical file presence and metadata usability disagree")
            if sar_without_optical:
                notes.append("SAR file exists at this month and the optical file does not")
            record = optical["metadata_flag"]
            rows.append({
                "site_id": site_id,
                "split": split_name,
                "historical_optical_date": date_text(year, month),
                "gap_months_to_target_sar_2019_12": gap_months((year, month), TARGET_SAR),
                "role": role,
                "optical_file_exists": optical["file_exists"],
                "optical_file_bytes": "" if optical["file_bytes"] is None else optical["file_bytes"],
                "optical_metadata_record": optical["metadata_record"],
                "optical_metadata_s2_flag": "" if record is None else record["s2"],
                "optical_bad_index": optical["bad_index"],
                "optical_masked": optical["masked"],
                "optical_metadata_usable": optical_usable,
                "sar_file_exists_at_historical_month": sar["file_exists"],
                "sar_metadata_usable_at_historical_month": sar_usable,
                "label_file_exists_at_historical_month": labels["file_exists"],
                "label_metadata_usable_at_historical_month": label_usable,
                "sar_file_exists_at_target_2019_12": target_sar["file_exists"],
                "sar_metadata_usable_at_target_2019_12": sar_metadata_usable(target_sar),
                "label_file_exists_at_2018_03": endpoint["label_2018_03_file"],
                "label_file_exists_at_2019_12": endpoint["label_2019_12_file"],
                "optical_file_and_metadata_usable": optical["file_exists"] and optical_usable,
                "disagreement_optical_file_vs_metadata": file_vs_metadata,
                "sar_file_without_optical_file": sar_without_optical,
                "notes": "; ".join(notes),
            })

    return rows, {
        "candidate_months": [date_text(*item) for item in months],
        "ambiguities": ambiguities,
        "fixed_endpoint_checks": fixed_checks,
    }


def summarize(rows: list[dict], extra: dict, split_hash: str) -> dict:
    def count(predicate) -> int:
        return sum(1 for row in rows if predicate(row))

    by_month = []
    for date in extra["candidate_months"]:
        subset = [row for row in rows if row["historical_optical_date"] == date]
        by_month.append({
            "historical_optical_date": date,
            "gap_months_to_target_sar_2019_12": subset[0]["gap_months_to_target_sar_2019_12"],
            "role": subset[0]["role"],
            "sites": len(subset),
            "optical_file_exists": count(lambda row, date=date: row["historical_optical_date"] == date and row["optical_file_exists"]),
            "optical_metadata_usable": count(lambda row, date=date: row["historical_optical_date"] == date and row["optical_metadata_usable"]),
            "optical_file_and_metadata_usable": count(lambda row, date=date: row["historical_optical_date"] == date and row["optical_file_and_metadata_usable"]),
            "sar_file_at_historical_month": count(lambda row, date=date: row["historical_optical_date"] == date and row["sar_file_exists_at_historical_month"]),
            "label_file_at_historical_month": count(lambda row, date=date: row["historical_optical_date"] == date and row["label_file_exists_at_historical_month"]),
            "sar_file_without_optical_file": count(lambda row, date=date: row["historical_optical_date"] == date and row["sar_file_without_optical_file"]),
        })

    disagreements = [
        row for row in rows
        if row["optical_file_exists"] and not row["optical_metadata_usable"]
    ]
    disagreement_reasons = {
        "metadata_s2_flag_false": sum(1 for row in disagreements if row["optical_metadata_s2_flag"] is False),
        "bad_data_s2_index": sum(1 for row in disagreements if row["optical_bad_index"]),
        "masked": sum(1 for row in disagreements if row["optical_masked"]),
        "metadata_record_missing": sum(1 for row in disagreements if not row["optical_metadata_record"]),
    }
    disagreement_rows = [
        {
            "site_id": row["site_id"],
            "split": row["split"],
            "historical_optical_date": row["historical_optical_date"],
            "metadata_s2_flag": row["optical_metadata_s2_flag"],
            "bad_index": row["optical_bad_index"],
            "masked": row["optical_masked"],
        }
        for row in disagreements
    ]

    return {
        "purpose": (
            "File and metadata eligibility inventory for historical optical months "
            "on the locked Scenario 2 sites. This is not a performance result and "
            "not a temporal-gap sensitivity study."
        ),
        "existing_experiment": {
            "name": "Scenario 2 trained interval",
            "historical_optical_date": date_text(*EXISTING_OPTICAL),
            "target_sar_date": date_text(*TARGET_SAR),
            "also_uses_sar_date": date_text(*EXISTING_OPTICAL),
            "label_dates": [date_text(*item) for item in LABEL_ENDPOINTS],
            "label_rule": "(buildings_T2 >= 0.5) XOR (buildings_T1 >= 0.5)",
            "gap_months": gap_months(EXISTING_OPTICAL, TARGET_SAR),
            "statement": (
                "March 2018 to December 2019 is the existing approximately "
                "21-month experiment. It is not a temporal-gap sweep."
            ),
        },
        "eligibility_rules": [
            "Sites are the locked train, validation, and test lists. The split file is read only.",
            "A candidate historical month is strictly earlier than December 2019.",
            "Optical file availability is existence of s2_{site}_{year}_{month}.tif with nonzero size.",
            "Optical metadata usability follows metadata.json s2=true, bad_data.json S2 index absent, and masked=false.",
            "Bad-data indices follow metadata record order, the same convention as temporal_pairing_analysis.py.",
            "SAR file availability and building-label file availability are separate columns.",
            "A SAR file does not make a month optically eligible.",
            "File presence and metadata usability are both reported. Disagreements are ambiguities, not resolved by this script.",
            "No model is trained and no detection metric is computed.",
        ],
        "split": {
            "path": str(SPLIT_FILE),
            "sha256": split_hash,
            "expected_sha256": EXPECTED_SPLIT_SHA256,
            "hash_matches_published_value": split_hash == EXPECTED_SPLIT_SHA256,
            "counts": {
                "train": 20,
                "validation": 4,
                "test": 4,
            },
        },
        "candidate_historical_months": extra["candidate_months"],
        "counts": {
            "sites": len({row["site_id"] for row in rows}),
            "candidate_months": len(extra["candidate_months"]),
            "rows": len(rows),
            "optical_file_exists": count(lambda row: row["optical_file_exists"]),
            "optical_metadata_usable": count(lambda row: row["optical_metadata_usable"]),
            "optical_file_and_metadata_usable": count(lambda row: row["optical_file_and_metadata_usable"]),
            "file_present_metadata_not_usable": count(lambda row: row["optical_file_exists"] and not row["optical_metadata_usable"]),
            "metadata_usable_file_missing": count(lambda row: row["optical_metadata_usable"] and not row["optical_file_exists"]),
            "sar_file_without_optical_file": count(lambda row: row["sar_file_without_optical_file"]),
        },
        "by_month": by_month,
        "fixed_endpoint_checks": extra["fixed_endpoint_checks"],
        "ambiguities": extra["ambiguities"] + [
            "Optical GeoTIFF presence and metadata usability disagree for "
            f"{len(disagreement_rows)} site-months. The reasons can overlap. "
            "This inventory does not decide which of those months are valid training inputs."
        ],
        "optical_file_metadata_disagreement": {
            "count": len(disagreement_rows),
            "reason_counts": disagreement_reasons,
            "site_months": disagreement_rows,
        },
        "outputs": {
            "csv": str(CSV_PATH),
            "json": str(JSON_PATH),
        },
    }


def write_outputs(rows: list[dict], report: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    JSON_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")


def verify(split: dict[str, list[str]]) -> list[str]:
    """Check the written report against the locked split and the eligibility rules."""
    failures = []
    current_hash = sha256_file(SPLIT_FILE)
    if current_hash != EXPECTED_SPLIT_SHA256:
        failures.append(f"split hash changed: {current_hash}")
    if [len(split[name]) for name in ("train", "validation", "test")] != [20, 4, 4]:
        failures.append("split counts are not 20/4/4")
    if not CSV_PATH.is_file() or not JSON_PATH.is_file():
        failures.append("csv or json output is missing")
        return failures
    report = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    if report["split"]["sha256"] != current_hash:
        failures.append("report hash does not match the split file")
    if report["existing_experiment"]["gap_months"] != 21:
        failures.append("existing experiment gap is not 21 months")
    if "sweep" in report["existing_experiment"]["statement"].lower() and "not a temporal-gap sweep" not in report["existing_experiment"]["statement"]:
        failures.append("existing interval was described as a sweep")
    with CSV_PATH.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != report["counts"]["rows"]:
        failures.append("csv row count does not match the json report")
    if len({row["site_id"] for row in rows}) != 28:
        failures.append("csv does not cover 28 sites")
    existing = [row for row in rows if row["role"] == "existing_trained_interval"]
    if len(existing) != 28:
        failures.append("existing March 2018 interval is not recorded once per site")
    if any(row["historical_optical_date"] != "2018-03" for row in existing):
        failures.append("existing interval rows are not March 2018")
    if any(row["gap_months_to_target_sar_2019_12"] != "21" for row in existing):
        failures.append("existing interval gap is not 21")
    forbidden = {"f1", "precision", "recall", "iou", "false_alarm_rate", "test_f1"}
    header = set(rows[0])
    if header & forbidden:
        failures.append(f"performance columns were written: {sorted(header & forbidden)}")
    copied = [
        row for row in rows
        if row["sar_file_without_optical_file"] == "True" and row["optical_file_exists"] == "True"
    ]
    if copied:
        failures.append("SAR-without-optical rows were marked as having optical files")
    return failures


def component_state(file_exists: bool, metadata_usable: bool) -> str:
    if file_exists and metadata_usable:
        return "confirmed"
    if file_exists != metadata_usable:
        return "ambiguous"
    return "ineligible"


def pair_status(states: dict[str, str]) -> str:
    values = list(states.values())
    if all(value == "confirmed" for value in values):
        return "confirmed_eligible"
    if any(value == "ineligible" for value in values):
        return "ineligible"
    return "ambiguous"


def flag_cooccurrence(metadata: dict, bad: dict, site_ids: list[str]) -> dict:
    """Relate s2, bad-S2 index, and masked on the locked sites. No quality is inferred."""
    counts = {
        "records": 0,
        "s2_false": 0,
        "s2_false_and_bad_index": 0,
        "s2_true_and_bad_index": 0,
        "bad_index": 0,
        "masked": 0,
        "masked_and_s2_true": 0,
        "masked_and_s2_false": 0,
        "masked_and_bad_index": 0,
    }
    for site_id in site_ids:
        records = metadata.get(site_id, [])
        bad_s2 = set(bad.get(site_id, {}).get("S2", []))
        counts["records"] += len(records)
        for index, record in enumerate(records):
            s2_flag = bool(record.get("s2"))
            is_bad = index in bad_s2
            is_masked = bool(record.get("masked"))
            if not s2_flag:
                counts["s2_false"] += 1
                if is_bad:
                    counts["s2_false_and_bad_index"] += 1
            elif is_bad:
                counts["s2_true_and_bad_index"] += 1
            if is_bad:
                counts["bad_index"] += 1
            if is_masked:
                counts["masked"] += 1
                counts["masked_and_s2_true" if s2_flag else "masked_and_s2_false"] += 1
                if is_bad:
                    counts["masked_and_bad_index"] += 1
    return counts


def raster_summary(path: Path) -> dict:
    import rasterio

    with rasterio.open(path) as dataset:
        array = dataset.read().astype("float32")
        summary = {
            "bands": int(dataset.count),
            "height": int(dataset.height),
            "width": int(dataset.width),
            "dtype": dataset.dtypes[0],
        }
    finite = np.isfinite(array)
    summary["nan_fraction"] = float((~finite).mean())
    summary["all_band_zero_fraction"] = float(np.all(array == 0, axis=0).mean())
    if finite.any():
        values = array[finite]
        summary["finite_min"] = float(values.min())
        summary["finite_max"] = float(values.max())
        summary["finite_mean"] = float(values.mean())
    else:
        summary["finite_min"] = None
        summary["finite_max"] = None
        summary["finite_mean"] = None
    return summary


def classify_pairs(rows: list[dict], extra: dict, metadata: dict, bad: dict) -> tuple[list[dict], dict, list[dict]]:
    endpoints = {item["site_id"]: item for item in extra["fixed_endpoint_checks"]}
    site_ids = list(endpoints)
    pair_rows = []
    for row in rows:
        endpoint = endpoints[row["site_id"]]
        states = {
            "optical_at_candidate_month": component_state(
                row["optical_file_exists"], row["optical_metadata_usable"]
            ),
            "sar_at_candidate_month": component_state(
                row["sar_file_exists_at_historical_month"],
                row["sar_metadata_usable_at_historical_month"],
            ),
            "sar_at_2019_12": component_state(
                row["sar_file_exists_at_target_2019_12"],
                row["sar_metadata_usable_at_target_2019_12"],
            ),
            "buildings_at_candidate_month": component_state(
                row["label_file_exists_at_historical_month"],
                row["label_metadata_usable_at_historical_month"],
            ),
            "buildings_at_2019_12": component_state(
                endpoint["label_2019_12_file"],
                endpoint["label_2019_12_metadata_usable"],
            ),
        }
        status = pair_status(states)
        blocking = [name for name, state in states.items() if state != "confirmed"]
        pair_rows.append({
            "site_id": row["site_id"],
            "split": row["split"],
            "historical_optical_date": row["historical_optical_date"],
            "gap_months_to_target_sar_2019_12": row["gap_months_to_target_sar_2019_12"],
            "role": row["role"],
            "pair_status": status,
            **{f"state_{name}": state for name, state in states.items()},
            "blocking_components": ";".join(blocking),
        })

    by_month = []
    for date in extra["candidate_months"]:
        subset = [row for row in pair_rows if row["historical_optical_date"] == date]
        by_month.append({
            "historical_optical_date": date,
            "gap_months_to_target_sar_2019_12": subset[0]["gap_months_to_target_sar_2019_12"],
            "role": subset[0]["role"],
            "sites_in_inventory": len(subset),
            "confirmed_eligible": sum(row["pair_status"] == "confirmed_eligible" for row in subset),
            "ambiguous": sum(row["pair_status"] == "ambiguous" for row in subset),
            "ineligible": sum(row["pair_status"] == "ineligible" for row in subset),
            "confirmed_eligible_train": sum(
                row["pair_status"] == "confirmed_eligible" and row["split"] == "train" for row in subset
            ),
            "confirmed_eligible_validation": sum(
                row["pair_status"] == "confirmed_eligible" and row["split"] == "validation" for row in subset
            ),
            "confirmed_eligible_test": sum(
                row["pair_status"] == "confirmed_eligible" and row["split"] == "test" for row in subset
            ),
        })

    disagreements = []
    raster_groups = {
        "s2_false_bad_index_unmasked": [],
        "s2_true_not_bad_masked": [],
        "s2_false_bad_index_masked": [],
        "other": [],
    }
    for row in rows:
        if not (row["optical_file_exists"] and not row["optical_metadata_usable"]):
            continue
        year, month = (int(part) for part in row["historical_optical_date"].split("-"))
        path = _s2_path(row["site_id"], year, month)
        summary = raster_summary(path)
        if row["optical_metadata_s2_flag"] is False and row["optical_bad_index"] and not row["optical_masked"]:
            group = "s2_false_bad_index_unmasked"
        elif row["optical_metadata_s2_flag"] is True and not row["optical_bad_index"] and row["optical_masked"]:
            group = "s2_true_not_bad_masked"
        elif row["optical_metadata_s2_flag"] is False and row["optical_bad_index"] and row["optical_masked"]:
            group = "s2_false_bad_index_masked"
        else:
            group = "other"
        entry = {
            "site_id": row["site_id"],
            "split": row["split"],
            "historical_optical_date": row["historical_optical_date"],
            "path": str(path),
            "metadata_s2_flag": row["optical_metadata_s2_flag"],
            "bad_s2_index": row["optical_bad_index"],
            "masked": row["optical_masked"],
            "reason_group": group,
            **summary,
        }
        disagreements.append(entry)
        raster_groups[group].append(summary)

    def raster_group_summary(items: list[dict]) -> dict:
        if not items:
            return {"count": 0}
        return {
            "count": len(items),
            "bands": sorted({item["bands"] for item in items}),
            "nan_fraction_min": min(item["nan_fraction"] for item in items),
            "nan_fraction_max": max(item["nan_fraction"] for item in items),
            "all_band_zero_fraction_max": max(item["all_band_zero_fraction"] for item in items),
            "finite_min": min(item["finite_min"] for item in items),
            "finite_max": max(item["finite_max"] for item in items),
        }

    reference_summaries = []
    for site_id in site_ids[:3]:
        path = _s2_path(site_id, *EXISTING_OPTICAL)
        reference_summaries.append({"site_id": site_id, "path": str(path), **raster_summary(path)})

    report = {
        "purpose": (
            "Pair classification for candidate historical optical months on the locked sites. "
            "This is an eligibility inventory, not a temporal-gap performance result."
        ),
        "metadata_rule_evidence": {
            "metadata_json_fields": ["aoi_id", "year", "month", "s1", "s2", "buildings", "masked"],
            "bad_data_json_fields": ["S1", "S2"],
            "s2": (
                "Boolean availability flag on each metadata.json timestep. "
                "The file does not record why the flag is false."
            ),
            "bad_s2_index": (
                "Integer position in that site's metadata.json record list, stored in bad_data.json under S2. "
                "The repository inspection says these indices must be excluded at load time and attributes "
                "the larger S2 count to cloud or mask issues. bad_data.json itself has no reason text."
            ),
            "masked": (
                "Boolean timestep flag in metadata.json. Pairing and inspection code treat masked=true "
                "as excluding the timestep. The JSON does not define the flag as cloud cover, no-data, or label masking."
            ),
            "cooccurrence_on_locked_sites_all_months": flag_cooccurrence(metadata, bad, site_ids),
            "inspection_sources": [
                "datasets/sen12_mt_urbanmapping/multimodal_cd_dataset/metadata.json",
                "datasets/sen12_mt_urbanmapping/multimodal_cd_dataset/bad_data.json",
                "docs/SCENARIO_2_DATASET_INSPECTION.md sections 9 and 10",
                "src/training/temporal_pairing_analysis.py is_s2_usable",
                "src/datasets/scenario2_dataset.py preprocess_site",
            ],
        },
        "loader_behavior": {
            "function": "src/datasets/scenario2_dataset.py:preprocess_site",
            "dates_read": ["2018-03", "2019-12"],
            "reads_metadata_json": False,
            "reads_bad_data_json": False,
            "other_months": "The loader has no path that opens a candidate month other than March 2018 and December 2019.",
            "missing_file": "rasterio.open raises if a required GeoTIFF is absent.",
            "present_file": "A present GeoTIFF at the hardcoded dates is read and accepted. The s2, bad-index, and masked flags are not consulted.",
            "label_rule": "(buildings_T2 >= 0.5) XOR (buildings_T1 >= 0.5), with both building maps read from the same two hardcoded dates.",
            "sar_rule": "VV and VH at both hardcoded dates. Validity is NaN absence before zero-fill, not the metadata flags.",
        },
        "pair_definition": {
            "name": "moved_t1_async_pair",
            "historical_date": "candidate month strictly before 2019-12",
            "target_sar_date": "2019-12",
            "required_components": [
                "optical file and optical metadata usable at the candidate month",
                "SAR file and SAR metadata usable at the candidate month",
                "SAR file and SAR metadata usable at 2019-12",
                "building file and building metadata usable at the candidate month",
                "building file and building metadata usable at 2019-12",
            ],
            "confirmed_eligible": "All five components have both the file and the metadata usability flag.",
            "ambiguous": "No component is missing on both file and metadata, but at least one component disagrees.",
            "ineligible": "At least one required component is absent as a file and not metadata-usable.",
            "not_claimed": (
                "A confirmed pair is not a trained gap and is not evidence of temporal-gap sensitivity. "
                "Keeping March 2018 SAR and building dates while swapping only the optical file would "
                "leave the label interval at 21 months, so that substitution is not counted as eligible."
            ),
        },
        "counts": {
            "rows": len(pair_rows),
            "confirmed_eligible": sum(row["pair_status"] == "confirmed_eligible" for row in pair_rows),
            "ambiguous": sum(row["pair_status"] == "ambiguous" for row in pair_rows),
            "ineligible": sum(row["pair_status"] == "ineligible" for row in pair_rows),
        },
        "by_month": by_month,
        "optical_disagreement": {
            "count": len(disagreements),
            "reason_group_raster_ranges": {
                group: raster_group_summary(items) for group, items in raster_groups.items()
            },
            "march_2018_reference_rasters": reference_summaries,
            "interpretation": (
                "These ranges describe file contents. They do not decide whether a metadata flag is correct."
            ),
        },
        "outputs": {
            "pair_csv": str(PAIR_CSV_PATH),
            "pair_json": str(PAIR_JSON_PATH),
            "disagreement_csv": str(DISAGREEMENT_CSV_PATH),
        },
    }
    return pair_rows, report, disagreements


def write_pair_outputs(pair_rows: list[dict], report: dict, disagreements: list[dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with PAIR_CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0]))
        writer.writeheader()
        writer.writerows(pair_rows)
    PAIR_JSON_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with DISAGREEMENT_CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(disagreements[0]))
        writer.writeheader()
        writer.writerows(disagreements)


def verify_pairs(report: dict) -> list[str]:
    failures = []
    counts = report["counts"]
    if counts["confirmed_eligible"] + counts["ambiguous"] + counts["ineligible"] != counts["rows"]:
        failures.append("pair statuses do not sum to the inventory rows")
    for item in report["by_month"]:
        total = item["confirmed_eligible"] + item["ambiguous"] + item["ineligible"]
        if total != item["sites_in_inventory"]:
            failures.append(f"{item['historical_optical_date']}: pair statuses do not cover the sites")
    if report["optical_disagreement"]["count"] != 85:
        failures.append(
            f"disagreement file count is {report['optical_disagreement']['count']}, expected 85"
        )
    return failures


def print_pair_summary(report: dict) -> None:
    counts = report["counts"]
    print("Moved-T1 pair classification")
    print(
        "A confirmed pair needs optical, SAR, and building inputs at the candidate month, "
        "plus SAR and buildings at 2019-12, with file and metadata agreeing."
    )
    print(
        f"Pairs: {counts['rows']}  confirmed_eligible={counts['confirmed_eligible']}  "
        f"ambiguous={counts['ambiguous']}  ineligible={counts['ineligible']}"
    )
    print("Month, gap, eligible, ambiguous, ineligible, eligible train/val/test")
    for item in report["by_month"]:
        print(
            f"  {item['historical_optical_date']}  gap={item['gap_months_to_target_sar_2019_12']:2d}  "
            f"eligible={item['confirmed_eligible']:2d}  ambiguous={item['ambiguous']:2d}  "
            f"ineligible={item['ineligible']:2d}  "
            f"{item['confirmed_eligible_train']}/{item['confirmed_eligible_validation']}/{item['confirmed_eligible_test']}"
        )
    print(f"Disagreement files inspected: {report['optical_disagreement']['count']}")
    print(f"Pair CSV:  {PAIR_CSV_PATH}")
    print(f"Pair JSON: {PAIR_JSON_PATH}")
    print(f"Disagreement CSV: {DISAGREEMENT_CSV_PATH}")


def print_summary(report: dict) -> None:
    counts = report["counts"]
    print("Scenario 2 historical-optical eligibility inventory")
    print(report["existing_experiment"]["statement"])
    print(
        f"Locked sites: {counts['sites']} "
        f"({report['split']['counts']['train']} train / "
        f"{report['split']['counts']['validation']} validation / "
        f"{report['split']['counts']['test']} test)"
    )
    print(f"Split sha256 matches published value: {report['split']['hash_matches_published_value']}")
    print(f"Candidate historical months before 2019-12: {counts['candidate_months']}")
    print(f"Site-month rows: {counts['rows']}")
    print(f"Optical file present: {counts['optical_file_exists']}")
    print(f"Optical metadata usable: {counts['optical_metadata_usable']}")
    print(f"Both file and metadata usable: {counts['optical_file_and_metadata_usable']}")
    print(f"File present but metadata not usable: {counts['file_present_metadata_not_usable']}")
    print(f"Disagreement reason counts: {report['optical_file_metadata_disagreement']['reason_counts']}")
    print(f"Metadata usable but file missing: {counts['metadata_usable_file_missing']}")
    print(f"SAR file without optical file: {counts['sar_file_without_optical_file']}")
    print(f"Ambiguity notes: {len(report['ambiguities'])}")
    print("Month, gap, role, optical file, optical metadata usable, both, SAR file, label file")
    for item in report["by_month"]:
        print(
            f"  {item['historical_optical_date']}  gap={item['gap_months_to_target_sar_2019_12']:2d}  "
            f"{item['role']}  file={item['optical_file_exists']:2d}  "
            f"meta={item['optical_metadata_usable']:2d}  "
            f"both={item['optical_file_and_metadata_usable']:2d}  "
            f"sar={item['sar_file_at_historical_month']:2d}  "
            f"label={item['label_file_at_historical_month']:2d}"
        )
    print(f"CSV:  {CSV_PATH}")
    print(f"JSON: {JSON_PATH}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit historical optical eligibility for the locked Scenario 2 sites.")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Rewrite the inventory, then check the report against the locked split.",
    )
    args = parser.parse_args()
    split_hash_before = sha256_file(SPLIT_FILE)
    split = load_split()
    metadata, bad = load_metadata()
    rows, extra = build_rows(split, metadata, bad)
    report = summarize(rows, extra, split_hash_before)
    write_outputs(rows, report)
    if sha256_file(SPLIT_FILE) != split_hash_before:
        print("ERROR: split file hash changed while the inventory was written.")
        return 1
    failures = verify(split)
    if args.verify:
        print("Verification mode: outputs were written, then checked.")
    print_summary(report)
    pair_rows, pair_report, disagreements = classify_pairs(rows, extra, metadata, bad)
    write_pair_outputs(pair_rows, pair_report, disagreements)
    pair_failures = verify_pairs(pair_report)
    print_pair_summary(pair_report)
    failures.extend(pair_failures)
    if sha256_file(SPLIT_FILE) != split_hash_before:
        print("ERROR: split file hash changed while pair outputs were written.")
        return 1
    if failures:
        print("Verification: FAIL")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("Verification: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
