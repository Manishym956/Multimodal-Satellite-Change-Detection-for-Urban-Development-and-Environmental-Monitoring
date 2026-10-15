# RQ2 Dataset Provenance Investigation

**Scope:** read-only provenance review. No dataset, loader, split, existing audit/results, or classifications were changed; no models were trained and no metrics were calculated.

## Findings

- **Authoritatively documented — upstream release:** Zenodo identifies [SEN12 Multi-Temporal Urban Mapping Dataset](https://zenodo.org/records/7794693), with monthly mean Sentinel-1, cloud-free Sentinel-2, and rasterized built-up labels. It lists `sen12_mt_urbanmapping.tar.gz` (18.1 GB), MD5 `5d34fc4537c7974c31824baaaea050f5`. That release summary does not define the disputed record-level quality fields.
- **Supported by repository implementation — index/date mapping:** `generate_manifest.py` uses zero-based `enumerate(records)` positions for `bad_data.json[site].S2`; it builds monthly paths from the record's year and month. For all 28 locked sites, metadata dates are unique and chronological; all 85 conflicting optical files map to one matching date record and exist at the corresponding path. This confirms the repository's mapping convention, not why the producer flagged a record.
- **Supported by repository implementation — loader behavior:** `scenario2_dataset.py` reads the fixed-date rasters directly and does not consult `metadata.json` or `bad_data.json`. This describes code behavior only; it does not validate the optical data.
- **Unresolved — `s2=false`, bad-S2 indices, and `masked=true`:** local notes/code use these as availability/exclusion signals, but neither the local dataset files nor the inspected Zenodo description defines specific reasons for individual flags. Do not interpret the 85 rasters as valid based on names or pixel appearance.
- **Unresolved — source-file authentication:** Zenodo publishes an archive MD5, but no matching archive, sidecar checksum, or upstream file manifest was found in the project workspace. The local manifest is generated from metadata and contains file paths/presence, not checksums. Local JSON SHA-256 values in the JSON deliverable are fingerprints only and do not authenticate the extracted rasters.

## The 85 conflicting optical records

| Flag combination | Count |
|---|---:|
| `s2=true, bad_s2_index=false, masked=true` | 24 |
| `s2=false, bad_s2_index=true, masked=false` | 45 |
| `s2=false, bad_s2_index=true, masked=true` | 16 |

The JSON deliverable lists each record with split, date, zero-based metadata index, exact optical path, file size, all three flags, and the available provenance assessment. The existing audit classifications were not changed.

## Evidence and remaining gaps

Local evidence: `datasets/sen12_mt_urbanmapping/multimodal_cd_dataset/metadata.json`, `bad_data.json`; `src/training/generate_manifest.py`; `src/training/temporal_pairing_analysis.py`; `src/datasets/scenario2_dataset.py`; `docs/SCENARIO_2_DATASET_INSPECTION.md`; `docs/SCENARIO_2_DATASET_MANIFEST.csv`; and `experiments/scenario_2_temporal_gap/optical_disagreement_files.csv`.

To resolve the semantics, obtain the dataset data dictionary, bad-index and masked-field generation code/logs, and per-record quality provenance. To establish source integrity, obtain the original Zenodo archive and verify its published MD5, then compare archive members or trusted per-file checksums with the local extraction.

**Split integrity:** SHA-256 `5d0841adf0c651166cb18ecea310c6ef8cce69172065fc2d699a29083c272e27`; expected `5d0841adf0c651166cb18ecea310c6ef8cce69172065fc2d699a29083c272e27`; match: true.
