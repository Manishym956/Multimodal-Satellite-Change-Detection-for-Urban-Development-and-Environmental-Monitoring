# Multimodal Satellite Change Detection for Urban Development and Environmental Monitoring

## 1. Overview

This project studies urban change detection from Sentinel-1 SAR and Sentinel-2 optical imagery on the Onera Satellite Change Detection (OSCD) benchmark.

The working baselines ask how well a standard U-Net can detect pixel-level change from SAR alone and from optical imagery alone. Both models use the same city split, the same loss, and the same full-image test evaluation. These runs are reference points for later multimodal experiments.

The preprocessing follows the numeric steps used by the DS-UNet OSCD pipeline. The network itself is a single-stream U-Net written for this project. It is not the DS-UNet architecture.

## 2. Research Question

Can historical Sentinel-2 optical imagery improve Sentinel-1 SAR-based urban change detection when synchronous optical imagery is unavailable, and how does the temporal gap between the historical optical image and target SAR observation affect detection performance?

## 3. Current Research Status

The repository currently contains:

- DS-UNet-faithful OSCD preprocessing
- a completed SAR-only U-Net baseline
- a completed optical-only U-Net baseline

SAR and optical fusion is not yet completed. The asynchronous historical-optical experiment is not yet completed.

## 4. Dataset

The benchmark is OSCD: 24 cities, bitemporal Sentinel-2 imagery, and pixel-level urban change masks. The official OSCD division has 14 training cities and 10 test cities. Section 6 records how those 14 training cities are divided for the baselines.

Sentinel-1 VV imagery aligned to those cities is used by the same preprocessing pipeline. Sentinel-2 contributes 13 optical bands at each date.

The imagery, masks, and Sentinel-1 GeoTIFFs are not included in this Git repository.

## 5. Preprocessing

`src/datasets/oscd_dataset.py` reproduces the current baseline preprocessing.

- Sentinel-2 uses 13 rectified bands, divided by 10000 and clipped to [0, 1].
- Sentinel-1 uses the VV band, resized to the change-mask width and height with bicubic interpolation.
- This preprocessing step does not reproject coordinates.
- Raw mask values 1 and 2 become 0 and 1 after subtracting 1.
- Training samples random 32×32 crops.
- Validation and the final test pass score each city on the full image.

## 6. Experimental Split

Four of the 14 official OSCD training cities are held out as validation. The selection is fixed with seed 42. Cities are not split by pixel or by crop.

**Train:** bordeaux, nantes, rennes, saclay_e, abudhabi, cupertino, pisa, hongkong, beirut, mumbai

**Validation:** beihai, bercy, aguasclaras, paris

**Test:** brasilia, montpellier, norcia, rio, saclay_w, valencia, dubai, lasvegas, milano, chongqing

The checkpoint is the epoch with the highest validation F1. The 10 official test cities are evaluated once after that checkpoint is selected.

## 7. Baselines

These figures are the current pooled test results. They are not the final answer to the research question.

**SAR-only U-Net.** Input is Sentinel-1 T1 concatenated with T2 (2 channels). Best validation epoch: 21. Validation F1: 0.0662. Final pooled test F1: 0.3558.

**Optical-only U-Net.** Input is Sentinel-2 T1 concatenated with T2 (26 channels). Best validation epoch: 50. Validation F1: 0.2387. Final pooled test F1: 0.2663.

Both models use BCEWithLogitsLoss, Adam, seed 42, 50 epochs, batch size 64, learning rate 1e-4, and 64 crops of 32×32 per training city per epoch.

## 8. Repository Structure

| Path | Role |
| --- | --- |
| `src/` | Dataset loading, U-Net, metrics, and baseline training |
| `configs/` | Reserved for experiment configuration |
| `experiments/` | Metrics, histories, and config records for completed baselines |
| `figures/` | Prediction figures for one test city per baseline |
| `paper/` | Reserved for manuscript material |
| `notebooks/` | Reserved for notebooks |

`datasets/`, `checkpoints/`, `.venv/`, and `outputs/` are excluded from Git.

## 9. Environment

The completed baselines were trained with Python 3.11 and PyTorch 2.5.1+cu124 on a CUDA-enabled NVIDIA GeForce RTX 4060 Laptop GPU.

Install the pinned packages from `requirements.txt` into a virtual environment before running the commands below.

## 10. Reproduction

From the project directory, with the local virtual environment:

```text
.venv\Scripts\python.exe src\training\train_sar_baseline.py --epochs 50 --batch-size 64 --lr 1e-4 --seed 42 --crop-size 32 --crops-per-city 64
```

```text
.venv\Scripts\python.exe src\training\train_optical_baseline.py --epochs 50 --batch-size 64 --lr 1e-4 --seed 42 --crop-size 32 --crops-per-city 64
```

These commands expect the local OSCD and Sentinel-1 files described above. Those files are not in Git.

## 11. Data availability

Large datasets, model checkpoints, the Python virtual environment, and runtime output directories are intentionally left out of Git. The tracked experiment tables and figures record the completed baseline runs. This README does not provide dataset download URLs.

## 12. Research roadmap

The following work is planned and is not implemented yet:

- SAR and optical fusion
- asynchronous fusion that uses historical optical imagery with SAR
- experiments on the temporal gap between the optical date and the SAR observation
- SAR polarization experiments, where the available Sentinel-1 bands support them
- ablation studies
