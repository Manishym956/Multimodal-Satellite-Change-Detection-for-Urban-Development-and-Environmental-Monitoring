# GPU / CPU Device Audit

Audit date: 2026-10-06
Environment: NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM), Windows, PyTorch 2.5.1+cu124, CUDA 12.4

---

## Audit finding: No GPU execution defect in Scenario 1

All completed Scenario 1 training scripts were audited. **No script hardcodes CPU for neural network execution.** Every script uses the pattern:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

and correctly:
- Moves the model with `.to(device)` before training
- Moves `image` and `label` tensors to `device` inside each batch loop and in evaluation
- Uses `map_location="cpu"` when loading checkpoints, then `.to(device)` afterwards
- Computes loss, backward, and optimizer step on the selected device

The Scenario 1 experiments ran on the RTX 4060, which is confirmed by the `gpu_name: "NVIDIA GeForce RTX 4060 Laptop GPU"` field recorded in every `metrics.json`.

---

## What was intentionally on CPU (not a defect)

| Code path | Location | CPU use | Reason |
|---|---|---|---|
| Rasterio / OpenCV reads | `oscd_dataset.py`, `temporal_pairing_analysis.py` | CPU | Standard I/O — correct |
| NumPy array construction | Dataset `__getitem__` | CPU | Before `.to(device)` in training loop — correct |
| `TF.to_tensor()` | Dataset loaders | CPU | Tensor construction before GPU transfer — correct |
| Shape checks (32×32 pass) in `smoke_test_transformers.py` Tests 3/4/8 | Smoke test | CPU | Deliberately testing CPU shapes before GPU tests — correct |
| Metric accumulation (TP/FP/FN/TN counts) | `metrics.py`, evaluation functions | CPU (via `.item()`) | Scalar extraction for counting — correct |
| Checkpoint state dict save | `save_checkpoint` | CPU copy | `.detach().cpu().clone()` before `torch.save` — correct |
| Gradient accumulation check | Verification functions | CPU (via `.item()`) | Scalar check only — correct |

---

## What was missing (gaps, not defects)

| Gap | Affected files | Status |
|---|---|---|
| No shared device utility | All scripts copy-paste the same one-liner | **Fixed**: `src/utils/device.py` |
| No device status banner at run start | Training scripts print nothing about GPU | **Fixed**: `get_device()` prints banner |
| No `--device` CLI argument | Users cannot request CPU without editing code | **Fixed**: `add_device_argument()` helper; added to new scripts |
| No explicit device placement test | No automated test confirming CUDA execution | **Fixed**: `src/training/test_device.py` |
| `pin_memory` not set | DataLoaders do not enable `pin_memory=True` | **Not changed** — see note |
| Scenario 2 noted here as unimplemented stubs | `train_scenario2_*.py` | **Superseded.** The loader and trainers exist. Completed runs are under `experiments/scenario_2/`, with `device: cuda` and `pos_weight: 94.0` in the saved metrics. |

**Note on `pin_memory`:** Pinned memory (`pin_memory=True` in DataLoader) enables faster host→GPU transfers via DMA. With `num_workers=0` (the current setting), workers run in the main process and `pin_memory` provides marginal benefit. It was not added to avoid unnecessary changes to Scenario 1 scripts. It can be enabled in Scenario 2 scripts if data loading is a bottleneck.

---

## Files created or changed

| File | Action | Change description |
|---|---|---|
| `src/utils/device.py` | **Created** | Shared `get_device(preference)` + `add_device_argument()` |
| `src/utils/__init__.py` | **Updated** | Exports `get_device`, `add_device_argument` |
| `src/training/test_device.py` | **Created** | 9-test device placement verification |
| `src/training/verify_backward_compat.py` | **Created** | 27-check backward-compat + loader validation |
| `src/datasets/scenario2_dataset.py` | **Created** | SEN12-MT dataset loader (async + sync modes) |
| `src/training/train_scenario2_dual_stream.py` | **Created** | Scenario 2 CNN training (async + sync) |
| `src/training/train_scenario2_cross_attention.py` | **Created** | Scenario 2 transformer training |
| `src/models/dual_stream_unet.py` | **Updated** | Constructor accepts `sar_channels`/`opt_channels`; forward uses instance attrs. Scenario 1 defaults (2+26=28ch) unchanged. |
| `src/models/dual_stream_unet_clean.py` | **Updated** | Same pattern. Scenario 1 defaults (2+13=15ch) unchanged. |
| `src/models/transformer_cross_attention.py` | **Updated** | Forward validation uses `self.sar_channels + self.opt_channels` instead of hardcoded `TOTAL_CHANNELS=15`. Scenario 1 (15ch) unchanged. |
| `docs/GPU_DEVICE_AUDIT.md` | **Updated** | Added verification results and Scenario 2 model changes |

**No Scenario 1 training scripts were modified.** Their device selection logic is correct and the completed experiment results are unaffected.

---

## Device verification test results (confirmed executed)

```
Execution device: GPU — NVIDIA GeForce RTX 4060 Laptop GPU (8.0 GB VRAM)
                  |  PyTorch 2.5.1+cu124  |  CUDA 12.4

Test 1: get_device('auto')        → PASS  (returned cuda)
Test 2: get_device('cpu')         → PASS  (returned cpu)
Test 3: Model params on CUDA      → PASS
Test 4: Input/label on CUDA       → PASS  (cuda:0)
Test 5: Forward pass on CUDA      → PASS  (output on cuda:0, finite)
Test 6: Loss/backward/optim       → PASS  (loss on cuda:0, finite grads, optimizer OK)
Test 7: CUDA memory observed      → PASS  (0.1 MB allocated, 2.0 MB reserved)
Test 8: CPU mode explicit         → PASS
Test 9: Checkpoint save/load      → PASS  (map_location='cpu', then moved to cuda:0)

All 9 tests PASSED
```

## Scenario 2 verify-only results (confirmed executed)

```
train_scenario2_cross_attention.py --verify-only
  Device banner: GPU — NVIDIA GeForce RTX 4060 Laptop GPU (8.0 GB VRAM) | CUDA 12.4
  1. Input channels:  14 (4 SAR + 10 S2 T1)  SAR [0,1], S2 T1 [0,1]
  2. S2 T2 status:    NOT USED (14-ch input, no T2 slots)
  3. Cross-attention: SAR → Query, Historical Optical → Key/Value ✓
  4. 32×32 forward:   (1,1,32,32) on cuda:0 ✓
  5. 489×488 forward: (1,1,489,488) ✓
  6. Backward pass:   loss=1.124, grads present ✓
  Parameter count: 1,154,945
  Exit code: 0

train_scenario2_dual_stream.py --mode async --verify-only
  Device banner: GPU — NVIDIA GeForce RTX 4060 Laptop GPU (8.0 GB VRAM) | CUDA 12.4
  1. Input channels:  14 (4 SAR + 10 S2 T1 [T2 withheld])
  3. 32×32 forward:   (1,1,32,32) on cuda:0 ✓
  4. Backward pass:   loss=1.115, grads present ✓
  Parameter count: 27,276,033
  Exit code: 0

train_scenario2_dual_stream.py --mode sync --verify-only
  Device banner: GPU — NVIDIA GeForce RTX 4060 Laptop GPU (8.0 GB VRAM) | CUDA 12.4
  1. Input channels:  24 (4 SAR + 10 S2 T1 + 10 S2 T2)
  3. 32×32 forward:   (1,1,32,32) on cuda:0 ✓
  4. Backward pass:   loss=1.988, grads present ✓
  Parameter count: 27,281,793
  Exit code: 0
```

## Backward-compatibility test results (confirmed executed)

`verify_backward_compat.py` — 27 checks, all PASS:
- DualStreamUNetClean default: 27,276,609 params, 15-ch forward correct
- DualStreamUNet default: 27,284,097 params, 28-ch forward correct
- All three Scenario 1 checkpoints reload cleanly (asynchronous_dual_stream_clean ep26, dual_stream_baseline ep31, transformer_cross_attention ep33)
- Scenario 2 overrides (14-ch async, 24-ch sync, 14-ch transformer) all correct
- Dataset loader: async=14ch, sync=24ch, no NaN after fill, label binary and aligned
- Training-split class ratio: 4,718,348 no-change / 50,387 change = 93.6 to one decimal place. Saved Scenario 2 metrics use training `pos_weight` 94.0. 93.6 is that ratio, not the weight passed to the loss.
- No path collision between Scenario 1 and Scenario 2 output directories

CUDA confirmed used: all Scenario 2 forward passes executed on cuda:0.

---

## Scenario 2 device requirements

`src/datasets/scenario2_dataset.py` and the Scenario 2 training scripts are implemented. Saved metrics record CUDA on the RTX 4060 Laptop GPU and training `pos_weight` 94.0. The device rules for those runs are:

1. Import and call `get_device(args.device)` — do not repeat the raw `torch.cuda.is_available()` one-liner.
2. Add `--device` via `add_device_argument(parser)` in `parse_args()`.
3. Move model, image tensors, and label tensors to the returned device before any computation.
4. Use `map_location="cpu"` when loading checkpoints, then `.to(device)`.
5. The pos_weight tensor for BCEWithLogitsLoss must also be moved to device: `torch.tensor([94.0]).to(device)`.

---

## Device status message you will see at every launch

When CUDA is available (normal case):
```
Execution device: GPU — NVIDIA GeForce RTX 4060 Laptop GPU (8.0 GB VRAM)  |  PyTorch 2.5.1+cu124  |  CUDA 12.4
```

When CPU is explicitly requested (`--device cpu`):
```
Execution device: CPU — CPU explicitly requested  |  PyTorch 2.5.1+cu124
```

When CUDA is unavailable (e.g. no GPU driver):
```
Execution device: CPU — CUDA unavailable  |  PyTorch 2.5.1+cu124
```
