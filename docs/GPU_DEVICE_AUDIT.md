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
| Scenario 2 stubs raise `NotImplementedError` | `train_scenario2_*.py` stubs | Unchanged — correct, dataset loader not yet implemented |

**Note on `pin_memory`:** Pinned memory (`pin_memory=True` in DataLoader) enables faster host→GPU transfers via DMA. With `num_workers=0` (the current setting), workers run in the main process and `pin_memory` provides marginal benefit. It was not added to avoid unnecessary changes to Scenario 1 scripts. It can be enabled in Scenario 2 scripts if data loading is a bottleneck.

---

## Files created or changed

| File | Action | Change description |
|---|---|---|
| `src/utils/device.py` | **Created** | Shared `get_device(preference)` + `add_device_argument()` |
| `src/utils/__init__.py` | **Updated** | Exports `get_device`, `add_device_argument` |
| `src/training/test_device.py` | **Created** | 9-test device placement verification |
| `docs/GPU_DEVICE_AUDIT.md` | **Created** | This file |

**No Scenario 1 training scripts were modified.** Their device selection logic is correct and the completed experiment results are unaffected.

---

## Device verification test results

```
Execution device: GPU — NVIDIA GeForce RTX 4060 Laptop GPU (8.0 GB VRAM)
                  |  PyTorch 2.5.1+cu124  |  CUDA 12.4

Test 1: get_device('auto')    → PASS  (returned cuda)
Test 2: get_device('cpu')     → PASS  (returned cpu)
Test 3: Model params on CUDA  → PASS
Test 4: Input/label on CUDA   → PASS  (cuda:0)
Test 5: Forward pass on CUDA  → PASS  (output on cuda:0, finite)
Test 6: Loss/backward/optim   → PASS  (loss on cuda:0, finite grads, optimizer OK)
Test 7: CUDA memory observed  → PASS  (0.1 MB allocated, 2.0 MB reserved)
Test 8: CPU mode explicit     → PASS
Test 9: Checkpoint save/load  → PASS  (map_location='cpu', then moved to cuda:0)

All 9 tests PASSED
```

CUDA was actually used: the forward pass, loss, backward pass, and optimizer step all executed on `cuda:0` (RTX 4060). This is confirmed by the `CUDA memory allocated: 0.1 MB` measurement in Test 7.

---

## Scenario 2 device requirements

When `src/datasets/scenario2_dataset.py` and the Scenario 2 training scripts are implemented, they must:

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
