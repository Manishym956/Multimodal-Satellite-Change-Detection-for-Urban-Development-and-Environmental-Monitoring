"""GPU/CPU device placement verification test.

Verifies:
  1. get_device() returns the correct torch.device.
  2. A model placed on CUDA has all parameters on CUDA.
  3. Input tensors reach the same device as the model.
  4. Forward pass, loss, backward pass, and optimizer step execute correctly.
  5. Model gradients are finite.
  6. CUDA memory allocation is observable after GPU forward pass.
  7. CPU mode works when explicitly requested.
  8. Model and tensor device types match at every step.

Run from project root:
    .venv\\Scripts\\python.exe src\\training\\test_device.py

Exit code 0 = all tests passed.
Exit code 1 = one or more tests failed.
"""

import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.device import get_device

# ── colour helpers ────────────────────────────────────────────────────────────
PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
INFO = "\033[94m[INFO]\033[0m"

failures: list[str] = []


def ok(msg: str) -> None:
    print(f"  {PASS} {msg}")


def fail(msg: str) -> None:
    print(f"  {FAIL} {msg}")
    failures.append(msg)


def info(msg: str) -> None:
    print(f"  {INFO} {msg}")


def header(title: str) -> None:
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print(f"{'=' * 65}")


# ── tiny model for testing ─────────────────────────────────────────────────────
class _TinyConv(nn.Module):
    """Minimal conv model: (B, 4, H, W) → (B, 1, H, W)."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ── Test 1: get_device("auto") ────────────────────────────────────────────────
header("Test 1: get_device('auto')")
device_auto = get_device("auto")
info(f"Returned device: {device_auto}")
if torch.cuda.is_available():
    if device_auto.type == "cuda":
        ok("auto → cuda (CUDA is available)")
    else:
        fail(f"auto should return cuda when CUDA is available, got {device_auto.type}")
else:
    if device_auto.type == "cpu":
        ok("auto → cpu (CUDA unavailable)")
    else:
        fail(f"auto should return cpu when CUDA is unavailable, got {device_auto.type}")

# ── Test 2: get_device("cpu") ─────────────────────────────────────────────────
header("Test 2: get_device('cpu')")
device_cpu = get_device("cpu")
if device_cpu.type == "cpu":
    ok("get_device('cpu') → cpu")
else:
    fail(f"Expected cpu, got {device_cpu.type}")

# ── Test 3: model parameters on correct device ────────────────────────────────
header("Test 3: Model parameters on selected device")
model = _TinyConv().to(device_auto)
param_devices = {p.device.type for p in model.parameters()}
info(f"Model parameter device types: {param_devices}")
if len(param_devices) == 1 and device_auto.type in param_devices:
    ok(f"All model parameters on {device_auto.type}")
else:
    fail(f"Parameter devices {param_devices} do not match selected device {device_auto.type}")

# ── Test 4: input tensor on same device as model ──────────────────────────────
header("Test 4: Input tensor device matches model device")
x = torch.randn(2, 4, 32, 32).to(device_auto)
label = torch.zeros(2, 1, 32, 32).to(device_auto)
label[0, 0, 10:15, 10:15] = 1.0

info(f"Input device:  {x.device}")
info(f"Label device:  {label.device}")
info(f"Model device:  {next(model.parameters()).device}")

if x.device.type == device_auto.type:
    ok(f"Input tensor is on {x.device.type}")
else:
    fail(f"Input tensor on {x.device.type}, expected {device_auto.type}")

if label.device.type == device_auto.type:
    ok(f"Label tensor is on {label.device.type}")
else:
    fail(f"Label tensor on {label.device.type}, expected {device_auto.type}")

# ── Test 5: forward pass ──────────────────────────────────────────────────────
header("Test 5: Forward pass")
model.eval()
try:
    with torch.no_grad():
        logits = model(x)
    info(f"Output device: {logits.device}")
    info(f"Output shape:  {tuple(logits.shape)}")
    if logits.device.type == device_auto.type:
        ok(f"Logits are on {logits.device.type}")
    else:
        fail(f"Logits on {logits.device.type}, expected {device_auto.type}")
    if torch.isfinite(logits).all():
        ok("Logits are finite")
    else:
        fail("Logits contain NaN or Inf")
except Exception as e:
    fail(f"Forward pass raised: {e}")

# ── Test 6: loss, backward, optimizer ─────────────────────────────────────────
header("Test 6: Loss, backward pass, optimizer step")
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.BCEWithLogitsLoss()

try:
    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    loss = criterion(logits, label)

    info(f"Loss device: {loss.device}")
    info(f"Loss value:  {loss.item():.6f}")

    if loss.device.type == device_auto.type:
        ok(f"Loss is on {loss.device.type}")
    else:
        fail(f"Loss on {loss.device.type}, expected {device_auto.type}")

    if torch.isfinite(loss):
        ok("Loss is finite")
    else:
        fail("Loss is NaN or Inf")

    loss.backward()

    # Gradient finiteness
    nan_grads = [
        n for n, p in model.named_parameters()
        if p.grad is not None and not torch.isfinite(p.grad).all()
    ]
    if not nan_grads:
        ok("All gradients are finite")
    else:
        fail(f"Non-finite gradients in: {nan_grads}")

    # Gradient non-zero
    zero_grads = [
        n for n, p in model.named_parameters()
        if p.grad is not None and p.grad.abs().sum().item() == 0
    ]
    nonzero_count = sum(
        1 for p in model.parameters()
        if p.grad is not None and p.grad.abs().sum().item() > 0
    )
    if nonzero_count > 0:
        ok(f"Non-zero gradients present ({nonzero_count} parameter groups)")
    else:
        fail("All gradients are zero")

    optimizer.step()
    ok("Optimizer step completed without error")

except Exception as e:
    fail(f"Training step raised: {e}")
    import traceback; traceback.print_exc()

# ── Test 7: CUDA memory observable when GPU selected ──────────────────────────
header("Test 7: CUDA memory allocation")
if device_auto.type == "cuda":
    mem_allocated = torch.cuda.memory_allocated(0) / 1024 ** 2
    mem_reserved  = torch.cuda.memory_reserved(0)  / 1024 ** 2
    info(f"CUDA memory allocated: {mem_allocated:.1f} MB")
    info(f"CUDA memory reserved:  {mem_reserved:.1f} MB")
    if mem_allocated > 0 or mem_reserved > 0:
        ok("CUDA memory is allocated — model/tensors are on GPU")
    else:
        fail("No CUDA memory allocated — model may not be on GPU")
else:
    info("CPU mode — CUDA memory check skipped")
    ok("CPU mode confirmed, no GPU memory expected")

# ── Test 8: CPU mode works correctly ─────────────────────────────────────────
header("Test 8: CPU mode explicit test")
try:
    cpu_model = _TinyConv().to(torch.device("cpu"))
    x_cpu = torch.randn(1, 4, 16, 16)
    with torch.no_grad():
        out_cpu = cpu_model(x_cpu)
    if out_cpu.device.type == "cpu":
        ok("CPU forward pass succeeded")
    else:
        fail(f"CPU model output on unexpected device: {out_cpu.device}")
    if torch.isfinite(out_cpu).all():
        ok("CPU output is finite")
    else:
        fail("CPU output contains NaN/Inf")
except Exception as e:
    fail(f"CPU mode test raised: {e}")

# ── Test 9: Checkpoint save/load with map_location ────────────────────────────
header("Test 9: Checkpoint save and load (map_location='cpu')")
import tempfile, os
try:
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = f.name
    torch.save({"model_state_dict": state}, tmp_path)
    loaded = torch.load(tmp_path, map_location="cpu", weights_only=False)
    cpu_model2 = _TinyConv()
    cpu_model2.load_state_dict(loaded["model_state_dict"])
    cpu_model2.to(device_auto)
    ok("Checkpoint saved and loaded with map_location='cpu', then moved to device")
    info(f"Loaded model device: {next(cpu_model2.parameters()).device}")
    os.unlink(tmp_path)
except Exception as e:
    fail(f"Checkpoint test raised: {e}")

# ── Summary ────────────────────────────────────────────────────────────────────
header("SUMMARY")
print(f"\n  Selected device:  {device_auto}")
if device_auto.type == "cuda":
    print(f"  GPU name:         {torch.cuda.get_device_name(0)}")
    print(f"  PyTorch version:  {torch.__version__}")
    print(f"  CUDA version:     {torch.version.cuda}")
print()

if failures:
    print(f"  {FAIL} {len(failures)} test(s) FAILED:")
    for msg in failures:
        print(f"    • {msg}")
    sys.exit(1)
else:
    print(f"  {PASS} All device placement tests PASSED")
    sys.exit(0)
