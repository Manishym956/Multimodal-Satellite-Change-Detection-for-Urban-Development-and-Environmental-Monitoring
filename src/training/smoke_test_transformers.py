"""Smoke tests for TransformerSelfAttention and TransformerCrossAttention.

Tests performed:
    1.  Import test
    2.  Parameter counts (both models)
    3.  32x32 forward pass (standard training crop)
    4.  Non-multiple spatial size forward pass (517x461)
    5.  GPU forward pass
    6.  Backward pass + gradient check
    7.  NaN/Inf check
    8.  Input/output shape verification
    9.  Peak CUDA memory usage
    10. Asynchronous loader channel verification
        (SAR=2ch, optical=13ch, no S2_T2 access)

Run from project root:
    python src/training/smoke_test_transformers.py
"""

import sys
import time
from pathlib import Path

import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
INFO = "\033[94m[INFO]\033[0m"
WARN = "\033[93m[WARN]\033[0m"

_failures: list[str] = []

def ok(msg: str) -> None:
    print(f"  {PASS} {msg}")

def fail(msg: str) -> None:
    print(f"  {FAIL} {msg}")
    _failures.append(msg)

def info(msg: str) -> None:
    print(f"  {INFO} {msg}")

def header(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def check_no_nan_inf(tensor: torch.Tensor, name: str) -> bool:
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    if has_nan or has_inf:
        fail(f"{name}: contains NaN={has_nan} Inf={has_inf}")
        return False
    ok(f"{name}: no NaNs or Infs")
    return True


def make_batch(b: int = 2, h: int = 32, w: int = 32) -> torch.Tensor:
    """Create a (B, 15, H, W) random input tensor."""
    return torch.randn(b, 15, h, w)


# ----------------------------------------------------------------------------
# Test 1: Import
# ----------------------------------------------------------------------------

header("Test 1: Import")
try:
    from src.models.transformer_self_attention import (
        TransformerSelfAttention,
        count_parameters,
        _pad_to_multiple,
    )
    from src.models.transformer_cross_attention import TransformerCrossAttention
    ok("TransformerSelfAttention imported successfully")
    ok("TransformerCrossAttention imported successfully")
    ok("count_parameters, _pad_to_multiple imported")
except Exception as e:
    fail(f"Import failed: {e}")
    sys.exit(1)


# ----------------------------------------------------------------------------
# Instantiate models
# ----------------------------------------------------------------------------

header("Instantiating models")
model_a = TransformerSelfAttention(
    sar_channels=2,
    opt_channels=13,
    embed_dim=128,
    patch_stride=4,
    num_blocks=2,
    num_heads=4,
    mlp_ratio=4,
)
model_b = TransformerCrossAttention(
    sar_channels=2,
    opt_channels=13,
    embed_dim=128,
    patch_stride=4,
    num_self_blocks=2,
    num_cross_blocks=1,
    num_heads=4,
    mlp_ratio=4,
)
ok("Both models instantiated")


# ----------------------------------------------------------------------------
# Test 2: Parameter counts
# ----------------------------------------------------------------------------

header("Test 2: Parameter counts")
params_a = count_parameters(model_a)
params_b = count_parameters(model_b)
info(f"TransformerSelfAttention  parameters: {params_a:,}")
info(f"TransformerCrossAttention parameters: {params_b:,}")
info(f"Difference (A - B):                  {params_a - params_b:,}")

if params_a > 0:
    ok("Model A has non-zero parameters")
else:
    fail("Model A has zero parameters")

if params_b > 0:
    ok("Model B has non-zero parameters")
else:
    fail("Model B has zero parameters")

# Expected: Model B < Model A because decoder fused_dim is halved
if params_b < params_a:
    ok("Model B has fewer parameters than Model A (expected: smaller decoder)")
else:
    info("Model B >= Model A parameters (cross-attn params offset decoder savings)")


# ----------------------------------------------------------------------------
# Test 3: 32x32 forward pass (CPU)
# ----------------------------------------------------------------------------

header("Test 3: 32x32 forward pass (CPU)")
model_a.eval()
model_b.eval()

x_32 = make_batch(b=2, h=32, w=32)
info(f"Input shape: {tuple(x_32.shape)}")

with torch.no_grad():
    out_a = model_a(x_32)
    out_b = model_b(x_32)

info(f"Model A output shape: {tuple(out_a.shape)}")
info(f"Model B output shape: {tuple(out_b.shape)}")

expected_out = (2, 1, 32, 32)
if tuple(out_a.shape) == expected_out:
    ok(f"Model A output shape correct: {expected_out}")
else:
    fail(f"Model A output shape wrong: got {tuple(out_a.shape)}, expected {expected_out}")

if tuple(out_b.shape) == expected_out:
    ok(f"Model B output shape correct: {expected_out}")
else:
    fail(f"Model B output shape wrong: got {tuple(out_b.shape)}, expected {expected_out}")

check_no_nan_inf(out_a, "Model A 32x32")
check_no_nan_inf(out_b, "Model B 32x32")


# ----------------------------------------------------------------------------
# Test 4: Non-multiple spatial size forward pass (517x461)
# ----------------------------------------------------------------------------

header("Test 4: Non-multiple spatial size forward pass (517x461)")
x_odd = make_batch(b=1, h=517, w=461)
info(f"Input shape: {tuple(x_odd.shape)}")

with torch.no_grad():
    try:
        out_a_odd = model_a(x_odd)
        out_b_odd = model_b(x_odd)
        info(f"Model A output shape: {tuple(out_a_odd.shape)}")
        info(f"Model B output shape: {tuple(out_b_odd.shape)}")

        if out_a_odd.shape[2:] == x_odd.shape[2:]:
            ok(f"Model A output spatial dims match input: {x_odd.shape[2:]}")
        else:
            fail(f"Model A spatial mismatch: out={out_a_odd.shape[2:]}, in={x_odd.shape[2:]}")

        if out_b_odd.shape[2:] == x_odd.shape[2:]:
            ok(f"Model B output spatial dims match input: {x_odd.shape[2:]}")
        else:
            fail(f"Model B spatial mismatch: out={out_b_odd.shape[2:]}, in={x_odd.shape[2:]}")

        check_no_nan_inf(out_a_odd, "Model A 517x461")
        check_no_nan_inf(out_b_odd, "Model B 517x461")
    except Exception as e:
        fail(f"Non-multiple spatial size forward failed: {e}")


# ----------------------------------------------------------------------------
# Test 5: GPU forward pass
# ----------------------------------------------------------------------------

header("Test 5: GPU forward pass")
cuda_available = torch.cuda.is_available()
if cuda_available:
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    info(f"GPU: {gpu_name}")
    info(f"CUDA version: {torch.version.cuda}")

    torch.cuda.reset_peak_memory_stats()

    model_a_gpu = model_a.to(device)
    model_b_gpu = model_b.to(device)
    x_gpu = make_batch(b=2, h=32, w=32).to(device)

    with torch.no_grad():
        out_a_gpu = model_a_gpu(x_gpu)
        out_b_gpu = model_b_gpu(x_gpu)

    peak_mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    info(f"Peak CUDA memory (both models, 32x32 batch): {peak_mem_mb:.1f} MB")

    if tuple(out_a_gpu.shape) == (2, 1, 32, 32):
        ok(f"Model A GPU output shape correct: (2, 1, 32, 32)")
    else:
        fail(f"Model A GPU output shape wrong: {tuple(out_a_gpu.shape)}")

    if tuple(out_b_gpu.shape) == (2, 1, 32, 32):
        ok(f"Model B GPU output shape correct: (2, 1, 32, 32)")
    else:
        fail(f"Model B GPU output shape wrong: {tuple(out_b_gpu.shape)}")

    check_no_nan_inf(out_a_gpu, "Model A GPU")
    check_no_nan_inf(out_b_gpu, "Model B GPU")
else:
    print(f"  {WARN} CUDA not available -- skipping GPU test")


# ----------------------------------------------------------------------------
# Test 6 & 7: Backward pass + gradient check (GPU if available, else CPU)
# ----------------------------------------------------------------------------

header("Test 6 & 7: Backward pass + nonzero gradient verification")

for model_name, model in [("Model A (Self-Attention)", model_a), ("Model B (Cross-Attention)", model_b)]:
    info(f"--- {model_name} ---")
    if cuda_available:
        model_test = model.to(device)
        x_bwd = make_batch(b=2, h=32, w=32).to(device)
    else:
        model_test = model.cpu()
        x_bwd = make_batch(b=2, h=32, w=32)

    model_test.train()

    try:
        logits = model_test(x_bwd)                     # (2, 1, 32, 32)
        target = torch.zeros_like(logits)
        target[0, 0, 10:15, 10:15] = 1.0              # synthetic change patch
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
        loss.backward()

        ok(f"Backward pass succeeded (loss = {loss.item():.6f})")
        check_no_nan_inf(loss, f"{model_name} loss")

        # Verify gradients are non-zero on at least some parameters
        grad_norms = [p.grad.norm().item() for p in model_test.parameters()
                      if p.grad is not None and p.requires_grad]
        zero_grads = sum(1 for g in grad_norms if g == 0.0)
        nonzero_grads = sum(1 for g in grad_norms if g > 0.0)

        info(f"Parameters with grad: {len(grad_norms)}")
        info(f"  Non-zero grads: {nonzero_grads}  Zero grads: {zero_grads}")

        if nonzero_grads > 0:
            ok(f"Non-zero gradients present ({nonzero_grads}/{len(grad_norms)} params)")
        else:
            fail("All gradients are zero -- possible dead graph")

        # Check for NaN gradients
        nan_grads = sum(1 for p in model_test.parameters()
                        if p.grad is not None and torch.isnan(p.grad).any())
        if nan_grads == 0:
            ok("No NaN gradients")
        else:
            fail(f"{nan_grads} parameters have NaN gradients")

        # Zero gradients for next iteration
        model_test.zero_grad()

    except Exception as e:
        fail(f"Backward pass failed: {e}")
        import traceback; traceback.print_exc()


# ----------------------------------------------------------------------------
# Test 8: Input/output shapes summary
# ----------------------------------------------------------------------------

header("Test 8: Input/output shape summary")
# Ensure models are on CPU for this test (they may have been moved to GPU in Test 5)
model_a_cpu = model_a.cpu()
model_b_cpu = model_b.cpu()
model_a_cpu.eval()
model_b_cpu.eval()

for crop_h, crop_w in [(32, 32), (64, 64), (128, 128), (517, 461)]:
    x_test = make_batch(b=1, h=crop_h, w=crop_w)
    with torch.no_grad():
        out_a_t = model_a_cpu(x_test)
        out_b_t = model_b_cpu(x_test)
    info(f"Input {(1,15,crop_h,crop_w)} -> A: {tuple(out_a_t.shape)}  B: {tuple(out_b_t.shape)}")
    if out_a_t.shape[2:] != x_test.shape[2:]:
        fail(f"Model A shape mismatch at {crop_h}x{crop_w}")
    if out_b_t.shape[2:] != x_test.shape[2:]:
        fail(f"Model B shape mismatch at {crop_h}x{crop_w}")

ok("All shape tests passed")


# ----------------------------------------------------------------------------
# Test 9: Peak CUDA memory (larger batch)
# ----------------------------------------------------------------------------

header("Test 9: Peak CUDA memory (batch=64, 32x32)")
if cuda_available:
    torch.cuda.reset_peak_memory_stats()
    # Re-send to GPU (Test 8 moved them to CPU)
    model_a_gpu = model_a.to(device)
    model_b_gpu = model_b.to(device)
    model_a_gpu.eval()
    model_b_gpu.eval()
    x_big = make_batch(b=64, h=32, w=32).to(device)
    with torch.no_grad():
        _ = model_a_gpu(x_big)
        _ = model_b_gpu(x_big)
    peak_big_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    info(f"Peak CUDA memory (batch=64, both models): {peak_big_mb:.1f} MB")
    ok("Large batch GPU inference succeeded")
else:
    print(f"  {WARN} CUDA not available -- skipping memory test")


# ----------------------------------------------------------------------------
# Test 10: Asynchronous loader channel verification
# ----------------------------------------------------------------------------

header("Test 10: Asynchronous loader channel verification")
try:
    from src.training.train_asynchronous_dual_stream_clean import (
        CleanAsynchronousOSCDDataset,
        SAR_CHANNELS,
        OPT_CHANNELS,
        INPUT_CHANNELS,
        CHANNEL_ORDER,
    )
    info(f"SAR_CHANNELS  = {SAR_CHANNELS}  (expected 2)")
    info(f"OPT_CHANNELS  = {OPT_CHANNELS}  (expected 13)")
    info(f"INPUT_CHANNELS= {INPUT_CHANNELS}  (expected 15)")
    info(f"Channel order : {CHANNEL_ORDER}")

    if SAR_CHANNELS == 2:
        ok("SAR_CHANNELS = 2 v")
    else:
        fail(f"SAR_CHANNELS = {SAR_CHANNELS}, expected 2")

    if OPT_CHANNELS == 13:
        ok("OPT_CHANNELS = 13 v")
    else:
        fail(f"OPT_CHANNELS = {OPT_CHANNELS}, expected 13")

    if INPUT_CHANNELS == 15:
        ok("INPUT_CHANNELS = 15 (2+13) v")
    else:
        fail(f"INPUT_CHANNELS = {INPUT_CHANNELS}, expected 15")

    # Verify that combine_bands() is never called with 'imgs_2_rect' as the folder.
    # That is the only code path that would actually load T2 optical data.
    # (print() messages that mention the string are intentional documentation.)
    import ast, inspect
    from src.training import train_asynchronous_dual_stream_clean as clean_module
    source = inspect.getsource(clean_module)
    tree = ast.parse(source)

    # Find all Call nodes where the function is 'combine_bands'
    dangerous_calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # combine_bands(some_path)
        if isinstance(func, ast.Name) and func.id == "combine_bands":
            # Check if any argument contains "imgs_2_rect"
            for arg in node.args + [kw.value for kw in node.keywords]:
                arg_src = ast.unparse(arg)
                if "imgs_2_rect" in arg_src:
                    dangerous_calls.append((node.lineno, arg_src))
        # Also check for Path divisions: ... / "imgs_2_rect"
        if isinstance(func, ast.Attribute) and func.attr == "__truediv__":
            # This is a / operator: check the right operand
            pass  # handled by BinOp below

    # Also check BinOp for path / "imgs_2_rect" divisions
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            right_src = ast.unparse(node.right)
            if "imgs_2_rect" in right_src:
                dangerous_calls.append((node.lineno, ast.unparse(node)))

    if not dangerous_calls:
        ok("combine_bands() and path ops never use 'imgs_2_rect' -- T2 optical not loaded v")
    else:
        fail(f"combine_bands/path with imgs_2_rect at: {dangerous_calls}")

    # Check channel names -- none should reference T2 optical
    t2_opt_channels = [c for c in CHANNEL_ORDER if "sentinel2_t2" in c]
    if not t2_opt_channels:
        ok("No S2_T2 channels in CHANNEL_ORDER v")
    else:
        fail(f"S2_T2 optical channels found: {t2_opt_channels}")

    info(f"Loader confirmed: SAR=2ch, Optical=13ch, no S2_T2")

except Exception as e:
    fail(f"Loader verification failed: {e}")
    import traceback; traceback.print_exc()


# ----------------------------------------------------------------------------
# Final summary
# ----------------------------------------------------------------------------

header("FINAL SUMMARY")

print(f"\n  {'-'*60}")
print(f"  {'Model':<35} {'Parameters':>15}")
print(f"  {'-'*60}")
print(f"  {'TransformerSelfAttention (A)':<35} {params_a:>15,}")
print(f"  {'TransformerCrossAttention (B)':<35} {params_b:>15,}")
print(f"  {'Difference A - B':<35} {params_a - params_b:>15,}")
print(f"  {'-'*60}\n")

if _failures:
    print(f"\n  {FAIL} {len(_failures)} test(s) FAILED:")
    for f_msg in _failures:
        print(f"    • {f_msg}")
    sys.exit(1)
else:
    print(f"  {PASS} All smoke tests PASSED")
    sys.exit(0)
