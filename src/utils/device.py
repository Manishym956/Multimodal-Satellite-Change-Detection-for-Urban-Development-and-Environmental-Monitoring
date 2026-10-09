from __future__ import annotations

import torch


def get_device(preference: str = "auto") -> torch.device:
    preference = preference.lower().strip()
    if preference not in ("auto", "cuda", "cpu"):
        raise ValueError(
            f"Invalid device preference '{preference}'. "
            "Choose one of: 'auto', 'cuda', 'cpu'."
        )

    if preference == "cpu":
        device = torch.device("cpu")
        _print_status(device, explicit_cpu=True)
        return device

    cuda_available = torch.cuda.is_available()

    if preference == "cuda" and not cuda_available:
        raise RuntimeError(
            "Device preference is 'cuda' but torch.cuda.is_available() is False. "
            "Check your PyTorch installation and CUDA drivers."
        )

    device = torch.device("cuda" if cuda_available else "cpu")
    _print_status(device, explicit_cpu=False)
    return device


def _print_status(device: torch.device, explicit_cpu: bool) -> None:
    """Print a one-line device banner plus GPU details when CUDA is selected."""
    if device.type == "cuda":
        gpu_name   = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(
            f"Execution device: GPU — {gpu_name} "
            f"({total_vram:.1f} GB VRAM)  |  "
            f"PyTorch {torch.__version__}  |  "
            f"CUDA {torch.version.cuda}",
            flush=True,
        )
    else:
        reason = "CPU explicitly requested" if explicit_cpu else "CUDA unavailable"
        print(
            f"Execution device: CPU — {reason}  |  PyTorch {torch.__version__}",
            flush=True,
        )


def add_device_argument(parser) -> None:
    """Add a --device argument to an argparse.ArgumentParser.

    Adds:
        --device {auto,cuda,cpu}   Execution device. Default: auto.
    """
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Execution device: auto (default) selects CUDA when available.",
    )
