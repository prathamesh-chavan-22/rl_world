from __future__ import annotations

import torch


VALID_DEVICE_CHOICES = ("auto", "cuda", "mps", "cpu")


def select_torch_device(preferred: str | None = "auto") -> torch.device:
    """Select the best PyTorch device.

    Auto mode prefers CUDA for future NVIDIA runs, then Apple MPS for local
    Apple GPU acceleration, then CPU as a final fallback.
    """
    choice = (preferred or "auto").lower()
    if choice not in VALID_DEVICE_CHOICES:
        raise ValueError(
            f"Unknown device '{preferred}'. Expected one of {', '.join(VALID_DEVICE_CHOICES)}."
        )

    if choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
        return torch.device("cuda")

    if choice == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested, but torch.backends.mps.is_available() is False.")
        return torch.device("mps")

    return torch.device("cpu")
