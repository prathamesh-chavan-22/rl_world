from __future__ import annotations

from dataclasses import dataclass

import torch


VALID_DEVICE_CHOICES = ("auto", "cuda", "mps", "cpu")
VALID_PRECISION_CHOICES = ("auto", "fp32", "fp16", "bf16")


@dataclass(frozen=True)
class PrecisionConfig:
    name: str
    dtype: torch.dtype | None
    use_grad_scaler: bool


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


def resolve_torch_precision(
    preferred: str | None,
    device: torch.device,
) -> PrecisionConfig:
    """Resolve requested precision for a concrete PyTorch device."""
    choice = (preferred or "auto").lower()
    if choice not in VALID_PRECISION_CHOICES:
        raise ValueError(
            f"Unknown precision '{preferred}'. Expected one of {', '.join(VALID_PRECISION_CHOICES)}."
        )

    if choice == "auto":
        if device.type == "cuda":
            if torch.cuda.is_bf16_supported():
                return PrecisionConfig("bf16", torch.bfloat16, use_grad_scaler=False)
            return PrecisionConfig("fp16", torch.float16, use_grad_scaler=True)
        if device.type == "mps":
            return PrecisionConfig("fp16", torch.float16, use_grad_scaler=False)
        return PrecisionConfig("fp32", None, use_grad_scaler=False)

    if choice == "fp32":
        return PrecisionConfig("fp32", None, use_grad_scaler=False)

    if choice == "fp16":
        if device.type == "cpu":
            raise RuntimeError("FP16 precision was requested, but CPU autocast is not supported here.")
        return PrecisionConfig("fp16", torch.float16, use_grad_scaler=device.type == "cuda")

    if choice == "bf16":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return PrecisionConfig("bf16", torch.bfloat16, use_grad_scaler=False)
        raise RuntimeError(
            f"BF16 precision was requested, but it is not supported on device '{device.type}'."
        )

    raise AssertionError("unreachable")
