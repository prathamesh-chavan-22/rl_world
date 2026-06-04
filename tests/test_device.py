import pytest
import torch

from src.config import Config
from src.dqn import DQNAgent
from src.device import resolve_torch_precision, select_torch_device


def test_auto_prefers_cuda_over_mps(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    device = select_torch_device("auto")

    assert device.type == "cuda"


def test_auto_uses_mps_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    device = select_torch_device("auto")

    assert device.type == "mps"


def test_auto_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    device = select_torch_device("auto")

    assert device.type == "cpu"


def test_explicit_unavailable_cuda_raises(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA"):
        select_torch_device("cuda")


def test_explicit_unavailable_mps_raises(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="MPS"):
        select_torch_device("mps")


def test_dqn_agent_uses_configured_device():
    cfg = Config()
    cfg.device = "cpu"

    dqn = DQNAgent(cfg)

    assert dqn.device.type == "cpu"


def test_auto_precision_prefers_cuda_bf16_when_supported(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)

    precision = resolve_torch_precision("auto", torch.device("cuda"))

    assert precision.name == "bf16"
    assert precision.dtype is torch.bfloat16
    assert precision.use_grad_scaler is False


def test_auto_precision_uses_cuda_fp16_with_scaler_without_bf16(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)

    precision = resolve_torch_precision("auto", torch.device("cuda"))

    assert precision.name == "fp16"
    assert precision.dtype is torch.float16
    assert precision.use_grad_scaler is True


def test_auto_precision_uses_mps_fp16_without_scaler():
    precision = resolve_torch_precision("auto", torch.device("mps"))

    assert precision.name == "fp16"
    assert precision.dtype is torch.float16
    assert precision.use_grad_scaler is False


def test_auto_precision_uses_cpu_fp32():
    precision = resolve_torch_precision("auto", torch.device("cpu"))

    assert precision.name == "fp32"
    assert precision.dtype is None
    assert precision.use_grad_scaler is False


def test_explicit_cpu_fp16_raises():
    with pytest.raises(RuntimeError, match="FP16"):
        resolve_torch_precision("fp16", torch.device("cpu"))


def test_dqn_agent_uses_configured_precision():
    cfg = Config()
    cfg.device = "cpu"
    cfg.precision = "fp32"

    dqn = DQNAgent(cfg)

    assert dqn.precision.name == "fp32"
