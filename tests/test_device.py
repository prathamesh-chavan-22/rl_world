import pytest
import torch

from src.config import Config
from src.dqn import DQNAgent
from src.device import select_torch_device


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
