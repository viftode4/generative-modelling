from dataclasses import FrozenInstanceError, fields

import pytest
import torch

from zigzag_project import models
from zigzag_project.models import ModelHandle, resolve_device


def test_resolve_device_respects_explicit_string():
    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_falls_back_to_cpu_with_no_accelerator(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert resolve_device() == torch.device("cpu")


def test_resolve_device_prefers_cuda_when_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_device() == torch.device("cuda")


def test_resolve_device_falls_back_to_mps_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_device() == torch.device("mps")


def test_model_handle_has_expected_fields():
    expected = {"pipe", "inverse_scheduler", "device", "dtype", "hf_id"}
    actual = {f.name for f in fields(ModelHandle)}
    assert expected == actual


def test_model_handle_is_frozen():
    handle = ModelHandle(
        pipe=object(),
        inverse_scheduler=object(),
        device=torch.device("cpu"),
        dtype=torch.float32,
        hf_id="test/model",
    )
    with pytest.raises(FrozenInstanceError):
        handle.hf_id = "other"  # type: ignore[misc]


def test_loader_function_exists():
    assert callable(models.load_dreamshaper_xl_turbo)
