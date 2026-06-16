from dataclasses import FrozenInstanceError, fields

import pytest
import torch
from diffusers import DDIMInverseScheduler, DDIMScheduler

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


def test_sd21_loader_function_exists():
    assert callable(models.load_sd21)


def test_sdxl_loader_function_exists():
    assert callable(models.load_sdxl)


class _FakePipe:
    """Minimal stand-in so load_sdxl can be tested without downloading weights."""

    def __init__(self, scheduler):
        self.scheduler = scheduler

    def to(self, _device):
        return self


def test_load_sdxl_pairs_real_inverse_scheduler(monkeypatch):
    """Regression guard: the SDXL handle must carry a true DDIMInverseScheduler.

    The earlier SDXL path (run_exp2's wrapper) silently used the forward DDIM
    scheduler as the inverse, which would corrupt Z-Sampling inversion. load_sdxl
    must instead derive a real DDIMInverseScheduler from the DDIM config.
    """
    fake = _FakePipe(scheduler=DDIMScheduler())
    monkeypatch.setattr(
        models.StableDiffusionXLPipeline,
        "from_pretrained",
        classmethod(lambda cls, *a, **k: fake),
    )

    handle = models.load_sdxl(device="cpu")

    assert handle.hf_id == "stabilityai/stable-diffusion-xl-base-1.0"
    assert isinstance(handle.pipe.scheduler, DDIMScheduler)
    assert isinstance(handle.inverse_scheduler, DDIMInverseScheduler)
    # The inverse scheduler must NOT be the same (forward) scheduler instance.
    assert handle.inverse_scheduler is not handle.pipe.scheduler


def test_sdxl_base_loader_function_exists():
    assert callable(models.load_sdxl_base)


def test_resolve_sdxl_dtype_uses_fp16_on_cuda_fp32_elsewhere():
    # SDXL's UNet produces NaN in fp16/bf16 on Apple MPS (verified empirically:
    # NaN at the first forward), so MPS/CPU must run fp32; CUDA keeps fast fp16.
    assert models.resolve_sdxl_dtype(torch.device("cuda")) == torch.float16
    assert models.resolve_sdxl_dtype(torch.device("mps")) == torch.float32
    assert models.resolve_sdxl_dtype(torch.device("cpu")) == torch.float32
