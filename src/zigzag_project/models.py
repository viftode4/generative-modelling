from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from diffusers import (
    DDIMInverseScheduler,
    DDIMScheduler,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
)


@dataclass(frozen=True)
class ModelHandle:
    pipe: Any  # StableDiffusionXLPipeline | StableDiffusionPipeline
    inverse_scheduler: DDIMInverseScheduler
    device: torch.device
    dtype: torch.dtype
    hf_id: str


def resolve_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_dreamshaper_xl_turbo(
    device: str | None = None,
    dtype: torch.dtype = torch.float16,
) -> ModelHandle:
    """Load DreamShaper-xl-v2-turbo with DDIM + DDIMInverse schedulers.

    DDIM is required for Z-Sampling because the inversion step needs a
    deterministic, invertible scheduler. Upstream uses the same pair for SDXL.
    """
    hf_id = "Lykon/dreamshaper-xl-v2-turbo"
    resolved_device = resolve_device(device)

    pipe = StableDiffusionXLPipeline.from_pretrained(
        hf_id,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    inverse_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(resolved_device)

    return ModelHandle(
        pipe=pipe,
        inverse_scheduler=inverse_scheduler,
        device=resolved_device,
        dtype=dtype,
        hf_id=hf_id,
    )


def load_sd21(
    device: str | None = None,
    dtype: torch.dtype = torch.float16,
) -> ModelHandle:
    """Load Stable Diffusion 2.1 with DDIM + DDIMInverse schedulers.

    SD2.1 uses a single CLIP ViT-H/14 text encoder and produces 768×768
    images natively.  DDIM is required for Z-Sampling because the inversion
    step needs a deterministic, invertible scheduler.
    """
    hf_id = "sd2-community/stable-diffusion-2-1"
    resolved_device = resolve_device(device)

    pipe = StableDiffusionPipeline.from_pretrained(
        hf_id,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    inverse_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(resolved_device)

    return ModelHandle(
        pipe=pipe,
        inverse_scheduler=inverse_scheduler,
        device=resolved_device,
        dtype=dtype,
        hf_id=hf_id,
    )
