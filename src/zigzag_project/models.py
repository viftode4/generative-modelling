from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from diffusers import (
    DDIMInverseScheduler,
    DDIMScheduler,
    HunyuanDiTPipeline,
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
    """Load DreamShaper-xl-v2-turbo with DDIM + DDIMInverse schedulers."""
    hf_id = "Lykon/dreamshaper-xl-v2-turbo"
    resolved_device = resolve_device(device)

    pipe = StableDiffusionXLPipeline.from_pretrained(
        hf_id,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    inverse_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(resolved_device)

    return ModelHandle(
        pipe=pipe,
        inverse_scheduler=inverse_scheduler,
        device=resolved_device,
        dtype=dtype,
        hf_id=hf_id,
    )


def load_sdxl(
    device: str | None = None,
    dtype: torch.dtype = torch.float16,
) -> ModelHandle:
    """Load SDXL base 1.0 with DDIM + DDIMInverse schedulers."""
    hf_id = "stabilityai/stable-diffusion-xl-base-1.0"
    resolved_device = resolve_device(device)

    pipe = StableDiffusionXLPipeline.from_pretrained(
        hf_id,
        torch_dtype=dtype,
        use_safetensors=True,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    inverse_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(resolved_device)

    return ModelHandle(
        pipe=pipe,
        inverse_scheduler=inverse_scheduler,
        device=resolved_device,
        dtype=dtype,
        hf_id=hf_id,
    )


def resolve_sdxl_dtype(device: torch.device) -> torch.dtype:
    """fp16 on CUDA, fp32 on MPS/CPU (fp16/bf16 produces NaN on Apple Silicon)."""
    return torch.float16 if device.type == "cuda" else torch.float32


def load_sdxl_base(
    device: str | None = None,
    dtype: torch.dtype | None = None,
) -> ModelHandle:
    """Load SDXL base 1.0 with DDIM + DDIMInverse schedulers. dtype=None picks fp32 on MPS/CPU."""
    hf_id = "stabilityai/stable-diffusion-xl-base-1.0"
    resolved_device = resolve_device(device)
    if dtype is None:
        dtype = resolve_sdxl_dtype(resolved_device)

    pipe = StableDiffusionXLPipeline.from_pretrained(
        hf_id,
        torch_dtype=dtype,
        use_safetensors=True,
        add_watermarker=False,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    inverse_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(resolved_device)

    if resolved_device.type in ("cuda", "mps"):
        pipe.enable_attention_slicing()

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
    """Load Stable Diffusion 2.1 with DDIM + DDIMInverse schedulers."""
    hf_id = "sd2-community/stable-diffusion-2-1"
    resolved_device = resolve_device(device)

    pipe = StableDiffusionPipeline.from_pretrained(
        hf_id,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    inverse_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(resolved_device)

    return ModelHandle(
        pipe=pipe,
        inverse_scheduler=inverse_scheduler,
        device=resolved_device,
        dtype=dtype,
        hf_id=hf_id,
    )


def load_hunyuan_dit(
    device: str | None = None,
    dtype: torch.dtype = torch.float16,
) -> ModelHandle:
    """Load Hunyuan-DiT with DDIM + DDIMInverse schedulers."""
    hf_id = "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers"
    resolved_device = resolve_device(device)

    pipe = HunyuanDiTPipeline.from_pretrained(
        hf_id,
        torch_dtype=dtype,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    inverse_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(resolved_device)

    return ModelHandle(
        pipe=pipe,
        inverse_scheduler=inverse_scheduler,
        device=resolved_device,
        dtype=dtype,
        hf_id=hf_id,
    )


load_hunyan = load_hunyuan_dit
