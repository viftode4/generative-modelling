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
    # Replace the default scheduler with DDIM first so the inverse scheduler
    # is derived from the DDIM config (prediction_type, set_alpha_to_one,
    # steps_offset, timestep_spacing) rather than the original Euler config.
    # Upstream registers both schedulers from the same DDIM config.
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


def load_sdxl_base(
    device: str | None = None,
    dtype: torch.dtype = torch.float16,
) -> ModelHandle:
    """Load Stable Diffusion XL base 1.0 with DDIM + DDIMInverse schedulers.

    This is the model used in the Z-Sampling paper's SDXL experiments
    (Tables 2/5, Figure 7). DDIM is required for Z-Sampling because the
    inversion step needs a deterministic, invertible scheduler; the inverse
    scheduler is derived from the DDIM config so forward/inverse share
    prediction_type, set_alpha_to_one, steps_offset, and timestep_spacing.
    """
    hf_id = "stabilityai/stable-diffusion-xl-base-1.0"
    resolved_device = resolve_device(device)

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
    # See load_dreamshaper_xl_turbo: derive the inverse scheduler from DDIM
    # rather than the default PNDM config so forward/inverse share parameters.
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
