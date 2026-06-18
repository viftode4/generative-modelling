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


def load_sdxl(
    device: str | None = None,
    dtype: torch.dtype = torch.float16,
) -> ModelHandle:
    """Load Stable Diffusion XL base 1.0 with DDIM + DDIMInverse schedulers.

    This is the backbone the Z-Sampling paper uses for SDXL (50 steps, gamma_1=5.5)
    and the model the upstream ``infer.py`` hardcodes. DDIM is required for
    Z-Sampling because the inversion step needs a deterministic, invertible
    scheduler. We pair a DDIM forward scheduler with a DDIMInverse scheduler
    derived from the same config, exactly as ``load_dreamshaper_xl_turbo`` does,
    so the inverse step uses matching parameters rather than the default config.

    Used by Experiment 8's efficiency runner. See ``load_sdxl_base`` for the
    per-device-dtype variant (fp32 on MPS/CPU) that the later SDXL experiments use.
    """
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
    """Pick a numerically-safe dtype for SDXL-base per device.

    SDXL's UNet overflows to NaN on the *first* forward pass in fp16 AND bf16
    on Apple Silicon / MPS (verified empirically on torch 2.12 — clean inputs
    in, NaN out, both reduced-precision dtypes; fp32 is clean). CUDA runs fp16
    correctly (as in the paper), so keep the fast path there.
    """
    return torch.float16 if device.type == "cuda" else torch.float32


def load_sdxl_base(
    device: str | None = None,
    dtype: torch.dtype | None = None,
) -> ModelHandle:
    """Load Stable Diffusion XL base 1.0 with DDIM + DDIMInverse schedulers.

    This is the model used in the Z-Sampling paper's SDXL experiments
    (Tables 2/5, Figure 7). DDIM is required for Z-Sampling because the
    inversion step needs a deterministic, invertible scheduler; the inverse
    scheduler is derived from the DDIM config so forward/inverse share
    prediction_type, set_alpha_to_one, steps_offset, and timestep_spacing.

    ``dtype=None`` selects per device via ``resolve_sdxl_dtype`` (fp32 on
    MPS/CPU to avoid the fp16 NaN, fp16 on CUDA).
    """
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


def load_hunyuan_dit(
    device: str | None = None,
    dtype: torch.dtype = torch.float16,
) -> ModelHandle:
    """Load Hunyuan-DiT with DDIM + DDIMInverse schedulers.

    Hunyuan-DiT uses a different Diffusers pipeline from SD/SDXL, so this
    loader only establishes the standard generation pipeline and scheduler
    pair. Z-Sampling still needs a Hunyuan-specific sampling function before
    this handle can be used for zigzag generation.
    """

    hf_id = "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers"
    resolved_device = resolve_device(device)

    pipe = HunyuanDiTPipeline.from_pretrained(
        hf_id,
        torch_dtype=dtype,
    )
    # Keep the scheduler setup parallel to SDXL/SD2.1. This gives standard
    # DDIM sampling plus a matching inverse scheduler for a future Z-Sampling
    # port, but the Hunyuan pipeline plumbing still needs separate validation.
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
