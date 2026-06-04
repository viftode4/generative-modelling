from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    rescale_noise_cfg,
)
from diffusers.utils.torch_utils import randn_tensor


def retrieve_timesteps(scheduler, num_inference_steps=None, device=None, timesteps=None, sigmas=None, **kwargs):
    """Vendored from diffusers>=0.26; not present in the pinned diffusers==0.22.3."""
    if timesteps is not None:
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
    elif sigmas is not None:
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
    timesteps = scheduler.timesteps
    num_inference_steps = len(timesteps)
    return timesteps, num_inference_steps

from zigzag_project.models import ModelHandle
from zigzag_project.resampling_utils import get_resampling_schedule, get_timestep_from_logical, get_resampling_config, \
    ResamplingConfig


@dataclass(frozen=True)
class ZSamplingConfig:
    """Hyperparameters for Zigzag Diffusion Sampling.

    Names match the upstream implementation at commit
    ``513eb51cef223e7a562b5eb9b31d5ac5fa6b70f4``:
    ``gamma_1`` guides denoising, ``gamma_2`` guides inversion,
    ``infer_step`` is the total DDIM step count, ``lambda_step`` controls how
    many early denoising steps get zigzag refinement, and ``t_max`` is the
    number of zag/zig rounds per refined step.
    """

    gamma_1: float = 5.5
    gamma_2: float = 0.0
    infer_step: int = 50
    lambda_step: int | None = None
    t_max: int = 1

    def resolved_lambda_step(self) -> int:
        validate_zsampling_config(self)
        if self.lambda_step is None:
            return self.infer_step - 1
        return self.lambda_step


def validate_zsampling_config(config: ZSamplingConfig) -> None:
    """Raise ``ValueError`` when Z-Sampling loop bounds would be invalid."""
    if config.infer_step < 2:
        raise ValueError("infer_step must be at least 2 so inversion can use the next timestep")
    if config.t_max < 0:
        raise ValueError("t_max must be non-negative")
    if config.lambda_step is not None and not 0 <= config.lambda_step < config.infer_step:
        raise ValueError("lambda_step must satisfy 0 <= lambda_step < infer_step")


def make_initial_latents(
    *,
    seed: int,
    height: int,
    width: int,
    dtype: torch.dtype,
    device: torch.device,
    batch_size: int = 1,
    channels: int = 4,
    vae_scale_factor: int = 8,
) -> torch.Tensor:
    """Create deterministic SD/SDXL latent noise for standard-vs-Z comparisons."""
    if height % vae_scale_factor != 0 or width % vae_scale_factor != 0:
        raise ValueError("height and width must be divisible by vae_scale_factor")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    latents = torch.randn(
        (batch_size, channels, height // vae_scale_factor, width // vae_scale_factor),
        generator=generator,
        dtype=dtype,
        device="cpu",
    )
    return latents.to(device)


def _guided_noise_prediction(
    *,
    pipe: Any,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    prompt_embeds: torch.Tensor,
    add_text_embeds: torch.Tensor,
    add_time_ids: torch.Tensor,
    guidance_scale: float,
    guidance_rescale: float,
    timestep_cond: torch.Tensor | None,
) -> torch.Tensor:
    latent_model_input = torch.cat([latents] * 2) if pipe.do_classifier_free_guidance else latents
    latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, timestep)
    noise_pred = pipe.unet(
        latent_model_input,
        timestep,
        encoder_hidden_states=prompt_embeds,
        timestep_cond=timestep_cond,
        cross_attention_kwargs=pipe.cross_attention_kwargs,
        added_cond_kwargs={"text_embeds": add_text_embeds, "time_ids": add_time_ids},
        return_dict=False,
    )[0]

    if pipe.do_classifier_free_guidance:
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        if guidance_rescale > 0.0:
            noise_pred = rescale_noise_cfg(
                noise_pred,
                noise_pred_text,
                guidance_rescale=guidance_rescale,
            )
    return noise_pred


def _decode_latents(pipe: Any, latents: torch.Tensor, output_type: str | None) -> Any:
    if output_type == "latent":
        return latents

    needs_upcasting = pipe.vae.dtype == torch.float16 and pipe.vae.config.force_upcast
    if needs_upcasting:
        pipe.upcast_vae()
        latents = latents.to(next(iter(pipe.vae.post_quant_conv.parameters())).dtype)
    elif latents.dtype != pipe.vae.dtype and torch.backends.mps.is_available():
        # Mirrors diffusers' SDXL MPS workaround for dtype drift during scheduler steps.
        pipe.vae = pipe.vae.to(latents.dtype)

    has_latents_mean = (
        hasattr(pipe.vae.config, "latents_mean")
        and pipe.vae.config.latents_mean is not None
    )
    has_latents_std = (
        hasattr(pipe.vae.config, "latents_std")
        and pipe.vae.config.latents_std is not None
    )
    if has_latents_mean and has_latents_std:
        latents_mean = (
            torch.tensor(pipe.vae.config.latents_mean)
            .view(1, 4, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = (
            torch.tensor(pipe.vae.config.latents_std)
            .view(1, 4, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents = latents * latents_std / pipe.vae.config.scaling_factor + latents_mean
    else:
        latents = latents / pipe.vae.config.scaling_factor

    image = pipe.vae.decode(latents, return_dict=False)[0]
    if needs_upcasting:
        pipe.vae.to(dtype=torch.float16)
    if pipe.watermark is not None:
        image = pipe.watermark.apply_watermark(image)
    return pipe.image_processor.postprocess(image, output_type=output_type)


@torch.no_grad()
def z_sample_sdxl(
    handle: ModelHandle,
    prompt: str | list[str] | None = None,
    *,
    config: ZSamplingConfig | None = None,
    prompt_2: str | list[str] | None = None,
    height: int | None = None,
    width: int | None = None,
    negative_prompt: str | list[str] | None = None,
    negative_prompt_2: str | list[str] | None = None,
    num_images_per_prompt: int = 1,
    eta: float = 0.0,
    generator: torch.Generator | list[torch.Generator] | None = None,
    latents: torch.Tensor | None = None,
    prompt_embeds: torch.Tensor | None = None,
    negative_prompt_embeds: torch.Tensor | None = None,
    pooled_prompt_embeds: torch.Tensor | None = None,
    negative_pooled_prompt_embeds: torch.Tensor | None = None,
    output_type: str | None = "pil",
    return_dict: bool = True,
    cross_attention_kwargs: dict[str, Any] | None = None,
    guidance_rescale: float = 0.0,
    original_size: tuple[int, int] | None = None,
    crops_coords_top_left: tuple[int, int] = (0, 0),
    target_size: tuple[int, int] | None = None,
    negative_original_size: tuple[int, int] | None = None,
    negative_crops_coords_top_left: tuple[int, int] = (0, 0),
    negative_target_size: tuple[int, int] | None = None,
    clip_skip: int | None = None,
    callback_on_step_end: (
        Callable[[Any, int, torch.Tensor, dict[str, torch.Tensor]], dict[str, torch.Tensor]]
        | None
    ) = None,
    callback_on_step_end_tensor_inputs: tuple[str, ...] = ("latents",),
) -> StableDiffusionXLPipelineOutput | tuple[Any]:
    """Run SDXL Z-Sampling using a regular diffusers pipeline.

    This is a small, repo-local adaptation of upstream
    ``StableDiffusionXLPipeline.z_sampling_call`` at commit
    ``513eb51cef223e7a562b5eb9b31d5ac5fa6b70f4``. It intentionally uses the
    installed diffusers pipeline's private helpers instead of modifying or
    vendor-copying the upstream submodule.
    """
    config = config or ZSamplingConfig()
    validate_zsampling_config(config)
    lambda_step = config.resolved_lambda_step()
    pipe = handle.pipe
    inverse_scheduler = handle.inverse_scheduler

    height = height or pipe.default_sample_size * pipe.vae_scale_factor
    width = width or pipe.default_sample_size * pipe.vae_scale_factor
    original_size = original_size or (height, width)
    target_size = target_size or (height, width)

    pipe.check_inputs(
        prompt,
        prompt_2,
        height,
        width,
        None,
        negative_prompt,
        negative_prompt_2,
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
        None,
        None,
        list(callback_on_step_end_tensor_inputs),
    )

    pipe._guidance_scale = config.gamma_1
    pipe._guidance_rescale = guidance_rescale
    pipe._clip_skip = clip_skip
    pipe._cross_attention_kwargs = cross_attention_kwargs
    pipe._denoising_end = None
    pipe._interrupt = False

    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = pipe._execution_device
    lora_scale = (
        pipe.cross_attention_kwargs.get("scale", None)
        if pipe.cross_attention_kwargs is not None
        else None
    )
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        lora_scale=lora_scale,
        clip_skip=pipe.clip_skip,
    )

    timesteps, num_inference_steps = retrieve_timesteps(
        pipe.scheduler,
        config.infer_step,
        device,
        None,
        None,
    )
    inverse_scheduler.set_timesteps(num_inference_steps, device=device)

    latents = pipe.prepare_latents(
        batch_size * num_images_per_prompt,
        pipe.unet.config.in_channels,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, eta)

    add_text_embeds = pooled_prompt_embeds
    if pipe.text_encoder_2 is None:
        text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
    else:
        text_encoder_projection_dim = pipe.text_encoder_2.config.projection_dim

    add_time_ids = pipe._get_add_time_ids(
        original_size,
        crops_coords_top_left,
        target_size,
        dtype=prompt_embeds.dtype,
        text_encoder_projection_dim=text_encoder_projection_dim,
    )
    if negative_original_size is not None and negative_target_size is not None:
        negative_add_time_ids = pipe._get_add_time_ids(
            negative_original_size,
            negative_crops_coords_top_left,
            negative_target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
    else:
        negative_add_time_ids = add_time_ids

    if pipe.do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
        add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

    prompt_embeds = prompt_embeds.to(device)
    add_text_embeds = add_text_embeds.to(device)
    add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

    timestep_cond = None
    if pipe.unet.config.time_cond_proj_dim is not None:
        guidance_scale_tensor = torch.tensor(pipe.guidance_scale - 1).repeat(
            batch_size * num_images_per_prompt
        )
        timestep_cond = pipe.get_guidance_scale_embedding(
            guidance_scale_tensor,
            embedding_dim=pipe.unet.config.time_cond_proj_dim,
        ).to(device=device, dtype=latents.dtype)

    pipe._num_timesteps = len(timesteps)
    num_warmup_steps = max(len(timesteps) - num_inference_steps * pipe.scheduler.order, 0)

    with pipe.progress_bar(total=num_inference_steps) as progress_bar:
        for i, timestep in enumerate(timesteps):
            noise_pred = _guided_noise_prediction(
                pipe=pipe,
                latents=latents,
                timestep=timestep,
                prompt_embeds=prompt_embeds,
                add_text_embeds=add_text_embeds,
                add_time_ids=add_time_ids,
                guidance_scale=config.gamma_1,
                guidance_rescale=guidance_rescale,
                timestep_cond=timestep_cond,
            )
            next_latents = pipe.scheduler.step(
                noise_pred,
                timestep,
                latents,
                **extra_step_kwargs,
                return_dict=False,
            )[0]
            if next_latents.dtype != latents.dtype and torch.backends.mps.is_available():
                next_latents = next_latents.to(latents.dtype)

            if i < lambda_step:
                for _ in range(config.t_max):
                    inverse_noise_pred = _guided_noise_prediction(
                        pipe=pipe,
                        latents=next_latents,
                        timestep=timestep,
                        prompt_embeds=prompt_embeds,
                        add_text_embeds=add_text_embeds,
                        add_time_ids=add_time_ids,
                        guidance_scale=config.gamma_2,
                        guidance_rescale=guidance_rescale,
                        timestep_cond=timestep_cond,
                    )
                    inv_latents = inverse_scheduler.step(
                        inverse_noise_pred,
                        timesteps[i + 1],
                        next_latents,
                        return_dict=False,
                    )[0]

                    forward_noise_pred = _guided_noise_prediction(
                        pipe=pipe,
                        latents=inv_latents,
                        timestep=timestep,
                        prompt_embeds=prompt_embeds,
                        add_text_embeds=add_text_embeds,
                        add_time_ids=add_time_ids,
                        guidance_scale=config.gamma_1,
                        guidance_rescale=guidance_rescale,
                        timestep_cond=timestep_cond,
                    )
                    next_latents = pipe.scheduler.step(
                        forward_noise_pred,
                        timestep,
                        inv_latents,
                        **extra_step_kwargs,
                        return_dict=False,
                    )[0]

            latents = next_latents
            if callback_on_step_end is not None:
                callback_kwargs = {key: locals()[key] for key in callback_on_step_end_tensor_inputs}
                callback_outputs = callback_on_step_end(pipe, i, timestep, callback_kwargs)
                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                add_text_embeds = callback_outputs.pop("add_text_embeds", add_text_embeds)
                add_time_ids = callback_outputs.pop("add_time_ids", add_time_ids)

            should_update_progress = (
                i == len(timesteps) - 1
                or ((i + 1) > num_warmup_steps and (i + 1) % pipe.scheduler.order == 0)
            )
            if should_update_progress:
                progress_bar.update()

    image = _decode_latents(pipe, latents, output_type)
    pipe.maybe_free_model_hooks()
    if not return_dict:
        return (image,)
    return StableDiffusionXLPipelineOutput(images=image)

def _guided_noise_prediction_sd21(
    *,
    pipe: Any,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    prompt_embeds: torch.Tensor,
    guidance_scale: float,
    guidance_rescale: float,
    timestep_cond: torch.Tensor | None,
) -> torch.Tensor:
    """U-Net noise prediction for SD2.1 (no add_text_embeds / add_time_ids)."""
    latent_model_input = torch.cat([latents] * 2) if pipe.do_classifier_free_guidance else latents
    latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, timestep)
    noise_pred = pipe.unet(
        latent_model_input,
        timestep,
        encoder_hidden_states=prompt_embeds,
        timestep_cond=timestep_cond,
        cross_attention_kwargs=pipe.cross_attention_kwargs,
        return_dict=False,
    )[0]

    if pipe.do_classifier_free_guidance:
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        if guidance_rescale > 0.0:
            noise_pred = rescale_noise_cfg(
                noise_pred,
                noise_pred_text,
                guidance_rescale=guidance_rescale,
            )
    return noise_pred


def _decode_latents_sd21(pipe: Any, latents: torch.Tensor, output_type: str | None) -> Any:
    """Decode latents to images for SD2.1 (simpler VAE path than SDXL)."""
    if output_type == "latent":
        return latents

    latents = latents / pipe.vae.config.scaling_factor
    image = pipe.vae.decode(latents, return_dict=False)[0]
    return pipe.image_processor.postprocess(image, output_type=output_type)


@torch.no_grad()
def z_sample_sd21(
    handle: ModelHandle,
    prompt: str | list[str] | None = None,
    *,
    config: ZSamplingConfig | None = None,
    height: int | None = None,
    width: int | None = None,
    negative_prompt: str | list[str] | None = None,
    num_images_per_prompt: int = 1,
    eta: float = 0.0,
    generator: torch.Generator | list[torch.Generator] | None = None,
    latents: torch.Tensor | None = None,
    prompt_embeds: torch.Tensor | None = None,
    negative_prompt_embeds: torch.Tensor | None = None,
    output_type: str | None = "pil",
    return_dict: bool = True,
    cross_attention_kwargs: dict[str, Any] | None = None,
    guidance_rescale: float = 0.0,
    clip_skip: int | None = None,
    callback_on_step_end: (
        Callable[[Any, int, torch.Tensor, dict[str, torch.Tensor]], dict[str, torch.Tensor]]
        | None
    ) = None,
    callback_on_step_end_tensor_inputs: tuple[str, ...] = ("latents",),
) -> StableDiffusionPipelineOutput | tuple[Any]:
    """Run SD2.1 Z-Sampling using a regular diffusers StableDiffusionPipeline.

    Adapted from ``z_sample_sdxl`` but stripped of all SDXL-specific features
    (dual text encoders, add_text_embeds, add_time_ids).  The core zigzag loop
    — denoise, invert, forward-denoise — is identical.

    Provenance: upstream ``utils/pipeline_stable_diffusion_xl.py::z_sampling_call``
    at commit ``513eb51``, translated to SD2.1 pipeline internals.
    """
    config = config or ZSamplingConfig(gamma_1=7.5)
    validate_zsampling_config(config)
    lambda_step = config.resolved_lambda_step()
    pipe = handle.pipe
    inverse_scheduler = handle.inverse_scheduler

    # SD2.1 defaults to 768 if pipe.unet.config.sample_size is 96 (96*8=768).
    default_size = pipe.unet.config.sample_size * pipe.vae_scale_factor
    height = height or default_size
    width = width or default_size

    pipe._guidance_scale = config.gamma_1
    pipe._guidance_rescale = guidance_rescale
    pipe._clip_skip = clip_skip
    pipe._cross_attention_kwargs = cross_attention_kwargs
    pipe._interrupt = False

    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = pipe._execution_device
    lora_scale = (
        pipe.cross_attention_kwargs.get("scale", None)
        if pipe.cross_attention_kwargs is not None
        else None
    )

    # SD2.1 encode_prompt returns (prompt_embeds, negative_prompt_embeds).
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=prompt,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        negative_prompt=negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        lora_scale=lora_scale,
        clip_skip=pipe.clip_skip,
    )

    timesteps, num_inference_steps = retrieve_timesteps(
        pipe.scheduler,
        config.infer_step,
        device,
        None,
        None,
    )
    inverse_scheduler.set_timesteps(num_inference_steps, device=device)

    latents = pipe.prepare_latents(
        batch_size * num_images_per_prompt,
        pipe.unet.config.in_channels,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, eta)

    if pipe.do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

    prompt_embeds = prompt_embeds.to(device)

    timestep_cond = None
    if pipe.unet.config.time_cond_proj_dim is not None:
        guidance_scale_tensor = torch.tensor(pipe.guidance_scale - 1).repeat(
            batch_size * num_images_per_prompt
        )
        timestep_cond = pipe.get_guidance_scale_embedding(
            guidance_scale_tensor,
            embedding_dim=pipe.unet.config.time_cond_proj_dim,
        ).to(device=device, dtype=latents.dtype)

    pipe._num_timesteps = len(timesteps)
    num_warmup_steps = max(len(timesteps) - num_inference_steps * pipe.scheduler.order, 0)

    with pipe.progress_bar(total=num_inference_steps) as progress_bar:
        for i, timestep in enumerate(timesteps):
            noise_pred = _guided_noise_prediction_sd21(
                pipe=pipe,
                latents=latents,
                timestep=timestep,
                prompt_embeds=prompt_embeds,
                guidance_scale=config.gamma_1,
                guidance_rescale=guidance_rescale,
                timestep_cond=timestep_cond,
            )
            next_latents = pipe.scheduler.step(
                noise_pred,
                timestep,
                latents,
                **extra_step_kwargs,
                return_dict=False,
            )[0]
            if next_latents.dtype != latents.dtype and torch.backends.mps.is_available():
                next_latents = next_latents.to(latents.dtype)

            if i < lambda_step:
                for _ in range(config.t_max):
                    # Inversion step: weak guidance (gamma_2) backward.
                    inverse_noise_pred = _guided_noise_prediction_sd21(
                        pipe=pipe,
                        latents=next_latents,
                        timestep=timestep,
                        prompt_embeds=prompt_embeds,
                        guidance_scale=config.gamma_2,
                        guidance_rescale=guidance_rescale,
                        timestep_cond=timestep_cond,
                    )
                    inv_latents = inverse_scheduler.step(
                        inverse_noise_pred,
                        timesteps[i + 1],
                        next_latents,
                        return_dict=False,
                    )[0]

                    # Forward denoising step: strong guidance (gamma_1).
                    forward_noise_pred = _guided_noise_prediction_sd21(
                        pipe=pipe,
                        latents=inv_latents,
                        timestep=timestep,
                        prompt_embeds=prompt_embeds,
                        guidance_scale=config.gamma_1,
                        guidance_rescale=guidance_rescale,
                        timestep_cond=timestep_cond,
                    )
                    next_latents = pipe.scheduler.step(
                        forward_noise_pred,
                        timestep,
                        inv_latents,
                        **extra_step_kwargs,
                        return_dict=False,
                    )[0]

            latents = next_latents
            if callback_on_step_end is not None:
                callback_kwargs = {key: locals()[key] for key in callback_on_step_end_tensor_inputs}
                callback_outputs = callback_on_step_end(pipe, i, timestep, callback_kwargs)
                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

            should_update_progress = (
                i == len(timesteps) - 1
                or ((i + 1) > num_warmup_steps and (i + 1) % pipe.scheduler.order == 0)
            )
            if should_update_progress:
                progress_bar.update()

    image = _decode_latents_sd21(pipe, latents, output_type)
    pipe.maybe_free_model_hooks()
    if not return_dict:
        return (image,)
    return StableDiffusionPipelineOutput(images=image)

logger = logging.getLogger(__name__)

def _renoise_latents(
    scheduler: Any,
    latents: torch.Tensor,
    from_timestep: torch.Tensor,
    to_timestep: torch.Tensor,
    generator: torch.Generator | list[torch.Generator] | None = None,
) -> torch.Tensor:
    """Forward diffusion from a less-noisy to a more-noisy timestep.

    Uses the DDIM-compatible marginal:
        x_to = sqrt(α_to / α_from) * x_from + sqrt(1 − α_to / α_from) * ε

    Uses randn_tensor (not torch.randn) so a single generator or a per-image
    list of generators is handled reproducibly. Keep in sync with
    resampling_utils.renoise_between_timesteps, which uses torch.randn and
    only supports single generators.
    """
    if not hasattr(scheduler, "alphas_cumprod"):
        raise NotImplementedError(
            "_renoise_latents requires scheduler.alphas_cumprod (DDIM/DDPM-style scheduler)."
        )
    t_from = int(from_timestep.item())
    t_to = int(to_timestep.item())
    alpha_from = scheduler.alphas_cumprod[t_from].to(device=latents.device, dtype=latents.dtype)
    alpha_to = scheduler.alphas_cumprod[t_to].to(device=latents.device, dtype=latents.dtype)
    if alpha_to > alpha_from:
        raise ValueError(
            f"Re-noising must move to a noisier timestep (alpha_to={alpha_to:.4f} > alpha_from={alpha_from:.4f})."
        )
    alpha_ratio = torch.clamp(alpha_to / alpha_from, min=0.0, max=1.0)
    noise = randn_tensor(latents.shape, generator=generator, device=latents.device, dtype=latents.dtype)
    return alpha_ratio.sqrt() * latents + (1.0 - alpha_ratio).sqrt() * noise


@torch.no_grad()
def resampling_sdxl(
    handle: ModelHandle,
    prompt: str | list[str] | None = None,
    *,
    config: ResamplingConfig | None = None,
    prompt_2: str | list[str] | None = None,
    height: int | None = None,
    width: int | None = None,
    negative_prompt: str | list[str] | None = None,
    negative_prompt_2: str | list[str] | None = None,
    num_images_per_prompt: int = 1,
    eta: float = 0.0,
    generator: torch.Generator | list[torch.Generator] | None = None,
    latents: torch.Tensor | None = None,
    prompt_embeds: torch.Tensor | None = None,
    negative_prompt_embeds: torch.Tensor | None = None,
    pooled_prompt_embeds: torch.Tensor | None = None,
    negative_pooled_prompt_embeds: torch.Tensor | None = None,
    output_type: str | None = "pil",
    return_dict: bool = True,
    cross_attention_kwargs: dict[str, Any] | None = None,
    guidance_rescale: float = 0.0,
    original_size: tuple[int, int] | None = None,
    crops_coords_top_left: tuple[int, int] = (0, 0),
    target_size: tuple[int, int] | None = None,
    negative_original_size: tuple[int, int] | None = None,
    negative_crops_coords_top_left: tuple[int, int] = (0, 0),
    negative_target_size: tuple[int, int] | None = None,
    clip_skip: int | None = None,
    callback_on_step_end: (
        Callable[[Any, int, torch.Tensor, dict[str, torch.Tensor]], dict[str, torch.Tensor]]
        | None
    ) = None,
    callback_on_step_end_tensor_inputs: tuple[str, ...] = ("latents",),
) -> StableDiffusionXLPipelineOutput | tuple[Any]:
    """Run SDXL Resampling (Lugmayr et al., RePaint, CVPR 2022) baseline.

    Uses a non-monotonic RePaint schedule: denoise jump_length steps, jump back
    jump_length steps, repeat jump_n_sample times. Re-noising uses the
    DDIM-compatible forward marginal q(x_t | x_s) on the current latent (no x0 anchor).
    jump_length and jump_n_sample default to the RePaint-style config values (10, 10).
    jump_length is clamped to num_inference_steps - 1 when necessary.
    """
    config = config or ResamplingConfig()
    pipe = handle.pipe

    if eta > 0.0:
        logger.warning(
            "resampling_sdxl: eta=%.3f > 0 makes the DDIM step stochastic; "
            "the deterministic re-noise formula is only an approximation in this regime. "
            "Use eta=0.0 for a paper-faithful baseline.",
            eta,
        )

    # 1. Input validation & pipeline state setup
    height = height or pipe.default_sample_size * pipe.vae_scale_factor
    width = width or pipe.default_sample_size * pipe.vae_scale_factor
    original_size = original_size or (height, width)
    target_size = target_size or (height, width)

    pipe.check_inputs(
        prompt, prompt_2, height, width, None, negative_prompt, negative_prompt_2,
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds,
        negative_pooled_prompt_embeds, None, None, list(callback_on_step_end_tensor_inputs),
    )

    pipe._guidance_scale = config.gamma_1
    pipe._guidance_rescale = guidance_rescale
    pipe._clip_skip = clip_skip
    pipe._cross_attention_kwargs = cross_attention_kwargs
    pipe._denoising_end = None
    pipe._interrupt = False

    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    # 2. Encode prompt
    device = pipe._execution_device
    lora_scale = pipe.cross_attention_kwargs.get("scale", None) if pipe.cross_attention_kwargs is not None else None

    (
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=prompt, prompt_2=prompt_2, device=device, num_images_per_prompt=num_images_per_prompt,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance, negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds, negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        lora_scale=lora_scale, clip_skip=pipe.clip_skip,
    )

    # 3. Timesteps & initial latents
    timesteps, num_inference_steps = retrieve_timesteps(pipe.scheduler, config.infer_step, device, None, None)

    latents = pipe.prepare_latents(
        batch_size * num_images_per_prompt, pipe.unet.config.in_channels,
        height, width, prompt_embeds.dtype, device, generator, latents,
    )
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, eta)

    # 4. SDXL added conditioning (time IDs, pooled embeds)
    add_text_embeds = pooled_prompt_embeds
    if pipe.text_encoder_2 is None:
        text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
    else:
        text_encoder_projection_dim = pipe.text_encoder_2.config.projection_dim

    add_time_ids = pipe._get_add_time_ids(original_size, crops_coords_top_left, target_size, dtype=prompt_embeds.dtype, text_encoder_projection_dim=text_encoder_projection_dim)
    if negative_original_size is not None and negative_target_size is not None:
        negative_add_time_ids = pipe._get_add_time_ids(negative_original_size, negative_crops_coords_top_left, negative_target_size, dtype=prompt_embeds.dtype, text_encoder_projection_dim=text_encoder_projection_dim)
    else:
        negative_add_time_ids = add_time_ids

    if pipe.do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
        add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

    prompt_embeds = prompt_embeds.to(device)
    add_text_embeds = add_text_embeds.to(device)
    add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

    timestep_cond = None
    if pipe.unet.config.time_cond_proj_dim is not None:
        guidance_scale_tensor = torch.tensor(pipe.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
        timestep_cond = pipe.get_guidance_scale_embedding(guidance_scale_tensor, embedding_dim=pipe.unet.config.time_cond_proj_dim).to(device=device, dtype=latents.dtype)

    # 5. Build the RePaint non-monotonic schedule
    effective_jump_length, effective_jump_n_sample, jump_was_clamped = config.resolved_jump_params()

    if jump_was_clamped:
        logger.warning("Clamped jump_length from %d to %d because infer_step=%d.",
            config.jump_length, effective_jump_length, config.infer_step,)

    schedule = get_resampling_schedule(num_inference_steps, jump_length=effective_jump_length, jump_n_sample=effective_jump_n_sample)
    transitions = list(zip(schedule[:-1], schedule[1:]))
    pipe._num_timesteps = len(transitions)

    # 6. Sampling loop
    with pipe.progress_bar(total=len(transitions)) as progress_bar:
        for t_last, t_cur in transitions:
            current_timestep = get_timestep_from_logical(timesteps, t_last)

            if t_cur < t_last:
                # Denoising step (schedule moving towards cleaner image)
                noise_pred = _guided_noise_prediction(
                    pipe=pipe, latents=latents, timestep=current_timestep, prompt_embeds=prompt_embeds,
                    add_text_embeds=add_text_embeds, add_time_ids=add_time_ids, guidance_scale=config.gamma_1,
                    guidance_rescale=guidance_rescale, timestep_cond=timestep_cond,
                )
                latents = pipe.scheduler.step(noise_pred, current_timestep, latents, **extra_step_kwargs, return_dict=False)[0]
                if latents.dtype != prompt_embeds.dtype and torch.backends.mps.is_available():
                    latents = latents.to(prompt_embeds.dtype)
            else:
                # Re-noising step: move directly between noisy states via α-ratio formula
                target_timestep = get_timestep_from_logical(timesteps, t_cur)
                latents = _renoise_latents(
                    scheduler=pipe.scheduler, latents=latents,
                    from_timestep=current_timestep, to_timestep=target_timestep,
                    generator=generator,
                )

            if callback_on_step_end is not None:
                callback_kwargs = {
                    key: locals()[key]
                    for key in callback_on_step_end_tensor_inputs
                    if key in locals()
                }
                callback_outputs = callback_on_step_end(pipe, t_last, current_timestep, callback_kwargs)
                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                add_text_embeds = callback_outputs.pop("add_text_embeds", add_text_embeds)
                add_time_ids = callback_outputs.pop("add_time_ids", add_time_ids)

            progress_bar.update()

    # 7. Decode & return
    image = _decode_latents(pipe, latents, output_type)
    pipe.maybe_free_model_hooks()
    if not return_dict:
        return (image,)
    return StableDiffusionXLPipelineOutput(images=image)

