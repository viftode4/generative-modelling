"""Lightweight sampler facade and registry for the project.

Keep established imports such as ``zigzag_project.zsampling`` working for
existing notebooks and scripts. New code may use this facade when it needs a
single sampler registry or method-specific config validation.
"""
from __future__ import annotations

from zigzag_project.samplers.registry import (
    SAMPLER_REGISTRY,
    SamplerSpec,
    get_sampler,
    validate_sampler_config,
)
from zigzag_project.samplers.resampling import (
    ResamplingConfig,
    get_resampling_schedule,
    get_timestep_from_logical,
    resampling_sdxl,
    validate_resampling_config,
)
from zigzag_project.samplers.standard import StandardSamplingConfig, standard_sdxl
from zigzag_project.samplers.zigzag import (
    ZSamplingConfig,
    make_initial_latents,
    retrieve_timesteps,
    validate_zsampling_config,
    z_sample_sd21,
    z_sample_sdxl,
)

__all__ = [
    "ResamplingConfig",
    "SAMPLER_REGISTRY",
    "SamplerSpec",
    "StandardSamplingConfig",
    "ZSamplingConfig",
    "get_resampling_schedule",
    "get_sampler",
    "get_timestep_from_logical",
    "make_initial_latents",
    "resampling_sdxl",
    "retrieve_timesteps",
    "standard_sdxl",
    "validate_resampling_config",
    "validate_sampler_config",
    "validate_zsampling_config",
    "z_sample_sd21",
    "z_sample_sdxl",
]
