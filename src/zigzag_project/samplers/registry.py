from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from zigzag_project.samplers.resampling import ResamplingConfig, resampling_sdxl
from zigzag_project.samplers.standard import StandardSamplingConfig, standard_sdxl
from zigzag_project.samplers.zigzag import ZSamplingConfig, z_sample_sdxl


@dataclass(frozen=True)
class SamplerSpec:
    name: str
    config_type: type
    function: Callable[..., Any]


SAMPLER_REGISTRY: dict[str, SamplerSpec] = {
    "standard": SamplerSpec(
        name="standard",
        config_type=StandardSamplingConfig,
        function=standard_sdxl,
    ),
    "zsampling": SamplerSpec(
        name="zsampling",
        config_type=ZSamplingConfig,
        function=z_sample_sdxl,
    ),
    "resampling": SamplerSpec(
        name="resampling",
        config_type=ResamplingConfig,
        function=resampling_sdxl,
    ),
}


def get_sampler(name: str) -> SamplerSpec:
    try:
        return SAMPLER_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(SAMPLER_REGISTRY))
        raise KeyError(f"Unknown sampler {name!r}; available: {available}") from exc


def validate_sampler_config(name: str, config: object) -> None:
    spec = get_sampler(name)
    if not isinstance(config, spec.config_type):
        raise TypeError(
            f"{name} sampler requires {spec.config_type.__name__}; "
            f"got {type(config).__name__}"
        )
