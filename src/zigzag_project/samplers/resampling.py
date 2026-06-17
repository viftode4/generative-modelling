"""Compatibility facade for the merged SDXL resampling implementation."""
from __future__ import annotations

from typing import Any

from zigzag_project.resampling_utils import (
    ResamplingConfig,
    get_resampling_schedule,
    get_timestep_from_logical,
)
from zigzag_project.zsampling import resampling_sdxl as _resampling_sdxl


def validate_resampling_config(config: ResamplingConfig) -> None:
    """Raise ``ValueError`` when resampling loop bounds would be invalid."""
    config.resolved_jump_params()


def resampling_sdxl(
    handle: Any,
    prompt: str | list[str] | None = None,
    *,
    config: ResamplingConfig | None = None,
    **kwargs: Any,
) -> Any:
    """Run SDXL resampling with explicit config-type validation.

    The implementation lives in ``zigzag_project.zsampling`` for compatibility
    with teammate notebooks/scripts. This facade only centralizes sampler names
    and prevents cross-method config mixups.
    """
    config = config or ResamplingConfig()
    if not isinstance(config, ResamplingConfig):
        raise TypeError(
            "resampling_sdxl config must be ResamplingConfig; "
            f"got {type(config).__name__}"
        )
    return _resampling_sdxl(handle, prompt=prompt, config=config, **kwargs)


__all__ = [
    "ResamplingConfig",
    "get_resampling_schedule",
    "get_timestep_from_logical",
    "resampling_sdxl",
    "validate_resampling_config",
]
