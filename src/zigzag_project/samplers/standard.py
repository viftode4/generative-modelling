from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from zigzag_project.models import ModelHandle


@dataclass(frozen=True)
class StandardSamplingConfig:
    """Shared config for a plain diffusion pipeline call."""

    gamma_1: float = 5.5
    infer_step: int = 50


def standard_sdxl(
    handle: ModelHandle,
    prompt: str | list[str] | None = None,
    *,
    config: StandardSamplingConfig | None = None,
    **kwargs: Any,
) -> Any:
    """Run the underlying SDXL-compatible pipeline without zigzag/resampling steps."""
    config = config or StandardSamplingConfig()
    if not isinstance(config, StandardSamplingConfig):
        raise TypeError(
            "standard_sdxl config must be StandardSamplingConfig; "
            f"got {type(config).__name__}"
        )

    call_kwargs = dict(kwargs)
    call_kwargs.setdefault("num_inference_steps", config.infer_step)
    call_kwargs.setdefault("guidance_scale", config.gamma_1)
    return handle.pipe(prompt=prompt, **call_kwargs)
