from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

# Configuration from the Resampling paper referenced by the ZigZag Sampling paper.
RESAMPLING_CONFIG = {
    "jump_length": 10,
    "jump_n_sample": 10,
}


@dataclass(frozen=True)
class ResamplingConfig:
    """Hyperparameters for the RePaint-style resampling baseline."""

    gamma_1: float = 5.5
    infer_step: int = 50
    jump_length: int = 10
    jump_n_sample: int = 10

    def resolved_jump_params(self) -> tuple[int, int, bool]:
        """Return validated jump parameters, clamped for short schedules."""
        if self.infer_step < 2:
            raise ValueError("infer_step must be at least 2 for resampling")
        if self.jump_length < 1:
            raise ValueError("jump_length must be at least 1")
        if self.jump_n_sample < 1:
            raise ValueError("jump_n_sample must be at least 1")

        effective_jump_length = min(self.jump_length, self.infer_step - 1)
        was_clamped = effective_jump_length != self.jump_length
        return effective_jump_length, self.jump_n_sample, was_clamped


def get_resampling_config() -> dict[str, int]:
    """Return default RePaint-style jump settings."""
    return dict(RESAMPLING_CONFIG)


def get_resampling_schedule(
    num_inference_steps: int,
    jump_length: int | None = None,
    jump_n_sample: int | None = None,
) -> list[int]:
    """Build a non-monotonic RePaint-style logical timestep schedule.

    High logical values represent noisier states; decreasing values denoise and
    increasing values re-noise. ``-1`` represents the final clean state after
    the last denoising step.
    """
    config = ResamplingConfig(
        infer_step=num_inference_steps,
        jump_length=RESAMPLING_CONFIG["jump_length"] if jump_length is None else jump_length,
        jump_n_sample=(
            RESAMPLING_CONFIG["jump_n_sample"] if jump_n_sample is None else jump_n_sample
        ),
    )
    jump_length, jump_n_sample, _ = config.resolved_jump_params()

    # Use jump_n_sample - 1 because the first visit happens during normal denoising.
    jumps = {
        step: jump_n_sample - 1 for step in range(0, num_inference_steps - jump_length, jump_length)
    }
    logical_step = num_inference_steps
    schedule: list[int] = []

    while logical_step >= 1:
        logical_step -= 1
        schedule.append(logical_step)

        if jumps.get(logical_step, 0) > 0:
            jumps[logical_step] -= 1
            for _ in range(jump_length):
                logical_step += 1
                schedule.append(logical_step)

    schedule.append(-1)
    return schedule


def get_timestep_from_logical(
    timesteps: Sequence[Any],
    logical_step_index: int,
) -> Any | None:
    """Map a logical resampling step to a concrete diffusers timestep."""
    num_inference_steps = len(timesteps)
    index = (
        num_inference_steps
        if logical_step_index == -1
        else num_inference_steps - 1 - logical_step_index
    )
    if index >= len(timesteps):
        return None
    return timesteps[index]
