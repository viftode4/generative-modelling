import pytest
import torch

from zigzag_project.zsampling import (
    ZSamplingConfig,
    make_initial_latents,
    validate_zsampling_config,
)


def test_zsampling_config_defaults_match_upstream_cli_defaults():
    config = ZSamplingConfig()
    assert config.gamma_1 == 5.5
    assert config.gamma_2 == 0.0
    assert config.infer_step == 50
    assert config.resolved_lambda_step() == 49
    assert config.t_max == 1


@pytest.mark.parametrize(
    "config, message",
    [
        (ZSamplingConfig(infer_step=1), "infer_step"),
        (ZSamplingConfig(t_max=-1), "t_max"),
        (ZSamplingConfig(infer_step=5, lambda_step=5), "lambda_step"),
        (ZSamplingConfig(infer_step=5, lambda_step=-1), "lambda_step"),
    ],
)
def test_validate_zsampling_config_rejects_invalid_loop_bounds(config, message):
    with pytest.raises(ValueError, match=message):
        validate_zsampling_config(config)


def test_explicit_lambda_step_resolves_unchanged():
    assert ZSamplingConfig(infer_step=10, lambda_step=3).resolved_lambda_step() == 3


def test_make_initial_latents_is_deterministic_and_correct_shape():
    kwargs = {
        "seed": 123,
        "height": 64,
        "width": 96,
        "dtype": torch.float32,
        "device": torch.device("cpu"),
    }
    first = make_initial_latents(**kwargs)
    second = make_initial_latents(**kwargs)
    assert first.shape == (1, 4, 8, 12)
    assert first.dtype == torch.float32
    assert first.device == torch.device("cpu")
    assert torch.equal(first, second)


def test_make_initial_latents_rejects_non_divisible_dimensions():
    with pytest.raises(ValueError, match="divisible"):
        make_initial_latents(
            seed=1,
            height=65,
            width=64,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
