import pytest
import torch

from zigzag_project.zsampling import (
    ZSamplingConfig,
    make_initial_latents,
    validate_zsampling_config,
    z_sample_sd21,
    z_sample_sdxl,
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


def test_z_sample_sd21_is_callable():
    assert callable(z_sample_sd21)


def test_z_sample_sd21_executes_same_zigzag_sequence_and_guidance_gap():
    scheduler = _RecordingScheduler()
    inverse_scheduler = _RecordingScheduler()
    pipe = _FakeSd21Pipe(scheduler)
    handle = _FakeHandle(pipe=pipe, inverse_scheduler=inverse_scheduler)

    result = z_sample_sd21(
        handle,  # type: ignore[arg-type]
        prompt="a photo of a bench",
        config=ZSamplingConfig(gamma_1=7.5, gamma_2=0.0, infer_step=4, lambda_step=3, t_max=1),
        height=8,
        width=8,
        latents=torch.zeros((1, 1, 1, 1)),
        output_type="latent",
    )

    assert result.images.shape == (1, 1, 1, 1)
    assert scheduler.timesteps_seen == [3, 3, 2, 2, 1, 1, 0]
    assert inverse_scheduler.timesteps_seen == [2, 1, 0]
    assert scheduler.model_outputs == pytest.approx([7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5])
    assert inverse_scheduler.model_outputs == pytest.approx([0.0, 0.0, 0.0])


def test_z_sample_sdxl_executes_paper_zigzag_sequence_and_guidance_gap():
    scheduler = _RecordingScheduler()
    inverse_scheduler = _RecordingScheduler()
    pipe = _FakeSdxlPipe(scheduler)
    handle = _FakeHandle(pipe=pipe, inverse_scheduler=inverse_scheduler)

    result = z_sample_sdxl(
        handle,  # type: ignore[arg-type]
        prompt="a photo of a bench",
        config=ZSamplingConfig(gamma_1=3.5, gamma_2=0.0, infer_step=4, lambda_step=3, t_max=1),
        height=8,
        width=8,
        latents=torch.zeros((1, 1, 1, 1)),
        output_type="latent",
    )

    assert result.images.shape == (1, 1, 1, 1)
    # Paper/upstream behavior for T=4, lambda=T-1:
    # denoise t3, inverse to t2, re-denoise t3;
    # denoise t2, inverse to t1, re-denoise t2;
    # denoise t1, inverse to t0, re-denoise t1;
    # final denoise t0 only.
    assert scheduler.timesteps_seen == [3, 3, 2, 2, 1, 1, 0]
    assert inverse_scheduler.timesteps_seen == [2, 1, 0]
    # The fake U-Net returns unconditional noise=0 and text noise=1. Classifier-free
    # guidance therefore collapses to exactly the guidance scale. This verifies the
    # paper's guidance gap implementation: gamma_1 on denoising/re-denoising and
    # gamma_2 on inversion.
    assert scheduler.model_outputs == pytest.approx([3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5])
    assert inverse_scheduler.model_outputs == pytest.approx([0.0, 0.0, 0.0])


def test_z_sample_sdxl_tmax_repeats_inverse_and_redeniose_rounds():
    scheduler = _RecordingScheduler()
    inverse_scheduler = _RecordingScheduler()
    pipe = _FakeSdxlPipe(scheduler)
    handle = _FakeHandle(pipe=pipe, inverse_scheduler=inverse_scheduler)

    z_sample_sdxl(
        handle,  # type: ignore[arg-type]
        prompt="a photo of a bench",
        config=ZSamplingConfig(gamma_1=4.0, gamma_2=1.0, infer_step=3, lambda_step=2, t_max=2),
        height=8,
        width=8,
        latents=torch.zeros((1, 1, 1, 1)),
        output_type="latent",
    )

    # For each refined timestep, the outer denoise is followed by two
    # inverse/re-denoise rounds. The final unrefined timestep denoises once.
    assert scheduler.timesteps_seen == [2, 2, 2, 1, 1, 1, 0]
    assert inverse_scheduler.timesteps_seen == [1, 1, 0, 0]


class _FakeHandle:
    def __init__(self, pipe, inverse_scheduler):
        self.pipe = pipe
        self.inverse_scheduler = inverse_scheduler
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.hf_id = "fake/sdxl"


class _FakeUnet:
    class _Config:
        in_channels = 1
        sample_size = 1
        time_cond_proj_dim = None

    config = _Config()

    def __call__(self, latent_model_input, *args, **kwargs):
        batch = latent_model_input.shape[0]
        assert batch % 2 == 0
        uncond = torch.zeros_like(latent_model_input[: batch // 2])
        text = torch.ones_like(latent_model_input[batch // 2 :])
        return (torch.cat([uncond, text], dim=0),)


class _RecordingScheduler:
    order = 1
    init_noise_sigma = 1.0

    def __init__(self):
        self.timesteps = torch.empty(0, dtype=torch.long)
        self.timesteps_seen: list[int] = []
        self.model_outputs: list[float] = []
        self._step_index = None

    def set_timesteps(self, num_inference_steps, device=None, **kwargs):
        self.timesteps = torch.arange(num_inference_steps - 1, -1, -1, device=device)

    def scale_model_input(self, latent_model_input, timestep):
        return latent_model_input

    def step(self, model_output, timestep, sample, **kwargs):
        self.timesteps_seen.append(int(timestep.item()))
        self.model_outputs.append(float(model_output.mean().item()))
        return (sample + model_output,)


class _FakeProgressBar:
    def __init__(self, total):
        self.total = total
        self.updates = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def update(self):
        self.updates += 1


class _FakeSdxlPipe:
    default_sample_size = 1
    vae_scale_factor = 8
    text_encoder_2 = None
    unet = _FakeUnet()
    watermark = None

    def __init__(self, scheduler):
        self.scheduler = scheduler
        self._guidance_scale = 1.0
        self._guidance_rescale = 0.0
        self._clip_skip = None
        self._cross_attention_kwargs = None
        self._denoising_end = None
        self._interrupt = False
        self._execution_device = torch.device("cpu")

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1.0

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def guidance_rescale(self):
        return self._guidance_rescale

    @property
    def clip_skip(self):
        return self._clip_skip

    @property
    def cross_attention_kwargs(self):
        return self._cross_attention_kwargs

    def check_inputs(self, *args, **kwargs):
        return None

    def encode_prompt(self, **kwargs):
        prompt_embeds = torch.ones((1, 2))
        negative_prompt_embeds = torch.zeros((1, 2))
        pooled_prompt_embeds = torch.ones((1, 2))
        negative_pooled_prompt_embeds = torch.zeros((1, 2))
        return (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents,
    ):
        if latents is not None:
            return latents.to(device=device, dtype=dtype)
        return torch.zeros(
            (batch_size, num_channels_latents, height, width),
            dtype=dtype,
            device=device,
        )

    def prepare_extra_step_kwargs(self, generator, eta):
        return {}

    def _get_add_time_ids(self, *args, **kwargs):
        return torch.zeros((1, 6))

    def progress_bar(self, total):
        return _FakeProgressBar(total)

    def maybe_free_model_hooks(self):
        return None


class _FakeSd21Pipe(_FakeSdxlPipe):
    def encode_prompt(self, **kwargs):
        prompt_embeds = torch.ones((1, 2))
        negative_prompt_embeds = torch.zeros((1, 2))
        return prompt_embeds, negative_prompt_embeds
