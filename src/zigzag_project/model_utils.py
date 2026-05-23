from datetime import datetime
from pathlib import Path
import gc
import torch
from diffusers import StableDiffusionXLPipeline, DDIMScheduler

SDXL = "sdxl"
DREAM_SHARPER = "dreamshaper-xl-v2-turbo"

PIPELINE_SDXL = "sdxl"

# The implementation currently assumes SDXL-compatible models because the current model set
# only contains SDXL and DreamShaper XL. If SD2.1, Hunyuan-DiT, or other backbones are added later,
# revisit the code, instead of only adding a new model_id.
# Other architectures may need StableDiffusionPipeline or DiffusionPipeline, different ecoding/decoding logic, etc.
# Also, re-check things like scheduler, safety checkers, default resolution, dtype/device constraints, etc.
SUPPORTED_PIPELINE_TYPES = { PIPELINE_SDXL }

MODEL_CONFIGS = {
    # The code released with the ZigZag Diffusion Sampling paper uses SDXL: https://github.com/xie-lab-ml/Zigzag-Diffusion-Sampling/tree/main
    # The setup for the other models is not certain, other than the parameters reported in the paper itself.
    SDXL: {
        "model_id": "stabilityai/stable-diffusion-xl-base-1.0",  # From the paper's implementation
        "pipeline_type": PIPELINE_SDXL,
        "num_inference_steps": 50,  # From the paper's implementation
        "guidance_scale": 5.5,  # From the paper's implementation
        "height": 1024,  # From the paper's implementation
        "width": 1024,  # From the paper's implementation
        "use_ddim": True  # The paper's implementation replaces the default scheduler with DDIM
    },
    DREAM_SHARPER: {
        "model_id": "Lykon/dreamshaper-xl-v2-turbo",  # Public model for DreamSharper XL Turbo
        "pipeline_type": PIPELINE_SDXL,
        "num_inference_steps": 4,  # Reported in the paper
        "guidance_scale": 3.5,  # Reported in the paper
        "height": 1024,  # Same resolution as for SDXL since DreamShaper XL Turbo is SDXL-based
        "width": 1024,  # Same resolution as for SDXL since DreamShaper XL Turbo is SDXL-based
        "use_ddim": True  # Same as for SDXL since DreamShaper XL Turbo, for comparable experiments
    }
    # "sd21": {}, TODO: Add SD2.1. Also check load_pipeline and the rest of the code to make it compatible.
    # "hunyuan-dit": {} TODO: Add Hunyuan-DiT. Also check load_pipeline and the rest of the code to make it compatible.
}

def list_model_configs():
    return MODEL_CONFIGS

def get_model_config(model_name):
    return dict(MODEL_CONFIGS[model_name])

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def get_torch_dtype(device=None):
    """Uses float16 on CUDA and float32 on CPU."""
    if device is None:
        device = get_device()
    return torch.float16 if device == "cuda" else torch.float32

def make_generator(seed=42, device=None):
    """Creates a torch generator (for reproducible latent sampling)."""
    if device is None:
        torch.device(get_device())
    return torch.Generator(device=device.type).manual_seed(seed)

def make_run_id():
    """Creates a timestamp used to avoid overwriting generated files."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_supported_pipeline_type(pipeline_type):
    """For throwing an error when a helper is used with an unsupported pipeline type."""
    if pipeline_type not in SUPPORTED_PIPELINE_TYPES:
        raise NotImplementedError(
            f"Unsupported pipeline type: {pipeline_type}. "
            "Check compatibility or add model-specific prompt encoding, latent creation, noise prediction, and decoding."
        )

def ensure_stable_diffusion_xl_pipeline(pipeline):
    """Throws error when an SDXL-only helper receives another pipeline type."""
    if not isinstance(pipeline, StableDiffusionXLPipeline):
        raise NotImplementedError(
            "This helper currently supports StableDiffusionXLPipeline only. "
            "For other architectures, add a pipeline-specific implementation."
        )

def load_pipeline_standard(model_name, model_id=None, pipeline_type=None, use_ddim=None, device=None):
    """ Just a wrapper for backward compatibility.  """
    return load_pipeline(model_name, model_id, pipeline_type, use_ddim, device)

def load_pipeline(model_name, model_id=None, pipeline_type=None, use_ddim=None, device=None):
    """
    Loads a basic text-to-image pipeline for standard sampling.
    """

    config = get_model_config(model_name)
    if model_id is None:
        model_id = config["model_id"]
    if pipeline_type is None:
        pipeline_type = config["pipeline_type"]
    if device is None:
        device = torch.device(get_device())
    if use_ddim is None:
        use_ddim = config["use_ddim"]
    torch_dtype = get_torch_dtype(device)

    ensure_supported_pipeline_type(pipeline_type)

    if pipeline_type == PIPELINE_SDXL:
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            use_safetensors=True
        )
    else:
        raise NotImplementedError(f"No load_pipeline_standard implemented for pipeline type: {pipeline_type}")

    if use_ddim:
        # The released Zigzag SDXL implementation uses DDIMScheduler.
        # This also matters for Z-Sampling-style inversion and Resampling-style jumps
        # because deterministic schedules are easier to compare fairly.
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

    pipeline = pipeline.to(device)

    if device == "cuda":
        pipeline.enable_attention_slicing()
        try:
            pipeline.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    return pipeline

def set_scheduler_timesteps(pipeline, num_inference_steps, device=None):
    """Sets scheduler timesteps and returns them."""
    if device is None:
        device = torch.device(get_device())
    pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
    return pipeline.scheduler.timesteps

def encode_prompt(pipeline, prompt, pipeline_type, device=None, negative_prompt=None):
    """
    Encodes one prompt for manual denoising.
    """

    ensure_supported_pipeline_type(pipeline_type)

    if device is None:
        device = torch.device(get_device())

    if pipeline_type != PIPELINE_SDXL:
        raise NotImplementedError(f"No encode_prompt implemented for pipeline type: {pipeline_type}")

    ensure_stable_diffusion_xl_pipeline(pipeline)

    embeddings, negative_embeddings, pooled_embeddings, negative_pooled_embeddings = pipeline.encode_prompt(
        prompt=prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt=negative_prompt
    )

    # Classifier-free guidance needs both unconditional and conditional embeddings,
    # so both are concatenated here (this is not model-specific).
    embeddings = torch.cat([negative_embeddings, embeddings])

    pooled_embeddings = torch.cat([negative_pooled_embeddings, pooled_embeddings])

    return {
        "embeddings": embeddings,
        "pooled_embeddings": pooled_embeddings,
    }

def get_additional_conditioning(pipeline, pipeline_type, height=None, width=None, dtype=None, device=None):
    """
    Builds extra conditioning needed by the model.

    Note: SDXL uses size/crop conditioning with added time IDs.
    """

    ensure_supported_pipeline_type(pipeline_type)

    if device is None:
        device = torch.device(get_device())

    if pipeline_type != PIPELINE_SDXL:
        raise NotImplementedError(f"No get_additional_conditioning implemented for pipeline type: {pipeline_type}")

    ensure_stable_diffusion_xl_pipeline(pipeline)

    original_size = (height, width)
    target_size = (height, width)
    top_left = (0, 0)

    time_ids = pipeline._get_add_time_ids(
        original_size,
        top_left,
        target_size,
        dtype=dtype,
        text_encoder_projection_dim=pipeline.text_encoder_2.config.projection_dim
    )

    time_ids = time_ids.to(device)
    time_ids = torch.cat([time_ids, time_ids])

    return {
        "time_ids": time_ids,
    }

def prepare_latents(pipeline, pipeline_type, height=None, width=None, generator=None, device=None, latents=None):
    """
    Creates/validates the initial latents.

    Note: Passing latents is useful for fair comparisons, because standard sampling,
    resampling, and Z-Sampling can then start from the exact same noise.
    """

    ensure_supported_pipeline_type(pipeline_type)
    if device is None:
        device = torch.device(device)

    if pipeline_type != PIPELINE_SDXL:
        raise NotImplementedError(f"No prepare_latents implemented for pipeline type: {pipeline_type}")

    ensure_stable_diffusion_xl_pipeline(pipeline)

    if latents is not None:
        return latents.detach().clone().to(device=device, dtype=pipeline.unet.dtype)

    n_channels = pipeline.unet.config.in_channels

    return pipeline.prepare_latents(
        batch_size=1,
        num_channels_latents=n_channels,
        height=height,
        width=width,
        dtype=pipeline.unet.dtype,
        device=device,
        generator=generator,
        latents=None
    )

def predict_noise(pipeline, latents, timestep, prompt_data, conditioning_data, guidance_scale, pipeline_type):
    """
    Predicts noise for one denoising step (classifier-free guidance).
    """

    ensure_supported_pipeline_type(pipeline_type)

    if pipeline_type != PIPELINE_SDXL:
        raise NotImplementedError(f"No predict_noise implemented for pipeline type: {pipeline_type}")

    ensure_stable_diffusion_xl_pipeline(pipeline)

    latent_model_input = torch.cat([latents] * 2)
    latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, timestep)

    pooled_embeddings_and_time_ids = {
        "text_embeds": prompt_data["pooled_embeddings"],
        "time_ids": conditioning_data["time_ids"],
    }

    noise_prediction = pipeline.unet(
        latent_model_input,
        timestep,
        encoder_hidden_states=prompt_data["embeddings"],
        added_cond_kwargs=pooled_embeddings_and_time_ids,
        return_dict=False
    )[0]

    noise_prediction_unconditional, noise_prediction_conditional = noise_prediction.chunk(2)
    return noise_prediction_unconditional + guidance_scale * (noise_prediction_conditional - noise_prediction_unconditional)

def apply_denoise_step(pipeline, latents, timestep, noise_prediction, generator=None):
    return pipeline.scheduler.step(
        noise_prediction,
        timestep,
        latents,
        generator=generator,
        return_dict=False
    )[0]

def decode_latents(pipeline, latents, pipeline_type):
    """Decodes final latents to an image."""

    ensure_supported_pipeline_type(pipeline_type)

    if pipeline_type != PIPELINE_SDXL:
        raise NotImplementedError(f"No decode_latents implemented for pipeline type: {pipeline_type}")

    ensure_stable_diffusion_xl_pipeline(pipeline)

    decoded = pipeline.vae.decode(latents / pipeline.vae.config.scaling_factor, return_dict=False)
    image = decoded[0]
    return pipeline.image_processor.postprocess(image, output_type="pil")[0]

def save_image(image, output_dir, model_name, method_name, prompt_index, run_id=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if run_id is None:
        run_id = make_run_id()

    path = output_dir / f"{model_name}_{method_name}_{prompt_index:03d}_{run_id}.png"
    image.save(path)

    return path


def generate_images_standard(pipeline, prompts, model_name, output_dir, seed=42, latents=None,
                             run_id=None, negative_prompt=None, device=None):
    """
    Generates images with the chosen model using standard sampling.
    Saves images and returns their file paths.
    """

    config = get_model_config(model_name)

    if device is None:
        device = torch.device(get_device())
    if run_id is None:
        run_id = make_run_id()

    generator = make_generator(seed, device)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result_paths = []

    for i, prompt in enumerate(prompts):
        prompt_latents = None

        # Accepts either one shared latent tensor for all prompts
        # or a list with one latent tensor per prompt
        if latents is not None:
            if isinstance(latents, (list, tuple)):
                prompt_latents = latents[i].detach().clone()
            else:
                prompt_latents = latents.detach().clone()

        result = pipeline(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=config["num_inference_steps"],
            guidance_scale=config["guidance_scale"],
            height=config["height"],
            width=config["width"],
            generator=generator,
            latents=prompt_latents
        )

        path = save_image(result.images[0], output_dir, model_name, "standard", i, run_id=run_id)
        result_paths.append(path)

    return result_paths


def free_pipeline(pipe=None):
    """
    Deletes a loaded pipeline and clears CUDA memory.

    Diffusion pipelines are large and can keep several GB of GPU memory allocated.
    Explicit cleanup helps avoid memory errors when loading multiple models sequentially in notebooks.
    """

    if pipe is not None:
        del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
