import gc
from datetime import datetime
from pathlib import Path

import torch
from diffusers import DDIMScheduler, StableDiffusionXLPipeline

SDXL = "sdxl"
DREAM_SHARPER = "dreamshaper-xl-v2-turbo"

MODEL_CONFIGS = {
    # The code released with the ZigZag Diffusion Sampling paper uses SDXL:
    # https://github.com/xie-lab-ml/Zigzag-Diffusion-Sampling/tree/main
    # The setup for the other models is not certain, other than the
    # parameters reported in the paper itself.
    SDXL: {
        "model_id": "stabilityai/stable-diffusion-xl-base-1.0",  # From the paper's implementation
        "num_inference_steps": 50,  # From the paper's implementation
        "guidance_scale": 5.5,  # From the paper's implementation
        "height": 1024,  # From the paper's implementation
        "width": 1024,  # From the paper's implementation
    },
    DREAM_SHARPER: {
        "model_id": "Lykon/dreamshaper-xl-v2-turbo",  # Public model for DreamSharper XL Turbo
        "num_inference_steps": 4,  # Reported in the paper
        "guidance_scale": 3.5,  # Reported in the paper
        "height": 1024,  # Same resolution as for SDXL since DreamShaper XL Turbo is SDXL-based
        "width": 1024,  # Same resolution as for SDXL since DreamShaper XL Turbo is SDXL-based
    }
    # "sd21": {}, TODO: Add SD2.1. Also check load_standard_pipeline and
    # the rest of the code to make it compatible.
    # Hunyuan-DiT has a canonical loader in models.py. This older SDXL-only
    # helper still needs a separate update before it can generate Hunyuan runs.
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

def load_pipeline_standard(model_name, model_id=None, use_ddim=True, device=None):
    """
    Loads a basic text-to-image pipeline for standard sampling.
    """

    config = get_model_config(model_name)
    if model_id is None:
        model_id = config["model_id"]
    if device is None:
        device = get_device()

    torch_dtype = get_torch_dtype(device)

    # NOTE: This loader currently assumes SDXL-compatible models because the current
    # model set only contains SDXL and DreamShaper XL. If SD2.1, Hunyuan-DiT, or other
    # backbones are added later, revisit this, instead of only adding a new model_id.
    # Other architectures may need StableDiffusionPipeline or DiffusionPipeline, etc.
    # Also, re-check things like scheduler, safety checkers, default resolution,
    # dtype/device constraints, etc.
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        use_safetensors=True,
    )

    if use_ddim:
        # The paper's released SDXL implementation uses DDIMScheduler
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    pipe = pipe.to(device)

    if device == "cuda":
        pipe.enable_attention_slicing()
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    return pipe


def generate_images_standard(pipeline, prompts, model_name, output_dir, seed=42, device=None):
    """
    Generates images with the chosen model using standard sampling.
    Saves images and returns their file paths.
    """

    config = get_model_config(model_name)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = torch.device(get_device())

    generator = torch.Generator(device=device.type).manual_seed(seed)

    result_paths = []
    for i, prompt in enumerate(prompts):
        result = pipeline(
            prompt,
            num_inference_steps=config["num_inference_steps"],
            guidance_scale=config["guidance_scale"],
            height=config["height"],
            width=config["width"],
            generator=generator,
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image = result.images[0]
        path = output_dir / f"{model_name}_{i:03d}_{timestamp}.png"
        image.save(path)
        result_paths.append(path)

    return result_paths


def free_pipeline(pipe=None):
    """
    Deletes a loaded pipeline and clears CUDA memory.

    Diffusion pipelines are large and can keep several GB of GPU memory allocated.
    Explicit cleanup helps avoid memory errors when loading multiple models
    sequentially in notebooks.
    """

    if pipe is not None:
        del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
