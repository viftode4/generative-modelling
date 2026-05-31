from zigzag_project.model_utils import *
from tqdm.auto import tqdm

# Configuration from the Resampling paper referenced by the ZigZag Sampling paper
RESAMPLING_CONFIG = {
    "jump_length": 10,
    "jump_n_sample": 10,
}

def get_resampling_config():
    return dict(RESAMPLING_CONFIG)

def get_resampling_schedule(num_inference_steps, jump_length=None, jump_n_sample=None):
    """
    Builds a resampling-style non-monotonic schedule for diffusion steps.

    In this schedule, high values mean noisier images and lower values mean cleaner:
    - When the schedule decreases -> denoise
    - When schedule increases -> re-noise

    A normal schedule might look like: 20 -> 19 -> 18
    But with resampling: 20 -> 19 -> 20 -> 19 -> 20 -> 19 -> 18

    -1 is the final (clean) state after the last denoising step.
    """

    if jump_length is None:
        jump_length = RESAMPLING_CONFIG["jump_length"]
    if jump_n_sample is None:
        jump_n_sample = RESAMPLING_CONFIG["jump_n_sample"]

    # We need to keep the schedule usable for low-step models such as DreamShaper (uses only 4 inference steps).
    # The original Resamplimg paper used jump_length=10 with 250 inference steps,
    # but a model using less than 10 steps cannot jump that much, so we need to clip the jump length.
    jump_length = min(jump_length, num_inference_steps - 1)

    # We add a resampling point every jump_length steps.
    # jumps[j] stores how many extra backward-forward cycles should happen at step j.
    # Example: jump_length=10 and jump_n_sample=3 give us 20 -> 19 -> 20 -> 19 -> 20 -> 19 -> 18
    jumps = {}
    for j in range(0, num_inference_steps - jump_length, jump_length):
        jumps[j] = jump_n_sample - 1 # We use jump_n_sample - 1 because the first visit already happens during normal denoising

    t = num_inference_steps
    schedule = []

    while t >= 1:
        t = t - 1
        schedule.append(t)

        if jumps.get(t, 0) > 0:
            jumps[t] = jumps[t] - 1
            for _ in range(jump_length):
                t = t + 1
                schedule.append(t)

    schedule.append(-1) # -1 represents the final, clean state
    return schedule

def get_timestep_from_logical(timesteps, logical_step_index):
    """
    Resampling schedules use logical step positions (0, 1, 2, ...),
    while diffusers use actual scheduler timestep values (999, 749, ...).

    E.g. logical schedule 20 -> 19 -> 20 -> 19 -> 18
    may correspond to scheduler timesteps: 532 -> 619 -> 532 -> 619 -> 744

    This helper converts a logical step index into the actual timestep value used by diffusion schedulers.
    """

    num_inference_steps = len(timesteps)

    if logical_step_index == -1: # -1 represents the final (clean) state in the resampling schedule
        index = num_inference_steps
    else:
        index = num_inference_steps - 1 - logical_step_index

    if index >= len(timesteps):
        return None

    return timesteps[index]


def apply_schedule_transition(
    pipeline,
    pipeline_type,
    latents,
    t_last,
    t_current,
    timesteps,
    prompt_data,
    conditioning_data,
    guidance_scale,
    generator=None
):
    """
    Applies one transition from the resampling schedule.
    """

    # If t_current < t_last, this is a reverse diffusion step, so we denoise
    if t_current < t_last:
        timestep = get_timestep_from_logical(timesteps, t_last)

        noise_prediction = predict_noise(
            pipeline,
            latents,
            timestep,
            prompt_data,
            conditioning_data,
            guidance_scale,
            pipeline_type=pipeline_type
        )

        return apply_denoise_step(pipeline, latents, timestep, noise_prediction, generator=generator)

    # If t_current > t_last, this is a forward diffusion jump, so we re-noise
    if t_current > t_last:
        timestep_last = get_timestep_from_logical(timesteps, t_last)
        timestep_current = get_timestep_from_logical(timesteps, t_current)
        return renoise_between_timesteps(pipeline.scheduler, latents, timestep_last, timestep_current, generator=generator)

    return latents

def renoise_between_timesteps(scheduler, latents, from_timestep, to_timestep, generator=None):
    """
    Resampling uses forward diffusion jumps in its schedule (adds noise to move latents to a noisier timestep).
    This helper implements the same idea for diffuser schedulers by using cumulative alpha values.
    Important: It is intended for deterministic DDIM-style schedules, not arbitrary ones.
    """

    # alphas_cumprod stores how much clean image signal remains at each timestep.
    # Larger alpha values mean cleaner images, while smaller alpha values mean noisier ones.
    if not hasattr(scheduler, "alphas_cumprod"):
        raise NotImplementedError("renoise_between_timesteps requires scheduler.alphas_cumprod from DDIM/DDPM-like scheduler.")

    from_timestep = int(from_timestep.item())
    to_timestep = int(to_timestep.item())

    alpha_from = scheduler.alphas_cumprod[from_timestep].to(device=latents.device, dtype=latents.dtype)
    alpha_to = scheduler.alphas_cumprod[to_timestep].to(device=latents.device, dtype=latents.dtype)

    if alpha_to > alpha_from:
        raise ValueError("alpha_to > alpha_from (re-noising should move from a less noisy state to a noisier state)")

    # In diffusion, image x at timestamp t is a mix of the clean image and noise:
    #   x_t = sqrt(alpha_t) * clean_image + sqrt(1 - alpha_t) * noise
    #
    # During resampling we move back to a noisier timestep by adding back noise:
    #   x_to = sqrt(alpha_to / alpha_from) * x_from + sqrt(1 - alpha_to / alpha_from) * noise

    alpha_ratio = torch.clamp(alpha_to / alpha_from, min=0.0, max=1.0)
    noise = torch.randn(latents.shape, generator=generator, device=latents.device, dtype=latents.dtype)

    return alpha_ratio.sqrt() * latents + (1.0 - alpha_ratio).sqrt() * noise

def generate_images_resampling(
    pipeline,
    prompts,
    model_name,
    output_dir,
    seed=42,
    latents=None,
    jump_length=None,
    jump_n_sample=None,
    run_id=None,
    negative_prompt=None,
    device=None,
    show_progress=True
):
    """
    Generates images with the chosen model using Resampling.
    Saves images and returns their file paths.
    """

    config = get_model_config(model_name)
    pipeline_type = config["pipeline_type"]
    if device is None:
        device = torch.device(get_device())
    if run_id is None:
        run_id = make_run_id()

    set_scheduler_timesteps(pipeline, config["num_inference_steps"], device=device)
    timesteps = pipeline.scheduler.timesteps
    schedule = get_resampling_schedule(config["num_inference_steps"], jump_length=jump_length, jump_n_sample=jump_n_sample)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result_paths = []

    for i, prompt in enumerate(prompts):
        generator = make_generator(seed + i, device)

        prompt_data = encode_prompt(
            pipeline,
            prompt,
            pipeline_type,
            device=device,
            negative_prompt=negative_prompt
        )

        conditioning_data = get_additional_conditioning(
            pipeline,
            pipeline_type,
            height=config["height"],
            width=config["width"],
            dtype=prompt_data["embeddings"].dtype,
            device=device
        )

        prompt_latents = None
        if latents is not None:
            if isinstance(latents, (list, tuple)):
                prompt_latents = latents[i]
            else:
                prompt_latents = latents

        current_latents = prepare_latents(
            pipeline,
            pipeline_type,
            height=config["height"],
            width=config["width"],
            generator=generator,
            device=device,
            latents=prompt_latents
        )

        with torch.no_grad():
            transitions = list(zip(schedule[:-1], schedule[1:]))

            if show_progress:
                transitions = tqdm(transitions, total=len(transitions), desc=f"Diffusion steps for prompt {i + 1}", leave=True)

            for t_last, t_cur in transitions:
                current_latents = apply_schedule_transition(
                    pipeline,
                    pipeline_type,
                    current_latents,
                    t_last,
                    t_cur,
                    timesteps,
                    prompt_data,
                    conditioning_data,
                    config["guidance_scale"],
                    generator=generator
                )

            image = decode_latents(pipeline, current_latents, pipeline_type)

        path = save_image(image, output_dir, model_name,"resampling", i, run_id=run_id)
        result_paths.append(path)

    return result_paths
