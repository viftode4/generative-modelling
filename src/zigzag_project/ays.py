from __future__ import annotations

AYS_SCHEDULE: list[int] = [999, 845, 730, 587, 443, 310, 193, 116, 53, 13]

def get_named_timesteps(schedule_name = None):
    if schedule_name is None:
        return None
    n = schedule_name.lower().replace("-", "_")
    if n in {"ays", "ays_sdxl", "ays_sdxl_10", "stable_diffusion_xl_ays"}:
        return list(AYS_SCHEDULE)
    raise ValueError(f"Unknown schedule_name: {schedule_name}")