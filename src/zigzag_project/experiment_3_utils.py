from __future__ import annotations

import gc
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
import pandas as pd
import torch
import pathlib
import shutil
import urllib.request
import hpsv2
import clip
import ImageReward as RM
from transformers import AutoModel, AutoProcessor
from PIL import Image
from tqdm.auto import tqdm
from zigzag_project.models import load_dreamshaper_xl_turbo
from zigzag_project.zsampling import ZSamplingConfig, z_sample_sdxl

AYS_SCHEDULE: list[int] = [999, 845, 730, 587, 443, 310, 193, 116, 53, 13]

EXPERIMENT_3_METHODS = {
    "standard_10step": {"use_ays": False, "use_zsampling": False},
    "ays_10step": {"use_ays": True, "use_zsampling": False},
    "zsampling_10step": {"use_ays": False, "use_zsampling": True},
    "ays_zsampling_10step": {"use_ays": True, "use_zsampling": True}
}

EXPERIMENT_3_METHOD_NAMES = {
    "standard_10step": "Standard",
    "zsampling_10step": "Z-Sampling",
    "ays_10step": "AYS",
    "ays_zsampling_10step": "AYS + Z-Sampling"
}

EXPERIMENT_3_METRICS = ("hps_v2", "aes", "pickscore", "imagereward")

@dataclass(frozen=True)
class Experiment3TableConfig:
    method: str
    project_root: str | Path
    output_root: str | Path
    seeds: tuple[int, ...] = (42,)
    pickapic_limit: int = 100
    drawbench_limit: int | None = None
    num_inference_steps: int = 10
    guidance_scale: float = 3.5
    inversion_guidance_scale: float = 0.0
    height: int = 1024
    width: int = 1024
    model_name: str = "dreamshaper-xl-v2-turbo"
    @property
    def use_ays(self):
        return EXPERIMENT_3_METHODS[self.method]["use_ays"]
    @property
    def use_zsampling(self):
        return EXPERIMENT_3_METHODS[self.method]["use_zsampling"]
    @property
    def run_path(self):
        return Path(self.output_root) / self.method
    @property
    def image_path(self):
        return self.run_path / "images"

    @property
    def manifest_path(self):
        return self.run_path / "manifest.jsonl"

    @property
    def scores_path(self):
        return self.run_path / "scores.jsonl"
    @property
    def metadata_path(self) -> Path:
        return self.run_path / "metadata.json"

def get_named_timesteps(schedule_name = None):
    if schedule_name is None:
        return None
    name = schedule_name.lower().replace("-", "_")
    if name in {"ays", "ays_sdxl", "ays_sdxl_10", "stable_diffusion_xl_ays"}:
        return list(AYS_SCHEDULE)
    raise ValueError(f"Unknown schedule_name: {schedule_name}")

def apend_jsonl(path, r):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())

def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def write_metadata(config, extra = None):
    config.run_path.mkdir(parents=True, exist_ok=True)
    x = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()},
        "method": EXPERIMENT_3_METHODS[config.method],
        "method_name": EXPERIMENT_3_METHOD_NAMES[config.method],
        "ays_timesteps": AYS_SCHEDULE if config.use_ays else None,
        "model": "Lykon/dreamshaper-xl-v2-turbo",
        "inference_steps": 10,
        "gamma_1": config.guidance_scale,
        "gamma_2": config.inversion_guidance_scale if config.use_zsampling else None,
        "lambda_step": config.num_inference_steps - 1 if config.use_zsampling else None,
        "zsampling_t_max": 1 if config.use_zsampling else None,
        "extra": extra or {}
    }
    config.metadata_path.write_text(json.dumps(x, indent=2, ensure_ascii=False), encoding="utf-8")


def load_prompt_file(path, dataset, n):
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")

    lines = [ line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() ]

    if not lines:
        raise ValueError(f"Prompt file is emty: {path}")

    first = lines[0].lower().strip()
    prompts: list[str]

    if first in {"prompt", "prompts", "caption"}:
        prompts = lines[1:]
    elif first.startswith("prompts,"):
        prompts = []
        for l in lines[1:]:
            if "," in l:
                prompt, _ = l.rsplit(",", 1)
            else:
                prompt = l
            prompts.append(prompt.strip().strip('"'))
    else:
        prompts = lines

    prompts = [ p.strip().strip('"')
        for p in prompts
        if p.strip() and p.strip().lower() != "nan"]

    if n is not None:
        prompts = prompts[:n]

    d = pd.DataFrame({"prompt": prompts})
    d.insert(0, "prompt_id", range(len(d)))
    d.insert(0, "dataset", dataset)
    return d

def load_experiment_3_prompts(root, pickapic_limit = 100, drawbench_limit = None):
    rp = Path(root)
    pickapic = load_prompt_file(rp / "data" / "prompts" / "pick-a-pic.csv", "pickapic", pickapic_limit)
    drawbench = load_prompt_file(rp / "data" / "prompts" / "draw-bench.csv", "drawbench", drawbench_limit)
    ps = pd.concat([pickapic, drawbench], ignore_index=True)
    ps.insert(0, "global_id", range(len(ps)))
    return ps

def image_name(dataset, prompt_id, s, method):
    return f"{dataset}_{prompt_id:04d}_seed{s}_{method}.png"

def run_generation_3(config):
    if config.method not in EXPERIMENT_3_METHODS:
        raise ValueError(f"Unknown method {config.method}")

    config.image_path.mkdir(parents=True, exist_ok=True)
    write_metadata(config)
    ps = load_experiment_3_prompts(config.project_root, config.pickapic_limit, config.drawbench_limit)

    completed = {
        (r["dataset"], int(r["prompt_id"]), int(r["seed"]), r["method"])
        for r in read_jsonl(config.manifest_path) }

    device = "cuda" if torch.cuda.is_available() else None
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    handle = load_dreamshaper_xl_turbo(device=device, dtype=dtype)

    print(f"Method: {config.method}")
    print(f"Output: {config.run_path}")
    print(f"Prompts: {len(ps)} (existing generated rows: {len(completed)})")

    try:
        iterator = ps.itertuples(index=False)
        for r in tqdm(list(iterator), desc=f"generate {config.method}"):
            for s in config.seeds:

                k = (r.dataset, int(r.prompt_id), int(s), config.method)
                if k in completed:
                    continue

                nmae = image_name(r.dataset, int(r.prompt_id), int(s), config.method)
                out_path = config.image_path / nmae
                generator_device = handle.device if str(handle.device) != "mps" else torch.device("cpu")
                generator = torch.Generator(device=generator_device).manual_seed(int(s))
                started = time.time()

                with torch.inference_mode():
                    zc = ZSamplingConfig(
                        gamma_1=config.guidance_scale,
                        gamma_2=config.inversion_guidance_scale,
                        infer_step=config.num_inference_steps,
                        lambda_step=config.num_inference_steps - 1 if config.use_zsampling else 0,
                        t_max=1,
                        schedule_name="ays_sdxl_10" if config.use_ays else None,
                    )
                    image = z_sample_sdxl(
                        handle,
                        prompt=r.prompt,
                        config=zc,
                        height=config.height,
                        width=config.width,
                        generator=generator,
                    ).images[0]

                image.save(out_path)
                x = {
                    "method": config.method,
                    "method_display_name": EXPERIMENT_3_METHOD_NAMES[config.method],
                    "dataset": r.dataset,
                    "prompt_id": int(r.prompt_id),
                    "global_id": int(r.global_id),
                    "prompt": r.prompt,
                    "seed": int(s),
                    "image_path": str(out_path),
                    "runtime_s": round(time.time() - started, 4),
                    "model_id": handle.hf_id,
                    "height": config.height,
                    "width": config.width,
                    "num_inference_steps": config.num_inference_steps,
                    "guidance_scale": config.guidance_scale,
                    "inversion_guidance_scale": config.inversion_guidance_scale if config.use_zsampling else None,
                    "lambda_step": config.num_inference_steps - 1 if config.use_zsampling else 0,
                    "t_max": 1 if config.use_zsampling else None,
                    "ays_timesteps": AYS_SCHEDULE if config.use_ays else None
                }
                apend_jsonl(config.manifest_path, x)
                completed.add(k)
    finally:
        del handle
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return config.manifest_path


class Experiment3Scores:
    """HPS v2, AES, PickScore, and ImageReward."""

    def __init__(self, device = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._hpsv2 = None
        self._aes = None
        self._pick = None
        self._ir = None

    def hps(self, image_path, prompt):
        if self._hpsv2 is None:
            path = pathlib.Path(hpsv2.__file__).parent / "src" / "open_clip" / "bpe_simple_vocab_16e6.txt.gz"
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    src = pathlib.Path(clip.__file__).parent / "bpe_simple_vocab_16e6.txt.gz"
                    shutil.copy(src, path)
                except Exception:
                    urllib.request.urlretrieve("https://github.com/tgxs002/HPSv2/raw/master/src/open_clip/bpe_simple_vocab_16e6.txt.gz", path)
            self._hpsv2 = hpsv2

        score = self._hpsv2.score([image_path], prompt, hps_version="v2.1")
        return float(torch.as_tensor(score).reshape(-1)[0].item()) * 100.0

    def aes(self, image_path, prompt = None):
        if self._aes is None:

            class AesMLP(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.layers = torch.nn.Sequential(
                        torch.nn.Linear(768, 1024),
                        torch.nn.Dropout(0.2),
                        torch.nn.Linear(1024, 128),
                        torch.nn.Dropout(0.2),
                        torch.nn.Linear(128, 64),
                        torch.nn.Dropout(0.1),
                        torch.nn.Linear(64, 16),
                        torch.nn.Linear(16, 1))

                def forward(self, x):
                    return self.layers(x)

            m, preprocess = clip.load("ViT-L/14", device=self.device)
            url = "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth"
            states = torch.hub.load_state_dict_from_url(url, map_location=self.device)
            mlp = AesMLP().to(self.device).eval()
            mlp.load_state_dict(states)
            self._aes = (m.eval(), preprocess, mlp)

        m, preprocess, mlp = self._aes

        with torch.no_grad():
            image = preprocess(Image.open(image_path).convert("RGB"))
            image = image.unsqueeze(0).to(self.device)
            e = m.encode_image(image).float()
            e = e / e.norm(dim=-1, keepdim=True)
            return float(mlp(e).item())

    def pickscore(self, image_path: str, prompt: str) -> float:
        if self._pick is None:
            processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
            m = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").to(self.device).eval()
            self._pick = (processor, m)
    
        processor, m = self._pick
    
        def as_e(x):
            if hasattr(x, "pooler_output") and x.pooler_output is not None:
                return x.pooler_output
            if hasattr(x, "last_hidden_state"):
                return x.last_hidden_state[:, 0]
            return x
    
        with torch.no_grad():
            inputs = processor(
                text=[prompt],
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt"
            ).to(self.device)
    
            image_inputs = processor(images=Image.open(image_path).convert("RGB"), return_tensors="pt").to(self.device)
            text_e = as_e(m.get_text_features(**inputs))
            image_e = as_e(m.get_image_features(**image_inputs))
            text_e = text_e / text_e.norm(dim=-1, keepdim=True)
            image_e = image_e / image_e.norm(dim=-1, keepdim=True)
            x = m.logit_scale.exp() * (text_e @ image_e.T)
            return float(x.item())

    def ir(self, image_path, prompt):
        if self._ir is None:
            self._ir = RM.load("ImageReward-v1.0", device=self.device)
        s = self._ir.score(prompt, image_path)
        return float(s) * 100.0

    def score_one(self, image_path, prompt):
        return {
            "hps_v2": self.hps(image_path, prompt),
            "aes": self.aes(image_path, prompt),
            "pickscore": self.pickscore(image_path, prompt),
            "imagereward": self.ir(image_path, prompt)
        }

def score(config):
    manifest = read_jsonl(config.manifest_path)
    done = {(r["dataset"], int(r["prompt_id"]), int(r["seed"]), r["method"])
        for r in read_jsonl(config.scores_path)}
    pending = [r for r in manifest if (r["dataset"], int(r["prompt_id"]), int(r["seed"]), r["method"]) not in done]
    print(f"Method: {config.method}")
    print(f"Generated rows: {len(manifest)} (existing scored rows: {len(done)}, pending: {len(pending)})")

    scorers = Experiment3Scores()
    for r in tqdm(pending, desc=f"score {config.method}"):
        scores = scorers.score_one(r["image_path"], r["prompt"])
        apend_jsonl(config.scores_path, {**r, **scores})
    return config.scores_path

def summarize_table(output_root):
    rows = []
    for method in EXPERIMENT_3_METHODS:
        scores_path = Path(output_root) / method / "scores.jsonl"
        rows.extend(read_jsonl(scores_path))

    d = pd.DataFrame(rows)
    if d.empty:
        raise FileNotFoundError("No scores found")

    summary = (d.groupby(["method", "method_display_name", "dataset"], sort=False)[list(EXPERIMENT_3_METRICS)]
        .mean(numeric_only=True)
        .reset_index())
    summary["method_order"] = summary["method"].map({m: i for i, m in enumerate(EXPERIMENT_3_METHODS)})
    summary = summary.sort_values(["method_order", "dataset"]).drop(columns=["method_order"])

    path = Path(output_root) / "experiment_3_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)

    summary.to_csv(path, index=False)
    return summary

def table_3(output_path):
    summary = summarize_table(output_path)
    d1 = summary[summary["dataset"] == "pickapic"].set_index("method")
    d2 = summary[summary["dataset"] == "drawbench"].set_index("method")
    rows = []

    for m in EXPERIMENT_3_METHODS:
        rows.append({
            "Method": EXPERIMENT_3_METHOD_NAMES[m],
            "Pick-a-Pic HPS v2": d1.loc[m, "hps_v2"] if m in d1.index else None,
            "Pick-a-Pic AES": d1.loc[m, "aes"] if m in d1.index else None,
            "Pick-a-Pic PickScore": d1.loc[m, "pickscore"] if m in d1.index else None,
            "Pick-a-Pic IR": d1.loc[m, "imagereward"] if m in d1.index else None,
            "DrawBench HPS v2": d2.loc[m, "hps_v2"] if m in d2.index else None,
            "DrawBench AES": d2.loc[m, "aes"] if m in d2.index else None,
            "DrawBench PickScore": d2.loc[m, "pickscore"] if m in d2.index else None,
            "DrawBench IR": d2.loc[m, "imagereward"] if m in d2.index else None
        })

    table = pd.DataFrame(rows)
    table.to_csv(Path(output_path) / "experiment_3_table.csv", index=False)
    return table