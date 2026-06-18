"""Image-quality scorers for Experiment #6: HPS v2, AES, PickScore.

ImageReward runs in an isolated env (see score_guidance_gap_imagereward.py).
Heavy libs imported lazily so this module stays unit-testable.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

ALL_METRICS = ("hps", "aes", "pick", "ir")


def make_metric_score_rows(
    manifest_rows: Iterable[Mapping[str, Any]],
    scorers: Mapping[str, Callable[[str, str], float]],
    *,
    all_metrics: Sequence[str] = ALL_METRICS,
) -> list[dict[str, Any]]:
    """Score each manifest image. Missing metrics default to NaN."""
    passthrough = (
        "prompt_id", "source_index", "tag", "prompt", "seed",
        "method", "gamma_2", "gamma_1", "image_path",
    )
    out: list[dict[str, Any]] = []
    for row in manifest_rows:
        scored = {k: row[k] for k in passthrough if k in row}
        for m in all_metrics:
            fn = scorers.get(m)
            scored[m] = float(fn(row["image_path"], row["prompt"])) if fn else math.nan
        out.append(scored)
    return out


def _shim_turtle_for_hpsv2() -> None:
    """Shim the stray turtle import in hpsv2's vendored open_clip (absent on headless Python)."""
    import sys
    import types

    try:
        import turtle  # noqa: F401
    except Exception:
        stub = types.ModuleType("turtle")
        stub.forward = lambda *args, **kwargs: None
        sys.modules["turtle"] = stub


class HpsV2Scorer:
    """HPS v2.1 scorer. Raw score in [0,1]; paper reports x100."""

    def __init__(self, hps_version: str = "v2.1") -> None:
        _shim_turtle_for_hpsv2()
        import hpsv2  # noqa: F401  (import validates availability + triggers BPE path)

        self._hpsv2 = hpsv2
        self._version = hps_version

    def __call__(self, image_path: str, prompt: str) -> float:
        import numpy as np

        raw = self._hpsv2.score([image_path], prompt, hps_version=self._version)
        return float(np.asarray(raw, dtype=float).reshape(-1)[0])


class AesScorer:
    """LAION aesthetic predictor (CLIP ViT-L/14 → MLP)."""

    def __init__(self, device: str = "cpu") -> None:
        import clip
        import torch

        self._torch = torch
        self.device = device
        self.clip_model, self.preprocess = clip.load("ViT-L/14", device=device)

        mlp = torch.nn.Sequential(
            torch.nn.Linear(768, 1024), torch.nn.Dropout(0.2),
            torch.nn.Linear(1024, 128), torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64), torch.nn.Dropout(0.1),
            torch.nn.Linear(64, 16), torch.nn.Linear(16, 1),
        )
        url = ("https://github.com/christophschuhmann/improved-aesthetic-predictor/"
               "raw/main/sac+logos+ava1-l14-linearMSE.pth")
        state = torch.hub.load_state_dict_from_url(url, map_location=device)
        remapped = {k.replace("layers.", ""): v for k, v in state.items()}
        mlp.load_state_dict(remapped)
        self.mlp = mlp.to(device).eval()

    def __call__(self, image_path: str, prompt: str) -> float:
        from PIL import Image

        torch = self._torch
        with torch.no_grad():
            pil = Image.open(image_path).convert("RGB")
            img = self.preprocess(pil).unsqueeze(0).to(self.device)
            emb = self.clip_model.encode_image(img).float()
            emb = emb / emb.norm(dim=-1, keepdim=True)
            return float(self.mlp(emb).item())


class PickScorer:
    """PickScore_v1."""

    def __init__(self, device: str = "cpu") -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        self._torch = torch
        self.device = device
        self.processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        self.model = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").eval().to(device)

    @staticmethod
    def _pooled(out: Any) -> Any:
        return out if hasattr(out, "shape") else out.pooler_output

    def __call__(self, image_path: str, prompt: str) -> float:
        from PIL import Image

        torch = self._torch
        with torch.no_grad():
            text_in = self.processor(text=prompt, padding=True, truncation=True,
                                     max_length=77, return_tensors="pt").to(self.device)
            t = self._pooled(self.model.get_text_features(**text_in))
            t = t / t.norm(dim=-1, keepdim=True)
            img = Image.open(image_path).convert("RGB")
            img_in = self.processor(images=img, return_tensors="pt").to(self.device)
            v = self._pooled(self.model.get_image_features(**img_in))
            v = v / v.norm(dim=-1, keepdim=True)
            return float((self.model.logit_scale.exp() * (t @ v.T)).item())


def build_main_scorers(device: str = "cpu") -> dict[str, Callable[[str, str], float]]:
    return {
        "hps": HpsV2Scorer(),
        "aes": AesScorer(device=device),
        "pick": PickScorer(device=device),
    }
