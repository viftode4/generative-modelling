"""Reusable image-quality / preference scorers for Experiment #6.

Four metrics: HPS v2, AES (LAION aesthetic predictor), PickScore, ImageReward.
ImageReward is NOT here — it needs transformers v4 and runs in an isolated
environment (see scripts/score_guidance_gap_imagereward.py). This module covers
the three metrics that run in the main transformers-v5 venv.

Heavy libraries are imported lazily inside the scorer __init__ so this module
imports with no models, no network, and stays unit-testable.
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
    """Score each manifest image with each provided scorer.

    ``scorers[name](image_path, prompt) -> float``. Metrics in ``all_metrics``
    with no scorer are written as NaN (e.g. IR when scored separately).
    Identity/provenance fields are passed through for later merge/aggregation.
    """
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


# --------------------------------------------------------------------------
# Model-backed scorers (lazy). Not unit-tested; exercised by the scoring script.
# --------------------------------------------------------------------------
def _shim_turtle_for_hpsv2() -> None:
    """Work around a packaging bug in hpsv2's vendored open_clip.

    ``hpsv2/src/open_clip/factory.py`` has a stray ``from turtle import forward``
    (the imported name is never used). ``turtle`` pulls in ``tkinter``/``_tkinter``,
    which is absent in headless / Homebrew Python builds, so the import crashes.
    If the real ``turtle`` is importable we leave it alone; otherwise we register a
    minimal stand-in so the unused import resolves.
    """
    import sys
    import types

    try:
        import turtle  # noqa: F401
    except Exception:
        stub = types.ModuleType("turtle")
        stub.forward = lambda *args, **kwargs: None
        sys.modules["turtle"] = stub


class HpsV2Scorer:
    """HPS v2.1. hpsv2.score returns a raw [0,1] score; paper reports it x100."""

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
    """LAION aesthetic predictor: OpenAI CLIP ViT-L/14 embedding -> MLP head."""

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
        # The published MLP stores weights under "layers.*"; remap to Sequential keys.
        remapped = {k.replace("layers.", ""): v for k, v in state.items()}
        mlp.load_state_dict(remapped)
        self.mlp = mlp.to(device).eval()

    def __call__(self, image_path: str, prompt: str) -> float:  # prompt unused (no-reference)
        from PIL import Image

        torch = self._torch
        with torch.no_grad():
            pil = Image.open(image_path).convert("RGB")
            img = self.preprocess(pil).unsqueeze(0).to(self.device)
            emb = self.clip_model.encode_image(img).float()
            emb = emb / emb.norm(dim=-1, keepdim=True)
            return float(self.mlp(emb).item())


class PickScorer:
    """PickScore_v1: raw CLIP logit (logit_scale * cos). Range ~21-23."""

    def __init__(self, device: str = "cpu") -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        self._torch = torch
        self.device = device
        self.processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        self.model = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").eval().to(device)

    @staticmethod
    def _pooled(out: Any) -> Any:
        # transformers v5 returns BaseModelOutputWithPooling instead of a tensor.
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
    """Instantiate the three main-env scorers (HPS, AES, PickScore)."""
    return {
        "hps": HpsV2Scorer(),
        "aes": AesScorer(device=device),
        "pick": PickScorer(device=device),
    }
