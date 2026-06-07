"""Run the GenEval object-detector evaluator on Vlad's standard-vs-Z pairs.

This script reads the manifest produced by ``scripts/run_vlad_geneval.py``
and scores both the standard and Z-Sampling image in each row against the
official GenEval rule for that prompt. The official rule is looked up by
``source_index`` (and prompt-string as fallback) inside the upstream
``external/geneval/prompts/evaluation_metadata.jsonl``.

The detection/color/position logic mirrors
``external/geneval/evaluation/evaluate_images.py``. It supports both the
upstream-compatible mmdet 2.x tuple/list output and modern mmdet 3.x
``DetDataSample`` output. On Vlad's Apple Silicon environment the working
path is `mmdet==2.28.2`, `mmcv-full==1.7.2`, and `--device cpu`.
Behaviour matches upstream:

* Mask2Former object detector.
* Threshold 0.3 for general tags, 0.9 for counting.
* CLIP color classifier (open_clip ``ViT-L-14`` + ``clip_benchmark.zeroshot_classification``)
  using the same three-prompt template as upstream.
* Position relations evaluated relative to the first matched group, with
  the same ``POSITION_THRESHOLD = 0.1`` and 0.5 cosine cutoff used upstream.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps

from zigzag_project.experiment_io import read_jsonl, write_jsonl
from zigzag_project.geneval_io import load_geneval_metadata

COLORS = [
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "purple",
    "pink",
    "brown",
    "black",
    "white",
]
DEFAULT_THRESHOLD = 0.3
COUNTING_THRESHOLD = 0.9
POSITION_THRESHOLD = 0.1
MAX_OBJECTS = 16
NMS_THRESHOLD = 1.0  # upstream default; no NMS

# Upstream Mask2Former checkpoint + config.
DEFAULT_CONFIG_RELATIVE = (
    "configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py"
)
DEFAULT_CHECKPOINT_URL = (
    "https://download.openmmlab.com/mmdetection/v2.0/mask2former/"
    "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/"
    "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth"
)
DEFAULT_CHECKPOINT_NAME = "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--geneval-root", default="external/geneval")
    parser.add_argument("--device", default=None, help="cuda | mps | cpu (auto if omitted)")
    parser.add_argument(
        "--checkpoint-dir",
        default=".cache/geneval",
        help="Directory used to cache the Mask2Former weights file.",
    )
    parser.add_argument("--model-config", default=None)
    parser.add_argument(
        "--clip-arch",
        default="ViT-L-14",
        help="open_clip architecture for the color classifier.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of manifest rows (for smoke tests).",
    )
    args = parser.parse_args()

    geneval_root = Path(args.geneval_root)
    metadata_path = geneval_root / "prompts" / "evaluation_metadata.jsonl"
    classnames_path = geneval_root / "evaluation" / "object_names.txt"
    if not metadata_path.exists():
        raise SystemExit(f"GenEval metadata not found at {metadata_path}")
    if not classnames_path.exists():
        raise SystemExit(f"GenEval object_names.txt not found at {classnames_path}")

    metadata_index = load_geneval_metadata(metadata_path)
    classnames = [line.strip() for line in classnames_path.read_text().splitlines() if line.strip()]

    device = _resolve_device(args.device)
    print(f"[geneval] device: {device}", file=sys.stderr)

    ckpt_path = _ensure_checkpoint(Path(args.checkpoint_dir))
    config_path = _resolve_model_config(args.model_config)

    detector = _init_detector(config_path=config_path, ckpt_path=ckpt_path, device=device)
    clip_bundle = _init_clip(arch=args.clip_arch, device=device)

    manifest_rows = read_jsonl(args.manifest)
    if args.limit is not None:
        manifest_rows = manifest_rows[: args.limit]

    score_rows: list[dict[str, Any]] = []
    t0 = time.time()
    for idx, row in enumerate(manifest_rows):
        meta = metadata_index.lookup(
            source_index=int(row["source_index"]) if "source_index" in row else None,
            prompt=str(row.get("prompt")),
        )
        if str(row["tag"]) != str(meta["tag"]):
            raise SystemExit(
                f"Tag mismatch row {idx}: manifest tag={row['tag']!r} vs upstream {meta['tag']!r}"
            )
        standard = _evaluate_one(
            row["standard_path"], meta, detector, clip_bundle, classnames, device
        )
        zigzag = _evaluate_one(
            row["zigzag_path"], meta, detector, clip_bundle, classnames, device
        )
        score_rows.append(
            {
                "prompt_id": row["prompt_id"],
                "source_index": int(row["source_index"]),
                "tag": str(row["tag"]),
                "prompt": str(row["prompt"]),
                "seed": int(row["seed"]),
                "standard_path": str(row["standard_path"]),
                "zigzag_path": str(row["zigzag_path"]),
                "standard_pass": bool(standard["correct"]),
                "zigzag_pass": bool(zigzag["correct"]),
                "standard_details": standard,
                "zigzag_details": zigzag,
            }
        )
        elapsed = time.time() - t0
        print(
            f"[geneval] row {idx + 1}/{len(manifest_rows)} "
            f"tag={row['tag']} std={int(standard['correct'])} "
            f"z={int(zigzag['correct'])} elapsed={elapsed:.1f}s",
            file=sys.stderr,
        )

    write_jsonl(args.out, score_rows)
    total = time.time() - t0
    print(f"[geneval] wrote {len(score_rows)} rows -> {args.out} in {total:.1f}s", file=sys.stderr)


# ---------------- helpers ----------------


def _resolve_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _ensure_checkpoint(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = cache_dir / DEFAULT_CHECKPOINT_NAME
    if ckpt_path.exists() and ckpt_path.stat().st_size > 0:
        return ckpt_path
    import urllib.request

    print(f"[geneval] downloading {DEFAULT_CHECKPOINT_URL} -> {ckpt_path}", file=sys.stderr)
    urllib.request.urlretrieve(DEFAULT_CHECKPOINT_URL, ckpt_path)
    return ckpt_path


def _resolve_model_config(explicit: str | None) -> str:
    if explicit:
        return explicit
    # mmdet ships its config tree under <package>/.mim/configs in pip installs,
    # and under <package>/../configs in source installs.
    import mmdet

    base = Path(mmdet.__file__).resolve().parent
    candidates = [
        base / ".mim" / DEFAULT_CONFIG_RELATIVE,
        base.parent / DEFAULT_CONFIG_RELATIVE,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise SystemExit(
        "Could not locate Mask2Former config in mmdet install. Tried: "
        + ", ".join(str(c) for c in candidates)
    )


def _init_detector(*, config_path: str, ckpt_path: Path, device: str) -> Any:
    from mmdet.apis import init_detector

    return init_detector(config_path, str(ckpt_path), device=device)


def _init_clip(*, arch: str, device: str) -> dict[str, Any]:
    import open_clip

    model, _, transform = open_clip.create_model_and_transforms(
        arch, pretrained="openai", device=device
    )
    tokenizer = open_clip.get_tokenizer(arch)
    return {
        "model": model,
        "transform": transform,
        "tokenizer": tokenizer,
        "device": device,
        "classifiers": {},
    }


def _evaluate_one(
    image_path: str,
    metadata: dict[str, Any],
    detector: Any,
    clip_bundle: dict[str, Any],
    classnames: list[str],
    device: str,
) -> dict[str, Any]:
    from mmdet.apis import inference_detector

    result = inference_detector(detector, image_path)
    detected = _extract_detections(result, classnames, metadata["tag"])
    image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    correct, reason, matched_groups = _evaluate_against_metadata(
        image, detected, metadata, clip_bundle
    )
    return {
        "filename": image_path,
        "tag": metadata["tag"],
        "prompt": metadata["prompt"],
        "correct": bool(correct),
        "reason": reason,
        "detected_counts": {cls: len(boxes) for cls, boxes in detected.items()},
        "matched_groups": [
            None if grp is None else [_box_to_list(obj[0]) for obj in grp]
            for grp in matched_groups
        ],
    }


def _extract_detections(
    result: Any, classnames: list[str], tag: str
) -> dict[str, list[tuple[np.ndarray, np.ndarray | None]]]:
    """Adapt mmdet output to the upstream per-class object list.

    GenEval's upstream script installs mmdetection 2.x, whose
    ``inference_detector`` returns either ``bbox_results`` or
    ``(bbox_results, segm_results)``. A previous local port targeted
    mmdetection 3.x ``DetDataSample`` instead. Supporting both matters on
    Apple Silicon because the locally available compiled ``mmcv-full`` wheel is
    from the mmdet-2.x era.
    """
    if hasattr(result, "pred_instances"):
        return _extract_detections_mmdet3(result, classnames, tag)
    return _extract_detections_mmdet2(result, classnames, tag)


def _extract_detections_mmdet3(
    result: Any, classnames: list[str], tag: str
) -> dict[str, list[tuple[np.ndarray, np.ndarray | None]]]:
    threshold = COUNTING_THRESHOLD if tag == "counting" else DEFAULT_THRESHOLD
    pred = result.pred_instances
    bboxes = pred.bboxes.detach().cpu().numpy()  # (N, 4)
    scores = pred.scores.detach().cpu().numpy()  # (N,)
    labels = pred.labels.detach().cpu().numpy()  # (N,)
    masks = (
        pred.masks.detach().cpu().numpy() if "masks" in pred else None
    )  # (N, H, W) bool

    detected: dict[str, list[tuple[np.ndarray, np.ndarray | None]]] = {}
    for class_idx, classname in enumerate(classnames):
        class_mask = labels == class_idx
        if not class_mask.any():
            continue
        class_scores = scores[class_mask]
        order = np.argsort(class_scores)[::-1]
        order = order[class_scores[order] > threshold]
        order = order[:MAX_OBJECTS]
        if len(order) == 0:
            continue
        sub_bboxes = bboxes[class_mask][order]
        sub_scores = class_scores[order]
        sub_masks = masks[class_mask][order] if masks is not None else None
        kept: list[tuple[np.ndarray, np.ndarray | None]] = []
        remaining = list(range(len(order)))
        while remaining:
            head = remaining.pop(0)
            head_box = np.concatenate([sub_bboxes[head], [sub_scores[head]]])
            head_mask = sub_masks[head] if sub_masks is not None else None
            kept.append((head_box, head_mask))
            if NMS_THRESHOLD < 1.0:
                remaining = [
                    j
                    for j in remaining
                    if _iou(sub_bboxes[head], sub_bboxes[j]) < NMS_THRESHOLD
                ]
        detected[classname] = kept
    return detected


def _extract_detections_mmdet2(
    result: Any, classnames: list[str], tag: str
) -> dict[str, list[tuple[np.ndarray, np.ndarray | None]]]:
    threshold = COUNTING_THRESHOLD if tag == "counting" else DEFAULT_THRESHOLD
    bbox_results = result[0] if isinstance(result, tuple) else result
    segm_results = result[1] if isinstance(result, tuple) and len(result) > 1 else None

    detected: dict[str, list[tuple[np.ndarray, np.ndarray | None]]] = {}
    for class_idx, classname in enumerate(classnames):
        class_bboxes = np.asarray(bbox_results[class_idx])
        if class_bboxes.size == 0:
            continue
        class_scores = class_bboxes[:, 4]
        order = np.argsort(class_scores)[::-1]
        order = order[class_scores[order] > threshold]
        order = order[:MAX_OBJECTS]
        if len(order) == 0:
            continue

        class_masks = None
        if segm_results is not None:
            class_masks = segm_results[class_idx]

        kept: list[tuple[np.ndarray, np.ndarray | None]] = []
        remaining = order.tolist()
        while remaining:
            head = remaining.pop(0)
            head_box = class_bboxes[head]
            head_mask = None if class_masks is None else np.asarray(class_masks[head])
            kept.append((head_box, head_mask))
            if NMS_THRESHOLD < 1.0:
                remaining = [
                    idx
                    for idx in remaining
                    if _iou(head_box[:4], class_bboxes[idx, :4]) < NMS_THRESHOLD
                ]
        detected[classname] = kept
    return detected


def _evaluate_against_metadata(
    image: Image.Image,
    detected: dict[str, list[tuple[np.ndarray, np.ndarray | None]]],
    metadata: dict[str, Any],
    clip_bundle: dict[str, Any],
) -> tuple[bool, str, list[list[tuple[np.ndarray, np.ndarray | None]] | None]]:
    correct = True
    reasons: list[str] = []
    matched_groups: list[list[tuple[np.ndarray, np.ndarray | None]] | None] = []
    for req in metadata.get("include", []):
        classname = req["class"]
        matched = True
        found = detected.get(classname, [])[: req["count"]]
        if len(found) < req["count"]:
            correct = matched = False
            reasons.append(f"expected {classname}>={req['count']}, found {len(found)}")
        else:
            if "color" in req:
                colors = _color_classification(image, found, classname, clip_bundle)
                if colors.count(req["color"]) < req["count"]:
                    correct = matched = False
                    reasons.append(
                        f"expected {req['color']} {classname}>={req['count']}, found "
                        + f"{colors.count(req['color'])} {req['color']}; and "
                        + ", ".join(f"{colors.count(c)} {c}" for c in COLORS if c in colors)
                    )
            if "position" in req and matched:
                expected_rel, target_group = req["position"]
                target = matched_groups[target_group]
                if target is None:
                    correct = matched = False
                    reasons.append(f"no target for {classname} to be {expected_rel}")
                else:
                    for obj in found:
                        for target_obj in target:
                            rels = _relative_position(obj, target_obj)
                            if expected_rel not in rels:
                                correct = matched = False
                                reasons.append(
                                    f"expected {classname} {expected_rel} target, found "
                                    + f"{' and '.join(rels) if rels else 'none'} target"
                                )
                                break
                        if not matched:
                            break
        matched_groups.append(found if matched else None)
    for req in metadata.get("exclude", []):
        classname = req["class"]
        if len(detected.get(classname, [])) >= req["count"]:
            correct = False
            reasons.append(
                f"expected {classname}<{req['count']}, found {len(detected[classname])}"
            )
    return correct, "\n".join(reasons), matched_groups


def _color_classification(
    image: Image.Image,
    objects: list[tuple[np.ndarray, np.ndarray | None]],
    classname: str,
    clip_bundle: dict[str, Any],
) -> list[str]:
    from clip_benchmark.metrics import zeroshot_classification as zsc

    classifiers = clip_bundle["classifiers"]
    if classname not in classifiers:
        classifiers[classname] = zsc.zero_shot_classifier(
            clip_bundle["model"],
            clip_bundle["tokenizer"],
            COLORS,
            [
                f"a photo of a {{c}} {classname}",
                f"a photo of a {{c}}-colored {classname}",
                "a photo of a {c} object",
            ],
            clip_bundle["device"],
        )
    clf = classifiers[classname]
    dataset = _CroppedDataset(image, objects, transform=clip_bundle["transform"])
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, num_workers=0)
    with torch.no_grad():
        pred, _ = zsc.run_classification(clip_bundle["model"], clf, loader, clip_bundle["device"])
    return [COLORS[int(idx.item())] for idx in pred.argmax(1)]


class _CroppedDataset(torch.utils.data.Dataset):
    """Composite + crop transform matching upstream's ``ImageCrops``."""

    def __init__(
        self,
        image: Image.Image,
        objects: list[tuple[np.ndarray, np.ndarray | None]],
        transform: Any,
        bgcolor: str = "#999",
    ) -> None:
        self._image = image.convert("RGB")
        self._blank = Image.new("RGB", image.size, color=bgcolor)
        self._objects = objects
        self._transform = transform

    def __len__(self) -> int:
        return len(self._objects)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        box, mask = self._objects[index]
        if mask is not None:
            mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
            image = Image.composite(self._image, self._blank, mask_img)
        else:
            image = self._image
        image = image.crop(tuple(int(v) for v in box[:4]))
        return self._transform(image), 0


def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    inter_x1 = max(box_a[0], box_b[0])
    inter_y1 = max(box_a[1], box_b[1])
    inter_x2 = min(box_a[2], box_b[2])
    inter_y2 = min(box_a[3], box_b[3])
    inter = max(inter_x2 - inter_x1 + 1, 0) * max(inter_y2 - inter_y1 + 1, 0)
    a_area = max(box_a[2] - box_a[0] + 1, 0) * max(box_a[3] - box_a[1] + 1, 0)
    b_area = max(box_b[2] - box_b[0] + 1, 0) * max(box_b[3] - box_b[1] + 1, 0)
    union = a_area + b_area - inter
    return inter / union if union else 0.0


def _relative_position(
    obj_a: tuple[np.ndarray, np.ndarray | None],
    obj_b: tuple[np.ndarray, np.ndarray | None],
) -> set[str]:
    """Mirror of upstream ``relative_position`` semantics."""
    boxes = np.array([obj_a[0], obj_b[0]])[:, :4].reshape(2, 2, 2)
    center_a, center_b = boxes.mean(axis=-2)
    dim_a, dim_b = np.abs(np.diff(boxes, axis=-2))[..., 0, :]
    offset = center_a - center_b
    revised_offset = np.maximum(
        np.abs(offset) - POSITION_THRESHOLD * (dim_a + dim_b), 0
    ) * np.sign(offset)
    if np.all(np.abs(revised_offset) < 1e-3):
        return set()
    norm = np.linalg.norm(offset)
    if norm == 0:
        return set()
    dx, dy = revised_offset / norm
    relations: set[str] = set()
    if dx < -0.5:
        relations.add("left of")
    if dx > 0.5:
        relations.add("right of")
    if dy < -0.5:
        relations.add("above")
    if dy > 0.5:
        relations.add("below")
    return relations


def _box_to_list(box: np.ndarray) -> list[float]:
    return [float(v) for v in box.tolist()]


if __name__ == "__main__":
    main()
