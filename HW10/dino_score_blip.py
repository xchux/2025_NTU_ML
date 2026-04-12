#!/usr/bin/env python3
"""Compute DINO and CLIP-T scores for BLIP generated results.

This script compares each generated image in results_blip/object-* with the
corresponding reference image defined in ml2025-hw10/metadata.json.

Outputs:
- dino_scores_blip_per_image.csv
- dino_scores_blip_summary.csv

Notes:
- Raw cosine similarities are in [-1, 1].
- DINO homework-style score is reported as cosine * 100.
- CLIP-T homework-style score is reported as cosine * 100.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor


DEFAULT_DINO_THRESHOLDS = {
    "object-1": 68.0,
    "object-2": 60.0,
    "object-3": 61.0,
    "object-4": 68.0,
    "object-5": 60.0,
    "object-6": 57.0,
}

DEFAULT_CLIP_T_THRESHOLDS = {
    "object-1": 18.0,
    "object-2": 17.0,
    "object-3": 18.0,
    "object-4": 19.0,
    "object-5": 19.0,
    "object-6": 17.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score results_blip images with DINOv2 and CLIP-T.")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("ml2025-hw10/metadata.json"),
        help="Path to metadata.json (default: ml2025-hw10/metadata.json)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results_blip"),
        help="Directory containing object-* generated images (default: results_blip)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="facebook/dinov2-base",
        help="Hugging Face model id for DINO backbone",
    )
    parser.add_argument(
        "--clip-model-name",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="Hugging Face model id for CLIP text-image scoring",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for embedding extraction",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="Device to run model on",
    )
    parser.add_argument(
        "--per-image-out",
        type=Path,
        default=Path("dino_scores_blip_per_image.csv"),
        help="Output CSV for per-image scores",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("dino_scores_blip_summary.csv"),
        help="Output CSV for per-object summary",
    )
    parser.add_argument(
        "--clip-t-threshold",
        type=float,
        default=None,
        help="Optional global CLIP-T threshold on 0-100 scale for pass/fail",
    )
    parser.add_argument(
        "--clip-t-thresholds-json",
        type=Path,
        default=None,
        help=(
            "Optional JSON path for per-object CLIP-T thresholds, "
            'e.g. {"object-1": 18, "object-2": 17}'
        ),
    )
    return parser.parse_args()


def load_metadata(metadata_path: Path) -> Dict[str, dict]:
    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_clip_thresholds(path: Path | None) -> Dict[str, float]:
    if path is None:
        return {}

    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if not isinstance(obj, dict):
        raise ValueError("CLIP-T thresholds JSON must be an object mapping object id to float.")

    out: Dict[str, float] = {}
    for k, v in obj.items():
        out[str(k)] = float(v)
    return out


def numeric_sort_key(path: Path) -> Tuple[int, str]:
    stem = path.stem
    if stem.isdigit():
        return int(stem), path.name
    return 10**9, path.name


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def get_embeddings(
    model: AutoModel,
    processor: AutoImageProcessor,
    images: List[Image.Image],
    device: str,
    batch_size: int,
) -> torch.Tensor:
    outputs: List[torch.Tensor] = []
    model.eval()

    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch_imgs = images[i : i + batch_size]
            inputs = processor(images=batch_imgs, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            model_out = model(**inputs)
            # CLS embedding is a stable global image representation for DINO models.
            cls_emb = model_out.last_hidden_state[:, 0, :]
            cls_emb = torch.nn.functional.normalize(cls_emb, p=2, dim=-1)
            outputs.append(cls_emb.cpu())

    return torch.cat(outputs, dim=0)


def cosine_scores(ref_emb: torch.Tensor, gen_embs: torch.Tensor) -> torch.Tensor:
    # Embeddings are already L2-normalized, so cosine similarity is a dot product.
    ref_vec = ref_emb.squeeze(0)
    return torch.matmul(gen_embs, ref_vec)


def _to_feature_tensor(output: object) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output

    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output

    if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
        return output.last_hidden_state[:, 0, :]

    raise TypeError(f"Cannot convert output type to feature tensor: {type(output)}")


def get_clip_text_embedding(
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    text: str,
    device: str,
) -> torch.Tensor:
    with torch.no_grad():
        text_inputs = clip_processor(text=[text], return_tensors="pt", padding=True)
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        text_out = clip_model.get_text_features(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs.get("attention_mask"),
        )
        text_emb = _to_feature_tensor(text_out)
        text_emb = torch.nn.functional.normalize(text_emb, p=2, dim=-1)
    return text_emb.cpu()


def get_clip_image_embeddings(
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    images: List[Image.Image],
    device: str,
    batch_size: int,
) -> torch.Tensor:
    outputs: List[torch.Tensor] = []
    clip_model.eval()

    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch_imgs = images[i : i + batch_size]
            image_inputs = clip_processor(images=batch_imgs, return_tensors="pt")
            image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
            image_out = clip_model.get_image_features(pixel_values=image_inputs["pixel_values"])
            image_emb = _to_feature_tensor(image_out)
            image_emb = torch.nn.functional.normalize(image_emb, p=2, dim=-1)
            outputs.append(image_emb.cpu())

    return torch.cat(outputs, dim=0)


def main() -> None:
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    if not args.metadata.exists():
        raise FileNotFoundError(f"Metadata file not found: {args.metadata}")
    if not args.results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {args.results_dir}")

    metadata = load_metadata(args.metadata)
    clip_thresholds = load_clip_thresholds(args.clip_t_thresholds_json)

    processor = AutoImageProcessor.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(args.device)
    clip_processor = CLIPProcessor.from_pretrained(args.clip_model_name)
    clip_model = CLIPModel.from_pretrained(args.clip_model_name).to(args.device)

    per_image_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    for obj_key in sorted(metadata.keys()):
        info = metadata[obj_key]
        ref_path = Path(info["path"])
        gen_dir = args.results_dir / obj_key

        if not ref_path.exists():
            print(f"[Skip] Missing reference image: {ref_path}")
            continue
        if not gen_dir.exists():
            print(f"[Skip] Missing generated folder: {gen_dir}")
            continue

        gen_paths = sorted(gen_dir.glob("*.jpg"), key=numeric_sort_key)
        if not gen_paths:
            print(f"[Skip] No generated images in: {gen_dir}")
            continue

        ref_image = load_rgb(ref_path)
        gen_images = [load_rgb(p) for p in gen_paths]

        ref_emb = get_embeddings(model, processor, [ref_image], args.device, batch_size=1)
        gen_embs = get_embeddings(model, processor, gen_images, args.device, batch_size=args.batch_size)

        dino_scores_raw = cosine_scores(ref_emb, gen_embs)
        dino_scores = dino_scores_raw * 100.0

        text_cond = info.get("text_cond", "")
        if not text_cond:
            print(f"[Warn] Empty text_cond for {obj_key}; CLIP-T scores may be less meaningful.")
        clip_text_emb = get_clip_text_embedding(clip_model, clip_processor, text_cond, args.device)
        clip_img_embs = get_clip_image_embeddings(
            clip_model,
            clip_processor,
            gen_images,
            args.device,
            batch_size=args.batch_size,
        )
        clip_t_scores_raw = cosine_scores(clip_text_emb, clip_img_embs)
        clip_t_scores = clip_t_scores_raw * 100.0

        for p, dino_s, clip_s in zip(gen_paths, dino_scores.tolist(), clip_t_scores.tolist()):
            per_image_rows.append(
                {
                    "object": obj_key,
                    "image_path": str(p),
                    "dino_score": float(dino_s),
                    "clip_t_score": float(clip_s),
                    "text_cond": text_cond,
                    "reference_path": str(ref_path),
                }
            )

        dino_mean = float(dino_scores.mean().item())
        dino_std = float(dino_scores.std(unbiased=False).item())
        dino_min = float(dino_scores.min().item())
        dino_max = float(dino_scores.max().item())

        clip_t_mean = float(clip_t_scores.mean().item())
        clip_t_std = float(clip_t_scores.std(unbiased=False).item())
        clip_t_min = float(clip_t_scores.min().item())
        clip_t_max = float(clip_t_scores.max().item())

        dino_threshold = float(DEFAULT_DINO_THRESHOLDS.get(obj_key, float("nan")))
        dino_pass = bool(dino_mean >= dino_threshold) if obj_key in DEFAULT_DINO_THRESHOLDS else None

        clip_t_threshold = None
        if obj_key in clip_thresholds:
            clip_t_threshold = float(clip_thresholds[obj_key])
        elif args.clip_t_threshold is not None:
            clip_t_threshold = float(args.clip_t_threshold)
        elif obj_key in DEFAULT_CLIP_T_THRESHOLDS:
            clip_t_threshold = float(DEFAULT_CLIP_T_THRESHOLDS[obj_key])

        clip_t_pass = None if clip_t_threshold is None else bool(clip_t_mean >= clip_t_threshold)
        overall_pass = None
        if dino_pass is not None and clip_t_pass is not None:
            overall_pass = bool(dino_pass and clip_t_pass)

        summary_rows.append(
            {
                "object": obj_key,
                "num_images": len(gen_paths),
                "mean_dino_score": dino_mean,
                "std_dino_score": dino_std,
                "min_dino_score": dino_min,
                "max_dino_score": dino_max,
                "mean_clip_t_score": clip_t_mean,
                "std_clip_t_score": clip_t_std,
                "min_clip_t_score": clip_t_min,
                "max_clip_t_score": clip_t_max,
                "dino_threshold": dino_threshold,
                "dino_pass": dino_pass,
                "clip_t_threshold": clip_t_threshold,
                "clip_t_pass": clip_t_pass,
                "overall_pass": overall_pass,
                "text_cond": text_cond,
                "reference_path": str(ref_path),
                "generated_dir": str(gen_dir),
            }
        )

        dino_pass_str = "N/A" if dino_pass is None else ("PASS" if dino_pass else "FAIL")
        clip_pass_str = "N/A" if clip_t_pass is None else ("PASS" if clip_t_pass else "FAIL")
        overall_pass_str = "N/A" if overall_pass is None else ("PASS" if overall_pass else "FAIL")

        print(
            f"[{obj_key}] n={len(gen_paths)} "
            f"DINO(mean={dino_mean:.4f}, std={dino_std:.4f}, min={dino_min:.4f}, max={dino_max:.4f}) "
            f"CLIP-T(mean={clip_t_mean:.4f}, std={clip_t_std:.4f}, min={clip_t_min:.4f}, max={clip_t_max:.4f}) "
            f"DINO[{dino_pass_str}] CLIP-T[{clip_pass_str}] OVERALL[{overall_pass_str}]"
        )

    if not summary_rows:
        raise RuntimeError("No valid objects were scored. Check metadata and results directory.")

    overall_mean = sum(row["mean_dino_score"] for row in summary_rows) / len(summary_rows)
    overall_clip_t_mean = sum(row["mean_clip_t_score"] for row in summary_rows) / len(summary_rows)
    print(f"\nOverall mean DINO score across objects: {overall_mean:.4f}")
    print(f"Overall mean CLIP-T score across objects: {overall_clip_t_mean:.4f}")
    print("CLIP-T threshold scale: 0-100 (cosine similarity * 100).")

    with args.per_image_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["object", "image_path", "dino_score", "clip_t_score", "text_cond", "reference_path"],
        )
        writer.writeheader()
        writer.writerows(per_image_rows)

    with args.summary_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "object",
                "num_images",
                "mean_dino_score",
                "std_dino_score",
                "min_dino_score",
                "max_dino_score",
                "mean_clip_t_score",
                "std_clip_t_score",
                "min_clip_t_score",
                "max_clip_t_score",
                "dino_threshold",
                "dino_pass",
                "clip_t_threshold",
                "clip_t_pass",
                "overall_pass",
                "text_cond",
                "reference_path",
                "generated_dir",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nSaved per-image scores: {args.per_image_out}")
    print(f"Saved summary scores: {args.summary_out}")


if __name__ == "__main__":
    main()
