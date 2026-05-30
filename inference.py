#!/usr/bin/env python3
"""
Inference: load a trained checkpoint, preprocess one image, output class + probabilities.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from config import ARCHITECTURES, merge_model_config
from dataset import build_eval_transforms
from models import build_model
from utils import (
    configure_torch_runtime,
    format_parameter_summary,
    get_best_device,
    load_model_weights,
    maybe_compile_model,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("inference")


def get_device() -> torch.device:
    return get_best_device()


def load_checkpoint_meta(path: str, map_location: torch.device) -> Dict:
    ckpt = torch.load(path, map_location=map_location)
    if "model_state_dict" not in ckpt:
        raise ValueError(
            f"Checkpoint {path} missing 'model_state_dict'. Use best_model.pt from training."
        )
    return ckpt


def preprocess_image(
    image_path: Path,
    transform,
    device: torch.device,
) -> torch.Tensor:
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    t = transform(img)
    return t.unsqueeze(0).to(device)


def predict_one(
    model: torch.nn.Module,
    batch: torch.Tensor,
    class_names: List[str],
) -> Tuple[str, float, np.ndarray, int]:
    model.eval()
    with torch.inference_mode():
        logits = model(batch)
        probs = F.softmax(logits, dim=1)
        confidence, pred_idx = probs.max(dim=1)
    idx = int(pred_idx.item())
    label = class_names[idx]
    conf = float(confidence.item())
    prob_vec = probs.cpu().numpy()[0]
    return label, conf, prob_vec, idx


def format_probabilities(class_names: List[str], prob_vec: np.ndarray, top_k: int = 5) -> str:
    pairs = sorted(zip(class_names, prob_vec), key=lambda x: -x[1])[:top_k]
    lines = [f"  {name}: {p:.4f}" for name, p in pairs]
    return "\n".join(lines)


def resolve_architecture(
    cli_architecture: str | None,
    ckpt_meta: Dict,
) -> str:
    if cli_architecture is not None:
        return cli_architecture
    arch = ckpt_meta.get("architecture")
    if arch:
        return str(arch)
    return "attention"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference on one dragon fruit disease image")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pt")
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument(
        "--architecture",
        type=str,
        choices=ARCHITECTURES,
        default=None,
        help="Model architecture (auto-detected from checkpoint meta if omitted)",
    )
    parser.add_argument(
        "--meta",
        type=str,
        default=None,
        help="Optional run_meta.json (uses data_config.image_size if present)",
    )
    parser.add_argument("--no_compile", action="store_true", help="Disable torch.compile")
    args = parser.parse_args()

    device = get_device()
    configure_torch_runtime(device)

    ckpt = load_checkpoint_meta(args.checkpoint, map_location=device)
    ckpt_meta = ckpt.get("meta", {}) if isinstance(ckpt.get("meta"), dict) else {}

    architecture = resolve_architecture(args.architecture, ckpt_meta)

    class_names: List[str] = ckpt.get("class_names") or ckpt_meta.get("class_names")
    if not class_names:
        raise ValueError("Checkpoint missing class_names in meta.")

    num_classes = int(ckpt.get("num_classes") or ckpt_meta.get("num_classes"))
    image_size = int(ckpt.get("image_size") or ckpt_meta.get("image_size", 224))
    model_cfg = merge_model_config(
        architecture,
        ckpt.get("model_config") or ckpt_meta.get("model_config") or {},
    )

    if args.meta:
        with open(args.meta, encoding="utf-8") as f:
            meta = json.load(f)
        image_size = int(meta.get("data_config", {}).get("image_size", image_size))
        if args.architecture is None and meta.get("architecture"):
            architecture = str(meta["architecture"])
            model_cfg = merge_model_config(architecture, meta.get("model_config", {}))

    transform = build_eval_transforms(image_size)
    model = build_model(architecture, num_classes, model_cfg, device=device)
    load_model_weights(model, ckpt["model_state_dict"])
    model = maybe_compile_model(model, device, enabled=not args.no_compile)

    logger.info("Architecture: %s", architecture)
    logger.info("Model parameters before testing | %s", format_parameter_summary(model))

    img_path = Path(args.image)
    batch = preprocess_image(img_path, transform, device)

    label, conf, prob_vec, pred_idx = predict_one(model, batch, class_names)

    logger.info("Image: %s", img_path)
    logger.info("Predicted class: %s (index %s)", label, pred_idx)
    logger.info("Confidence (top-1 probability): %.4f", conf)
    logger.info("Top-%s probabilities:\n%s", args.top_k, format_probabilities(class_names, prob_vec, args.top_k))


if __name__ == "__main__":
    main()
