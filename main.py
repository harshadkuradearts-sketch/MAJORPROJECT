#!/usr/bin/env python3
"""
Convenience entrypoint for training and single-image inference.

Examples:

    python main.py --mode train --architecture attention --data_root ./data --epochs 30 --run_name experiment
    python main.py --mode train --architecture efficientnet --data_root ./data --run_name exp1
    python main.py --mode infer --checkpoint ./checkpoints/attention/experiment/best_model.pt --image ./sample.jpg
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import shutil
from collections import Counter
from pathlib import Path

from config import ARCHITECTURES, DataConfig
from dataset import build_manifest


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("launcher")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run training or inference from one launcher")
    parser.add_argument("--mode", type=str, choices=("train", "infer", "test"), default="train")
    parser.add_argument(
        "--architecture",
        type=str,
        choices=ARCHITECTURES,
        default="attention",
        help="Model architecture (efficientnet | attention | vit)",
    )
    parser.add_argument("--data_root", type=str, default="./data", help="Root folder with class subdirs")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--run_name", type=str, default="experiment")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision (CUDA)")
    parser.add_argument("--no_compile", action="store_true", help="Disable torch.compile")
    parser.add_argument("--skip_dataset_audit", action="store_true", help="Skip the dataset audit step")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path for inference")
    parser.add_argument("--image", type=str, default=None, help="Image path for inference")
    parser.add_argument("--top_k", type=int, default=5, help="Top-k probabilities to display in inference")
    parser.add_argument("--meta", type=str, default=None, help="Optional run_meta.json for inference")
    return parser.parse_args()


def run_dataset_audit(data_root: str) -> None:
    data_cfg = DataConfig(data_root=data_root)
    records, class_names, _ = build_manifest(data_cfg.data_root, data_cfg)
    counts = Counter(record.class_name for record in records)

    logger.info("Dataset audit | classes=%d %s", len(class_names), class_names)
    logger.info("Dataset audit | total_images_after_cleaning=%d", len(records))
    logger.info("Dataset audit | class_counts=%s", dict(counts))


def clear_previous_training_artifacts(architecture: str, run_name: str) -> None:
    script_dir = Path(__file__).resolve().parent
    targets = [
        script_dir / "checkpoints" / architecture / run_name,
        script_dir / "logs" / architecture / run_name,
    ]

    for target in targets:
        if target.exists():
            shutil.rmtree(target)
            logger.info("Cleared previous training artifacts: %s", target)


def run_training(args: argparse.Namespace) -> None:
    script_dir = Path(__file__).resolve().parent
    train_script = script_dir / "train.py"

    cmd = [
        sys.executable,
        str(train_script),
        "--architecture",
        args.architecture,
        "--data_root",
        args.data_root,
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--run_name",
        args.run_name,
    ]
    if args.lr is not None:
        cmd.extend(["--lr", str(args.lr)])
    if args.resume is not None:
        cmd.extend(["--resume", args.resume])
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])
    if args.no_amp:
        cmd.append("--no_amp")
    if args.no_compile:
        cmd.append("--no_compile")

    subprocess.run(cmd, check=True)


def run_inference(args: argparse.Namespace) -> None:
    if not args.checkpoint:
        raise ValueError("--checkpoint is required when --mode infer/test")
    if not args.image:
        raise ValueError("--image is required when --mode infer/test")

    script_dir = Path(__file__).resolve().parent
    inference_script = script_dir / "inference.py"

    cmd = [
        sys.executable,
        str(inference_script),
        "--checkpoint",
        args.checkpoint,
        "--image",
        args.image,
        "--top_k",
        str(args.top_k),
    ]
    if args.architecture:
        cmd.extend(["--architecture", args.architecture])
    if args.meta is not None:
        cmd.extend(["--meta", args.meta])
    if args.no_compile:
        cmd.append("--no_compile")

    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()

    if args.mode == "train":
        if args.resume is None:
            clear_previous_training_artifacts(args.architecture, args.run_name)
        else:
            logger.info(
                "Resume requested; preserving existing artifacts for %s/%s",
                args.architecture,
                args.run_name,
            )

        if not args.skip_dataset_audit:
            run_dataset_audit(args.data_root)

        run_training(args)
        return

    if args.mode in {"infer", "test"}:
        run_inference(args)
        return


if __name__ == "__main__":
    main()
