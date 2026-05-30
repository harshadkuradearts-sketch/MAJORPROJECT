#!/usr/bin/env python3
"""
CLI entrypoint for training any supported classifier architecture.
"""

from __future__ import annotations

import argparse
import logging

from config import ARCHITECTURES, default_config
from trainer import run_training

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("train")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train dragon fruit disease classifier")
    parser.add_argument(
        "--architecture",
        type=str,
        choices=ARCHITECTURES,
        default="attention",
        help="Model architecture to train",
    )
    parser.add_argument("--data_root", type=str, default=None, help="Root folder with class subdirs")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision (CUDA)")
    parser.add_argument("--no_compile", action="store_true", help="Disable torch.compile")
    parser.add_argument(
        "--run_name",
        type=str,
        default="run_001",
        help="Subfolder under checkpoints/<architecture>/ and logs/<architecture>/",
    )
    args = parser.parse_args()

    cfg = default_config()
    cfg.architecture = args.architecture
    if args.data_root:
        cfg.data.data_root = args.data_root
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    else:
        cfg.train.batch_size = 8
    if args.lr is not None:
        cfg.train.learning_rate = args.lr
    if args.seed is not None:
        cfg.data.random_seed = args.seed

    run_training(
        cfg,
        args.architecture,
        args.run_name,
        resume=args.resume,
        no_amp=args.no_amp,
        no_compile=args.no_compile,
    )


if __name__ == "__main__":
    main()
