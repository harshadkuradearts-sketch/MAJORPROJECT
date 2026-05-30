#!/usr/bin/env python3
"""
Shared training pipeline for all classifier architectures.

Features: mixed precision (CUDA), AdamW with layer-wise LR decay, cosine schedule with
linear warmup, early stopping, TensorBoard, tqdm, stratified dataloaders from dataset.py.
"""

from __future__ import annotations

import logging
import math
import os
import time
from contextlib import nullcontext
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import ARCHITECTURES, Config, model_config_for, model_config_to_dict
from dataset import prepare_data_pipeline
from models import build_model, param_groups_lrd, set_encoder_trainable
from utils import (
    LiveTrainingMonitor,
    compute_metrics,
    configure_torch_runtime,
    confusion_matrix_numpy,
    format_parameter_summary,
    get_best_device,
    load_checkpoint,
    load_model_weights,
    maybe_compile_model,
    plot_confusion_matrix,
    save_checkpoint,
    save_run_meta,
    set_seed,
    summarize_model_parameters,
    unwrap_compiled_model,
)

logger = logging.getLogger("trainer")


def _host_rss_gb() -> float:
    try:
        import psutil  # type: ignore[import-untyped]

        return psutil.Process(os.getpid()).memory_info().rss / (1024**3)
    except (ImportError, OSError, RuntimeError):
        return float("nan")


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: Optional[GradScaler],
    device: torch.device,
    max_grad_norm: float,
    use_amp: bool,
    *,
    log_first_batch_stats: bool = False,
) -> float:
    model.train()
    losses: List[float] = []
    amp_ctx = autocast if (use_amp and device.type == "cuda") else nullcontext

    t_fetch = time.perf_counter()
    for batch_idx, (xb, yb) in enumerate(tqdm(loader, desc="Train", leave=False)):
        t_after_fetch = time.perf_counter()
        fetch_s = t_after_fetch - t_fetch

        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with amp_ctx():
            logits = model(xb)
            loss = criterion(logits, yb)

        if scaler is not None and device.type == "cuda":
            scaler.scale(loss).backward()
            if max_grad_norm > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        scheduler.step()
        losses.append(loss.detach().item())

        if log_first_batch_stats and batch_idx == 0:
            parts = [
                f"first_train_batch: dataloader_fetch_s={fetch_s:.3f}",
                f"batches_per_epoch={len(loader)}",
                f"x_shape={tuple(xb.shape)}",
                f"y_shape={tuple(yb.shape)}",
                f"dtype={xb.dtype}",
                f"rss_GB={_host_rss_gb():.3f}",
            ]
            if device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
                parts.append(f"cuda_alloc_MB={torch.cuda.memory_allocated() / (1024**2):.1f}")
                parts.append(f"cuda_reserved_MB={torch.cuda.memory_reserved() / (1024**2):.1f}")
            logger.info(" | ".join(parts))

        del logits, loss, xb, yb
        t_fetch = time.perf_counter()

    return float(np.mean(losses))


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    use_amp: bool,
    desc: str = "Eval",
) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    losses: List[float] = []
    preds_all: List[torch.Tensor] = []
    labels_all: List[torch.Tensor] = []
    amp_ctx = autocast if (use_amp and device.type == "cuda") else nullcontext

    for xb, yb in tqdm(loader, desc=desc, leave=False):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        with amp_ctx():
            logits = model(xb)
            loss = criterion(logits, yb)
        losses.append(loss.item())
        preds_all.append(logits.argmax(dim=1).detach().cpu())
        labels_all.append(yb.detach().cpu())
        del logits, loss, xb, yb

    y_pred = torch.cat(preds_all).numpy()
    y_true = torch.cat(labels_all).numpy()
    metrics = compute_metrics(y_true, y_pred, num_classes)
    metrics["loss"] = float(np.mean(losses))
    return metrics["loss"], metrics, y_true, y_pred


def build_losses(
    cfg: Config,
    class_weights: Optional[torch.Tensor],
    device: torch.device,
) -> Tuple[nn.CrossEntropyLoss, nn.CrossEntropyLoss]:
    use_sampler = cfg.train.use_weighted_sampler
    w_train = None
    w_eval = None
    if cfg.train.use_class_weights and not use_sampler:
        w_train = class_weights.to(device) if class_weights is not None else None
        w_eval = w_train
        logger.info("Using inverse-frequency class weights in loss (no weighted sampler).")
    elif cfg.train.use_class_weights and use_sampler:
        logger.info(
            "Weighted sampler enabled; class weights disabled in loss to avoid double emphasis."
        )

    criterion_train = nn.CrossEntropyLoss(
        weight=w_train,
        label_smoothing=cfg.train.label_smoothing,
    )
    criterion_eval = nn.CrossEntropyLoss(weight=w_eval)
    return criterion_train, criterion_eval


def run_training(
    cfg: Config,
    architecture: str,
    run_name: str,
    *,
    resume: Optional[str] = None,
    no_amp: bool = False,
    no_compile: bool = False,
) -> None:
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture {architecture!r}. Choose from {ARCHITECTURES}")

    seed = cfg.data.random_seed
    set_seed(seed)

    device = get_best_device()
    configure_torch_runtime(device)
    use_amp = cfg.train.amp and not no_amp and device.type == "cuda"

    if cfg.train.use_weighted_sampler and cfg.train.use_class_weights:
        logger.info(
            "Mutual exclusion: disabling use_class_weights because WeightedRandomSampler is enabled."
        )
    if cfg.train.use_weighted_sampler:
        cfg.train.use_class_weights = False

    model_cfg = model_config_for(architecture)

    logger.info(
        "Pre-processing stage started | architecture=%s root=%s",
        architecture,
        cfg.data.data_root,
    )
    train_loader, val_loader, test_loader, extra = prepare_data_pipeline(
        data_root=cfg.data.data_root,
        data_cfg=cfg.data,
        batch_size=cfg.train.batch_size,
        use_weighted_sampler=cfg.train.use_weighted_sampler,
        seed=seed,
        num_workers=2,
        persistent_workers=False,
        eval_batch_size=cfg.train.batch_size,
    )
    logger.info(
        "Pre-processing stage complete | classes=%s train=%s val=%s test=%s",
        len(extra["class_names"]),
        len(train_loader.dataset),
        len(val_loader.dataset),
        len(test_loader.dataset),
    )

    num_classes: int = extra["num_classes"]
    class_names: List[str] = extra["class_names"]
    model_cfg.num_classes = num_classes

    model = build_model(architecture, num_classes, model_cfg, device=device)
    model = maybe_compile_model(model, device, enabled=not no_compile)

    param_stats = summarize_model_parameters(model)
    logger.info("Model parameters before training | %s", format_parameter_summary(model))
    logger.info(
        "Run setup | architecture=%s device=%s amp=%s compiled=%s",
        architecture,
        device,
        use_amp,
        not no_compile,
    )

    criterion_train, criterion_eval = build_losses(cfg, extra["class_weights"], device)

    base_model = unwrap_compiled_model(model)
    param_groups = param_groups_lrd(
        base_model,
        architecture,
        lr=cfg.train.learning_rate,
        encoder_lr_multiplier=cfg.train.backbone_lr_multiplier,
        weight_decay=cfg.train.weight_decay,
    )
    optimizer = AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)

    steps_per_epoch = max(len(train_loader), 1)
    total_steps = cfg.train.epochs * steps_per_epoch
    warmup_steps = max(1, cfg.train.warmup_epochs * steps_per_epoch)
    warmup_start = cfg.train.warmup_start_factor

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return warmup_start + (1.0 - warmup_start) * float(step) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    scaler: Optional[GradScaler] = GradScaler() if use_amp else None

    ckpt_dir = os.path.join(cfg.train.checkpoint_dir, architecture, run_name)
    log_dir = os.path.join(cfg.train.log_dir, architecture, run_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    save_run_meta(
        os.path.join(ckpt_dir, "run_meta.json"),
        {
            "architecture": architecture,
            "class_names": class_names,
            "num_classes": num_classes,
            "model_config": model_config_to_dict(model_cfg),
            "train_config": asdict(cfg.train),
            "data_config": asdict(cfg.data),
            "seed": seed,
            "device": str(device),
            "torch_version": torch.__version__,
            "compiled": not no_compile,
            "parameter_summary": param_stats,
        },
    )

    writer = SummaryWriter(log_dir=log_dir)
    writer.add_text(
        "run/summary",
        (
            f"architecture={architecture} device={device} amp={use_amp} "
            f"compiled={not no_compile} params={param_stats['trainable']:,}"
        ),
    )
    monitor = LiveTrainingMonitor(ckpt_dir, title=f"Training progress - {architecture}/{run_name}")

    start_epoch = 0
    best_val_loss = float("inf")
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_precision_macro": [],
        "val_recall_macro": [],
        "val_f1_macro": [],
    }

    if resume:
        ckpt = load_checkpoint(resume, map_location=device)
        load_model_weights(model, ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if scaler is not None and ckpt.get("scaler_state_dict"):
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt.get("epoch", -1) + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        if ckpt.get("history"):
            history = ckpt["history"]
        logger.info("Resumed from epoch %s, best_val_loss=%s", start_epoch, best_val_loss)

    patience_left = cfg.train.early_stopping_patience

    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(start_epoch, cfg.train.epochs):
        if cfg.train.freeze_backbone_epochs > 0:
            freeze = epoch < cfg.train.freeze_backbone_epochs
            set_encoder_trainable(base_model, architecture, not freeze)
            if epoch == 0 or epoch == cfg.train.freeze_backbone_epochs:
                logger.info("Encoder trainable=%s", not freeze)

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion_train,
            optimizer,
            scheduler,
            scaler,
            device,
            cfg.train.gradient_clip_max_norm,
            use_amp,
            log_first_batch_stats=False,
        )

        val_loss, val_metrics, _, _ = evaluate_model(
            model,
            val_loader,
            criterion_eval,
            device,
            num_classes,
            use_amp,
            desc="Val",
        )

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("acc/val", val_metrics["accuracy"], epoch)
        writer.add_scalar("precision_macro/val", val_metrics["precision_macro"], epoch)
        writer.add_scalar("recall_macro/val", val_metrics["recall_macro"], epoch)
        writer.add_scalar("f1_macro/val", val_metrics["f1_macro"], epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
        writer.flush()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_precision_macro"].append(val_metrics["precision_macro"])
        history["val_recall_macro"].append(val_metrics["recall_macro"])
        history["val_f1_macro"].append(val_metrics["f1_macro"])

        logger.info(
            "Epoch %s | train_loss=%.4f val_loss=%.4f val_acc=%.4f val_f1_macro=%.4f",
            epoch,
            train_loss,
            val_loss,
            val_metrics["accuracy"],
            val_metrics["f1_macro"],
        )
        monitor.update(
            epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_metrics=val_metrics,
            lr=optimizer.param_groups[0]["lr"],
        )

        improved = best_val_loss - val_loss > cfg.train.min_delta
        if improved:
            best_val_loss = val_loss
            patience_left = cfg.train.early_stopping_patience
            best_path = os.path.join(ckpt_dir, "best_model.pt")
            save_checkpoint(
                best_path,
                epoch=epoch,
                model=model,
                optimizer=None,
                scheduler=None,
                scaler=None,
                best_val_loss=best_val_loss,
                meta={
                    "architecture": architecture,
                    "class_names": class_names,
                    "num_classes": num_classes,
                    "model_config": model_config_to_dict(model_cfg),
                    "image_size": cfg.data.image_size,
                    "parameter_summary": param_stats,
                    "compiled": not no_compile,
                },
            )
            save_checkpoint(
                os.path.join(ckpt_dir, "checkpoint.pt"),
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                best_val_loss=best_val_loss,
                history=history,
                meta={
                    "architecture": architecture,
                    "class_names": class_names,
                    "num_classes": num_classes,
                },
            )
            logger.info("Saved new best model (val_loss=%.6f) -> %s", best_val_loss, best_path)
        else:
            patience_left -= 1
            logger.info("No val improvement (%s epochs left before stop)", patience_left)

        if patience_left <= 0:
            logger.info("Early stopping.")
            break

    writer.close()
    monitor.finalize()

    best_ckpt_path = os.path.join(ckpt_dir, "best_model.pt")
    if os.path.isfile(best_ckpt_path):
        best = torch.load(best_ckpt_path, map_location=device)
        load_model_weights(model, best["model_state_dict"])
        logger.info("Loaded best checkpoint for test evaluation.")

    logger.info("Model parameters before testing | %s", format_parameter_summary(model))

    test_loss, test_metrics, y_true, y_pred = evaluate_model(
        model,
        test_loader,
        criterion_eval,
        device,
        num_classes,
        use_amp,
        desc="Test",
    )

    logger.info(
        "Test | loss=%.4f acc=%.4f precision_macro=%.4f recall_macro=%.4f f1_macro=%.4f",
        test_loss,
        test_metrics["accuracy"],
        test_metrics["precision_macro"],
        test_metrics["recall_macro"],
        test_metrics["f1_macro"],
    )

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    per_class_f1 = {
        name: float(report[name]["f1-score"])
        for name in class_names
        if name in report
    }
    logger.info(
        "Per-class F1 | %s",
        ", ".join(f"{name}={score:.3f}" for name, score in per_class_f1.items()),
    )

    cm = confusion_matrix_numpy(y_true, y_pred, num_classes)
    plot_confusion_matrix(
        cm,
        class_names,
        os.path.join(ckpt_dir, "confusion_matrix_test.png"),
        normalize=False,
    )
    plot_confusion_matrix(
        cm,
        class_names,
        os.path.join(ckpt_dir, "confusion_matrix_test_normalized.png"),
        normalize=True,
    )

    save_run_meta(
        os.path.join(ckpt_dir, "test_metrics.json"),
        {
            "architecture": architecture,
            "loss": test_loss,
            **{k: v for k, v in test_metrics.items()},
            "classification_report": report,
            "per_class_f1": per_class_f1,
        },
    )
