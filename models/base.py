"""
Shared training helpers for all classifier architectures.
"""

from __future__ import annotations

from typing import Dict, List

import torch.nn as nn


def set_encoder_trainable(model: nn.Module, architecture: str, trainable: bool) -> None:
    """Freeze or unfreeze the pretrained encoder; keep classification head trainable when frozen."""
    if architecture == "attention":
        if not hasattr(model, "backbone"):
            raise AttributeError(f"{type(model).__name__} has no backbone attribute")
        for p in model.backbone.parameters():
            p.requires_grad = trainable
        return

    if not hasattr(model, "backbone"):
        raise AttributeError(f"{type(model).__name__} has no backbone attribute")

    for name, p in model.backbone.named_parameters():
        if "classifier" in name or name.startswith("head."):
            p.requires_grad = True
        else:
            p.requires_grad = trainable


def param_groups_lrd(
    model: nn.Module,
    architecture: str,
    lr: float,
    encoder_lr_multiplier: float,
    weight_decay: float,
) -> List[Dict]:
    """Lower LR for pretrained encoder, full LR for attention blocks / classification head."""
    encoder_params: List[nn.Parameter] = []
    head_params: List[nn.Parameter] = []

    if architecture == "attention":
        for name, p in model.named_parameters():
            if name.startswith("backbone."):
                encoder_params.append(p)
            else:
                head_params.append(p)
    else:
        for name, p in model.named_parameters():
            if "classifier" in name or name.startswith("head."):
                head_params.append(p)
            else:
                encoder_params.append(p)

    return [
        {
            "params": encoder_params,
            "lr": lr * encoder_lr_multiplier,
            "weight_decay": weight_decay,
        },
        {"params": head_params, "lr": lr, "weight_decay": weight_decay},
    ]


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())
