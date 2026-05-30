"""
Backward-compatible shim for the attention model.

Prefer: from models import build_model
"""

from __future__ import annotations

from typing import List, Dict

import torch.nn as nn

from models.attention import (
    EfficientNetMHSA,
    build_attention_model,
)
from models.base import count_parameters
from models.base import param_groups_lrd as _param_groups_lrd
from models.base import set_encoder_trainable as _set_encoder_trainable

# Legacy alias: original API built only the attention model.
build_model = build_attention_model


def set_backbone_trainable(model: EfficientNetMHSA, trainable: bool) -> None:
    _set_encoder_trainable(model, "attention", trainable)


def param_groups_lrd(
    model: EfficientNetMHSA,
    lr: float,
    backbone_lr_multiplier: float,
    weight_decay: float,
) -> List[Dict]:
    return _param_groups_lrd(
        model,
        "attention",
        lr,
        backbone_lr_multiplier,
        weight_decay,
    )


__all__ = [
    "EfficientNetMHSA",
    "build_model",
    "build_attention_model",
    "count_parameters",
    "param_groups_lrd",
    "set_backbone_trainable",
]
