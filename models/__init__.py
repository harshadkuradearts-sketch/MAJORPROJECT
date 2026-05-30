"""
Model registry for multi-architecture dragon fruit disease classification.
"""

from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn

from config import (
    ARCHITECTURES,
    AttentionModelConfig,
    EfficientNetConfig,
    ViTConfig,
    model_config_for,
)
from models.attention import EfficientNetMHSA, build_attention_model
from models.base import count_parameters, param_groups_lrd, set_encoder_trainable
from models.efficientnet import EfficientNetClassifier, build_efficientnet_model
from models.vit import ViTClassifier, build_vit_model

ModelConfigType = Union[EfficientNetConfig, AttentionModelConfig, ViTConfig]


def build_model(
    architecture: str,
    num_classes: int,
    model_cfg: Optional[ModelConfigType] = None,
    device: Optional[torch.device] = None,
) -> nn.Module:
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture {architecture!r}. Choose from {ARCHITECTURES}")

    cfg = model_cfg or model_config_for(architecture)

    if architecture == "efficientnet":
        model = build_efficientnet_model(num_classes, cfg, device=None)  # type: ignore[arg-type]
    elif architecture == "attention":
        model = build_attention_model(num_classes, cfg, device=None)  # type: ignore[arg-type]
    elif architecture == "vit":
        model = build_vit_model(num_classes, cfg, device=None)  # type: ignore[arg-type]
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    if device is not None:
        model = model.to(device)
    return model


__all__ = [
    "ARCHITECTURES",
    "EfficientNetClassifier",
    "EfficientNetMHSA",
    "ViTClassifier",
    "build_model",
    "count_parameters",
    "param_groups_lrd",
    "set_encoder_trainable",
]
