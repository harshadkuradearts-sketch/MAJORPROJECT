"""
Pretrained Vision Transformer classifier (timm) with custom num_classes head.

Transfer learning: ImageNet weights, classification head replaced for num_classes.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

try:
    import timm
except ImportError as e:  # pragma: no cover
    raise ImportError("Please install timm: pip install timm>=0.9.0") from e

from config import ViTConfig


class ViTClassifier(nn.Module):
    """ViT / DeiT baseline via timm with a replaced classification head."""

    def __init__(
        self,
        num_classes: int,
        model_cfg: Optional[ViTConfig] = None,
        *,
        model_name: Optional[str] = None,
        pretrained: Optional[bool] = None,
        dropout: Optional[float] = None,
        drop_path_rate: Optional[float] = None,
    ):
        super().__init__()
        cfg = model_cfg or ViTConfig()
        self.model_name = model_name or cfg.model_name
        pretrained_flag = cfg.pretrained if pretrained is None else pretrained
        drop_p = cfg.dropout if dropout is None else dropout
        drop_path = cfg.drop_path_rate if drop_path_rate is None else drop_path_rate

        self.backbone = timm.create_model(
            self.model_name,
            pretrained=pretrained_flag,
            num_classes=num_classes,
            drop_rate=drop_p,
            drop_path_rate=drop_path,
        )
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def build_vit_model(
    num_classes: int,
    model_cfg: Optional[ViTConfig] = None,
    device: Optional[torch.device] = None,
) -> ViTClassifier:
    model = ViTClassifier(num_classes=num_classes, model_cfg=model_cfg)
    if device is not None:
        model = model.to(device)
    return model
