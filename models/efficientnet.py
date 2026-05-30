"""
Pretrained EfficientNet classifier (timm) with custom num_classes head.

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

from config import EfficientNetConfig


class EfficientNetClassifier(nn.Module):
    """Pure EfficientNet baseline via timm with a replaced classification head."""

    def __init__(
        self,
        num_classes: int,
        model_cfg: Optional[EfficientNetConfig] = None,
        *,
        backbone: Optional[str] = None,
        pretrained: Optional[bool] = None,
    ):
        super().__init__()
        cfg = model_cfg or EfficientNetConfig()
        self.backbone_name = backbone or cfg.backbone
        pretrained_flag = cfg.pretrained if pretrained is None else pretrained

        self.backbone = timm.create_model(
            self.backbone_name,
            pretrained=pretrained_flag,
            num_classes=num_classes,
        )
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def build_efficientnet_model(
    num_classes: int,
    model_cfg: Optional[EfficientNetConfig] = None,
    device: Optional[torch.device] = None,
) -> EfficientNetClassifier:
    model = EfficientNetClassifier(num_classes=num_classes, model_cfg=model_cfg)
    if device is not None:
        model = model.to(device)
    return model
