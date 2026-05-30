"""
EfficientNet-B0 (pretrained) + Multi-Head Self-Attention + dense classification head.

Spatial CNN features are projected to `embed_dim`, flattened to a token sequence,
refined with batch-first multi-head self-attention (residual + LayerNorm), then
mean-pooled and classified. Suitable for Grad-CAM on the backbone feature maps.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

try:
    import timm
except ImportError as e:  # pragma: no cover
    raise ImportError("Please install timm: pip install timm>=0.9.0") from e

from config import AttentionModelConfig


class EfficientNetMHSA(nn.Module):
    """
    EfficientNet backbone without global pool -> 1x1 projection -> MHSA over spatial tokens
    -> dense head.
    """

    def __init__(
        self,
        num_classes: int,
        model_cfg: Optional[AttentionModelConfig] = None,
        *,
        backbone: Optional[str] = None,
        pretrained: Optional[bool] = None,
        embed_dim: Optional[int] = None,
        num_heads: Optional[int] = None,
        attn_dropout: Optional[float] = None,
        dropout: Optional[float] = None,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        cfg = model_cfg or AttentionModelConfig()
        self.backbone_name = backbone or cfg.backbone
        pretrained_flag = cfg.pretrained if pretrained is None else pretrained
        self.embed_dim = cfg.embed_dim if embed_dim is None else embed_dim
        self.num_heads = cfg.num_heads if num_heads is None else num_heads
        attn_drop = cfg.attn_dropout if attn_dropout is None else attn_dropout
        drop_p = cfg.dropout if dropout is None else dropout
        hidden = cfg.hidden_dim if hidden_dim is None else hidden_dim

        self.backbone = timm.create_model(
            self.backbone_name,
            pretrained=pretrained_flag,
            num_classes=0,
            global_pool="",
        )
        c_in = self.backbone.num_features
        self.proj = nn.Conv2d(c_in, self.embed_dim, kernel_size=1)
        self.norm_pre = nn.LayerNorm(self.embed_dim)
        self.attn = nn.MultiheadAttention(
            self.embed_dim,
            self.num_heads,
            dropout=attn_drop,
            batch_first=True,
        )
        self.norm_post = nn.LayerNorm(self.embed_dim)
        self.drop = nn.Dropout(drop_p)

        self.head = nn.Sequential(
            nn.Linear(self.embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(hidden, num_classes),
        )
        self.num_classes = num_classes

        self._last_feat: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone.forward_features(x)
        if feat.dim() != 4:
            raise RuntimeError(
                f"Backbone must return spatial features (B,C,H,W); got {tuple(feat.shape)}. "
                "Ensure timm model is created with global_pool=''."
            )
        self._last_feat = feat

        z = self.proj(feat)
        seq = z.flatten(2).transpose(1, 2)
        seq = self.norm_pre(seq)
        attn_out, _ = self.attn(seq, seq, seq, need_weights=False)
        seq = self.norm_post(seq + attn_out)
        pooled = seq.mean(dim=1)
        pooled = self.drop(pooled)
        logits = self.head(pooled)
        return logits

    def forward_features_dict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.forward(x)
        return {
            "logits": logits,
            "spatial_backbone": self._last_feat,
        }

    def spatial_feature_maps(self) -> Optional[torch.Tensor]:
        """Last backbone spatial features (B, C, H, W); set after forward."""
        if self._last_feat is None or self._last_feat.dim() != 4:
            return None
        return self._last_feat

    def grad_cam_module(self) -> nn.Module:
        """Module whose output activations are used for Grad-CAM (backbone body)."""
        return self.backbone


def build_attention_model(
    num_classes: int,
    model_cfg: Optional[AttentionModelConfig] = None,
    device: Optional[torch.device] = None,
) -> EfficientNetMHSA:
    model = EfficientNetMHSA(num_classes=num_classes, model_cfg=model_cfg)
    if device is not None:
        model = model.to(device)
    return model
