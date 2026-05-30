"""
Central configuration for dragon fruit disease classification.
Paths should be set via environment variables or CLI; defaults are placeholders.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, Optional, Tuple, Union


ARCHITECTURES: Tuple[str, ...] = ("efficientnet", "attention", "vit")


@dataclass
class DataConfig:
    """Dataset paths, splits, and cleaning thresholds."""

    data_root: str = os.environ.get("DATA_ROOT", "./data")
    image_extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    image_size: int = 224
    num_workers: int = max(1, min(8, (os.cpu_count() or 4)))
    pin_memory: bool = True
    remove_duplicate_files: bool = True
    remove_perceptual_duplicates: bool = False
    perceptual_hash_size: int = 8
    skip_corrupted: bool = True
    blur_detection: bool = False
    blur_laplacian_threshold: float = 50.0
    random_seed: int = 42


@dataclass
class EfficientNetConfig:
    """Pure EfficientNet baseline (timm transfer learning)."""

    num_classes: Optional[int] = None
    backbone: str = "tf_efficientnet_b2_ns"
    pretrained: bool = True


@dataclass
class AttentionModelConfig:
    """EfficientNet-B0 + Multi-Head Self-Attention + dense classifier."""

    num_classes: Optional[int] = None
    backbone: str = "efficientnet_b0"
    pretrained: bool = True
    embed_dim: int = 256
    num_heads: int = 4
    attn_dropout: float = 0.1
    dropout: float = 0.35
    hidden_dim: int = 384


@dataclass
class ViTConfig:
    """ViT / DeiT baseline (timm transfer learning)."""

    num_classes: Optional[int] = None
    model_name: str = "deit_small_patch16_224"
    pretrained: bool = True
    dropout: float = 0.3
    drop_path_rate: float = 0.1


# Backward-compatible alias for the original attention model config.
ModelConfig = AttentionModelConfig

ModelConfigType = Union[EfficientNetConfig, AttentionModelConfig, ViTConfig]


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    batch_size: int = 16
    epochs: int = 100
    learning_rate: float = 3e-4
    backbone_lr_multiplier: float = 0.25
    weight_decay: float = 5e-2
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    warmup_start_factor: float = 0.1
    early_stopping_patience: int = 18
    min_delta: float = 1e-4
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    use_class_weights: bool = True
    use_weighted_sampler: bool = True
    label_smoothing: float = 0.08
    amp: bool = True
    gradient_clip_max_norm: float = 1.0
    freeze_backbone_epochs: int = 0


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    architecture: str = "attention"


def default_config() -> Config:
    return Config()


def model_config_for(architecture: str) -> ModelConfigType:
    if architecture == "efficientnet":
        return EfficientNetConfig()
    if architecture == "attention":
        return AttentionModelConfig()
    if architecture == "vit":
        return ViTConfig()
    raise ValueError(f"Unknown architecture {architecture!r}. Choose from {ARCHITECTURES}")


def model_config_to_dict(model_cfg: ModelConfigType) -> Dict[str, Any]:
    return asdict(model_cfg)


def merge_model_config(architecture: str, saved: Optional[Dict[str, Any]]) -> ModelConfigType:
    cfg = model_config_for(architecture)
    if not saved:
        return cfg
    valid_fields = {f.name for f in fields(cfg)}
    for key, value in saved.items():
        if key in valid_fields:
            setattr(cfg, key, value)
    return cfg


IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)
