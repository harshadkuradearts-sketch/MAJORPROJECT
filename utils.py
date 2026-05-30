"""
Reusable training utilities: metrics, plots, seeding, checkpoints, Grad-CAM.
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import colormaps
from matplotlib import pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Runtime helpers
# ---------------------------------------------------------------------------


def get_best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def configure_torch_runtime(device: torch.device) -> None:
    """Apply small backend settings that improve throughput when available."""
    if device.type == "cuda":
        try:
            torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
            torch.backends.cudnn.allow_tf32 = True  # type: ignore[attr-defined]
        except AttributeError:
            pass
        try:
            torch.backends.cudnn.benchmark = True
        except AttributeError:
            pass
        try:
            torch.set_float32_matmul_precision("high")
        except (AttributeError, RuntimeError):
            pass


def unwrap_compiled_model(model: nn.Module) -> nn.Module:
    return getattr(model, "_orig_mod", model)


def maybe_compile_model(
    model: nn.Module,
    device: torch.device,
    *,
    enabled: bool = True,
    mode: str = "reduce-overhead",
) -> nn.Module:
    """Compile the model when the backend is expected to benefit from it."""
    if not enabled:
        logger.info("Torch compile disabled by configuration.")
        return model

    if hasattr(model, "_orig_mod"):
        logger.info("Model is already compiled; skipping recompilation.")
        return model

    if device.type == "mps":
        logger.info("Skipping torch.compile on MPS for stability.")
        return model

    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        logger.info("torch.compile is unavailable in this PyTorch build.")
        return model

    try:
        compiled = compile_fn(model, mode=mode, dynamic=False)
        logger.info("Compiled model with torch.compile(mode=%s).", mode)
        return compiled
    except (RuntimeError, TypeError, ValueError) as exc:
        logger.warning("torch.compile failed; continuing with eager model: %s", exc)
        return model


def summarize_model_parameters(model: nn.Module) -> Dict[str, int]:
    base_model = unwrap_compiled_model(model)
    total = sum(p.numel() for p in base_model.parameters())
    trainable = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    backbone = 0
    if hasattr(base_model, "backbone"):
        backbone = sum(p.numel() for p in base_model.backbone.parameters())
        if hasattr(base_model, "head") and isinstance(base_model.head, nn.Module):
            backbone -= sum(p.numel() for p in base_model.head.parameters())
    head = max(total - backbone, 0)
    return {
        "total": int(total),
        "trainable": int(trainable),
        "backbone": int(backbone),
        "head": int(head),
    }


def format_parameter_summary(model: nn.Module) -> str:
    stats = summarize_model_parameters(model)
    non_trainable = max(stats["total"] - stats["trainable"], 0)
    return (
        f"trainable={stats['trainable']:,} "
        f"non_trainable={non_trainable:,} "
        f"total={stats['total']:,}"
    )


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Make runs comparable across Colab / local (best effort; non-deterministic on GPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True  # slower; enable if you need full determinism


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(
    y_true: Union[np.ndarray, List],
    y_pred: Union[np.ndarray, List],
    num_classes: int,
) -> Dict[str, float]:
    """
    Return accuracy, macro/weighted precision, recall, F1.
    """
    yt = np.asarray(y_true).flatten()
    yp = np.asarray(y_pred).flatten()
    out: Dict[str, float] = {}
    out["accuracy"] = float(accuracy_score(yt, yp))
    avg_modes = ("macro", "weighted")
    for avg in avg_modes:
        out[f"precision_{avg}"] = float(
            precision_score(yt, yp, average=avg, zero_division=0, labels=np.arange(num_classes))
        )
        out[f"recall_{avg}"] = float(
            recall_score(yt, yp, average=avg, zero_division=0, labels=np.arange(num_classes))
        )
        out[f"f1_{avg}"] = float(
            f1_score(yt, yp, average=avg, zero_division=0, labels=np.arange(num_classes))
        )
    return out


def confusion_matrix_numpy(
    y_true: Union[np.ndarray, List],
    y_pred: Union[np.ndarray, List],
    num_classes: int,
) -> np.ndarray:
    yt = np.asarray(y_true).flatten()
    yp = np.asarray(y_pred).flatten()
    return confusion_matrix(yt, yp, labels=np.arange(num_classes))


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: str,
    figsize: Tuple[int, int] = (10, 8),
    normalize: bool = False,
) -> None:
    """Plot confusion matrix heatmap and save to disk."""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
        data = cm_norm
        fmt = ".2f"
    else:
        data = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True",
        xlabel="Predicted",
        title="Confusion matrix" + (" (normalized)" if normalize else ""),
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = data.max() / 2.0 if data.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(data[i, j], fmt),
                ha="center",
                va="center",
                color="white" if data[i, j] > thresh else "black",
            )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: str,
) -> None:
    """Plot loss / metrics vs epoch from a history dict of lists."""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    n = len(history.get("train_loss", []))
    if n == 0:
        return
    epochs = np.arange(1, n + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if "train_loss" in history:
        axes[0].plot(epochs, history["train_loss"], label="train")
    if "val_loss" in history:
        axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].set_title("Loss")

    key_acc = "val_accuracy" if "val_accuracy" in history else None
    key_f1 = "val_f1_macro" if "val_f1_macro" in history else None
    if key_acc:
        axes[1].plot(epochs, history[key_acc], label="acc")
    if key_f1:
        axes[1].plot(epochs, history[key_f1], label="F1 macro")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].legend()
    axes[1].set_title("Validation metrics")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


class LiveTrainingMonitor:
    """Headless training history tracker that writes plots to disk."""

    def __init__(self, run_dir: str, title: str = "Training progress") -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir = self.run_dir / "snapshots"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = self.run_dir / "training_history.json"
        self.final_plot_path = self.run_dir / "training_curves.png"

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_precision_macro": [],
            "val_recall_macro": [],
            "val_f1_macro": [],
            "lr": [],
        }

        self._fig = None
        self._axes = None
        self._enabled = False

    def update(
        self,
        epoch: int,
        *,
        train_loss: float,
        val_loss: float,
        val_metrics: Dict[str, float],
        lr: float,
    ) -> None:
        self.history["train_loss"].append(float(train_loss))
        self.history["val_loss"].append(float(val_loss))
        self.history["val_accuracy"].append(float(val_metrics["accuracy"]))
        self.history["val_precision_macro"].append(float(val_metrics["precision_macro"]))
        self.history["val_recall_macro"].append(float(val_metrics["recall_macro"]))
        self.history["val_f1_macro"].append(float(val_metrics["f1_macro"]))
        self.history["lr"].append(float(lr))

        self._persist_history()
        self._save_snapshot(epoch)

        # Headless mode: only persist metrics and write the final static plot.

    def finalize(self) -> None:
        self._persist_history()
        self._save_snapshot(len(self.history["train_loss"]) - 1, final=True)
        plot_training_curves(self.history, str(self.final_plot_path))

    def _persist_history(self) -> None:
        with open(self.history_path, "w", encoding="utf-8") as handle:
            json.dump(self.history, handle, indent=2)

    def _save_snapshot(self, epoch: int, final: bool = False) -> None:
        return

    def _render(self) -> None:
        return


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


def save_checkpoint(
    path: str,
    *,
    epoch: int,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[Any],
    scaler: Optional[torch.cuda.amp.GradScaler],
    best_val_loss: float,
    history: Optional[Dict[str, List[float]]] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload: Dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": unwrap_compiled_model(model).state_dict(),
        "best_val_loss": best_val_loss,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()
    if history is not None:
        payload["history"] = history
    if meta is not None:
        payload["meta"] = meta
    torch.save(payload, path)
    logger.info("Saved checkpoint to %s", path)


def load_checkpoint(
    path: str,
    map_location: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location)
    return ckpt


def load_model_weights(model: nn.Module, state_dict: Dict[str, Tensor], strict: bool = True) -> None:
    target_model = unwrap_compiled_model(model)
    missing, unexpected = target_model.load_state_dict(state_dict, strict=strict)
    if missing or unexpected:
        logger.warning("load_state_dict: missing=%s unexpected=%s", missing, unexpected)


# ---------------------------------------------------------------------------
# Grad-CAM (CNN backbone feature maps before 1x1 projection)
# ---------------------------------------------------------------------------


@torch.no_grad()
def _denormalize_image(
    tensor_chw: Tensor,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
) -> np.ndarray:
    """CHW tensor -> HxWx3 uint8 for overlay."""
    t = tensor_chw.detach().cpu().clone()
    for i in range(3):
        t[i] = t[i] * std[i] + mean[i]
    t = t.clamp(0, 1).numpy().transpose(1, 2, 0)
    return (t * 255).astype(np.uint8)


def compute_grad_cam(
    model: nn.Module,
    input_batch: Tensor,
    target_class: int,
    *,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
    image_idx: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Grad-CAM on inputs to `model.proj` (EfficientNet spatial features).

    Returns:
        cam_hw: 2D numpy heatmap in [0,1], shape (H_in, W_in)
        overlay_rgb: uint8 HxWx3 for visualization
    """
    if input_batch.dim() != 4:
        raise ValueError("input_batch must be (N,C,H,W)")

    model.eval()
    x = input_batch.clone().detach().requires_grad_(True)

    activations: List[Tensor] = []
    gradients: List[Tensor] = []

    def forward_pre_hook(_module, inputs):
        t = inputs[0]
        activations.append(t)

    def full_backward_hook(_module, grad_input, _grad_output):
        if grad_input[0] is not None:
            gradients.append(grad_input[0])

    if not hasattr(model, "proj"):
        raise AttributeError("Model must have a `proj` Conv2d after backbone (EfficientNetMHSA).")

    h_pre = model.proj.register_forward_pre_hook(forward_pre_hook)
    h_bwd = model.proj.register_full_backward_hook(full_backward_hook)

    logits = model(x)
    model.zero_grad(set_to_none=True)
    score = logits[image_idx, target_class]
    score.backward(retain_graph=False)

    h_pre.remove()
    h_bwd.remove()

    if not activations or not gradients:
        raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

    act = activations[0]
    grad = gradients[0]
    # Global average pool of gradients per channel -> importance weights
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = (weights * act).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = cam[image_idx, 0]
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    h_in, w_in = x.shape[2], x.shape[3]
    cam_up = F.interpolate(
        cam.unsqueeze(0).unsqueeze(0),
        size=(h_in, w_in),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    cam_np = cam_up.detach().cpu().numpy()

    rgb = _denormalize_image(x[image_idx], mean, std)
    heat = colormaps["jet"](cam_np)[..., :3]
    heat = (heat * 255).astype(np.uint8)
    overlay = (0.45 * rgb + 0.55 * heat).astype(np.uint8)

    return cam_np, overlay


def save_grad_cam_pair(
    cam_hw: np.ndarray,
    overlay_rgb: np.ndarray,
    save_path_raw: str,
    save_path_overlay: str,
) -> None:
    from PIL import Image

    os.makedirs(os.path.dirname(save_path_raw) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(save_path_overlay) or ".", exist_ok=True)
    plt.imsave(save_path_raw, cam_hw, cmap="jet")
    Image.fromarray(overlay_rgb).save(save_path_overlay)


# ---------------------------------------------------------------------------
# JSON meta (class names, etc.)
# ---------------------------------------------------------------------------


def save_run_meta(path: str, meta: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)


def load_run_meta(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
