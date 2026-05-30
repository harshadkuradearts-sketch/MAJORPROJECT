"""
Dataset loading, cleaning, stratified splits, and augmentation for dragon fruit disease images.

Expects a folder layout::

    data_root/
        class_a/
            img1.jpg
        class_b/
            ...

Robust to corrupted files, optional duplicate removal, optional blur filtering,
stratified train/val/test (70/15/15), and stronger augmentation for minority classes.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

try:
    import cv2
except ImportError:  # pragma: no cover - optional until requirements installed
    cv2 = None

from config import DataConfig, IMAGENET_MEAN, IMAGENET_STD

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Image constants & helpers
# ---------------------------------------------------------------------------


def _pil_open_rgb(path: Union[str, Path]) -> Image.Image:
    """Open image as RGB; raises if invalid."""
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def is_image_readable(path: Union[str, Path]) -> bool:
    """Return True if PIL can load and convert the file to RGB."""
    try:
        with Image.open(path) as img:
            ImageOps.exif_transpose(img).convert("RGB")
        return True
    except Exception:
        return False


def file_md5(path: Union[str, Path], chunk_size: int = 1 << 20) -> str:
    """MD5 hash of raw file bytes (exact duplicates)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def average_hash(path: Union[str, Path], hash_size: int = 8) -> str:
    """
    Simple perceptual hash: resize to hash_size x hash_size grayscale, compare to mean.
    Good for near-duplicate detection without heavy deps.
    """
    img = _pil_open_rgb(path)
    g = np.array(img.resize((hash_size, hash_size), Image.Resampling.BILINEAR).convert("L"))
    mean = g.mean()
    bits = g.flatten() > mean
    return "".join("1" if b else "0" for b in bits)


def hamming_distance(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))


def _laplacian_variance_numpy(gray: np.ndarray) -> float:
    """Variance of Laplacian on grayscale float array (H, W), NumPy only."""
    if gray.ndim != 2:
        raise ValueError("expected 2D grayscale array")
    h, w = gray.shape
    if h < 3 or w < 3:
        return 0.0
    g = gray.astype(np.float64, copy=False)
    lap = (
        g[0:-2, 1:-1]
        + g[2:, 1:-1]
        + g[1:-1, 0:-2]
        + g[1:-1, 2:]
        - 4.0 * g[1:-1, 1:-1]
    )
    return float(lap.var())


def laplacian_variance(path: Union[str, Path]) -> float:
    """
    Variance of Laplacian; low values often indicate heavy blur.
    Uses OpenCV if available; otherwise a NumPy Laplacian on grayscale pixels.
    """
    if cv2 is not None:
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise ValueError(f"cv2 could not read: {path}")
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    img = _pil_open_rgb(path)
    g = np.asarray(img.convert("L"), dtype=np.float64)
    return _laplacian_variance_numpy(g)


# ---------------------------------------------------------------------------
# Cleaning & manifest
# ---------------------------------------------------------------------------


@dataclass
class ImageRecord:
    path: Path
    class_idx: int
    class_name: str


def discover_class_folders(
    data_root: Union[str, Path],
    extensions: Sequence[str],
) -> Tuple[List[str], Dict[str, int]]:
    """
    Discover class names from immediate subdirectories that contain at least one image.
    """
    root = Path(data_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"data_root does not exist or is not a directory: {root}")

    class_names = sorted(
        d.name for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")
    )
    if not class_names:
        raise RuntimeError(f"No class subfolders found under {root}")

    ext_set = {e.lower() for e in extensions}
    valid_classes = []
    for name in class_names:
        cdir = root / name
        has_img = any(f.suffix.lower() in ext_set for f in cdir.rglob("*") if f.is_file())
        if has_img:
            valid_classes.append(name)

    if not valid_classes:
        raise RuntimeError(f"No images found under class folders in {root}")

    class_to_idx = {n: i for i, n in enumerate(valid_classes)}
    return valid_classes, class_to_idx


def iter_image_paths(
    data_root: Union[str, Path],
    class_to_idx: Dict[str, int],
    extensions: Sequence[str],
) -> List[Path]:
    root = Path(data_root).resolve()
    ext_set = {e.lower() for e in extensions}
    paths: List[Path] = []
    for class_name in class_to_idx:
        cdir = root / class_name
        for f in cdir.rglob("*"):
            if f.is_file() and f.suffix.lower() in ext_set:
                paths.append(f)
    return paths


def clean_image_paths(
    paths: Sequence[Path],
    class_to_idx: Dict[str, int],
    data_root: Path,
    *,
    remove_duplicate_files: bool = True,
    remove_perceptual_duplicates: bool = True,
    perceptual_hash_size: int = 8,
    max_perceptual_hamming: int = 5,
    skip_corrupted: bool = True,
    blur_detection: bool = False,
    blur_laplacian_threshold: float = 50.0,
) -> Tuple[List[ImageRecord], Dict[str, int]]:
    """
    Filter paths: corrupted, optional blur, exact duplicates, perceptual duplicates.

    Returns records and a stats dict with counts of dropped reasons.
    """
    stats: Counter = Counter()
    kept: List[ImageRecord] = []
    seen_md5: Dict[str, Path] = {}
    phash_buckets: List[Tuple[str, Path]] = []

    for path in paths:
        try:
            rel = path.relative_to(data_root)
        except ValueError:
            stats["outside_root"] += 1
            continue
        parts = rel.parts
        if len(parts) < 2:
            stats["bad_layout"] += 1
            continue
        class_name = parts[0]
        if class_name not in class_to_idx:
            stats["unknown_class"] += 1
            continue

        if skip_corrupted and not is_image_readable(path):
            stats["corrupted"] += 1
            continue

        if blur_detection:
            try:
                var = laplacian_variance(path)
                if var < blur_laplacian_threshold:
                    stats["too_blurry"] += 1
                    continue
            except Exception:
                stats["blur_check_failed"] += 1
                continue

        if remove_duplicate_files:
            try:
                digest = file_md5(path)
            except OSError:
                stats["read_error"] += 1
                continue
            if digest in seen_md5:
                stats["duplicate_file"] += 1
                continue
            seen_md5[digest] = path

        if remove_perceptual_duplicates:
            try:
                ph = average_hash(path, hash_size=perceptual_hash_size)
            except Exception:
                stats["phash_failed"] += 1
                continue
            dup = False
            for existing_hash, _existing_path in phash_buckets:
                if hamming_distance(ph, existing_hash) <= max_perceptual_hamming:
                    stats["perceptual_duplicate"] += 1
                    dup = True
                    break
            if dup:
                continue
            phash_buckets.append((ph, path))

        kept.append(
            ImageRecord(
                path=path,
                class_idx=class_to_idx[class_name],
                class_name=class_name,
            )
        )

    logger.info(
        "Cleaning summary: kept=%d dropped=%s",
        len(kept),
        dict(stats),
    )
    return kept, dict(stats)


def build_manifest(
    data_root: Union[str, Path],
    data_cfg: Optional[DataConfig] = None,
) -> Tuple[List[ImageRecord], List[str], Dict[str, int]]:
    """
    Discover classes, list images, apply cleaning. Returns records and metadata.
    """
    data_cfg = data_cfg or DataConfig()
    root = Path(data_root).resolve()
    logger.info("Pre-processing | scanning data root: %s", root)
    class_names, class_to_idx = discover_class_folders(root, data_cfg.image_extensions)
    raw_paths = iter_image_paths(root, class_to_idx, data_cfg.image_extensions)
    logger.info(
        "Pre-processing | discovered classes=%d raw_images=%d",
        len(class_names),
        len(raw_paths),
    )

    records, _stats = clean_image_paths(
        raw_paths,
        class_to_idx,
        root,
        remove_duplicate_files=data_cfg.remove_duplicate_files,
        remove_perceptual_duplicates=data_cfg.remove_perceptual_duplicates,
        perceptual_hash_size=data_cfg.perceptual_hash_size,
        skip_corrupted=data_cfg.skip_corrupted,
        blur_detection=data_cfg.blur_detection,
        blur_laplacian_threshold=data_cfg.blur_laplacian_threshold,
    )
    logger.info(
        "Pre-processing | cleaned_images=%d kept_classes=%d",
        len(records),
        len(class_names),
    )
    return records, class_names, class_to_idx


def stratified_split_indices(
    labels: Sequence[int],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified 70/15/15 style split (ratios configurable)."""
    s = train_ratio + val_ratio + test_ratio
    if abs(s - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {s}")

    labels_arr = np.asarray(labels)
    indices = np.arange(len(labels_arr))

    test_size = val_ratio + test_ratio
    idx_train, idx_temp = train_test_split(
        indices,
        test_size=test_size,
        stratify=labels_arr,
        random_state=seed,
        shuffle=True,
    )
    rel_test = test_ratio / test_size
    labels_temp = labels_arr[idx_temp]
    idx_val, idx_test = train_test_split(
        idx_temp,
        test_size=rel_test,
        stratify=labels_temp,
        random_state=seed,
        shuffle=True,
    )
    return idx_train, idx_val, idx_test


# ---------------------------------------------------------------------------
# Transforms: torchvision (train with aug; val/test resize + center crop)
# ---------------------------------------------------------------------------


def build_train_transforms(
    image_size: int,
    augmentation_strength: str,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> Callable:
    """
    Training pipeline: light RandomResizedCrop, flips, rotation, color jitter, optional blur.
    Tensor + ImageNet normalize last. augmentation_strength: 'weak' | 'strong'.
    """
    jitter = (
        {"brightness": 0.3, "contrast": 0.3, "saturation": 0.25, "hue": 0.06}
        if augmentation_strength == "strong"
        else {"brightness": 0.2, "contrast": 0.2, "saturation": 0.15, "hue": 0.04}
    )
    rot = 22 if augmentation_strength == "strong" else 15
    blur_p = 0.22 if augmentation_strength == "strong" else 0.12
    blur_k = 3

    return T.Compose(
        [
            T.RandomResizedCrop(
                image_size,
                scale=(0.88, 1.0),
                ratio=(0.92, 1.08),
                interpolation=T.InterpolationMode.BILINEAR,
            ),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=rot, interpolation=T.InterpolationMode.BILINEAR),
            T.ColorJitter(**jitter),
            T.RandomApply(
                [T.GaussianBlur(kernel_size=blur_k, sigma=(0.1, 0.6))],
                p=blur_p,
            ),
            T.ToTensor(),
            T.Normalize(mean=list(mean), std=list(std)),
        ]
    )


def build_eval_transforms(
    image_size: int,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> Callable:
    """
    Validation / test: resize (ImageNet-style), center crop to image_size, normalize.
    """
    resize = int(round(image_size * 256 / 224))
    return T.Compose(
        [
            T.Resize(resize, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=list(mean), std=list(std)),
        ]
    )


# ---------------------------------------------------------------------------
# Lazy datasets: only paths + labels in RAM; decode in __getitem__
# ---------------------------------------------------------------------------


def _load_rgb_or_none(path: str) -> Optional[Image.Image]:
    """Load RGB image or return None if corrupted / unreadable."""
    try:
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img)
            return img.convert("RGB")
    except Exception as exc:
        logger.warning("Corrupted or unreadable image skipped: %s (%s)", path, exc)
        return None


class LazyImageClassificationDataset(Dataset):
    """
    Memory-efficient dataset: stores only string paths and int labels.
    PIL load + transforms run inside __getitem__ (true lazy I/O).
    """

    def __init__(
        self,
        image_paths: Sequence[str],
        labels: Sequence[int],
        image_size: int,
        *,
        train: bool,
        transform_eval: Optional[Callable] = None,
        transform_train_weak: Optional[Callable] = None,
        transform_train_strong: Optional[Callable] = None,
        minority_label_indices: Optional[set] = None,
    ):
        if len(image_paths) != len(labels):
            raise ValueError("image_paths and labels must have the same length")

        self.image_paths: List[str] = [str(p) for p in image_paths]
        self.labels: List[int] = list(labels)
        self.image_size = image_size
        self.train = train

        if train:
            if transform_train_weak is None or transform_train_strong is None:
                raise ValueError("train=True requires transform_train_weak and transform_train_strong")
            self.transform_eval = None
            self.tw = transform_train_weak
            self.ts = transform_train_strong
            self.minority = minority_label_indices or set()
        else:
            if transform_eval is None:
                raise ValueError("train=False requires transform_eval")
            self.transform_eval = transform_eval
            self.tw = self.ts = None
            self.minority = set()

        # Same-class indices for corrupted-file fallback (list of int per label; compact)
        by_label: Dict[int, List[int]] = defaultdict(list)
        for i, y in enumerate(self.labels):
            by_label[y].append(i)
        self._by_label: Dict[int, List[int]] = dict(by_label)

    def __len__(self) -> int:
        return len(self.image_paths)

    def _transform_for_label(self, label: int) -> Callable:
        if not self.train:
            return self.transform_eval  # type: ignore[return-value]
        return self.ts if label in self.minority else self.tw

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        label = self.labels[idx]
        order = [idx]
        order.extend(i for i in self._by_label.get(label, ()) if i != idx)

        for j in order[: min(64, len(order))]:
            path = self.image_paths[j]
            pil = _load_rgb_or_none(path)
            if pil is None:
                continue
            tensor = self._transform_for_label(label)(pil)
            return tensor, label

        logger.error(
            "No readable image for label %s after %d attempts; returning zeros tensor for idx %d",
            label,
            len(order),
            idx,
        )
        z = torch.zeros(3, self.image_size, self.image_size, dtype=torch.float32)
        # Match normalized black-ish (ImageNet mean scaled)
        for c in range(3):
            z[c] = (0.0 - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
        return z, label


def minority_class_indices(labels: Sequence[int], fraction_threshold: float = 0.8) -> set:
    """
    Classes with count below (median_count * fraction_threshold) get stronger augmentation.
    """
    counts = Counter(labels)
    if not counts:
        return set()
    median = np.median(list(counts.values()))
    cutoff = max(1.0, median * fraction_threshold)
    return {c for c, n in counts.items() if n < cutoff}


def compute_class_weights(labels: Sequence[int], num_classes: int) -> torch.Tensor:
    """Inverse-frequency weights normalized to mean 1.0 (for CrossEntropyLoss)."""
    counts = Counter(labels)
    weights = torch.zeros(num_classes, dtype=torch.float32)
    for c in range(num_classes):
        n = counts.get(c, 1)
        weights[c] = 1.0 / float(n)
    weights *= num_classes / weights.sum()
    return weights


def make_weighted_sampler(labels: Sequence[int]) -> WeightedRandomSampler:
    """Sampler oversampling low-frequency classes."""
    labels_list = list(labels)
    sample_weights = []
    class_counts = Counter(labels_list)
    for y in labels_list:
        sample_weights.append(1.0 / class_counts[y])
    sample_weights_t = torch.tensor(sample_weights, dtype=torch.double)
    return WeightedRandomSampler(sample_weights_t, num_samples=len(sample_weights), replacement=True)


def _pin_memory_flag(enabled_in_cfg: bool) -> bool:
    if not enabled_in_cfg:
        return False
    # Pin memory accelerates host->GPU copies on CUDA; harmless otherwise.
    return torch.cuda.is_available()


def paths_and_labels_from_indices(
    records: Sequence[ImageRecord],
    indices: np.ndarray,
) -> Tuple[List[str], List[int]]:
    """Build path/label lists for lazy datasets (only strings + ints in memory)."""
    paths = [str(records[int(i)].path) for i in indices]
    labels = [records[int(i)].class_idx for i in indices]
    return paths, labels


def _dataloader_worker_kwargs(
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
) -> Dict[str, Any]:
    """Colab-stable DataLoader worker settings."""
    pw = bool(persistent_workers and num_workers > 0)
    out: Dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": pw,
    }
    if num_workers > 0:
        out["prefetch_factor"] = 2
    return out


def log_dataloader_batch_stats(
    loader: DataLoader,
    device: torch.device,
    tag: str = "train",
) -> None:
    """
    Debug: first-batch shape, host RAM, CUDA VRAM, timing, loader length.
    """
    try:
        import psutil  # type: ignore[import-untyped]

        proc = psutil.Process(os.getpid())
        rss_gb = proc.memory_info().rss / (1024**3)
    except Exception:
        rss_gb = float("nan")

    n_batches = len(loader)
    logger.info("[%s] DataLoader batches=%s batch_size=%s workers=%s", tag, n_batches, loader.batch_size, loader.num_workers)

    t0 = time.perf_counter()
    it = iter(loader)
    xb, yb = next(it)
    dt = time.perf_counter() - t0
    logger.info(
        "[%s] first batch load_time=%.3fs tensor_shape=%s labels_shape=%s dtype=%s",
        tag,
        dt,
        tuple(xb.shape),
        tuple(yb.shape),
        xb.dtype,
    )
    logger.info("[%s] host_RSS_GB (approx)=%.3f", tag, rss_gb)

    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
        alloc = torch.cuda.memory_allocated() / (1024**2)
        reserved = torch.cuda.memory_reserved() / (1024**2)
        logger.info("[%s] cuda_mem_alloc_MB=%.1f reserved_MB=%.1f", tag, alloc, reserved)

    del it, xb, yb


def create_dataloaders(
    records: Sequence[ImageRecord],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    class_names: Sequence[str],
    data_cfg: DataConfig,
    batch_size: int,
    use_weighted_sampler: bool,
    *,
    eval_batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    persistent_workers: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, object]]:
    """
    Lazy-loading datasets + Colab-oriented DataLoaders (workers=2, pin_memory=CUDA only).
    """
    num_classes = len(class_names)
    class_to_idx = {n: i for i, n in enumerate(class_names)}

    train_paths, train_labels_list = paths_and_labels_from_indices(records, train_idx)
    train_labels = train_labels_list
    minority = minority_class_indices(train_labels)

    image_size = data_cfg.image_size
    tw = build_train_transforms(image_size, "weak")
    ts = build_train_transforms(image_size, "strong")
    te = build_eval_transforms(image_size)

    train_ds = LazyImageClassificationDataset(
        train_paths,
        train_labels_list,
        image_size,
        train=True,
        transform_train_weak=tw,
        transform_train_strong=ts,
        minority_label_indices=minority,
    )

    val_paths, val_labels = paths_and_labels_from_indices(records, val_idx)
    test_paths, test_labels = paths_and_labels_from_indices(records, test_idx)

    val_ds = LazyImageClassificationDataset(
        val_paths,
        val_labels,
        image_size,
        train=False,
        transform_eval=te,
    )
    test_ds = LazyImageClassificationDataset(
        test_paths,
        test_labels,
        image_size,
        train=False,
        transform_eval=te,
    )

    class_weights = compute_class_weights(train_labels, num_classes)

    sampler = make_weighted_sampler(train_labels) if use_weighted_sampler else None

    pin = _pin_memory_flag(data_cfg.pin_memory)
    nw = num_workers if num_workers is not None else data_cfg.num_workers
    eval_bs = eval_batch_size if eval_batch_size is not None else batch_size
    dl_common_train = _dataloader_worker_kwargs(nw, pin, persistent_workers)
    dl_common_eval = _dataloader_worker_kwargs(nw, pin, persistent_workers)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        drop_last=False,
        **dl_common_train,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=eval_bs,
        shuffle=False,
        drop_last=False,
        **dl_common_eval,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=eval_bs,
        shuffle=False,
        drop_last=False,
        **dl_common_eval,
    )

    extra = {
        "class_weights": class_weights,
        "minority_class_indices": minority,
        "num_classes": num_classes,
        "class_to_idx": class_to_idx,
        "class_names": list(class_names),
        "train_idx": np.asarray(train_idx, dtype=np.int64),
        "val_idx": np.asarray(val_idx, dtype=np.int64),
        "test_idx": np.asarray(test_idx, dtype=np.int64),
        "dataloader_num_workers": nw,
        "train_batch_size": batch_size,
        "eval_batch_size": eval_bs,
        "use_weighted_sampler": use_weighted_sampler,
    }
    return train_loader, val_loader, test_loader, extra


def prepare_data_pipeline(
    data_root: Optional[str] = None,
    data_cfg: Optional[DataConfig] = None,
    batch_size: int = 8,
    use_weighted_sampler: bool = True,
    seed: Optional[int] = None,
    *,
    eval_batch_size: Optional[int] = None,
    num_workers: int = 2,
    persistent_workers: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, object]]:
    """
    High-level: manifest -> stratified split -> DataLoaders + metadata.

    Defaults favor Google Colab free tier (batch_size=8, num_workers=2, persistent_workers=False).
    """
    data_cfg = data_cfg or DataConfig()
    split_seed = int(seed) if seed is not None else data_cfg.random_seed
    root = data_root or data_cfg.data_root

    logger.info("Pre-processing | building splits and dataloaders")

    records, class_names, _ = build_manifest(root, data_cfg)
    labels = [r.class_idx for r in records]

    tr, va, te = stratified_split_indices(
        labels,
        data_cfg.train_ratio,
        data_cfg.val_ratio,
        data_cfg.test_ratio,
        split_seed,
    )

    logger.info(
        "Pre-processing | split sizes train=%d val=%d test=%d seed=%d",
        len(tr),
        len(va),
        len(te),
        split_seed,
    )

    return create_dataloaders(
        records,
        tr,
        va,
        te,
        class_names,
        data_cfg,
        batch_size=batch_size,
        use_weighted_sampler=use_weighted_sampler,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )


# CLI helper for standalone dataset audit/clean report
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    cfg = DataConfig(data_root=os.environ.get("DATA_ROOT", "./data"))
    recs, names, _ = build_manifest(cfg.data_root, cfg)
    print(f"Classes ({len(names)}): {names}")
    print(f"Total images after cleaning: {len(recs)}")
    ys = [r.class_idx for r in recs]
    print("Class counts:", Counter(ys))
