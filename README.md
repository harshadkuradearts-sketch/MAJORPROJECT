# MAJORPROJECT

Multi-architecture research framework for dragon fruit disease image classification.

## Architectures

| `--architecture` | Model | timm backbone |
|------------------|-------|---------------|
| `attention` | EfficientNet-B0 + Multi-Head Self-Attention | `efficientnet_b0` |
| `efficientnet` | Pure EfficientNet baseline | `tf_efficientnet_b2_ns` |
| `vit` | Vision Transformer baseline | `deit_small_patch16_224` |

## Training

```bash
# Attention model (original ~89% setup)
python main.py --mode train --architecture attention --data_root ./data --epochs 30 --batch_size 8 --run_name experiment

# EfficientNet baseline
python main.py --mode train --architecture efficientnet --data_root ./data --epochs 30 --run_name exp1

# ViT baseline
python main.py --mode train --architecture vit --data_root ./data --epochs 30 --run_name exp1
```

Checkpoints and logs are stored per architecture:

```
checkpoints/<architecture>/<run_name>/
logs/<architecture>/<run_name>/
```

## Inference

```bash
python main.py --mode infer \
  --checkpoint ./checkpoints/attention/experiment/best_model.pt \
  --image ./sample.jpg
```

Architecture is auto-detected from checkpoint metadata. Override with `--architecture` if needed.

## Project layout

```
models/
  attention.py      # EfficientNet + MHSA (original model)
  efficientnet.py   # Pure EfficientNet baseline
  vit.py            # ViT / DeiT baseline
  base.py           # Shared freeze / LR helpers
  __init__.py       # Registry (build_model)
trainer.py          # Shared training loop
train.py            # Training CLI
inference.py        # Single-image inference
dataset.py          # Data pipeline (shared)
config.py           # Per-architecture configs
```
