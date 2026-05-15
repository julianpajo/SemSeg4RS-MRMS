"""
models/utils.py
---------------

Shared training utilities:

  - build_optimizer : AdamW with differentiated parameter groups
  - build_scheduler : linear warmup + cosine decay
  - SegMetrics      : mIoU, per-class IoU, pixel accuracy
  - AverageMeter    : loss/metric accumulator
  - save_checkpoint : save model and training state
  - load_checkpoint : load checkpoint
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def build_optimizer(
    model: nn.Module,
    cfg: dict,
) -> AdamW:
    """
    Build an AdamW optimizer.

    If the model exposes parameter_groups(), that method is used to create
    differentiated learning-rate groups. Otherwise, all model parameters are
    optimized with a single learning rate.

    Expected cfg keys
    -----------------
    lr :
        Base learning rate, usually for decoder/head parameters.
    lr_backbone :
        Learning rate for backbone/encoder parameters. Defaults to lr / 10.
    weight_decay :
        Weight decay. Defaults to 0.01.

    Notes
    -----
    Models may implement parameter_groups(lr_encoder, lr_decoder, ...).
    This wrapper calls parameter_groups() only with arguments accepted by the
    model-specific signature.

    Parameters
    ----------
    model :
        Model to optimize.
    cfg :
        Optimizer configuration dictionary.

    Returns
    -------
    torch.optim.AdamW
        Initialized AdamW optimizer.
    """
    lr = cfg.get("lr", 1e-4)
    lr_backbone = cfg.get("lr_backbone", lr / 10)
    weight_decay = cfg.get("weight_decay", 0.01)

    if hasattr(model, "parameter_groups"):
        # Each model can expose groups with differentiated learning rates.
        groups = model.parameter_groups(
            **{
                k: v
                for k, v in {
                    "lr_encoder": lr_backbone,
                    "lr_decoder": lr,
                    "lr_sae": lr,
                    "lr_backbone": lr_backbone,
                    "lr_head": lr,
                    "lr_rein": lr,
                    "weight_decay": weight_decay,
                }.items()
                if k in _pg_signature(model)
            }
        )
    else:
        groups = [
            {
                "params": model.parameters(),
                "lr": lr,
                "weight_decay": weight_decay,
            }
        ]

    return AdamW(groups, lr=lr, weight_decay=weight_decay)


def _pg_signature(model: nn.Module) -> set:
    """
    Return the parameter names accepted by model.parameter_groups().

    Parameters
    ----------
    model :
        Model exposing a parameter_groups method.

    Returns
    -------
    set
        Set of accepted parameter names. Returns an empty set if the signature
        cannot be inspected.
    """
    import inspect

    try:
        sig = inspect.signature(model.parameter_groups)
        return set(sig.parameters.keys())
    except Exception:
        return set()


# ---------------------------------------------------------------------------
# Scheduler: linear warmup + cosine decay
# ---------------------------------------------------------------------------

def build_scheduler(
    optimizer: AdamW,
    num_steps: int,
    warmup_steps: int = 500,
    min_lr_ratio: float = 0.01,
) -> LambdaLR:
    """
    Build a linear-warmup plus cosine-decay learning-rate scheduler.

    The schedule is:

      - linear warmup for the first warmup_steps steps;
      - cosine decay down to lr * min_lr_ratio.

    Parameters
    ----------
    optimizer :
        Optimizer to schedule.
    num_steps :
        Total number of training steps, usually epochs × steps_per_epoch.
    warmup_steps :
        Number of warmup steps.
    min_lr_ratio :
        Minimum learning-rate ratio relative to the base learning rate.

    Returns
    -------
    torch.optim.lr_scheduler.LambdaLR
        Learning-rate scheduler.
    """
    import math

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)

        progress = float(step - warmup_steps) / max(1, num_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class SegMetrics:
    """
    Accumulate predictions and compute mIoU, per-class IoU and pixel accuracy.

    A confusion matrix is used for efficiency.

    Parameters
    ----------
    num_classes :
        Number of semantic classes.
    ignore_index :
        Label value ignored during metric computation.
    """

    def __init__(self, num_classes: int, ignore_index: int = 255):
        """
        Initialize the segmentation metrics accumulator.

        Parameters
        ----------
        num_classes :
            Number of semantic classes.
        ignore_index :
            Label value ignored during metric computation.
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self) -> None:
        """
        Reset the internal confusion matrix.
        """
        self.conf_matrix = np.zeros(
            (self.num_classes, self.num_classes),
            dtype=np.int64,
        )

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Update the confusion matrix with a new batch.

        Parameters
        ----------
        preds :
            Predicted class tensor with shape (B, H, W).
        targets :
            Ground-truth label tensor with shape (B, H, W).
        """
        p = preds.cpu().numpy().flatten()
        t = targets.cpu().numpy().flatten()

        # Filter ignore_index.
        valid = t != self.ignore_index
        p, t = p[valid], t[valid]

        # Update confusion matrix.
        idx = t * self.num_classes + p
        np.add.at(self.conf_matrix.ravel(), idx, 1)

    def compute(self) -> Dict[str, float]:
        """
        Compute segmentation metrics.

        Returns
        -------
        dict
            Dictionary containing:

            miou :
                Mean IoU over all classes.
            per_class :
                List of per-class IoU values.
            pixel_acc :
                Global pixel accuracy.
        """
        cm = self.conf_matrix.astype(np.float64)

        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        denom = tp + fp + fn
        iou = np.where(denom > 0, tp / denom, np.nan)

        miou = float(np.nanmean(iou))
        pixel_acc = float(tp.sum() / cm.sum()) if cm.sum() > 0 else 0.0

        return {
            "miou": miou,
            "per_class": iou.tolist(),
            "pixel_acc": pixel_acc,
        }


# ---------------------------------------------------------------------------
# AverageMeter
# ---------------------------------------------------------------------------

class AverageMeter:
    """
    Accumulate scalar values and compute their running average.
    """

    def __init__(self):
        """
        Initialize the meter and reset its internal state.
        """
        self.reset()

    def reset(self) -> None:
        """
        Reset accumulated sum and count.
        """
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        """
        Add a new value to the accumulator.

        Parameters
        ----------
        val :
            Scalar value.
        n :
            Number of elements represented by val.
        """
        self.sum += val * n
        self.count += n

    @property
    def avg(self) -> float:
        """
        Return the current average.

        Returns
        -------
        float
            Average value. If count is zero, the denominator is clamped to 1.
        """
        return self.sum / max(self.count, 1)


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: LambdaLR,
    epoch: int,
    metrics: dict,
    cfg: dict,
) -> None:
    """
    Save the full training state.

    Parameters
    ----------
    path :
        Output checkpoint path.
    model :
        Model to save.
    optimizer :
        Optimizer whose state will be saved.
    scheduler :
        Scheduler whose state will be saved.
    epoch :
        Current epoch index.
    metrics :
        Metrics dictionary to store in the checkpoint.
    cfg :
        Training configuration dictionary.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "metrics": metrics,
            "config": cfg,
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[AdamW] = None,
    scheduler: Optional[LambdaLR] = None,
    device: str = "cpu",
) -> dict:
    """
    Load a checkpoint.

    The model state is always loaded. Optimizer and scheduler states are loaded
    in-place if the corresponding objects are provided.

    Parameters
    ----------
    path :
        Checkpoint path.
    model :
        Model into which the checkpoint state_dict will be loaded.
    optimizer :
        Optional optimizer to restore.
    scheduler :
        Optional scheduler to restore.
    device :
        Device used for map_location.

    Returns
    -------
    dict
        Checkpoint dictionary containing fields such as "epoch", "metrics" and
        "config".
    """
    ckpt = torch.load(path, map_location=device)

    model.load_state_dict(ckpt["model"])

    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    if scheduler and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])

    print(f"[checkpoint] Loaded from '{path}' (epoch {ckpt.get('epoch', '?')})")

    return ckpt