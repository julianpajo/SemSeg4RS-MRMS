"""
pl_modules/components/schedulers.py
-------------------------------------
Learning rate scheduler components for semantic segmentation training.

Each scheduler is implemented as a configuration class that inherits from
``Scheduler`` and exposes a ``build()`` method returning a callable
``optimizer -> LRScheduler``. The ``interval`` property declares whether
the scheduler must be stepped every ``"step"`` (batch) or every ``"epoch"``,
and is consumed by ``LightningSegConfig`` to configure the Lightning trainer
accordingly.

Available components
--------------------
PolyLR
    Polynomial LR decay with optional linear warmup. Standard in
    mmsegmentation-style training. Must be stepped every batch
    (``interval = "step"``). Recommended for DOFA, CrossEarth, and
    SegFormer-SAE to match the original paper training recipes.

CosineAnnealing
    Plain ``torch.optim.lr_scheduler.CosineAnnealingLR`` wrapped in the
    component interface. Stepped every epoch. Useful as a lightweight
    alternative when warmup is not needed.

CosineWarmup
    Linear warmup for a configurable number of epochs followed by cosine
    annealing. Stepped every epoch. A common alternative to PolyLR when
    working outside the mmseg ecosystem.

Usage
-----
    from pl_modules.components.schedulers import PolyLR, CosineWarmup

    # PolyLR — compute max_iters before creating the config
    max_iters = len(train_loader) * EPOCHS
    scheduler_fn = PolyLR(max_iters=max_iters, warmup_iters=1500).build()
    scheduler_interval = "step"

    # CosineWarmup
    scheduler_fn = CosineWarmup(epochs=50, warmup_epochs=5).build()
    scheduler_interval = "epoch"
"""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch.optim.lr_scheduler import (
    CosineAnnealingLR as TorchCosineAnnealingLR,
    LinearLR,
    LRScheduler,
    SequentialLR,
)

from .base import Scheduler


# ============================================================
# PolyLR
# ============================================================

class _PolyLR(LRScheduler):
    """Polynomial LR decay with optional linear warmup."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_iters: int,
        power: float = 0.9,
        min_lr: float = 0.0,
        warmup_iters: int = 0,
        warmup_ratio: float = 1e-6,
        last_epoch: int = -1,
    ):
        self.max_iters = max_iters
        self.power = power
        self.min_lr = min_lr
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio

        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        curr = self.last_epoch

        if curr < self.warmup_iters:
            alpha = (
                self.warmup_ratio
                + (1.0 - self.warmup_ratio)
                * curr
                / max(self.warmup_iters, 1)
            )
            return [base_lr * alpha for base_lr in self.base_lrs]

        progress = min(
            curr - self.warmup_iters,
            self.max_iters - self.warmup_iters,
        )
        denom = max(self.max_iters - self.warmup_iters, 1)
        scale = (1.0 - progress / denom) ** self.power

        return [
            max(self.min_lr, base_lr * scale)
            for base_lr in self.base_lrs
        ]


class PolyLR(Scheduler):
    """
    Polynomial LR decay with optional linear warmup.

    This scheduler must usually be stepped every batch.

    Parameters
    ----------
    max_iters:
        Total number of training steps.

    power:
        Polynomial decay exponent.

    warmup_iters:
        Number of warmup steps.

    warmup_ratio:
        Initial LR multiplier during warmup.

    min_lr:
        Minimum learning rate floor.
    """

    def __init__(
        self,
        max_iters: int,
        power: float = 0.9,
        warmup_iters: int = 1500,
        warmup_ratio: float = 1e-6,
        min_lr: float = 0.0,
    ):
        self.max_iters = max_iters
        self.power = power
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio
        self.min_lr = min_lr

    @property
    def interval(self) -> str:
        return "step"

    def build(self) -> Callable[[torch.optim.Optimizer], Any]:
        cfg = self

        def _build(optimizer: torch.optim.Optimizer) -> _PolyLR:
            return _PolyLR(
                optimizer=optimizer,
                max_iters=cfg.max_iters,
                power=cfg.power,
                warmup_iters=cfg.warmup_iters,
                warmup_ratio=cfg.warmup_ratio,
                min_lr=cfg.min_lr,
            )

        return _build


# ============================================================
# CosineAnnealing
# ============================================================

class CosineAnnealing(Scheduler):
    """
    Plain cosine annealing learning-rate decay.

    This is the standard PyTorch CosineAnnealingLR wrapped in the project's
    scheduler component interface.

    It is stepped every epoch by default.

    Parameters
    ----------
    epochs:
        Total number of training epochs. Used as ``T_max``.

    min_lr:
        Minimum learning rate at the end of cosine decay.

    Example
    -------
    scheduler_fn = CosineAnnealing(
        epochs=20,
        min_lr=0.0,
    ).build()

    scheduler_interval = "epoch"
    """

    def __init__(
        self,
        epochs: int,
        min_lr: float = 0.0,
    ):
        self.epochs = epochs
        self.min_lr = min_lr

    @property
    def interval(self) -> str:
        return "epoch"

    def build(self) -> Callable[[torch.optim.Optimizer], Any]:
        cfg = self

        def _build(optimizer: torch.optim.Optimizer) -> TorchCosineAnnealingLR:
            return TorchCosineAnnealingLR(
                optimizer=optimizer,
                T_max=max(1, cfg.epochs),
                eta_min=cfg.min_lr,
            )

        return _build


# ============================================================
# CosineWarmup
# ============================================================

class CosineWarmup(Scheduler):
    """
    Linear warmup followed by cosine annealing.

    This scheduler is stepped every epoch by default.

    Parameters
    ----------
    epochs:
        Total number of training epochs.

    warmup_epochs:
        Number of linear warmup epochs.

    min_lr_ratio:
        Final minimum LR as a ratio of the first optimizer param-group LR.

    warmup_start_factor:
        Initial LR multiplier for the warmup stage.
    """

    def __init__(
        self,
        epochs: int,
        warmup_epochs: int = 5,
        min_lr_ratio: float = 0.01,
        warmup_start_factor: float = 0.01,
    ):
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.min_lr_ratio = min_lr_ratio
        self.warmup_start_factor = warmup_start_factor

    @property
    def interval(self) -> str:
        return "epoch"

    def build(self) -> Callable[[torch.optim.Optimizer], Any]:
        cfg = self

        def _build(optimizer: torch.optim.Optimizer) -> SequentialLR:
            base_lr = optimizer.param_groups[0]["lr"]

            warmup = LinearLR(
                optimizer,
                start_factor=cfg.warmup_start_factor,
                end_factor=1.0,
                total_iters=max(1, cfg.warmup_epochs),
            )

            cosine = TorchCosineAnnealingLR(
                optimizer,
                T_max=max(1, cfg.epochs - cfg.warmup_epochs),
                eta_min=base_lr * cfg.min_lr_ratio,
            )

            return SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[cfg.warmup_epochs],
            )

        return _build