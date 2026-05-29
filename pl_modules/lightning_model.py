"""
pl_modules/lightning_model.py
------------------------------
Generic Lightning wrapper for dense semantic segmentation models.

This module provides a model-agnostic ``LightningModule`` that works with
any segmentation architecture exposing a standard ``model(x, **kwargs) ->
logits`` interface. Architecture-specific logic such as Mask2Former query
matching must live in dedicated subclasses.

The training configuration is fully decoupled from the module through
``LightningSegConfig``, which accepts prebuilt loss, optimizer, and
scheduler objects produced by the component classes in
``pl_modules.components``.

Requirements
------------
- lightning
- torch
- torchmetrics

Usage
-----
    from pl_modules.lightning_model import LightningSegConfig, LightningSegModel
    from pl_modules.components import (
        CrossEntropyLoss, AdamW, PolyLR,
    )

    max_iters = len(train_loader) * EPOCHS

    config = LightningSegConfig(
        num_classes        = 2,
        ignore_index       = 255,
        criterion          = CrossEntropyLoss(ignore_index=255).build(),
        optimizer_fn       = AdamW(lr=6e-5, weight_decay=0.05).build(),
        scheduler_fn       = PolyLR(max_iters=max_iters).build(),
        scheduler_interval = "step",
    )

    lit_model = LightningSegModel(model=model, config=config)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import lightning as lt
import torch
import torch.nn.functional as F
import torchmetrics


@dataclass
class LightningSegConfig:
    """
    Generic semantic segmentation Lightning config.

    This class is intentionally model-agnostic. It assumes a standard
    segmentation model with:

        logits = model(x)
        loss   = criterion(logits, y)

    Model-specific behavior, such as Mask2Former query/mask losses, must live
    in specialized LightningModules.
    """

    num_classes: int
    ignore_index: int

    criterion: torch.nn.Module
    optimizer_fn: Callable[[torch.nn.Module], torch.optim.Optimizer]
    scheduler_fn: Optional[Callable[[torch.optim.Optimizer], Any]] = None
    scheduler_interval: str = "epoch"
    scheduler_frequency: int = 1


class LightningSegModel(lt.LightningModule):
    """
    Generic LightningModule for standard dense semantic segmentation.

    Expected model interface:
        model(x, **kwargs) -> logits

    Expected criterion interface:
        criterion(logits, target) -> scalar loss

    Do not put architecture-specific logic here.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: LightningSegConfig,
    ):
        super().__init__()

        self.model = model
        self.config = config
        self.criterion = config.criterion

        self.num_classes = config.num_classes
        self.ignore_index = config.ignore_index

        self.train_acc = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
        )
        self.train_miou = torchmetrics.JaccardIndex(
            task="multiclass",
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
        )

        self.val_acc = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
        )
        self.val_miou = torchmetrics.JaccardIndex(
            task="multiclass",
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
        )

        self.save_hyperparameters(ignore=["model", "config", "criterion"])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_extra(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Extract model-specific keyword arguments from the batch.

        Currently handles wavelengths for DOFA: if ``"wavelengths"`` is
        present in the batch, the first sample's wavelengths are converted
        to a Python list of floats, as expected by ``torchgeo`` internals.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Batch dictionary produced by the dataloader.

        Returns
        -------
        Dict[str, Any]
            Keyword arguments to forward to the model alongside the image.
        """
        extra = {}

        if "wavelengths" in batch:
            extra["wavelengths"] = batch["wavelengths"][0].tolist()

        return extra

    @staticmethod
    def _resize_logits_if_needed(
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Bilinearly upsample logits to match the target spatial resolution.

        Some decoder architectures (e.g. SegFormer, Mask2Former) produce
        logits at a lower resolution than the input. This method aligns them
        before computing the loss or metrics.

        Parameters
        ----------
        logits : torch.Tensor
            Predicted logits with shape ``(B, C, H', W')``.
        target : torch.Tensor
            Ground-truth labels with shape ``(B, H, W)``.

        Returns
        -------
        torch.Tensor
            Logits resized to ``(B, C, H, W)`` if needed, otherwise unchanged.
        """
        if logits.shape[-2:] != target.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        return logits

    def _compute_loss_and_logits(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run the forward pass and compute the loss.

        Parameters
        ----------
        x : torch.Tensor
            Input image batch ``(B, C, H, W)``.
        y : torch.Tensor
            Ground-truth label batch ``(B, H, W)``.
        **kwargs
            Additional model inputs, e.g. ``wavelengths`` for DOFA.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Scalar loss and logits ``(B, num_classes, H, W)``.
        """
        logits = self.model(x, **kwargs)
        logits = self._resize_logits_if_needed(logits, y)

        loss = self.criterion(logits, y)

        return loss, logits

    # ------------------------------------------------------------------
    # Lightning API
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass delegated to the wrapped model.

        Parameters
        ----------
        x : torch.Tensor
            Input image batch ``(B, C, H, W)``.
        **kwargs
            Additional model inputs forwarded verbatim.

        Returns
        -------
        torch.Tensor
            Logits ``(B, num_classes, H, W)``.
        """
        return self.model(x, **kwargs)

    def training_step(self, batch, batch_idx):
        """
        Compute training loss and update running metrics.

        Logs ``train_loss`` at step and epoch level.
        Accuracy and mIoU are accumulated and logged at epoch end
        by ``on_train_epoch_end``.
        """
        x = batch["image"]
        y = batch["label"].long()

        extra = self._extract_extra(batch)

        loss, logits = self._compute_loss_and_logits(x, y, **extra)

        self.train_acc.update(logits, y)
        self.train_miou.update(logits, y)

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Compute validation loss and update running metrics.

        Logs ``val_loss`` at epoch level only.
        Accuracy and mIoU are accumulated and logged at epoch end
        by ``on_validation_epoch_end``.
        """
        x = batch["image"]
        y = batch["label"].long()

        extra = self._extract_extra(batch)

        loss, logits = self._compute_loss_and_logits(x, y, **extra)

        self.val_acc.update(logits, y)
        self.val_miou.update(logits, y)

        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

        return loss

    def on_train_epoch_end(self):
        """
        Log and reset training accuracy and mIoU at the end of each epoch.
        """
        train_acc = self.train_acc.compute()
        train_miou = self.train_miou.compute()

        self.log("epoch_train_acc", train_acc, prog_bar=True)
        self.log("epoch_train_miou", train_miou, prog_bar=True)

        self.train_acc.reset()
        self.train_miou.reset()

    def on_validation_epoch_end(self):
        """
        Log and reset validation accuracy and mIoU at the end of each epoch.
        """
        val_acc = self.val_acc.compute()
        val_miou = self.val_miou.compute()

        self.log("val_acc", val_acc, prog_bar=True)
        self.log("val_miou", val_miou, prog_bar=True)

        self.val_acc.reset()
        self.val_miou.reset()

    def predict_step(self, batch, batch_idx):
        """
        Run inference on a batch.

        Parameters
        ----------
        batch : dict
            Batch dictionary containing at least ``"image"``.
        batch_idx : int
            Batch index.

        Returns
        -------
        torch.Tensor
            Raw logits ``(B, num_classes, H, W)``.
        """
        x = batch["image"]
        extra = self._extract_extra(batch)

        return self.model(x, **extra)

    def configure_optimizers(self):
        """
        Configure optimizer and optional learning rate scheduler.

        Uses ``config.optimizer_fn`` and ``config.scheduler_fn`` from
        ``LightningSegConfig``. The scheduler interval and frequency are
        also read from the config, allowing both epoch-level (CosineWarmup)
        and step-level (PolyLR) schedulers.

        Returns
        -------
        torch.optim.Optimizer | dict
            Optimizer alone if no scheduler is configured, otherwise a
            Lightning optimizer-scheduler dictionary.
        """
        optimizer = self.config.optimizer_fn(self.model)

        if self.config.scheduler_fn is None:
            return optimizer

        scheduler = self.config.scheduler_fn(optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": self.config.scheduler_interval,
                "frequency": self.config.scheduler_frequency,
            },
        }