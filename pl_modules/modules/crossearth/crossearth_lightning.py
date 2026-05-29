"""
pl_modules/modules/crossearth/crossearth_lightning.py
------------------------------------------------------
CrossEarth LightningModule with Mask2Former training interface.

CrossEarth uses a Mask2Former decoder that requires a two-pass forward
during training: one pass through the backbone and decoder to produce
query/mask predictions consumed by the Hungarian-matched loss, and a
second dense forward pass to produce per-pixel logits used for metrics.

This module provides:

    CrossEarthLightningConfig
        Dataclass that consolidates all model, backbone, and training
        component configuration in one place.

    CrossEarthLightning
        LightningModule subclass that overrides the training and
        validation steps to implement the Mask2Former interface while
        reusing metric initialization, optimizer/scheduler setup, and
        epoch-end logging from ``LightningSegModel``.

Requirements
------------
- lightning
- torch
- models.crossearth (CrossEarthSeg)
- pl_modules.lightning_model (LightningSegModel)
- pl_modules.components (Mask2FormerLossConfig)

Usage
-----
    from pl_modules.components import (
        CrossEarthLightning,
        CrossEarthLightningConfig,
    )
    from pl_modules.components import (
        AdamW, Mask2FormerLossConfig, PolyLR,
    )

    config = CrossEarthLightningConfig(
        num_classes             = 2,
        ignore_index            = 255,
        in_channels             = 4,
        use_pretrained_backbone = True,
        freeze_backbone         = False,
        criterion               = Mask2FormerLossConfig(num_classes=2).build(),
        optimizer_fn            = AdamW(lr=6e-5, weight_decay=0.05).build(),
        scheduler_fn            = PolyLR(max_iters=10000).build(),
        scheduler_interval      = "step",
    )

    model = CrossEarthLightning(config=config)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, cast

import torch
import torch.nn.functional as F

from models.crossearth import CrossEarthSeg
from pl_modules.lightning_model import (
    LightningSegConfig,
    LightningSegModel,
)



@dataclass
class CrossEarthLightningConfig:
    """
    Configuration for CrossEarth Mask2Former Lightning training.

    CrossEarth is trained through the Mask2Former interface:

        features = model.backbone_rein(x)
        outputs  = model.decoder.forward_train(features)
        losses   = criterion(outputs, target)
        loss     = losses["loss"]

    Dense semantic logits are still produced through:

        logits = model(x)

    and are used only for metrics.
    """

    # Task
    num_classes: int
    ignore_index: int
    in_channels: int

    # Model
    variant: str = "dinov2_vitl14_reg"
    decoder_dim: Optional[int] = None
    num_tokens: int = 100
    token_dim: int = 256
    dropout: float = 0.0
    patch_embed_init: str = "rgb_mean"

    # Backbone loading/training
    use_pretrained_backbone: bool = True
    freeze_backbone: bool = True
    train_patch_embed: bool = True
    force_reload: bool = False

    # Training components
    criterion: Optional[torch.nn.Module] = None
    optimizer_fn: Optional[
        Callable[[torch.nn.Module], torch.optim.Optimizer]
    ] = None
    scheduler_fn: Optional[Callable[[torch.optim.Optimizer], Any]] = None
    scheduler_interval: str = "epoch"
    scheduler_frequency: int = 1


class CrossEarthLightning(LightningSegModel):
    """
    CrossEarth Mask2Former LightningModule.

    The class still inherits from LightningSegModel to reuse:
        - metric initialization
        - optimizer/scheduler configuration
        - epoch-end metric logging

    but overrides training/validation steps because Mask2Former does not train
    with direct dense-logit cross entropy.
    """

    def __init__(self, config: CrossEarthLightningConfig):
        if config.criterion is None:
            raise ValueError("CrossEarthLightningConfig.criterion cannot be None.")

        if config.optimizer_fn is None:
            raise ValueError("CrossEarthLightningConfig.optimizer_fn cannot be None.")

        model = self._build_model(config)

        base_config = LightningSegConfig(
            num_classes=config.num_classes,
            ignore_index=config.ignore_index,
            criterion=config.criterion,
            optimizer_fn=config.optimizer_fn,
            scheduler_fn=config.scheduler_fn,
            scheduler_interval=config.scheduler_interval,
            scheduler_frequency=config.scheduler_frequency,
        )

        super().__init__(
            model=model,
            config=base_config,
        )

        self.crossearth_config = config

    @staticmethod
    def _build_model(config: CrossEarthLightningConfig) -> CrossEarthSeg:
        """
        Instantiate the CrossEarthSeg model from the given configuration.

        If ``use_pretrained_backbone`` is True, DINOv2 weights are downloaded
        from torch.hub via ``CrossEarthSeg.from_pretrained()``. Otherwise,
        the backbone is loaded without pretrained weights.

        Parameters
        ----------
        config : CrossEarthLightningConfig
            Full model and backbone configuration.

        Returns
        -------
        CrossEarthSeg
            Instantiated segmentation model.
        """
        if config.use_pretrained_backbone:
            return CrossEarthSeg.from_pretrained(
                variant=config.variant,
                num_classes=config.num_classes,
                decoder="mask2former",
                decoder_dim=config.decoder_dim,
                num_tokens=config.num_tokens,
                token_dim=config.token_dim,
                dropout=config.dropout,
                in_channels=config.in_channels,
                patch_embed_init=config.patch_embed_init,
                freeze_backbone=config.freeze_backbone,
                train_patch_embed=config.train_patch_embed,
                force_reload=config.force_reload,
            )

        model = CrossEarthSeg(
            variant=config.variant,
            num_classes=config.num_classes,
            decoder_dim=config.decoder_dim,
            num_tokens=config.num_tokens,
            token_dim=config.token_dim,
            dropout=config.dropout,
            in_channels=config.in_channels,
            patch_embed_init=config.patch_embed_init,
            freeze_backbone=config.freeze_backbone,
            train_patch_embed=config.train_patch_embed,
        )

        model.backbone_rein.load_backbone(
            pretrained=False,
            force_reload=config.force_reload,
        )

        return model

    @staticmethod
    def _extract_extra(batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Extract optional model keyword arguments from the batch.

        Currently handles wavelengths for wave-dynamic models. CrossEarth
        does not use wavelengths but the method is kept for interface
        consistency with ``LightningSegModel``.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Batch dictionary from the dataloader.

        Returns
        -------
        Dict[str, Any]
            Keyword arguments to pass alongside the image tensor.
        """
        extra: Dict[str, Any] = {}

        if "wavelengths" in batch:
            extra["wavelengths"] = batch["wavelengths"][0].tolist()

        return extra

    @staticmethod
    def _resize_logits_if_needed(
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Bilinearly upsample logits to the target spatial resolution if needed.

        Parameters
        ----------
        logits : torch.Tensor
            Predicted logits ``(B, C, H', W')``.
        target : torch.Tensor
            Ground-truth labels ``(B, H, W)``.

        Returns
        -------
        torch.Tensor
            Logits at resolution ``(B, C, H, W)``.
        """
        if logits.shape[-2:] != target.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        return logits

    def _mask2former_forward_train(
        self,
        x: torch.Tensor,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Run the Mask2Former training forward pass.

        Passes the image through the backbone and Rein adapter, then through
        the decoder's ``forward_train()`` method to produce the query
        classification predictions and mask predictions required by the
        Hungarian-matched loss.

        Parameters
        ----------
        x : torch.Tensor
            Input image batch ``(B, C, H, W)``.
        **kwargs
            Additional model inputs forwarded to the backbone.

        Returns
        -------
        Dict[str, Any]
            Decoder outputs containing ``pred_logits``, ``pred_masks``,
            ``all_cls_preds``, and ``all_mask_preds``.
        """
        backbone_rein = cast(Callable[[torch.Tensor], Any], self.model.backbone_rein)
        features = backbone_rein(x)

        decoder = cast(Any, self.model.decoder)
        outputs = decoder.forward_train(features)

        return cast(Dict[str, Any], outputs)

    def _compute_loss_and_logits(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute Mask2Former loss and dense logits for metrics.

        Performs two forward passes:

            1. ``_mask2former_forward_train()`` → query/mask outputs → loss;
            2. ``model(x)`` → dense logits → used only for accuracy and mIoU.

        Parameters
        ----------
        x : torch.Tensor
            Input image batch ``(B, C, H, W)``.
        y : torch.Tensor
            Ground-truth label batch ``(B, H, W)``.
        **kwargs
            Additional model inputs.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]
            Scalar total loss, dense logits ``(B, C, H, W)``, and the full
            Mask2Former loss dictionary.
        """
        outputs = self._mask2former_forward_train(x, **kwargs)

        raw_losses = self.criterion(outputs, y)
        losses = cast(Dict[str, torch.Tensor], raw_losses)

        loss = losses["loss"]

        logits = self.model(x, **kwargs)
        logits = self._resize_logits_if_needed(logits, y)

        return loss, logits, losses

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Compute Mask2Former training loss, update metrics, and log losses.

        Logs ``train_loss`` at step and epoch level, plus individual
        Mask2Former loss components via ``_log_mask2former_losses()``.
        Accuracy and mIoU are accumulated and logged at epoch end.
        """
        x = batch["image"]
        y = batch["label"].long()

        extra = self._extract_extra(batch)

        loss, logits, losses = self._compute_loss_and_logits(
            x,
            y,
            **extra,
        )

        self.train_acc.update(logits, y)
        self.train_miou.update(logits, y)

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )

        self._log_mask2former_losses(
            losses=losses,
            prefix="train",
            on_step=True,
            on_epoch=True,
        )

        return loss

    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Compute Mask2Former validation loss, update metrics, and log losses.

        Logs ``val_loss`` at epoch level only, plus individual loss
        components. Accuracy and mIoU are accumulated and logged at epoch end.
        """
        x = batch["image"]
        y = batch["label"].long()

        extra = self._extract_extra(batch)

        loss, logits, losses = self._compute_loss_and_logits(
            x,
            y,
            **extra,
        )

        self.val_acc.update(logits, y)
        self.val_miou.update(logits, y)

        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

        self._log_mask2former_losses(
            losses=losses,
            prefix="val",
            on_step=False,
            on_epoch=True,
        )

        return loss

    def predict_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Run dense inference on a batch.

        Uses the standard ``model(x)`` forward pass to produce per-pixel
        logits, bypassing the Mask2Former training interface.

        Returns
        -------
        torch.Tensor
            Dense logits ``(B, num_classes, H, W)``.
        """
        x = batch["image"]
        extra = self._extract_extra(batch)

        return self.model(x, **extra)

    def _log_mask2former_losses(
        self,
        losses: Dict[str, torch.Tensor],
        prefix: str,
        on_step: bool,
        on_epoch: bool,
    ) -> None:
        """
        Log all individual Mask2Former loss components.

        Skips the aggregated ``"loss"`` key since it is already logged
        separately as ``train_loss`` or ``val_loss``. Logs ``loss_cls``,
        ``loss_mask``, and ``loss_dice`` in the progress bar; all auxiliary
        layer losses are logged silently.

        Parameters
        ----------
        losses : Dict[str, torch.Tensor]
            Full loss dictionary returned by ``Mask2FormerLoss.forward()``.
        prefix : str
            Metric prefix, either ``"train"`` or ``"val"``.
        on_step : bool
            Whether to log at step level.
        on_epoch : bool
            Whether to log at epoch level.
        """
        for name, value in losses.items():
            if name == "loss":
                continue

            self.log(
                f"{prefix}_{name}",
                value,
                prog_bar=name in {"loss_cls", "loss_mask", "loss_dice"},
                on_step=on_step,
                on_epoch=on_epoch,
            )