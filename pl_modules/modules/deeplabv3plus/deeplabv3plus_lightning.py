"""
pl_modules/modules/deeplabv3plus/deeplabv3plus_lightning.py
------------------------------------------------------------
DeepLabV3+ LightningModule and configuration dataclass.

DeepLabV3+ is a standard dense segmentation model that follows the
conventional ``logits = model(x)`` interface, so it reuses the generic
``LightningSegModel`` training loop without any overrides. This module
exists to keep model instantiation and configuration encapsulated and
consistent with the other model-specific Lightning wrappers in the project.

Requirements
------------
- lightning
- torch
- pk_seg.models.simple_models_library.deeplabv3plus (DeepLabV3Plus)
- pk_seg.models.pl_modules.lightning_model (LightningSegModel)

Usage
-----
    from pk_seg.models.pl_modules.modules.deeplabv3plus.deeplabv3plus_lightning import (
        DeepLabV3PlusLightning,
        DeepLabV3PlusLightningConfig,
    )
    from pl_modules.components import (
        CrossEntropyLoss, AdamW, PolyLR,
    )

    config = DeepLabV3PlusLightningConfig(
        num_classes         = 2,
        ignore_index        = 255,
        in_channels         = 4,
        backbone            = "resnet101",
        pretrained_backbone = False,
        criterion           = CrossEntropyLoss(ignore_index=255).build(),
        optimizer_fn        = AdamW(lr=1e-4, weight_decay=1e-4).build(),
        scheduler_fn        = PolyLR(max_iters=10000).build(),
        scheduler_interval  = "step",
    )

    model = DeepLabV3PlusLightning(config=config)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch

from pk_seg.models.simple_models_library.deeplabv3plus import DeepLabV3Plus
from pk_seg.models.pl_modules.lightning_model import LightningSegConfig, LightningSegModel


@dataclass
class DeepLabV3PlusLightningConfig:
    """
    Configuration for DeepLabV3Plus Lightning training.

    This config builds:
        - DeepLabV3Plus model
        - LightningSegConfig
        - LightningSegModel behavior through inheritance

    DeepLabV3Plus is a standard dense semantic segmentation model:

        logits = model(x)
        loss   = criterion(logits, target)

    Therefore it can reuse the generic LightningSegModel implementation.
    """

    # Task
    num_classes: int
    ignore_index: int
    in_channels: int

    # Model
    backbone: str = "resnet101"
    pretrained_backbone: bool = False

    # Training components
    criterion: Optional[torch.nn.Module] = None
    optimizer_fn: Optional[
        Callable[[torch.nn.Module], torch.optim.Optimizer]
    ] = None
    scheduler_fn: Optional[Callable[[torch.optim.Optimizer], Any]] = None
    scheduler_interval: str = "epoch"
    scheduler_frequency: int = 1


class DeepLabV3PlusLightning(LightningSegModel):
    """
    DeepLabV3Plus-specific LightningModule.

    It keeps the codebase modular while reusing the generic segmentation
    training loop from LightningSegModel.
    """

    def __init__(self, config: DeepLabV3PlusLightningConfig):
        """
        Instantiate DeepLabV3Plus and wrap it in a ``LightningSegModel``.

        Builds the ``DeepLabV3Plus`` model from the given configuration,
        constructs a ``LightningSegConfig`` from the training components,
        and delegates to ``LightningSegModel.__init__()``.

        Parameters
        ----------
        config : DeepLabV3PlusLightningConfig
            Full model and training configuration.

        Raises
        ------
        ValueError
            If ``criterion`` or ``optimizer_fn`` is None.
        """
        if config.criterion is None:
            raise ValueError("DeepLabV3PlusLightningConfig.criterion cannot be None.")

        if config.optimizer_fn is None:
            raise ValueError("DeepLabV3PlusLightningConfig.optimizer_fn cannot be None.")

        model = DeepLabV3Plus(
            backbone=config.backbone,
            in_channels=config.in_channels,
            num_classes=config.num_classes,
            pretrained_backbone=config.pretrained_backbone,
        )

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

        self.deeplab_config = config