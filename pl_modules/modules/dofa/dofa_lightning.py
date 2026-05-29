"""
pl_modules/modules/dofa/dofa_lightning.py
------------------------------------------
DOFA LightningModule and configuration dataclass.

DOFA (Dynamic One-For-All) is a wave-dynamic foundation model for remote
sensing that accepts images with any number of spectral bands, provided
the corresponding central wavelengths in micrometers are supplied at
inference time. Despite this wave-dynamic input, DOFA follows a standard
dense segmentation interface at the model level:

    logits = model(x, wavelengths=wavelengths)
    loss   = criterion(logits, target)

Wavelengths are extracted from the batch by ``LightningSegModel._extract_extra()``
and forwarded automatically via ``**kwargs``, so no override of the training
or validation steps is needed.

Requirements
------------
- lightning
- torch
- torchgeo  (DOFA backbone weights)
- pk_seg.models.simple_models_library.dofa (DOFASeg)
- pk_seg.models.pl_modules.lightning_model (LightningSegModel)

Usage
-----
    from pk_seg.models.pl_modules.modules.dofa.dofa_lightning import (
        DOFALightning,
        DOFALightningConfig,
    )
    from pl_modules.components import (
        CrossEntropyLoss, AdamW, PolyLR,
    )

    config = DOFALightningConfig(
        num_classes    = 2,
        ignore_index   = 255,
        variant        = "base",
        pretrained     = True,
        freeze_backbone= True,
        criterion      = CrossEntropyLoss(ignore_index=255).build(),
        optimizer_fn   = AdamW(lr=6e-5, weight_decay=0.05).build(),
        scheduler_fn   = PolyLR(max_iters=10000).build(),
        scheduler_interval = "step",
    )

    model = DOFALightning(config=config)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch

from pk_seg.models.simple_models_library.dofa import DOFASeg
from pk_seg.models.pl_modules.lightning_model import LightningSegConfig, LightningSegModel


@dataclass
class DOFALightningConfig:
    """
    Configuration for DOFA Lightning training.

    DOFA is treated as a standard dense semantic segmentation model:

        logits = model(x)
        loss   = criterion(logits, target)

    Therefore it can reuse the generic LightningSegModel implementation.
    """

    # Task
    num_classes: int
    ignore_index: int

    # Model
    variant: str = "base"
    pretrained: bool = False
    freeze_backbone: bool = False
    decoder: str = "mla"

    # Training components
    criterion: Optional[torch.nn.Module] = None
    optimizer_fn: Optional[
        Callable[[torch.nn.Module], torch.optim.Optimizer]
    ] = None
    scheduler_fn: Optional[Callable[[torch.optim.Optimizer], Any]] = None
    scheduler_interval: str = "epoch"
    scheduler_frequency: int = 1


class DOFALightning(LightningSegModel):
    """
    DOFA-specific LightningModule.

    It keeps the project modular while reusing the generic segmentation
    training loop from LightningSegModel.
    """

    def __init__(self, config: DOFALightningConfig):
        """
        Instantiate DOFASeg and wrap it in a ``LightningSegModel``.

        Builds the ``DOFASeg`` model from the given configuration, constructs
        a ``LightningSegConfig`` from the training components, and delegates
        to ``LightningSegModel.__init__()``.

        Wavelengths are not handled here — they are passed automatically by
        ``LightningSegModel._extract_extra()`` when the batch contains a
        ``"wavelengths"`` key.

        Parameters
        ----------
        config : DOFALightningConfig
            Full model and training configuration.

        Raises
        ------
        ValueError
            If ``criterion`` or ``optimizer_fn`` is None.
        """
        if config.criterion is None:
            raise ValueError("DOFALightningConfig.criterion cannot be None.")

        if config.optimizer_fn is None:
            raise ValueError("DOFALightningConfig.optimizer_fn cannot be None.")

        model = DOFASeg(
            variant=config.variant,
            num_classes=config.num_classes,
            pretrained=config.pretrained,
            freeze_backbone=config.freeze_backbone,
            decoder=config.decoder,
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

        self.dofa_config = config