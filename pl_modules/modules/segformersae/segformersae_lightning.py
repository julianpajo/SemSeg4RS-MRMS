"""
pl_modules/modules/segformersae/segformersae_lightning.py
----------------------------------------------------------
SegFormer-SAE LightningModule and configuration dataclass.

SegFormer-SAE extends SegFormer with two custom components:

    SAEModule (Spectral Awareness Embedding)
        Projects multispectral or RGBNIR input into the first MiT encoder
        stage, decoupling the spectral input size from the pretrained
        patch embedding.

    BRDDecoder (Boundary-aware Refinement Decoder)
        Optional decoder that improves segmentation accuracy along class
        boundaries by explicitly refining boundary pixel predictions.

Despite these extensions, SegFormer-SAE follows the standard dense
segmentation interface:

    logits = model(x)
    loss   = criterion(logits, target)

and can therefore reuse the generic ``LightningSegModel`` training loop
without any overrides.

Requirements
------------
- lightning
- torch
- transformers  (HuggingFace MiT encoder)
- pk_seg.models.simple_models_library.segformer_sae (SegFormerSAE)
- pk_seg.models.pl_modules.lightning_model (LightningSegModel)

Usage
-----
    from pk_seg.models.pl_modules.modules.segformersae.segformersae_lightning import (
        SegFormerSAELightning,
        SegFormerSAELightningConfig,
    )
    from pl_modules.components import (
        ImbalanceLoss, AdamW, PolyLR,
    )

    config = SegFormerSAELightningConfig(
        num_classes             = 2,
        ignore_index            = 255,
        in_channels             = 4,
        variant                 = "mit-b2",
        use_pretrained_backbone = True,
        use_brd                 = True,
        criterion               = ImbalanceLoss(
                                      num_classes=2,
                                      class_weights=[1.0, 2.5],
                                  ).build(),
        optimizer_fn            = AdamW(
                                      lr=6e-5,
                                      lr_encoder=6e-6,
                                      lr_decoder=6e-4,
                                      lr_sae=6e-4,
                                  ).build(),
        scheduler_fn            = PolyLR(max_iters=10000).build(),
        scheduler_interval      = "step",
    )

    model = SegFormerSAELightning(config=config)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch


from pk_seg.models.simple_models_library.segformer_sae import SegFormerSAE
from pk_seg.models.pl_modules.lightning_model import LightningSegConfig, LightningSegModel


@dataclass
class SegFormerSAELightningConfig:
    """
    Configuration for SegFormer-SAE Lightning training.

    SegFormer-SAE is treated as a standard dense semantic segmentation model:

        logits = model(x)
        loss   = criterion(logits, target)

    Therefore it reuses the generic LightningSegModel implementation.
    """

    # Task
    num_classes: int
    ignore_index: int
    in_channels: int

    # Model
    variant: str = "mit-b2"
    hf_model_name: str = "nvidia/mit-b2"
    use_pretrained_backbone: bool = False
    use_brd: bool = True

    # Training components
    criterion: Optional[torch.nn.Module] = None
    optimizer_fn: Optional[
        Callable[[torch.nn.Module], torch.optim.Optimizer]
    ] = None
    scheduler_fn: Optional[Callable[[torch.optim.Optimizer], Any]] = None
    scheduler_interval: str = "epoch"
    scheduler_frequency: int = 1


class SegFormerSAELightning(LightningSegModel):
    """
    SegFormer-SAE-specific LightningModule.

    It keeps the project modular while reusing the generic segmentation
    training loop from LightningSegModel.
    """

    def __init__(self, config: SegFormerSAELightningConfig):
        """
        Instantiate SegFormerSAE and wrap it in a ``LightningSegModel``.

        If ``use_pretrained_backbone`` is True, the MiT encoder weights are
        loaded from HuggingFace via ``SegFormerSAE.from_pretrained()``, which
        skips the first patch embedding as it is incompatible with the SAE
        input projection. Otherwise, the encoder is randomly initialized.

        Parameters
        ----------
        config : SegFormerSAELightningConfig
            Full model and training configuration.

        Raises
        ------
        ValueError
            If ``criterion`` or ``optimizer_fn`` is None.
        """
        if config.criterion is None:
            raise ValueError("SegFormerSAELightningConfig.criterion cannot be None.")

        if config.optimizer_fn is None:
            raise ValueError("SegFormerSAELightningConfig.optimizer_fn cannot be None.")

        if config.use_pretrained_backbone:
            model = SegFormerSAE.from_pretrained(
                hf_model_name=config.hf_model_name,
                in_channels=config.in_channels,
                num_classes=config.num_classes,
                use_brd=config.use_brd,
            )
        else:
            model = SegFormerSAE(
                variant=config.variant,
                in_channels=config.in_channels,
                num_classes=config.num_classes,
                use_brd=config.use_brd,
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

        self.segformer_sae_config = config